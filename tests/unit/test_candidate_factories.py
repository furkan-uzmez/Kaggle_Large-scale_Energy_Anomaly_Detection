"""Synthetic model/regime acceptance contracts; no data or output IO."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError

from candidate_fixtures import CORE_COLUMNS, binary_target, core_frame
from project.experiments.candidate_contracts import (
    BlockedExperimentError,
    CandidateSpec,
    FitBudget,
    FitCounter,
)
from project.features.distributions import build_distribution_registry
from project.features.registry import FeatureRegistry, build_lead_baseline_registry
from project.features.target_encoding import build_target_encoding_registry
from project.models.candidate_factories import (
    balanced_train_weights,
    build_candidate_arms,
    fit_candidate_arm,
)

NATIVE = "feat_missingness_passthrough_comparison"
WEIGHTS = "regime_natural_vs_class_weighted"
TREES = "model_extratrees_core_features"


def spec(recipe=NATIVE, params=(), pool=None, family=None):
    return CandidateSpec(recipe, pool or CORE_COLUMNS,
                         family or ("extratrees" if recipe == TREES else "hist_gradient_boosting"),
                         params)


def counter(count=2):
    result = FitCounter(FitBudget(count, 0, 5., 128 * 1024**2))
    result.reserve(count)
    return result


def test_known_balanced_weights_are_training_only_float64():
    values = balanced_train_weights(pd.Series([0, 0, 0, 1], dtype="int64"))
    np.testing.assert_allclose(values, [2 / 3, 2 / 3, 2 / 3, 2])
    assert values.dtype == np.dtype("float64")


@pytest.mark.parametrize("labels", [[0]*4, [1]*4, [0,2], [False,True], [0.,1.], []])
def test_invalid_weight_labels_fail(labels):
    with pytest.raises((ValueError, TypeError)):
        balanced_train_weights(pd.Series(labels))


def test_native_imputed_factor_is_real_and_models_are_identical():
    frame = core_frame()
    frame.loc[0, "meter_reading"] = np.nan
    labels = binary_target(frame); original = frame.copy(deep=True)
    arms = build_candidate_arms(spec(), build_lead_baseline_registry())
    assert tuple(arms) == ("control", "challenger")
    budget = counter()
    matrices = []
    for arm in arms.values():
        fit_candidate_arm(arm, frame, labels, groups=frame.attrs["groups"], fit_counter=budget)
        matrices.append(arm[:-1].transform(frame))
        assert arm.predict_proba(frame).shape == (16, 2)
        assert arm.classes_.tolist() == [0, 1]
        model = arm.named_steps["model"]
        assert isinstance(model, HistGradientBoostingClassifier)
        assert model.get_params() == HistGradientBoostingClassifier(early_stopping=False, random_state=42).get_params()
        assert model.do_early_stopping_ is False
        assert model.n_iter_ == model.max_iter
    assert isinstance(matrices[0], np.ndarray)
    assert matrices[0].shape == matrices[1].shape
    assert np.isnan(matrices[0]).any() and np.isfinite(matrices[1]).all()
    assert matrices[0].shape[1] == 11  # eight numeric + three observed category columns
    assert budget.used_model_fits == 2
    pd.testing.assert_frame_equal(frame, original)


def test_all_null_numeric_columns_are_retained_by_both_preprocessors():
    frame = core_frame().assign(floor_count=np.nan)
    arms = build_candidate_arms(spec(), build_lead_baseline_registry())
    native = arms['control'][:-1].fit_transform(frame)
    imputed = arms['challenger'][:-1].fit_transform(frame)
    assert native.shape == imputed.shape == (16, 11)
    assert np.isnan(native[:, 3]).all()
    assert (imputed[:, 3] == 0).all()


def test_weighted_fit_uses_final_model_sample_weight_only(monkeypatch):
    frame = core_frame(); labels = binary_target(frame); labels.iloc[:8] = 0
    observed = []
    original_fit = HistGradientBoostingClassifier.fit
    def recording(model, X, y, sample_weight=None, **kwargs):
        observed.append(None if sample_weight is None else sample_weight.copy())
        return original_fit(model, X, y, sample_weight=sample_weight, **kwargs)
    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", recording)
    arms = build_candidate_arms(spec(WEIGHTS), build_lead_baseline_registry())
    budget = counter()
    for arm in arms.values():
        fit_candidate_arm(arm, frame, labels, fit_counter=budget)
        assert arm.named_steps["model"].class_weight is None
    assert observed[0] is None
    np.testing.assert_allclose(observed[1], np.where(labels == 0, 2 / 3, 2.))
    np.testing.assert_allclose(arms['control'][:-1].transform(frame), arms['challenger'][:-1].transform(frame))


def test_extratrees_explicit_small_fit_and_default_bounds():
    arms = build_candidate_arms(spec(TREES), build_lead_baseline_registry())
    model = arms['challenger'].named_steps['model']
    assert isinstance(model, ExtraTreesClassifier)
    for name, value in {'n_estimators':64,'max_depth':12,'min_samples_leaf':2,
                        'max_features':'sqrt','bootstrap':False,'class_weight':None,
                        'n_jobs':1,'random_state':42}.items():
        assert model.get_params()[name] == value
    small = build_candidate_arms(spec(TREES, (('n_estimators',8),)), build_lead_baseline_registry())
    frame = core_frame(); labels = binary_target(frame)
    fit_candidate_arm(small['challenger'], frame, labels, fit_counter=counter(1))
    assert len(small['challenger'].named_steps['model'].estimators_) == 8
    assert small['challenger'].predict_proba(frame).shape == (16, 2)


@pytest.mark.parametrize('params', [(('n_jobs',2),),(('n_estimators',257),),(('n_estimators',True),),
                                   (('max_depth',25),),(('min_samples_leaf',0),),(('min_samples_leaf',129),),
                                   (('bootstrap',True),),(('class_weight','balanced'),),(('unknown',1),)])
def test_extratrees_rejects_invalid_configs(params):
    with pytest.raises(ValueError):
        build_candidate_arms(spec(TREES,params), build_lead_baseline_registry())


@pytest.mark.parametrize('candidate', [spec('unknown'),spec(family='unknown'),
                                      spec(params=(('early_stopping',True),)),
                                      spec(params=(('random_state',True),)),
                                      spec(params=(('max_iter',8),))])
def test_unknown_or_nonfrozen_hgb_configuration_is_rejected(candidate):
    with pytest.raises(ValueError):
        build_candidate_arms(candidate, build_lead_baseline_registry())


def test_freshness_clone_and_fitted_checks():
    first = build_candidate_arms(spec(), build_lead_baseline_registry())
    second = build_candidate_arms(spec(), build_lead_baseline_registry())
    for name in first:
        assert first[name] is not second[name]
        assert first[name].named_steps['model'] is not second[name].named_steps['model']
        assert first[name].named_steps['preprocess'] is not first['challenger' if name=='control' else 'control'].named_steps['preprocess']
        with pytest.raises(NotFittedError):
            clone(first[name]).predict_proba(core_frame())


def test_dynamic_derived_column_is_routed_and_forgery_is_blocked():
    registry = build_distribution_registry('bin'); pool=(*CORE_COLUMNS,'meter_reading_bin')
    arms = build_candidate_arms(spec(pool=pool), registry)
    frame=core_frame(); frame['meter_reading_bin']=np.arange(16)%2
    arm=arms['challenger']; fit_candidate_arm(arm,frame,binary_target(core_frame()),fit_counter=counter(1))
    assert any('meter_reading_bin' in n for n in arm[:-1].get_feature_names_out())
    forged=FeatureRegistry(replace(registry.get(n),semantic_meaning='forged') if n=='meter_reading_bin' else registry.get(n) for n in pool)
    with pytest.raises(ValueError,match='authoritative|metadata'):
        build_candidate_arms(spec(pool=pool),forged)
    te=build_target_encoding_registry()
    with pytest.raises(ValueError,match='target|authoritative'):
        build_candidate_arms(spec(pool=te.names),te)


def test_nullable_missing_unseen_category_and_exact_schema():
    frame=core_frame(); frame['meter_reading']=frame.meter_reading.astype('Float64')
    frame.loc[0,'meter_reading']=pd.NA
    frame['primary_use']=frame.primary_use.astype('string');frame.loc[0,'primary_use']=pd.NA
    arms=build_candidate_arms(spec(),build_lead_baseline_registry());arm=arms['challenger']
    fit_candidate_arm(arm,frame,binary_target(frame),fit_counter=counter(1))
    probe=frame.iloc[:2].copy();probe['primary_use']=pd.Series(['unseen',None],index=probe.index,dtype='string')
    assert np.isfinite(arm.predict_proba(probe)).all()
    with pytest.raises(ValueError,match='schema'):
        arm.predict_proba(probe[list(reversed(probe.columns))])
    with pytest.raises(ValueError):
        arm.predict_proba(probe.assign(building_id=1))


def test_failed_model_attempt_is_debited_and_unreserved_fit_is_blocked(monkeypatch):
    arm=build_candidate_arms(spec(),build_lead_baseline_registry())['control']
    def fail(*args,**kwargs):
        raise RuntimeError('model failure')
    monkeypatch.setattr(HistGradientBoostingClassifier,'fit',fail)
    budget=counter(1);frame=core_frame()
    with pytest.raises(RuntimeError,match='model failure'):
        fit_candidate_arm(arm,frame,binary_target(frame),fit_counter=budget)
    assert budget.used_model_fits==1
    with pytest.raises(ValueError,match='reserved|consumed'):
        fit_candidate_arm(arm,frame,binary_target(frame),fit_counter=budget)


def test_group_alignment_and_predictor_identity_fail_before_fit():
    frame=core_frame(); arm=build_candidate_arms(spec(),build_lead_baseline_registry())['control']
    with pytest.raises(ValueError,match='index'):
        fit_candidate_arm(arm,frame,binary_target(frame),groups=frame.attrs['groups'].iloc[::-1],fit_counter=counter(1))
    broken=frame.copy();broken.index=[0]*16
    with pytest.raises(ValueError,match='index'):
        fit_candidate_arm(arm,broken,binary_target(frame),fit_counter=counter(1))


def test_model_parameter_mutations_cannot_bypass_frozen_configuration():
    frame=core_frame()
    for recipe, change in [(NATIVE, {'model__early_stopping': True}),
                           (TREES, {'model__n_jobs': 2})]:
        arm=build_candidate_arms(spec(recipe),build_lead_baseline_registry())['challenger']
        arm.set_params(**change)
        with pytest.raises(ValueError,match='configuration|parameter'):
            fit_candidate_arm(arm,frame,binary_target(frame),fit_counter=counter(1))


@pytest.mark.parametrize('change', [
    {'model__n_estimators': 200},
    {'model__max_depth': 20},
    {'model__min_samples_leaf': 50},
    {'model__n_estimators': 200, 'model__max_depth': 20, 'model__min_samples_leaf': 50},
    {'model__n_estimators': 65},
])
def test_in_bounds_extratrees_drift_cannot_bypass_the_construction_snapshot(change):
    """TREE_BOUNDS maxima are ceilings for a new versioned config, not live drift."""
    frame = core_frame()
    arm = build_candidate_arms(spec(TREES), build_lead_baseline_registry())['challenger']
    arm.set_params(**change)
    with pytest.raises(ValueError, match='configuration|parameter'):
        fit_candidate_arm(arm, frame, binary_target(frame), fit_counter=counter(1))
    assert not hasattr(arm.named_steps['model'], 'estimators_')


def test_spec_declared_small_tree_override_stays_legal_against_its_own_snapshot():
    frame = core_frame()
    arm = build_candidate_arms(spec(TREES, (('n_estimators', 8),)),
                               build_lead_baseline_registry())['challenger']
    fit_candidate_arm(arm, frame, binary_target(frame), fit_counter=counter(1))
    assert len(arm.named_steps['model'].estimators_) == 8
    # The frozen snapshot tracks the declared 8, so drifting back to the default is rejected.
    drifted = build_candidate_arms(spec(TREES, (('n_estimators', 8),)),
                                   build_lead_baseline_registry())['challenger']
    drifted.set_params(model__n_estimators=64)
    with pytest.raises(ValueError, match='configuration|parameter'):
        fit_candidate_arm(drifted, frame, binary_target(frame), fit_counter=counter(1))


def test_native_arm_blocks_all_null_numeric_column_before_debiting_the_fit():
    frame = core_frame().assign(floor_count=np.nan)
    arm = build_candidate_arms(spec(), build_lead_baseline_registry())['control']
    budget = counter(1)
    with pytest.raises(BlockedExperimentError) as excinfo:
        fit_candidate_arm(arm, frame, binary_target(frame), fit_counter=budget)
    assert excinfo.value.code == 'native_all_null_numeric_column'
    assert 'floor_count' in str(excinfo.value)
    # Raised before the model fit, so no attempt is debited and nothing is fitted.
    assert budget.used_model_fits == 0
    assert not hasattr(arm.named_steps['model'], 'n_iter_')
    # No imputation leaked into the native arm and no reroute occurred.
    assert arm.named_steps['preprocess'].transformers[0][1] == 'passthrough'


def test_native_block_names_every_offending_column_and_spares_valid_arms():
    frame = core_frame().assign(floor_count=np.nan, year_built=np.nan)
    native = build_candidate_arms(spec(), build_lead_baseline_registry())['control']
    with pytest.raises(BlockedExperimentError) as excinfo:
        fit_candidate_arm(native, frame, binary_target(frame), fit_counter=counter(1))
    assert 'floor_count' in str(excinfo.value) and 'year_built' in str(excinfo.value)
    # The imputed arm legitimately retains all-null columns and must stay fittable.
    imputed = build_candidate_arms(spec(), build_lead_baseline_registry())['challenger']
    fit_candidate_arm(imputed, frame, binary_target(frame), fit_counter=counter(1))
    assert np.isfinite(imputed.predict_proba(frame)).all()
    # A single finite training value is sufficient for the native arm.
    partial = core_frame(); partial['floor_count'] = np.nan
    partial.loc[0, 'floor_count'] = 2.
    survivor = build_candidate_arms(spec(), build_lead_baseline_registry())['control']
    fit_candidate_arm(survivor, partial, binary_target(partial), fit_counter=counter(1))
    assert np.isfinite(survivor.predict_proba(partial)).all()


@pytest.mark.parametrize('recipe', [NATIVE, WEIGHTS, TREES])
def test_live_class_weight_drift_cannot_double_the_declared_weighting(recipe):
    """The frozen snapshot subsumes the former standalone class_weight guard."""
    frame = core_frame()
    arm = build_candidate_arms(spec(recipe), build_lead_baseline_registry())['challenger']
    assert dict(arm.frozen_params)['model__class_weight'] is None
    arm.set_params(model__class_weight='balanced')
    with pytest.raises(ValueError, match='configuration|parameter'):
        fit_candidate_arm(arm, frame, binary_target(frame), fit_counter=counter(1))
    assert not hasattr(arm.named_steps['model'], 'classes_')


@pytest.mark.parametrize('change', [
    {'preprocess__numeric__strategy': 'mean'},
    {'preprocess__categorical__imputer__strategy': 'constant'},
    {'preprocess__categorical__one_hot__handle_unknown': 'error'},
    {'preprocess__categorical__imputer__keep_empty_features': False},
    {'preprocess__numeric__keep_empty_features': False},
    {'preprocess__remainder': 'passthrough'},
    {'schema__categorical': ()},
])
def test_preprocessing_mutations_cannot_add_a_second_confounding_factor(change):
    frame = core_frame()
    arm = build_candidate_arms(spec(), build_lead_baseline_registry())['challenger']
    arm.set_params(**change)
    with pytest.raises(ValueError, match='configuration|parameter'):
        fit_candidate_arm(arm, frame, binary_target(frame), fit_counter=counter(1))


def test_weight_mode_cannot_drift_from_the_regime_implied_by_comparison_factor():
    frame = core_frame(); labels = binary_target(frame)
    # A non-weight control must not silently become weighted.
    control = build_candidate_arms(spec(), build_lead_baseline_registry())['control']
    control.set_params(weight_mode='balanced')
    with pytest.raises(ValueError, match='regime|weight'):
        fit_candidate_arm(control, frame, labels, fit_counter=counter(1))
    # The declared balanced challenger must not silently lose its weighting.
    weighted = build_candidate_arms(spec(WEIGHTS), build_lead_baseline_registry())['challenger']
    weighted.set_params(weight_mode='natural')
    with pytest.raises(ValueError, match='regime|weight'):
        fit_candidate_arm(weighted, frame, labels, fit_counter=counter(1))
    # The weight control for the weight recipe is legitimately natural.
    natural = build_candidate_arms(spec(WEIGHTS), build_lead_baseline_registry())['control']
    fit_candidate_arm(natural, frame, labels, fit_counter=counter(1))
    assert natural.weight_mode == 'natural'


def test_all_missing_category_and_unseen_inference_are_supported():
    frame=core_frame();frame['primary_use']=pd.Series([pd.NA]*16,dtype='string',index=frame.index)
    arm=build_candidate_arms(spec(),build_lead_baseline_registry())['challenger']
    fit_candidate_arm(arm,frame,binary_target(frame),fit_counter=counter(1))
    assert np.isfinite(arm.predict_proba(frame.iloc[:2].assign(primary_use='unknown'))).all()


class _Consuming(TransformerMixin, BaseEstimator):
    """Declared group consumer: records exactly what the router handed it."""

    def __init__(self):
        self.seen = 'unset'
        self.predictors = 'unset'

    def fit(self, X, y=None, groups=None):
        self.seen = groups
        self.predictors = tuple(X.columns)
        return self

    def fit_transform(self, X, y=None, groups=None):
        return self.fit(X, y, groups=groups).transform(X)

    def transform(self, X):
        return X


class _NonConsuming(TransformerMixin, BaseEstimator):
    """Non-consumer: a routed kwarg would raise TypeError instead of recording."""

    def __init__(self):
        self.called = 'unset'

    def fit(self, X, y=None):
        self.called = ()
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform(self, X):
        return X


def test_group_metadata_routes_only_to_a_declared_consumer():
    frame = core_frame(); groups = frame.attrs['groups']
    arm = build_candidate_arms(spec(), build_lead_baseline_registry())['challenger']
    consuming, plain = _Consuming(), _NonConsuming()
    arm.steps.insert(0, ('plain', plain))
    arm.steps.insert(0, ('consuming', consuming))
    fit_candidate_arm(arm, frame, binary_target(frame), groups=groups, fit_counter=counter(1))
    # The declared consumer received the exact group metadata object contents.
    assert isinstance(consuming.seen, pd.Series)
    pd.testing.assert_series_equal(consuming.seen, groups)
    # The non-consumer completed fit; any routed kwarg would have raised TypeError.
    assert plain.called == ()
    # Group metadata never becomes a predictor column for any step.
    assert consuming.predictors == CORE_COLUMNS
    assert 'building_id' not in frame.columns
    assert arm.named_steps['model'].n_features_in_ == 11


def test_dense_category_expansion_has_a_pre_fit_cap():
    count = 4097
    frame = pd.DataFrame({'primary_use': [f'use-{i}' for i in range(count)]})
    arm = build_candidate_arms(spec(pool=('primary_use',)), build_lead_baseline_registry())['control']
    with pytest.raises(ValueError, match='cardinality'):
        fit_candidate_arm(arm, frame, pd.Series(np.arange(count) % 2), fit_counter=counter(1))


def test_compound_repeated_level_names_and_raw_domain_guards():
    frame = core_frame()
    frame.index = pd.MultiIndex.from_tuples([(7, i) for i in range(16)], names=['row', 'row'])
    labels = pd.Series(np.arange(16) % 2, index=frame.index)
    arm = build_candidate_arms(spec(), build_lead_baseline_registry())['challenger']
    fit_candidate_arm(arm, frame, labels, fit_counter=counter(1))
    for column, bad in [('meter_reading', np.inf), ('year', True), ('primary_use', 3)]:
        with pytest.raises((ValueError, TypeError), match=column):
            arm.predict_proba(frame.assign(**{column: bad}))


def test_all_null_boolean_storage_cannot_masquerade_as_a_category():
    frame = core_frame()
    frame['primary_use'] = pd.Series([pd.NA] * 16, index=frame.index, dtype='boolean')
    arm = build_candidate_arms(spec(), build_lead_baseline_registry())['challenger']
    with pytest.raises(TypeError, match='primary_use'):
        fit_candidate_arm(arm, frame, binary_target(frame), fit_counter=counter(1))
@pytest.mark.parametrize('mutation', ['selectors', 'class', 'order'])
def test_core_branch_structure_is_frozen(mutation):
    frame = core_frame()
    arm = build_candidate_arms(spec(), build_lead_baseline_registry())['challenger']
    branches = list(arm.named_steps['preprocess'].transformers)
    name, transformer, columns = branches[0]
    if mutation == 'selectors':
        branches[0] = (name, transformer, ('meter_reading',))
    elif mutation == 'class':
        from sklearn.impute import SimpleImputer

        class ReplacementImputer(SimpleImputer):
            pass

        branches[0] = (name, ReplacementImputer(**transformer.get_params()), columns)
    else:
        branches.reverse()
    arm.set_params(preprocess__transformers=branches)
    budget = counter(1)
    with pytest.raises(ValueError, match='configuration'):
        fit_candidate_arm(arm, frame, binary_target(frame), fit_counter=budget)
    assert budget.used_model_fits == 0


@pytest.mark.parametrize('direct', [False, True])
def test_native_support_metadata_cannot_be_disabled(direct):
    frame = core_frame().assign(floor_count=np.nan)
    arm = build_candidate_arms(spec(), build_lead_baseline_registry())['control']
    arm.set_params(native_numeric=())
    budget = counter(1)
    with pytest.raises(ValueError, match='configuration'):
        if direct:
            arm.fit(frame, binary_target(frame))
        else:
            fit_candidate_arm(arm, frame, binary_target(frame), fit_counter=budget)
    assert budget.used_model_fits == 0


@pytest.mark.parametrize(('recipe', 'api'), [
    (WEIGHTS, 'predict'), (WEIGHTS, 'predict_proba'),
    (WEIGHTS, 'decision_function'), (WEIGHTS, 'score'),
    (TREES, 'predict'), (TREES, 'predict_proba'),
    (TREES, 'predict_log_proba'), (TREES, 'score'),
])
@pytest.mark.parametrize('mutation', ['weight', 'preprocess', 'model'])
def test_inference_rejects_configuration_drift(recipe, api, mutation):
    frame = core_frame(); labels = binary_target(frame)
    arm = build_candidate_arms(spec(recipe), build_lead_baseline_registry())['challenger']
    assert hasattr(arm, api)
    arm.fit(frame, labels)
    changes = {
        'weight': {'weight_mode': 'natural' if recipe == WEIGHTS else 'balanced'},
        'preprocess': {'preprocess__numeric__strategy': 'mean'},
        'model': {'model__learning_rate': .2} if recipe == WEIGHTS else {'model__max_depth': 20},
    }
    arm.set_params(**changes[mutation])
    with pytest.raises(ValueError, match='configuration|regime'):
        getattr(arm, api)(frame, labels) if api == 'score' else getattr(arm, api)(frame)


def test_clone_retains_structure_and_native_inference_accepts_all_null_batch():
    frame = core_frame()
    arm = clone(build_candidate_arms(spec(), build_lead_baseline_registry())['control'])
    arm.fit(frame, binary_target(frame))
    probe = frame.iloc[:2].assign(floor_count=np.nan)
    assert np.isfinite(arm.predict_proba(probe)).all()


def test_cloned_small_tree_arm_keeps_producer_extension_and_api_availability():
    frame = core_frame(); groups = frame.attrs['groups']
    arm = build_candidate_arms(spec(TREES, (('n_estimators', 8),)),
                               build_lead_baseline_registry())['challenger']
    arm.steps.insert(0, ('producer', _Consuming()))
    arm = clone(arm)
    fit_candidate_arm(arm, frame, binary_target(frame), groups=groups, fit_counter=counter(1))
    pd.testing.assert_series_equal(arm.named_steps['producer'].seen, groups)
    assert len(arm.named_steps['model'].estimators_) == 8
    assert np.isfinite(arm.predict_proba(frame)).all()
    assert not hasattr(arm, 'decision_function')
    assert not hasattr(build_candidate_arms(spec(), build_lead_baseline_registry())['control'],
                       'predict_log_proba')
@pytest.mark.parametrize('dtype', [np.complex128, np.dtype('complex128'), np.float32,
                                  'float64', object()])
def test_complex_encoder_dtype_is_rejected_before_helper_debit(dtype):
    frame = core_frame()
    arm = build_candidate_arms(spec(), build_lead_baseline_registry())['challenger']
    arm.set_params(preprocess__categorical__one_hot__dtype=dtype)
    budget = counter(1)
    with pytest.raises(ValueError, match='configuration'):
        fit_candidate_arm(arm, frame, binary_target(frame), fit_counter=budget)
    assert budget.used_model_fits == 0


@pytest.mark.parametrize('api', ['predict', 'predict_proba', 'decision_function', 'score'])
def test_postfit_encoder_dtype_mutation_is_rejected(api):
    frame = core_frame(); labels = binary_target(frame)
    arm = build_candidate_arms(spec(), build_lead_baseline_registry())['challenger']
    arm.fit(frame, labels)
    arm.set_params(preprocess__categorical__one_hot__dtype=np.complex128)
    with pytest.raises(ValueError, match='configuration'):
        getattr(arm, api)(frame, labels) if api == 'score' else getattr(arm, api)(frame)


def test_equivalent_dtype_declaration_is_clone_stable():
    frame = core_frame()
    arm = build_candidate_arms(spec(), build_lead_baseline_registry())['challenger']
    arm.set_params(preprocess__categorical__one_hot__dtype=np.dtype('float64'))
    arm = clone(arm)
    fit_candidate_arm(arm, frame, binary_target(frame), fit_counter=counter(1))
    assert np.isfinite(arm.predict_proba(frame)).all()
