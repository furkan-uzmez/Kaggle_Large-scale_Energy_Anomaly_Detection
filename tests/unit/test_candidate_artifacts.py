"""Real synthetic Polars/Arrow corruption and exclusive streaming boundaries."""

from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from candidate_fixtures import synthetic_partitions
from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    HPO_EXPERIMENT,
    ROW_COLUMNS,
    validate_candidate_oof,
    write_candidate_oof_batches,
)


def bundle(full=False, hpo=False):
    repeats, folds = ([0, 1, 2], [0, 1, 2, 3]) if full else ([0], [0, 1])
    manifest = {
        'schema_version': 1, 'execution_mode': 'full' if full else 'synthetic_smoke',
        'non_comparative': not full, 'n03_eligible': full,
        'source_notebook': 'notebooks/02_baseline_and_candidate_design.ipynb',
        'source_cell_id': 'candidate-design-cell', 'run_id': 'synthetic-test-run',
        'split_sha256': 'a' * 64, 'data_sha256': 'b' * 64, 'config_sha256': 'c' * 64,
        'vault_row_indices': [999], 'vault_building_ids': ['vault-building'],
        'protocol': {'repeat_ids': repeats, 'fold_ids': folds,
                     'split_seeds': {str(r): 42 + r for r in repeats}},
        'candidates': {},
    }
    for name, kind in [('p', 'positive_probability'), ('m', 'decision_margin')]:
        metadata = {
            'recipe_id': 'core', 'model_family': 'hgb' if name == 'p' else 'svc',
            'model_config_id': name + '-config',
            'experiment_id': HPO_EXPERIMENT if hpo and name == 'p' else 'compatibility-test',
            'arm_id': 'challenger', 'score_kind': kind, 'positive_label': 1,
        }
        if hpo and name == 'p':
            metadata['fold_fit_config_sha256'] = {
                f'{r}:{f}': f'{1 + r * 4 + f:064x}' for r in repeats for f in folds
            }
        else:
            metadata['model_fit_config_sha256'] = 'd' * 64
        manifest['candidates'][name] = metadata
    base = pl.from_pandas(pd.concat(
        [p.validation_keys.assign(target=p.validation_keys.row_index % 2)
         for p in synthetic_partitions()])).sort('row_index')
    expected_records, records = [], []
    for repeat in repeats:
        for row in base.iter_rows(named=True):
            expected = {**row, 'repeat': repeat,
                        'fold': int(row['row_index'] // 2 % len(folds)),
                        'split_seed': 42 + repeat,
                        **{k: manifest[k] for k in ('split_sha256', 'data_sha256',
                                                   'config_sha256', 'run_id')}}
            expected_records.append(expected)
            for candidate, meta in manifest['candidates'].items():
                prediction = (.2 if expected['target'] == 0 else .8) if candidate == 'p' else (
                    -2. if expected['target'] == 0 else 2.)
                fit_hash = meta.get('model_fit_config_sha256') or meta['fold_fit_config_sha256'][
                    f"{repeat}:{expected['fold']}"]
                records.append({**expected, 'candidate_id': candidate,
                                **{k: v for k, v in meta.items() if 'sha256' not in k},
                                'prediction': prediction, 'model_fit_config_sha256': fit_hash,
                                **{k: manifest[k] for k in ('execution_mode', 'non_comparative',
                                                           'n03_eligible')}})
    table = pa.Table.from_pylist(records, schema=CANDIDATE_OOF_SCHEMA)
    expected = pl.from_arrow(pa.Table.from_pylist(expected_records,
        schema=pa.schema([CANDIDATE_OOF_SCHEMA.field(n) for n in ROW_COLUMNS])))
    return pl.from_arrow(table), expected, manifest


def test_metrics_are_recomputed_after_alignment_per_repeat_and_fold():
    oof, expected, manifest = bundle(full=True)
    original = oof.clone(); frozen = deepcopy(manifest)
    report = validate_candidate_oof(oof.reverse(), expected.reverse(), manifest)
    assert report['oof_record_count'] == 96
    assert len(report['repeat_metrics']) == 6 and len(report['fold_metrics']) == 24
    probability = report['candidate_metrics']['p']
    assert probability['mean_repeat_metrics']['roc_auc'] == 1.
    assert probability['mean_repeat_metrics']['brier_score'] == pytest.approx(.04)
    margin = report['candidate_metrics']['m']
    assert margin['mean_repeat_metrics']['roc_auc'] == 1.
    assert margin['mean_repeat_metrics']['log_loss'] is None
    assert margin['guardrail_status'] == 'requires_calibrated_probability'
    assert margin['probability_pool_eligible'] is False
    assert probability['probability_pool_eligible'] is True
    assert report['summary_basis'] == 'equal_mean_of_repeats_not_pooled_rows'
    assert oof.equals(original) and manifest == frozen


def test_diagnostic_protocol_never_becomes_eligible():
    oof, expected, manifest = bundle()
    report = validate_candidate_oof(oof, expected, manifest)
    assert len(report['repeat_metrics']) == 2 and len(report['fold_metrics']) == 4
    assert report['candidate_metrics']['p']['probability_pool_eligible'] is False


@pytest.mark.parametrize('corruption', ['duplicate', 'missing', 'extra', 'missing_arm',
    'target', 'building_id', 'timestamp', 'repeat', 'fold', 'split_seed', 'config_sha256',
    'data_sha256', 'split_sha256', 'run_id', 'model_fit_config_sha256', 'score_kind',
    'positive_label', 'recipe_id', 'arm_id', 'experiment_id', 'non_comparative',
    'n03_eligible', 'prediction_dtype', 'target_dtype', 'nonfinite', 'range'])
def test_corrupted_artifacts_fail_closed(corruption):
    oof, expected, manifest = bundle()
    if corruption == 'duplicate':
        oof = pl.concat([oof, oof.head(1)])
    elif corruption == 'missing':
        oof = oof.slice(1)
    elif corruption == 'extra':
        oof = pl.concat([oof, oof.head(1).with_columns(pl.lit(100, dtype=pl.Int64).alias('row_index'))])
    elif corruption == 'missing_arm':
        oof = oof.filter(pl.col('candidate_id') == 'p')
    elif corruption == 'prediction_dtype':
        oof = oof.with_columns(pl.col('prediction').cast(pl.Float32))
    elif corruption == 'target_dtype':
        oof = oof.with_columns(pl.col('target').cast(pl.Boolean))
    else:
        values = {
            'target': 1, 'building_id': 'wrong', 'timestamp': oof['timestamp'][0].replace(year=2025),
            'repeat': 2, 'fold': 7, 'split_seed': 99, 'config_sha256': 'e'*64,
            'data_sha256': 'e'*64, 'split_sha256': 'e'*64, 'run_id': 'wrong',
            'model_fit_config_sha256': 'e'*64, 'score_kind': 'decision_margin',
            'positive_label': 0, 'recipe_id': 'wrong', 'arm_id': 'wrong', 'experiment_id': 'wrong',
            'non_comparative': False, 'n03_eligible': True, 'nonfinite': float('nan'), 'range': 1.1,
        }
        column = 'prediction' if corruption in ('range', 'nonfinite') else corruption
        oof = oof.with_columns(pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(values[corruption], dtype=oof.schema[column]))
            .otherwise(pl.col(column)).alias(column))
    with pytest.raises((ValueError, TypeError)):
        validate_candidate_oof(oof, expected, manifest)


@pytest.mark.parametrize('corruption', ['missing_repeat', 'missing_fold', 'expected_duplicate',
                                     'expected_target_drift', 'vault_row', 'vault_building',
                                     'source', 'synthetic_eligible', 'bad_hash', 'version'])
def test_authoritative_manifest_and_protocol_are_validated(corruption):
    oof, expected, manifest = bundle(full=corruption in ('missing_repeat', 'missing_fold'))
    if corruption == 'missing_repeat':
        oof = oof.filter(pl.col('repeat') != 2)
    elif corruption == 'missing_fold':
        oof = oof.filter(pl.col('fold') != 3)
    elif corruption == 'expected_duplicate':
        expected = pl.concat([expected, expected.head(1)])
    elif corruption == 'expected_target_drift':
        expected = expected.with_columns((1-pl.col('target')).alias('target'))
    elif corruption == 'vault_row':
        manifest['vault_row_indices'] = [0]
    elif corruption == 'vault_building':
        manifest['vault_building_ids'] = [expected['building_id'][0]]
    elif corruption == 'source':
        manifest['source_notebook'] = 'invented'
    elif corruption == 'synthetic_eligible':
        manifest['n03_eligible'] = True
    elif corruption == 'bad_hash':
        manifest['config_sha256'] = 'not-a-hash'
    else:
        manifest['schema_version'] = 2
    with pytest.raises((ValueError, TypeError)):
        validate_candidate_oof(oof, expected, manifest)


def test_hpo_requires_manifested_fold_hashes_not_a_global_fit_hash():
    oof, expected, manifest = bundle(hpo=True)
    assert validate_candidate_oof(oof, expected, manifest)['oof_record_count'] == 32
    wrong = oof.with_columns(pl.lit('d'*64).alias('model_fit_config_sha256'))
    with pytest.raises(ValueError, match='fit|hash'):
        validate_candidate_oof(wrong, expected, manifest)
    del manifest['candidates']['p']['fold_fit_config_sha256']['0:1']
    with pytest.raises(ValueError, match='fit|hash'):
        validate_candidate_oof(oof, expected, manifest)


def arrow(oof):
    return oof.to_arrow().cast(CANDIDATE_OOF_SCHEMA)


def test_streamed_batches_preserve_float64_and_all_keys(tmp_path):
    oof, expected, manifest = bundle(full=True, hpo=True)
    table = arrow(oof)
    consumed = []

    def batches():
        for index, batch in enumerate(table.to_batches(max_chunksize=7)):
            consumed.append(index)
            yield batch

    path = tmp_path / 'candidate.parquet'
    result = write_candidate_oof_batches(path, batches(), CANDIDATE_OOF_SCHEMA)
    assert result == {'status': 'written_unvalidated', 'oof_record_count': 96}
    assert len(consumed) > 1
    assert pq.ParquetFile(path).schema_arrow.equals(CANDIDATE_OOF_SCHEMA, check_metadata=True)
    persisted = pl.read_parquet(path)
    assert persisted['prediction'].dtype == pl.Float64 and persisted.equals(oof)
    assert validate_candidate_oof(persisted, expected, manifest)['oof_record_count'] == 96


def test_writer_never_overwrites_or_traverses_symlink(tmp_path):
    oof, _, _ = bundle()
    path = tmp_path / 'existing.parquet'; path.write_bytes(b'keep')
    with pytest.raises((FileExistsError, ValueError)):
        write_candidate_oof_batches(path, [arrow(oof)], CANDIDATE_OOF_SCHEMA)
    assert path.read_bytes() == b'keep'
    link = tmp_path / 'link.parquet'; link.symlink_to(path)
    with pytest.raises((FileExistsError, ValueError)):
        write_candidate_oof_batches(link, [arrow(oof)], CANDIDATE_OOF_SCHEMA)
    assert path.read_bytes() == b'keep'


@pytest.mark.parametrize('corruption', ['empty', 'empty_batch', 'duplicate_across_batches',
                                     'float32', 'bad_hash', 'schema', 'nonfinite'])
def test_stream_failures_raise_without_completed_claim(tmp_path, corruption):
    oof, _, _ = bundle()
    table = arrow(oof); schema = CANDIDATE_OOF_SCHEMA
    batches = [table.slice(0, 8), table.slice(8)]
    if corruption == 'empty':
        batches = []
    elif corruption == 'empty_batch':
        batches = [table.slice(0, 0)]
    elif corruption == 'duplicate_across_batches':
        batches.append(table.slice(0, 1))
    elif corruption == 'float32':
        batches = [table.set_column(table.schema.get_field_index('prediction'), 'prediction',
                                   pa.array(oof['prediction'].to_list(), type=pa.float32()))]
    elif corruption == 'bad_hash':
        batches = [arrow(oof.with_columns(pl.lit('invalid').alias('config_sha256')))]
    elif corruption == 'schema':
        schema = schema.remove(schema.get_field_index('target'))
    else:
        batches = [arrow(oof.with_columns(pl.lit(np.inf).alias('prediction')))]
    with pytest.raises((ValueError, TypeError)):
        write_candidate_oof_batches(tmp_path / 'failure.parquet', iter(batches), schema)


def test_empty_oof_or_expected_rows_are_rejected():
    oof, expected, manifest = bundle()
    with pytest.raises(ValueError):
        validate_candidate_oof(oof.head(0), expected, manifest)
    with pytest.raises(ValueError):
        validate_candidate_oof(oof, expected.head(0), manifest)


def test_manifest_does_not_coerce_nonstring_json_keys():
    oof, expected, manifest = bundle()
    manifest['protocol']['split_seeds'] = {0: 42}
    with pytest.raises((ValueError, TypeError), match='key'):
        validate_candidate_oof(oof, expected, manifest)


@pytest.mark.parametrize(('column', 'value'), [('repeat', 1), ('fold', 2), ('split_seed', 2**32)])
def test_writer_rejects_locally_impossible_diagnostic_protocol(tmp_path, column, value):
    oof, _, _ = bundle()
    oof = oof.with_columns(pl.lit(value, dtype=pl.Int64).alias(column))
    with pytest.raises(ValueError, match='protocol|seed'):
        write_candidate_oof_batches(tmp_path / 'invalid.parquet', [arrow(oof)], CANDIDATE_OOF_SCHEMA)


def test_nanosecond_key_changes_are_not_truncated_away():
    oof, expected, manifest = bundle()
    changed = oof.with_columns((pl.col('timestamp').cast(pl.Int64) + 1)
                              .cast(pl.Datetime('ns')).alias('timestamp'))
    with pytest.raises(ValueError, match='alignment'):
        validate_candidate_oof(changed, expected, manifest)


def test_stream_metadata_drift_and_iterator_failure_leave_only_partial_evidence(tmp_path):
    oof, _, _ = bundle()
    changed = oof.slice(8).with_columns(pl.lit('different-config').alias('model_config_id'))
    with pytest.raises(ValueError, match='metadata'):
        write_candidate_oof_batches(tmp_path / 'drift.parquet', [arrow(oof.head(8)), arrow(changed)],
                                    CANDIDATE_OOF_SCHEMA)

    def failing_source():
        yield arrow(oof.head(8))
        raise RuntimeError('source failed')

    path = tmp_path / 'partial.parquet'
    with pytest.raises(RuntimeError, match='source failed'):
        write_candidate_oof_batches(path, failing_source(), CANDIDATE_OOF_SCHEMA)
    assert path.exists()
    with pytest.raises(ValueError, match='coverage'):
        validate_candidate_oof(pl.read_parquet(path), bundle()[1], bundle()[2])


def test_integer_building_ids_preserve_key_types_and_writer_confines_parent(tmp_path):
    from project.models.candidate_artifacts import candidate_oof_schema

    oof, expected, manifest = bundle()
    expression = pl.col('building_id').str.extract(r'(\d+)$').cast(pl.Int64)
    oof, expected = oof.with_columns(expression), expected.with_columns(expression)
    assert validate_candidate_oof(oof, expected, manifest)['oof_record_count'] == 32
    schema = candidate_oof_schema(building_id_type=pa.int64())
    path = tmp_path / 'integers.parquet'
    write_candidate_oof_batches(path, [oof.to_arrow().cast(schema)], schema)
    assert pl.read_parquet(path)['building_id'].dtype == pl.Int64
    directory_link = tmp_path / 'parent-link'; directory_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='confined'):
        write_candidate_oof_batches(directory_link / 'outside.parquet', [arrow(bundle()[0])],
                                    CANDIDATE_OOF_SCHEMA)


def test_arrow_schema_version_cannot_be_discarded_by_conversion():
    oof, expected, manifest = bundle()
    corrupted = arrow(oof).replace_schema_metadata({b'candidate_schema_version': b'2'})
    with pytest.raises(ValueError, match='schema'):
        validate_candidate_oof(corrupted, expected, manifest)
