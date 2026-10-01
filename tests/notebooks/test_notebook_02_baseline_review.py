"""Artifact-boundary tests for Notebook 02's read-only baseline presentation."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock

import matplotlib.pyplot as plt
import nbformat
import polars as pl
import pytest
from IPython.display import Markdown

from project.notebook_charts import plot_comparative_count_bars

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks/02_baseline_and_candidate_design.ipynb"


def render_review(review):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    functions = [
        node
        for cell in notebook.cells if cell.cell_type == "code"
        for node in ast.parse(cell.source).body
        if isinstance(node, ast.FunctionDef) and node.name == "render_baseline_review"
    ]
    assert functions, "Notebook must render validated baseline artifacts conditionally"
    outputs = []
    figures = []
    reader = Mock(return_value=review)
    namespace = {
        "load_baseline_review": reader,
        "display": outputs.append,
        "Markdown": Markdown,
        "pl": pl,
        "plot_comparative_count_bars": plot_comparative_count_bars,
        "show_figure": lambda figure: (figures.append(figure), outputs.append(figure)),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)  # noqa: S102
    result = namespace["render_baseline_review"](Path("."))
    reader.assert_called_once_with(Path("."))
    return result, outputs, figures


def completed_review():
    ids = ["dummy_prior", "logistic_l2", "hist_gradient_boosting"]
    return {
        "status": "completed", "run_id": "full-test", "run_path": "outputs/baselines/full-test",
        "reference_candidate_id": "hist_gradient_boosting", "positive_prevalence": 0.123,
        "development_rows": 1399631, "development_buildings": 160, "fold_count": 4,
        "repeat_count": 3, "split_seeds": [42, 43, 44],
        "resources": {"elapsed_seconds": 11.5, "process_peak_rss_mb": 1024, "model_fit_count": 36,
                      "execution_guard": {"cpu_threads": 4, "max_rss_bytes": None,
                                          "max_wall_seconds": 2700, "rss_limit_enforced": False}},
        "metrics": pl.DataFrame({
            "candidate_id": ids, "model_family": ["dummy", "linear_model", "ensemble_gbdt"],
            "roc_auc": [0.5, 0.713, 0.821], "average_precision": [0.123, 0.31, 0.49],
            "log_loss": [0.4, 0.3, 0.2], "brier_score": [0.11, 0.09, 0.07],
            "fold_roc_auc_mean": [0.5, 0.7, 0.8], "fold_roc_auc_std": [0.0, 0.04, 0.03],
            "fit_wall_seconds": [0.1, 2.0, 8.0],
        }),
        "repeat_metrics": pl.DataFrame([
            {"candidate_id": candidate, "repeat": repeat, "split_seed": 42 + repeat,
             "roc_auc": [0.5, 0.713, 0.821][i] + (repeat - 1) * 0.01,
             "average_precision": [0.123, 0.31, 0.49][i] + (repeat - 1) * 0.01,
             "log_loss": [0.4, 0.3, 0.2][i], "brier_score": [0.11, 0.09, 0.07][i]}
            for i, candidate in enumerate(ids) for repeat in range(3)
        ]),
        "fold_metrics": pl.DataFrame([
            {"candidate_id": candidate, "repeat": repeat, "split_seed": 42 + repeat,
             "fold": fold, "roc_auc": 0.6,
             "average_precision": 0.2, "log_loss": 0.3, "brier_score": 0.1}
            for candidate in ids for repeat in range(3) for fold in range(4)
        ]),
    }


def test_pending_artifact_reports_reason_without_metrics_or_plots():
    review = {"status": "not_run", "reason": "No completed valid full run"}
    result, outputs, figures = render_review(review)
    assert result == review
    assert not figures
    assert not any(isinstance(output, pl.DataFrame) for output in outputs)
    text = "\n".join(output.data for output in outputs)
    assert "not run" in text.casefold()
    assert review["reason"] in text
    assert "Observation" in text and "Interpretation" in text and "Action" in text


def test_completed_artifact_renders_exact_scores_references_and_full_evidence():
    review = completed_review()
    result, outputs, figures = render_review(review)
    try:
        assert result is review
        assert len(figures) == 2
        for figure, metric, title, reference in zip(
            figures, ["roc_auc", "average_precision"],
            ["Mean repeat-level development OOF ROC-AUC", "Mean repeat-level development OOF Average Precision"],
            [0.5, review["positive_prevalence"]], strict=True,
        ):
            axis = figure.axes[0]
            assert axis.get_title() == title
            assert axis.get_xlim() == (0.0, 1.0)
            assert figure.get_size_inches().tolist() == [13.0, 5.0]
            assert [patch.get_width() for patch in axis.patches] == pytest.approx(review["metrics"][metric].to_list())
            assert any(line.get_linestyle() == "--" and line.get_xdata()[0] == reference for line in axis.lines)
        tables = [output for output in outputs if isinstance(output, pl.DataFrame)]
        assert any(table.equals(review["metrics"]) for table in tables)
        assert any(table.equals(review["fold_metrics"]) for table in tables)
        assert any(table.equals(review["repeat_metrics"]) for table in tables)
        text = "\n".join(output.data for output in outputs if isinstance(output, Markdown))
        for phrase in ("0.821", "0.490", "hist_gradient_boosting", "standard deviation", "not a confidence interval",
                       "estimator seed 42", "split seeds", "12 dependent", "not statistically significant", "FE", "PCA", "FS", "not run", "production"):
            assert phrase in text
        provenance = tables[-1].to_dicts()
        assert any(row["field"] == "run_path" and row["value"] == review["run_path"] for row in provenance)
        assert any(row["field"] == "resources" and "36" in row["value"] for row in provenance)
        assert outputs.index(figures[-1]) < next(i for i, output in enumerate(outputs) if isinstance(output, Markdown))
        assert next(i for i, output in enumerate(outputs) if isinstance(output, Markdown)) < next(i for i, output in enumerate(outputs) if isinstance(output, pl.DataFrame))
    finally:
        for figure in figures:
            plt.close(figure)


def test_ranking_reference_reports_worse_probability_guardrails_and_worst_folds():
    review = completed_review()
    review['metrics'] = review['metrics'].with_columns(
        pl.Series('log_loss', [0.103, 0.107, 0.196]),
        pl.Series('brier_score', [0.021, 0.022, 0.043]),
        pl.Series('roc_auc', [0.495, 0.713, 0.821]),
    )
    review['fold_metrics'] = review['fold_metrics'].with_columns(
        pl.when((pl.col('candidate_id') == 'hist_gradient_boosting') & (pl.col('fold') == 2))
        .then(0.777).otherwise(pl.col('log_loss')).alias('log_loss'),
        pl.when((pl.col('candidate_id') == 'hist_gradient_boosting') & (pl.col('fold') == 3))
        .then(0.222).otherwise(pl.col('brier_score')).alias('brier_score'),
        pl.when(pl.col('candidate_id') == 'dummy_prior').then(0.5)
        .otherwise(pl.col('roc_auc')).alias('roc_auc'),
    )
    _, outputs, figures = render_review(review)
    try:
        text = '\n'.join(output.data for output in outputs if isinstance(output, Markdown))
        for phrase in ('0.196000', '0.043000', '0.107000', '0.103000',
                        'worse', 'repeat 0 / fold 2', '0.777000', 'repeat 0 / fold 3', '0.222000',
                       'training priors', 'score inversion', 'operational decision cutoff',
                       'calibration hypothesis', 'not run'):
            assert phrase in text
    finally:
        for figure in figures:
            plt.close(figure)


def test_static_narrative_and_tail_acknowledge_measured_baseline_evidence():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = '\n'.join(cell.source for cell in notebook.cells)
    assert 'eligible baseline metrics are reviewed separately above' in source
    assert 'candidate-campaign run' in source
    assert 'Until a comparative OOF artifact exists, comparative outcomes stay reported as not run' not in source
    assert '4 GiB RAM' not in source
    assert 'no RAM limit enforced' in source
    assert 'No candidate-campaign execution while its caps or approval remain pending' in source
