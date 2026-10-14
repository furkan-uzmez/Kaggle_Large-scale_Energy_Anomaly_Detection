# LEAD Energy Anomaly Detection

Research code for the Kaggle [Large-scale Energy Anomaly Detection (LEAD)](https://www.kaggle.com/competitions/energy-anomaly-detection) competition: given hourly building meter readings, predict whether a reading is anomalous. The repository focuses on leakage-aware, building-isolated validation, auditable experiment campaigns, and notebooks that review the resulting artifacts.

This is a personal research repository, not a packaged library or a supported product. It does not bundle the raw competition data, trained model files, or `outputs/` run artifacts. The notebooks do retain cached tables and figures.

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Quick start (no data needed)](#quick-start-no-data-needed)
- [Repository map](#repository-map)
- [Getting the competition data](#getting-the-competition-data)
- [Data preparation and baselines](#data-preparation-and-baselines)
- [Notebooks](#notebooks)
- [Optional backend environments](#optional-backend-environments)
- [Submitting to Kaggle (read first)](#submitting-to-kaggle-read-first)
- [Configuration and outputs](#configuration-and-outputs)
- [Tests](#tests)
- [Limitations](#limitations)
- [License](#license)

## Requirements

- Linux or another POSIX system (the repository has been used on Linux only).
- [uv](https://docs.astral.sh/uv/) and Python 3.13 (`requires-python = ">=3.13"`).
- For real data work: a Kaggle account that has accepted the competition rules, and the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) on your own `PATH`.

## Install

```bash
uv sync --locked --group dev
```

This installs the base environment from [`pyproject.toml`](pyproject.toml) and `uv.lock`: Polars, pandas, scikit-learn, PyArrow, Jupyter tooling and pytest/ruff. It does **not** install LightGBM, XGBoost, CatBoost, PyTorch, AutoGluon or cuML; see [Optional backend environments](#optional-backend-environments).

## Quick start (no data needed)

Run the bounded synthetic candidate smoke. It uses generated fixtures in a temporary directory and does not read competition data, train on real data, or contact Kaggle:

```bash
uv run python scripts/campaign.py --candidate-synthetic-smoke
```

The command prints one JSON object and exits with status `0` when its `status` field is `completed` (non-zero otherwise). This synthetic smoke has been run successfully against the locked environment; the data-dependent pipelines below have not been re-run for this README.

## Repository map

| Path | Purpose |
| --- | --- |
| [`src/project/`](src/project) | Importable package: data loading/validation/splits, feature registry, model wrappers, experiment and campaign contracts, submission helpers. |
| [`scripts/`](scripts) | Command-line entry points (data prep, baseline training, campaigns, submission generation). Use `--help` on each. |
| [`configs/`](configs) | Data contract (file hashes, schemas), split configs, resource budget, and experiment/campaign YAML under `configs/experiments/`. |
| [`notebooks/`](notebooks) | Five numbered review notebooks (see below). |
| [`tests/`](tests) | `unit/`, `integration/` and `notebooks/` pytest suites. |
| [`docs/evidence/`](docs/evidence) | Three runtime-required records: construction-year encoding, approved AutoGluon profile, and the current development audit. |

These records are inputs to campaign admission and integrity checks, not optional execution logs. Keep their exact bytes: the code and reviewed configurations bind them by SHA-256. Superseded audits and historical smoke reports are not bundled.

## Getting the competition data

Data is **not** distributed here. Obtain it yourself from the competition page, under the competition's own rules; the MIT license in this repository does not grant any rights to the data.

```bash
kaggle competitions download -c energy-anomaly-detection -p data/
unzip data/energy-anomaly-detection.zip -d data/energy-anomaly-detection
```

The code expects these files in `data/energy-anomaly-detection/`: `train.csv`, `test.csv`, `sample_submission.csv`, `train_features.csv`, `test_features.csv`. Their expected SHA-256 hashes and row counts are in [`configs/data_contract.yaml`](configs/data_contract.yaml) and are checked on preparation. `data/` and `outputs/` are git-ignored.

## Data preparation and baselines

Command order, each step reading the previous step's files under `outputs/`:

```bash
# 1. Bounded validation of the raw files (no full Parquet written)
uv run python scripts/prepare_data.py --smoke --sample-rows 10000

# 2. Full validation; writes joined Parquet to outputs/prepared/ and lineage to outputs/lineage/
uv run python scripts/prepare_data.py --materialize

# 3. Building-isolated development/vault manifest in outputs/splits/
uv run python scripts/prepare_data.py --build-splits

# 4. Repeated development folds. The default configs/split_repeated.yaml reads
#    outputs/splits/building_folds_repeated.parquet, a frozen source that a fresh
#    checkout does not have. Bootstrap from the manifest written in step 3 instead:
cat > outputs/splits/bootstrap_repeated.yaml <<'EOF'
source_manifest: outputs/splits/building_folds.parquet
prepared_train: outputs/prepared/train_joined.parquet
output_dir: outputs/splits
n_dev_folds: 4
split_seeds: [42, 43, 44]
EOF
uv run python scripts/prepare_data.py --build-repeated-splits --split-config outputs/splits/bootstrap_repeated.yaml

# 5. Baseline OOF experiments on development folds only
uv run python scripts/train.py --smoke --max-buildings 8   # one bounded dev fold
uv run python scripts/train.py --full                       # 4 dev folds x 3 repeats
```

`--full` requires the approved budget file `configs/experiments/baseline_execution_budget.yaml` (the default `--budget`). Baseline candidates are defined in [`configs/experiments/baselines.yaml`](configs/experiments/baselines.yaml) and use only scikit-learn models.

The many other `scripts/train_*.py` and `scripts/*campaign*` entry points (GBDT, MLP, ensembles, AutoGluon, full campaigns) and their YAML in `configs/experiments/` are research artifacts with their own admission/budget checks and extra dependencies. This README does not claim a single command that reproduces all models or reported scores; inspect each script's `--help` and config before running it.

## Notebooks

Run from the repository root (`uv run jupyter lab`). The notebooks review artifacts under `outputs/` and have different prerequisites:

- Notebook 01 needs the downloaded raw CSVs, prepared data and join lineage (step 2),
  plus `outputs/splits/building_folds_repeated.parquet` and `split_metadata_repeated.json`
  (steps 3-4). It also needs the domain-shift artifacts from
  [`scripts/adversarial_validate.py`](scripts/adversarial_validate.py):
  `outputs/shift/adversarial_validation.json`, `domain_oof.parquet` and `fold_averaged_importance.csv`.
- Notebook 02 needs the splits and baseline artifacts (steps 3-5).
- Notebooks 03-05 need artifacts from model campaigns not covered by the commands above.

Shift diagnostic, run after `--materialize` (not part of the no-data quickstart):

```bash
uv run python scripts/adversarial_validate.py
```

It reads the full prepared train and test Parquet from `outputs/prepared/` and fits a building-level 5-fold train-vs-test domain classifier (options: `--prepared-dir`, `--output-dir`, `--folds` fixed at 5, `--seed`). It writes to `outputs/shift/` and is a distribution-shift diagnostic, not a competition model score. This command has only had its `--help` checked, not a full run, for this README.

| Notebook | Topic |
| --- | --- |
| [`01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb) | Data understanding and outlier contract |
| [`02_baseline_and_candidate_design.ipynb`](notebooks/02_baseline_and_candidate_design.ipynb) | Baselines and candidate experiment design |
| [`03_model_comparison_and_error_analysis.ipynb`](notebooks/03_model_comparison_and_error_analysis.ipynb) | Model comparison and error analysis |
| [`04_ensemble_review.ipynb`](notebooks/04_ensemble_review.ipynb) | Ensemble review |
| [`05_final_results.ipynb`](notebooks/05_final_results.ipynb) | Final results summary |

Notebooks 03-05 are intended to consume artifacts produced by the model campaigns, so they can only work if those artifacts exist locally (the generated `outputs/` tree is not published). Notebook 04 also imports `scipy`, which the lock file pulls in as a dependency of another package.

## Optional backend environments

The base environment stays free of heavy model backends. Install them yourself in separate environments as needed.

| Backend | Where the code expects it |
| --- | --- |
| LightGBM, XGBoost, CatBoost, PyTorch | Imported by the GBDT, MLP, and ensemble scripts. Not declared in `pyproject.toml`; add them to an environment of your choosing. |
| AutoGluon | Separate Python 3.12 environment at `.venv-autogluon/`, with AutoGluon 1.4.0. The full-campaign code validates this fixed interpreter location and the restricted profile in [`autogluon_full_profile_v1.json`](docs/evidence/autogluon_full_profile_v1.json). |
| cuML RBF SVM (GPU) | Separate interpreter at `.venv_gpu_cuml/`, with cuML and CuPy and a visible GPU. Used by `src/project/features/gpu_svm_bridge.py`; set `DISABLE_GPU_SVM=1` to skip it. |

These environments are git-ignored and are not reproduced by `uv.lock`.

## Submitting to Kaggle (read first)

> **Warning:** `scripts/generate_and_submit_all_remaining.py`, `generate_and_submit_feature_experiments.py` and `generate_and_submit_final_6.py` train models on the prepared data and, **unless run with `--smoke`**, upload the resulting CSV files to the competition using the `kaggle` executable found on your `PATH`. They consume your submission quota and can take a long time. They require the prepared Parquet files from the steps above and the optional backends listed above.

- Authenticate the Kaggle CLI yourself (your own `~/.kaggle/kaggle.json` or Kaggle's supported environment variables). No credentials are stored in this repository, and none should be committed.
- Run `--smoke` first to generate on a tiny sample without uploading. `scripts/generate_submissions.py` also offers `--dry-run` and `--smoke`.
- Verify the competition's current rules and daily submission limits before uploading anything.

## Configuration and outputs

| Location | What |
| --- | --- |
| `configs/data_contract.yaml` | Raw-file hashes, row counts and schemas. |
| `configs/split.yaml`, `configs/split_repeated.yaml` | Building-isolated split and repeated-fold settings. |
| `configs/resource_budget.yaml` | Resource limits used by campaign admission. |
| `configs/experiments/` | Baseline, candidate, HPO, ensemble and campaign YAML. |
| `outputs/prepared/`, `outputs/splits/`, `outputs/lineage/` | Prepared data, split manifests, row lineage. |
| `outputs/runs/`, `outputs/candidates/`, `outputs/submissions/` | Run artifacts, candidate results, generated submission CSVs. |

All of `outputs/` is generated and git-ignored. Whether generated outputs derived from competition data may be shared has not been resolved for this repository; treat them as private unless you verify the competition rules.

## Tests

```bash
uv run pytest tests/unit/test_paths.py tests/unit/test_splits.py tests/unit/test_submission.py tests/unit/test_campaign_cli.py
```

The full `uv run pytest` suite is broader than those focused tests and is not known to pass in a fresh checkout. Some reviewed full-campaign tests and routes require local planning artifacts that are not bundled in this repository, and others expect local data, generated outputs or optional backends. A known failure is `test_supported_id_inventory` (10 vs. 11 expected entries).

## Limitations

- Research code for one competition; interfaces, configs and artifacts change without compatibility guarantees.
- Not security-hardened. Do not run campaign scripts on untrusted configs, artifacts or notebooks; they execute subprocesses and read/write files under the repository.
- Only the Linux, Python 3.13 base path described above has been used by the author; other platforms are untested.
- No claim is made here about competition scores or the reproducibility of any specific result.

## License

Code is released under the [MIT License](LICENSE), Copyright (c) 2026 furkan-uzmez. The license covers this repository's code only. The competition data, and any rights to redistribute it or work derived from it, are governed by the Kaggle competition rules, not by this license.
