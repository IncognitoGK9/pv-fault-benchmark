# Model_comparison — PV Shading-Fault Classification (Réunion dataset)

Classical-ML benchmark on the real Réunion domestic rooftop PV plant
electrical-production dataset (Graillet et al., 2022, *Data in Brief*,
46:108723; Zenodo DOI 10.5281/zenodo.7828879), file
`dt2_electrical_production_inverter_1_with_faults.csv`.

Every result reported in the paper comes from real sensor readings on a
real 4kW PV plant with real, physically-staged shading-fault episodes —
not simulated or synthetic data.

Seven classifiers are compared — Logistic Regression, k-Nearest
Neighbours, Decision Tree, SVM, MLP, Random Forest, and XGBoost — across
two label schemes (binary, tier) and two resampling conditions (with and
without SMOTE), each under repeated group-aware nested cross-validation,
with a wider hyperparameter search, matched-pairs effect sizes alongside
the Wilcoxon p-values, and a SHAP-based cross-model feature-attribution
analysis. The `results/` folder in this repository already contains the
completed run that the paper's tables and figures are drawn from.

## Reproducing the results

Put `dt2_electrical_production_inverter_1_with_faults.csv` in a `data/`
subfolder next to these scripts (or edit `RAW_CSV` at the top of `config.py`
to point at wherever you saved it). The raw file is ~200MB and is **not**
included in this folder — only the much smaller subsample this pipeline
produces is written here.

```
pip install -r requirements.txt
```

`config.py` has a `QUICK_TEST_MODE` flag. Leave it `False` to reproduce the
run whose results are checked into `results/` and reported in the paper;
set it `True` first if you just want to confirm everything executes on
your machine before committing to the long run (it cuts the outer-CV
repeats to 1 and the search budget to 3 draws, so it finishes in minutes
instead of hours).

```
python prepare_dataset.py     # ~1-2 min: loads the 200MB raw CSV, segments
                               #   it into contiguous fault/no-fault episodes,
                               #   and writes data/reunion_subsample.csv
                               #   (~20k rows) — see "Why subsample?" below.

python run_experiment.py      # the main benchmark: 7 classifiers x 2 label
                               #   schemes x 2 SMOTE conditions = 28 runs,
                               #   each under repeated group-aware nested CV
                               #   (config.N_CV_REPEATS x config.N_OUTER_FOLDS
                               #   folds). Writes results/*.json and
                               #   results/all_runs_summary.csv

python shap_analysis.py       # cross-model SHAP feature attribution —
                               #   run AFTER run_experiment.py, BEFORE
                               #   visualize.py. Writes results/shap_importance.json

python fault_type_error_analysis.py   # per-fault-code and per-tier error
                               #   breakdown (Random Forest only). Writes
                               #   results/fault_type_error_analysis.json

python feature_ablation_rf.py # raw-vs-engineered feature ablation
                               #   (Random Forest only). Writes
                               #   results/feature_ablation_rf.json

python visualize.py           # builds figures/*.png from the results/ files
```

### Measured runtime

With `QUICK_TEST_MODE = False` (the paper-quality settings: 3 CV repeats,
15-draw random search), `run_experiment.py` took 26,666 s (≈7.4 h) on the
development machine (Table V in the paper), driven mostly by SVM under the
3-class tier scheme and by Random Forest's wider ensemble sizes.
`shap_analysis.py` adds roughly 15–25 more minutes, mostly spent on the
three models (kNN, SVM, MLP) that have no closed-form SHAP algorithm and
fall back to a sampling-based explainer. If this is too slow on your
machine, the easiest levers (all in `config.py`) are: lower `N_CV_REPEATS`
to 1 or 2, lower `N_RANDOM_SEARCH_ITER`, or temporarily comment `"SVM"` out
of `MODEL_NAMES`.

## Why subsample?

The raw file has 2,988,798 valid rows sampled roughly every 6 seconds over
559 days, but the fault-labelled portions all come from only **17** distinct,
deliberately-staged shading episodes (confirmed by segmenting the series on
every change of fault-vs-no-fault state — see `prepare_dataset.py`). Two
things follow from that:

1. **Leakage risk.** Adjacent rows inside one episode are near-duplicates
   (same shading condition, a few seconds apart). A plain random row split
   would put near-identical rows on both sides of a train/test split. This
   pipeline instead treats every contiguous episode as an indivisible group
   (`episode_id`) and always splits with `GroupKFold`/`StratifiedGroupKFold`,
   so a given episode is never partly in train and partly in test.
2. **Compute cost.** Nested cross-validation with hyperparameter search
   across 7 classifiers (several of which scale poorly past a few tens of
   thousands of rows) on ~3 million rows is not tractable on a laptop in
   reasonable time. `prepare_dataset.py` caps how many rows each episode may
   contribute (`MAX_ROWS_PER_FAULT_SEGMENT`, `MAX_ROWS_PER_NOFAULT_SEGMENT`
   in `config.py`) and caps the total no-fault volume
   (`MAX_NOFAULT_SEGMENTS_ROWS`), while keeping **every** one of the 17 real
   fault episodes represented. The result is a ~20,800-row subsample with
   roughly a third of rows fault-labelled — real rows, just fewer of them —
   and the full-dataset statistics (row counts, date range, per-fault-code
   counts) are saved to `data/subsample_metadata.json` before any capping,
   precisely so the paper can report the true class balance alongside the
   subsampling procedure. The paper's Section III.A also states directly why
   subsampling is preferred here over alternatives such as a weighted loss
   function or a balanced-bagging ensemble (neither solves the runtime
   problem across all seven model families) or distributed/chunked training
   (adds infrastructure complexity outside this project's single-machine
   reproducibility target).

## Label schemes (`config.py`)

- **binary** — `Fault == 0` vs. any nonzero fault code.
- **tier** — a 3-way severity grouping taken directly from the dataset's own
  documented fault taxonomy (Graillet et al., Table 3): No-Fault / Constant
  Partial Shading (codes 2.1–3.2) / Severe (uniform shading, code 1, and
  intermittent shading, code 4). This grouping is a stated modelling choice,
  not a natural constant — see the paper's Section III.C.

## Files

| File | Purpose |
|---|---|
| `config.py` | All paths, constants, label-scheme, CV, and SHAP settings |
| `prepare_dataset.py` | Loads raw CSV, segments, subsamples, writes `data/` |
| `features.py` | Electrical-channel feature engineering (ratios, rolling stats, rate of change), computed within-episode only |
| `target.py` | Builds the binary / tier label arrays |
| `models.py` | The 7 classifiers + their hyperparameter search grids |
| `pipeline.py` | Repeated StratifiedGroupKFold nested CV + imbalanced-learn SMOTE pipeline |
| `evaluate.py` | Per-fold metrics, aggregation, confusion matrices, Wilcoxon tests, matched-pairs effect sizes |
| `run_experiment.py` | Orchestrates all 28 runs, writes `results/` |
| `shap_analysis.py` | Cross-model SHAP feature attribution — run after `run_experiment.py` |
| `fault_type_error_analysis.py` | Per-fault-code and per-tier error breakdown (Random Forest only) |
| `feature_ablation_rf.py` | Raw-vs-engineered feature ablation (Random Forest only) |
| `visualize.py` | Builds `figures/*.png` from `results/` (and `shap_importance.json`) |

## Output (`results/`)

- `all_runs_summary.csv` — one row per (scheme, SMOTE, model) with mean±std of every metric
- `summary_metrics.json` — same, nested JSON
- `wilcoxon_accuracy.json`, `wilcoxon_mcc.json` — pairwise significance (primary condition: binary + SMOTE)
- `effect_size_accuracy.json`, `effect_size_mcc.json` — matched-pairs rank-biserial effect size for the same pairwise comparisons (Kerby, 2014), reported alongside the p-values so a non-significant result can be read together with how large the underlying difference actually was
- `confusion_matrices_binary_smote.json` — pooled outer-fold confusion matrices
- `feature_importance.json` — legacy: Logistic Regression coefficients + Decision Tree importances (not directly comparable to each other — kept for backward compatibility only)
- `shap_importance.json` — the primary, cross-model-comparable feature-attribution measure (written by `shap_analysis.py`)
- `best_hyperparameters.json` — the winning `RandomizedSearchCV` configuration for every outer fold, model, and condition
- `most_common_hyperparameters.json` — the modal (most frequently selected) configuration per model/condition across the outer folds; underlies paper Table IV-b
- `feature_ablation_rf.json` — raw-vs-engineered feature ablation (Table X)
- `fault_type_error_analysis.json` — per-fault-code and per-tier breakdown (Tables XI, XI-b)
- `figures/*.png` — all paper figures

All of the above are already populated in this repository from the run
reported in the paper (Table V gives the software/hardware environment and
measured runtime); re-running the scripts above with the same
`RANDOM_SEED` (42) reproduces them.
