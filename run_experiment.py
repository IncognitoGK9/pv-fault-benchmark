"""
run_experiment.py
Orchestrates the full comparison: 7 classifiers x 2 label schemes x 2 SMOTE
conditions = 28 conditioned runs (paper Section V.D), each under repeated
group-aware nested cross-validation (StratifiedGroupKFold, config.N_CV_REPEATS
repeats of config.N_OUTER_FOLDS folds; see config.py and pipeline.py for the
full reasoning). After this script completes, run shap_analysis.py for the
cross-model feature-attribution analysis and visualize.py for the figures.
"""
import os
import json
import time
import warnings
from collections import Counter

# scikit-learn 1.6+ flags an internal method (BaseEstimator._validate_data)
# that imbalanced-learn's SMOTE still calls internally; it is harmless noise.
# It must be suppressed via the PYTHONWARNINGS env var, not just
# warnings.filterwarnings() below, because pipeline.py's
# RandomizedSearchCV(n_jobs=-1) runs the inner search in separate worker
# subprocesses (joblib/loky), and each worker is a fresh interpreter that
# only inherits filters set through the environment at its own startup.
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning:sklearn.base")
warnings.filterwarnings(
    "ignore",
    message=".*_validate_data.*is deprecated.*",
    category=FutureWarning,
)

import numpy as np
import pandas as pd

import config
import features
import target as target_mod
import pipeline as pl
import evaluate as ev


def _mode_params(param_dicts):
    """Most common value per hyperparameter key across a list of per-fold
    best_params dicts; ties broken by order of first appearance (stable,
    deterministic). Used to summarise best_hyperparameters.json into
    most_common_hyperparameters.json.
    """
    keys = param_dicts[0].keys()
    mode = {}
    for k in keys:
        values = [str(pd_[k]) for pd_ in param_dicts]
        mode[k] = Counter(values).most_common(1)[0][0]
    return mode


def main():
    """Run all 20 (model x scheme x SMOTE) conditions under nested CV and
    write every results/*.json and results/all_runs_summary.csv artefact
    the paper's Results section (Section VI) and figures depend on.
    """
    t0 = time.time()
    print("Loading subsample ...")
    df = pd.read_csv(config.SUBSAMPLE_CSV, parse_dates=["time"])

    X_df, feature_cols = features.build_features(df)
    groups = df["episode_id"].values
    print(f"Rows: {len(df)}  Features: {len(feature_cols)}  Episodes: {len(np.unique(groups))}")

    all_results = {}      # all_results[scheme][smote][model] = fold_results
    all_aggs = {}          # all_aggs[scheme][smote][model] = aggregate(fold_results)
    all_best_params = {}   # all_best_params[scheme][smote][model] = [per-fold best_params, one per outer fold]

    for scheme in config.LABEL_SCHEMES:
        y = target_mod.get_target(df, scheme)
        n_classes = len(np.unique(y))
        all_results[scheme] = {}
        all_aggs[scheme] = {}
        all_best_params[scheme] = {}

        for use_smote in config.SMOTE_CONDITIONS:
            smote_key = "smote" if use_smote else "no_smote"
            all_results[scheme][smote_key] = {}
            all_aggs[scheme][smote_key] = {}
            all_best_params[scheme][smote_key] = {}

            for model_name in config.MODEL_NAMES:
                t1 = time.time()
                fold_results = pl.run_nested_cv(
                    X_df.values, y, groups, model_name, use_smote)
                agg = ev.aggregate(fold_results, n_classes=n_classes)
                all_results[scheme][smote_key][model_name] = fold_results
                all_aggs[scheme][smote_key][model_name] = agg
                all_best_params[scheme][smote_key][model_name] = [
                    fr["best_params"] for fr in fold_results
                ]
                dt = time.time() - t1
                print(f"[{scheme:6s} | {smote_key:8s} | {model_name:18s}] "
                      f"acc={agg['accuracy_mean']:.3f}+/-{agg['accuracy_std']:.3f}  "
                      f"mcc={agg['mcc_mean']:.3f}+/-{agg['mcc_std']:.3f}  ({dt:.1f}s)")

    # ---- Aggregated metrics (drop raw per-fold arrays for a slim summary) ----
    summary = {}
    for scheme in all_aggs:
        summary[scheme] = {}
        for smote_key in all_aggs[scheme]:
            summary[scheme][smote_key] = {}
            for model_name, agg in all_aggs[scheme][smote_key].items():
                summary[scheme][smote_key][model_name] = {
                    k: v for k, v in agg.items() if not k.endswith("_values")
                }
    with open(f"{config.RESULTS_DIR}/summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Selected hyperparameters per outer fold, plus the mode across folds ----
    # (a byproduct of the RandomizedSearchCV already run above, persisted
    # rather than discarded; underlies paper Table IV-b).
    with open(f"{config.RESULTS_DIR}/best_hyperparameters.json", "w") as f:
        json.dump(all_best_params, f, indent=2)

    most_common_params = {}
    for scheme in all_best_params:
        most_common_params[scheme] = {}
        for smote_key in all_best_params[scheme]:
            most_common_params[scheme][smote_key] = {
                model_name: _mode_params(param_list)
                for model_name, param_list in all_best_params[scheme][smote_key].items()
            }
    with open(f"{config.RESULTS_DIR}/most_common_hyperparameters.json", "w") as f:
        json.dump(most_common_params, f, indent=2)

    # ---- Pairwise Wilcoxon + matched-pairs effect size on the primary
    # condition (binary + SMOTE). Effect sizes are reported alongside the
    # p-values (not as a replacement) so a non-significant p-value can be
    # read together with how large/consistent the underlying difference
    # actually was.
    primary_aggs = all_aggs["binary"]["smote"]
    wil_acc = ev.pairwise_wilcoxon(primary_aggs, "accuracy_values")
    wil_mcc = ev.pairwise_wilcoxon(primary_aggs, "mcc_values")
    eff_acc = ev.pairwise_effect_sizes(primary_aggs, "accuracy_values")
    eff_mcc = ev.pairwise_effect_sizes(primary_aggs, "mcc_values")
    with open(f"{config.RESULTS_DIR}/wilcoxon_accuracy.json", "w") as f:
        json.dump(wil_acc, f, indent=2)
    with open(f"{config.RESULTS_DIR}/wilcoxon_mcc.json", "w") as f:
        json.dump(wil_mcc, f, indent=2)
    with open(f"{config.RESULTS_DIR}/effect_size_accuracy.json", "w") as f:
        json.dump(eff_acc, f, indent=2)
    with open(f"{config.RESULTS_DIR}/effect_size_mcc.json", "w") as f:
        json.dump(eff_mcc, f, indent=2)

    # ---- Confusion matrices for the primary condition ----
    cms = {}
    for model_name, fr in all_results["binary"]["smote"].items():
        cm = ev.pooled_confusion_matrix(fr, labels=[0, 1])
        cms[model_name] = cm.tolist()
    with open(f"{config.RESULTS_DIR}/confusion_matrices_binary_smote.json", "w") as f:
        json.dump(cms, f, indent=2)

    # ---- Feature importance (legacy, model-specific measures) ----
    # Logistic Regression's standardised coefficients and the Decision
    # Tree's impurity-based importances are still saved here for backward
    # compatibility, but are NOT directly comparable to each other (they are
    # different mathematical quantities on different scales) and are
    # superseded as the paper's primary feature-attribution analysis by the
    # cross-model SHAP comparison in shap_analysis.py -- run that script
    # separately after this one completes.
    y_bin = target_mod.get_target(df, "binary")
    lr_pipe = pl.build_pipeline("LogisticRegression", True)
    lr_pipe.fit(X_df.values, y_bin)
    lr_coef = lr_pipe.named_steps["clf"].coef_.ravel().tolist()

    dt_pipe = pl.build_pipeline("DecisionTree", True)
    dt_pipe.fit(X_df.values, y_bin)
    dt_importance = dt_pipe.named_steps["clf"].feature_importances_.tolist()

    with open(f"{config.RESULTS_DIR}/feature_importance.json", "w") as f:
        json.dump({
            "feature_names": feature_cols,
            "logreg_coef": lr_coef,
            "dtree_importance": dt_importance,
        }, f, indent=2)

    # ---- Raw fold-level metrics for every one of the 20 runs (tables/plots) ----
    rows = []
    for scheme in all_aggs:
        for smote_key in all_aggs[scheme]:
            for model_name, agg in all_aggs[scheme][smote_key].items():
                rows.append({
                    "scheme": scheme, "smote": smote_key, "model": model_name,
                    **{k: v for k, v in agg.items() if not k.endswith("_values")}
                })
    pd.DataFrame(rows).to_csv(f"{config.RESULTS_DIR}/all_runs_summary.csv", index=False)

    print(f"\nDone in {time.time() - t0:.1f}s. Results in {config.RESULTS_DIR}/")
    print("Next: run `python shap_analysis.py` for cross-model feature "
          "attribution, then `python visualize.py` for the figures.")


if __name__ == "__main__":
    main()
