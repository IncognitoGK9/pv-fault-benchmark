"""
fault_type_error_analysis.py

Supplementary analysis (NOT part of the main 28-run benchmark): break the
pooled binary confusion matrix (Table VII) down by the ORIGINAL fault code,
to answer "which fault type is easiest/hardest to detect?" -- a question
the binary Fault/No-Fault confusion matrix alone cannot answer, since it
collapses all eight fault codes into one "Fault" class.

Two complementary breakdowns are produced, both for Random Forest only
(same rationale as feature_ablation_rf.py: it is the best-performing model
and this is a supplementary check, not a new benchmark campaign), with NO
hyperparameter search (fixed modal configuration, Table IV-b):

  1. Per-fault-code RECALL under the existing binary classifier: of all
     rows genuinely labelled with fault code X (e.g. 2.1, or 4.0), what
     fraction did the binary Fault/No-Fault classifier correctly flag as
     "Fault"? This is well-defined for every individual fault code,
     including "No-fault" itself (= specificity), and directly answers
     "which fault type is hardest to detect" without changing the
     classification target.

  2. Genuine multiclass Precision/Recall/F1 per TIER class (No-Fault,
     Constant-Partial-Shading, Severe -- the paper's existing three-tier
     scheme, Section III.C), using the SAME modal Random Forest
     hyperparameters as the tier+SMOTE condition already reported in
     Table IX. Precision is only well-defined in a genuine multiclass
     setting, which is why this is reported at the tier level (mapping:
     Severe = Uniform shading [1.0] + Intermittent partial shading [4.0];
     Constant-Partial-Shading = codes 2.1/2.2/2.3/3.1/3.2) rather than
     invented for the binary classifier's two-class output.

Usage: run from the same project folder as config.py, features.py,
target.py, models.py, pipeline.py, alongside run_experiment.py:

    python fault_type_error_analysis.py

Output: results/fault_type_error_analysis.json, plus a printed summary.
These numbers underlie the paper's Results table / Discussion subsection
on per-fault-type performance (Tables XI, XI-b; Section VII.J).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import config
import features
import target as target_mod

RF_FIXED_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    max_features="log2",
    random_state=config.RANDOM_SEED,
    n_jobs=1,
)


def out_of_fold_predictions(X, y, groups, n_classes):
    """Repeated group-aware CV (same outer-split protocol as pipeline.py),
    fixed Random Forest hyperparameters, SMOTE inside each training fold
    only. Returns pooled (row_index, y_true, y_pred) across every outer
    fold of every repeat, so every row is predicted exactly
    config.N_CV_REPEATS times (once per repeat) as an out-of-fold
    prediction -- matching how Table VII's pooled confusion matrix is built.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    all_idx, all_true, all_pred = [], [], []
    for repeat in range(config.N_CV_REPEATS):
        outer = StratifiedGroupKFold(
            n_splits=config.N_OUTER_FOLDS, shuffle=True,
            random_state=config.RANDOM_SEED + repeat,
        )
        for train_idx, test_idx in outer.split(X, y, groups):
            pipe = ImbPipeline([
                ("smote", SMOTE(random_state=config.RANDOM_SEED, k_neighbors=3)),
                ("clf", RandomForestClassifier(**RF_FIXED_PARAMS)),
            ])
            pipe.fit(X[train_idx], y[train_idx])
            y_pred = pipe.predict(X[test_idx])
            all_idx.append(test_idx)
            all_true.append(y[test_idx])
            all_pred.append(y_pred)

    return (np.concatenate(all_idx), np.concatenate(all_true), np.concatenate(all_pred))


def per_fault_code_recall(df, row_idx, y_true_bin, y_pred_bin):
    """Analysis 1: recall per original fault code under the binary
    classifier's predictions, pooled across all repeats/folds.
    """
    fault_codes = df["Fault"].values[row_idx]
    out = {}
    for code in sorted(np.unique(fault_codes)):
        mask = fault_codes == code
        n = int(mask.sum())
        detected = int(((y_pred_bin[mask] == 1)).sum()) if code != 0.0 else int(((y_pred_bin[mask] == 0)).sum())
        rate = detected / n if n else float("nan")
        label = config.FAULT_CODE_NAMES.get(code, str(code))
        metric_name = "specificity (correctly No-Fault)" if code == 0.0 else "recall (correctly flagged Fault)"
        out[str(code)] = {
            "name": label, "n_rows": n, metric_name: rate,
        }
    return out


def main():
    df = pd.read_csv(config.SUBSAMPLE_CSV, parse_dates=["time"])
    X, feature_cols = features.build_features(df)
    groups = df["episode_id"].values

    print("Analysis 1: per-fault-code recall under the binary classifier ...")
    y_bin = target_mod.binary_target(df)
    idx_bin, true_bin, pred_bin = out_of_fold_predictions(X, y_bin, groups, n_classes=2)
    recall_by_code = per_fault_code_recall(df, idx_bin, true_bin, pred_bin)
    for code, info in recall_by_code.items():
        metric_key = [k for k in info if k not in ("name", "n_rows")][0]
        print(f"  Fault {code} ({info['name']}, n={info['n_rows']}): "
              f"{metric_key} = {info[metric_key]:.3f}")

    print("\nAnalysis 2: multiclass Precision/Recall/F1 per tier class ...")
    y_tier = target_mod.tier_target(df)
    idx_tier, true_tier, pred_tier = out_of_fold_predictions(X, y_tier, groups, n_classes=3)
    tier_labels = [0, 1, 2]
    tier_names = [config.TIER_NAMES[t] for t in tier_labels]
    report = classification_report(
        true_tier, pred_tier, labels=tier_labels, target_names=tier_names,
        output_dict=True, zero_division=0,
    )
    print(classification_report(true_tier, pred_tier, labels=tier_labels,
                                 target_names=tier_names, zero_division=0))

    results = {
        "per_fault_code_recall_binary_classifier": recall_by_code,
        "per_tier_class_classification_report": report,
        "note": "Random Forest only, fixed modal hyperparameters (Table IV-b), "
                "no hyperparameter search; pooled out-of-fold predictions across "
                f"{config.N_CV_REPEATS * config.N_OUTER_FOLDS} outer folds.",
    }
    out_path = os.path.join(config.RESULTS_DIR, "fault_type_error_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
