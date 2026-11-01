"""
feature_ablation_rf.py

Supplementary analysis (NOT part of the main 28-run benchmark): does the
engineered features (features.py) help Random Forest, relative to the 7
raw electrical channels alone? Configuration B uses the 18-column temporal
block (16 rolling-window + 2 rate-of-change); the 4 derived instantaneous
features appear only in configuration C, so they are not isolated.

Design constraints (deliberate):
  - Random Forest ONLY (the single best-performing model in the primary
    benchmark; Table VI). No other classifier is touched.
  - NO hyperparameter search. The Random Forest hyperparameters are FIXED
    at the modal configuration already reported in the paper's Table IV-b
    (n_estimators=300, max_depth=8, min_samples_leaf=10, max_features="log2"),
    selected under the SAME nested search as every other result in the
    paper -- reusing it here, rather than re-searching per feature config,
    is what keeps this a lightweight supplementary check rather than a new
    benchmark campaign.
  - SAME outer evaluation protocol as the main experiment: repeated,
    group-aware StratifiedGroupKFold (config.N_CV_REPEATS x
    config.N_OUTER_FOLDS = 15 paired folds when QUICK_TEST_MODE is False),
    SMOTE applied inside each training fold only, binary label scheme
    (the paper's primary condition, Table VI) -- so Configuration C's
    numbers below should closely reproduce Table VI's Random Forest row
    (accuracy 0.838, MCC 0.526) as an internal sanity check. If they don't
    land close to that, something about the environment/data differs from
    the original run and should be investigated before trusting A/B.
  - Does NOT touch prepare_dataset.py, run_experiment.py, or any of the
    other 27 conditions. Only reads the already-produced
    data/reunion_subsample.csv.

Usage: run from the same project folder as config.py, features.py,
target.py, models.py, pipeline.py (i.e. the folder containing
run_experiment.py):

    python feature_ablation_rf.py

Output: results/feature_ablation_rf.json, plus a printed summary table.
These numbers underlie the paper's Table X and Section VII.F/I.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import config
import features
import target as target_mod

# Fixed Random Forest configuration -- Table IV-b modal config, primary
# condition (binary label, SMOTE). NOT re-searched per feature configuration.
RF_FIXED_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    max_features="log2",
    random_state=config.RANDOM_SEED,
    n_jobs=1,
)

TEMPORAL_COLS = []
for col in ["Pg", "Ia", "Vg", "Fg"]:
    for w in config.ROLLING_WINDOWS:
        TEMPORAL_COLS += [f"{col}_rmean{w}", f"{col}_rstd{w}"]
TEMPORAL_COLS += ["Pg_diff1", "Ia_diff1"]

CONFIGS = {
    "A_raw_only": list(config.RAW_FEATURES),                       # 7 cols
    "B_temporal_only": list(TEMPORAL_COLS),                        # 18 cols
    "C_combined": None,                                            # all 29 -- filled in below
}


def run_one_config(X_full, feature_cols, cols_to_use, y, groups):
    """Repeated group-aware CV for Random Forest, fixed hyperparameters,
    SMOTE inside each training fold, restricted to `cols_to_use` columns.
    Mirrors pipeline.run_nested_cv's outer-split logic exactly, minus the
    inner RandomizedSearchCV.
    """
    col_idx = [feature_cols.index(c) for c in cols_to_use]
    X = np.asarray(X_full)[:, col_idx]
    y = np.asarray(y)
    groups = np.asarray(groups)

    accs, mccs, aucs = [], [], []
    for repeat in range(config.N_CV_REPEATS):
        outer = StratifiedGroupKFold(
            n_splits=config.N_OUTER_FOLDS, shuffle=True,
            random_state=config.RANDOM_SEED + repeat,
        )
        for train_idx, test_idx in outer.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe = ImbPipeline([
                ("smote", SMOTE(random_state=config.RANDOM_SEED, k_neighbors=3)),
                ("clf", RandomForestClassifier(**RF_FIXED_PARAMS)),
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_score = pipe.predict_proba(X_test)[:, 1]

            accs.append(accuracy_score(y_test, y_pred))
            mccs.append(matthews_corrcoef(y_test, y_pred))
            try:
                aucs.append(roc_auc_score(y_test, y_score))
            except Exception:
                aucs.append(np.nan)

    return {
        "n_features": len(cols_to_use),
        "feature_names": cols_to_use,
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "mcc_mean": float(np.mean(mccs)), "mcc_std": float(np.std(mccs)),
        "auc_roc_mean": float(np.nanmean(aucs)), "auc_roc_std": float(np.nanstd(aucs)),
        "n_folds": len(accs),
    }


def main():
    df = pd.read_csv(config.SUBSAMPLE_CSV, parse_dates=["time"])
    X_full, feature_cols = features.build_features(df)
    CONFIGS["C_combined"] = list(feature_cols)

    y = target_mod.binary_target(df)
    groups = df["episode_id"].values

    results = {}
    for name, cols in CONFIGS.items():
        print(f"Running Random Forest, fixed hyperparameters, feature set '{name}' "
              f"({len(cols)} features) ...")
        results[name] = run_one_config(X_full, feature_cols, cols, y, groups)
        r = results[name]
        print(f"  accuracy = {r['accuracy_mean']:.3f} +/- {r['accuracy_std']:.3f}   "
              f"MCC = {r['mcc_mean']:.3f} +/- {r['mcc_std']:.3f}   "
              f"AUC-ROC = {r['auc_roc_mean']:.3f} +/- {r['auc_roc_std']:.3f}")

    out_path = os.path.join(config.RESULTS_DIR, "feature_ablation_rf.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print("\nSanity check: Configuration C (combined, all 29 features) should be close to "
          "the paper's Table VI Random Forest row (accuracy 0.838, MCC 0.526). "
          "Small differences are expected (this script skips the inner hyperparameter "
          "search and uses the fixed modal configuration instead), but a large "
          "discrepancy means something else differs and should be checked before "
          "trusting Configurations A and B.")


if __name__ == "__main__":
    main()
