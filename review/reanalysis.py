"""
review/reanalysis.py
Post-hoc checks on the stored results (no re-training of the 28 runs):
  1. Holm-adjusted Wilcoxon p-values over all 21 pairs (binary + SMOTE).
  2. (Friedman test not possible: per-fold values are not stored.)
  3. Confound probe: grouped-CV Random Forest using only non-electrical
     context features (time of day, daily cumulative energy Eg) and only
     operating-level features, to measure how much of the fault/no-fault
     separation is explainable by acquisition context rather than shading.
  4. Per-episode operating summary (median Pg, Va; time window).
Writes review/reanalysis_results.json.
"""
import json, os, sys
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config, features

out = {}
s = json.load(open(os.path.join(ROOT, "results/summary_metrics.json")))["binary"]["smote"]
models = list(s.keys())
for metric in ["accuracy", "mcc"]:
    # Per-fold values are not persisted by run_experiment.py, so the Holm
    # step-down is applied to the stored raw Wilcoxon p-values.
    w = json.load(open(os.path.join(ROOT, f"results/wilcoxon_{metric}.json")))
    pairs = [f"{a} vs {b}" for a, b in combinations(models, 2)]
    ps = [w[a][b] for a, b in combinations(models, 2)]
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    out[f"holm_{metric}"] = {pairs[i]: {"p_raw": round(float(ps[i]), 4), "p_holm": round(float(adj[i]), 4)}
                             for i in order}

df = pd.read_csv(config.SUBSAMPLE_CSV, parse_dates=["time"])
X_all, cols = features.build_features(df)
df = df.loc[X_all.index]  # build_features re-sorts by episode/time
df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60
y = (df["Fault"] != 0).astype(int).values; g = df["episode_id"].values
sets = {"hour_only": df[["hour"]].values,
        "Eg_only": X_all[["Eg"]].values,
        "hour_Eg": np.c_[df["hour"].values, X_all["Eg"].values],
        "Va_only": X_all[["Va"]].values,
        "Pg_Ia_only": X_all[["Pg", "Ia"]].values,
        "raw7": X_all[config.RAW_FEATURES].values,
        "raw7_minus_Eg": X_all[[c for c in config.RAW_FEATURES if c != "Eg"]].values}
probe = {}
for name, X in sets.items():
    acc, mcc, auc = [], [], []
    for r in range(config.N_CV_REPEATS):
        cv = StratifiedGroupKFold(config.N_OUTER_FOLDS, shuffle=True, random_state=config.RANDOM_SEED + r)
        for tr, te in cv.split(X, y, g):
            clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                         max_features="log2", random_state=config.RANDOM_SEED, n_jobs=-1)
            clf.fit(X[tr], y[tr]); p = clf.predict(X[te]); sc = clf.predict_proba(X[te])[:, 1]
            acc.append(accuracy_score(y[te], p)); mcc.append(matthews_corrcoef(y[te], p))
            auc.append(roc_auc_score(y[te], sc))
    probe[name] = {k: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)]
                   for k, v in [("accuracy", acc), ("mcc", mcc), ("auc", auc)]}
    print(name, probe[name])
out["confound_probe_rf_fixed_modal_no_smote"] = probe

ep = df.groupby("episode_id").agg(code=("Fault", "first"), start=("time", "min"), end=("time", "max"),
                                  hour_min=("hour", "min"), hour_max=("hour", "max"),
                                  Pg_median=("Pg", "median"), Va_median=("Va", "median"),
                                  share_Pg_below_20W=("Pg", lambda v: float((v < 20).mean())))
ep["start"] = ep["start"].astype(str); ep["end"] = ep["end"].astype(str)
out["per_episode"] = ep.round(3).reset_index().to_dict(orient="records")
out["class_medians"] = df.groupby("fault_bin")[["hour", "Pg", "Va", "Ia"]].median().round(2).to_dict()
json.dump(out, open(os.path.join(ROOT, "review/reanalysis_results.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k.startswith("holm")}, indent=1))


# ----------------------------------------------------------------------
# 5. Per-episode recall of the binary Random Forest (same fixed modal
#    configuration and SMOTE-in-fold protocol as fault_type_error_analysis.py)
# 6. Extreme values produced by the eps-guarded ratio features
# ----------------------------------------------------------------------
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
Xf = X_all.values
hits = pd.Series(0.0, index=range(len(df))); cnt = pd.Series(0, index=range(len(df)))
for r in range(config.N_CV_REPEATS):
    cv = StratifiedGroupKFold(config.N_OUTER_FOLDS, shuffle=True, random_state=config.RANDOM_SEED + r)
    for tr, te in cv.split(Xf, y, g):
        pipe = ImbPipeline([("smote", SMOTE(random_state=config.RANDOM_SEED, k_neighbors=3)),
                            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                                           max_features="log2", random_state=config.RANDOM_SEED,
                                                           n_jobs=-1))])
        pipe.fit(Xf[tr], y[tr]); p = pipe.predict(Xf[te])
        hits.iloc[te] += (p == y[te]); cnt.iloc[te] += 1
df["_correct"] = (hits / cnt).values
per_ep = df.groupby("episode_id").agg(code=("Fault", "first"), share_correct=("_correct", "mean"))
out["per_episode_rf_share_correct"] = per_ep.round(3).reset_index().to_dict(orient="records")
print(per_ep.round(3).to_string())
out["ratio_feature_extremes"] = {c: {"n_abs_gt_1e3": int((X_all[c].abs() > 1e3).sum()),
                                     "max_abs": float(X_all[c].abs().max())}
                                 for c in ["ratio_Ig_Ia", "ratio_Pg_VaIa"]}
print(out["ratio_feature_extremes"])
json.dump(out, open(os.path.join(ROOT, "review/reanalysis_results.json"), "w"), indent=1, default=str)
