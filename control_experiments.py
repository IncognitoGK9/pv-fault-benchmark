"""
control_experiments.py

Three control analyses on the EXISTING 20,800-row benchmark
(data/reunion_subsample.csv). No raw CSV and no hyperparameter search: every
classifier uses its modal binary-scheme setting (results/most_common_
hyperparameters.json, paper Table IV-b), with SMOTE inside the training fold
and the same outer protocol as the paper (3 repeats x 5-fold stratified
group k-fold, seed 42 + repeat).

  Experiment 1  Hour-matched check. Same protocol on (a) the full benchmark
                and (b) the hour-matched benchmark: all 6,800 fault rows plus
                only the no-fault rows logged between 10:00 and 17:00 local
                time. Compares accuracy, MCC, ranking, the SHAP rank of Va,
                and an hour-of-day-only classifier.
  Experiment 2  Correlated-fold tests. From the per-fold results of 1(a):
                corrected resampled t-test (Nadeau & Bengio, 2003) with Holm
                adjustment, and Bayesian correlated t-test (Corani &
                Benavoli, 2015) with a 0.01 region of practical equivalence.
  Experiment 3  Random-split control. Same models with a plain stratified
                5-fold split that ignores episodes, against 1(a).

Outputs in results/controls/ (or results/controls_quick/ with --quick):
  log.txt, per_fold.json, experiment1_hour_matched.json,
  experiment2_corrected_tests.json, experiment3_random_split.json,
  SUMMARY.txt (plain-language answers).

Usage (normally via run_all.sh):  python control_experiments.py [--quick]
"""
import ast, itertools, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE)
import config, features, target as tm, pipeline as pl, evaluate as ev
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

QUICK = "--quick" in sys.argv
REPEATS = 1 if QUICK else 3
FOLDS = config.N_OUTER_FOLDS
SEED = config.RANDOM_SEED
MODELS = list(config.MODEL_NAMES)
RANDOM_SPLIT_MODELS = [m for m in MODELS if m != "SVM"]   # SVM omitted only to save time
OUT = os.path.join(config.RESULTS_DIR, "controls_quick" if QUICK else "controls")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "log.txt"), "w")
T0 = time.time()
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:6.1f} min] {msg}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
def dump(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=2, default=float)

# ------------------------------------------------------------ settings
FALLBACK = {
    "LogisticRegression": {"clf__penalty": "l2", "clf__C": 0.001},
    "kNN": {"clf__weights": "uniform", "clf__p": 1, "clf__n_neighbors": 3},
    "DecisionTree": {"clf__min_samples_leaf": 1, "clf__max_depth": 3, "clf__criterion": "gini"},
    "SVM": {"clf__gamma": 0.01, "clf__C": 0.1},
    "MLP": {"clf__learning_rate_init": 0.0005, "clf__hidden_layer_sizes": (64,), "clf__alpha": 0.0001},
    "RandomForest": {"clf__n_estimators": 300, "clf__min_samples_leaf": 10, "clf__max_features": "log2", "clf__max_depth": 8},
    "XGBoost": {"clf__subsample": 0.7, "clf__n_estimators": 100, "clf__max_depth": 3, "clf__learning_rate": 0.01, "clf__colsample_bytree": 0.85},
}
def cast(v):
    if not isinstance(v, str): return v
    if v == "None": return None
    try: return ast.literal_eval(v)
    except (ValueError, SyntaxError): return v
try:
    mc = json.load(open(os.path.join(config.RESULTS_DIR, "most_common_hyperparameters.json")))["binary"]["smote"]
    PARAMS = {m: {k: cast(v) for k, v in mc[m].items()} for m in MODELS}
    src = "results/most_common_hyperparameters.json"
except Exception:
    PARAMS, src = FALLBACK, "built-in copy of Table IV-b"
log(f"Mode: {'QUICK (1 repeat)' if QUICK else 'FULL (3 repeats)'}; settings from {src}")

# ---------------------------------------------------------------- data
df = pd.read_csv(config.SUBSAMPLE_CSV)
X_df, cols = features.build_features(df)
d = df.loc[X_df.index]
X = X_df.values
y = np.asarray(tm.get_target(d, "binary"))
g = d["episode_id"].values
lt = pd.to_datetime(d["time"], utc=True).dt.tz_convert("Indian/Reunion")
hour = (lt.dt.hour + lt.dt.minute / 60).values
FULL = np.ones(len(y), bool)
MATCHED = (y == 1) | ((hour >= 10) & (hour < 17))
log(f"Full benchmark: {FULL.sum():,} rows ({(y==1).sum():,} fault). "
    f"Hour-matched: {MATCHED.sum():,} rows ({(y[MATCHED]==0).sum():,} no-fault between 10:00 and 17:00)")

def make(m):
    p = pl.build_pipeline(m, use_smote=True, seed=SEED); p.set_params(**PARAMS[m]); return p

def run_cv(mask, splitter, models, label, use_groups=True):
    Xm, ym, gm, hm = X[mask], y[mask], g[mask], hour[mask]
    res = {m: [] for m in models}; res["HourOnly"] = []
    for r in range(REPEATS):
        cv = splitter(r)
        splits = cv.split(Xm, ym, gm) if use_groups else cv.split(Xm, ym)
        for f, (tr, te) in enumerate(splits):
            for m in models:
                t = time.time(); pipe = make(m).fit(Xm[tr], ym[tr])
                pred = pipe.predict(Xm[te])
                try: score = pipe.predict_proba(Xm[te])
                except Exception: score = None
                met = ev.fold_metrics(ym[te], pred, score, n_classes=2)
                met.update(n_train=int(len(tr)), n_test=int(len(te)), seconds=round(time.time() - t, 1))
                res[m].append({k: (float(v) if v is not None else None) for k, v in met.items()})
            hp = DecisionTreeClassifier(max_depth=3, random_state=0).fit(hm[tr].reshape(-1, 1), ym[tr]).predict(hm[te].reshape(-1, 1))
            res["HourOnly"].append({k: float(v) for k, v in ev.fold_metrics(ym[te], hp, None, 2).items() if k != "auc_roc"})
            log(f"  {label}: repeat {r+1}/{REPEATS}, fold {f+1}/{FOLDS} done")
    return res

def means(res):
    return {m: {k: round(float(np.nanmean([fr[k] for fr in v])), 3) for k in ("accuracy", "mcc")} for m, v in res.items()}

def shap_va_rank(mask):
    import shap
    out = {}
    Xm, ym = X[mask], y[mask]
    rng = np.random.default_rng(0); idx = rng.choice(len(ym), min(2000, len(ym)), replace=False)
    for m in ("DecisionTree", "RandomForest", "XGBoost", "LogisticRegression"):
        try:
            pipe = make(m).fit(Xm, ym); clf = pipe.named_steps["clf"]
            Xe = pipe.named_steps["scaler"].transform(Xm[idx]) if "scaler" in pipe.named_steps else Xm[idx]
            if m == "LogisticRegression":
                sv = shap.LinearExplainer(clf, Xe[:100]).shap_values(Xe)
            else:
                sv = shap.TreeExplainer(clf).shap_values(Xe)
            sv = sv[1] if isinstance(sv, list) else np.asarray(sv)
            if sv.ndim == 3: sv = sv[:, :, 1]
            imp = np.abs(sv).mean(0); share = imp / imp.sum(); order = list(np.argsort(-share))
            out[m] = {"Va_rank": order.index(cols.index("Va")) + 1,
                      "top5": [[cols[i], round(float(share[i]), 3)] for i in order[:5]]}
        except Exception as exc:  # noqa: BLE001
            out[m] = {"error": str(exc)}
    return out

grouped = lambda r: StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + r)
plain = lambda r: StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + r)

# ------------------------------------------------------- Experiment 1
log("Experiment 1a: full benchmark, episode-grouped folds")
res_full = run_cv(FULL, grouped, MODELS, "1a full")
dump("per_fold.json", {"full_grouped": res_full})
log("Experiment 1b: hour-matched benchmark, episode-grouped folds")
res_match = run_cv(MATCHED, grouped, MODELS, "1b hour-matched")
dump("per_fold.json", {"full_grouped": res_full, "hour_matched_grouped": res_match})
log("Experiment 1c: SHAP rank of Va (tree and linear models)")
shap_full, shap_match = shap_va_rank(FULL), shap_va_rank(MATCHED)
mf, mm = means(res_full), means(res_match)
rank = lambda mm_: [m for m, _ in sorted(((m, mm_[m]["accuracy"]) for m in MODELS), key=lambda x: -x[1])]
rho = stats.spearmanr([mf[m]["accuracy"] for m in MODELS], [mm[m]["accuracy"] for m in MODELS]).correlation
exp1 = {"full": mf, "hour_matched": mm,
        "majority_accuracy": {"full": float((y == 0).mean()), "hour_matched": float((y[MATCHED] == 0).mean())},
        "ranking_full": rank(mf), "ranking_hour_matched": rank(mm), "spearman_accuracy_rank": float(rho),
        "shap_full": shap_full, "shap_hour_matched": shap_match}
dump("experiment1_hour_matched.json", exp1)
log(f"Experiment 1 done. Spearman rank correlation of accuracy (full vs hour-matched): {rho:.2f}")

# ------------------------------------------------------- Experiment 2
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj
exp2 = {}
for metric in ("accuracy", "mcc"):
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        da = np.array([fr[metric] for fr in res_full[a]]); db = np.array([fr[metric] for fr in res_full[b]])
        diff = da - db; k = len(diff)
        ntr = np.mean([fr["n_train"] for fr in res_full[a]]); nte = np.mean([fr["n_test"] for fr in res_full[a]])
        s2 = diff.var(ddof=1); mean = diff.mean()
        if s2 == 0:
            p_nb = 1.0 if mean == 0 else 0.0; pl_, pe, pr_ = (0.0, 1.0, 0.0) if abs(mean) <= 0.01 else ((1.0, 0.0, 0.0) if mean < 0 else (0.0, 0.0, 1.0))
        else:
            t_nb = mean / np.sqrt((1 / k + nte / ntr) * s2)
            p_nb = float(2 * stats.t.sf(abs(t_nb), k - 1))
            rho_ = nte / (nte + ntr)
            post = stats.t(df=k - 1, loc=mean, scale=np.sqrt((1 / k + rho_ / (1 - rho_)) * s2))
            pl_, pr_ = float(post.cdf(-0.01)), float(post.sf(0.01)); pe = 1 - pl_ - pr_
        try: p_w = float(stats.wilcoxon(da, db).pvalue)
        except ValueError: p_w = 1.0
        rows.append({"a": a, "b": b, "mean_diff": round(float(mean), 4), "p_wilcoxon": p_w,
                     "p_corrected_t": p_nb, "P_a_better": round(pr_, 3), "P_equivalent": round(pe, 3), "P_b_better": round(pl_, 3)})
    for key, src_ in (("p_wilcoxon_holm", "p_wilcoxon"), ("p_corrected_t_holm", "p_corrected_t")):
        for r_, adj in zip(rows, holm([r_[src_] for r_ in rows])): r_[key] = adj
    exp2[metric] = rows
dump("experiment2_corrected_tests.json", exp2)
log("Experiment 2 done")

# ------------------------------------------------------- Experiment 3
log("Experiment 3: random (episode-ignoring) split")
res_rand = run_cv(FULL, plain, RANDOM_SPLIT_MODELS, "3 random split", use_groups=False)
mr = means(res_rand)
exp3 = {m: {"grouped_accuracy": mf[m]["accuracy"], "random_accuracy": mr[m]["accuracy"],
            "inflation_points": round(100 * (mr[m]["accuracy"] - mf[m]["accuracy"]), 1),
            "grouped_mcc": mf[m]["mcc"], "random_mcc": mr[m]["mcc"]} for m in RANDOM_SPLIT_MODELS}
dump("experiment3_random_split.json", exp3)
dump("per_fold.json", {"full_grouped": res_full, "hour_matched_grouped": res_match, "full_random": res_rand})
log("Experiment 3 done")

# ---------------------------------------------------------- summary
L = []
L.append("CONTROL EXPERIMENTS: PLAIN-LANGUAGE SUMMARY")
L.append(f"Mode: {'QUICK (1 repeat; not for the paper)' if QUICK else 'FULL (3 repeats x 5 folds)'}; fixed modal settings, SMOTE in training folds.\n")
L.append("EXPERIMENT 1: does the result depend on time of day?")
L.append(f"  {'Model':20s} {'Acc full':>9s} {'Acc matched':>12s} {'MCC full':>9s} {'MCC matched':>12s}")
for m in MODELS + ["HourOnly"]:
    L.append(f"  {m:20s} {mf[m]['accuracy']:9.3f} {mm[m]['accuracy']:12.3f} {mf[m]['mcc']:9.3f} {mm[m]['mcc']:12.3f}")
L.append(f"  Majority-class accuracy: full {exp1['majority_accuracy']['full']:.3f}, matched {exp1['majority_accuracy']['hour_matched']:.3f}")
L.append(f"  Ranking full:    {', '.join(exp1['ranking_full'])}")
L.append(f"  Ranking matched: {', '.join(exp1['ranking_hour_matched'])}")
L.append(f"  Spearman rank correlation: {rho:.2f}")
for m in ("DecisionTree", "RandomForest", "XGBoost", "LogisticRegression"):
    a, b = shap_full.get(m, {}), shap_match.get(m, {})
    L.append(f"  Va SHAP rank, {m}: full {a.get('Va_rank','?')}, matched {b.get('Va_rank','?')}")
L.append("\nEXPERIMENT 2: which differences survive tests built for correlated folds? (full benchmark)")
for metric in ("accuracy", "mcc"):
    sig = [f"{r_['a']} vs {r_['b']} (adj p={r_['p_corrected_t_holm']:.3f})" for r_ in exp2[metric] if r_["p_corrected_t_holm"] < 0.05]
    L.append(f"  {metric}: significant after corrected t-test + Holm: {', '.join(sig) if sig else 'none'}")
upper = ["RandomForest", "XGBoost", "LogisticRegression", "DecisionTree"]
for r_ in exp2["accuracy"]:
    if r_["a"] in upper and r_["b"] in upper:
        L.append(f"  {r_['a']} vs {r_['b']}: P(practically equivalent, |diff|<=0.01) = {r_['P_equivalent']:.2f}")
L.append("\nEXPERIMENT 3: does ignoring episodes inflate accuracy?")
for m, v in exp3.items():
    L.append(f"  {m:20s} grouped {v['grouped_accuracy']:.3f}  random {v['random_accuracy']:.3f}  inflation {v['inflation_points']:+.1f} points")
L.append(f"\nTotal time: {(time.time()-T0)/60:.1f} min")
open(os.path.join(OUT, "SUMMARY.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L), flush=True)
log("CONTROLS DONE")
