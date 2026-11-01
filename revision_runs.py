"""
revision_runs.py

One consolidated set of revision analyses on the EXISTING 20,800-row
benchmark (data/reunion_subsample.csv). It does not repeat the nested
hyperparameter search: every classifier uses its modal binary-scheme setting
(results/most_common_hyperparameters.json, paper Table IV-b), with SMOTE
inside the training fold, exactly as in control_experiments.py. The outer
protocol is the paper's (3 repeats x 5 folds, seed 42 + repeat) unless a
section says otherwise.

  R1  Fold audit (no training). Composition of every outer test fold of the
      main benchmark, binary and tier schemes, with the identical splitter
      and seeds used by run_experiment.py.
  R2  Pooled out-of-fold metrics of the main benchmark, from the saved
      pooled confusion matrices (Table VII). No training.
  R3  Data audit (no training). Per-episode date, local time, median grid
      power and PV-side voltage; the behaviour of the Eg register; PV-side
      voltage by class at comparable output (grid power >= 600 W); rows with
      zero output; sampling interval.
  R4  Grouped folds, all 29 features (reproduces control Experiment 1a) with
      every out-of-fold prediction saved, so that per-episode and
      per-fault-code recall can be reported for all seven classifiers.
  R5  Grouped folds without Eg (28 features): is any result carried by the
      daily cumulative energy register?
  R6  Reference baseline: one-split decision stump on PV-side voltage only,
      same folds.
  R7  Balanced-episode folds: a splitter that puts at least one no-fault
      episode in every test fold, so every fold has both classes. Repeats
      the seven models and the stump, with Wilcoxon, corrected resampled
      t-test and Bayesian correlated t-test.
  R8  Random split ignoring episodes, all seven classifiers (completes
      control Experiment 3, which omitted the SVM).
  R9  Figures for the paper.

Outputs: results/revision/ (SUMMARY.txt plus JSON/CSV files and the saved
predictions) and figures/revision/.

Usage (Python 3.9 environment of requirements.txt):
    python revision_runs.py            # full run, about 30 to 60 min on a laptop
    python revision_runs.py --quick    # 1 repeat, for a smoke test only
"""
import ast, itertools, json, os, sys, time, warnings
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE)
import config, features, target as tm, pipeline as pl, evaluate as ev
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from scipy import stats
from joblib import Parallel, delayed

QUICK = "--quick" in sys.argv
REPEATS = 1 if QUICK else 3
FOLDS = config.N_OUTER_FOLDS
SEED = config.RANDOM_SEED
MODELS = list(config.MODEL_NAMES)
NAMES = {"LogisticRegression": "Logistic Regression", "kNN": "k-Nearest Neighbours",
         "DecisionTree": "Decision Tree", "SVM": "SVM (RBF)", "MLP": "MLP",
         "RandomForest": "Random Forest", "XGBoost": "XGBoost", "VaStump": "Va stump"}
N_JOBS = int(os.environ.get("REVISION_JOBS", "2"))
OUT = os.path.join(config.RESULTS_DIR, "revision_quick" if QUICK else "revision")
FIG = os.path.join(config.FIG_DIR, "revision_quick" if QUICK else "revision")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(OUT, "log.txt"), "w")
T0 = time.time()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:6.1f} min] {msg}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


def dump(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=2, default=float)


# ------------------------------------------------------------ settings
def cast(v):
    if not isinstance(v, str): return v
    if v == "None": return None
    try: return ast.literal_eval(v)
    except (ValueError, SyntaxError): return v


mc = json.load(open(os.path.join(config.RESULTS_DIR, "most_common_hyperparameters.json")))["binary"]["smote"]
PARAMS = {m: {k: cast(v) for k, v in mc[m].items()} for m in MODELS}

# ---------------------------------------------------------------- data
df = pd.read_csv(config.SUBSAMPLE_CSV)
X_df, COLS = features.build_features(df)
assert (X_df.index.values == np.arange(len(df))).all(), "feature rows must stay in CSV order"
d = df.loc[X_df.index].reset_index(drop=True)
X_ALL = X_df.values
y = np.asarray(tm.get_target(d, "binary"))
y_tier = np.asarray(tm.get_target(d, "tier"))
g = d["episode_id"].values
code = d["Fault"].values
lt = pd.to_datetime(d["time"], utc=True).dt.tz_convert("Indian/Reunion")
VA = COLS.index("Va")
EG = COLS.index("Eg")
COLS_NO_EG = [c for c in COLS if c != "Eg"]
X_NO_EG = X_df[COLS_NO_EG].values
log(f"Mode: {'QUICK' if QUICK else 'FULL'}; {len(y):,} rows, {len(COLS)} features, "
    f"{len(np.unique(g))} episodes; settings from results/most_common_hyperparameters.json")


# ------------------------------------------------------------ splitters
def grouped(r):
    return StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + r)


def plain(r):
    return StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + r)


class BalancedEpisodeSplit:
    """Episode-grouped K-fold in which every test fold holds at least one
    no-fault episode (7 no-fault episodes over 5 folds: 2, 2, 1, 1, 1) and
    the 17 fault episodes are dealt so that row counts stay comparable."""

    def __init__(self, n_splits, seed):
        self.k, self.seed = n_splits, seed

    def split(self, X, yv, groups):
        rng = np.random.default_rng(self.seed)
        eps = pd.Series(yv).groupby(groups).first()
        nf = rng.permutation(eps.index[eps.values == 0].values)
        fe = rng.permutation(eps.index[eps.values == 1].values)
        fold_of = {}
        for i, e in enumerate(nf): fold_of[e] = i % self.k
        n_nf = np.bincount([fold_of[e] for e in nf], minlength=self.k)
        order = list(np.argsort(n_nf, kind="stable"))   # folds with fewer no-fault rows first
        for i, e in enumerate(fe): fold_of[e] = order[i % self.k]
        fid = np.array([fold_of[e] for e in groups])
        for k in range(self.k):
            yield np.where(fid != k)[0], np.where(fid == k)[0]


def balanced(r):
    return BalancedEpisodeSplit(FOLDS, SEED + r)


def splits_for(splitter, use_groups=True):
    out = []
    for r in range(REPEATS):
        cv = splitter(r)
        it = cv.split(X_ALL, y, g) if use_groups else cv.split(X_ALL, y)
        out += [(r, f, tr, te) for f, (tr, te) in enumerate(it)]
    return out


# --------------------------------------------------------- model running
def make(m):
    if m == "VaStump":
        return DecisionTreeClassifier(max_depth=1, random_state=SEED)
    p = pl.build_pipeline(m, use_smote=True, seed=SEED); p.set_params(**PARAMS[m]); return p


def fit_predict(m, Xmat, tr, te):
    t = time.time()
    if m == "VaStump":
        clf = make(m).fit(X_ALL[tr][:, [VA]], y[tr])
        pred = clf.predict(X_ALL[te][:, [VA]]); score = clf.predict_proba(X_ALL[te][:, [VA]])
        extra = {"threshold_V": float(clf.tree_.threshold[0])}
    else:
        pipe = make(m).fit(Xmat[tr], y[tr])
        pred = pipe.predict(Xmat[te])
        try: score = pipe.predict_proba(Xmat[te])
        except Exception: score = None
        extra = {}
    met = ev.fold_metrics(y[te], pred, score, n_classes=2)
    met = {k: (float(v) if v is not None else None) for k, v in met.items()}
    met.update(n_train=int(len(tr)), n_test=int(len(te)), two_class=bool(len(np.unique(y[te])) == 2),
               seconds=round(time.time() - t, 1), **extra)
    return met, pred


def run_setting(label, splitter, models, Xmat, use_groups=True):
    sp = splits_for(splitter, use_groups)
    jobs = [(m, r, f, tr, te) for m in models for (r, f, tr, te) in sp]
    log(f"{label}: {len(models)} models x {len(sp)} folds")
    outs = Parallel(n_jobs=N_JOBS)(delayed(fit_predict)(m, Xmat, tr, te) for (m, r, f, tr, te) in jobs)
    per_fold = {m: [] for m in models}
    preds = {m: np.full((REPEATS, len(y)), -1, dtype=np.int8) for m in models}
    for (m, r, f, tr, te), (met, pred) in zip(jobs, outs):
        met.update(repeat=r, fold=f); per_fold[m].append(met); preds[m][r, te] = pred
    log(f"{label}: done")
    return per_fold, preds


def pooled(pred_rep):
    """Pooled out-of-fold metrics over all repeats (rows x repeats)."""
    yt = np.tile(y, pred_rep.shape[0]); yp = pred_rep.ravel()
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {"accuracy": (tp + tn) / len(yt), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1), "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
            "mcc": matthews_corrcoef(yt, yp), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def fold_summary(folds):
    out = {}
    for k in ("accuracy", "precision", "recall", "f1", "mcc", "auc_roc"):
        allv = np.array([f[k] if f[k] is not None else np.nan for f in folds], float)
        two = np.array([f[k] if f[k] is not None else np.nan for f in folds if f["two_class"]], float)
        out[k] = {"mean_all_folds": float(np.nanmean(allv)), "sd_all_folds": float(np.nanstd(allv)),
                  "n_all": int(np.sum(~np.isnan(allv))),
                  "mean_two_class_folds": float(np.nanmean(two)) if len(two) else None,
                  "sd_two_class_folds": float(np.nanstd(two)) if len(two) else None, "n_two_class": int(len(two))}
    return out


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def pairwise_tests(per_fold, models):
    res = {}
    for metric in ("accuracy", "mcc"):
        rows = []
        for a, b in itertools.combinations(models, 2):
            da = np.array([f[metric] for f in per_fold[a]]); db = np.array([f[metric] for f in per_fold[b]])
            diff = da - db; k = len(diff)
            ntr = np.mean([f["n_train"] for f in per_fold[a]]); nte = np.mean([f["n_test"] for f in per_fold[a]])
            s2 = diff.var(ddof=1); mean = diff.mean()
            if s2 == 0:
                p_nb = 1.0 if mean == 0 else 0.0
                pl_, pe, pr_ = (0.0, 1.0, 0.0) if abs(mean) <= 0.01 else ((1.0, 0.0, 0.0) if mean < 0 else (0.0, 0.0, 1.0))
            else:
                t_nb = mean / np.sqrt((1 / k + nte / ntr) * s2)
                p_nb = float(2 * stats.t.sf(abs(t_nb), k - 1))
                rho = nte / (nte + ntr)
                post = stats.t(df=k - 1, loc=mean, scale=np.sqrt((1 / k + rho / (1 - rho)) * s2))
                pl_, pr_ = float(post.cdf(-0.01)), float(post.sf(0.01)); pe = 1 - pl_ - pr_
            try: p_w = float(stats.wilcoxon(da, db).pvalue)
            except ValueError: p_w = 1.0
            rows.append({"a": a, "b": b, "mean_diff": float(mean), "n_folds": int(k),
                         "n_nonzero_diffs": int(np.sum(diff != 0)), "p_wilcoxon": p_w, "p_corrected_t": p_nb,
                         "P_a_better": pr_, "P_equivalent": pe, "P_b_better": pl_})
        for key, src in (("p_wilcoxon_holm", "p_wilcoxon"), ("p_corrected_t_holm", "p_corrected_t")):
            for r_, adj in zip(rows, holm([r_[src] for r_ in rows])): r_[key] = adj
        res[metric] = rows
    return res


def per_unit_recall(preds, models):
    """Recall per fault episode (specificity per no-fault episode) and per
    fault code, pooled over repeats."""
    ep_rows, code_rows = [], []
    for m in models:
        P = preds[m]; ok = (P == np.tile(y, (P.shape[0], 1)))
        for e in np.unique(g):
            msk = g == e
            ep_rows.append({"model": m, "episode_id": int(e), "label": int(y[msk][0]), "code": float(code[msk][0]),
                            "rate": float(ok[:, msk].mean())})
        for c in np.unique(code):
            msk = code == c
            code_rows.append({"model": m, "code": float(c), "n_rows_pooled": int(msk.sum() * P.shape[0]),
                              "n_episodes": int(len(np.unique(g[msk]))), "rate": float(ok[:, msk].mean())})
    return pd.DataFrame(ep_rows), pd.DataFrame(code_rows)


S = []   # plain-language summary lines
# =================================================================== R1
log("R1 fold audit")
audit = []
for scheme, yy in (("binary", y), ("tier", y_tier)):
    for r in range(3):
        for f, (tr, te) in enumerate(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED + r).split(X_ALL, yy, g)):
            audit.append({"scheme": scheme, "repeat": r + 1, "fold": f + 1, "n_test": int(len(te)),
                          "no_fault_episodes": int(len(set(g[te][y[te] == 0]))),
                          "fault_episodes": int(len(set(g[te][y[te] == 1]))),
                          "fault_share": round(float(y[te].mean()), 3)})
audit = pd.DataFrame(audit); audit.to_csv(os.path.join(OUT, "R1_fold_audit.csv"), index=False)
nb = int(((audit.scheme == "binary") & (audit.no_fault_episodes == 0)).sum())
nt = int(((audit.scheme == "tier") & (audit.no_fault_episodes == 0)).sum())
S += ["R1 FOLD AUDIT (main benchmark splitter, seeds 42 to 44)",
      f"  Binary scheme: {nb} of 15 outer test folds contain no no-fault episode (only fault rows).",
      f"  Tier scheme:   {nt} of 15 outer test folds contain no no-fault episode.",
      "  In a fault-only fold MCC is undefined (scikit-learn returns 0), precision is 1 whenever any",
      "  fault is flagged, and AUC-ROC cannot be computed (NaN, skipped by the mean).", ""]

# =================================================================== R2
log("R2 pooled metrics from saved confusion matrices")
cm = json.load(open(os.path.join(config.RESULTS_DIR, "confusion_matrices_binary_smote.json")))
r2 = {}
for m in MODELS:
    (tn, fp), (fn, tp) = np.array(cm[m])
    n = tn + fp + fn + tp
    r2[m] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "n": int(n),
             "accuracy": (tp + tn) / n, "precision": tp / (tp + fp), "recall": tp / (tp + fn),
             "specificity": tn / (tn + fp), "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
             "mcc": float((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))}
dump("R2_pooled_main_benchmark.json", r2)
S += ["R2 POOLED OUT-OF-FOLD METRICS, MAIN BENCHMARK (from Table VII counts, 62,400 predictions)",
      f"  {'Model':22s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s} {'MCC':>6s}"]
for m in MODELS:
    v = r2[m]; S.append(f"  {NAMES[m]:22s} {v['accuracy']:6.3f} {v['precision']:6.3f} {v['recall']:6.3f} {v['specificity']:6.3f} {v['mcc']:6.3f}")
S.append("")

# =================================================================== R3
log("R3 data audit")
loc = lt.dt.hour + lt.dt.minute / 60
dt = pd.to_datetime(d["time"], utc=True)
step = dt.groupby(g).diff().dt.total_seconds()
ep = []
for e in np.unique(g):
    msk = g == e; sub = d[msk]
    ep.append({"episode_id": int(e), "label": int(y[msk][0]), "code": float(code[msk][0]), "n_rows": int(msk.sum()),
               "date": str(lt[msk].iloc[0].date()), "start_local": lt[msk].iloc[0].strftime("%H:%M"),
               "end_local": lt[msk].iloc[-1].strftime("%Y-%m-%d %H:%M"),
               "median_Pg_W": float(sub.Pg.median()), "median_Va_V": float(sub.Va.median()),
               "Va_q25": float(sub.Va.quantile(.25)), "Va_q75": float(sub.Va.quantile(.75)),
               "Eg_first": float(sub.Eg.iloc[0]), "Eg_last": float(sub.Eg.iloc[-1]),
               "Eg_resets_inside": int((sub.Eg.diff() < -0.5).sum())})
ep = pd.DataFrame(ep); ep.to_csv(os.path.join(OUT, "R3_episode_table.csv"), index=False)
hi = d.Pg >= 600
r3 = {"median_sampling_interval_s": float(step.median()),
      "rolling_windows_s": {w: float(w * step.median()) for w in config.ROLLING_WINDOWS},
      "Va_by_class_Pg_ge_600": {int(c): {"rows": int(((y == c) & hi).sum()),
                                          "episodes": int(len(np.unique(g[(y == c) & hi]))),
                                          "median": float(d.Va[(y == c) & hi].median()),
                                          "q25": float(d.Va[(y == c) & hi].quantile(.25)),
                                          "q75": float(d.Va[(y == c) & hi].quantile(.75))} for c in (0, 1)},
      "rows_Pg_le_5W": {int(c): int(((y == c) & (d.Pg <= 5)).sum()) for c in (0, 1)},
      "Eg_reset_count_overnight_episodes": int(ep.Eg_resets_inside.sum()),
      "Eg_spearman_with_local_hour_no_fault_daytime": float(stats.spearmanr(d.Eg[(y == 0) & (d.Pg > 5)], loc[(y == 0) & (d.Pg > 5)]).correlation),
      "Va_threshold_rule_in_sample": {str(t): float(((d.Va < t).astype(int) == y).mean()) for t in (160, 165, 170, 175)}}
med0 = r3["Va_by_class_Pg_ge_600"][0]["median"]
drops = {}
for c in sorted(set(code[y == 1])):
    msk = (code == c) & hi
    if msk.sum(): drops[str(c)] = {"median_Va_Pg_ge_600": float(d.Va[msk].median()),
                                   "drop_pct_vs_no_fault": float(100 * (1 - d.Va[msk].median() / med0)), "rows": int(msk.sum())}
r3["Va_drop_by_code_Pg_ge_600"] = drops
dump("R3_data_audit.json", r3)
S += ["R3 DATA AUDIT (no model)",
      f"  Median sampling interval: {r3['median_sampling_interval_s']:.0f} s, so the 5- and 10-sample windows span "
      f"{r3['rolling_windows_s'][5]:.0f} s and {r3['rolling_windows_s'][10]:.0f} s.",
      f"  Eg resets to zero inside overnight episodes {r3['Eg_reset_count_overnight_episodes']} time(s); Spearman correlation of Eg "
      f"with local hour (no-fault, daytime rows) = {r3['Eg_spearman_with_local_hour_no_fault_daytime']:.2f}: a daily cumulative register.",
      "  PV-side voltage at comparable output (grid power >= 600 W):"]
for c, nm in ((0, "no fault"), (1, "fault")):
    v = r3["Va_by_class_Pg_ge_600"][c]
    S.append(f"    {nm:9s} median {v['median']:.0f} V (IQR {v['q25']:.0f} to {v['q75']:.0f}), {v['rows']:,} rows, {v['episodes']} episodes")
S.append("  Va drop by fault code at grid power >= 600 W (against the no-fault median):")
for c, v in drops.items():
    S.append(f"    code {c}: median {v['median_Va_Pg_ge_600']:.0f} V, drop {v['drop_pct_vs_no_fault']:.1f}% ({v['rows']} rows)")
S.append(f"  Rows with grid power <= 5 W: no fault {r3['rows_Pg_le_5W'][0]}, fault {r3['rows_Pg_le_5W'][1]}")
S.append("  Single in-sample Va threshold (not cross-validated): " +
         ", ".join(f"Va < {t} V acc {v:.3f}" for t, v in r3["Va_threshold_rule_in_sample"].items()))
S += ["  Per-episode table: R3_episode_table.csv", ""]

# ============================================================== R4 to R8
ALL = MODELS + ["VaStump"]
pf4, pr4 = run_setting("R4 grouped, 29 features (+ R6 Va stump)", grouped, ALL, X_ALL)
pf5, pr5 = run_setting("R5 grouped, without Eg", grouped, MODELS, X_NO_EG)
pf7, pr7 = run_setting("R7 balanced-episode folds", balanced, ALL, X_ALL)
pf8, pr8 = run_setting("R8 random split ignoring episodes", plain, MODELS, X_ALL, use_groups=False)
np.savez_compressed(os.path.join(OUT, "oof_predictions.npz"),
                    **{f"{s}__{m}": p[m] for s, p in (("R4", pr4), ("R5", pr5), ("R7", pr7), ("R8", pr8)) for m in p},
                    y=y, episode_id=g, fault_code=code)
dump("per_fold.json", {"R4_grouped_29": pf4, "R5_grouped_no_Eg": pf5, "R7_balanced": pf7, "R8_random": pf8})

summ = {s: {m: {"folds": fold_summary(pf[m]), "pooled": pooled(pr[m])} for m in pf}
        for s, pf, pr in (("R4", pf4, pr4), ("R5", pf5, pr5), ("R7", pf7, pr7), ("R8", pf8, pr8))}
for m in ALL:
    summ["R4"][m]["stump_thresholds_V"] = [f.get("threshold_V") for f in pf4[m]] if m == "VaStump" else None
tests4 = pairwise_tests(pf4, ALL)
tests7 = pairwise_tests(pf7, ALL)
dump("R4_to_R8_summary.json", summ)
dump("R4_pairwise_tests.json", tests4)
dump("R7_pairwise_tests.json", tests7)

ep4, code4 = per_unit_recall(pr4, ALL)
ep4 = ep4.merge(ep[["episode_id", "date", "start_local", "median_Pg_W", "median_Va_V"]], on="episode_id")
ep4.to_csv(os.path.join(OUT, "R4_per_episode_recall.csv"), index=False)
code4.to_csv(os.path.join(OUT, "R4_per_code_recall.csv"), index=False)


def table(s, models, title):
    L = [title, f"  {'Model':22s} {'Acc':>6s} {'MCC':>6s} {'AUC':>6s} | {'Acc2c':>6s} {'MCC2c':>6s} | {'pAcc':>6s} {'pMCC':>6s}"]
    for m in models:
        f = summ[s][m]["folds"]; p = summ[s][m]["pooled"]
        L.append(f"  {NAMES[m]:22s} {f['accuracy']['mean_all_folds']:6.3f} {f['mcc']['mean_all_folds']:6.3f} "
                 f"{f['auc_roc']['mean_all_folds']:6.3f} | {f['accuracy']['mean_two_class_folds']:6.3f} "
                 f"{f['mcc']['mean_two_class_folds']:6.3f} | {p['accuracy']:6.3f} {p['mcc']:6.3f}")
    L.append("  Acc/MCC/AUC: mean over all folds (AUC over two-class folds only); 2c: two-class folds only; p: pooled out-of-fold.")
    return L


def sig_lines(tests):
    L = []
    for metric in ("accuracy", "mcc"):
        w = [f"{NAMES[r['a']]} vs {NAMES[r['b']]} (adj p {r['p_wilcoxon_holm']:.3f})" for r in tests[metric] if r["p_wilcoxon_holm"] < .05]
        t = [f"{NAMES[r['a']]} vs {NAMES[r['b']]} (adj p {r['p_corrected_t_holm']:.3f})" for r in tests[metric] if r["p_corrected_t_holm"] < .05]
        L.append(f"  {metric}: Wilcoxon + Holm significant: {'; '.join(w) if w else 'none'}")
        L.append(f"  {metric}: corrected t-test + Holm significant: {'; '.join(t) if t else 'none'}")
    top = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]
    for r in tests["accuracy"]:
        if r["a"] in top and r["b"] in top or "VaStump" in (r["a"], r["b"]):
            L.append(f"  accuracy {NAMES[r['a']]} vs {NAMES[r['b']]}: P(a better) {r['P_a_better']:.2f}, "
                     f"P(equivalent) {r['P_equivalent']:.2f}, P(b better) {r['P_b_better']:.2f}")
    return L


S += table("R4", ALL, "R4/R6 GROUPED FOLDS (paper protocol), MODAL SETTINGS, 29 FEATURES, PLUS Va STUMP")
th = [t for t in summ["R4"]["VaStump"]["stump_thresholds_V"] if t is not None]
S.append(f"  Va stump thresholds learned per fold: {min(th):.1f} to {max(th):.1f} V")
S += sig_lines(tests4) + [""]
S += table("R5", MODELS, "R5 GROUPED FOLDS WITHOUT Eg (28 FEATURES)")
S.append("  Change in pooled MCC when Eg is removed: " + ", ".join(
    f"{NAMES[m]} {summ['R5'][m]['pooled']['mcc'] - summ['R4'][m]['pooled']['mcc']:+.3f}" for m in MODELS))
S.append("")
S += table("R7", ALL, "R7 BALANCED-EPISODE FOLDS (every test fold has both classes)")
S += sig_lines(tests7) + [""]
S += ["R8 RANDOM SPLIT IGNORING EPISODES, ALL SEVEN CLASSIFIERS (against R4 grouped folds)"]
for m in MODELS:
    a_g = summ["R4"][m]["folds"]["accuracy"]["mean_all_folds"]; a_r = summ["R8"][m]["folds"]["accuracy"]["mean_all_folds"]
    S.append(f"  {NAMES[m]:22s} grouped {a_g:.3f}  random {a_r:.3f}  inflation {100*(a_r-a_g):+.1f} points")
S.append("")
S += ["R4 PER-FAULT-CODE RECALL (specificity for code 0.0), pooled over repeats"]
pv = code4.pivot(index="code", columns="model", values="rate")[ALL]
S.append("  code  " + " ".join(f"{NAMES[m][:10]:>10s}" for m in ALL))
for c, row in pv.iterrows():
    S.append(f"  {c:4.1f}  " + " ".join(f"{row[m]:10.3f}" for m in ALL))
S.append("")
S += ["R4 PER-EPISODE RECALL (fault episodes) with episode conditions"]
pe = ep4[ep4.label == 1].pivot(index="episode_id", columns="model", values="rate")[ALL]
info = ep.set_index("episode_id")
S.append(f"  {'ep':>3s} {'code':>4s} {'date':10s} {'start':5s} {'Pg W':>6s} {'Va V':>5s}  " + " ".join(f"{NAMES[m][:6]:>6s}" for m in ALL))
for e, row in pe.iterrows():
    i = info.loc[e]
    S.append(f"  {e:3d} {i.code:4.1f} {i.date:10s} {i.start_local:5s} {i.median_Pg_W:6.0f} {i.median_Va_V:5.0f}  " +
             " ".join(f"{row[m]:6.2f}" for m in ALL))
S.append("")

# =================================================================== R9
log("R9 figures")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

# Figure A: Va against grid power by class and fault group
fig, ax = plt.subplots(figsize=(6.5, 4.2))
groups_ = [("No fault", y == 0, "#9a9a9a", 4),
           ("Constant partial shading", np.isin(code, [2.1, 2.2, 2.3, 3.1, 3.2]), "#1f77b4", 5),
           ("Intermittent shading", code == 4.0, "#ff7f0e", 5),
           ("Uniform shading", code == 1.0, "#d62728", 7)]
rng = np.random.default_rng(0)
for nm, msk, col, s in groups_:
    idx = np.where(msk)[0]
    if len(idx) > 3000: idx = rng.choice(idx, 3000, replace=False)
    ax.scatter(d.Pg.values[idx], d.Va.values[idx], s=s, c=col, alpha=.45, label=nm, linewidths=0)
m23 = code == 2.3
ax.scatter(d.Pg.values[m23], d.Va.values[m23], s=9, facecolors="none", edgecolors="k", linewidths=.5, label="Code 2.3 episode")
ax.set_xlabel("Grid power Pg (W)"); ax.set_ylabel("PV-side voltage Va (V)")
ax.legend(loc="lower right", fontsize=7, markerscale=2, frameon=False)
ax.set_title("PV-side voltage against grid power, benchmark rows")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "figA_va_vs_power.png"), dpi=300); plt.close(fig)

# Figure B: grouped against random split accuracy
fig, ax = plt.subplots(figsize=(6.5, 3.2))
xs = np.arange(len(MODELS))
ag = [summ["R4"][m]["folds"]["accuracy"]["mean_all_folds"] for m in MODELS]
ar = [summ["R8"][m]["folds"]["accuracy"]["mean_all_folds"] for m in MODELS]
ax.bar(xs - .2, ag, .4, label="Episode-grouped folds", color="#1f77b4")
ax.bar(xs + .2, ar, .4, label="Random split ignoring episodes", color="#bbbbbb")
for i in range(len(MODELS)): ax.text(xs[i] + .2, ar[i] + .005, f"+{100*(ar[i]-ag[i]):.1f}", ha="center", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels([NAMES[m] for m in MODELS], rotation=20, ha="right")
ax.set_ylim(.6, 1.03); ax.set_ylabel("Mean accuracy (15 folds)"); ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "figB_grouped_vs_random.png"), dpi=300); plt.close(fig)

# Figure C: per-episode recall heatmap
fig, ax = plt.subplots(figsize=(6.8, 4.6))
mat = pe.values
im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(ALL))); ax.set_xticklabels([NAMES[m] for m in ALL], rotation=30, ha="right")
ax.set_yticks(range(len(pe))); ax.set_yticklabels([f"code {info.loc[e].code:.1f}, {info.loc[e].date}, {info.loc[e].median_Pg_W:.0f} W" for e in pe.index], fontsize=7)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if mat[i, j] < .5 else "k")
fig.colorbar(im, ax=ax, label="Recall")
ax.set_title("Recall per fault episode (label: fault code, date, median grid power)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "figC_per_episode_recall.png"), dpi=300); plt.close(fig)

S.append(f"Total time: {(time.time()-T0)/60:.1f} min")
open(os.path.join(OUT, "SUMMARY.txt"), "w").write("\n".join(S) + "\n")
print("\n".join(S), flush=True)
log("REVISION RUNS DONE")
