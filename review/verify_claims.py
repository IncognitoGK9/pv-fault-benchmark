"""
review/verify_claims.py
Recomputes, from the repository's own files, every number that is new or
changed in the arXiv version (arxiv/main.tex) and prints PASS/FAIL per claim.
Sources: results/*.json|csv, data/*, review/reanalysis_results.json.
Run: python review/verify_claims.py   (needs pandas only)
"""
import json, os, sys
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(os.path.join(R, p)))
csv = pd.read_csv(os.path.join(R, "results/all_runs_summary.csv"))
def m(s, k, mod, met): return float(csv[(csv.scheme == s) & (csv.smote == k) & (csv.model == mod)][met + "_mean"].iloc[0])
wa, wm = J("results/wilcoxon_accuracy.json"), J("results/wilcoxon_mcc.json")
ea, em = J("results/effect_size_accuracy.json"), J("results/effect_size_mcc.json")
meta, ra = J("data/subsample_metadata.json"), J("review/reanalysis_results.json")
fte = J("results/fault_type_error_analysis.json")
sub = pd.read_csv(os.path.join(R, "data/reunion_subsample.csv"), parse_dates=["time"])
sub["hour"] = sub.time.dt.hour + sub.time.dt.minute / 60
M = ["LogisticRegression", "kNN", "DecisionTree", "SVM", "MLP", "RandomForest", "XGBoost"]
fails = 0
def chk(desc, cond, val=""):
    global fails; fails += (not cond); print(("PASS " if cond else "FAIL ") + desc + (f"  [{val}]" if val != "" else ""))

fs = meta["full_dataset_stats"]
chk("raw rows 2,988,798", fs["n_rows_raw"] == 2988798)
chk("35 runs, 17 fault", fs["n_segments_total"] == 35 and fs["n_fault_segments"] == 17)
chk("no-fault share 97.17%", round(100 * fs["fault_value_counts"]["0.0"] / fs["n_rows_raw"], 2) == 97.17)
seg = meta["segments"]; fault = [s for s in seg if s["label"] == 1]
chk("fault window 8 Feb - 19 May 2022", min(s["start"] for s in fault)[:10] == "2022-02-08" and max(s["end"] for s in fault)[:10] == "2022-05-19")
chk("window = 100 days", (pd.Timestamp("2022-05-19") - pd.Timestamp("2022-02-08")).days == 100)
chk("subsample 20,800 rows, 24 episodes", meta["subsample_n_rows"] == 20800 and meta["n_episodes_kept"] == 24)
chk("tier counts 14000/4000/2800", meta["subsample_tier_counts"] == {"No-Fault": 14000, "Constant-Partial-Shading": 4000, "Severe (Uniform/Intermittent)": 2800})
chk("majority baseline 0.673", round(14000 / 20800, 3) == 0.673)
eps = {c: sum(1 for s in fault if s["fault_codes"][0] == c) for c in [1.0, 2.1, 2.2, 2.3, 3.1, 3.2, 4.0]}
chk("episodes per code 2/1/1/1/3/4/5", list(eps.values()) == [2, 1, 1, 1, 3, 4, 5], eps)
dt = sub.groupby("episode_id").time.diff().dt.total_seconds()
chk("sampling interval IQR 5-8 s", dt.quantile(.25) == 5 and dt.quantile(.75) == 8)
f, n = sub[sub.fault_bin == 1], sub[sub.fault_bin == 0]
chk("fault rows within 10:00-17:00", f.hour.min() >= 10 and f.hour.max() <= 17, (round(f.hour.min(), 2), round(f.hour.max(), 2)))
chk("no-fault hour range 05:51-18:11", round(n.hour.min(), 2) == 5.85 and round(n.hour.max(), 2) == 18.18)
chk("median hour 12.6 vs 13.4", round(f.hour.median(), 1) == 12.6 and round(n.hour.median(), 1) == 13.4)
chk("hour s.d. 1.7 vs 3.7", round(f.hour.std(), 1) == 1.7 and round(n.hour.std(), 1) == 3.7)
chk("median Pg 982 vs 274 W", round(f.Pg.median()) == 982 and round(n.Pg.median()) == 274)
chk("median Va 151 vs 184 V", f.Va.median() == 151 and n.Va.median() == 184)
chk("no-fault Vg s.d. about 4 V", round(n.Vg.std()) == 4, round(n.Vg.std(), 2))
dur = sub[sub.fault_bin == 1].groupby("episode_id").time.agg(lambda s: (s.max() - s.min()).total_seconds() / 60)
chk("fault slice 33-53 min", round(dur.min()) == 33 and round(dur.max()) == 53, (round(dur.min(), 1), round(dur.max(), 1)))
ns = [s for s in seg if s["label"] == 0]
chk("5 of 7 no-fault slices reach before 10:00 or after 17:00",
    sum(1 for e in sub[sub.fault_bin == 0].groupby("episode_id").hour.agg(["min", "max"]).itertuples() if e.min < 10 or e.max > 17) == 5)
# Wilcoxon / Holm
chk("RF vs XGB accuracy p = .421", round(wa["RandomForest"]["XGBoost"], 3) == .421)
up = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]
pa = min(wa[a][b] for i, a in enumerate(up) for b in up[i + 1:]); pm = min(wm[a][b] for i, a in enumerate(up) for b in up[i + 1:])
chk("upper-group min p: acc >= .42, MCC >= .29", pa >= .42 and pm >= .285, (round(pa, 3), round(pm, 3)))
chk("LR never significant", all(wa["LogisticRegression"][b] > .05 and wm["LogisticRegression"][b] > .05 for b in M[1:]))
chk("LR vs kNN/SVM/MLP acc p .055/.055/.277", [round(wa["LogisticRegression"][b], 3) for b in ["kNN", "SVM", "MLP"]] == [.055, .055, .277])
chk("effect sizes .93/1.0/.79/.73", [round(-ea[a]["RandomForest" if i < 2 else "XGBoost"], 2) for i, a in enumerate(["kNN", "MLP", "kNN", "MLP"])] == [.93, 1.0, .79, .73])
h = ra["holm_accuracy"]; hm = ra["holm_mcc"]
surv = [k for k, v in h.items() if v["p_holm"] < .05]
chk("Holm acc survivors = RF>MLP, RF>kNN", sorted(surv) == ["MLP vs RandomForest", "kNN vs RandomForest"], surv)
chk("Holm MCC: none survive, min .070", min(v["p_holm"] for v in hm.values()) == .0703)
# Metrics
chk("SMOTE lowers binary MCC for 6/7", sum(m("binary", "smote", x, "mcc") < m("binary", "no_smote", x, "mcc") for x in M) == 6)
chk("removing SMOTE raises acc for all but XGB", [x for x in M if m("binary", "no_smote", x, "accuracy") <= m("binary", "smote", x, "accuracy")] == ["XGBoost"])
chk("tier no-SMOTE DT .763/.527, LR .758/.483", round(m("tier", "no_smote", "DecisionTree", "accuracy"), 3) == .763 and round(m("tier", "no_smote", "DecisionTree", "mcc"), 3) == .527 and round(m("tier", "no_smote", "LogisticRegression", "accuracy"), 3) == .758)
chk("DT binary->tier drop 6 pts no-SMOTE, 20 with", round(100 * (m("binary", "no_smote", "DecisionTree", "accuracy") - m("tier", "no_smote", "DecisionTree", "accuracy"))) == 6 and round(100 * (m("binary", "smote", "DecisionTree", "accuracy") - m("tier", "smote", "DecisionTree", "accuracy"))) == 20)
pc = fte["per_fault_code_recall_binary_classifier"]
chk("code 3.1 recall 0.947 (< 0.95)", round(pc["3.1"]["recall (correctly flagged Fault)"], 3) == .947)
# Per-episode
pe = {d["episode_id"]: d["share_correct"] for d in ra["per_episode_rf_share_correct"]}
fe = {s["episode_id"]: s["fault_codes"][0] for s in fault}
missed = sorted(e for e in fe if pe[e] <= .007)
chk("missed episodes 5(2.3), 9(1.0), 19(4.0), 23(4.0)", missed == [5, 9, 19, 23], missed)
chk("other fault episodes >= .98 except ep31 .842", all(pe[e] >= .98 for e in fe if e not in missed + [31]) and pe[31] == .842)
chk("per-episode reproduces Table 13 code 3.1/4.0/1.0", abs((pe[29] + pe[31] + pe[33]) / 3 - .947) < .002 and abs(sum(pe[e] for e in fe if fe[e] == 4.0) / 5 - .595) < .003 and (pe[7] + pe[9]) / 2 == .5)
ep = {d["episode_id"]: d for d in ra["per_episode"]}
chk("ep5 Pg 217 Va 181; ep19 Pg 179 Va 169; ep23 Va 176", (ep[5]["Pg_median"], ep[5]["Va_median"], ep[19]["Pg_median"], ep[19]["Va_median"], ep[23]["Va_median"]) == (217, 181, 179, 169, 176))
chk("uniform eps zero output; ep7 Va 0, ep9 Va 150", ep[7]["share_Pg_below_20W"] == 1 and ep[9]["share_Pg_below_20W"] == 1 and ep[7]["Va_median"] == 0 and ep[9]["Va_median"] == 150)
chk("400-row slice = 2-15% of intermittent episodes", round(100 * 400 / max(s["n_original"] for s in fault if s["fault_codes"][0] == 4.0)) == 2 and round(100 * 400 / min(s["n_original"] for s in fault if s["fault_codes"][0] == 4.0)) == 15)
# Probe
p = ra["confound_probe_rf_fixed_modal_no_smote"]
chk("probe hour MCC .443, Va .724, raw6 .744", (p["hour_only"]["mcc"][0], p["Va_only"]["mcc"][0], p["raw7_minus_Eg"]["mcc"][0]) == (.443, .724, .744))
chk("hour-only MCC beats kNN, MLP, SVM (primary)", all(p["hour_only"]["mcc"][0] > m("binary", "smote", x, "mcc") for x in ["kNN", "MLP", "SVM"]))
chk("Va-only and raw6 beat every tuned model (MCC)", min(p["Va_only"]["mcc"][0], p["raw7_minus_Eg"]["mcc"][0]) > max(m("binary", "smote", x, "mcc") for x in M))
# SHAP
s = J("results/shap_importance.json"); fn = s["feature_names"]
share = {k: sum(s[x][fn.index(k)] / sum(s[x]) for x in M) / 7 for k in ["Va", "Eg", "volt_dev"]}
chk("SHAP share Va 23%, Eg 9%, Vg-Va 8%", [round(100 * share[k]) for k in ["Va", "Eg", "volt_dev"]] == [23, 9, 8], share)
chk("SHAP totals: LR 40.8, XGB 14.1 (log-odds) vs others < 1.1", round(sum(s["LogisticRegression"]), 1) == 40.8 and round(sum(s["XGBoost"]), 1) == 14.1 and max(sum(s[x]) for x in M if x not in ("LogisticRegression", "XGBoost")) < 1.1)
# Tables 6, 7 and 10 in arxiv/main.tex: every cell against all_runs_summary.csv
import re
tex = open(os.path.join(R, "arxiv/main.tex")).read()
names = dict(zip(["Logistic Regression", "kNN", "Decision Tree", "SVM (RBF)", "MLP", "Random Forest", "XGBoost"], M))
f3 = lambda v: f"{v:.3f}"[1:] if v < 1 else f"{v:.3f}"
for cap, k in [("Binary scheme with SMOTE; mean", "smote"), ("Binary scheme without SMOTE", "no_smote")]:
    blk = tex[tex.index(cap):]; blk = blk[:blk.index("\\end{tabular}")]
    bad = []
    for disp, mod in names.items():
        row = re.search("^" + re.escape(disp) + r" & (.*?) \\\\", blk, re.M).group(1)
        exp = " & ".join(f"{f3(m('binary', k, mod, met))}$\\pm${f3(float(csv[(csv.scheme=='binary')&(csv.smote==k)&(csv.model==mod)][met+'_std'].iloc[0]))}"
                         for met in ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc"])
        if row != exp: bad.append((disp, row, exp))
    chk(f"LaTeX table '{cap}' matches CSV (42 cells)", not bad, bad)
blk = tex[tex.index("Binary versus three-tier scheme"):]; blk = blk[:blk.index("\\end{tabular}")]
bad = []
for disp, mod in names.items():
    row = re.search("^" + re.escape(disp) + r" & (.*?) \\\\", blk, re.M).group(1)
    exp = " & ".join(f3(m(s, k, mod, met)) for k in ["smote", "no_smote"] for met in ["accuracy", "mcc"] for s in ["binary", "tier"])
    if row.split(" & ") != exp.split(" & "):
        # table column order is acc-bin, acc-tier, mcc-bin, mcc-tier per condition
        exp2 = " & ".join(f3(m(s, k, mod, met)) for k in ["smote", "no_smote"] for met in ["accuracy", "mcc"] for s in ["binary", "tier"])
        bad.append((disp, row, exp2))
chk("LaTeX label-scheme table matches CSV (56 cells)", not bad, bad)
print(f"\n{fails} failure(s)"); sys.exit(1 if fails else 0)
