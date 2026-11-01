"""
paper_figures_revision.py
Publication versions of the three figures added in revision (Figures 6 to 8
of the manuscript), drawn from the saved outputs of revision_runs.py
(results/revision/) and the benchmark CSV. No model is fitted.

  fig6_per_episode_recall.png   recall per fault episode, all classifiers
  fig7_grouped_vs_random.png    accuracy with episode-grouped folds and with a random split
  fig8_va_vs_power.png          PV-side voltage against grid power, by fault group

Usage: python paper_figures_revision.py   (writes to figures/revision/)
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", sys.argv[1] if len(sys.argv) > 1 else "revision")
OUT = os.path.join(HERE, "figures", "revision")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
MODELS = ["LogisticRegression", "kNN", "DecisionTree", "SVM", "MLP", "RandomForest", "XGBoost"]
NAMES = {"LogisticRegression": "Logistic\nRegression", "kNN": "kNN", "DecisionTree": "Decision\nTree",
         "SVM": "SVM", "MLP": "MLP", "RandomForest": "Random\nForest", "XGBoost": "XGBoost",
         "VaStump": "Voltage\nthreshold"}

# Figure 6: per-episode recall
ep = pd.read_csv(os.path.join(RES, "R4_per_episode_recall.csv"))
info = pd.read_csv(os.path.join(RES, "R3_episode_table.csv")).set_index("episode_id")
cols = MODELS + ["VaStump"]
pe = ep[ep.label == 1].pivot(index="episode_id", columns="model", values="rate")[cols]
fig, ax = plt.subplots(figsize=(6.8, 4.9))
im = ax.imshow(pe.values, aspect="auto", cmap="Greys", vmin=0, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([NAMES[m] for m in cols], fontsize=7.5)
ax.set_yticks(range(len(pe)))
ax.set_yticklabels([f"{info.loc[e].code:.1f}   {info.loc[e].date}   {info.loc[e].median_Pg_W:5.0f} W" for e in pe.index],
                   fontsize=7, family="DejaVu Sans Mono")
for i in range(pe.shape[0]):
    for j in range(pe.shape[1]):
        v = pe.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3, color="w" if v > .55 else "k")
ax.set_ylabel("Fault code, date, median grid power")
fig.colorbar(im, ax=ax, label="Recall", fraction=.04, pad=.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_per_episode_recall.png"), dpi=300); plt.close(fig)

# Figure 7: grouped against random split (differences of the reported, rounded means)
s = json.load(open(os.path.join(RES, "R4_to_R8_summary.json")))
ag = [round(s["R4"][m]["folds"]["accuracy"]["mean_all_folds"], 3) for m in MODELS]
ar = [round(s["R8"][m]["folds"]["accuracy"]["mean_all_folds"], 3) for m in MODELS]
fig, ax = plt.subplots(figsize=(6.5, 3.1))
x = np.arange(len(MODELS))
ax.bar(x - .2, ag, .4, label="Episode-grouped folds", color="#404040")
ax.bar(x + .2, ar, .4, label="Random split ignoring episodes", color="#bdbdbd")
for i in range(len(MODELS)):
    ax.text(x[i] + .2, ar[i] + .006, f"+{100*(ar[i]-ag[i]):.1f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([NAMES[m] for m in MODELS], fontsize=7.5)
ax.set_ylim(.6, 1.09); ax.set_ylabel("Mean accuracy, 15 folds")
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_grouped_vs_random.png"), dpi=300); plt.close(fig)

# Figure 8: PV-side voltage against grid power
d = pd.read_csv(os.path.join(HERE, "data", "reunion_subsample.csv"))
code = d.Fault.values
groups = [("No fault", code == 0.0, "#9e9e9e", "o", 4),
          ("Constant partial shading", np.isin(code, [2.1, 2.2, 3.1, 3.2]), "#1f77b4", "o", 4),
          ("Intermittent shading", code == 4.0, "#e6550d", "o", 4),
          ("Uniform shading", code == 1.0, "#a50f15", "s", 8),
          ("Constant partial shading, code 2.3", code == 2.3, "k", "x", 10)]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(6.5, 4.0))
for nm, m, c, mk, sz in groups:
    idx = np.where(m)[0]
    if len(idx) > 3000: idx = rng.choice(idx, 3000, replace=False)
    ax.scatter(d.Pg.values[idx], d.Va.values[idx], s=sz, c=c, marker=mk, alpha=.5 if mk == "o" else .8,
               linewidths=.5 if mk == "x" else 0, label=nm)
ax.set_xlabel("Grid power Pg (W)"); ax.set_ylabel("PV-side voltage Va (V)")
ax.legend(loc="lower right", fontsize=7, markerscale=2, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig8_va_vs_power.png"), dpi=300); plt.close(fig)
print("figures written to", OUT)
