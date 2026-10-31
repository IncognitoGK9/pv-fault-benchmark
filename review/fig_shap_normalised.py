"""
review/fig_shap_normalised.py
Replacement for paper Fig. 3. The stored mean |SHAP| values are in different
units per model (log-odds for Logistic Regression and XGBoost, probability
for the others), so raw magnitudes cannot be compared or summed across
models. Each model's vector is normalised to sum to 1 (share of that model's
total attribution) and shown as a heatmap of the 12 features with the
largest mean share. Reads results/shap_importance.json only.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "results/shap_importance.json")))
fn = d["feature_names"]
models = ["LogisticRegression", "kNN", "DecisionTree", "SVM", "MLP", "RandomForest", "XGBoost"]
labels = ["LR", "kNN", "DT", "SVM", "MLP", "RF", "XGB"]
M = np.array([np.array(d[m]) / np.sum(d[m]) for m in models]).T
order = np.argsort(-M.mean(axis=1))[:12]
pretty = {"Va": "$V_a$", "Eg": "$E_g$", "volt_dev": "$V_g-V_a$", "Ia": "$I_a$", "Pg": "$P_g$",
          "Vg": "$V_g$", "Ig": "$I_g$", "Fg": "$F_g$", "ratio_Ig_Ia": "$I_g/I_a$",
          "ratio_Pg_VaIa": "$P_g/(V_aI_a)$", "freq_dev": "$|F_g-50|$"}
def lab(f):
    if f in pretty: return pretty[f]
    base, rest = f.split("_", 1)
    sub = {"Pg": "P_g", "Ia": "I_a", "Vg": "V_g", "Fg": "F_g"}[base]
    if rest.startswith("rmean"): return rf"$\mu_{{{rest[5:]}}}({sub})$"
    if rest.startswith("rstd"): return rf"$\sigma_{{{rest[4:]}}}({sub})$"
    return rf"$\Delta {sub}$"
fig, ax = plt.subplots(figsize=(6.4, 4.6))
im = ax.imshow(M[order], cmap="Blues", aspect="auto", vmin=0)
ax.set_xticks(range(7), labels); ax.set_yticks(range(12), [lab(fn[i]) for i in order])
for i, r in enumerate(order):
    for j in range(7):
        v = M[r, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                             color="white" if v > 0.3 else "black")
cb = fig.colorbar(im, ax=ax); cb.set_label("Share of model's total mean |SHAP|")
fig.tight_layout()
out = os.path.join(ROOT, "figures/fig_shap_normalised.png")
fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf"))
print("wrote", out)
