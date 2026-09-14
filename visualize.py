"""
visualize.py
Generates the paper's figures from the JSON/CSV artefacts written by
run_experiment.py: metric comparison bars, confusion-matrix heatmaps,
feature importance, and the label-scheme/SMOTE ablation charts.

Run this AFTER run_experiment.py has completed.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def fig1_metric_comparison():
    """Grouped bar chart of all six metrics (mean +/- std) for every model,
    primary condition (binary label, with SMOTE). Writes fig_metric_comparison.png.
    """
    df = pd.read_csv(f"{config.RESULTS_DIR}/all_runs_summary.csv")
    prim = df[(df.scheme == "binary") & (df.smote == "smote")]
    metrics = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "mcc_mean", "auc_roc_mean"]
    errs = [m.replace("_mean", "_std") for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.15
    for i, (_, row) in enumerate(prim.iterrows()):
        vals = row[metrics].values.astype(float)
        errvals = row[errs].values.astype(float)
        ax.bar(x + i * width, vals, width, yerr=errvals, capsize=3, label=row["model"])
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([m.replace("_mean", "") for m in metrics])
    ax.set_ylabel("Score")
    ax.set_title("Comparative performance across metrics (binary scheme, with SMOTE)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{config.FIG_DIR}/fig_metric_comparison.png", dpi=150)
    plt.close(fig)


def fig3_confusion_matrices():
    """Row-normalised confusion-matrix heatmap per model, primary condition
    (binary label, with SMOTE). Writes fig_confusion_matrices.png.
    """
    with open(f"{config.RESULTS_DIR}/confusion_matrices_binary_smote.json") as f:
        cms = json.load(f)
    fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 3.2))
    for ax, (model_name, cm) in zip(axes, cms.items()):
        cm = np.array(cm, dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center")
        ax.set_title(model_name, fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No-Fault", "Fault"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No-Fault", "Fault"])
    fig.suptitle("Aggregated confusion matrices, row-normalised (binary, SMOTE)")
    fig.tight_layout()
    fig.savefig(f"{config.FIG_DIR}/fig_confusion_matrices.png", dpi=150)
    plt.close(fig)


def fig4_feature_importance():
    """Top-12 Logistic Regression coefficients and Decision Tree impurity
    importances, side by side. Writes fig_feature_importance.png.
    """
    with open(f"{config.RESULTS_DIR}/feature_importance.json") as f:
        fi = json.load(f)
    names = fi["feature_names"]
    lr = np.array(fi["logreg_coef"])
    dt = np.array(fi["dtree_importance"])

    order = np.argsort(-np.abs(lr))[:12]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh([names[i] for i in order][::-1], lr[order][::-1])
    axes[0].set_title("Logistic Regression: standardized coefficients")
    order2 = np.argsort(-dt)[:12]
    axes[1].barh([names[i] for i in order2][::-1], dt[order2][::-1])
    axes[1].set_title("Decision Tree: impurity-based importance")
    fig.suptitle("Feature importance (binary scheme, with SMOTE)")
    fig.tight_layout()
    fig.savefig(f"{config.FIG_DIR}/fig_feature_importance.png", dpi=150)
    plt.close(fig)


def fig56_ablation():
    """Two ablation line charts: MCC with vs. without SMOTE (binary scheme),
    and accuracy for the binary vs. tier label scheme (with SMOTE).
    Writes fig_ablation_smote_mcc.png and fig_ablation_label_scheme.png.
    """
    df = pd.read_csv(f"{config.RESULTS_DIR}/all_runs_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    for smote_key, marker in [("smote", "o"), ("no_smote", "x")]:
        sub = df[(df.scheme == "binary") & (df.smote == smote_key)]
        ax.plot(sub["model"], sub["mcc_mean"], marker=marker, label=f"MCC ({smote_key})")
    ax.set_ylabel("MCC"); ax.set_title("Effect of SMOTE on MCC (binary scheme)")
    ax.legend(); ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(f"{config.FIG_DIR}/fig_ablation_smote_mcc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for scheme, marker in [("binary", "o"), ("tier", "s")]:
        sub = df[(df.scheme == scheme) & (df.smote == "smote")]
        ax.plot(sub["model"], sub["accuracy_mean"], marker=marker, label=f"Accuracy ({scheme})")
    ax.set_ylabel("Accuracy"); ax.set_title("Accuracy: binary vs. tier label scheme (with SMOTE)")
    ax.legend(); ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(f"{config.FIG_DIR}/fig_ablation_label_scheme.png", dpi=150)
    plt.close(fig)


def main():
    """Build all five paper figures from results/ into figures/."""
    fig1_metric_comparison()
    fig3_confusion_matrices()
    fig4_feature_importance()
    fig56_ablation()
    print(f"Figures written to {config.FIG_DIR}/")


if __name__ == "__main__":
    main()
