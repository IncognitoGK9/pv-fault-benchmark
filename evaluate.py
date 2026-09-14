"""
evaluate.py
Per-fold metric computation, aggregation across outer folds, aggregated
confusion matrices, pairwise Wilcoxon significance testing, and matched-
pairs effect sizes (rank-biserial correlation) for the same comparisons.
"""
import numpy as np
from scipy.stats import wilcoxon, rankdata
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, matthews_corrcoef, roc_auc_score,
                              confusion_matrix)


def fold_metrics(y_true, y_pred, y_score=None, n_classes=2):
    """Compute accuracy, precision, recall, F1, MCC, and AUC-ROC for one fold.

    precision/recall/F1 are binary-averaged for n_classes == 2 (positive
    class = 1) and macro-averaged otherwise. AUC-ROC uses the positive-class
    score for binary problems and a macro one-vs-rest score for multiclass;
    it is set to NaN if y_score is not supplied or scoring fails.
    """
    average = "binary" if n_classes == 2 else "macro"
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    auc = np.nan
    if y_score is not None:
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_score[:, 1])
            else:
                auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        except Exception:
            auc = np.nan
    out["auc_roc"] = auc
    return out


def aggregate(fold_results, n_classes=2):
    """Reduce a list of per-fold results (as returned by pipeline.run_nested_cv)
    to per-metric mean, std, and the raw per-fold values (suffixed _mean,
    _std, _values respectively), for reporting and for pairwise_wilcoxon.
    """
    rows = [fold_metrics(fr["y_true"], fr["y_pred"], fr["y_score"], n_classes) for fr in fold_results]
    agg = {}
    for k in rows[0]:
        vals = np.array([r[k] for r in rows], dtype=float)
        agg[f"{k}_mean"] = float(np.nanmean(vals))
        agg[f"{k}_std"] = float(np.nanstd(vals))
        agg[f"{k}_values"] = vals.tolist()
    return agg


def pooled_confusion_matrix(fold_results, labels):
    """Concatenate y_true/y_pred across all outer folds and return one
    confusion matrix over the pooled predictions (rows/cols ordered by `labels`).
    """
    y_true_all = np.concatenate([fr["y_true"] for fr in fold_results])
    y_pred_all = np.concatenate([fr["y_pred"] for fr in fold_results])
    return confusion_matrix(y_true_all, y_pred_all, labels=labels)


def pairwise_wilcoxon(metric_by_model, metric_key="accuracy_values"):
    """Paired Wilcoxon signed-rank test between every pair of models, on the
    per-fold values of one metric (see aggregate()).

    metric_by_model: {model_name: agg_dict}, as produced by aggregate().
    Returns: {model_a: {model_b: p_value}}, with p=None on the diagonal and
    p=1.0 for a pair whose per-fold values are numerically identical
    (wilcoxon() is undefined when all differences are zero).
    """
    models_ = list(metric_by_model.keys())
    result = {m: {} for m in models_}
    for i, m1 in enumerate(models_):
        for m2 in models_:
            if m1 == m2:
                result[m1][m2] = None
                continue
            a = np.array(metric_by_model[m1][metric_key])
            b = np.array(metric_by_model[m2][metric_key])
            if np.allclose(a, b):
                p = 1.0
            else:
                try:
                    _, p = wilcoxon(a, b)
                except ValueError:
                    p = np.nan
            result[m1][m2] = float(p)
    return result


def matched_pairs_rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation (Kerby, 2014) -- the standard
    effect-size companion to the Wilcoxon signed-rank test. Reporting it
    alongside every p-value lets a non-significant result be read together
    with how large/consistent the underlying difference actually was,
    rather than relying on the p-value alone.

    r = (sum of ranks favouring a) - (sum of ranks favouring b), all divided
    by the total rank sum. Ranges -1 (b always better) to +1 (a always
    better); 0 means the wins and losses are rank-balanced. Zero-difference
    pairs are dropped before ranking, matching scipy.stats.wilcoxon's
    default (zero_method="wilcox") handling of ties.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    r_plus = ranks[nonzero > 0].sum()
    r_minus = ranks[nonzero < 0].sum()
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total > 0 else 0.0


def pairwise_effect_sizes(metric_by_model, metric_key="accuracy_values"):
    """Matched-pairs rank-biserial correlation between every pair of models,
    on the per-fold values of one metric. Mirrors pairwise_wilcoxon's
    structure/signature so the two are always reported side by side (paper
    Table VIII).

    metric_by_model: {model_name: agg_dict}, as produced by aggregate().
    Returns: {model_a: {model_b: r}}, with r=None on the diagonal.
    """
    models_ = list(metric_by_model.keys())
    result = {m: {} for m in models_}
    for m1 in models_:
        for m2 in models_:
            if m1 == m2:
                result[m1][m2] = None
                continue
            a = metric_by_model[m1][metric_key]
            b = metric_by_model[m2][metric_key]
            result[m1][m2] = matched_pairs_rank_biserial(a, b)
    return result
