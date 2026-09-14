"""
shap_analysis.py
Model-agnostic feature attribution via SHAP (Lundberg & Lee, 2017), computed
once for every classifier on the primary condition (binary label scheme,
with SMOTE). SHAP is used here instead of comparing Logistic Regression's
coefficients directly against the Decision Tree's impurity-based
importances, because those are two different mathematical quantities on
different scales and are not a valid basis for a cross-model comparison.
SHAP values are additive, signed, per-instance attributions defined
identically for every model, so mean |SHAP value| per feature is comparable
across all seven classifiers (paper Section VI.D).

Run this AFTER run_experiment.py has produced data/reunion_subsample.csv,
and BEFORE visualize.py (which reads this script's output to draw the
feature-attribution figure).

    python shap_analysis.py

Explainer choice per model:
  - DecisionTree, RandomForest, XGBoost: shap.TreeExplainer (exact, fast).
  - LogisticRegression: shap.LinearExplainer (exact for linear models).
  - kNN, SVM, MLP: no closed-form SHAP algorithm exists for these model
    families, so a generic sampling-based shap.Explainer is used instead,
    evaluated against a subsampled background and explanation set
    (config.SHAP_BACKGROUND_SIZE / SHAP_EXPLAIN_SIZE) to keep runtime
    tractable -- standard practice for non-tree SHAP on datasets of this
    size. Because only a subset of rows is explained for these three
    models, their mean |SHAP value| is a sample estimate rather than the
    exact full-dataset value the tree/linear models get; this is noted as
    a stated caveat in the paper (Section VIII, Threats to Validity).

A model whose SHAP computation fails for any reason (e.g. a library-version
incompatibility) is skipped with a clear warning printed to the console,
rather than aborting the whole script -- this is a supplementary analysis,
not the primary comparison, and a partial result is more useful than none.
"""
import json

import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "The 'shap' package is required for this script. Run: pip install "
        "shap>=0.44 (see requirements.txt)."
    ) from exc

import config
import features
import target as target_mod
import pipeline as pl


def _background_and_explain_sets(X, seed=config.RANDOM_SEED):
    """Deterministically subsample a small, disjoint background set and
    evaluation set for the sampling-based (non-tree, non-linear) explainer.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    bg_size = min(config.SHAP_BACKGROUND_SIZE, n)
    bg_idx = rng.choice(n, size=bg_size, replace=False)
    remaining = np.setdiff1d(np.arange(n), bg_idx)
    ex_size = min(config.SHAP_EXPLAIN_SIZE, len(remaining))
    ex_idx = rng.choice(remaining, size=ex_size, replace=False)
    return X[bg_idx], X[ex_idx]


def _mean_abs_positive_class(shap_values):
    """Normalise the many shapes the SHAP library can return (a list of
    per-class arrays, a stacked (n, features, classes) array, or a plain
    (n, features) array for a single-output model) down to one mean
    absolute per-feature attribution for the positive (Fault) class.
    """
    if isinstance(shap_values, list):
        arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return np.abs(arr).mean(axis=0)


def compute_shap_importance(X, y, feature_cols, seed=config.RANDOM_SEED):
    """Fit every model in config.MODEL_NAMES on the full primary-condition
    data (binary scheme, SMOTE applied inside the pipeline) and compute
    mean |SHAP value| per feature for each.

    Returns {"feature_names": [...], model_name: [mean_abs_shap, ...]}.
    """
    X = np.asarray(X)
    out = {"feature_names": list(feature_cols)}

    for model_name in config.MODEL_NAMES:
        try:
            pipe = pl.build_pipeline(model_name, use_smote=True, seed=seed)
            pipe.fit(X, y)
            clf = pipe.named_steps["clf"]

            # Feed the explainer the same feature space the classifier was
            # actually trained on (post-scaling, where a scaler is used).
            X_input = (pipe.named_steps["scaler"].transform(X)
                       if "scaler" in pipe.named_steps else X)

            if model_name in ("DecisionTree", "RandomForest", "XGBoost"):
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(X_input)
                mean_abs = _mean_abs_positive_class(sv)

            elif model_name == "LogisticRegression":
                bg, _ = _background_and_explain_sets(X_input, seed)
                explainer = shap.LinearExplainer(clf, bg)
                sv = explainer.shap_values(X_input)
                mean_abs = _mean_abs_positive_class(sv)

            else:  # kNN, SVM, MLP -- no closed-form SHAP algorithm
                bg, ex = _background_and_explain_sets(X_input, seed)
                predict_fn = clf.predict_proba if hasattr(clf, "predict_proba") else clf.predict
                explainer = shap.Explainer(predict_fn, bg)
                explanation = explainer(ex)
                mean_abs = _mean_abs_positive_class(explanation.values)

            out[model_name] = mean_abs.tolist()
            print(f"[shap] {model_name}: done ({len(mean_abs)} features).")
        except Exception as exc:  # noqa: BLE001 -- deliberately broad; see module docstring
            print(f"[shap] {model_name}: SKIPPED due to error: {exc}")

    return out


def main():
    print("Loading subsample ...")
    df = pd.read_csv(config.SUBSAMPLE_CSV, parse_dates=["time"])
    X_df, feature_cols = features.build_features(df)
    y_bin = target_mod.get_target(df, "binary")

    print(f"Computing SHAP importance for {len(config.MODEL_NAMES)} models "
          f"on {len(df)} rows / {len(feature_cols)} features ...")
    shap_importance = compute_shap_importance(X_df.values, y_bin, feature_cols)

    out_path = f"{config.RESULTS_DIR}/shap_importance.json"
    with open(out_path, "w") as f:
        json.dump(shap_importance, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
