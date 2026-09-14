"""
pipeline.py
Group-aware nested cross-validation: a repeated outer StratifiedGroupKFold
over episode_id (so a fault/no-fault episode is never split across train
and test, and the fault/no-fault ratio stays similar across folds), inner
GroupKFold + RandomizedSearchCV for hyperparameter tuning, and an
imbalanced-learn Pipeline so SMOTE is fit only on each training fold.

The outer split is repeated config.N_CV_REPEATS times with a different
random fold assignment each time, so the paired significance testing in
evaluate.py has more than N_OUTER_FOLDS paired observations to work with.
"""
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import config
import models


def build_pipeline(model_name, use_smote, seed=config.RANDOM_SEED):
    """Assemble the (optional scaler) -> (optional SMOTE) -> classifier
    pipeline for one model/condition. Scaling is applied only for the
    distance/gradient-sensitive models in config.SCALED_MODELS. Using an
    imbalanced-learn Pipeline (not sklearn's) ensures SMOTE resamples the
    training fold only, never the validation fold, under cross-validation.
    """
    steps = []
    if model_name in config.SCALED_MODELS:
        steps.append(("scaler", StandardScaler()))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=seed, k_neighbors=3)))
    steps.append(("clf", models.get_estimator(model_name, seed)))
    return ImbPipeline(steps)


def run_nested_cv(X, y, groups, model_name, use_smote, seed=config.RANDOM_SEED):
    """Run repeated (config.N_CV_REPEATS x config.N_OUTER_FOLDS) group-aware
    nested cross-validation for one model/condition.

    For each outer fold, in each repeat: an inner GroupKFold +
    RandomizedSearchCV (scored on macro F1) selects hyperparameters using
    only the outer-training data, so the outer test fold is never used for
    model selection; the resulting best estimator is then scored once on
    the held-out outer test fold.

    The outer split uses StratifiedGroupKFold(shuffle=True) with a distinct
    random_state per repeat (seed + repeat_index), so repeats give genuinely
    different fold assignments rather than the same partition N times, while
    still keeping every episode wholly inside one fold and keeping the
    fault/no-fault ratio comparable across folds.

    Returns a list of per-outer-fold dicts: repeat, fold (a running index
    unique across all repeat*fold combinations), y_true, y_pred, y_score
    (or None if the estimator has no predict_proba), best_params (the
    winning RandomizedSearchCV configuration for that fold), n_train, n_test.
    With the default config.N_CV_REPEATS=3 and config.N_OUTER_FOLDS=5, this
    returns 15 dicts instead of 5.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    fold_results = []
    running_fold = 0

    for repeat in range(config.N_CV_REPEATS):
        outer = StratifiedGroupKFold(
            n_splits=config.N_OUTER_FOLDS, shuffle=True, random_state=seed + repeat
        )
        for train_idx, test_idx in outer.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]

            pipe = build_pipeline(model_name, use_smote, seed)
            param_grid = models.get_param_grid(model_name)

            # Inner fold count is capped by how many episode groups are
            # actually available inside this outer training split (floored
            # at 2). Plain GroupKFold (not stratified) is used for the inner
            # loop since it is only for hyperparameter selection, not for
            # the paired significance test.
            n_inner_groups = len(np.unique(groups_train))
            n_inner_splits = min(config.N_INNER_FOLDS, n_inner_groups)
            inner = GroupKFold(n_splits=max(2, n_inner_splits))

            search = RandomizedSearchCV(
                pipe, param_grid, n_iter=config.N_RANDOM_SEARCH_ITER,
                cv=inner.split(X_train, y_train, groups_train),
                scoring="f1_macro", random_state=seed, n_jobs=-1, error_score=0.0,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_

            y_pred = best.predict(X_test)
            try:
                y_score = best.predict_proba(X_test)
            except Exception:
                y_score = None

            fold_results.append({
                "repeat": repeat,
                "fold": running_fold,
                "y_true": y_test,
                "y_pred": y_pred,
                "y_score": y_score,
                "best_params": search.best_params_,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })
            running_fold += 1
    return fold_results
