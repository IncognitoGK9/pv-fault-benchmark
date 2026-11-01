"""
models.py
The seven classifiers under comparison -- five classical/interpretable
models plus Random Forest and XGBoost -- with their randomized-search
hyperparameter grids (searched under nested/inner CV; see pipeline.py).
"""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

import config


def get_estimator(name, seed=config.RANDOM_SEED):
    """Return a freshly-constructed, unfitted classifier for `name`.

    `name` must be one of config.MODEL_NAMES. All stochastic estimators are
    seeded for reproducibility; solver-agnostic defaults are used except
    where a higher max_iter is needed for convergence (Logistic Regression,
    MLP) or where early stopping is used to control MLP overfitting.
    """
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "kNN":
        return KNeighborsClassifier()
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=seed)
    if name == "MLP":
        return MLPClassifier(max_iter=500, random_state=seed, early_stopping=True)
    if name == "RandomForest":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if name == "XGBoost":
        if not _HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Run: pip install xgboost>=2.0 "
                "(see requirements.txt), or remove 'XGBoost' from "
                "config.MODEL_NAMES to skip it."
            )
        return XGBClassifier(
            random_state=seed, n_jobs=1, eval_metric="logloss",
            tree_method="hist",
        )
    raise ValueError(name)


def get_param_grid(name):
    """Return the RandomizedSearchCV parameter grid for `name` (paper Table IV).

    Keys are prefixed `clf__` to match the `clf` step name in the
    imbalanced-learn Pipeline built by pipeline.build_pipeline(). Grid sizes
    range from 6 (Logistic Regression, exhaustive) to 324 (XGBoost)
    combinations. With 15 random draws per outer fold, coverage is unequal:
    100% for Logistic Regression but 4.6% for XGBoost (paper Table IV).
    """
    if name == "LogisticRegression":
        return {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__penalty": ["l2"]}
    if name == "kNN":
        return {"clf__n_neighbors": [3, 5, 11, 21, 31, 51],
                 "clf__weights": ["uniform", "distance"],
                 "clf__p": [1, 2]}
    if name == "DecisionTree":
        return {"clf__max_depth": [3, 4, 6, 8, 12, 16, None],
                 "clf__min_samples_leaf": [1, 5, 10, 20],
                 "clf__criterion": ["gini", "entropy"]}
    if name == "SVM":
        return {"clf__C": [0.1, 1, 10, 100],
                 "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1]}
    if name == "MLP":
        return {"clf__hidden_layer_sizes": [(16,), (32,), (64,), (16, 8), (32, 16)],
                 "clf__alpha": [0.0001, 0.001, 0.01, 0.1],
                 "clf__learning_rate_init": [0.0005, 0.001, 0.01]}
    if name == "RandomForest":
        # n_estimators is capped at 300 (not 500+) deliberately: profiling
        # on this project's hardware showed a single 500-tree fit taking
        # ~5x longer than a 100-tree fit, which multiplied across
        # config.N_RANDOM_SEARCH_ITER candidates x config.N_CV_REPEATS
        # repeats would make Random Forest the runtime bottleneck of the
        # whole 28-condition experiment; 300 keeps it in the same runtime
        # ballpark as the other tree-based models while still searching a
        # meaningful ensemble-size range.
        return {"clf__n_estimators": [100, 200, 300],
                 "clf__max_depth": [None, 8, 16],
                 "clf__min_samples_leaf": [1, 5, 10],
                 "clf__max_features": ["sqrt", "log2"]}
    if name == "XGBoost":
        return {"clf__n_estimators": [100, 300, 500],
                 "clf__max_depth": [3, 6, 9],
                 "clf__learning_rate": [0.01, 0.05, 0.1, 0.3],
                 "clf__subsample": [0.7, 0.85, 1.0],
                 "clf__colsample_bytree": [0.7, 0.85, 1.0]}
    raise ValueError(name)
