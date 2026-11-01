"""
preflight.py

Fast checks (about 1 minute) run before the long jobs, so that a missing
package, missing file or broken model fails immediately instead of an hour
in. Exits with code 0 if everything is ready, 1 otherwise.
"""
import os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE)
ok = True
def check(name, fn):
    global ok
    t = time.time()
    try:
        msg = fn()
        print(f"  OK    {name}{': ' + msg if msg else ''} ({time.time()-t:.1f} s)", flush=True)
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc(limit=2)

print(f"Python {sys.version.split()[0]} at {sys.executable}")

def imports():
    import numpy, pandas, scipy, sklearn, imblearn, xgboost, shap, matplotlib  # noqa: F401
    return (f"numpy {numpy.__version__}, pandas {pandas.__version__}, scipy {scipy.__version__}, "
            f"scikit-learn {sklearn.__version__}, imbalanced-learn {imblearn.__version__}, "
            f"xgboost {xgboost.__version__}, shap {shap.__version__}")
check("packages import", imports)

def project():
    import config, features, target, models, pipeline, evaluate, shap_analysis, prepare_dataset  # noqa: F401
    return f"QUICK_TEST_MODE={config.QUICK_TEST_MODE}"
check("project modules import", project)

def files():
    import config
    missing = [p for p in (config.SUBSAMPLE_CSV, os.path.join(config.RESULTS_DIR, "summary_metrics.json"),
                           os.path.join(config.RESULTS_DIR, "shap_importance.json")) if not os.path.exists(p)]
    if missing: raise FileNotFoundError(", ".join(missing))
    return "subsample and saved results present"
check("project files", files)

def raw():
    import config, pandas as pd
    if not os.path.exists(config.RAW_CSV):
        raise FileNotFoundError("raw CSV not found; looked in: " + " | ".join(config.RAW_CSV_CANDIDATES))
    head = pd.read_csv(config.RAW_CSV, nrows=2000)
    need = ["time", "Fault"] + list(config.RAW_FEATURES)
    miss = [c for c in need if c not in head.columns]
    if miss: raise ValueError(f"raw CSV lacks columns {miss}")
    pd.to_datetime(head["time"], utc=True)
    return f"{config.RAW_CSV} ({os.path.getsize(config.RAW_CSV)/1e6:.0f} MB)"
# raw CSV is not needed by control_experiments.py

def fit_models():
    import numpy as np, pandas as pd, config, features, target as tm, pipeline as pl
    df = pd.read_csv(config.SUBSAMPLE_CSV)
    X, cols = features.build_features(df); d = df.loc[X.index]
    y = np.asarray(tm.get_target(d, "binary")); X = X.values
    rng = np.random.default_rng(0); idx = rng.choice(len(y), 1500, replace=False)
    for m in config.MODEL_NAMES:
        pl.build_pipeline(m, use_smote=True).fit(X[idx], y[idx]).predict_proba(X[idx[:50]])
    return f"{len(config.MODEL_NAMES)} models fit with SMOTE"
check("model pipelines fit", fit_models)

def shap_small():
    import numpy as np, pandas as pd, shap, config, features, target as tm, pipeline as pl
    df = pd.read_csv(config.SUBSAMPLE_CSV)
    X, cols = features.build_features(df); d = df.loc[X.index]
    y = np.asarray(tm.get_target(d, "binary")); X = X.values[:1500]; y = y[:1500]
    if len(set(y)) < 2:
        rng = np.random.default_rng(0); full = features.build_features(df)[0].values
        yy = np.asarray(tm.get_target(df.loc[features.build_features(df)[0].index], "binary"))
        idx = rng.choice(len(yy), 1500, replace=False); X, y = full[idx], yy[idx]
    clf = pl.build_pipeline("RandomForest", use_smote=True).fit(X, y).named_steps["clf"]
    shap.TreeExplainer(clf).shap_values(X[:50])
    return "TreeExplainer works"
check("SHAP explainer", shap_small)

print("PREFLIGHT PASSED" if ok else "PREFLIGHT FAILED")
sys.exit(0 if ok else 1)
