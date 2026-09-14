"""
config.py
Configuration for the PV shading-fault classical-ML comparison study.

Dataset: Reunion domestic rooftop PV plant electrical production data
(Graillet et al., 2022, Data in Brief, 46:108723; Zenodo DOI
10.5281/zenodo.7828879), file dt2_electrical_production_inverter_1_with_faults.csv.
"""
import glob
import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
RAW_CSV_NAME = "dt2_electrical_production_inverter_1_with_faults.csv"
DATA_DIR = os.path.join(_HERE, "data")
FIG_DIR = os.path.join(_HERE, "figures")
RESULTS_DIR = os.path.join(_HERE, "results")
SUBSAMPLE_CSV = os.path.join(DATA_DIR, "reunion_subsample.csv")
SUBSAMPLE_META = os.path.join(DATA_DIR, "subsample_metadata.json")

for d in (DATA_DIR, FIG_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# RAW_CSV is resolved by checking a short list of candidate locations (next
# to this script, a data/ subfolder, and a couple of Google Drive mount
# patterns) and taking the first that exists. If none match on your
# machine, either copy the CSV into this folder or edit RAW_CSV directly.
_DRIVE_RELATIVE = os.path.join(
    "1. Strathmore", "COURSE WORK", "MACHINE LEARNING NOTES", "Model_comparison"
)

_raw_candidates = [
    os.path.join(_HERE, RAW_CSV_NAME),
    os.path.join(DATA_DIR, RAW_CSV_NAME),
    os.path.join("/Users/incognito/My Drive", _DRIVE_RELATIVE, RAW_CSV_NAME),
    os.path.expanduser(os.path.join("~/My Drive", _DRIVE_RELATIVE, RAW_CSV_NAME)),
]
_raw_candidates += glob.glob(
    os.path.expanduser(
        os.path.join("~/Library/CloudStorage/GoogleDrive-*/My Drive", _DRIVE_RELATIVE, RAW_CSV_NAME)
    )
)

RAW_CSV_CANDIDATES = _raw_candidates
RAW_CSV = next((p for p in _raw_candidates if os.path.exists(p)), _raw_candidates[0])

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
RANDOM_SEED = 42

# ----------------------------------------------------------------------
# Subsampling (computational tractability; see README.md and paper Section III.A)
#
# The raw file has 2,988,814 rows at ~6s sampling. Faults occur in only 17
# contiguous, deliberately-staged episodes (plus 18 contiguous no-fault
# segments) across the 559-day span, so a random row split would leak
# information (adjacent rows within an episode are near-duplicates).
# Mitigation: (a) segment the series into contiguous same-label runs,
# (b) cap how many rows any single segment may contribute, and (c) treat
# each retained segment as an indivisible group for cross-validation
# (GroupKFold), so no episode is ever split across train and test.
# ----------------------------------------------------------------------
MAX_ROWS_PER_FAULT_SEGMENT = 400     # cap per contiguous fault episode
MAX_ROWS_PER_NOFAULT_SEGMENT = 2000  # cap per contiguous no-fault run
MAX_NOFAULT_SEGMENTS_ROWS = 15000    # overall cap on retained no-fault rows

# ----------------------------------------------------------------------
# Label scheme definitions (from Graillet et al., 2022, Table 3)
#   0    : no-fault
#   1    : uniform shading (entire line obscured)
#   2.1  : constant partial shading - 1 whole module
#   2.2  : constant partial shading - 2 whole modules
#   2.3  : constant partial shading - 3 whole modules
#   3.1  : constant partial shading - 1/3 of a module
#   3.2  : constant partial shading - 2/3 of a module
#   4    : intermittent partial shading (shadow moves with the sun)
# ----------------------------------------------------------------------
FAULT_CODE_NAMES = {
    0.0: "No-fault",
    1.0: "Uniform shading",
    2.1: "Partial shading (2.1: 1 module)",
    2.2: "Partial shading (2.2: 2 modules)",
    2.3: "Partial shading (2.3: 3 modules)",
    3.1: "Partial shading (3.1: 1/3 module)",
    3.2: "Partial shading (3.2: 2/3 module)",
    4.0: "Intermittent partial shading",
}

# Three-tier severity scheme used as the "ordinal" companion to the binary
# Fault/No-Fault target (paper Section III.C). Codes 2.x/3.x are all
# constant partial shading confined to a fixed fraction of the array;
# codes 1 and 4 represent a materially larger, less localised loss of
# production and are grouped as the more severe tier. This grouping is a
# stated modelling choice, not a property of the raw data.
TIER_MAP = {
    0.0: 0,   # No-Fault
    2.1: 1, 2.2: 1, 2.3: 1, 3.1: 1, 3.2: 1,   # Constant partial shading
    1.0: 2, 4.0: 2,                            # Severe (uniform / intermittent)
}
TIER_NAMES = {0: "No-Fault", 1: "Constant-Partial-Shading", 2: "Severe (Uniform/Intermittent)"}

# ----------------------------------------------------------------------
# Feature columns (raw electrical channels; see Graillet et al., Table 2)
#   Ia: PV-side current (A)         Ig: grid-injected current (A)
#   Eg: grid-injected energy (kWh)  Fg: grid frequency (Hz)
#   Pg: grid-injected power (W)     Va: PV-side voltage (V)  Vg: grid voltage (V)
# ----------------------------------------------------------------------
RAW_FEATURES = ["Ia", "Ig", "Eg", "Fg", "Pg", "Va", "Vg"]
ROLLING_WINDOWS = [5, 10]   # samples (~30s, ~60s at the nominal 6s cadence)

# ----------------------------------------------------------------------
# Cross-validation protocol
#
# N_OUTER_FOLDS is set to 5 (rather than a more conventional 10) because
# there are only 17 real fault episodes in total; group-aware CV needs
# enough distinct groups per fold, and 5 was the largest split found to
# leave an adequate number of fault episodes on both sides of every fold.
#
# N_CV_REPEATS repeats the entire outer split N times with a different
# random fold assignment each time (StratifiedGroupKFold with shuffle=True,
# a different random_state per repeat), so the paired significance testing
# in evaluate.py has N_CV_REPEATS * N_OUTER_FOLDS paired observations
# instead of just N_OUTER_FOLDS. A Wilcoxon test on only 5 paired folds has
# limited power to distinguish a real small effect from noise; 3 repeats
# raises the paired sample to 15 at the cost of roughly 3x the runtime of
# the full experiment.
# ----------------------------------------------------------------------
# QUICK_TEST_MODE: set True for a fast end-to-end sanity check of the whole
# pipeline (a few minutes) BEFORE committing to the full run. With 7 models
# x 2 schemes x 2 SMOTE conditions x 3 repeats x 15-iteration random search,
# the full run takes several hours on commodity hardware -- see README.md
# "Expected runtime" for measured figures. Leave False for the run whose
# results go into the paper.
QUICK_TEST_MODE = False

N_OUTER_FOLDS = 5
N_CV_REPEATS = 1 if QUICK_TEST_MODE else 3   # 1 for a quick sanity check, 3 for the full run
N_INNER_FOLDS = 3
N_RANDOM_SEARCH_ITER = 3 if QUICK_TEST_MODE else 15   # bounded for runtime on commodity hardware

LABEL_SCHEMES = ["binary", "tier"]
SMOTE_CONDITIONS = [True, False]

# Seven classifiers: five classical/interpretable models (Logistic
# Regression, kNN, Decision Tree, SVM, MLP) plus Random Forest and XGBoost,
# the two most common modern tabular-classification baselines (paper
# Section II.A). Both are tree ensembles and so are intentionally left out
# of SCALED_MODELS below, exactly like the single Decision Tree already was.
MODEL_NAMES = ["LogisticRegression", "kNN", "DecisionTree", "SVM", "MLP",
               "RandomForest", "XGBoost"]
SCALED_MODELS = {"LogisticRegression", "kNN", "SVM", "MLP"}

# ----------------------------------------------------------------------
# SHAP feature-attribution analysis (shap_analysis.py)
#
# TreeExplainer (tree models) and LinearExplainer (Logistic Regression) are
# exact and fast at full sample size. kNN, SVM, and MLP have no closed-form
# SHAP algorithm, so a generic sampling-based explainer is used instead,
# evaluated against a subsampled background and a subsampled explanation
# set to keep runtime tractable -- standard practice for non-tree SHAP on
# datasets of this size.
# ----------------------------------------------------------------------
SHAP_BACKGROUND_SIZE = 100
SHAP_EXPLAIN_SIZE = 300
