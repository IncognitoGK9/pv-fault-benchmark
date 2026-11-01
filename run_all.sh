#!/bin/bash
# run_all.sh: build a working Python environment outside Google Drive, run
# the pre-flight checks, then run the three control experiments
# (control_experiments.py), keeping the Mac awake throughout.
# Start it with start_runs.sh (runs in the background); watch with monitor.sh.
set -uo pipefail
# Ignore any Python settings left in the terminal (e.g. a PYTHONPATH pointing
# at the old Google Drive venv), so only the clean environment below is used.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV PYTHONSTARTUP CONDA_PREFIX
export PYTHONNOUSERSITE=1
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"
LOGD="$DIR/results/logs"; mkdir -p "$LOGD"
STATUS="$LOGD/status.txt"
stamp() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

# Keep the Mac awake (display, idle, disk and system sleep) while this script runs.
if command -v caffeinate >/dev/null 2>&1; then caffeinate -dimsu -w $$ & stamp "caffeinate on"; fi

stamp "=== RUN STARTED (pid $$) ==="

# ---- 1. Python environment, kept outside Google Drive (Drive breaks venvs)
VENV="${PV_VENV:-$HOME/.venvs/pv-fault}"
PY="$VENV/bin/python"
need_check='import numpy, pandas, scipy, sklearn, imblearn, xgboost, shap, matplotlib'
if [ -x "$PY" ] && "$PY" -c "$need_check" >/dev/null 2>&1; then
  stamp "environment OK: $VENV"
else
  stamp "building environment in $VENV (first time only, 2 to 5 min)"
  PY_BASE=""
  for c in /Library/Developer/CommandLineTools/usr/bin/python3 python3.9 python3.10 python3.11 python3.12 /usr/bin/python3 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || continue
      case "$v" in 3.9|3.10|3.11|3.12) PY_BASE="$c"; break;; esac
    fi
  done
  if [ -z "$PY_BASE" ]; then stamp "FAILED: no Python 3.9 to 3.12 found"; exit 1; fi
  stamp "base Python: $PY_BASE ($("$PY_BASE" --version 2>&1))"
  rm -rf "$VENV"; mkdir -p "$(dirname "$VENV")"
  "$PY_BASE" -m venv "$VENV" >>"$LOGD/setup.log" 2>&1 || { stamp "FAILED: could not create venv (see setup.log)"; exit 1; }
  "$PY" -m pip install --upgrade pip >>"$LOGD/setup.log" 2>&1
  if ! "$PY" -m pip install --ignore-installed -r requirements.txt >>"$LOGD/setup.log" 2>&1; then
    stamp "pinned install failed; installing compatible latest versions instead"
    "$PY" -m pip install pandas numpy scikit-learn imbalanced-learn scipy matplotlib xgboost "shap>=0.44" >>"$LOGD/setup.log" 2>&1 \
      || { stamp "FAILED: package install (see setup.log)"; exit 1; }
  fi
  # XGBoost on macOS needs the OpenMP runtime
  if ! "$PY" -c "import xgboost" >>"$LOGD/setup.log" 2>&1; then
    if command -v brew >/dev/null 2>&1; then stamp "installing libomp for XGBoost"; brew install libomp >>"$LOGD/setup.log" 2>&1; fi
    "$PY" -c "import xgboost" >>"$LOGD/setup.log" 2>&1 || { stamp "FAILED: XGBoost cannot load (run: brew install libomp)"; exit 1; }
  fi
  "$PY" -c "$need_check" >>"$LOGD/setup.log" 2>&1 || { stamp "FAILED: packages still not importable (see setup.log)"; exit 1; }
  stamp "environment built"
fi
"$PY" -m pip freeze > "$LOGD/environment_freeze.txt" 2>/dev/null

# ---- 2. Pre-flight checks (about 1 min)
stamp "preflight started"
if ! "$PY" -u preflight.py > "$LOGD/preflight.log" 2>&1; then
  stamp "FAILED: preflight (see results/logs/preflight.log)"; exit 1
fi
stamp "preflight passed"

# ---- 3. Control experiments (hour-matched check, corrected tests, random split)
stamp "control experiments started"
"$PY" -u control_experiments.py ${PV_SMOKE:+--quick} > "$LOGD/controls.log" 2>&1
rc=$?; [ $rc -eq 0 ] && stamp "control experiments finished: see results/controls/SUMMARY.txt" || stamp "control experiments FAILED (exit $rc; see results/logs/controls.log)"

stamp "=== RUN COMPLETE ==="
