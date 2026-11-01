#!/bin/bash
# start_revision.sh: run revision_runs.py once on this Mac, in the background,
# with the clean environment built by run_all.sh (~/.venvs/pv-fault).
# Keep the Mac plugged in and the lid open. About 30 to 40 min.
# Watch progress with:  tail -f results/revision/log.txt
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV PYTHONSTARTUP CONDA_PREFIX
export PYTHONNOUSERSITE=1
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"
PY="${PV_VENV:-$HOME/.venvs/pv-fault}/bin/python"
if ! "$PY" -c 'import numpy, pandas, scipy, sklearn, imblearn, xgboost, matplotlib' >/dev/null 2>&1; then
  echo "Environment not found at $PY. Run 'bash start_runs.sh' once first (it builds it), or set PV_VENV."; exit 1
fi
mkdir -p results/logs
export REVISION_JOBS="${REVISION_JOBS:-4}"
if command -v caffeinate >/dev/null 2>&1; then
  nohup caffeinate -dimsu "$PY" revision_runs.py > results/logs/revision.out 2>&1 &
else
  nohup "$PY" revision_runs.py > results/logs/revision.out 2>&1 &
fi
echo "Started (pid $!). Progress: tail -f results/revision/log.txt   Result: results/revision/SUMMARY.txt"
