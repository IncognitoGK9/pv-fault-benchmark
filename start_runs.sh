#!/bin/bash
# start_runs.sh: launch run_all.sh in the background so it survives closing
# the terminal. Keep the Mac plugged in and the lid open.
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"
mkdir -p results/logs
if [ -f results/logs/run_all.pid ] && kill -0 "$(cat results/logs/run_all.pid)" 2>/dev/null; then
  echo "A run is already in progress (pid $(cat results/logs/run_all.pid)). Use: bash monitor.sh"; exit 1
fi
: > results/logs/status.txt
nohup bash run_all.sh > results/logs/run_all.out 2>&1 &
echo $! > results/logs/run_all.pid
echo "Started (pid $!). Monitor with:  bash monitor.sh"
