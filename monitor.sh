#!/bin/bash
# monitor.sh: clean live progress of the control experiments (no warnings).
# Prints one snapshot and returns to the prompt; run again to refresh.
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"
L=results/logs; LOG=$L/controls.log
show() {
  if [ -f $L/run_all.pid ] && kill -0 "$(cat $L/run_all.pid)" 2>/dev/null; then st="RUNNING"
  elif grep -q "RUN COMPLETE" $L/status.txt 2>/dev/null; then st="FINISHED"; else st="NOT RUNNING"; fi
  R=3; grep -q "QUICK" "$LOG" 2>/dev/null && R=1
  per=$((5 * R)); total=$((3 * per))
  a=$(grep -c "1a full: repeat" "$LOG" 2>/dev/null); b=$(grep -c "1b hour-matched: repeat" "$LOG" 2>/dev/null); c=$(grep -c "3 random split: repeat" "$LOG" 2>/dev/null)
  a=${a:-0}; b=${b:-0}; c=${c:-0}; done=$((a + b + c)); left=$((total - done))
  mins=$(grep -o '+ *[0-9.]* min' "$LOG" 2>/dev/null | tail -n 1 | tr -dc '0-9.'); mins=${mins:-0}
  eta=$(awk -v m="$mins" -v d="$done" -v t="$total" 'BEGIN{if(d>0) printf "%.0f", m/d*(t-d); else print "?"}')
  errs=$(grep -c -E "Traceback|FAILED|Error:" "$LOG" $L/status.txt 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
  echo "$(date '+%H:%M:%S')  $st"
  echo "Folds done: $done of $total   remaining: $left   elapsed: $mins min   time left: ~$eta min"
  echo "Exp 1a full: $a/$per   Exp 1b hour-matched: $b/$per   Exp 3 random split: $c/$per"
  echo "Last step: $(grep -v -i warn "$LOG" 2>/dev/null | grep '^\[' | tail -n 1 | cut -c24-)"
  if [ "$errs" -gt 0 ]; then echo "ERRORS: $errs  (see results/logs/controls.log)"; elif grep -q "CONTROLS DONE" "$LOG" 2>/dev/null; then echo "DONE: results/controls/SUMMARY.txt"; else echo "Errors: none"; fi
}
show
