#!/usr/bin/env bash
# pdll now, then pdll 0.3.1, on the 21 OpenML-CC18 tasks with at most 1,100 rows; one process per run.
# Runs whose latest record is a success are skipped, so an interrupted run resumes; OUT=<new file> reruns everything.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
PDLL031=${PDLL031:-pdll031}  # folder holding the extracted pdll 0.3.1 wheel (see README)
OUT=${OUT:-results/v031_vs_v1.jsonl}
TASKS="125920 3913 14954 146819 9946 9971 11 29 15 2079 10101 37 3560 3549 53 49 3022 31 9957 9981 146800"
records() { grep "\"task\": $1, .*\"pdll\": \"$2\"" "$OUT" 2>/dev/null; }
succeeded() { records "$1" "$2" | tail -1 | grep -q '"status": "ok"'; }
run() {  # version label timeout task
  local version=$1 label=$2 limit=$3 task=$4
  succeeded "$task" "$label" && return
  local before
  before=$(records "$task" "$label" | wc -l)
  timeout "$limit" "$PY" -I -W ignore run_task.py --version "$version" --pdll031 "$PDLL031" --out "$OUT" "$task" > "logs/$task-$label.log" 2>&1
  local code=$?
  if [ $code -eq 124 ]; then echo "{\"task\": $task, \"pdll\": \"$label\", \"status\": \"timeout\", \"seconds\": $limit}" >> "$OUT"
  elif [ "$(records "$task" "$label" | wc -l)" -eq "$before" ]; then echo "{\"task\": $task, \"pdll\": \"$label\", \"status\": \"crashed\", \"exit\": $code}" >> "$OUT"; fi
  echo "$(date +%H:%M) $label $task exit $code"
}
mkdir -p logs results
for t in $TASKS; do run now 1.0.0 1800 "$t"; done
for t in $TASKS; do run 0.3.1 0.3.1 3600 "$t"; done
echo ALL_DONE
