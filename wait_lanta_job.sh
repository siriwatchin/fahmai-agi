#!/bin/bash
set -euo pipefail

JOB_ID="${1:-5826671}"
INTERVAL_SEC="${2:-60}"
PIPELINE_DIR="${PIPELINE_DIR:-/project/zz992000-zdevb/fahmai-agi/pipeline-qwen2.5}"
LOG_FILE="$PIPELINE_DIR/wait_job_${JOB_ID}.log"

cd "$PIPELINE_DIR"

echo "watch_job=$JOB_ID interval=${INTERVAL_SEC}s started_at=$(date -Is)" | tee -a "$LOG_FILE"

while true; do
  now="$(date -Is)"
  queue_line="$(squeue -h -j "$JOB_ID" -o '%i|%P|%j|%u|%T|%M|%R' || true)"

  if [ -n "$queue_line" ]; then
    echo "$now squeue $queue_line" | tee -a "$LOG_FILE"
  else
    state="$(sacct -n -P -j "$JOB_ID" --format=JobID,JobName,Partition,State,Elapsed,ExitCode,Reason | head -n 1 || true)"
    echo "$now sacct ${state:-job_not_found}" | tee -a "$LOG_FILE"

    out_file="$(ls -1 "$PIPELINE_DIR"/lanta_smoke."$JOB_ID".out "$PIPELINE_DIR"/lanta_smoke_dev."$JOB_ID".out "$PIPELINE_DIR"/lanta_100."$JOB_ID".out 2>/dev/null | head -n 1 || true)"
    err_file="$(ls -1 "$PIPELINE_DIR"/lanta_smoke."$JOB_ID".err "$PIPELINE_DIR"/lanta_smoke_dev."$JOB_ID".err "$PIPELINE_DIR"/lanta_100."$JOB_ID".err 2>/dev/null | head -n 1 || true)"

    if [ -n "$out_file" ]; then
      echo "--- stdout tail: $out_file ---" | tee -a "$LOG_FILE"
      tail -80 "$out_file" | tee -a "$LOG_FILE"
    fi
    if [ -n "$err_file" ]; then
      echo "--- stderr tail: $err_file ---" | tee -a "$LOG_FILE"
      tail -80 "$err_file" | tee -a "$LOG_FILE"
    fi
    break
  fi

  sleep "$INTERVAL_SEC"
done

echo "watch_job=$JOB_ID finished_at=$(date -Is)" | tee -a "$LOG_FILE"
