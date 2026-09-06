#!/bin/bash
# Monitors the running S19 OOC-synth FINN build (PID given as $1), printing a
# step-progress snapshot every 30 minutes, and exits as soon as the build
# process itself exits (success or failure) so the caller gets notified
# immediately rather than waiting for the next 30-min tick.
PID="$1"
LOG="$2"
STDOUT_LOG="$3"
INTERVAL=1800  # 30 minutes

echo "Monitoring PID=$PID"
echo "Build log: $LOG"
echo "Stdout log: $STDOUT_LOG"

while kill -0 "$PID" 2>/dev/null; do
    for _ in $(seq 1 60); do
        kill -0 "$PID" 2>/dev/null || break
        sleep 30
    done
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') — still running ==="
    grep 'Running step' "$LOG" | tail -3
done

echo "=== BUILD PROCESS EXITED at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "--- last 80 lines of build_dataflow.log ---"
tail -80 "$LOG"
echo "--- last 80 lines of stdout/stderr log ---"
tail -80 "$STDOUT_LOG"
