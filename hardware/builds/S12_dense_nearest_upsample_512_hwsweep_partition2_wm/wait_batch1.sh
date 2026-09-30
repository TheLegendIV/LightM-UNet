#!/bin/bash
PIDS="3753540 3753542 3753544 3753546"
while true; do
  ALIVE=0
  for p in $PIDS; do
    if kill -0 $p 2>/dev/null; then ALIVE=$((ALIVE+1)); fi
  done
  echo "$(date +%T) alive=$ALIVE"
  if [ $ALIVE -eq 0 ]; then break; fi
  sleep 30
done
echo BATCH1_DONE
