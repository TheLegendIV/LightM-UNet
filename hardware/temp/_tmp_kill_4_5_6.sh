s#!/bin/bash
# Kill partition 4/5/6 subprocess trees, then freeze (SIGSTOP) their python workers
# so they never pick up further queued partitions. Partition 1 (worker 614) untouched.
killtree() {
  local pid=$1
  for child in $(pgrep -P "$pid"); do
    killtree "$child"
  done
  kill -9 "$pid" 2>/dev/null
}

killtree 238605   # partition 4 vivado_stitch_proj tree (under worker 613)
killtree 401124   # partition 5 verilator rtlsim (under worker 616)
killtree 454898   # partition 6 vivado_stitch_proj tree (under worker 615)

kill -STOP 613
kill -STOP 615
kill -STOP 616

echo "done: killed partition 4/5/6 subprocess trees, froze workers 613/615/616"
