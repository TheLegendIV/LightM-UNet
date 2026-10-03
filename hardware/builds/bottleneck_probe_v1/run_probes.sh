#!/bin/bash
# Launch the bottleneck probe builds in the FINN container (host-side driver, bash / Git Bash).
#   CONTAINER=<name from `docker ps`> bash run_probes.sh                 # all 15 cases, rtlsim only
#   CONTAINER=... CASES="bottleneck_cin32_d8_int4" STOP_AFTER=folding bash run_probes.sh   # cheap gate-3 dry run
# Env: CONTAINER (required) | CASES (default: all 15) | STOP_AFTER (convert|folding|fifo|stitch|rtlsim, default rtlsim)
#      EXTRA (extra args for finn_bottleneck_probe_build.py, e.g. "--ooc" or "--skip-scale 0.5" or "--no-merge")
#      JOBS (parallel builds, default 4) | TIMEOUT (seconds per build, default 14400)
# Results end up in the container under .../enet/finn_deployment_outputs/<case>_<tag>_<ts>/probe_result.json;
# copy them back with collect_probe_outputs.sh.
set -eo pipefail
: "${CONTAINER:?set CONTAINER to the running FINN container name (docker ps)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW="$HERE/../.."
ENET=/home/thelegendiv/finn/notebooks/enet
STOP_AFTER="${STOP_AFTER:-rtlsim}"
JOBS="${JOBS:-4}"
TIMEOUT="${TIMEOUT:-14400}"
EXTRA="${EXTRA:-}"
if [ -z "$CASES" ]; then
  CASES=""
  for d in 1 2 4 8 16; do for b in 4 6 8; do CASES="$CASES bottleneck_cin32_d${d}_int${b}"; done; done
fi

echo "=== deploy flat .py files + inputs into $CONTAINER:$ENET ==="
for f in "$HERE/finn_bottleneck_probe_build.py" "$HW/finn_compose_thresholds.py" \
         "$HW/finn_enet_build.py" "$HW/finn_enet_build_fixups.py" "$HW/finn_enet_convert_to_hw_rtl_mvau.py" \
         "$HW/finn_s12_build_steps.py" "$HW/finn_stage_partition.py" "$HW/finn_partition_build_steps.py"; do
  docker cp "$f" "$CONTAINER:$ENET/"
done
for c in $CASES; do
  for s in .onnx _probe.json _folding.json; do docker cp "$HERE/inputs/$c$s" "$CONTAINER:$ENET/"; done
done

run_case() {
  local c="$1"
  docker exec -e HOME=/tmp/home_dir "$CONTAINER" bash -c \
    "source /tools/Xilinx/Vivado/2022.2/settings64.sh >/dev/null 2>&1; cd $ENET && \
     timeout $TIMEOUT python3 finn_bottleneck_probe_build.py $c --stop-after $STOP_AFTER $EXTRA" \
    > "/tmp/probe_${c}.log" 2>&1 && echo "OK   $c" || echo "FAIL $c (see /tmp/probe_${c}.log; exit code 124 = timeout, i.e. possible deadlock)"
}
export -f run_case
export CONTAINER ENET STOP_AFTER TIMEOUT EXTRA
echo "=== running: $CASES  (stop-after $STOP_AFTER, $JOBS parallel) ==="
printf '%s\n' $CASES | xargs -P "$JOBS" -I{} bash -c 'run_case {}'
