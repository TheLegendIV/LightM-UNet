#!/bin/bash
# Copy every probe_result.json (+ report/ + landed folding) out of the container, BEFORE the container is recreated.
#   CONTAINER=<name> bash collect_probe_outputs.sh
set -eo pipefail
: "${CONTAINER:?set CONTAINER}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENET=/home/thelegendiv/finn/notebooks/enet
DEST="$HERE/results"
mkdir -p "$DEST"
for d in $(docker exec "$CONTAINER" bash -c "ls -d $ENET/finn_deployment_outputs/bottleneck_cin32_* $ENET/finn_deployment_outputs/dn_cin16_* $ENET/finn_deployment_outputs/up_cin32_* $ENET/finn_deployment_outputs/up_cin16_* $ENET/finn_deployment_outputs/init_cin1_* $ENET/finn_deployment_outputs/fnl_cin4_* 2>/dev/null"); do
  n=$(basename "$d")
  mkdir -p "$DEST/$n"
  docker cp "$CONTAINER:$d/probe_result.json" "$DEST/$n/" 2>/dev/null || echo "no probe_result.json in $n"
  docker cp "$CONTAINER:$d/report" "$DEST/$n/" 2>/dev/null || true
  # raw Vivado utilization (the OOC json DSP field is wrong): lives under the ephemeral FINN_BUILD_DIR
  docker exec "$CONTAINER" bash -c "find $d/finn_build_tmp -name 'utilization_placed.rpt' 2>/dev/null | head -1" \
    | while read -r rpt; do [ -n "$rpt" ] && docker cp "$CONTAINER:$rpt" "$DEST/$n/" ; done
done
echo "collected into $DEST"
