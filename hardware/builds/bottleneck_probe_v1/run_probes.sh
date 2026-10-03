#!/bin/bash
# Launch the bottleneck probe builds in the FINN container (host-side driver, bash / Git Bash).
#   CONTAINER=<name from `docker ps`> bash run_probes.sh                 # all 15 cases, rtlsim only
#   CONTAINER=... CASES="bottleneck_cin32_d8_int4" STOP_AFTER=folding bash run_probes.sh   # cheap gate-3 dry run
# Env: CONTAINER (required) | CASES (default: all 15 regular) | SET=dn (the 6 downsampling probes) | SET=up (the 6 up4 upsampling probes) | SET=up5 (the 3 up5 probes) | SET=int (the 3 initial-block probes) | SET=fnl (the 6 final-deconvolution probes) | SET=pool (the 2 Pool-route probes) | STOP_AFTER (convert|folding|fifo|stitch|rtlsim, default rtlsim)
#      EXTRA (extra args for finn_bottleneck_probe_build.py, e.g. "--ooc" or "--skip-scale 0.5" or "--no-merge")
#      JOBS (parallel builds, default 4, HARD CAP 4 -- run ONE set at a time) | TIMEOUT (seconds per build, default 14400)
#      EXTRA defaults to "--ooc" (build + OOC synthesis for every probe); EXTRA="" = rtlsim only
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
EXTRA="${EXTRA---ooc}"      # default: build + OOC synthesis for every probe (EXTRA="" for rtlsim only)
if [ "$JOBS" -gt 4 ]; then echo "JOBS=$JOBS > 4: at most 4 builds at a time (Vivado/HLS memory)"; exit 2; fi
if [ -z "$CASES" ]; then
  CASES=""
  if [ "${SET:-reg}" = "pool" ]; then # SET=pool: the 2 Pool-route probes (MaxPool lowered to depthwise SWG + Pool_hls with PE): initial block + down2-like block, INT4
    CASES="init_cin1_cout4_in256_int4_pool dn_cin16_cout32_in64_int4_mvau_pool"
  elif [ "${SET:-reg}" = "fnl" ]; then  # SET=fnl: the 6 final-deconv probes (4->5, 128x128 in -> 256x256 out): INT4/6/8 x {bias, nobias}
    for b in 4 6 8; do for v in bias nobias; do CASES="$CASES fnl_cin4_cout5_in128_int${b}_${v}"; done; done
  elif [ "${SET:-reg}" = "int" ]; then  # SET=int: the 3 initial-block probes (1->4, 256x256 in -> 128x128 out), INT4/6/8
    for b in 4 6 8; do CASES="$CASES init_cin1_cout4_in256_int${b}"; done
  elif [ "${SET:-reg}" = "up5" ]; then  # SET=up5: 3 probes, ENet U4 up5 block (16->4, 64x64 in -> 128x128 out), INT4/6/8, conv decoder
    for b in 4 6 8; do CASES="$CASES up_cin16_cout4_in64_int${b}_conv"; done
  elif [ "${SET:-reg}" = "up" ]; then   # SET=up: the 6 upsampling probes (32->16, 32x32 in -> 64x64 out): INT4/6/8 x {conv, noconv}
    for b in 4 6 8; do for v in conv noconv; do CASES="$CASES up_cin32_cout16_in32_int${b}_${v}"; done; done
  elif [ "${SET:-reg}" = "dn" ]; then   # SET=dn: the 6 downsampling probes (16->32, 64x64 in): INT4/6/8 x {fmpad, mvau} skip
    for b in 4 6 8; do for v in fmpad mvau; do CASES="$CASES dn_cin16_cout32_in64_int${b}_${v}"; done; done
  else                                # default: the 15 regular-bottleneck probes
    for d in 1 2 4 8 16; do for b in 4 6 8; do CASES="$CASES bottleneck_cin32_d${d}_int${b}"; done; done
  fi
fi

echo "=== deploy flat .py files + inputs into $CONTAINER:$ENET ==="
for f in "$HERE/finn_bottleneck_probe_build.py" "$HW/finn_compose_thresholds.py" "$HW/finn_channel_pad.py" \
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
