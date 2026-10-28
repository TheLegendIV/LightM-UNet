#!/bin/bash
for c in "$@"; do
  echo "===$c==="
  tail -n 20 "/tmp/probe_${c}.log" 2>/dev/null
  echo
done
