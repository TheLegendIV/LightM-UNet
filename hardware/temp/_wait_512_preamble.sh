#!/bin/bash
while ps -eo args | grep 'finn_s12_preamble.py' | grep -v grep > /dev/null; do sleep 30; done
echo "preamble process ended"
grep -vE "Warning|warn" /tmp/preamble_512.log | tail -8 | cut -c1-200
