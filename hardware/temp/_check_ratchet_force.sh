#!/bin/bash
echo "--- run log tail:"; tail -25 /tmp/ratchet256_run_p2_force.log | cut -c1-220
echo "--- jobs:"; ps aux | grep -E "finn_s12_build.py" | grep -v grep | sed -E 's/.*(finn_s12_build.py.*)/\1/' | cut -c1-200
