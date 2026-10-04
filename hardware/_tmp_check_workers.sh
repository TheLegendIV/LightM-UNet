#!/bin/bash
pids=$(pgrep -f "python3 finn_s12_build.py" | tr '\n' ',' | sed 's/,$//')
ps -o pid,ppid,etimes,cmd -p "$pids"
echo "---VIVADO COUNT---"
pgrep -af "unwrapped.*vivado" | wc -l
