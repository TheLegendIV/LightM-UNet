#!/bin/bash
date +%H:%M:%S
ps aux | grep -E "verify_export|assign_stage|python3 -" | grep -v grep | cut -c1-160
