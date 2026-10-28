#!/bin/bash
ps -eo etime,args | grep finn_add_iodma | grep -v grep | cut -c1-80
echo "-- log"
grep -vE '^\s*$' /tmp/iodma_512.log | tail -15 | cut -c1-220
