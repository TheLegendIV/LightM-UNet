#!/bin/bash
for p in $(ps -eo pid,args | grep 'while pgrep' | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 1
ps -eo pid,args | grep 'while pgrep' | grep -v grep | wc -l
