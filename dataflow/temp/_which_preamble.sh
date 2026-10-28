cd /home/thelegendiv/finn/notebooks/enet
grep -a -m4 -E '^Preamble:|^Conv order|^Folding json' /tmp/ooc_S12_u8in_full.log | cut -c1-300
ls -lt --time-style=+%F_%T /tmp/*.log | head -12
sed -n 280,335p finn_s12_build.py
