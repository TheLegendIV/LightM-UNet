cd /home/thelegendiv/finn
F=finn-rtllib/mvu/add_multi.sv
git diff --stat -- $F
python3 /tmp/_patch_add_multi2.py $F && python3 /tmp/_patch_add_multi3.py $F
git diff -- $F | head -60
git add $F
git commit -q -m "add_multi.sv: scalar-valued leaf_load() at module scope instead of unpacked-array constant function under generate (Verilator 5 compatibility; no functional change)" -- $F
git log --oneline | head -3
