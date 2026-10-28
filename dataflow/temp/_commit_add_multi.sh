cd /home/thelegendiv/finn
A=$(git log -1 --format='%an' a75d71030); E=$(git log -1 --format='%ae' a75d71030)
echo "author: $A <$E>"
git -c user.name="$A" -c user.email="$E" commit -q -m "add_multi.sv: scalar-valued leaf_load() at module scope instead of unpacked-array constant function under generate (Verilator 5 compatibility; no functional change)" -- finn-rtllib/mvu/add_multi.sv
git log --oneline | head -3
git status --short | head
