cd /home/thelegendiv/finn
git log -5 --format='%h %ad %s' --date=short -- finn-rtllib/mvu
echo ---
git log -1 --format='%h %ad %s' --date=short
git describe --tags --always 2>/dev/null
git branch --show-current
git status --short finn-rtllib | head
echo ---
ls finn-rtllib/mvu
grep -c "en" finn-rtllib/mvu/mvu_vvu_axi.sv
