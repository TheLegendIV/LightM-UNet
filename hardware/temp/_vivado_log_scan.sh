P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
cd "$P" || exit 1
ls
for r in $(find . -maxdepth 4 -type d -name 'synth_1' -o -maxdepth 4 -type d -name 'impl_1' | head); do
  echo "== $r"
  ls -l --time-style=+%F_%T "$r"/runme.log "$r"/vivado.log 2>/dev/null | cut -c1-140
  for f in "$r"/runme.log; do
    [ -f "$f" ] || continue
    echo "CRITICAL WARNING count: $(grep -c 'CRITICAL WARNING' "$f")"
    echo "20-1741 (module name collision) count: $(grep -c '20-1741' "$f")"
    grep 'CRITICAL WARNING' "$f" | sed -E 's/\[[^]]*\] //' | cut -c1-150 | sort | uniq -c | sort -rn | head -12
  done
done
