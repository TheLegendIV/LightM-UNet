for i in $(seq 1 240); do
  grep -q "^rc=" /tmp/golden_p7mvu.log && break
  sleep 10
done
tail -30 /tmp/golden_p7mvu.log | cut -c1-400
cat /tmp/golden_pp_p7mvu/summary.json 2>/dev/null | head -60
