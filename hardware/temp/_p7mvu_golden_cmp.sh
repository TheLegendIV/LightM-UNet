python3 - <<'EOF'
import json
for tag, p in (("old", "/tmp/golden_pp/summary.json"), ("new", "/tmp/golden_pp_p7mvu/summary.json")):
    try:
        d = json.load(open(p))
    except Exception as e:
        print(tag, "missing", e); continue
    for c, v in sorted(d.items()):
        r = v.get("7")
        if r: print(tag, c, "agree %.6f bad %d" % (r["agreement"], r["n_bad"]), r.get("first_bad"), r.get("top_diffs(rtl-sw)"))
EOF
ls /tmp/golden_pp_p7mvu/case1 /tmp/golden_pp_p7mvu/case1/p7 | head -20
sed -n 1,40p /home/thelegendiv/finn/notebooks/enet/rtlsim_taps.py
grep -nE "add_argument" /home/thelegendiv/finn/notebooks/enet/rtlsim_taps.py /home/thelegendiv/finn/notebooks/enet/tap_compare.py
