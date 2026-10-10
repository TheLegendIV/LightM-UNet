LOG=/tmp/p7mvu_build.log
for i in $(seq 1 360); do
  if grep -qE "rtlsim (PASS|DEADLOCK|FAILED)|Traceback" $LOG; then break; fi
  if ! pgrep -f finn_s12_build >/dev/null; then break; fi
  sleep 10
done
grep -nE "OOC synth done|rtlsim|Traceback" $LOG | cut -c1-1500 | tail -8
