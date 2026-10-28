LOG=/tmp/p7mvu_build.log
for i in $(seq 1 180); do
  if grep -qE "build done, starting OOC|Traceback|Error" $LOG; then break; fi
  if ! pgrep -f finn_s12_build >/dev/null; then break; fi
  sleep 10
done
grep -nE "build done|Traceback|Error|OOC synth done|rtlsim" $LOG | tail -10
tail -c 600 $LOG
