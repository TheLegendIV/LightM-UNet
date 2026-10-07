#!/bin/bash
cd /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.runs
echo "=== Per-run filemgmt 20-1741 counts ==="
for d in */runme.log; do
  c=$(grep -c "filemgmt 20-1741" "$d" 2>/dev/null)
  if [ "$c" != "0" ] && [ -n "$c" ]; then
    echo "$d : $c"
  fi
done
echo "=== Distinct colliding filenames across ALL logs ==="
grep -h "File '" */runme.log 2>/dev/null | grep "filemgmt 20-1741" -A0 | true
grep -h "used by one or more modules" -B2 */runme.log 2>/dev/null | grep "File '" | sed -E "s/.*File '([^']+)'.*/\1/" | sort | uniq -c | sort -rn | head -40
