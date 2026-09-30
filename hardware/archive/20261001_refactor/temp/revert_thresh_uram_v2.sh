#!/bin/bash
set -e
DIRS=(
  /tmp/finn_dev_thelegendiv/vivado_stitch_proj_8ykqj8vi
  /tmp/finn_dev_thelegendiv/vivado_stitch_proj__3dsyxpp
  /tmp/finn_dev_thelegendiv/vivado_stitch_proj_d9ohtyc0
  /tmp/finn_dev_thelegendiv/vivado_stitch_proj_xf0fsb7n
  /tmp/finn_dev_thelegendiv/vivado_stitch_proj_2169cp1l
  /tmp/finn_dev_thelegendiv/vivado_stitch_proj_eb74ppz6
  /tmp/finn_dev_thelegendiv/vivado_stitch_proj_d3ax231a
  /tmp/finn_dev_thelegendiv/vivado_stitch_proj_l6ibgc02
)
for d in "${DIRS[@]}"; do
  echo "=== $d ==="
  if [ -d "$d.bak_preuram_fix" ]; then
    echo "backup already exists, skipping backup"
  else
    cp -a "$d" "$d.bak_preuram_fix"
    echo "backed up"
  fi
  before=$(grep -rlE "DEPTH_TRIGGER_URAM\([1-9]|DEPTH_TRIGGER_BRAM\([1-9]" "$d" 2>/dev/null | wc -l)
  find "$d" -name "*Thresholding_rtl_*_0.v" -print0 | xargs -0 sed -i \
    -e 's/\.DEPTH_TRIGGER_URAM([0-9]*)/.DEPTH_TRIGGER_URAM(0)/' \
    -e 's/\.DEPTH_TRIGGER_BRAM([0-9]*)/.DEPTH_TRIGGER_BRAM(0)/' \
    -e 's/DEPTH_TRIGGER_URAM=[0-9]*,/DEPTH_TRIGGER_URAM=0,/' \
    -e 's/DEPTH_TRIGGER_BRAM=[0-9]*,/DEPTH_TRIGGER_BRAM=0,/'
  after=$(grep -rlE "DEPTH_TRIGGER_URAM\([1-9]|DEPTH_TRIGGER_BRAM\([1-9]" "$d" 2>/dev/null | wc -l)
  echo "files with nonzero trigger before=$before after=$after"
done
echo "ALL DONE"
