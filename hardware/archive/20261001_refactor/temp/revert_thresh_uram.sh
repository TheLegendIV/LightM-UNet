#!/bin/bash
set -e
cd /tmp/finn_dev_thelegendiv/combined_stitch_proj_dud9_c9n
if [ -d renamed_src.bak_preuram_fix ]; then
  echo "Backup already exists, skipping backup step."
else
  echo "Backing up renamed_src -> renamed_src.bak_preuram_fix ..."
  cp -a renamed_src renamed_src.bak_preuram_fix
  echo "Backup done."
fi

echo "Patching DEPTH_TRIGGER_URAM/BRAM back to 0 (auto) in all Thresholding_rtl instances..."
find renamed_src -name "*Thresholding_rtl_*_0.v" -print0 | xargs -0 sed -i \
  -e 's/\.DEPTH_TRIGGER_URAM([0-9]*)/.DEPTH_TRIGGER_URAM(0)/' \
  -e 's/\.DEPTH_TRIGGER_BRAM([0-9]*)/.DEPTH_TRIGGER_BRAM(0)/' \
  -e 's/DEPTH_TRIGGER_URAM=[0-9]*,/DEPTH_TRIGGER_URAM=0,/' \
  -e 's/DEPTH_TRIGGER_BRAM=[0-9]*,/DEPTH_TRIGGER_BRAM=0,/'

echo "Patch done. Verifying no nonzero triggers remain:"
if grep -rlE "DEPTH_TRIGGER_URAM\([1-9]|DEPTH_TRIGGER_BRAM\([1-9]" renamed_src/; then
  echo "WARNING: some nonzero triggers still remain (see above)"
else
  echo "OK: all DEPTH_TRIGGER_URAM/BRAM values are now 0 (auto)"
fi
