#!/bin/bash
for f in \
  /tmp/finn_dev_thelegendiv/code_gen_ipgen_DuplicateStreams_hls_0_9imqhfgr/project_DuplicateStreams_hls_0/sol1/impl/ip/component.xml \
  /tmp/finn_dev_thelegendiv/code_gen_ipgen_AddStreams_hls_0_7rr8i5jm/project_AddStreams_hls_0/sol1/impl/ip/component.xml \
  /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_hls_0_h6s8cvt4/project_MVAU_hls_0/sol1/impl/ip/component.xml \
  /tmp/finn_dev_thelegendiv/code_gen_ipgen_DuplicateStreams_hls_0_2pc4w53x/project_DuplicateStreams_hls_0/sol1/impl/ip/component.xml \
  /tmp/finn_dev_thelegendiv/code_gen_ipgen_AddStreams_hls_0_j8s__otq/project_AddStreams_hls_0/sol1/impl/ip/component.xml \
  /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_hls_0_bez4stje/project_MVAU_hls_0/sol1/impl/ip/component.xml ; do
  echo "== $f =="
  if [ -f "$f" ]; then
    grep -m1 'spirit:library\|<spirit:name>' "$f" | head -3
    grep -A1 '<spirit:library>hls</spirit:library>' "$f" | head -2
  else
    echo "  MISSING"
  fi
done
