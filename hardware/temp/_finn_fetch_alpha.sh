cd /home/thelegendiv/finn
timeout 300 git fetch --no-tags --depth=1 origin refs/tags/v1.0.0-alpha 2>&1 | tail -3
git rev-parse FETCH_HEAD
echo --- rtllib mvu diff stat
git diff --stat HEAD FETCH_HEAD -- finn-rtllib/mvu | tail -30
echo --- py diff stat
git diff --stat HEAD FETCH_HEAD -- src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_rtl.py src/finn/custom_op/fpgadataflow/rtl/vectorvectoractivation_rtl.py | tail
echo --- upstream mvu files
git ls-tree -r --name-only FETCH_HEAD finn-rtllib/mvu
echo --- upstream python refs
git grep -n "mvu_\|dsp48\|dsp58" FETCH_HEAD -- src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_rtl.py | head -60
echo --- vivado version mentions
git grep -n "2024\|2023" FETCH_HEAD -- finn-rtllib/mvu | head -20
git grep -n "vivado_version\|VIVADO_VERSION\|2024" FETCH_HEAD -- docker/Dockerfile.finn run-docker.sh | head
