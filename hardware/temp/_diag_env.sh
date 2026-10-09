#!/bin/bash
export HOME=/tmp/home_dir
cd /home/thelegendiv/finn/notebooks/enet
python3 -c "
from google.protobuf.internal import api_implementation as a; print('protobuf', a.Type())
import time; t=time.time(); s=0
for i in range(10**7): s+=i
print('loop10M %.2fs'%(time.time()-t))
"
env | grep -i -E 'protobuf|PYTHON' 
cat /proc/loadavg
nproc
cd /home/thelegendiv/finn/deps/qonnx && git status --short | head; git log -1 --format=%h' '%cd
pip list 2>/dev/null | grep -i -E '^(onnx|protobuf|numpy) '
