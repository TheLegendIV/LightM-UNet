#!/bin/bash
O=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs
T=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
echo "== deployment outputs (256, nearest/bilinear/mvu2)"; ls -d $O/*256* | sed "s#$O/##"
echo "== build tmp"; ls -d $T/*256* | sed "s#$T/##"
echo "== search p7 mvu2 refs"; grep -rIl "mvu2\|p7mvu2" /home/thelegendiv/finn/notebooks/enet/*.py 2>/dev/null | head
