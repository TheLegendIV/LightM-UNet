#!/bin/bash
python3 -c "
import finn.transformation.fpgadataflow.set_fifo_depths as m
print('FILE:', m.__file__)
"
FILE=$(python3 -c "import finn.transformation.fpgadataflow.set_fifo_depths as m; print(m.__file__)")
grep -n "rtlsim_batch_size\|verilator_fifosim\|def apply\|batch" "$FILE" | head -60
