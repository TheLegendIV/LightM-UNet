#!/bin/bash
cd ~/finn/notebooks/enet
echo "=== finn_enet_ip_build_partitioned_8way.py imports ==="
grep -E '^(from|import) [a-zA-Z_]' finn_enet_ip_build_partitioned_8way.py | grep -v -E 'numpy|torch|onnx|finn\.|qonnx|brevitas|^import (os|sys|json|re|time|shutil|copy|glob|subprocess|argparse|logging|dataclasses|pickle|math|traceback|warnings|random|itertools|functools|collections|typing|datetime|concurrent)'
echo
echo "=== finn_stage_partition.py imports ==="
grep -E '^(from|import) [a-zA-Z_]' finn_stage_partition.py | grep -v -E 'numpy|torch|onnx|finn\.|qonnx|brevitas|^import (os|sys|json|re|time|shutil|copy|glob|subprocess|argparse|logging|dataclasses|pickle|math|traceback|warnings|random|itertools|functools|collections|typing|datetime|concurrent)'
echo
echo "=== finn_enet_convert_to_hw_rtl_mvau.py imports ==="
grep -E '^(from|import) [a-zA-Z_]' finn_enet_convert_to_hw_rtl_mvau.py | grep -v -E 'numpy|torch|onnx|finn\.|qonnx|brevitas|^import (os|sys|json|re|time|shutil|copy|glob|subprocess|argparse|logging|dataclasses|pickle|math|traceback|warnings|random|itertools|functools|collections|typing|datetime|concurrent)'
echo
echo "=== who imports finn_partition_build_steps_rtl_mvau ==="
grep -l "finn_partition_build_steps_rtl_mvau" *.py 2>/dev/null | grep -v "^finn_partition_build_steps_rtl_mvau.py$"
echo
echo "=== who imports finn_cost_model / finn_native_cost_estimator ==="
grep -l "finn_cost_model\|finn_native_cost_estimator" *.py 2>/dev/null
echo
echo "=== who imports analyze_fifo_topology ==="
grep -l "analyze_fifo_topology" *.py 2>/dev/null
echo
echo "=== who imports dump_node_attrs ==="
grep -l "dump_node_attrs" *.py 2>/dev/null
echo
echo "=== who imports append_result_row ==="
grep -l "append_result_row" *.py 2>/dev/null
echo
echo "=== who imports regen_stitched_ip_8way_full_v3 or is imported ==="
grep -l "regen_stitched_ip_8way_full_v3" *.py 2>/dev/null
echo
echo "=== who imports finn_ooc_...v3 (non-nouram) besides zynqbuild ==="
grep -l "finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3\b" *.py 2>/dev/null
