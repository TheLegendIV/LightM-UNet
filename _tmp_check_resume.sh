#!/bin/bash
echo "=== finn_build_tmp entries modified since Sep 22 09:00 (Tree B run window) ==="
find ~/finn/notebooks/enet/finn_build_tmp -maxdepth 1 -newermt "2026-09-22 09:00:00" | wc -l
find ~/finn/notebooks/enet/finn_build_tmp -maxdepth 1 -newermt "2026-09-22 09:00:00" | head -20
echo
echo "=== OUTPUT_DIR for the last (nouram_512x512) run - find intermediate_models & any partition checkpoints ==="
find ~/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_512x512_20260922_102900 -maxdepth 3 2>/dev/null
echo
echo "=== also check the earlier 011241 attempt dir (Tree A) for comparison ==="
find ~/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_512x512_20260922_011241 -maxdepth 3 2>/dev/null
echo
echo "=== any *.onnx checkpoint files anywhere under these two output dirs ==="
find ~/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_512x512_* -iname "*.onnx" 2>/dev/null
