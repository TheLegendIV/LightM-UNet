"""Regenerate the 'original' (true ENet-paper-scale) INT8 ONNX export and its
FINN fully-unfolded resource-estimate report, after the channel-width bug fix
(2026-08-17): finn_export_original_enet_int8.py previously used the E1
sweep-grid widths (20,72,144,72,20) instead of the real baseline
(16,64,128,64,16) -- see that file's module docstring for the full story.

Chains the two already-existing pipeline scripts in one shot:
  1. finn_export_original_enet_int8.py  -- overwrites quantEnet_original_int8.onnx
     in place with the corrected-width model (also INT8).
  2. finn_estimate_original_enet_unfolded.py -- re-runs FINN's fully-unfolded
     (PE=SIMD=1) analytical estimate against the freshly exported ONNX and
     writes a new report/per_node_breakdown.csv under
     finn_deployment_outputs/estimates_unfolded_quantEnet_original_int8_<timestamp>/.

Run this from inside the FINN container, in the same directory as
finn_enet_build.py (i.e. /home/thelegendiv/finn/notebooks/enet/ per
FINN_REPO_INDEX.md), with finn_enet_prod_export.py,
finn_export_original_enet_int8.py, and finn_estimate_original_enet_unfolded.py
all copied alongside it first:

    docker cp hardware/finn_enet_prod_export.py <container>:/home/thelegendiv/finn/notebooks/enet/
    docker cp hardware/finn_export_original_enet_int8.py <container>:/home/thelegendiv/finn/notebooks/enet/
    docker cp hardware/finn_estimate_original_enet_unfolded.py <container>:/home/thelegendiv/finn/notebooks/enet/
    docker cp hardware/regenerate_original_enet_int8.py <container>:/home/thelegendiv/finn/notebooks/enet/
    docker exec -e HOME=/tmp/home_dir <container> python3 /home/thelegendiv/finn/notebooks/enet/regenerate_original_enet_int8.py

After it finishes, copy the results back out of the container for the repo:
    docker cp <container>:/home/thelegendiv/finn/notebooks/enet/quantEnet_original_int8.onnx hardware/outputs/finn_exports/
    docker cp <container>:/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/estimates_unfolded_quantEnet_original_int8_<timestamp>/ hardware/outputs/quantEnet_original_int8_unfolded_report_NEW/
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script_name: str) -> None:
    script_path = HERE / script_name
    print(f"\n{'=' * 70}\nRunning {script_name}\n{'=' * 70}")
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(HERE))


if __name__ == "__main__":
    run("finn_export_original_enet_int8.py")
    run("finn_estimate_original_enet_unfolded.py")
    print("\nDone. Both steps completed with the corrected channels=(16,64,128,64,16) baseline.")
