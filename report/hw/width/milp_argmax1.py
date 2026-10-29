"""Run ILP/finn_milp.py with LabelSelect (final argmax) priced at 1 cycle per fold (the assumption of the S12_dense_512_uniform_cap1_v2 artifacts) instead of the measured 2 (2026-10-16).
HYPOTHETICAL: only used to see each width's own compute-limited max FPS once the argmax wall (524,288 cycles) is out of the way. Everything else is the current model.
Usage: python3 report/hw/width/milp_argmax1.py <finn_milp.py args>"""
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "ILP"))
import finn_cost_model as fcm  # noqa: E402

fcm._LABELSELECT_CYCLES_PER_FOLD = 1
sys.argv = [str(REPO / "ILP/finn_milp.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
