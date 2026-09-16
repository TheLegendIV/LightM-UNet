"""Export the DENSE (non-factored, separable_dilated=False) half of the S12
context-block probe pair at INT6 -- see finn_export_probe_s12_context_common.py
for the shared network/estimate/export logic, and
finn_export_probe_s12_context_separable_int6.py for its matched counterpart.

Usage:
    python3 finn_export_probe_s12_context_dense_int6.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finn_export_probe_s12_context_common import run, write_combined_summary  # noqa: E402

MODEL_NAME = "probe_s12_context_dense_int6"
BIT_WIDTH = 6

if __name__ == "__main__":
    dense_result = run(MODEL_NAME, separable_dilated=False, weight_bit_width=BIT_WIDTH, act_bit_width=BIT_WIDTH)

    import json
    sep_path = Path(__file__).resolve().parent / "outputs" / "finn_exports" / "_separable_result_int6.json"
    dense_path = Path(__file__).resolve().parent / "outputs" / "finn_exports" / "_dense_result_int6.json"
    dense_path.parent.mkdir(parents=True, exist_ok=True)
    dense_path.write_text(json.dumps(dense_result, indent=2))
    if sep_path.exists():
        separable_result = json.loads(sep_path.read_text())
        write_combined_summary(dense_result, separable_result, out_name="probe_s12_context_int6_analytical_estimate.json")
    else:
        print("\n(separable variant not yet run -- combined summary will be written once it is)")
