"""Export the SEPARABLE ((K,1)+(1,K)-factored, separable_dilated=True) half
of the S12 context-block probe pair at INT8 -- see
finn_export_probe_s12_context_common.py for the shared network/estimate/export
logic, and finn_export_probe_s12_context_dense_int8.py for its matched
counterpart.

Usage:
    python3 finn_export_probe_s12_context_separable_int8.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finn_export_probe_s12_context_common import run, write_combined_summary  # noqa: E402

MODEL_NAME = "probe_s12_context_separable_int8"
BIT_WIDTH = 8

if __name__ == "__main__":
    separable_result = run(MODEL_NAME, separable_dilated=True, weight_bit_width=BIT_WIDTH, act_bit_width=BIT_WIDTH)

    import json
    dense_path = Path(__file__).resolve().parent / "outputs" / "finn_exports" / "_dense_result_int8.json"
    sep_path = Path(__file__).resolve().parent / "outputs" / "finn_exports" / "_separable_result_int8.json"
    sep_path.parent.mkdir(parents=True, exist_ok=True)
    sep_path.write_text(json.dumps(separable_result, indent=2))
    if dense_path.exists():
        dense_result = json.loads(dense_path.read_text())
        write_combined_summary(dense_result, separable_result, out_name="probe_s12_context_int8_analytical_estimate.json")
    else:
        print("\n(dense variant not yet run -- combined summary will be written once it is)")
