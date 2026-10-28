"""Quick, targeted verification (NOT a rebuild) that _find_up_block_fmpad_chain now correctly resolves
BOTH FMPadPix (FMPadding_Pixel_hls) and FMPad_u (FMPadding_rtl) for up4's 2x2-lowered conv in partition 5's
prefifo_autosize graph, instead of the old single-hop _find_dense_swu_fmpad which only found+mislabeled the
nearer node. Run inside the FINN container (needs qonnx/finn on path)."""
import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from finn_s12_build_steps import _find_up_block_fmpad_chain, _find_dense_swu_fmpad

MODEL = ("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
         "S12_256_analytical_namefix_20261006_195156/intermediate_models/"
         "supported_op_partitions/partition_5_prefifo_autosize.onnx")

model = ModelWrapper(MODEL)
mvau_u = model.get_nodes_by_op_type("MVAU_rtl") + model.get_nodes_by_op_type("MVAU_hls")
target = None
for n in mvau_u:
    if n.name == "MVAU_rtl_11":
        target = n
        break
if target is None:
    print("Could not find MVAU_rtl_11 -- listing MVAU-ish nodes:")
    for n in mvau_u:
        print(" ", n.name, n.op_type)
    sys.exit(1)

print(f"Target MVAU node: {target.name} ({target.op_type})")

old_fmpad, old_swu = _find_dense_swu_fmpad(model, target)
print(f"OLD single-hop  -> fmpad={getattr(old_fmpad, 'name', None)}/{getattr(old_fmpad, 'op_type', None)}  "
      f"swu={getattr(old_swu, 'name', None)}/{getattr(old_swu, 'op_type', None)}")

fmpadpix, fmpad_u, swu = _find_up_block_fmpad_chain(model, target)
print(f"NEW two-hop     -> fmpadpix={getattr(fmpadpix, 'name', None)}/{getattr(fmpadpix, 'op_type', None)}  "
      f"fmpad_u={getattr(fmpad_u, 'name', None)}/{getattr(fmpad_u, 'op_type', None)}  "
      f"swu={getattr(swu, 'name', None)}/{getattr(swu, 'op_type', None)}")

expected_fmpadpix = "FMPadding_Pixel_hls_0"
expected_fmpad_u = "FMPadding_rtl_3"
ok = (getattr(fmpadpix, "name", None) == expected_fmpadpix and
      getattr(fmpad_u, "name", None) == expected_fmpad_u)
print(f"RESULT: {'PASS' if ok else 'FAIL'} (expected fmpadpix={expected_fmpadpix}, fmpad_u={expected_fmpad_u})")
