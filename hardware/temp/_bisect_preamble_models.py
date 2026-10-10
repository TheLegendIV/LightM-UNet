"""Per-step bisection: execute every intermediate ONNX of a preamble on the verify-reference inputs and report agreement with PyTorch."""
import warnings
warnings.filterwarnings("ignore")
import glob, os, sys, time
import numpy as np
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes
import verify_export as ve

PRE = sys.argv[1]
REF = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2

ref = np.load(REF)
u, pred, gt = ref["u"][:N], ref["pred"][:N], ref["gt"][:N]
files = sorted(glob.glob(os.path.join(PRE, "intermediate_models", "*.onnx")), key=os.path.getmtime)
print(f"{len(files)} models, {N} cases; torch fg-dice {ve.dice_report(pred, gt)['fg_dice']:.4f}", flush=True)
for f in files:
    name = os.path.basename(f)
    t0 = time.time()
    try:
        model = ModelWrapper(f).transform(InferShapes())
        for node in model.graph.node:
            for i, t in enumerate(node.input):
                if t == "":
                    node.input[i] = f"{node.name}_empty{i}"
                    model.set_initializer(node.input[i], np.zeros((0,), dtype=np.float32))
        model.check_all_tensor_shapes_specified(fix_missing_init_shape=True)
        in_name, out_name = model.graph.input[0].name, model.graph.output[0].name
        in_shape = model.get_tensor_shape(in_name)
        if model.get_tensor_datatype(out_name) != DataType["FLOAT32"] and not model.get_nodes_by_op_type("LabelSelect"):
            model.set_tensor_datatype(out_name, DataType["FLOAT32"])
        got = np.stack([ve._to_class_map(execute_onnx(model, {in_name: u[i].astype(np.float32).reshape(in_shape)})[out_name], ve.NUM_CLASSES)
                        for i in range(len(u))])
        rep = ve.dice_report(got, gt)
        print(f"{name:62s} agree {float((got == pred).mean()):.6f} fg_dice {rep['fg_dice']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"{name:62s} ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
