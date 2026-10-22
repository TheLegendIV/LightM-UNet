"""Convert the INT8 QONNX export of the uniform-INT8 QAT checkpoint (the network of the FPGA build s12_512_bilinear_int8_maxfps) to standard QuantizeLinear / DequantizeLinear (QCDQ) ONNX files that
ONNX Runtime (CPU, int8 kernels) and TensorRT (explicit quantization, INT8) can run, and check them against the QONNX reference.

Checkpoint: nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_256_perlayer_12_dense_relu_warmstart150ep_bilinear_uniform_int8_perlayer_ft15ep (checkpoint_best), exported by
hardware/builds/S12_dense_512_bilinear_int8_maxfps_v1/finn_export_S12_dense_512_bilinear_int8_maxfps_v1_ft15ep.py (UINT8 input contract: u = clip(round(z/s_in) + 128, 0, 255), the same input the board gets;
scale in deployment/img_intf/utils/input_profiles.json).

Steps: QONNX -> QCDQ (qonnx QuantToQCDQ); the one constant 32-bit bias Quant is folded; every BatchNormalization is folded into the preceding Conv / ConvTranspose WITHOUT changing the int8 weights
(y = a * (conv(s_w * w_int, x) + b0 - mean) + beta  ==  conv(DQ(w_int * sign(a), scale = s_w * |a| per output channel), x) + b'), so each block is a true INT8 conv + ReLU + activation quantizer;
an ArgMax over the class axis is appended (the FPGA chain ends in an argmax node and returns a class map). Two variants:
  *_ort.onnx  the conv bias as int32 (DequantizeLinear, scale = s_x * s_w per channel), the standard QDQ form ONNX Runtime fuses into QLinearConv (VNNI int8 kernels);
  *_trt.onnx  the conv bias as a float constant (TensorRT folds it; it does not take quantized biases).

Run (lightmunet_dev, repo at /workspace/LightM-UNet):  python analysis/benchmark/export_int8_qcdq.py
Output: analysis/benchmark/int8/s12_512_bilinear_int8_qcdq_{ort,trt}.onnx
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper as nh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.quant import quant
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.qonnx_to_qcdq import QuantToQCDQ

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "hardware/builds/S12_dense_512_bilinear_int8_maxfps_v1/outputs/quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_u8in.onnx"
REF = SRC.with_name(SRC.stem + "_verify_ref.npz")
OUT_DIR = Path(__file__).resolve().parent / "int8"


def qonnx_to_qcdq() -> onnx.ModelProto:
    m = ModelWrapper(str(SRC))
    m = m.transform(QuantToQCDQ())
    # QuantToQCDQ leaves the one 32-bit Quant (the final bias, a constant): fold it into a float initializer
    for n in list(m.graph.node):
        if n.op_type == "Quant" and m.get_initializer(n.input[0]) is not None:
            x, scale, zp, bw = (m.get_initializer(n.input[i]) for i in range(4))
            attrs = {a.name: a for a in n.attribute}
            q = quant(x, scale, zp, bw, signed=attrs["signed"].i, narrow=attrs["narrow"].i, rounding_mode=attrs["rounding_mode"].s.decode())
            m.set_initializer(n.output[0], np.asarray(q, dtype=np.float32))
            m.graph.node.remove(n)
    m = m.transform(InferShapes())
    return m.model


def fold_batchnorm(model: onnx.ModelProto, bias_mode: str, skip_deconv: bool = False) -> onnx.ModelProto:
    """bias_mode 'int32': conv bias as DequantizeLinear(int32, scale = s_x * s_w per channel); 'float': plain float bias."""
    g = model.graph
    init = {i.name: i for i in g.initializer}
    const = {n.output[0]: n for n in g.node if n.op_type == "Constant"}

    def val(name):
        if name in init:
            return nh.to_array(init[name])
        if name in const:
            return nh.to_array(const[name].attribute[0].t)
        return None

    prod = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    remove, new_before, new_inits, n_fold = set(), {}, [], 0
    for c in g.node:
        if c.op_type not in ("Conv", "ConvTranspose") or (skip_deconv and c.op_type == "ConvTranspose"):
            continue
        users = consumers.get(c.output[0], [])
        if len(users) != 1 or users[0].op_type != "BatchNormalization":
            continue
        bn = users[0]
        dq, = [prod[c.input[1]]]
        clip = prod[dq.input[0]]
        q = prod[clip.input[0]]
        assert dq.op_type == "DequantizeLinear" and clip.op_type == "Clip" and q.op_type == "QuantizeLinear", (c.name, dq.op_type, clip.op_type, q.op_type)
        w, s = val(q.input[0]), float(val(q.input[1]))
        w_int = np.clip(np.rint(w / s), -127, 127)
        gamma, beta, mean, var = (val(bn.input[i]).astype(np.float64) for i in (1, 2, 3, 4))
        eps = next((a.f for a in bn.attribute if a.name == "epsilon"), 1e-5)
        a = gamma / np.sqrt(var + eps)
        axis = 0 if c.op_type == "Conv" else 1                       # output-channel axis of the weight
        b0 = val(c.input[2]).astype(np.float64) if len(c.input) > 2 else np.zeros_like(a)
        bias = a * (b0 - mean) + beta
        sign = np.where(a < 0, -1.0, 1.0)
        shape = [1] * w_int.ndim
        shape[axis] = -1
        w_new = (w_int * sign.reshape(shape)).astype(np.int8)
        w_scale = (s * np.abs(a)).astype(np.float32)
        tag = c.name or c.output[0]
        inits = [nh.from_array(w_new, f"{tag}_w_int8"), nh.from_array(w_scale, f"{tag}_w_scale"), nh.from_array(np.zeros_like(a, dtype=np.int8), f"{tag}_w_zp")]
        nodes = [helper.make_node("DequantizeLinear", [f"{tag}_w_int8", f"{tag}_w_scale", f"{tag}_w_zp"], [f"{tag}_w_dq"], axis=axis, name=f"{tag}_w_dq_node")]
        if bias_mode == "int32":
            xdq = prod[c.input[0]]
            assert xdq.op_type == "DequantizeLinear", (c.name, xdq.op_type)
            s_x = float(val(xdq.input[1]))
            b_scale = (s_x * w_scale.astype(np.float64))
            b_int = np.clip(np.rint(bias / b_scale), -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
            inits += [nh.from_array(b_int, f"{tag}_b_int32"), nh.from_array(b_scale.astype(np.float32), f"{tag}_b_scale"), nh.from_array(np.zeros_like(b_int), f"{tag}_b_zp")]
            nodes.append(helper.make_node("DequantizeLinear", [f"{tag}_b_int32", f"{tag}_b_scale", f"{tag}_b_zp"], [f"{tag}_b_dq"], axis=0, name=f"{tag}_b_dq_node"))
            bias_name = f"{tag}_b_dq"
        else:
            inits.append(nh.from_array(bias.astype(np.float32), f"{tag}_bias"))
            bias_name = f"{tag}_bias"
        new_inits += inits
        new_before[id(c)] = nodes
        c.input[1] = f"{tag}_w_dq"
        if len(c.input) > 2:
            c.input[2] = bias_name
        else:
            c.input.append(bias_name)
        c.output[0] = bn.output[0]
        remove |= {id(dq), id(clip), id(q), id(bn)}
        n_fold += 1
    nodes_out = []
    for n in g.node:
        if id(n) in remove:
            continue
        nodes_out += new_before.get(id(n), [])
        nodes_out.append(n)
    del g.node[:]
    g.node.extend(nodes_out)
    g.initializer.extend(new_inits)
    print(f"[{bias_mode}] folded {n_fold} BatchNormalization nodes into per-channel int8 weight scales")
    return model


def bake_weight_quant(model: onnx.ModelProto) -> onnx.ModelProto:
    """Weights that still go constant -> QuantizeLinear -> Clip(-127, 127) -> DequantizeLinear (the convolutions without a BatchNorm to fold): bake the integers into an int8 initializer so the weight path is just
    int8 initializer -> DequantizeLinear. (TensorRT evaluates the first convolution to zeros with the Q -> Clip -> DQ chain on its weights.) The integers are unchanged, the Clip is a no-op on them."""
    g = model.graph
    init = {i.name: i for i in g.initializer}
    const = {n.output[0]: n for n in g.node if n.op_type == "Constant"}

    def get(name):
        return nh.to_array(const[name].attribute[0].t) if name in const else nh.to_array(init[name])

    prod = {o: n for n in g.node for o in n.output}
    remove, n_baked = set(), 0
    for d in g.node:
        if d.op_type != "DequantizeLinear" or d.input[0] not in prod or prod[d.input[0]].op_type != "Clip":
            continue
        clip = prod[d.input[0]]
        q = prod.get(clip.input[0])
        if q is None or q.op_type != "QuantizeLinear" or (q.input[0] not in const and q.input[0] not in init):
            continue
        w, s_w = get(q.input[0]), get(q.input[1])
        w_int = np.clip(np.rint(w / s_w), -127, 127).astype(np.int8)
        name = f"{d.output[0]}_w_int8"
        g.initializer.append(nh.from_array(w_int, name))
        d.input[0] = name
        remove |= {id(clip), id(q)}
        n_baked += 1
    keep = [n for n in g.node if id(n) not in remove]
    del g.node[:]
    g.node.extend(keep)
    print(f"baked {n_baked} constant weight quantizers (Q -> Clip -> DQ) into int8 initializers")
    return model


def uint8_to_int8(model: onnx.ModelProto) -> onnx.ModelProto:
    """TensorRT has no UINT8 Q/DQ ("TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point, will use INT8 instead"): the post-ReLU activation quantizers (unsigned 8-bit, levels
    0..255, zero point 0) would saturate at 127 * s. Re-express them as signed INT8 with zero point 0 and TWICE the scale (range 0..254 * s kept, resolution halved: 7 effective bits on those tensors)."""
    g = model.graph
    const = {n.output[0]: n for n in g.node if n.op_type == "Constant"}
    init = {i.name: i for i in g.initializer}

    def get(name):
        return nh.to_array(const[name].attribute[0].t) if name in const else nh.to_array(init[name])

    def put(name, arr):
        t = nh.from_array(arr, name)
        if name in const:
            const[name].attribute[0].t.CopyFrom(t)
        else:
            init[name].CopyFrom(t)

    done, n_sites = set(), 0
    for n in g.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear") and len(n.input) > 2:
            zp = get(n.input[2])
            if zp.dtype == np.uint8 and n.input[2] not in done:
                assert not zp.any(), "unsigned quantizer with a non-zero zero point"
                put(n.input[1], (2.0 * get(n.input[1])).astype(np.float32))
                put(n.input[2], np.zeros_like(zp, dtype=np.int8))
                done.add(n.input[2])
                n_sites += 1
    del g.value_info[:]           # the stale uint8 type annotations of the re-typed tensors
    print(f"uint8 -> int8 (scale x2) on {n_sites} activation quantizers")
    return model


def add_argmax(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    logits = g.output[0].name
    g.node.append(helper.make_node("ArgMax", [logits], ["pred"], axis=1, keepdims=0, name="argmax"))
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("pred", onnx.TensorProto.INT64, [1, 512, 512]))
    return model


def cleanup(model: onnx.ModelProto) -> onnx.ModelProto:
    del model.graph.quantization_annotation[:]      # QONNX metadata; also breaks onnx.utils.extract_model
    del model.graph.value_info[:]
    used = {i for n in model.graph.node for i in n.input} | {o.name for o in model.graph.output}
    keep = [n for n in model.graph.node if not (n.op_type == "Constant" and n.output[0] not in used)]
    del model.graph.node[:]
    model.graph.node.extend(keep)
    init_used = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(init_used)
    return model


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = np.load(REF)
    for mode, suffix in (("int32", "ort"), ("float", "trt"), ("float", "trt_nodeconvfold")):
        model = qonnx_to_qcdq()
        model = bake_weight_quant(fold_batchnorm(model, mode, skip_deconv=(suffix == "trt_nodeconvfold")))
        if suffix.startswith("trt"):
            model = uint8_to_int8(model)
        model = cleanup(add_argmax(model))
        onnx.checker.check_model(model)
        out = OUT_DIR / f"s12_512_bilinear_int8_qcdq_{suffix}.onnx"
        onnx.save(model, str(out))
        print(f"{out.name}: ops {sorted({n.op_type for n in model.graph.node})}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
        agree = []
        for k in range(12):
            pred = sess.run(None, {sess.get_inputs()[0].name: ref["u"][k].astype(np.float32)[None, None]})[0][0]
            agree.append(float((pred == ref["pred"][k]).mean()))
        print(f"  argmax agreement with the QONNX reference over 12 images: mean {np.mean(agree):.5f}, min {np.min(agree):.5f}")


if __name__ == "__main__":
    main()
