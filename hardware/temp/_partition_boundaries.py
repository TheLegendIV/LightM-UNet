import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

D = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions/"
for k in range(8):
    m = ModelWrapper(D + f"partition_{k}.onnx")
    i, o = m.graph.input[0].name, m.graph.output[0].name
    fn = m.find_consumer(i); ln = m.find_producer(o)
    fo, fi = None, None
    try:
        fi = getCustomOp(fn).get_folded_input_shape(); fo = getCustomOp(ln).get_folded_output_shape()
    except Exception as e:
        print("  folded err", e)
    print(f"p{k}: in={i} {m.get_tensor_shape(i)} {m.get_tensor_datatype(i)} firstnode={fn.op_type} folded_in={fi} | "
          f"out={o} {m.get_tensor_shape(o)} {m.get_tensor_datatype(o)} lastnode={ln.op_type} folded_out={fo}")
