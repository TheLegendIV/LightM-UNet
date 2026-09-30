from qonnx.core.modelwrapper import ModelWrapper

path = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_preamble_20260913_091715/intermediate_models/step_enet_convert_to_hw.onnx"
m = ModelWrapper(path)
targets = {"MVAU_71_param0", "MVAU_83_param0"}
for idx, n in enumerate(m.graph.node):
    if n.op_type in ("MatrixVectorActivation","MVAU","VVAU"):
        for inp in n.input:
            if inp in targets:
                print(f"node_idx={idx} name={n.name} op_type={n.op_type} weight_tensor={inp} inputs={list(n.input)} outputs={list(n.output)}")