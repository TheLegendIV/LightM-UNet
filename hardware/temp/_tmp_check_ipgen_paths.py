import os
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_144349"
part_dir = os.path.join(OUTPUT_DIR, "intermediate_models", "supported_op_partitions")

for i in range(8):
    p = os.path.join(part_dir, f"partition_{i}.onnx")
    if not os.path.exists(p):
        print(f"partition_{i}: MISSING onnx at {p}")
        continue
    m = ModelWrapper(p)
    hw_nodes = [n for n in m.graph.node if n.domain.startswith("finn")]
    n_missing_ipgen = 0
    n_missing_codegen = 0
    n_total = 0
    sample = []
    for n in hw_nodes:
        inst = getCustomOp(n)
        n_total += 1
        for attr_name in ("ipgen_path", "code_gen_dir_ipgen", "code_gen_dir_cppsim"):
            try:
                val = inst.get_nodeattr(attr_name)
            except Exception:
                continue
            if val:
                exists = os.path.exists(val)
                if len(sample) < 3:
                    sample.append((n.name, attr_name, val, exists))
                if attr_name == "ipgen_path" and not exists:
                    n_missing_ipgen += 1
                if attr_name == "code_gen_dir_ipgen" and not exists:
                    n_missing_codegen += 1
    print(f"partition_{i}: n_hw_nodes={n_total} sample={sample}")
