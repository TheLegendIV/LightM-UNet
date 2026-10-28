import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from qonnx.core.modelwrapper import ModelWrapper

BASE = "finn_deployment_outputs/stitched_ip_quantEnet_O8_native_20260729_123711/intermediate_models/"
OUT_DIR = "/tmp/graph_previews"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    ("step_enet_tidy.onnx", "1. step_enet_tidy (post ONNX import, tidy-up)"),
    ("step_enet_streamline.onnx", "2. step_enet_streamline (note the Transpose churn)"),
    ("step_enet_convert_to_hw.onnx", "3. step_enet_convert_to_hw (ops -> FINN HW ops)"),
    ("step_specialize_layers.onnx", "4. step_specialize_layers (HLS/RTL backend chosen)"),
    ("step_hw_ipgen.onnx", "5. step_hw_ipgen (final, post per-node synthesis - 325 nodes)"),
]

CATEGORY_COLORS = {
    "hw_rtl": "#2ca02c",
    "hw_hls": "#1f77b4",
    "transpose": "#d62728",
    "other": "#7f7f7f",
}


def categorize(node):
    if node.domain == "finn.custom_op.fpgadataflow.rtl":
        return "hw_rtl"
    if node.domain == "finn.custom_op.fpgadataflow.hls":
        return "hw_hls"
    if node.op_type == "Transpose":
        return "transpose"
    return "other"


def render(fname, title, out_path):
    model = ModelWrapper(BASE + fname)
    graph = model.graph

    tensor_producer = {}
    for i, n in enumerate(graph.node):
        for o in n.output:
            tensor_producer[o] = i

    G = nx.DiGraph()
    for i, n in enumerate(graph.node):
        G.add_node(i, cat=categorize(n))
    for i, n in enumerate(graph.node):
        for inp in n.input:
            if inp in tensor_producer:
                G.add_edge(tensor_producer[inp], i)

    # layered layout by longest-path depth (topological generations)
    depth = {}
    for i in nx.topological_sort(G):
        preds = list(G.predecessors(i))
        depth[i] = 0 if not preds else max(depth[p] for p in preds) + 1

    layer_counts = {}
    pos = {}
    for i in sorted(G.nodes, key=lambda x: depth[x]):
        d = depth[i]
        layer_counts[d] = layer_counts.get(d, 0) + 1
        pos[i] = (d, -layer_counts[d])

    colors = [CATEGORY_COLORS[G.nodes[i]["cat"]] for i in G.nodes]

    n_nodes = len(G.nodes)
    width = max(14, depth and (max(depth.values()) + 1) * 0.6 or 14)
    height = max(6, max(layer_counts.values()) * 0.35) if layer_counts else 6

    fig, ax = plt.subplots(figsize=(width, height))
    nx.draw(
        G, pos, ax=ax, node_color=colors, node_size=40, edge_color="#bbbbbb",
        width=0.5, arrows=False, with_labels=False,
    )
    handles = [mpatches.Patch(color=c, label=k) for k, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    ax.set_title(f"{title}\n{n_nodes} nodes", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path} ({n_nodes} nodes)")


for fname, title in MODELS:
    out_path = os.path.join(OUT_DIR, fname.replace(".onnx", ".png"))
    render(fname, title, out_path)
