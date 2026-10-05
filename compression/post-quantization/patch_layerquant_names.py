"""Run-time workaround for LayerQuantENet.py / LayerQuantEnetFINN.py in the current working tree: the 'Sync' commit 66e0590e8b dropped the
`from QuantENet / ENet / CombinedQuantENet import (...)` block, VALID_CONTEXT_PATTERNS and ACT_SITE_TYPES from LayerQuantENet.py (they are intact at b16b35468a), so
LayerQuantEnetFINN cannot be constructed in this checkout. Importing this module injects the missing names in-process (a no-op once the files are restored), without
editing either file. Use it in front of any calibrate_* / export script run from this checkout:

    python3 - <<'EOF'
    import runpy, sys
    sys.path.insert(0, "compression/post-quantization")
    import patch_layerquant_names            # noqa: F401
    sys.argv = ["calibrate_12_dense_relu_nearest_conv_upsample_256_perlayer.py", "--model-class", "finn", ...]
    runpy.run_path("compression/post-quantization/calibrate_12_dense_relu_nearest_conv_upsample_256_perlayer.py", run_name="__main__")
    EOF
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "enet"))
import nnunetv2.nets.CombinedQuantENet as _cq  # noqa: E402
import nnunetv2.nets.ENet as _en  # noqa: E402
import nnunetv2.nets.LayerQuantENet as _lq  # noqa: E402
import nnunetv2.nets.LayerQuantEnetFINN as _fin  # noqa: E402
import nnunetv2.nets.QuantENet as _qe  # noqa: E402


def patch_missing_names() -> None:
    """LayerQuantENet.py in the working tree lost its `from QuantENet / ENet / CombinedQuantENet import (...)` block, VALID_CONTEXT_PATTERNS and ACT_SITE_TYPES in
    commit 66e0590e8b (intact at b16b35468a), and LayerQuantEnetFINN.py uses the same names through it. Inject what is missing (no-op when the files are intact)
    instead of editing them."""
    import brevitas.nn as qnn
    for mod in (_qe, _cq, _en):
        for name in dir(mod):
            if not name.startswith("__") and not hasattr(_lq, name):
                setattr(_lq, name, getattr(mod, name))
    # CombinedQuantENet's tuple predates dense_dilation_half (the S12 256 context pattern), which _make_layer_context_stage handles: extend it.
    _lq.VALID_CONTEXT_PATTERNS = tuple(dict.fromkeys((*getattr(_lq, "VALID_CONTEXT_PATTERNS", ()), "dense_dilation_half")))
    if not hasattr(_lq, "ACT_SITE_TYPES"):
        _lq.ACT_SITE_TYPES = (qnn.QuantReLU, qnn.QuantIdentity, _qe.QuantDecomposedLeakyAct, _qe.QuantFusedLeakyAct, qnn.QuantEltwiseAdd)
    for mod in (_lq, _qe, _cq, _en):
        for name in dir(mod):
            if not name.startswith("__") and not hasattr(_fin, name):
                setattr(_fin, name, getattr(mod, name))


patch_missing_names()
