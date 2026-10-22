"""STOCK FINN resource estimates: FINN v0.10.1's own `lut_estimation() / bram_estimation() / uram_estimation() / dsp_estimation()` of every node type, CALLED from the checked-out FINN source
(FINN/src, the build's version), not re-implemented. A node is a duck-typed subclass of the real FINN op class whose `get_nodeattr` and datatype getters read a plain dict, so no ONNX model, FINN
install or simulator is needed (the unavailable simulator packages are stubbed; they are never executed by an estimator).

Node types and what FINN itself says about them (this is the point of "stock"):
  MVAU_rtl        LUT 0 (lut_estimation returns 0);  DSP = ceil(PE / 4) * SIMD;  BRAM = FINN-R weight-memory formula (ram_style "auto" is estimated like "block")
  VVAU_hls        FINN-R MVU formula 300 + 1.1 * (...), DSP = PE * ceil((W + A) / 48), BRAM / URAM as in vectorvectoractivation.py
  ConvolutionInputGenerator_rtl   LUT = 300 + line-buffer LUTRAM (ram_style distributed), BRAM / URAM per ram_style; geometry = the PADDED feature map (FMPadding is a separate node in FINN's graph)
  Thresholding_rtl   LUT = LUTRAM count, BRAM, URAM from get_memory_estimate(); the bridge pins depth_trigger_bram and leaves depth_trigger_uram = 0, for which FINN's selection
                     (`elif depth >= depth_trigger_uram`) is true for every stage: the estimator then puts EVERY threshold stage in URAM (a known quirk of the stock estimator, reproduced here)
  StreamingDataWidthConverter   LUT per its shift-based formula
  StreamingFIFO   LUT / BRAM / URAM per impl_style (rtl up to 256 words, Vivado FIFO IP block RAM above, depths split as the real build's SplitLargeFIFOs leaves them)
  ChannelwiseOp_hls (the final bias add)   comparator + LUTRAM cost
  AddStreams, DuplicateStreams, StreamingConcat, UpsampleNearestNeighbour, LabelSelect, StreamingMaxPool, Pool, FMPadding   no estimator in v0.10.1: 0 (HWCustomOp default)
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import math  # noqa: F401
import sys
import types
import warnings
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[4]
FPGA_PART = "xczu7ev-ffvc1156-2-e"
MAX_RTL_FIFO_DEPTH = 256        # FINN max_qsrl_depth: deeper FIFOs become Vivado FIFO IP (block RAM)
FIFO_VIVADO_MAX_PIECE = 32768

_STUB_ROOTS = ("pyverilator", "pyxsi", "pyxsi_utils", "vcd")


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Simulator packages FINN imports at module level but an estimator never calls."""

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in _STUB_ROOTS:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = mock.MagicMock(name=spec.name)
        module.__path__, module.__spec__, module.__name__ = [], spec, spec.name
        return module

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _StubFinder())
sys.path.insert(0, str(REPO_ROOT / "FINN" / "src"))
warnings.filterwarnings("ignore", message=".*rounding-up FIFO depth.*")      # FINN's own note when a Vivado FIFO depth is rounded to a power of two

from qonnx.core.datatype import DataType  # noqa: E402
from finn.custom_op.fpgadataflow.hls.channelwise_op_hls import ChannelwiseOp_hls  # noqa: E402
from finn.custom_op.fpgadataflow.hls.vectorvectoractivation_hls import VVAU_hls  # noqa: E402
from finn.custom_op.fpgadataflow.rtl.convolutioninputgenerator_rtl import ConvolutionInputGenerator_rtl  # noqa: E402
from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl  # noqa: E402
from finn.custom_op.fpgadataflow.rtl.thresholding_rtl import Thresholding_rtl  # noqa: E402
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import StreamingDataWidthConverter  # noqa: E402
from finn.custom_op.fpgadataflow.rtl.streamingfifo_rtl import StreamingFIFO_rtl  # noqa: E402  (the class in FINN's graph; it defines get_adjusted_depth used by the base estimators)

ZERO = {"LUT": 0.0, "BRAM": 0.0, "URAM": 0.0, "DSP": 0.0}


def _node(cls, attrs: dict, **getters):
    """A duck-typed instance of the REAL FINN op class: nodeattrs come from `attrs`, `getters` (name -> value) replace the datatype / width getters."""
    ns = {"get_nodeattr": lambda self, name: self._attrs[name]}
    ns.update({name: (lambda value: (lambda self, *a, **k: value))(value) for name, value in getters.items()})
    sub = type(f"Stock_{cls.__name__}", (cls,), ns)
    obj = sub.__new__(sub)
    obj._attrs = attrs
    obj.onnx_node = types.SimpleNamespace(name="stock_node")      # only read by FINN's rounding warnings
    return obj


def _estimate(node) -> dict:
    return {"LUT": float(node.lut_estimation()), "BRAM": float(node.bram_estimation()), "URAM": float(node.uram_estimation()), "DSP": float(node.dsp_estimation(FPGA_PART))}


def _add(*parts: dict) -> dict:
    return {k: sum(p[k] for p in parts) for k in ZERO}


def mvau_rtl(pe: int, simd: int, w_bits: int, mw: int, mh: int) -> dict:
    node = _node(MVAU_rtl, {"PE": pe, "SIMD": simd, "MW": mw, "MH": mh, "mem_mode": "internal_decoupled", "ram_style": "auto"}, get_weight_datatype=DataType[f"INT{w_bits}"])
    return _estimate(node)


def vvau_hls(pe: int, simd: int, w_bits: int, a_bits: int, channels: int, kh: int, kw: int) -> dict:
    node = _node(VVAU_hls, {"PE": pe, "SIMD": simd, "Channels": channels, "Kernel": (kh, kw), "resType": "dsp", "mem_mode": "internal_decoupled", "ram_style": "auto", "noActivation": 1},
                 get_weight_datatype=DataType[f"INT{w_bits}"], get_input_datatype=DataType[f"UINT{a_bits}"], get_accumulator_datatype=DataType["INT32"])
    return _estimate(node)


def sliding_window_rtl(simd: int, a_bits: int, cin: int, kh: int, kw: int, hpad: int, wpad: int, sh: int, sw: int, dh: int, dw: int, parallel_window: bool, depthwise: bool, ram_style: str) -> dict:
    """ConvolutionInputGenerator_rtl on the PADDED feature map (hpad, wpad). A 1x1 kernel has no SWG node in FINN: zeros."""
    if kh == 1 and kw == 1:
        return dict(ZERO)
    node = _node(ConvolutionInputGenerator_rtl, {"SIMD": simd, "M": 1, "depthwise": int(depthwise), "IFMChannels": cin, "IFMDim": (hpad, wpad), "Stride": (sh, sw), "Dilation": (dh, dw),
                                                 "ConvKernelDim": (kh, kw), "parallel_window": int(parallel_window), "ram_style": ram_style, "dynamic_mode": 0},
                 get_input_datatype=DataType[f"UINT{a_bits}"])
    return _estimate(node)


def thresholding_rtl(pe: int, channels: int, in_bits: int, out_bits: int, ram_style: str) -> dict:
    trigger = 1024 if ram_style == "block" else 999_999       # THR_DEPTH_TRIGGER_BRAM / _DISTRIBUTED, what the bridge pins; depth_trigger_uram stays 0 (nodeattr default)
    node = _node(Thresholding_rtl, {"PE": pe, "NumChannels": channels, "depth_trigger_bram": trigger, "depth_trigger_uram": 0, "uniform_thres": 0},
                 get_weight_datatype=DataType[f"INT{in_bits}"], get_output_datatype=DataType[f"UINT{out_bits}"])
    return _estimate(node)


def dwc(in_width: int, out_width: int) -> dict:
    node = _node(StreamingDataWidthConverter, {"inWidth": in_width, "outWidth": out_width}, get_instream_width=in_width, get_outstream_width=out_width)
    return {"LUT": float(node.lut_estimation()), "BRAM": 0.0, "URAM": 0.0, "DSP": 0.0}


def _split_fifo(depth: int) -> list[int]:
    """The pieces FINN's SplitLargeFIFOs leaves for one requested depth (bottleneck.split_fifo_depth: the rule read off the real build)."""
    if depth <= MAX_RTL_FIFO_DEPTH:
        return [depth]
    out = []
    while depth > FIFO_VIVADO_MAX_PIECE:
        out.append(FIFO_VIVADO_MAX_PIECE)
        depth -= FIFO_VIVADO_MAX_PIECE
    if depth > MAX_RTL_FIFO_DEPTH:
        piece = 1 << (depth.bit_length() - 1)
        out.append(piece)
        depth -= piece
    if depth > 0:
        out.append(depth)
    return out


def streaming_fifo(width_bits: int, depth: int) -> dict:
    parts = []
    for piece in _split_fifo(depth):
        impl = "rtl" if piece <= MAX_RTL_FIFO_DEPTH else "vivado"
        node = _node(StreamingFIFO_rtl, {"depth": piece, "impl_style": impl, "ram_style": "block"}, get_instream_width=width_bits)
        parts.append(_estimate(node))
    return _add(*parts)


def channelwise_bias(pe: int, channels: int, in_bits: int) -> dict:
    node = _node(ChannelwiseOp_hls, {"PE": pe, "NumChannels": channels, "ram_style": "distributed"}, get_input_datatype=DataType[f"INT{in_bits}"])
    return _estimate(node)
