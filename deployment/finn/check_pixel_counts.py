"""
Standalone, no-hardware, no-network sanity check for the FINN accelerator's
input/output pixel and byte counts.

Uses the exact same io_shape_dict (driver.py) and packing routines
(finn.util.data_packing) that the real PYNQ driver uses on the board, so the
byte counts here are guaranteed to match what run.c must send/receive over
LwIP -- without needing a board, bitstream, or TCP connection.

Run from this directory: python check_pixel_counts.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qonnx.core.datatype import DataType
from finn.util.data_packing import finnpy_to_packed_bytearray, packed_bytearray_to_finnpy

# Mirrors driver.py's io_shape_dict exactly, without importing driver.py itself
# (driver.py pulls in driver_base.py -> pynq, which requires real PYNQ hardware).
io_shape_dict = {
    "idt": [DataType["UINT8"]],
    "odt": [DataType["INT24"]],
    "ishape_normal": [(1, 64, 64, 1)],
    "oshape_normal": [(1, 64, 64, 2)],
}


def describe(label, shape, dtype, n_bytes):
    n_elems = int(np.prod(shape))
    print(f"{label}: shape={shape} dtype={dtype} elements={n_elems} bytes={n_bytes}")


def main():
    idt = io_shape_dict["idt"][0]
    odt = io_shape_dict["odt"][0]
    ishape_normal = io_shape_dict["ishape_normal"][0]
    oshape_normal = io_shape_dict["oshape_normal"][0]

    # ---- Input: 64x64 UINT8 image, exactly what idma0 expects. ----
    rng = np.random.default_rng(0)
    input_img = rng.integers(0, 256, size=ishape_normal, dtype=np.uint8).astype(np.float32)

    ibuf_packed = finnpy_to_packed_bytearray(input_img, idt)
    n_input_bytes = ibuf_packed.size
    describe("INPUT  (normal, pre-pack)", ishape_normal, idt, "n/a")
    print(f"INPUT  (packed for idma0): bytes={n_input_bytes}")
    assert n_input_bytes == 4096, f"expected 4096 bytes, got {n_input_bytes}"

    # ---- Output: dummy odma0 payload, exactly what odma0 produces. ----
    output_vals = rng.integers(-100, 100, size=oshape_normal).astype(np.float32)
    obuf_packed = finnpy_to_packed_bytearray(output_vals, odt)
    n_output_bytes = obuf_packed.size
    describe("OUTPUT (normal, pre-pack)", oshape_normal, odt, "n/a")
    print(f"OUTPUT (packed from odma0): bytes={n_output_bytes}")
    assert n_output_bytes == 24576, f"expected 24576 bytes, got {n_output_bytes}"

    # ---- Round-trip the output packing to confirm pixel count on unpack. ----
    unpacked = packed_bytearray_to_finnpy(obuf_packed, odt, output_shape=oshape_normal)
    assert unpacked.shape == oshape_normal
    assert np.array_equal(unpacked, output_vals)

    print()
    print(f"Input : {int(np.prod(ishape_normal))} pixels -> {n_input_bytes} bytes "
          f"({idt.bitwidth()}-bit each)")
    print(f"Output: {int(np.prod(oshape_normal))} pixels -> {n_output_bytes} bytes "
          f"({odt.bitwidth()}-bit each)")
    print(f"Byte ratio out/in = {n_output_bytes / n_input_bytes:g}x "
          f"(pixel ratio = {np.prod(oshape_normal) / np.prod(ishape_normal):g}x, "
          f"bit-width ratio = {odt.bitwidth() / idt.bitwidth():g}x)")
    print("OK: all pixel/byte counts match run.c's ACCEL_INPUT_BYTES/ACCEL_OUTPUT_BYTES.")


if __name__ == "__main__":
    main()
