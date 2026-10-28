"""Re-decode golden_per_partition dumps with explicit byte order: RTL dump is little-endian (memcpy), the software stream is FINN's default big-endian."""
import sys
import numpy as np

root = sys.argv[1]
# (partition, bytes per element, signed, bits) of each partition's output boundary
specs = {1: (2, 9), 2: (1, 7), 3: (2, 9), 4: (2, 9), 5: (2, 9), 6: (2, 9)}


def signed(v, bits):
    v = v.astype(np.int64)
    return np.where(v >= (1 << (bits - 1)), v - (1 << bits), v)


for k, (nb, bits) in specs.items():
    rtl = np.fromfile(f"{root}/p{k}_rtl.raw", np.uint8).reshape(-1, nb).astype(np.int64)
    sw = np.fromfile(f"{root}/p{k}_sw.raw", np.uint8).reshape(-1, nb).astype(np.int64)
    if nb == 1:
        r, s = rtl[:, 0], sw[:, 0]
    else:
        r = rtl[:, 0] | (rtl[:, 1] << 8)      # little-endian (testbench memcpy)
        s = (sw[:, 0] << 8) | sw[:, 1]        # big-endian (finnpy_to_packed_bytearray default)
    r, s = signed(r, bits), signed(s, bits)
    print(f"p{k}: n={r.size} exact={np.mean(r == s):.6f} max|d|={int(np.abs(r - s).max())} sw[min,max]=({s.min()},{s.max()}) rtl[min,max]=({r.min()},{r.max()})")
