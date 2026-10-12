"""Generate synthetic 256x256 uint8 grayscale test images for the batch
ACCEL_LOOPBACK smoke test (see async_read_write_single.py's __main__).

Raw, headerless output (exactly 65536 bytes = ACCEL_INPUT_BYTES per image) --
a radial gradient with a bright ring, loosely mimicking a vessel
cross-section so it's visually distinguishable from flat/random noise when
inspected. Multiple phase-shifted variants are written so the batch client
has more than one *.raw file to stream per run.
"""
import math
import os

SIZE = 256
NUM_IMAGES = 4
# Directory matches async_read_write_single.py's input_dir (".\input\", relative
# to deployment/img_intf/ when the client script is run).
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "input")


def make_image(phase: float = 0.0) -> bytearray:
    cx = cy = (SIZE - 1) / 2.0
    r_max = math.sqrt(cx ** 2 + cy ** 2)
    data = bytearray(SIZE * SIZE)

    # Dark background, bright ring around mid-radius, dim core -- deterministic.
    for y in range(SIZE):
        for x in range(SIZE):
            r_norm = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / r_max
            val = 40 + 180 * math.exp(-((r_norm - 0.55) ** 2) / (2 * 0.08 ** 2))
            val += 15 * math.sin(x / 3.0 + phase) * math.cos(y / 5.0 + phase)
            data[y * SIZE + x] = max(0, min(255, int(round(val))))
    return data


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for i in range(NUM_IMAGES):
        phase = i * (math.pi / NUM_IMAGES)
        data = make_image(phase)
        assert len(data) == SIZE * SIZE == 65536
        out_path = os.path.join(OUT_DIR, f"medical_slice_{SIZE}x{SIZE}_{i:02d}.raw")
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Wrote {len(data)} bytes to {out_path}")

