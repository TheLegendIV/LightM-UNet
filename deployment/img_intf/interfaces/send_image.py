"""Send ONE image to the board and save/show the returned class map.

The file also holds BatchImageStreamer (v3 marker protocol client), which
deployment/collect_results.py reuses to stream whole datasets.

Usage (repo root):
  python deployment/img_intf/interfaces/send_image.py [image.png|image_u8.raw] [--ip 192.168.0.111] [--port 7]

A PNG is preprocessed to the board's UINT8 input (img_intf/utils/preprocess.py);
a .raw is sent as is. The class map is written to output/<--name>/raw/<stem>.raw
(default name s12_512_nn). Default image: first imagesTs case of Dataset509.
"""
import os
import sys
import glob
import socket
import struct
import time
from typing import List, Optional

# Keep in sync with the server-side contract
# (deployment/FINN_interface/lwip_test/src/run.c: NETWORK_INPUT_BYTES/NETWORK_OUTPUT_BYTES).
# The server accepts one full image in a single idma0/odma0 transfer
# (on-chip LabelSelect_hls argmax) and returns one class-id map, 1 byte/pixel
# (values 0..4), so input and output are the same size.
IMAGE_H = int(os.getenv("IMAGE_H", "512"))
IMAGE_W = int(os.getenv("IMAGE_W", "512"))
ACCEL_INPUT_BYTES = int(os.getenv("ACCEL_INPUT_BYTES", str(IMAGE_H * IMAGE_W)))    # 262144
ACCEL_OUTPUT_BYTES = int(os.getenv("ACCEL_OUTPUT_BYTES", str(IMAGE_H * IMAGE_W)))  # 262144 (1 byte/pixel class id)

"""
Per-image-marker image transfer protocol (raw payloads only, format-agnostic
per-file but geometry is fixed for the whole session):

1. Send ONE length-prefixed header (magic + fixed per-image payload size +
   width/height). Sent exactly once per connection/session. The 4-byte
   length prefix is kept on the wire for framing consistency but carries no
   meaning -- the header size is fixed.
2. Wait for the header ACK ("ACK" + expected_output_bytes u32 BE).
3. For each image (client owns the count -- there is no num_images field):
   send a 1-byte MARKER_IMAGE followed by its raw bytes, then read back the
   server's fixed per-image output size, then read the per-image ACK
   ("ACK", sent by the server after every image).
4. After all images have been sent, send a 1-byte MARKER_STOP and read the
   final stop ACK ("ACK", the same token reused for simplicity).

See deployment/FINN_interface/test/src/run.c for the matching server-side
state machine (WAIT_BATCH_HEADER -> WAIT_MARKER -> WAIT_PIXELS -> RUN_ACCEL
-> SEND_PIXELS -> WAIT_MARKER -> ...).
"""
# Imported as the img_intf package (deployment/ on sys.path) so a caller's own
# `utils` module is never shadowed.
_DEPLOYMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _DEPLOYMENT_DIR not in sys.path:
    sys.path.insert(0, _DEPLOYMENT_DIR)
from img_intf.utils.batch_img_header import build_batch_header, MARKER_IMAGE, MARKER_STOP  # noqa: E402


def _render_progress_bar(done: int, total: int, width: int = 40) -> str:
    if total <= 0:
        filled = 0
    else:
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(ratio * width)

    bar = "=" * filled + "-" * (width - filled)
    return f"0%  [{bar}]  100%"


def _print_progress(prefix: str, done: int, total: int, *, width: int = 40):
    line = f"{prefix} {_render_progress_bar(done, total, width=width)} ({done}/{total})"
    print("\r" + line + " " * 8, end="", flush=True)


def _finish_progress_line():
    print()


class BatchImageStreamer:
    def __init__(
        self,
        ip: str,
        port: int,
        input_dir: Optional[str],
        output_dir: str,
        pattern: str = "*.raw",
        payload_length: int = ACCEL_INPUT_BYTES,
        expected_output_bytes: int = ACCEL_OUTPUT_BYTES,
        verify_loopback: bool = False,
        paths: Optional[List[str]] = None,
    ):
        self.ip = ip
        self.port = port
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.pattern = pattern
        self.payload_length = payload_length
        self.expected_output_bytes = expected_output_bytes
        self.verify_loopback = verify_loopback
        self.paths = paths  # explicit file list; overrides input_dir/pattern

    def discover_images(self) -> List[str]:
        if self.paths is not None:
            if not self.paths:
                raise FileNotFoundError("empty image list")
            return list(self.paths)
        paths = sorted(glob.glob(os.path.join(self.input_dir, self.pattern)))
        if not paths:
            raise FileNotFoundError(f"No files matching {self.pattern!r} found in {self.input_dir}")
        return paths

    def validate_images(self, paths: List[str]):
        for path in paths:
            size = os.path.getsize(path)
            if size != self.payload_length:
                raise ValueError(
                    f"{path}: size {size} bytes does not match required payload_length "
                    f"{self.payload_length} bytes (accelerator input is fixed-size)"
                )

    def _send_all(self, sock: socket.socket, data: bytes):
        sock.sendall(data)

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed during recv")
            buf.extend(chunk)
        return bytes(buf)

    def send_batch_header(self, sock: socket.socket):
        header_bytes = build_batch_header(self.payload_length)
        print(f"[INFO] Sending header (payload_length={self.payload_length})...")
        self._send_all(sock, struct.pack("!I", len(header_bytes)) + header_bytes)
        print("[INFO] Header sent.")

    def wait_for_batch_ack(self, sock: socket.socket) -> int:
        """Read the header ACK ("ACK" + output_bytes u32 BE) and return
        the server's declared per-image output size."""
        reply = self._recv_exact(sock, 7)
        if reply[:3] != b"ACK":
            raise ConnectionError("Did not receive valid ACK from server.")
        return struct.unpack("!I", reply[3:])[0]

    def wait_for_ack(self, sock: socket.socket):
        """Read the reused 3-byte "ACK" token (per-image ack or stop ack)."""
        reply = self._recv_exact(sock, 3)
        if reply != b"ACK":
            raise ConnectionError("Did not receive valid ACK from server.")

    def send_marker(self, sock: socket.socket, marker: int):
        self._send_all(sock, struct.pack("!B", marker))

    def verify_loopback_result(self, sent: bytes, received: bytes, image_name: str) -> bool:
        # Server's ACCEL_LOOPBACK mode echoes the raw input straight back
        # (input/output are the same size now), so a direct byte compare is
        # sufficient -- no padding to crop anymore.
        if len(received) != len(sent):
            print(f"[WARN] {image_name}: received {len(received)} bytes, expected {len(sent)}")

        ok = received == sent
        if ok:
            print(f"[PASS] {image_name}: loopback verified ({len(sent)} bytes match).")
        else:
            mismatches = sum(1 for a, b in zip(sent, received) if a != b)
            print(f"[FAIL] {image_name}: loopback mismatch, {mismatches}/{len(sent)} bytes differ.")

        return ok

    def run(self):
        paths = self.discover_images()
        self.validate_images(paths)
        os.makedirs(self.output_dir, exist_ok=True)

        num_images = len(paths)
        print(f"[INFO] Found {num_images} image(s) matching {self.pattern!r} in {self.input_dir}")

        t_batch_start = time.perf_counter()
        all_ok = True

        with socket.create_connection((self.ip, self.port)) as sock:
            print(f"[INFO] Connected to {self.ip}:{self.port}")

            self.send_batch_header(sock)
            server_output_bytes = self.wait_for_batch_ack(sock)
            if server_output_bytes != self.expected_output_bytes:
                print(f"[WARN] Server output size {server_output_bytes} differs from "
                      f"expected {self.expected_output_bytes}; using server's value.")
                self.expected_output_bytes = server_output_bytes
            print("[INFO] Header ACKed by server; streaming images...")

            for i, path in enumerate(paths, start=1):
                with open(path, "rb") as f:
                    data = f.read()

                t0 = time.perf_counter()
                self.send_marker(sock, MARKER_IMAGE)
                self._send_all(sock, data)
                result = self._recv_exact(sock, self.expected_output_bytes)
                self.wait_for_ack(sock)
                elapsed_s = time.perf_counter() - t0

                image_name = os.path.basename(path)
                out_path = os.path.join(self.output_dir, image_name)
                with open(out_path, "wb+") as f:
                    f.write(result)

                _print_progress("[BATCH]", i, num_images)
                print(f"  {image_name}: round-trip {elapsed_s * 1000.0:.3f} ms")

                if self.verify_loopback:
                    ok = self.verify_loopback_result(data, result, image_name)
                    all_ok = all_ok and ok

            _finish_progress_line()

            # Client owns the image count; STOP marker + ack ends the session.
            self.send_marker(sock, MARKER_STOP)
            self.wait_for_ack(sock)
            print(f"[INFO] Sent STOP marker after {num_images} image(s).")

        total_s = time.perf_counter() - t_batch_start
        print(f"[INFO] Batch of {num_images} image(s) complete in {total_s * 1000.0:.3f} ms.")
        print("[INFO] Connection closed.")

        if self.verify_loopback:
            status = "PASS" if all_ok else "FAIL"
            print(f"[{status}] Batch loopback verification: {'all' if all_ok else 'not all'} images matched.")

        return all_ok


def main():
    import argparse
    import numpy as np
    from img_intf.utils.preprocess import png_to_u8

    repo_root = os.path.abspath(os.path.join(_DEPLOYMENT_DIR, '..'))
    default_png = os.path.join(repo_root, "data", "nnUNet_raw", "Dataset509_ARCADE_1x1_4c",
                               "imagesTs", "test_1_p0000_0000.png")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=default_png, help="PNG (preprocessed here) or headerless u8 .raw")
    ap.add_argument("--ip", default="192.168.0.111")
    ap.add_argument("--port", type=int, default=7)
    ap.add_argument("--name", default="s12_512_nn", help="output/<name>/raw destination")
    ap.add_argument("--input-profile", help="input scale profile in utils/input_profiles.json (default: --name)")
    a = ap.parse_args()

    out_dir = os.path.join(_DEPLOYMENT_DIR, "img_intf", "output", a.name, "raw")
    stem = os.path.splitext(os.path.basename(a.image))[0]
    if a.image.lower().endswith(".png"):
        u8 = png_to_u8(a.image, a.input_profile or a.name)
        if u8.size != ACCEL_INPUT_BYTES:
            raise ValueError(f"{a.image}: {u8.shape} does not match {IMAGE_H}x{IMAGE_W}")
        in_dir = os.path.join(_DEPLOYMENT_DIR, "img_intf", "input")
        os.makedirs(in_dir, exist_ok=True)
        raw_path = os.path.join(in_dir, stem + ".raw")
        u8.tofile(raw_path)
    else:
        raw_path = a.image

    BatchImageStreamer(a.ip, a.port, None, out_dir, paths=[raw_path]).run()
    out = np.fromfile(os.path.join(out_dir, os.path.basename(raw_path)), dtype=np.uint8)
    counts = {int(k): int(v) for k, v in zip(*np.unique(out, return_counts=True))}
    print(f"[INFO] {stem}: class pixel counts {counts}")


if __name__ == "__main__":
    main()
