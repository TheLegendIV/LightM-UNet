"""Client of the test-split benchmark server (deployment/FINN_interface/lwip_collect_results, "IMGC" protocol).

Not for live streaming (use stream_images.py with lwip_stream for that). One session does, in this order:

  1. LOAD     upload every image into the board's DDR          (network, not timed as inference)
  2. LATENCY  every image alone through the empty pipeline; the PL latency monitor gives per-image ticks
  3. RUN      the whole split back-to-back, `repeats` times; the monitor gives, per pass, the PL ticks from the first
              input beat to the last output beat of the split (no network, no CPU in the timed part for dma_mode 0)
  4. FETCH    download all class maps                          (network)
  5. STOP

Ticks are PL-clock ticks (input_first, output_first, output_last) as 32-bit counters; subtract modulo 2**32.
Wire format: see the header of the server's src/run.c and img_intf/utils/batch_img_header.py.
"""
from __future__ import annotations

import os
import socket
import struct
import sys
import time
from typing import Callable, Sequence

import numpy as np

_DEPLOYMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _DEPLOYMENT_DIR not in sys.path:
    sys.path.insert(0, _DEPLOYMENT_DIR)
from img_intf.utils.batch_img_header import (  # noqa: E402
    build_batch_header, MAGIC_COLLECT, MARKER_FETCH, MARKER_LATENCY, MARKER_LOAD, MARKER_RUN, MARKER_STOP,
)

DMA_MODE_BATCH = 0  # one idma/odma start with numReps = n images
DMA_MODE_FRAME = 1  # one idma/odma start per image


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError(f"connection closed after {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def _ticks(sock: socket.socket, tag: bytes) -> np.ndarray:
    head = _recv_exact(sock, 7)
    if head[:3] != tag:
        raise ConnectionError(f"expected {tag!r} reply, got {head!r}")
    count = struct.unpack("!I", head[3:])[0]
    return np.frombuffer(_recv_exact(sock, 12 * count), dtype=">u4").astype(np.int64).reshape(count, 3)


def run_split_benchmark(ip: str, port: int, frames: Sequence[bytes], dma_mode: int = DMA_MODE_BATCH,
                        repeats: int = 5, width: int = 0, height: int = 0, connect_retries: int = 10,
                        log: Callable[[str], None] = print, calibrate: bool = False) -> dict:
    """Run the whole split on the board and return
    {"outputs": [bytes per image], "latency_ticks": (n, 3) int64, "run_ticks": (passes, 3) int64,
     "run_calls": [{"reps", "wall_s", "ticks"}, ...], "upload_s", "fetch_s"}.
    With `calibrate`, a short RUN of 2 passes precedes the main RUN of `repeats` passes (all passes count as
    samples); the PC times both RUN commands, so the slope (wall_main - wall_short) / (repeats - 2) is the true
    seconds per pass on the PC's clock, independent of the PL clock assumption (see implied_pl_clock_mhz).
    Raises ConnectionError on a server that is not lwip_collect_results."""
    n = len(frames)
    payload = len(frames[0])
    sock = None
    for attempt in range(connect_retries):  # the server refuses a new session while the previous one is closing
        try:
            sock = socket.create_connection((ip, port), timeout=120)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            header = build_batch_header(payload, width, height, magic=MAGIC_COLLECT)
            sock.sendall(struct.pack("!I", len(header)) + header)
            reply = _recv_exact(sock, 7)
            break
        except (ConnectionError, socket.timeout, OSError):
            if sock is not None:
                sock.close()
            sock = None
            if attempt == connect_retries - 1:
                raise
            time.sleep(0.5)
    try:
        if reply[:3] != b"ACK":
            raise ConnectionError(f"bad header ACK: {reply!r}")
        out_bytes = struct.unpack("!I", reply[3:])[0]

        t0 = time.perf_counter()
        for frame in frames:
            sock.sendall(struct.pack("!B", MARKER_LOAD) + frame)
        sock.sendall(struct.pack("!B", MARKER_LATENCY))  # in-order: the server finishes the loads first
        latency_ticks = _ticks(sock, b"LAT")
        upload_s = time.perf_counter() - t0
        log(f"[COLLECT] {n} images loaded, latency pass done ({upload_s:.1f} s incl. network)")

        run_calls = []
        plan = [2, repeats] if calibrate and repeats > 2 else [repeats]
        for reps in plan:
            t0 = time.perf_counter()
            sock.sendall(struct.pack("!BBB", MARKER_RUN, dma_mode, reps))
            ticks = _ticks(sock, b"RUN")
            wall = time.perf_counter() - t0
            if len(ticks) != reps:
                raise ConnectionError(f"RUN completed {len(ticks)} of {reps} passes")
            run_calls.append({"reps": reps, "wall_s": wall, "ticks": ticks})
        run_ticks = np.concatenate([c["ticks"] for c in run_calls])

        t0 = time.perf_counter()
        sock.sendall(struct.pack("!B", MARKER_FETCH))
        head = _recv_exact(sock, 7)
        if head[:3] != b"OUT" or struct.unpack("!I", head[3:])[0] != n:
            raise ConnectionError(f"bad FETCH reply {head!r}")
        blob = _recv_exact(sock, n * out_bytes)
        fetch_s = time.perf_counter() - t0
        outputs = [blob[i * out_bytes:(i + 1) * out_bytes] for i in range(n)]

        sock.sendall(struct.pack("!B", MARKER_STOP))
        _recv_exact(sock, 3)
    finally:
        sock.close()
    return {"outputs": outputs, "latency_ticks": latency_ticks, "run_ticks": run_ticks, "run_calls": run_calls,
            "upload_s": upload_s, "fetch_s": fetch_s}


def ticks_to_intervals(ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(input_first -> output_first, input_first -> output_last) in PL ticks; modulo 2**32 handles counter wrap."""
    return (ticks[:, 1] - ticks[:, 0]) % 2**32, (ticks[:, 2] - ticks[:, 0]) % 2**32


def implied_pl_clock_mhz(run_calls: Sequence[dict]):
    """PL tick rate implied by the PC's wall clock: ticks per pass (monitor) over seconds per pass (PC), where the
    seconds per pass come from the slope between two RUN commands of different length, so the fixed cost of a RUN
    (cache flush, command round trip) cancels. Returns (MHz, seconds_per_pass) or (None, None) without two RUNs."""
    if len(run_calls) < 2:
        return None, None
    lo, hi = min(run_calls, key=lambda c: c["reps"]), max(run_calls, key=lambda c: c["reps"])
    if hi["reps"] == lo["reps"]:
        return None, None
    sec_per_pass = (hi["wall_s"] - lo["wall_s"]) / (hi["reps"] - lo["reps"])
    ticks = np.concatenate([lo["ticks"], hi["ticks"]])
    _, full = ticks_to_intervals(ticks)
    if sec_per_pass <= 0 or full.min() <= 0:
        return None, None
    return float(full.mean()) / sec_per_pass / 1e6, sec_per_pass
