"""Client of the test-split benchmark server (deployment/FINN_interface/lwip_collect_results, "IMGC" protocol).

Not for live streaming (use stream_images.py with lwip_stream for that). One session does, in this order:

  0. POWER    (only with a PowerPlan) idle window: the board's INA226 rails sampled with the accelerator idle
  1. LOAD     upload every image into the board's DDR          (network, not timed as inference)
  2. LATENCY  every image alone through the empty pipeline; the PL latency monitor gives per-image ticks
  3. RUN      the whole split back-to-back, `repeats` times; the monitor gives, per pass, the PL ticks from the first
              input beat to the last output beat of the split (no network, no CPU in the timed part for dma_mode 0)
  3b. POWER   (only with a PowerPlan) inference window: the split runs back-to-back while the rails are sampled, then
              an optional second idle window
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
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

_DEPLOYMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _DEPLOYMENT_DIR not in sys.path:
    sys.path.insert(0, _DEPLOYMENT_DIR)
from img_intf.utils.batch_img_header import (  # noqa: E402
    build_batch_header, MAGIC_COLLECT, MARKER_FETCH, MARKER_LATENCY, MARKER_LOAD, MARKER_POWER, MARKER_RUN, MARKER_STOP,
)

DMA_MODE_BATCH = 0  # one idma/odma start with numReps = n images
DMA_MODE_FRAME = 1  # one idma/odma start per image


class HandshakeError(ConnectionError):
    """The server did not accept the IMGC header: another app is running on the board (e.g. lwip_stream) or a previous
    session is still active. Raised before any command was sent."""
POWER_MODE_IDLE = 0  # POWER: accelerator idle
POWER_MODE_LOAD = 1  # POWER: the loaded split runs back-to-back for the whole warm-up + window
POWER_MAX_TOTAL_S = 100  # the server clamps warm-up + window to this (its main loop is blocked meanwhile)
POWER_CHUNK_S = 5  # longest single POWER command: the board's main loop (and so its TCP) is blocked for that long


@dataclass(frozen=True)
class PowerPlan:
    """Board power measurement inside the benchmark session: idle window, inference window (after a warm-up that is
    run but not measured, so the board is at operating temperature), optional idle window after the load. Every window
    is sent as POWER commands of at most `chunk_s` seconds (see power_window)."""
    idle_s: int = 10
    warmup_s: int = 5
    load_s: int = 20
    idle_after: bool = True
    chunk_s: int = POWER_CHUNK_S

    def __post_init__(self):
        if not (0 < self.idle_s <= POWER_MAX_TOTAL_S and 0 < self.load_s and self.warmup_s >= 0
                and self.warmup_s + self.load_s <= POWER_MAX_TOTAL_S and 0 < self.chunk_s <= POWER_MAX_TOTAL_S):
            raise ValueError(f"power plan out of range: idle_s 1..{POWER_MAX_TOTAL_S}, warmup_s + load_s <= {POWER_MAX_TOTAL_S}: {self}")


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


def _s64(hi: int, lo: int) -> int:
    v = (int(hi) << 32) | int(lo)
    return v - (1 << 64) if v >= 1 << 63 else v


def power_command(sock: socket.socket, mode: int, warmup_s: int, window_s: int) -> dict:
    """POWER: sample the board's INA226 rails for `window_s` seconds after `warmup_s` seconds of warm-up. Mode 0 = idle;
    mode 1 = the loaded split runs back-to-back during warm-up and window. Returns {"mode", "passes", "images",
    "ticks_sum" (PL ticks, first input beat -> last output beat summed over the counted passes), "sweeps", "window_s"
    (the board's own clock), "wall_s" (this host's clock for the whole command), "devices": [{"channel", "addr", "n",
    "sum_vbus", "sum_vshunt", "sum_prod", "errors"}, ...]} (raw INA226 sums, see img_intf/utils/zcu106_power.py)."""
    old_timeout = sock.gettimeout()
    sock.settimeout(max(old_timeout or 0, warmup_s + window_s + 60))  # the server's main loop is blocked meanwhile
    t0 = time.perf_counter()
    try:
        sock.sendall(struct.pack("!BBBB", MARKER_POWER, mode, warmup_s, window_s))
        head = _recv_exact(sock, 7)
        if head[:3] != b"PWR":
            raise ConnectionError(f"expected b'PWR' reply, got {head!r}")
        ndev = struct.unpack("!I", head[3:])[0]
        words = np.frombuffer(_recv_exact(sock, 4 * (7 + 9 * ndev)), dtype=">u4").astype(np.uint64)
    finally:
        sock.settimeout(old_timeout)
    wall_s = time.perf_counter() - t0
    hdr = [int(w) for w in words[:7]]
    devices = []
    for i in range(ndev):
        w = [int(x) for x in words[7 + 9 * i: 16 + 9 * i]]
        devices.append({"channel": w[0] >> 8, "addr": w[0] & 0xFF, "n": w[1], "sum_vbus": (w[2] << 32) | w[3],
                        "sum_vshunt": _s64(w[4], w[5]), "sum_prod": _s64(w[6], w[7]), "errors": w[8]})
    return {"mode": hdr[0], "passes": hdr[1], "images": hdr[2], "ticks_sum": (hdr[3] << 32) | hdr[4], "sweeps": hdr[5],
            "window_s": hdr[6] / 1000, "wall_s": wall_s, "devices": devices}


def merge_power(parts: Sequence[dict]) -> dict:
    """Add up the raw sums of several POWER replies of the same mode (same devices in the same order)."""
    out = {"mode": parts[0]["mode"], "passes": 0, "images": 0, "ticks_sum": 0, "sweeps": 0, "window_s": 0.0, "wall_s": 0.0,
           "devices": [dict(d, n=0, sum_vbus=0, sum_vshunt=0, sum_prod=0, errors=0) for d in parts[0]["devices"]]}
    for p in parts:
        for key in ("passes", "images", "ticks_sum", "sweeps", "window_s", "wall_s"):
            out[key] += p[key]
        for acc, d in zip(out["devices"], p["devices"]):
            if (acc["channel"], acc["addr"]) != (d["channel"], d["addr"]):
                raise ConnectionError("POWER replies list the monitors in a different order")
            for key in ("n", "sum_vbus", "sum_vshunt", "sum_prod", "errors"):
                acc[key] += d[key]
    return out


def power_window(sock: socket.socket, mode: int, warmup_s: int, window_s: int, chunk_s: int = POWER_CHUNK_S) -> dict:
    """power_command() for a long window, split into commands of at most `chunk_s` seconds that are merged. The board
    blocks its main loop while it samples, so it cannot send TCP ACKs / answer the host's retransmissions meanwhile; one
    command of ~20 s or more makes the host's TCP stack give up and reset the connection. Between two chunks the network
    is served and the next chunk starts (mode 1: the accelerator restarts, the warm-up is kept running with unmeasured
    chunks, a pass in flight at a chunk end is not counted)."""
    left = warmup_s
    while left > 0:
        power_command(sock, mode, 0, min(chunk_s, left))   # unmeasured: keeps the load / temperature up
        left -= min(chunk_s, left)
    parts, left = [], window_s
    while left > 0:
        step = min(chunk_s, left)
        parts.append(power_command(sock, mode, 0, step))
        left -= step
    return merge_power(parts)


def run_split_benchmark(ip: str, port: int, frames: Sequence[bytes], dma_mode: int = DMA_MODE_BATCH,
                        repeats: int = 5, width: int = 0, height: int = 0, connect_retries: int = 10,
                        log: Callable[[str], None] = print, calibrate: bool = False,
                        power: PowerPlan | None = None) -> dict:
    """Run the whole split on the board and return
    {"outputs": [bytes per image], "latency_ticks": (n, 3) int64, "run_ticks": (passes, 3) int64,
     "run_calls": [{"reps", "wall_s", "ticks"}, ...], "upload_s", "fetch_s", "power": None or {"idle": ..., "load": ...,
     "idle_after": ... or None} (each a power_window() result)}.
    With `calibrate`, a short RUN of 2 passes precedes the main RUN of `repeats` passes (all passes count as
    samples); the PC times both RUN commands, so the slope (wall_main - wall_short) / (repeats - 2) is the true
    seconds per pass on the PC's clock, independent of the PL clock assumption (see implied_pl_clock_mhz).
    With `power`, the board's rails are sampled around the benchmark: an idle window first (before the images are
    loaded), then, after the RUN, the inference window (split back-to-back) and an optional second idle window.
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
        except (ConnectionError, socket.timeout, OSError) as exc:
            if sock is not None:
                sock.close()
            sock = None
            if attempt == connect_retries - 1:
                raise HandshakeError(
                    f"{ip}:{port} does not accept the lwip_collect_results (IMGC) header ({exc}). The board is probably running "
                    "another app (lwip_stream answers only IMGS headers) or the previous collect session is still open: "
                    "run the lwip_collect_results application and reset the board if needed") from exc
            time.sleep(0.5)
    power_result = None
    try:
        if reply[:3] != b"ACK":
            raise ConnectionError(f"bad header ACK: {reply!r}")
        out_bytes = struct.unpack("!I", reply[3:])[0]

        idle = None
        if power is not None:
            log(f"[POWER] idle window {power.idle_s} s ...")
            idle = power_window(sock, POWER_MODE_IDLE, 0, power.idle_s, power.chunk_s)

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

        if power is not None:
            log(f"[POWER] inference window: {power.warmup_s} s warm-up + {power.load_s} s measured, split back-to-back ...")
            load = power_window(sock, POWER_MODE_LOAD, power.warmup_s, power.load_s, power.chunk_s)
            idle_after = None
            if power.idle_after:
                log(f"[POWER] idle window after the load, {power.idle_s} s ...")
                idle_after = power_window(sock, POWER_MODE_IDLE, 0, power.idle_s, power.chunk_s)
            power_result = {"idle": idle, "load": load, "idle_after": idle_after}

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
            "upload_s": upload_s, "fetch_s": fetch_s, "power": power_result}


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
