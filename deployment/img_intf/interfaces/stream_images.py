"""Asynchronous, full-duplex streaming client for the pipelined lwIP server.

Matches deployment/FINN_interface/lwip_stream/src/run.c ("IMGS" protocol):

  C->S  [u32 BE length prefix][16-byte header: "IMGS", payload_length, width, height]
  S->C  "ACK" + u32 BE output_bytes_per_frame
  C->S  per frame, without waiting for anything: 0x01 + payload_length bytes
  S->C  per frame, in order, no prefix/ack: output_bytes_per_frame bytes
  C->S  0x00 (stop)
  S->C  "ACK" after the last result
  S->C  optional trailer right behind the ACK: "HWL" + 3 x u32 BE PL-clock ticks (first input beat, first output
        beat, last output beat of the session's first frame, from the axi_late_mon block): the single-frame
        inference latency. Read when read_hw_trailer=True (the CLI does); other clients just close after the ACK.

The sender and the receiver run as separate asyncio tasks, so the sender may
block on TCP backpressure (server ring full) while results keep draining.

Usage (repo root):
  python deployment/img_intf/interfaces/stream_images.py [--ip 192.168.0.111] [--port 7]
         [--frames N] [--name s12_512_bilinear_mixed68] [--no-save] IMAGE_OR_DIR ...

Inputs are headerless u8 .raw files (payload size), PNGs (preprocessed with
the --input-profile, default --name) or directories of those. The list is
cycled until --frames frames were sent (default: one pass). Each frame is
loaded/preprocessed just before it is sent (the next one is prefetched while
the current one is on the wire) and every result is written to
output/<name>/raw/<input stem>.raw as soon as it is received.

--overlay saves each result blended over its input image as output/<name>/overlay/<stem>_overlay.png
(--alpha sets the opacity); --show displays the same blend live in a window. For .raw inputs the
preprocessed u8 image is the background. --mp4 FILE also encodes the blended frames into a video
(--fps, default 30; needs `pip install imageio-ffmpeg`).
"""
import asyncio
import glob
import inspect
import os
import socket
import struct
import sys
import time
from typing import Awaitable, Callable, List, Optional, Sequence, Union

_DEPLOYMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _DEPLOYMENT_DIR not in sys.path:
    sys.path.insert(0, _DEPLOYMENT_DIR)
from img_intf.utils.batch_img_header import (  # noqa: E402
    build_batch_header, MAGIC_STREAM, MARKER_IMAGE, MARKER_STOP,
)

IMAGE_H = int(os.getenv("IMAGE_H", "512"))
IMAGE_W = int(os.getenv("IMAGE_W", "512"))
ACCEL_INPUT_BYTES = int(os.getenv("ACCEL_INPUT_BYTES", str(IMAGE_H * IMAGE_W)))
PL_CLOCK_MHZ = float(os.getenv("PL_CLOCK_MHZ", "187.481262"))  # accelerator clock: converts the server's PL ticks to time
HW_TRAILER_WAIT_S = 0.5  # how long to wait for the optional hardware-latency trailer behind the stop ACK

FrameSource = Union[bytes, str]  # raw bytes, or a path handed to `loader`
ResultCallback = Callable[[int, bytes], Union[None, Awaitable[None]]]


def _read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class StreamResult:
    def __init__(self):
        self.frames = 0
        self.elapsed_s = 0.0
        self.latencies_s: List[float] = []  # send-start -> result-complete, per frame
        self.done_times: List[float] = []    # perf_counter() at which each result was complete
        self.stalls = 0                     # sends that blocked on server backpressure
        self.stalled_s = 0.0
        self.hw_ticks = None                # (input_first, output_first, output_last) PL ticks of the first frame
        self.send_times: List[float] = []   # perf_counter() at which each frame started to be sent
        self.prep_start: dict = {}          # frame index -> perf_counter() at the start of its preprocessing
        self.prep_s: dict = {}              # frame index -> preprocessing duration in s
        self.app = None                     # set by the CLI: postprocess/overlay timings and app wall clock

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def steady_fps(self) -> float:
        """Result rate once the pipeline is full: the first result (start-up, pipeline fill) is excluded."""
        span = self.done_times[-1] - self.done_times[0] if len(self.done_times) > 1 else 0.0
        return (len(self.done_times) - 1) / span if span > 0 else 0.0

    @property
    def hw_latency_ms(self):
        """(input -> first output beat, input -> last output beat) of the first frame in ms, or None."""
        if self.hw_ticks is None:
            return None
        t_in, t_first, t_last = self.hw_ticks  # unsigned 32-bit subtraction handles one wrap of the PL counter
        to_ms = lambda t: (t % 2**32) / PL_CLOCK_MHZ / 1000  # noqa: E731
        return to_ms(t_first - t_in), to_ms(t_last - t_in)


class AsyncImageStreamer:
    """Stream `num_frames` frames (cycling through `frames`) to the server.

    `frames` items are bytes or paths; paths are turned into payload bytes by
    `loader(path)` (default: read the raw file) in a worker thread right before
    sending, with the next frame prefetched during the current send.
    If `output_dir` is set, each result is written there as soon as it arrives
    (<input stem>.raw, <stem>_<cycle>.raw on later passes, frame_<i>.raw for
    bytes inputs). `on_result(index, data)` is called in order for every
    returned frame; it may be a coroutine (a slow one throttles the receiver
    and, through TCP, the server). A send that blocks longer than
    `stall_warn_s` means the server's ring is full: it is reported through
    `log` (at most once per second). `rcvbuf` shrinks the socket receive
    buffer (testing aid, as is `sndbuf`).
    """

    def __init__(
        self,
        ip: str,
        port: int,
        frames: Sequence[FrameSource],
        num_frames: Optional[int] = None,
        payload_length: int = ACCEL_INPUT_BYTES,
        on_result: Optional[ResultCallback] = None,
        rcvbuf: Optional[int] = None,
        width: int = 0,
        height: int = 0,
        timeout_s: Optional[float] = None,
        loader: Callable[[str], bytes] = _read_file,
        output_dir: Optional[str] = None,
        stall_warn_s: float = 0.05,
        log: Callable[[str], None] = print,
        sndbuf: Optional[int] = None,
        read_hw_trailer: bool = False,
    ):
        if not frames:
            raise ValueError("no frames to send")
        self.ip = ip
        self.port = port
        self.frames = list(frames)
        self.num_frames = len(self.frames) if num_frames is None else num_frames
        self.payload_length = payload_length
        self.on_result = on_result
        self.rcvbuf = rcvbuf
        self.sndbuf = sndbuf
        self.width = width
        self.height = height
        self.timeout_s = timeout_s
        self.loader = loader
        self.output_dir = output_dir
        self.stall_warn_s = stall_warn_s
        self.log = log
        self.read_hw_trailer = read_hw_trailer
        self.output_bytes = payload_length
        self._prep_start: dict = {}  # frame index -> perf_counter() when its loader (preprocessing) started
        self._prep_s: dict = {}      # frame index -> loader duration in s
        self._last_stall_log = float("-inf")

    def _frame_bytes(self, index: int) -> bytes:
        src = self.frames[index % len(self.frames)]
        if isinstance(src, (bytes, bytearray)):
            data = bytes(src)
        else:
            t = time.perf_counter()
            data = self.loader(src)  # file read + preprocessing
            self._prep_start[index] = t
            self._prep_s[index] = time.perf_counter() - t
        if len(data) != self.payload_length:
            raise ValueError(f"frame {index}: {len(data)} bytes, expected {self.payload_length}")
        return data

    def _prefetch(self, loop, index: int):
        if isinstance(self.frames[index % len(self.frames)], (bytes, bytearray)):
            fut = loop.create_future()
            fut.set_result(self._frame_bytes(index))
            return fut
        return loop.run_in_executor(None, self._frame_bytes, index)  # file read / preprocessing

    def _out_name(self, index: int) -> str:
        n = len(self.frames)
        src = self.frames[index % n]
        if not isinstance(src, str):
            return f"frame_{index:06d}.raw"
        stem = os.path.splitext(os.path.basename(src))[0]
        cycle = index // n
        return f"{stem}.raw" if cycle == 0 else f"{stem}_{cycle:04d}.raw"

    def _save(self, index: int, data: bytes):
        with open(os.path.join(self.output_dir, self._out_name(index)), "wb") as f:
            f.write(data)

    def _note_stall(self, index: int, waited_s: float, result: StreamResult):
        result.stalls += 1
        result.stalled_s += waited_s
        now = time.perf_counter()
        if now - self._last_stall_log >= 1.0:
            self._last_stall_log = now
            self.log(f"[WARN] frame {index}: send stalled {waited_s * 1000:.0f} ms "
                     f"(server buffer full, backpressure); {result.stalls} stalled send(s) so far")

    async def _send_frames(self, writer: asyncio.StreamWriter, send_times: List[float],
                           result: StreamResult):
        loop = asyncio.get_event_loop()
        nxt = self._prefetch(loop, 0)
        for i in range(self.num_frames):
            data = await nxt
            if i + 1 < self.num_frames:
                nxt = self._prefetch(loop, i + 1)  # overlaps with the send below
            send_times.append(time.perf_counter())
            writer.write(struct.pack("!B", MARKER_IMAGE))
            writer.write(data)
            t = time.perf_counter()
            await writer.drain()  # blocks while the server's ring/TCP window is full
            waited = time.perf_counter() - t
            if waited > self.stall_warn_s:
                self._note_stall(i, waited, result)
        writer.write(struct.pack("!B", MARKER_STOP))
        await writer.drain()

    async def _receive_results(self, reader: asyncio.StreamReader,
                               send_times: List[float], result: StreamResult):
        loop = asyncio.get_event_loop()
        for i in range(self.num_frames):
            data = await reader.readexactly(self.output_bytes)
            now = time.perf_counter()
            result.latencies_s.append(now - send_times[i])
            result.done_times.append(now)
            result.frames += 1
            if self.output_dir:
                await loop.run_in_executor(None, self._save, i, data)
            if self.on_result is not None:
                ret = self.on_result(i, data)
                if inspect.isawaitable(ret):
                    await ret
        ack = await reader.readexactly(3)
        if ack != b"ACK":
            raise ConnectionError(f"expected stop ACK, got {ack!r}")
        if self.read_hw_trailer:
            result.elapsed_s = time.perf_counter() - self._t0  # stop the clock before waiting for the trailer
            try:  # optional "HWL" + 3 x u32 BE PL ticks (input_first, output_first, output_last) behind the ACK
                trailer = await asyncio.wait_for(reader.readexactly(15), HW_TRAILER_WAIT_S)
            except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                trailer = b""
            if trailer[:3] == b"HWL":
                result.hw_ticks = struct.unpack("!3I", trailer[3:])

    async def run(self) -> StreamResult:
        coro = self._run()
        if self.timeout_s is not None:
            return await asyncio.wait_for(coro, self.timeout_s)
        return await coro

    async def _run(self) -> StreamResult:
        reader, writer = await self._connect()
        # drain() then waits until the OS accepted the whole frame, so a full server ring shows up as a stall.
        writer.transport.set_write_buffer_limits(high=0)
        result = StreamResult()
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        try:
            header = build_batch_header(self.payload_length, self.width, self.height,
                                        magic=MAGIC_STREAM)
            writer.write(struct.pack("!I", len(header)) + header)
            await writer.drain()

            reply = await reader.readexactly(7)
            if reply[:3] != b"ACK":
                raise ConnectionError(f"bad header ACK: {reply!r}")
            self.output_bytes = struct.unpack("!I", reply[3:])[0]

            send_times: List[float] = []
            result.send_times = send_times
            result.prep_start, result.prep_s = self._prep_start, self._prep_s
            t0 = self._t0 = time.perf_counter()
            sender = asyncio.ensure_future(self._send_frames(writer, send_times, result))
            receiver = asyncio.ensure_future(self._receive_results(reader, send_times, result))
            try:
                await asyncio.gather(sender, receiver)
            except BaseException:
                sender.cancel()
                receiver.cancel()
                raise
            if not result.elapsed_s:
                result.elapsed_s = time.perf_counter() - t0
            return result
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _connect(self):
        if self.rcvbuf is None and self.sndbuf is None:
            return await asyncio.open_connection(self.ip, self.port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.rcvbuf is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf)
        if self.sndbuf is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sndbuf)
        sock.setblocking(False)
        await asyncio.get_event_loop().sock_connect(sock, (self.ip, self.port))
        return await asyncio.open_connection(sock=sock)


def _collect_inputs(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.raw"))))
            out.extend(sorted(glob.glob(os.path.join(p, "*.png"))))
        else:
            out.append(p)
    if not out:
        raise FileNotFoundError("no input images found")
    return out


def _show_live(q, title: str, total: int) -> None:
    """Tk window that shows the newest (index, name, rgb) from `q`; None on `q` marks the end of the stream."""
    import queue
    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(title)
    view = tk.Label(root)
    view.pack()
    status = tk.Label(root, anchor="w", justify="left")
    status.pack(fill="x")
    legend = "red LAD | green RCA | blue LCX | yellow LM"
    state = {"text": "", "done": False}

    def poll():
        newest = None
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                state["done"] = True
            else:
                newest = item
        if newest is not None:
            i, name, rgb = newest
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            view.configure(image=photo)
            view.image = photo  # keep a reference
            state["text"] = f"{name}  ({i + 1}/{total})"
        status.configure(text=f"{state['text']}{'  - done' if state['done'] else ''}\n{legend}")
        if not state["done"]:
            root.after(5, poll)

    poll()
    root.mainloop()


def _pct(sorted_vals: List[float], q: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def _app_stats(res: StreamResult) -> dict:
    """Whole-application numbers: preprocessing + PC->board->PC loop + postprocessing (overlay / video / viewer).
    A frame's application latency runs from the start of its preprocessing to the end of its postprocessing
    (to the result arriving when there is no postprocessing)."""
    n = res.frames
    app = res.app or {}
    render_s = app.get("render_s", {})
    render_done = app.get("render_done", {})
    starts = [res.prep_start.get(i, res.send_times[i]) for i in range(n)]
    ends = [render_done.get(i, res.done_times[i]) for i in range(n)]
    wall = app["t1"] - app["t0"] if "t0" in app and "t1" in app else max(ends) - min(starts)
    lat = sorted(e - s for s, e in zip(starts, ends))
    span = max(ends) - min(ends)
    out = {
        "wall_s": wall, "fps": n / wall if wall > 0 else 0.0,
        "steady_fps": (n - 1) / span if n > 1 and span > 0 else 0.0,
        "latency_first_ms": (ends[0] - starts[0]) * 1000, "latency_median_ms": lat[len(lat) // 2] * 1000,
        "latency_p95_ms": _pct(lat, 0.95) * 1000, "latency_max_ms": lat[-1] * 1000,
        "prep_ms": None, "post_ms": None,
    }
    if res.prep_s:
        v = sorted(res.prep_s.values())
        out["prep_ms"] = (sum(v) / len(v) * 1000, v[len(v) // 2] * 1000, v[-1] * 1000)
    if render_s:
        v = sorted(render_s.values())
        out["post_ms"] = (sum(v) / len(v) * 1000, v[len(v) // 2] * 1000, v[-1] * 1000)
    return out


def _report(res: StreamResult, out_dir: Optional[str], metrics_path: Optional[str] = None,
            payload_bytes: int = ACCEL_INPUT_BYTES) -> None:
    """The first block is the PC -> board -> PC loop (network, board CPU, accelerator, network); the second block is
    the whole application around it (preprocessing and postprocessing/overlay on the PC)."""
    lat = sorted(res.latencies_s)
    p95 = _pct(lat, 0.95)
    first_ms = res.latencies_s[0] * 1000  # frame 0 enters an empty pipeline: the unloaded loop latency
    steady = res.steady_fps
    mb_s = steady * payload_bytes / 1e6  # payload per direction
    print(f"[INFO] PC->PC throughput: {res.frames} frame(s) in {res.elapsed_s:.3f} s -> {res.fps:.2f} fps overall")
    if steady:
        print(f"[INFO] PC->PC steady state (first result excluded): {steady:.2f} fps = {1000 / steady:.2f} ms/frame, "
              f"{mb_s:.1f} MB/s ({mb_s * 8:.0f} Mbit/s) per direction")
    print(f"[INFO] PC->PC latency, first frame (empty pipeline, send start -> result complete): {first_ms:.1f} ms")
    print(f"[INFO] PC->PC turnaround under load (includes queueing behind earlier frames): "
          f"min {lat[0] * 1000:.1f}, median {lat[len(lat) // 2] * 1000:.1f}, p95 {p95 * 1000:.1f}, "
          f"max {lat[-1] * 1000:.1f} ms")
    if res.stalls:
        print(f"[WARN] {res.stalls} send(s) stalled by a full server buffer, "
              f"{res.stalled_s:.2f} s total")
    hw = res.hw_latency_ms
    if hw is not None:
        print(f"[INFO] inference latency of that first frame, accelerator only (PL clock {PL_CLOCK_MHZ} MHz): "
              f"{hw[1]:.3f} ms full image, {hw[0]:.3f} ms to the first output pixel; "
              f"the other {first_ms - hw[1]:.1f} ms of the loop is network + board CPU")
    else:
        print("[INFO] no hardware-latency trailer from the server (firmware without the latency monitor?)")

    ap = _app_stats(res)
    stage = lambda t: f"mean {t[0]:.1f}, median {t[1]:.1f}, max {t[2]:.1f} ms"  # noqa: E731
    print("[INFO] --- whole application (preprocess + board loop + postprocess/overlay) ---")
    if ap["prep_ms"]:
        print(f"[INFO] preprocessing per frame (worker thread, overlapped with the loop): {stage(ap['prep_ms'])}")
    if ap["post_ms"]:
        print(f"[INFO] postprocessing/overlay per frame (worker thread, overlapped): {stage(ap['post_ms'])}")
    print(f"[INFO] application throughput: {res.frames} frame(s) in {ap['wall_s']:.3f} s -> {ap['fps']:.2f} fps overall"
          + (f", steady state {ap['steady_fps']:.2f} fps = {1000 / ap['steady_fps']:.2f} ms/frame" if ap["steady_fps"] else ""))
    print(f"[INFO] application latency, preprocess start -> result processed: first frame {ap['latency_first_ms']:.1f} ms; "
          f"under load median {ap['latency_median_ms']:.1f}, p95 {ap['latency_p95_ms']:.1f}, max {ap['latency_max_ms']:.1f} ms")
    if steady and ap["steady_fps"] and ap["steady_fps"] < 0.95 * steady:
        interval_ms = 1000 / steady
        limits = [name for name, t in (("preprocessing", ap["prep_ms"]), ("postprocessing/overlay", ap["post_ms"]))
                  if t and t[0] > interval_ms]
        print(f"[INFO] the PC side slows the loop from {steady:.1f} to {ap['steady_fps']:.1f} fps"
              + (f": {' and '.join(limits)} take longer per frame than the {interval_ms:.1f} ms loop interval" if limits else ""))
    if out_dir:
        print(f"[INFO] class maps in {out_dir}")
    if metrics_path:
        import json
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({
                "frames": res.frames,
                "pc_to_pc": {
                    "elapsed_s": round(res.elapsed_s, 4), "fps": round(res.fps, 3), "steady_fps": round(steady, 3),
                    "mb_per_s_per_direction": round(mb_s, 3), "latency_first_frame_ms": round(first_ms, 2),
                    "turnaround_min_ms": round(lat[0] * 1000, 2),
                    "turnaround_median_ms": round(lat[len(lat) // 2] * 1000, 2),
                    "turnaround_p95_ms": round(p95 * 1000, 2), "turnaround_max_ms": round(lat[-1] * 1000, 2),
                    "stalls": res.stalls,
                },
                "application": {
                    "wall_s": round(ap["wall_s"], 4), "fps": round(ap["fps"], 3),
                    "steady_fps": round(ap["steady_fps"], 3), "latency_first_ms": round(ap["latency_first_ms"], 2),
                    "latency_median_ms": round(ap["latency_median_ms"], 2),
                    "latency_p95_ms": round(ap["latency_p95_ms"], 2), "latency_max_ms": round(ap["latency_max_ms"], 2),
                    "preprocess_ms_mean_median_max": None if not ap["prep_ms"] else [round(x, 2) for x in ap["prep_ms"]],
                    "postprocess_ms_mean_median_max": None if not ap["post_ms"] else [round(x, 2) for x in ap["post_ms"]],
                },
                "accelerator": {
                    "pl_clock_mhz": PL_CLOCK_MHZ,
                    "hw_ticks": None if res.hw_ticks is None else list(res.hw_ticks),
                    "first_output_ms": None if hw is None else round(hw[0], 4),
                    "latency_ms": None if hw is None else round(hw[1], 4),
                },
            }, f, indent=2)
        print(f"[INFO] metrics written to {metrics_path}")


def main():
    import argparse
    import numpy as np
    from PIL import Image
    from img_intf.utils.preprocess import png_to_u8
    from img_intf.utils.raw_to_png import overlay

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".raw / .png files or directories")
    ap.add_argument("--ip", default="192.168.0.111")
    ap.add_argument("--port", type=int, default=7)
    ap.add_argument("--frames", type=int, default=None, help="frames to send (inputs are cycled); default one pass")
    ap.add_argument("--name", default="s12_512_bilinear_int8_maxfps", help="output/<name>/stream destination and default input profile")
    ap.add_argument("--input-profile", help="input scale profile in utils/input_profiles.json (default: --name)")
    ap.add_argument("--no-save", action="store_true", help="do not write returned class maps")
    ap.add_argument("--overlay", action="store_true", help="save results blended over the inputs (output/<name>/overlay)")
    ap.add_argument("--show", action="store_true", help="display the blended results live in a window")
    ap.add_argument("--alpha", type=float, default=0.5, help="opacity of the class colours in the overlay")
    ap.add_argument("--mp4", help="encode the blended results into this mp4 file")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the --mp4 video")
    a = ap.parse_args()

    sources = _collect_inputs(a.inputs)
    profile = a.input_profile or a.name

    def load(path: str) -> bytes:
        if not path.lower().endswith(".png"):
            return _read_file(path)
        u8 = png_to_u8(path, profile)
        if u8.size != ACCEL_INPUT_BYTES:
            raise ValueError(f"{path}: {u8.shape} does not match {IMAGE_H}x{IMAGE_W}")
        return np.ascontiguousarray(u8, dtype=np.uint8).tobytes()

    out_dir = os.path.join(_DEPLOYMENT_DIR, "img_intf", "output", a.name, "raw")
    overlay_dir = os.path.join(_DEPLOYMENT_DIR, "img_intf", "output", a.name, "overlay")

    def input_gray(path: str) -> np.ndarray:
        if path.lower().endswith(".png"):
            return np.asarray(Image.open(path).convert("L"))
        return np.fromfile(path, dtype=np.uint8).reshape(IMAGE_H, IMAGE_W)

    live = None  # queue to the viewer
    video = None
    if a.mp4:
        import imageio_ffmpeg
        os.makedirs(os.path.dirname(os.path.abspath(a.mp4)), exist_ok=True)
        video = imageio_ffmpeg.write_frames(a.mp4, (IMAGE_W, IMAGE_H), fps=a.fps, macro_block_size=1)
        video.send(None)

    def render(i: int, data: bytes) -> None:
        stem = os.path.splitext(streamer._out_name(i))[0]
        rgb = overlay(input_gray(sources[i % len(sources)]),
                      np.frombuffer(data, np.uint8).reshape(IMAGE_H, IMAGE_W), a.alpha)
        if a.overlay:
            Image.fromarray(rgb).save(os.path.join(overlay_dir, f"{stem}_overlay.png"))
        if video is not None:
            video.send(rgb)
        if live is not None:
            live.put((i, stem, rgb))

    from concurrent.futures import ThreadPoolExecutor
    renderer = ThreadPoolExecutor(1)  # keeps blending/PNG writing off the receive loop, in order

    jobs = []
    render_s: dict = {}     # frame index -> postprocessing (blend/save/video/viewer) duration
    render_done: dict = {}  # frame index -> perf_counter() when its postprocessing finished
    app_t: dict = {}        # application wall clock: t0 before the stream starts, t1 after the last frame is processed

    def render_timed(i: int, data: bytes) -> None:
        t = time.perf_counter()
        render(i, data)
        render_done[i] = time.perf_counter()
        render_s[i] = render_done[i] - t

    def on_result(i: int, data: bytes) -> None:
        jobs.append(renderer.submit(render_timed, i, data))

    def finish_render(res: StreamResult) -> None:
        renderer.shutdown(wait=True)
        if video is not None:
            video.close()
        for j in jobs:
            j.result()  # re-raise any rendering error
        app_t["t1"] = time.perf_counter()
        res.app = {**app_t, "render_s": render_s, "render_done": render_done}

    if a.overlay:
        os.makedirs(overlay_dir, exist_ok=True)
    streamer = AsyncImageStreamer(
        a.ip, a.port, sources, num_frames=a.frames,
        width=IMAGE_W, height=IMAGE_H,
        loader=load, output_dir=None if a.no_save else out_dir,
        on_result=on_result if (a.overlay or a.show or a.mp4) else None,
        read_hw_trailer=True,
    )
    saved = None if a.no_save else out_dir
    metrics_path = None if a.no_save else os.path.join(os.path.dirname(out_dir), "stream_metrics.json")
    if not a.show:
        app_t["t0"] = time.perf_counter()
        res = asyncio.run(streamer.run())
        finish_render(res)
        _report(res, saved, metrics_path)
        return

    import queue
    import threading
    live = queue.Queue()
    box = {}

    def worker():
        try:
            app_t["t0"] = time.perf_counter()
            box["res"] = asyncio.run(streamer.run())
            finish_render(box["res"])
        except BaseException as e:  # reported after the window closes
            box["err"] = e
        finally:
            live.put(None)

    threading.Thread(target=worker, daemon=True).start()
    _show_live(live, f"{a.name} - class overlay", streamer.num_frames)
    if "err" in box:
        raise box["err"]
    if "res" in box:
        _report(box["res"], saved, metrics_path)


if __name__ == "__main__":
    main()
