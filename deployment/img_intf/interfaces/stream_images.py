"""Asynchronous, full-duplex streaming client for the pipelined lwIP server.

Matches deployment/FINN_interface/lwip_stream/src/run.c ("IMGS" protocol):

  C->S  [u32 BE length prefix][16-byte header: "IMGS", payload_length, width, height]
  S->C  "ACK" + u32 BE output_bytes_per_frame
  C->S  per frame, without waiting for anything: 0x01 + payload_length bytes
  S->C  per frame, in order, no prefix/ack: output_bytes_per_frame bytes
  C->S  0x00 (stop)
  S->C  "ACK" after the last result

The sender and the receiver run as separate asyncio tasks, so the sender may
block on TCP backpressure (server ring full) while results keep draining.

Usage (repo root):
  python deployment/img_intf/interfaces/stream_images.py [--ip 192.168.0.111] [--port 7]
         [--frames N] [--name s12_512_bilinear_mixed68] [--no-save] IMAGE_OR_DIR ...

Inputs are headerless u8 .raw files (payload size), PNGs (preprocessed with
the --input-profile, default --name) or directories of those. The list is
cycled until --frames frames were sent (default: one pass).
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

FrameSource = Union[bytes, str]  # raw bytes, or path to a headerless .raw file
ResultCallback = Callable[[int, bytes], Union[None, Awaitable[None]]]


class StreamResult:
    def __init__(self):
        self.frames = 0
        self.elapsed_s = 0.0
        self.latencies_s: List[float] = []  # send-start -> result-complete, per frame

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0


class AsyncImageStreamer:
    """Stream `num_frames` frames (cycling through `frames`) to the server.

    `frames` items are bytes or paths to raw files (read lazily at send time).
    `on_result(index, data)` is called in order for every returned frame; it
    may be a coroutine (a slow one throttles the receiver and, through TCP,
    the server). `rcvbuf` shrinks the socket receive buffer (testing aid).
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
        self.width = width
        self.height = height
        self.timeout_s = timeout_s
        self.output_bytes = payload_length

    def _frame_bytes(self, index: int) -> bytes:
        src = self.frames[index % len(self.frames)]
        if isinstance(src, (bytes, bytearray)):
            data = bytes(src)
        else:
            with open(src, "rb") as f:
                data = f.read()
        if len(data) != self.payload_length:
            raise ValueError(f"frame {index}: {len(data)} bytes, expected {self.payload_length}")
        return data

    async def _send_frames(self, writer: asyncio.StreamWriter, send_times: List[float]):
        for i in range(self.num_frames):
            data = self._frame_bytes(i)
            send_times.append(time.perf_counter())
            writer.write(struct.pack("!B", MARKER_IMAGE))
            writer.write(data)
            await writer.drain()  # blocks while the server's ring/TCP window is full
        writer.write(struct.pack("!B", MARKER_STOP))
        await writer.drain()

    async def _receive_results(self, reader: asyncio.StreamReader,
                               send_times: List[float], result: StreamResult):
        for i in range(self.num_frames):
            data = await reader.readexactly(self.output_bytes)
            result.latencies_s.append(time.perf_counter() - send_times[i])
            result.frames += 1
            if self.on_result is not None:
                ret = self.on_result(i, data)
                if inspect.isawaitable(ret):
                    await ret
        ack = await reader.readexactly(3)
        if ack != b"ACK":
            raise ConnectionError(f"expected stop ACK, got {ack!r}")

    async def run(self) -> StreamResult:
        coro = self._run()
        if self.timeout_s is not None:
            return await asyncio.wait_for(coro, self.timeout_s)
        return await coro

    async def _run(self) -> StreamResult:
        reader, writer = await self._connect()
        result = StreamResult()
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
            t0 = time.perf_counter()
            sender = asyncio.ensure_future(self._send_frames(writer, send_times))
            receiver = asyncio.ensure_future(self._receive_results(reader, send_times, result))
            try:
                await asyncio.gather(sender, receiver)
            except BaseException:
                sender.cancel()
                receiver.cancel()
                raise
            result.elapsed_s = time.perf_counter() - t0
            return result
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _connect(self):
        if self.rcvbuf is None:
            return await asyncio.open_connection(self.ip, self.port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf)
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


def main():
    import argparse
    import numpy as np
    from img_intf.utils.preprocess import png_to_u8

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".raw / .png files or directories")
    ap.add_argument("--ip", default="192.168.0.111")
    ap.add_argument("--port", type=int, default=7)
    ap.add_argument("--frames", type=int, default=None, help="frames to send (inputs are cycled); default one pass")
    ap.add_argument("--name", default="s12_512_bilinear_mixed68", help="output/<name>/stream destination and default input profile")
    ap.add_argument("--input-profile", help="input scale profile in utils/input_profiles.json (default: --name)")
    ap.add_argument("--no-save", action="store_true", help="do not write returned class maps")
    a = ap.parse_args()

    sources: List[FrameSource] = []
    for path in _collect_inputs(a.inputs):
        if path.lower().endswith(".png"):
            u8 = png_to_u8(path, a.input_profile or a.name)
            if u8.size != ACCEL_INPUT_BYTES:
                raise ValueError(f"{path}: {u8.shape} does not match {IMAGE_H}x{IMAGE_W}")
            sources.append(np.ascontiguousarray(u8, dtype=np.uint8).tobytes())
        else:
            sources.append(path)

    out_dir = os.path.join(_DEPLOYMENT_DIR, "img_intf", "output", a.name, "stream")
    if not a.no_save:
        os.makedirs(out_dir, exist_ok=True)

    def save(index: int, data: bytes):
        with open(os.path.join(out_dir, f"frame_{index:06d}.raw"), "wb") as f:
            f.write(data)

    streamer = AsyncImageStreamer(
        a.ip, a.port, sources, num_frames=a.frames,
        width=IMAGE_W, height=IMAGE_H,
        on_result=None if a.no_save else save,
    )
    res = asyncio.run(streamer.run())
    lat = sorted(res.latencies_s)
    print(f"[INFO] {res.frames} frame(s) in {res.elapsed_s:.3f} s -> {res.fps:.2f} fps; "
          f"latency median {lat[len(lat) // 2] * 1000:.1f} ms, max {lat[-1] * 1000:.1f} ms")
    if not a.no_save:
        print(f"[INFO] class maps in {out_dir}")


if __name__ == "__main__":
    main()
