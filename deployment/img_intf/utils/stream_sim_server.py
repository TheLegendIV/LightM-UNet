"""Localhost model of the pipelined lwIP streaming server (lwip_stream/src/run.c).

Speaks the "IMGS" protocol (see interfaces/stream_images.py) and mimics the
firmware's structure so the client and flow control can be tested without the
board:

  - a ring of `ring_slots` frames: a slot is taken when a frame is accepted
    and only released after its result was handed to TCP (write + drain);
  - when the ring is full the server stops reading the socket, so the client
    stalls through TCP backpressure (like withholding tcp_recved());
  - one "accelerator" stage processes frames in order, `proc_delay_s` each;
    the result overwrites the frame (in place) with `byte % 5` (stand-in for
    the on-chip argmax class map);
  - STOP is acknowledged ("ACK") only after all results were sent.

Usage: python stream_sim_server.py [--host 127.0.0.1] [--port 5002] [--slots 5] [--delay-ms 10]
"""
import asyncio
import os
import socket
import struct
import sys

_DEPLOYMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _DEPLOYMENT_DIR not in sys.path:
    sys.path.insert(0, _DEPLOYMENT_DIR)
from img_intf.utils.batch_img_header import (  # noqa: E402
    parse_batch_header, MAGIC_STREAM, MARKER_IMAGE, MARKER_STOP,
)

CLASS_TABLE = bytes(b % 5 for b in range(256))


def process_frame(data: bytes) -> bytes:
    """What the simulated accelerator does to one frame."""
    return data.translate(CLASS_TABLE)


class StreamSimServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, ring_slots: int = 5,
                 proc_delay_s: float = 0.0, payload_length: int = 512 * 512,
                 sndbuf: int = None):
        self.host = host
        self.port = port
        self.ring_slots = ring_slots
        self.proc_delay_s = proc_delay_s
        self.payload_length = payload_length
        self.sndbuf = sndbuf  # shrink SO_SNDBUF (testing aid, like lwIP's TCP_SND_BUF)
        self.occupied = 0
        self.max_occupied = 0
        self.rx_stalls = 0
        self.frames_in = 0
        self.frames_out = 0
        self.sessions = 0
        self.error = None
        self._server = None

    async def start(self):
        self._server = await asyncio.start_server(self._handle, self.host, self.port,
                                                  limit=64 * 1024)
        self.port = self._server.sockets[0].getsockname()[1]

    async def close(self):
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader, writer):
        if self.sndbuf is not None:
            sock = writer.get_extra_info("socket")
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sndbuf)
        try:
            while True:
                try:
                    prefix = await reader.readexactly(4)  # length prefix, ignored
                except asyncio.IncompleteReadError:
                    break
                header = await reader.readexactly(16)
                hdr = parse_batch_header(header, magic=MAGIC_STREAM)
                if hdr.payload_length != self.payload_length:
                    raise ValueError(f"payload_length {hdr.payload_length} != {self.payload_length}")
                writer.write(b"ACK" + struct.pack("!I", self.payload_length))
                await writer.drain()
                self.sessions += 1
                await self._session(reader, writer)
        except Exception as exc:  # noqa: BLE001 - surfaced to the test via .error
            self.error = exc
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _session(self, reader, writer):
        slots = asyncio.Semaphore(self.ring_slots)
        ready: asyncio.Queue = asyncio.Queue()
        done: asyncio.Queue = asyncio.Queue()

        async def accelerator():
            while True:
                frame = await ready.get()
                if frame is None:
                    await done.put(None)
                    return
                if self.proc_delay_s:
                    await asyncio.sleep(self.proc_delay_s)
                await done.put(process_frame(frame))

        async def transmitter():
            while True:
                result = await done.get()
                if result is None:
                    writer.write(b"ACK")
                    await writer.drain()
                    return
                writer.write(result)
                await writer.drain()
                self.frames_out += 1
                self.occupied -= 1
                slots.release()

        tasks = [asyncio.ensure_future(accelerator()), asyncio.ensure_future(transmitter())]
        try:
            while True:
                marker = (await reader.readexactly(1))[0]
                if marker == MARKER_STOP:
                    await ready.put(None)
                    break
                if marker != MARKER_IMAGE:
                    raise ValueError(f"unknown marker 0x{marker:02x}")
                if slots.locked():
                    self.rx_stalls += 1
                await slots.acquire()  # ring full -> stop reading -> TCP backpressure
                self.occupied += 1
                self.max_occupied = max(self.max_occupied, self.occupied)
                frame = await reader.readexactly(self.payload_length)
                self.frames_in += 1
                await ready.put(frame)
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5002)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--delay-ms", type=float, default=10.0, help="simulated per-frame accelerator time")
    ap.add_argument("--payload", type=int, default=512 * 512)
    a = ap.parse_args()

    async def run():
        srv = StreamSimServer(a.host, a.port, a.slots, a.delay_ms / 1000.0, a.payload)
        await srv.start()
        print(f"[SERVER] streaming sim listening on {srv.host}:{srv.port} "
              f"({a.slots} slots, {a.delay_ms} ms/frame)")
        try:
            await asyncio.Event().wait()
        finally:
            await srv.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")


if __name__ == "__main__":
    main()
