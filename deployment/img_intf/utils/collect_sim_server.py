"""Simulated lwip_collect_results server ("IMGC" protocol) for testing collect_results.py / collect_split.py.

LOAD stores images; LATENCY answers a fixed single-image latency per image; RUN answers PL ticks for
latency + (n - 1) * interval per pass; FETCH returns process_frame() of each stored image (class table like
stream_sim_server). Ticks are deterministic, so a client can check its arithmetic.
"""
import asyncio
import os
import struct
import sys

_DEPLOYMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _DEPLOYMENT_DIR not in sys.path:
    sys.path.insert(0, _DEPLOYMENT_DIR)
from img_intf.utils.batch_img_header import (  # noqa: E402
    MAGIC_COLLECT, MARKER_FETCH, MARKER_LATENCY, MARKER_LOAD, MARKER_RUN, MARKER_STOP, parse_batch_header,
)
from img_intf.utils.stream_sim_server import process_frame  # noqa: E402


class CollectSimServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, payload_length: int = 512 * 512,
                 latency_ticks: int = 1696933, interval_ticks: int = 460000, first_tick: int = 134,
                 identical_outputs: bool = False):
        self.host, self.port, self.payload_length = host, port, payload_length
        self.latency_ticks, self.interval_ticks, self.first_tick = latency_ticks, interval_ticks, first_tick
        self.identical_outputs = identical_outputs  # emulate a DMA whose numReps does not advance the buffer
        self.error = None
        self._server = None

    async def start(self):
        self._server = await asyncio.start_server(self._handle, self.host, self.port, limit=1 << 20)
        self.port = self._server.sockets[0].getsockname()[1]

    async def close(self):
        self._server.close()
        await self._server.wait_closed()

    def _pack(self, first_out_offset, last_offset):
        m = 2**32  # the PL counter wraps
        f = self.first_tick
        return struct.pack("!3I", f % m, (f + first_out_offset) % m, (f + last_offset) % m)

    async def _handle(self, reader, writer):
        try:
            await reader.readexactly(4)
            hdr = parse_batch_header(await reader.readexactly(16), magic=MAGIC_COLLECT)
            if hdr.payload_length != self.payload_length:
                raise ValueError(f"payload_length {hdr.payload_length} != {self.payload_length}")
            writer.write(b"ACK" + struct.pack("!I", self.payload_length))
            images = []
            while True:
                marker = (await reader.readexactly(1))[0]
                if marker == MARKER_LOAD:
                    images.append(await reader.readexactly(self.payload_length))
                elif marker == MARKER_LATENCY:
                    t = self._pack(self.latency_ticks // 2, self.latency_ticks)
                    writer.write(b"LAT" + struct.pack("!I", len(images)) + t * len(images))
                elif marker == MARKER_RUN:
                    _mode, reps = await reader.readexactly(2)
                    t = self._pack(self.latency_ticks // 2, self.latency_ticks + (len(images) - 1) * self.interval_ticks)
                    writer.write(b"RUN" + struct.pack("!I", reps) + t * reps)
                elif marker == MARKER_FETCH:
                    outs = [process_frame(images[0] if self.identical_outputs else im) for im in images]
                    writer.write(b"OUT" + struct.pack("!I", len(outs)) + b"".join(outs))
                elif marker == MARKER_STOP:
                    writer.write(b"ACK")
                    await writer.drain()
                    break
                else:
                    raise ValueError(f"unknown command 0x{marker:02x}")
                await writer.drain()
        except Exception as exc:  # noqa: BLE001 - surfaced to the test via .error
            self.error = exc
        finally:
            writer.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5003)
    a = ap.parse_args()

    async def run():
        srv = CollectSimServer(a.host, a.port)
        await srv.start()
        print(f"[collect-sim] listening on {srv.host}:{srv.port}")
        await asyncio.Event().wait()

    asyncio.run(run())


if __name__ == "__main__":
    main()
