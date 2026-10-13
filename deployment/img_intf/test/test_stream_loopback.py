import asyncio
import os
import sys
import unittest

_DEPLOYMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _DEPLOYMENT_DIR not in sys.path:
    sys.path.insert(0, _DEPLOYMENT_DIR)
from img_intf.interfaces.stream_images import AsyncImageStreamer  # noqa: E402
from img_intf.utils.stream_sim_server import StreamSimServer, process_frame  # noqa: E402

RING_SLOTS = 5


def make_frames(n: int, size: int):
    """n distinct frames (first bytes encode the index, rest is a ramp)."""
    frames = []
    for i in range(n):
        body = bytes((i * 7 + k) & 0xFF for k in range(size))
        frames.append(body)
    return frames


class StreamLoopbackTest(unittest.TestCase):
    def _run(self, *, payload, n_frames, delay_s=0.0, slots=RING_SLOTS,
             consume_delay_s=0.0, sndbuf=None, rcvbuf=None, cycle=None):
        frames = make_frames(cycle or n_frames, payload)
        received = []

        async def on_result(i, data):
            received.append(data)
            if consume_delay_s:
                await asyncio.sleep(consume_delay_s)

        async def go():
            srv = StreamSimServer("127.0.0.1", 0, slots, delay_s, payload, sndbuf=sndbuf)
            await srv.start()
            try:
                client = AsyncImageStreamer(
                    "127.0.0.1", srv.port, frames, num_frames=n_frames,
                    payload_length=payload, on_result=on_result, rcvbuf=rcvbuf,
                    timeout_s=60,
                )
                res = await client.run()
            finally:
                await srv.close()
            return srv, res

        srv, res = asyncio.run(go())
        self.assertIsNone(srv.error)
        self.assertEqual(res.frames, n_frames)
        self.assertEqual(len(received), n_frames)
        for i, data in enumerate(received):  # in order, correct content
            self.assertEqual(data, process_frame(frames[i % len(frames)]), f"frame {i}")
        self.assertEqual((srv.frames_in, srv.frames_out), (n_frames, n_frames))
        self.assertLessEqual(srv.max_occupied, slots)
        return srv, res

    def test_in_order_small_frames(self):
        srv, _ = self._run(payload=4096, n_frames=50, delay_s=0.001)
        self.assertEqual(srv.sessions, 1)

    def test_full_size_frames(self):
        self._run(payload=512 * 512, n_frames=8, delay_s=0.002, cycle=3)

    def test_ring_overlaps_frames(self):
        """Client does not wait per frame: the ring fills while the slow
        accelerator works, and total time is ~frames*delay, not 2x."""
        srv, res = self._run(payload=4096, n_frames=30, delay_s=0.01)
        self.assertEqual(srv.max_occupied, RING_SLOTS)
        self.assertGreater(srv.rx_stalls, 0)
        self.assertLess(res.elapsed_s, 30 * 0.01 * 2.0)

    def test_backpressure_slow_consumer(self):
        """A client that reads results slowly must stall the sender through
        the full ring (no unbounded buffering, no deadlock, no data loss)."""
        srv, res = self._run(payload=65536, n_frames=24, delay_s=0.0,
                             consume_delay_s=0.02, sndbuf=8192, rcvbuf=8192)
        self.assertEqual(srv.max_occupied, RING_SLOTS)
        self.assertGreater(srv.rx_stalls, 0)

    def test_single_slot_ring(self):
        srv, _ = self._run(payload=4096, n_frames=10, delay_s=0.002, slots=1)
        self.assertEqual(srv.max_occupied, 1)

    def test_two_sessions_same_connection_server(self):
        """STOP returns the server to the header state; a second client works."""
        frames = make_frames(3, 4096)

        async def go():
            srv = StreamSimServer("127.0.0.1", 0, RING_SLOTS, 0.0, 4096)
            await srv.start()
            try:
                for _ in range(2):
                    r = await AsyncImageStreamer("127.0.0.1", srv.port, frames,
                                                 payload_length=4096, timeout_s=30).run()
                    self.assertEqual(r.frames, 3)
            finally:
                await srv.close()
            return srv

        srv = asyncio.run(go())
        self.assertIsNone(srv.error)
        self.assertEqual(srv.sessions, 2)


if __name__ == "__main__":
    unittest.main()
