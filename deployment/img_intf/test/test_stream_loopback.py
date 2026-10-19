import asyncio
import os
import sys
import tempfile
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
    def setUp(self):
        self.logs = []

    def _run(self, *, payload, n_frames, delay_s=0.0, slots=RING_SLOTS,
             consume_delay_s=0.0, sndbuf=None, rcvbuf=None, cycle=None):
        frames = make_frames(cycle or n_frames, payload)
        received = []

        async def on_result(i, data):
            received.append(data)
            if consume_delay_s:
                await asyncio.sleep(consume_delay_s)

        async def go():
            srv = StreamSimServer("127.0.0.1", 0, slots, delay_s, payload,
                                  sndbuf=sndbuf, rcvbuf=sndbuf)
            await srv.start()
            try:
                client = AsyncImageStreamer(
                    "127.0.0.1", srv.port, frames, num_frames=n_frames,
                    payload_length=payload, on_result=on_result, rcvbuf=rcvbuf,
                    timeout_s=60, log=self.logs.append, sndbuf=rcvbuf,
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
        srv, res = self._run(payload=4096, n_frames=30, delay_s=0.02)
        self.assertEqual(srv.max_occupied, RING_SLOTS)
        self.assertGreater(srv.rx_stalls, 0)
        self.assertLess(res.elapsed_s, 30 * 0.02 * 2.0)

    def test_backpressure_slow_consumer(self):
        """A client that reads results slowly must stall the sender through
        the full ring (no unbounded buffering, no deadlock, no data loss)."""
        srv, res = self._run(payload=65536, n_frames=40, delay_s=0.0,
                             consume_delay_s=0.02, sndbuf=8192, rcvbuf=8192)
        self.assertEqual(srv.max_occupied, RING_SLOTS)
        self.assertGreater(srv.rx_stalls, 0)
        self.assertGreater(res.stalls, 0)
        self.assertTrue(any("stalled" in m and "[WARN]" in m for m in self.logs))

    def test_single_slot_ring(self):
        srv, _ = self._run(payload=4096, n_frames=10, delay_s=0.002, slots=1)
        self.assertEqual(srv.max_occupied, 1)

    def test_loader_and_direct_raw_output(self):
        """Frames are loaded lazily via `loader`; each result lands in
        output_dir as <input stem>.raw (<stem>_0001.raw on the second pass)."""
        payload = 4096
        names = ["a.png", "b.png", "c.png"]
        loaded = []
        frames = dict(zip(names, make_frames(3, payload)))

        def loader(path):
            loaded.append(path)
            return frames[path]

        async def go(out_dir):
            srv = StreamSimServer("127.0.0.1", 0, RING_SLOTS, 0.0, payload)
            await srv.start()
            try:
                return await AsyncImageStreamer(
                    "127.0.0.1", srv.port, names, num_frames=5, payload_length=payload,
                    loader=loader, output_dir=out_dir, timeout_s=30).run()
            finally:
                await srv.close()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "raw")
            res = asyncio.run(go(out_dir))
            self.assertEqual(res.frames, 5)
            self.assertEqual(loaded, ["a.png", "b.png", "c.png", "a.png", "b.png"])
            self.assertEqual(sorted(os.listdir(out_dir)),
                             ["a.raw", "a_0001.raw", "b.raw", "b_0001.raw", "c.raw"])
            for stem in ("a", "b", "c"):
                with open(os.path.join(out_dir, stem + ".raw"), "rb") as f:
                    self.assertEqual(f.read(), process_frame(frames[stem + ".png"]))

    def test_paced_source(self):
        """rate_fps spaces the sends: N frames need ~(N-1)/rate, and the frames stay correct and in order."""
        payload, n, rate = 4096, 12, 50.0
        frames = make_frames(n, payload)

        async def go():
            srv = StreamSimServer("127.0.0.1", 0, RING_SLOTS, 0.0, payload)
            await srv.start()
            try:
                res = await AsyncImageStreamer("127.0.0.1", srv.port, frames, payload_length=payload,
                                               timeout_s=30, rate_fps=rate).run()
            finally:
                await srv.close()
            return srv, res

        srv, res = asyncio.run(go())
        self.assertIsNone(srv.error)
        self.assertEqual(res.frames, n)
        gaps = [b - a for a, b in zip(res.send_times, res.send_times[1:])]
        self.assertGreaterEqual(res.send_times[-1] - res.send_times[0], (n - 1) / rate * 0.95)
        self.assertLess(res.send_times[-1] - res.send_times[0], (n - 1) / rate * 1.5)
        self.assertEqual(len(gaps), n - 1)

    def test_loop_forever_stops_cleanly(self):
        """loop_forever cycles the inputs until stop_event is set; every frame in flight is still received
        in order, the stop handshake completes and the server saw exactly as many frames in as out."""
        import threading
        payload = 4096
        frames = make_frames(3, payload)
        stop = threading.Event()
        received = []

        def on_result(i, data):
            received.append(data)
            if i == 20:
                stop.set()

        async def go():
            srv = StreamSimServer("127.0.0.1", 0, RING_SLOTS, 0.001, payload)
            await srv.start()
            try:
                res = await AsyncImageStreamer("127.0.0.1", srv.port, frames, payload_length=payload,
                                               on_result=on_result, timeout_s=30,
                                               loop_forever=True, stop_event=stop).run()
            finally:
                await srv.close()
            return srv, res

        srv, res = asyncio.run(go())
        self.assertIsNone(srv.error)
        self.assertGreater(res.frames, 20)
        self.assertEqual(len(received), res.frames)
        self.assertEqual((srv.frames_in, srv.frames_out), (res.frames, res.frames))
        for i, data in enumerate(received):
            self.assertEqual(data, process_frame(frames[i % 3]), f"frame {i}")

    def test_loop_forever_stopped_before_first_frame(self):
        import threading
        stop = threading.Event()
        stop.set()

        async def go():
            srv = StreamSimServer("127.0.0.1", 0, RING_SLOTS, 0.0, 4096)
            await srv.start()
            try:
                res = await AsyncImageStreamer("127.0.0.1", srv.port, make_frames(2, 4096), payload_length=4096,
                                               timeout_s=30, loop_forever=True, stop_event=stop).run()
            finally:
                await srv.close()
            return srv, res

        srv, res = asyncio.run(go())
        self.assertIsNone(srv.error)
        self.assertEqual((res.frames, srv.frames_in), (0, 0))

    def test_loop_forever_output_names_overwrite(self):
        import threading
        payload = 4096
        names = ["a.png", "b.png"]
        frames = dict(zip(names, make_frames(2, payload)))
        stop = threading.Event()

        async def go(out_dir):
            srv = StreamSimServer("127.0.0.1", 0, RING_SLOTS, 0.0, payload)
            await srv.start()
            try:
                return await AsyncImageStreamer(
                    "127.0.0.1", srv.port, names, payload_length=payload, loader=frames.__getitem__,
                    output_dir=out_dir, on_result=lambda i, d: stop.set() if i == 6 else None,
                    timeout_s=30, loop_forever=True, stop_event=stop).run()
            finally:
                await srv.close()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "raw")
            res = asyncio.run(go(out_dir))
            self.assertGreater(res.frames, 6)
            self.assertEqual(sorted(os.listdir(out_dir)), ["a.raw", "b.raw"])

    def test_two_sequential_sessions(self):
        """Server accepts a second session after a clean STOP."""
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


class OverlayTest(unittest.TestCase):
    def test_background_untouched_and_classes_blended(self):
        import numpy as np
        from img_intf.utils.raw_to_png import PALETTE, overlay
        gray = np.full((4, 4), 100, np.uint8)
        cls = np.zeros((4, 4), np.uint8)
        cls[1, 1] = 1
        cls[2, 2] = 4
        out = overlay(gray, cls, 0.5)
        self.assertEqual(out.shape, (4, 4, 3))
        self.assertTrue((out[0, 0] == 100).all())
        self.assertEqual(out[1, 1].tolist(), ((100 + PALETTE[1].astype(int)) // 2).tolist())
        self.assertEqual(out[2, 2].tolist(), ((100 + PALETTE[4].astype(int)) // 2).tolist())


if __name__ == "__main__":
    unittest.main()
