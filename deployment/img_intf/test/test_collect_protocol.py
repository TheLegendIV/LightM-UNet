import os
import sys
import threading
import unittest
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from img_intf.interfaces.collect_split import run_split_benchmark, ticks_to_intervals  # noqa: E402
from img_intf.utils.collect_sim_server import CollectSimServer  # noqa: E402
from img_intf.utils.stream_sim_server import process_frame  # noqa: E402


class CollectProtocolTest(unittest.TestCase):
    def _run(self, frames, **server_kwargs):
        loop = asyncio.new_event_loop()
        srv = CollectSimServer(payload_length=len(frames[0]), **server_kwargs)
        ready = threading.Event()

        def serve():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(srv.start())
            ready.set()
            loop.run_forever()

        threading.Thread(target=serve, daemon=True).start()
        ready.wait(5)
        try:
            return run_split_benchmark("127.0.0.1", srv.port, frames, repeats=3, log=lambda s: None), srv
        finally:
            loop.call_soon_threadsafe(loop.stop)

    def test_outputs_and_ticks(self):
        frames = [bytes([i]) * 64 for i in range(5)]
        r, srv = self._run(frames)
        self.assertEqual(r["outputs"], [process_frame(f) for f in frames])
        first, full = ticks_to_intervals(r["latency_ticks"])
        self.assertEqual(len(full), 5)
        self.assertTrue((full == 1696933).all())
        _, run_full = ticks_to_intervals(r["run_ticks"])
        self.assertEqual(len(run_full), 3)
        self.assertTrue((run_full == 1696933 + 4 * 460000).all())
        self.assertIsNone(srv.error)

    def test_tick_counter_wrap(self):
        frames = [b"\x01" * 16, b"\x02" * 16]
        r, _ = self._run(frames, first_tick=2**32 - 100)
        _, full = ticks_to_intervals(r["latency_ticks"])
        self.assertTrue((full == 1696933).all())


if __name__ == "__main__":
    unittest.main()
