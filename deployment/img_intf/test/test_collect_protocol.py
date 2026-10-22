import os
import sys
import tempfile
import threading
import unittest
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from img_intf.interfaces.collect_split import PowerPlan, _s64, run_split_benchmark, ticks_to_intervals  # noqa: E402
from img_intf.utils import power_report, zcu106_power  # noqa: E402
from img_intf.utils.collect_sim_server import CollectSimServer, sim_rail_amps  # noqa: E402
from img_intf.utils.stream_sim_server import process_frame  # noqa: E402


class CollectProtocolTest(unittest.TestCase):
    def _run(self, frames, bench=None, **server_kwargs):
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
            return run_split_benchmark("127.0.0.1", srv.port, frames, repeats=3, log=lambda s: None, **(bench or {})), srv
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
        self.assertIsNone(r["power"])
        self.assertEqual(srv.power_commands, [])

    def test_tick_counter_wrap(self):
        frames = [b"\x01" * 16, b"\x02" * 16]
        r, _ = self._run(frames, first_tick=2**32 - 100)
        _, full = ticks_to_intervals(r["latency_ticks"])
        self.assertTrue((full == 1696933).all())


def _expected_board_w(mode):
    return sum(volts * sim_rail_amps(name, mode) for _ref, name, volts, _r in zcu106_power.RAILS.values())


class PowerProtocolTest(CollectProtocolTest):
    FRAMES = [bytes([i]) * 64 for i in range(5)]
    PASS_TICKS = 1696933 + 4 * 460000

    def test_power_windows_and_arithmetic(self):
        r, srv = self._run(self.FRAMES, bench={"power": PowerPlan(idle_s=3, warmup_s=1, load_s=4)})
        self.assertIsNone(srv.error)
        self.assertEqual(srv.power_commands, [(0, 0, 3), (1, 0, 1), (1, 0, 4), (0, 0, 3)])   # warm-up chunk, then the measured chunk
        self.assertEqual(r["outputs"], [process_frame(f) for f in self.FRAMES])      # the power windows leave the FETCH intact
        s = {k: zcu106_power.summarize(r["power"][k]["devices"]) for k in ("idle", "load", "idle_after")}
        for k in s:
            self.assertEqual(s[k]["n_rails"], 18)
            self.assertEqual(s[k]["voltage_warnings"], [])
        self.assertAlmostEqual(s["idle"]["board_power_w"], _expected_board_w(0), delta=0.005 * _expected_board_w(0))
        self.assertAlmostEqual(s["idle_after"]["board_power_w"], _expected_board_w(0), delta=0.005 * _expected_board_w(0))
        self.assertAlmostEqual(s["load"]["board_power_w"], _expected_board_w(1), delta=0.005 * _expected_board_w(1))
        load = r["power"]["load"]
        self.assertGreater(load["passes"], 0)
        self.assertEqual(load["images"], load["passes"] * 5)
        self.assertEqual(load["ticks_sum"], load["passes"] * self.PASS_TICKS)
        self.assertEqual(r["power"]["idle"]["images"], 0)

    def test_long_windows_are_chunked_and_merged(self):
        """No single POWER command is longer than chunk_s (the board's main loop is blocked for its whole length); the
        chunks are merged into one window with summed samples / passes / ticks."""
        plan = PowerPlan(idle_s=5, warmup_s=3, load_s=7, idle_after=False, chunk_s=2)
        r, srv = self._run(self.FRAMES, bench={"power": plan})
        self.assertIsNone(srv.error)
        self.assertEqual(srv.power_commands, [(0, 0, 2), (0, 0, 2), (0, 0, 1),            # idle 5 = 2 + 2 + 1
                                              (1, 0, 2), (1, 0, 1),                       # warm-up 3 (not measured)
                                              (1, 0, 2), (1, 0, 2), (1, 0, 2), (1, 0, 1)])   # load 7
        self.assertTrue(all(w <= 2 for _m, _wu, w in srv.power_commands))
        self.assertAlmostEqual(r["power"]["idle"]["window_s"], 5.0)
        self.assertAlmostEqual(r["power"]["load"]["window_s"], 7.0)
        self.assertEqual(r["power"]["load"]["sweeps"], 7 * 50)
        self.assertEqual(r["power"]["load"]["ticks_sum"], r["power"]["load"]["passes"] * self.PASS_TICKS)
        s = zcu106_power.summarize(r["power"]["load"]["devices"])
        self.assertEqual(s["n_rails"], 18)
        self.assertEqual(s["n_samples"], 7 * 50)
        self.assertAlmostEqual(s["board_power_w"], _expected_board_w(1), delta=0.005 * _expected_board_w(1))

    def test_power_tables_and_csv_upsert(self):
        r, _ = self._run(self.FRAMES, bench={"power": PowerPlan(idle_s=2, warmup_s=1, load_s=3, idle_after=False)})
        rows, rail_rows = power_report.power_tables("m", "board", r["power"], "ts", log=lambda s: None)
        by = {x["condition"]: x for x in rows}
        self.assertEqual(set(by), {power_report.COND_IDLE, power_report.COND_LOAD})
        idle, load = by[power_report.COND_IDLE], by[power_report.COND_LOAD]
        self.assertEqual(idle["increase_above_idle_w"], 0.0)
        self.assertAlmostEqual(load["increase_above_idle_w"], load["board_power_w"] - idle["board_power_w"], places=3)
        fps = 5 / (self.PASS_TICKS / (power_report.PL_CLOCK_MHZ * 1e6))
        self.assertAlmostEqual(load["throughput_fps"], fps, delta=0.01)
        self.assertAlmostEqual(load["energy_per_image_mj"], load["board_power_w"] / fps * 1000, delta=0.01)
        self.assertAlmostEqual(load["dynamic_energy_per_image_mj"], load["increase_above_idle_w"] / fps * 1000, delta=0.01)
        self.assertEqual(len(rail_rows), 18 * 2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "power_results.csv"
            for _ in range(2):  # idempotent: re-running replaces the rows of the same (model, config, condition, source)
                power_report.upsert_csv(path, power_report.POWER_COLUMNS, power_report.POWER_KEY, rows)
            other, _ = power_report.power_tables("other", "board", r["power"], "ts", log=lambda s: None)
            power_report.upsert_csv(path, power_report.POWER_COLUMNS, power_report.POWER_KEY, other)
            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1 + 2 * 2)
            self.assertEqual(lines[0].split(",")[:4], ["model_name", "config", "condition", "source"])

    def test_external_meter_rows(self):
        idle, load = power_report.external_power_rows("m", "board", 10.0, 14.0, 250.0, "ts")
        self.assertEqual((idle["increase_above_idle_w"], load["increase_above_idle_w"]), (0.0, 4.0))
        self.assertAlmostEqual(load["energy_per_image_mj"], 14.0 / 250.0 * 1000)
        self.assertAlmostEqual(load["dynamic_energy_per_image_mj"], 4.0 / 250.0 * 1000)
        self.assertEqual(load["source"], power_report.SRC_EXTERNAL)

    def test_missing_rail_is_reported_and_dead_bus_fails(self):
        warnings = []
        r, _ = self._run(self.FRAMES, bench={"power": PowerPlan(idle_s=2, warmup_s=1, load_s=2, idle_after=False)},
                         absent_rails={(1, 0x40)})
        s = zcu106_power.summarize(r["power"]["idle"]["devices"])
        self.assertEqual((s["n_rails"], s["missing"]), (17, [(1, 0x40)]))
        rows, _ = power_report.power_tables("m", "board", r["power"], "ts", log=warnings.append)
        self.assertTrue(any("did not answer" in w for w in warnings))
        self.assertEqual(rows[0]["n_rails"], 17)

        r, _ = self._run(self.FRAMES, bench={"power": PowerPlan(idle_s=2, warmup_s=1, load_s=2)}, power_bus_dead=True)
        with self.assertRaises(RuntimeError):
            power_report.power_tables("m", "board", r["power"], "ts", log=lambda s: None)

    def test_plan_validation_and_signed_sums(self):
        for bad in (dict(idle_s=0), dict(load_s=0), dict(warmup_s=90, load_s=20), dict(idle_s=101)):
            with self.assertRaises(ValueError):
                PowerPlan(**bad)
        self.assertEqual(_s64(0xFFFFFFFF, 0xFFFFFFFE), -2)
        self.assertEqual(_s64(0, 7), 7)


class RailMathTest(unittest.TestCase):
    def test_conversion(self):
        # VCCINT: 2 mOhm shunt. 1000 samples of bus 0.85 V (680 LSB) and 4 A (4 A * 2 mOhm = 8 mV = 3200 LSB)
        dev = {"channel": 1, "addr": 0x40, "n": 1000, "sum_vbus": 680 * 1000, "sum_vshunt": 3200 * 1000,
               "sum_prod": 680 * 3200 * 1000, "errors": 0}
        r = zcu106_power.rail_summary(dev)
        self.assertEqual(r["rail"], "VCCINT")
        self.assertAlmostEqual(r["bus_v"], 0.85)
        self.assertAlmostEqual(r["current_a"], 4.0)
        self.assertAlmostEqual(r["power_w"], 3.4)
        self.assertTrue(r["voltage_ok"])

    def test_mean_of_product_not_product_of_means(self):
        # two samples: (V 0.9, I 8 A) and (V 1.1, I 0 A) -> mean power 0.5 * 0.9 * 8 = 3.6 W, while mean V * mean I = 1.0 * 4 = 4 W
        vb_hi, vb_lo, vs = round(0.9 / 1.25e-3), round(1.1 / 1.25e-3), round(0.04 / 2.5e-6)   # 8 A * 5 mOhm = 40 mV
        dev = {"channel": 0, "addr": 0x42, "n": 2, "sum_vbus": vb_hi + vb_lo, "sum_vshunt": vs, "sum_prod": vb_hi * vs, "errors": 0}
        r = zcu106_power.rail_summary(dev)
        self.assertAlmostEqual(r["power_w"], 0.5 * 0.9 * 8.0, places=3)

    def test_no_samples_and_voltage_warning(self):
        self.assertIsNone(zcu106_power.rail_summary({"channel": 0, "addr": 0x40, "n": 0, "sum_vbus": 0, "sum_vshunt": 0,
                                                     "sum_prod": 0, "errors": 3}))
        wrong = {"channel": 1, "addr": 0x40, "n": 1, "sum_vbus": round(1.8 / 1.25e-3), "sum_vshunt": 0, "sum_prod": 0, "errors": 0}
        self.assertFalse(zcu106_power.rail_summary(wrong)["voltage_ok"])


if __name__ == "__main__":
    unittest.main()
