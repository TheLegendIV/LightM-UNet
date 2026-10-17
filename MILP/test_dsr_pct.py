"""DSR feasibility on the 512 net (there is no floor). Run in lightmunet_dev:  cd MILP && python3 test_dsr_pct.py

S12 dense nearest-upsample noconv 512x512 (config_12_dense_relu_nearest_upsample), uniform INT6, 180 fps at 100 MHz (555,555 cycles/frame: the 512x512 LabelSelect floor is 524,288 cycles = 190.7 fps, so 250 fps
is infeasible by itself), mvau-wwidth-max 96, min-resources. There is no exemption for nodes that cannot
satisfy the rule, so a small pct is INFEASIBLE: regular5.0.conv (1 -> 1, 3x3) tops out at 196,608 cycles within F (its next fold is 590,339) and `final` cannot go below ~263,171 cycles (1 cycle per
512x512 output pixel), hence the chain conv -> expand.0 -> final needs 1 + pct >= 263,171 / 196,608 (pct >= ~34 %). 25 % is infeasible, 50 % is feasible, and so is DSR off (50 % takes a faster fold on the dilated convs, see the test).
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SENS = HERE / "artifacts" / "layer_sensitivity_12_dense_relu_nearest_conv_upsample_256.json"


def solve(tmp: Path, name: str, *dsr: str) -> dict:
    out = tmp / f"{name}.json"
    cmd = [sys.executable, str(HERE / "finn_milp.py"), "--config", "config_12_dense_relu_nearest_upsample", "--sensitivity-file", str(SENS), "--candidate-bits", "6", "--force-dsp", "--min-resources",
           "--max-lut-fraction", "1.0", "--max-bram-fraction", "1.0", "--max-dsp-fraction", "1.0", "--max-uram-fraction", "1.0", "--target-fps", "180", "--clock-mhz", "100",
           "--mvau-wwidth-max", "96", "--time-limit", "300", "--gap-rel", "0.001", "--out-file", str(out), *dsr]
    res = subprocess.run(cmd, cwd=HERE.parent, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stderr[-1500:]
    return json.loads(out.read_text())


class TestDsrPct(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.r = {p: solve(t, f"p{p}", "--dsr-pct", p) for p in ("1", "25", "50")}
        cls.off = solve(t, "off", "--dsr-pct", "none")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_small_pct_is_infeasible(self):
        self.assertEqual(self.r["1"]["status"], "Infeasible")
        self.assertEqual(self.r["25"]["status"], "Infeasible")

    def test_50_percent_is_feasible_and_so_is_off(self):
        # At the old 250 FPS budget the 50 % rule differed from OFF only in the dilated 3x3 convs. The reachable 512x512 budget is now 180 FPS (the LabelSelect floor), where min-resources
        # picks slower folds and the rule changes other nodes too (in both directions), so only feasibility is asserted; the rule itself is checked edge by edge in test_argmax_dsr.py.
        self.assertEqual(self.r["50"]["status"], "Optimal")
        self.assertEqual(self.off["status"], "Optimal")

    def test_final_layer_cannot_beat_one_cycle_per_output_pixel(self):
        self.assertGreaterEqual(self.off["per_layer"]["final"]["cycles"], 512 * 512)


if __name__ == "__main__":
    unittest.main()
