"""ZCU106 board power from the 18 INA226 rail monitors: rail table and the conversion of the firmware's raw register sums.

The monitors sit behind the PCA9544 mux u60 (0x75) of PS I2C0 (zynqmp-zcu106-revA.dts): channel 0 = PS_PMBUS, channel 1 =
PL_PMBUS. The firmware (lwip_collect_results/src/power_mon.c) only accumulates raw register values per device:
    n                  samples
    sum_vbus           sum of the bus voltage register     (1.25 mV / LSB)
    sum_vshunt         sum of the shunt voltage register   (2.5 uV / LSB, signed)
    sum_prod           sum of bus register * shunt register (so the mean POWER is the mean of the product)
and this module turns them into volts / amps / watts with the shunt resistor of each rail (device tree: 5 mOhm, VCCINT 2 mOhm).

"Board power" here is the SUM OF THE MONITORED RAILS (PS + PL): it is measured after the voltage regulators, so it excludes
their conversion losses and the loads that have no monitor (fan, clocks, PHYs, ...). It is exact for differences between
two conditions on the same board (increase above idle); for the power drawn from the 12 V supply use an external meter
(collect_results.py --ext-power-*). The rail NAMES are those of the ZCU102 / ZCU106 power plan (same INA226 addresses); the
mean bus voltage is checked against the nominal rail voltage so a wrong name shows up as a warning.
"""
from __future__ import annotations

from typing import Iterable, Optional

VBUS_LSB_V = 1.25e-3
VSHUNT_LSB_V = 2.5e-6

# (channel, i2c address) -> (INA226 ref des, rail, nominal volts, shunt resistor in ohm)
RAILS = {
    (0, 0x40): ("u76", "VCCPSINTFP", 0.85, 0.005),
    (0, 0x41): ("u77", "VCCPSINTLP", 0.85, 0.005),
    (0, 0x42): ("u78", "VCCPSAUX", 1.8, 0.005),
    (0, 0x43): ("u87", "VCCPSPLL", 1.2, 0.005),
    (0, 0x44): ("u85", "MGTRAVCC", 0.85, 0.005),
    (0, 0x45): ("u86", "MGTRAVTT", 1.8, 0.005),
    (0, 0x46): ("u93", "VCCO_PSDDR_504", 1.2, 0.005),
    (0, 0x47): ("u88", "VCCOPS", 1.8, 0.005),
    (0, 0x4A): ("u15", "VCCOPS3", 3.3, 0.005),
    (0, 0x4B): ("u92", "VCCPSDDRPLL", 1.8, 0.005),
    (1, 0x40): ("u79", "VCCINT", 0.85, 0.002),
    (1, 0x41): ("u81", "VCCBRAM", 0.85, 0.005),
    (1, 0x42): ("u80", "VCCAUX", 1.8, 0.005),
    (1, 0x43): ("u84", "VCC1V2", 1.2, 0.005),
    (1, 0x44): ("u16", "VCC3V3", 3.3, 0.005),
    (1, 0x45): ("u65", "VADJ_FMC", 1.8, 0.005),
    (1, 0x46): ("u74", "MGTAVCC", 0.9, 0.005),
    (1, 0x47): ("u75", "MGTAVTT", 1.2, 0.005),
}
VOLTAGE_TOLERANCE = 0.15  # a rail whose mean bus voltage is further from nominal gets a warning


def rail_summary(dev: dict) -> Optional[dict]:
    """One device of a POWER reply ({"channel","addr","n","sum_vbus","sum_vshunt","sum_prod","errors"}) -> mean volts, amps
    and watts of that rail, or None when the device answered no sample."""
    n = dev["n"]
    if n <= 0:
        return None
    ref, rail, nominal_v, r_ohm = RAILS.get((dev["channel"], dev["addr"]), ("?", f"ch{dev['channel']}_0x{dev['addr']:02x}", None, 0.005))
    vbus = dev["sum_vbus"] / n * VBUS_LSB_V
    amps = dev["sum_vshunt"] / n * VSHUNT_LSB_V / r_ohm
    watts = dev["sum_prod"] / n * VBUS_LSB_V * VSHUNT_LSB_V / r_ohm
    ok = nominal_v is None or abs(vbus - nominal_v) <= VOLTAGE_TOLERANCE * nominal_v
    return {"rail": rail, "ref": ref, "channel": dev["channel"], "addr": dev["addr"], "n": n, "errors": dev["errors"],
            "bus_v": vbus, "current_a": amps, "power_w": watts, "nominal_v": nominal_v, "voltage_ok": ok}


def summarize(devices: Iterable[dict]) -> dict:
    """Per-rail table + the sum over all rails that answered: {"rails": [...], "board_power_w", "n_rails", "n_samples"
    (samples of the slowest rail), "missing": [(channel, addr), ...], "voltage_warnings": [rail, ...]}."""
    rails, missing = [], []
    for dev in devices:
        r = rail_summary(dev)
        if r is None:
            missing.append((dev["channel"], dev["addr"]))
        else:
            rails.append(r)
    return {"rails": rails, "board_power_w": sum(r["power_w"] for r in rails), "n_rails": len(rails),
            "n_samples": min((r["n"] for r in rails), default=0), "missing": missing,
            "voltage_warnings": [r["rail"] for r in rails if not r["voltage_ok"]]}
