"""Board power rows for deployment/collect_results.py --power: the POWER replies of the board -> power_results.csv / power_rails.csv.

Pure functions + CSV helpers (no board, no torch), so they are unit-tested on their own (img_intf/test/test_collect_protocol.py).
Definitions (also in collect_results.py's docstring):
  board_power_w                 sum of the monitored INA226 rails (see zcu106_power.py), per window
  increase_above_idle_w         board_power_w - board_power_w of the Idle window
  throughput_fps                images / (sum of the PL-tick pass times) of the passes inside the measured inference window
  energy_per_image_mj           board_power_w / throughput_fps; dynamic_energy_per_image_mj uses the increase above idle
"""
from __future__ import annotations

import csv
from pathlib import Path

from img_intf.utils import zcu106_power

PL_CLOCK_MHZ = 187.481262  # accelerator clock of the current bitstream (PL ticks -> seconds)

POWER_COLUMNS = ["model_name", "config", "condition", "source", "board_power_w", "increase_above_idle_w", "throughput_fps",
                 "energy_per_image_mj", "dynamic_energy_per_image_mj", "n_rails", "n_samples", "window_s", "passes", "timestamp"]
POWER_KEY = ("model_name", "config", "condition", "source")
POWER_RAIL_COLUMNS = ["model_name", "config", "condition", "rail", "ref", "channel", "addr", "bus_v", "current_a", "power_w",
                      "n_samples", "errors", "voltage_ok"]
POWER_RAIL_KEY = ("model_name", "config", "condition", "rail")
COND_IDLE = "Idle"
COND_LOAD = "Inference (back-to-back)"
COND_IDLE_AFTER = "Idle (after inference)"
SRC_RAILS = "ina226_rails"
SRC_EXTERNAL = "external_meter"


def upsert_csv(path: Path, columns: list, key: tuple, new_rows: list) -> None:
    """Replace the rows whose `key` columns equal those of a new row (idempotent re-runs), keep everything else."""
    new_keys = {tuple(str(r[k]) for k in key) for r in new_rows}
    rows = []
    if path.is_file():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if tuple(r.get(k) for k in key) not in new_keys]
    rows.extend({c: ("" if r.get(c) is None else r[c]) for c in columns} for r in new_rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def power_tables(model_name: str, config: str, power: dict, timestamp: str, pl_clock_mhz: float = PL_CLOCK_MHZ,
                 log=print) -> tuple[list, list]:
    """The board's POWER replies (idle / inference / idle after, raw INA226 sums) -> (power_results rows, per-rail rows).
    Raises RuntimeError when the monitors did not answer or the inference window completed no pass."""
    windows = [(COND_IDLE, power["idle"]), (COND_LOAD, power["load"])]
    if power.get("idle_after"):
        windows.append((COND_IDLE_AFTER, power["idle_after"]))
    summaries = {cond: zcu106_power.summarize(w["devices"]) for cond, w in windows}
    for cond, s in summaries.items():
        if s["n_rails"] == 0:
            raise RuntimeError(f"[{cond}] no power monitor answered: PS I2C0 / the 0x75 mux is not reachable (see the "
                               "board's UART log) -- the firmware needs the POWER build of lwip_collect_results")
        if s["missing"]:
            log(f"[WARN] {cond}: {len(s['missing'])} monitor(s) did not answer: " + ", ".join(f"ch{c}/0x{a:02x}" for c, a in s["missing"]))
        if s["voltage_warnings"]:
            log(f"[WARN] {cond}: bus voltage far from the nominal rail voltage on {s['voltage_warnings']} -- is the rail table "
                "(img_intf/utils/zcu106_power.py) right for this board?")
    load = power["load"]
    if load["passes"] == 0:
        raise RuntimeError("the inference window completed no pass of the split (--power-load-s shorter than one pass?)")
    fps = load["images"] / (load["ticks_sum"] / (pl_clock_mhz * 1e6))
    p_idle = summaries[COND_IDLE]["board_power_w"]

    rows, rail_rows = [], []
    for cond, w in windows:
        s = summaries[cond]
        p = s["board_power_w"]
        inc = p - p_idle
        row = {"model_name": model_name, "config": config, "condition": cond, "source": SRC_RAILS,
               "board_power_w": round(p, 4), "increase_above_idle_w": round(inc, 4), "n_rails": s["n_rails"],
               "n_samples": s["n_samples"], "window_s": round(w["window_s"], 2), "timestamp": timestamp}
        if cond == COND_LOAD:
            row.update({"throughput_fps": round(fps, 2), "energy_per_image_mj": round(p / fps * 1000, 3),
                        "dynamic_energy_per_image_mj": round(inc / fps * 1000, 3), "passes": load["passes"]})
        rows.append(row)
        for r in s["rails"]:
            rail_rows.append({"model_name": model_name, "config": config, "condition": cond, "rail": r["rail"], "ref": r["ref"],
                              "channel": r["channel"], "addr": f"0x{r['addr']:02x}", "bus_v": round(r["bus_v"], 4),
                              "current_a": round(r["current_a"], 4), "power_w": round(r["power_w"], 5),
                              "n_samples": r["n"], "errors": r["errors"], "voltage_ok": int(r["voltage_ok"])})
    return rows, rail_rows


def external_power_rows(model_name: str, config: str, idle_w: float, load_w: float, fps: float, timestamp: str) -> list:
    """Rows for a plug / 12 V input meter read by hand (same conditions, throughput from the same run)."""
    base = {"model_name": model_name, "config": config, "source": SRC_EXTERNAL, "timestamp": timestamp}
    return [
        {**base, "condition": COND_IDLE, "board_power_w": round(idle_w, 3), "increase_above_idle_w": 0.0},
        {**base, "condition": COND_LOAD, "board_power_w": round(load_w, 3), "increase_above_idle_w": round(load_w - idle_w, 3),
         "throughput_fps": round(fps, 2), "energy_per_image_mj": round(load_w / fps * 1000, 3),
         "dynamic_energy_per_image_mj": round((load_w - idle_w) / fps * 1000, 3)},
    ]


def print_rail_table(rail_rows: list, log=print) -> None:
    """Per rail: bus voltage, idle and inference power (the rails behind the board_power_w sum)."""
    by = {}
    for r in rail_rows:
        by.setdefault(r["rail"], {})[r["condition"]] = r
    log(f"[POWER] {'Rail':16s} {'Bus V':>7s} {'Idle W':>8s} {'Inference W':>12s} {'Delta W':>8s}")
    for rail, c in by.items():
        idle, load = c.get(COND_IDLE), c.get(COND_LOAD)
        if idle and load:
            log(f"[POWER] {rail:16s} {float(load['bus_v']):>7.3f} {float(idle['power_w']):>8.3f} {float(load['power_w']):>12.3f} "
                f"{float(load['power_w']) - float(idle['power_w']):>+8.3f}" + ("" if int(load["voltage_ok"]) else "   <- voltage off nominal"))


def print_power_table(rows: list, log=print) -> None:
    log(f"[POWER] {'Condition':26s} {'Source':14s} {'Board power':>12s} {'Above idle':>11s} {'Throughput':>12s} {'Energy/image':>13s}")
    for r in rows:
        fps, e = r.get("throughput_fps"), r.get("energy_per_image_mj")
        log(f"[POWER] {r['condition']:26s} {r['source']:14s} {r['board_power_w']:>10.3f} W {r['increase_above_idle_w']:>+9.3f} W "
            + (f"{fps:>8.1f} fps {e:>9.2f} mJ" if fps else f"{'-':>12s} {'-':>13s}"))
