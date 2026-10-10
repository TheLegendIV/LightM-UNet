"""Chain old P0,P2..P7 with the fixed P1 through rtlsim; report agreement vs ref pred and fg-Dice vs gt."""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import rtlsim_golden as rg

ap = argparse.ArgumentParser()
ap.add_argument("--old", required=True)
ap.add_argument("--new-p1", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--cases", default="0,1")
ap.add_argument("--mix", default="/tmp/chain_mixed")
ap.add_argument("--out", default="/tmp/chain_mixed_out")
ap.add_argument("--valid-pct", type=int, default=100)
ap.add_argument("--ready-pct", type=int, default=100)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--max-idle", type=int, default=3_000_000)
a = ap.parse_args()

mix = Path(a.mix)
mix.mkdir(parents=True, exist_ok=True)
for n in range(8):
    link = mix / f"GenericPartition_{n}"
    tgt = Path(a.new_p1 if n == 1 else a.old).resolve() / f"GenericPartition_{n}"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(tgt, link)
    print(f"GenericPartition_{n} -> {tgt}", flush=True)

with ThreadPoolExecutor(8) as ex:
    parts = list(ex.map(lambda n: rg.build_partition_tb(mix / f"GenericPartition_{n}"), range(8)))

d = np.load(a.ref)
ag, dr, dt = [], [], []
for i in rg.parse_range(a.cases):
    work = Path(a.out) / f"case{i}"
    work.mkdir(parents=True, exist_ok=True)
    inp = work / "in.raw"
    d["u"][i].astype(np.uint8).reshape(-1).tofile(inp)
    ref = d["pred"][i].astype(np.uint8).reshape(-1)
    gt = d["gt"][i].astype(np.uint8).reshape(-1)
    print(f"[chain] case{i}", flush=True)
    final = rg.run_chain(parts, inp, work, argparse.Namespace(
        valid_pct=a.valid_pct, ready_pct=a.ready_pct, seed=a.seed, max_idle=a.max_idle))
    pred = np.fromfile(final, np.uint8)
    assert pred.size == ref.size, (pred.size, ref.size)
    agree = float((pred == ref).mean())
    print(f"[chain] case{i}: agreement(rtl vs torch) {agree:.6f} "
          f"fg-dice rtl {rg.dice_fg(pred, gt):.4f} torch {rg.dice_fg(ref, gt):.4f} "
          f"classes rtl {np.bincount(pred, minlength=5)[:5].tolist()} "
          f"torch {np.bincount(ref, minlength=5)[:5].tolist()} gt {np.bincount(gt, minlength=5)[:5].tolist()}",
          flush=True)
    ag.append(agree)
    dr.append(rg.dice_fg(pred, gt))
    dt.append(rg.dice_fg(ref, gt))
ag, dr, dt = np.array(ag), np.array(dr), np.array(dt)
print(f"[chain] SUMMARY n={len(ag)} mean agreement {ag.mean():.6f} min {ag.min():.6f} | "
      f"mean fg-dice rtl {np.nanmean(dr):.4f} torch {np.nanmean(dt):.4f} drop {np.nanmean(dt) - np.nanmean(dr):.4f}")
ok = ag.mean() >= 0.995 and (np.nanmean(dt) - np.nanmean(dr)) <= 0.01
print("CHAIN", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
