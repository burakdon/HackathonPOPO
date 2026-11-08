"""Fit the SP model to published patient proportions and compare with Dell's own fits.

Builds (or loads) the cached response-probability grid, fits every row of
data/calibration/dell2004_table2_sp_fits.csv, and prints our (s, p) next to Dell's.

Usage:  uv run python -m scripts.fit_dell_sp [--csv PATH] [--items 175]
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from patientmodel.dell_sp import NAMING_CATEGORIES, DellSPModel, load_dell_sp_config
from patientmodel.fit import counts_from_proportions, fit_counts, load_or_build_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/calibration/dell2004_table2_sp_fits.csv")
    ap.add_argument("--items", type=int, default=175, help="PNT items per patient (counts are rebuilt from proportions)")
    args = ap.parse_args()

    cfg, fit_cfg = load_dell_sp_config()
    model = DellSPModel(cfg)
    t0 = time.time()
    table = load_or_build_grid(model, fit_cfg["grid_step"], fit_cfg["trials_per_cell"], fit_cfg["seed"])
    print(f"grid {table.probs.shape[0]}x{table.probs.shape[1]} cells, {table.trials_per_cell} trials each ({time.time() - t0:.0f} s)\n")

    rows = [r for r in csv.DictReader(l for l in open(args.csv) if not l.startswith("#"))]
    print(f"{'patient':12s} {'Dell s':>7s} {'Dell p':>7s} | {'our s':>6s} {'our p':>6s}  {'s 95%':>14s}  {'p 95%':>14s}  rmsd  correct obs/pred")
    ds, dp = [], []
    for r in rows:
        obs = {k: float(r[k]) for k in NAMING_CATEGORIES}
        counts = counts_from_proportions(obs, args.items)
        fit = fit_counts(counts, table, fit_cfg["probability_floor"])
        tot = sum(obs.values())
        ds.append(fit.s - float(r["s"])); dp.append(fit.p - float(r["p"]))
        print(f"{r['patient']:12s} {float(r['s']):7.3f} {float(r['p']):7.3f} | {fit.s:6.3f} {fit.p:6.3f}  "
              f"[{fit.s_interval[0]:.3f}, {fit.s_interval[1]:.3f}]  [{fit.p_interval[0]:.3f}, {fit.p_interval[1]:.3f}]  "
              f"{fit.rmsd:.3f}  {obs['correct'] / tot:.2f}/{fit.predicted['correct']:.2f}")
    print(f"\nmean |delta s| = {np.mean(np.abs(ds)):.4f}   mean |delta p| = {np.mean(np.abs(dp)):.4f}   (grid step {fit_cfg['grid_step']})")


if __name__ == "__main__":
    main()
