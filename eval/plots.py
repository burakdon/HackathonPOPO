"""Static figures for the README (matplotlib PNGs).

Visual conventions: one categorical hue per policy in a fixed order (validated palette), thin
2 px lines, recessive gridlines and axes, text in neutral ink (never the series colour), one
y-axis per panel, small multiples instead of dual axes.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order
SERIES_LIGHT = "#86b6ef"   # lighter step of series 1 (individual seeds behind the mean)


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def read_curve(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def plot_learning_curves(curve_paths: list[Path], out: Path, title: str) -> None:
    """Small multiples (one metric per panel) of validation metrics vs training steps.
    Thin light lines = individual seeds, bold line = mean over seeds."""
    curves = [read_curve(p) for p in curve_paths]
    panels = [("return", "Proxy reward per session", "return"),
              ("time_in_band", "Time in band (75-90% rolling accuracy)", "fraction of trials"),
              ("control_error", "True control error |p - 0.85| (sim-only)", "mean |p - 0.85|"),
              ("learning_gain", "Learning gain (true objective, sim-only)", "points x 100")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), facecolor=SURFACE)
    n = min(len(c["timesteps"]) for c in curves)
    steps = curves[0]["timesteps"][:n] / 1e6
    for ax, (key, name, ylabel) in zip(axes.flat, panels):
        scale = 100.0 if key == "learning_gain" else 1.0
        ys = np.array([c[key][:n] * scale for c in curves])
        for y in ys:
            ax.plot(steps, y, color=SERIES_LIGHT, linewidth=1.0)
        ax.plot(steps, ys.mean(axis=0), color=SERIES[0], linewidth=2.0)
        _style(ax, name, ylabel)
        ax.set_xlabel("training steps (millions)", fontsize=9, color=INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, f"Validation patients (held out from training), {len(curves)} seeds: "
             "light lines = individual seeds, dark line = mean.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_variant_curves(variants: dict[str, list[Path]], out: Path, title: str) -> None:
    """Small multiples of validation metrics vs training steps, one colour per reward variant
    (fixed categorical order). Thin faint lines = seeds, bold line = mean. Proxy return is not shown
    because returns under different rewards are not comparable."""
    panels = [("time_in_band", "Time in band (observable)", "fraction of trials"),
              ("control_error", "True control error |p - 0.85| (sim-only)", "mean |p - 0.85|"),
              ("learning_gain", "Learning gain (sim-only)", "points x 100"),
              ("probe_lift", "Probe lift: recall gain on review items (sim-only)", "points x 100")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), facecolor=SURFACE)
    for ax, (key, name, ylabel) in zip(axes.flat, panels):
        scale = 100.0 if key in ("learning_gain", "probe_lift") else 1.0
        ends = []
        for color, (label, paths) in zip(SERIES, variants.items()):
            curves = [read_curve(p) for p in paths]
            n = min(len(c["timesteps"]) for c in curves)
            steps = curves[0]["timesteps"][:n] / 1e6
            ys = np.array([c[key][:n] * scale for c in curves])
            for y in ys:
                ax.plot(steps, y, color=color, linewidth=0.8, alpha=0.35)
            mean = ys.mean(axis=0)
            ax.plot(steps, mean, color=color, linewidth=2.0, label=label)
            ends.append((mean[-1], label, steps[-1]))
        if key == "time_in_band":   # direct labels on one panel, nudged apart; the legend covers the rest
            lo, hi = ax.get_ylim()
            gap, placed = 0.06 * (hi - lo), []
            for y, label, x in sorted(ends):
                y_text = max(y, placed[-1] + gap) if placed else y
                placed.append(y_text)
                ax.annotate(label, (x, y), xytext=(x + 0.04, y_text), fontsize=7.5, color=INK_2, va="center",
                            annotation_clip=False)
            ax.set_xlim(right=steps[-1] * 1.42)
            ax.set_xticks(np.arange(0, steps[-1] + 0.01, 0.5))
        _style(ax, name, ylabel)
        ax.set_xlabel("training steps (millions)", fontsize=9, color=INK_2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Held-out validation patients; 3 seeds per variant (faint lines), mean in bold.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


VARIANT_LABELS = {"full": "proxy reward", "full_probe": "proxy + probe reward", "privileged": "privileged reward"}


def main() -> None:
    import argparse
    from core.config import load_yaml, resolve
    cfg = load_yaml("configs/ppo.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg["seeds"])
    ap.add_argument("--out", default="results/learning_curves.png")
    args = ap.parse_args()
    runs = resolve(cfg["output_dir"])
    variants = {}
    for v, label in VARIANT_LABELS.items():
        tag = "" if v == "full" else f"{v}_"
        paths = [runs / f"{tag}seed_{s}" / "val_curve.csv" for s in args.seeds]
        if all(p.exists() for p in paths):
            variants[label] = paths
    plot_variant_curves(variants, resolve(args.out), "MaskablePPO learning curves by training reward")
    # Commit the combined curve data next to the figure so it can be re-plotted without retraining.
    rows = []
    for label, paths in variants.items():
        for s, p in zip(args.seeds, paths):
            c = read_curve(p)
            for i in range(len(c["timesteps"])):
                rows.append({"variant": label, "seed": s, **{k: float(v[i]) for k, v in c.items()}})
    csv_out = resolve(args.out).with_suffix(".csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {resolve(args.out)} and {csv_out}")


if __name__ == "__main__":
    main()
