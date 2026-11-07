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


def _legend_top(fig, handles, labels, ncol=3) -> None:
    fig.legend(handles, labels, loc="upper right", ncol=ncol, frameon=False, fontsize=8.5, labelcolor=INK_2)


def plot_comparison_bars(cfg: dict, csv_path: Path, out: Path) -> None:
    """Grouped bars (cohort groups, one bar per policy) with 95% bootstrap CI whiskers, one panel per
    metric. Lower is better for control error and quit rate; higher for learning gain and probe lift."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    get = {(r["cohort"], r["policy"], r["metric"]): r for r in rows}
    panels = [("control_error", "True control error |p - 0.85| (lower is better)", 1.0),
              ("learning_gain", "Learning gain over the item bank, x100 (higher is better)", 100.0),
              ("probe_lift", "Probe lift: recall gain on review items, x100 (higher is better)", 100.0),
              ("quit", "Sessions abandoned (lower is better)", 1.0)]
    cohorts, policies = cfg["cohorts"], cfg["policies"]
    n = len(policies)
    width = 0.8 / n
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURFACE)
    for ax, (metric, title, sc) in zip(axes.flat, panels):
        for j, (p, color) in enumerate(zip(policies, SERIES + ["#52514e"])):
            xs = np.arange(len(cohorts)) - 0.4 + width * (j + 0.5)
            means = np.array([sc * float(get[(c, p["key"], metric)]["mean"]) for c in cohorts])
            lo = np.array([sc * float(get[(c, p["key"], metric)]["ci_low"]) for c in cohorts])
            hi = np.array([sc * float(get[(c, p["key"], metric)]["ci_high"]) for c in cohorts])
            ax.bar(xs, means, width=width * 0.88, color=color, label=p["label"], edgecolor=SURFACE, linewidth=1.0)
            ax.errorbar(xs, means, yerr=[means - lo, hi - means], fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=1.5)
        ax.set_xticks(np.arange(len(cohorts)))
        ax.set_xticklabels([c.replace("_", " ") for c in cohorts], fontsize=8.5, color=INK_2)
        _style(ax, title, "")
        ax.axhline(0, color=AXIS, linewidth=0.8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    _legend_top(fig, handles, labels)
    fig.suptitle("Policy comparison on held-out simulated patients", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, f"{cfg['n_patients']} held-out patients per cohort, mean with 95% bootstrap CI; "
             "PPO averaged over 3 training seeds per patient; the oracle reads the hidden patient state.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_rolling_accuracy(cfg: dict, ev: dict, out: Path) -> None:
    """Rolling accuracy (last `window` trials) vs trial number, mean over patients (and seeds) with a
    light 95% band, the 75-90% target band shaded. Sessions that ended early drop out of later trials."""
    cohorts, policies, W = cfg["cohorts"], cfg["policies"], ev["window"]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(12, 4.2), facecolor=SURFACE, sharey=True)
    x = np.arange(W, ev["n_trials"] + 1)
    for ax, cohort in zip(np.atleast_1d(axes), cohorts):
        ax.axhspan(0.75, 0.90, color=GRID, alpha=0.6, linewidth=0, label="target band 75-90%")
        for p, color in zip(policies, SERIES + ["#52514e"]):
            curves = ev["curves"][(cohort, p["key"])]
            mean = np.nanmean(curves, axis=0)
            n = np.sum(~np.isnan(curves), axis=0)
            se = np.nanstd(curves, axis=0) / np.sqrt(np.maximum(n, 1))
            ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, mean, color=color, linewidth=2.0, label=p["label"])
        _style(ax, cohort.replace("_", " "), "rolling accuracy (last 8 trials)" if ax is np.atleast_1d(axes)[0] else "")
        ax.set_xlabel("trial", fontsize=9, color=INK_2)
        ax.set_ylim(0.2, 1.02)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    _legend_top(fig, handles, labels, ncol=4)
    fig.suptitle("Rolling accuracy over the session", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, f"{cfg['n_patients']} held-out patients per cohort; lines = mean, light bands = 95% CI; "
             "includes review probes; sessions that were abandoned drop out of later trials.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.82))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_learning_peak_ablation(rows: list[dict], cfg: dict, out: Path) -> None:
    """Learning gain vs the simulator's learning-peak assumption, one panel per cohort, one line per
    policy (fixed categorical order), 95% CI whiskers."""
    rows = [r for r in rows if r["ablation"] == "learning_peak" and r["metric"] == "learning_gain"]
    peaks = sorted({float(r["setting"]) for r in rows})
    pols = list(dict.fromkeys((r["policy"], r["label"]) for r in rows))
    cohorts = cfg["cohorts"]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(12, 4.3), facecolor=SURFACE)
    colors = SERIES + ["#52514e", "#0b0b0b"]   # matched PPO variants in neutral inks, dashed / dotted
    for ax, cohort in zip(np.atleast_1d(axes), cohorts):
        for (key, label), color in zip(pols, colors):
            pts = sorted(((float(r["setting"]), r) for r in rows if r["cohort"] == cohort and r["policy"] == key),
                         key=lambda t: t[0])
            if not pts:
                continue
            x = [p for p, _ in pts]
            y = np.array([100 * r["mean"] for _, r in pts])
            lo = np.array([100 * r["ci_low"] for _, r in pts])
            hi = np.array([100 * r["ci_high"] for _, r in pts])
            style = {"ppo_sim_matched": "--", "ppo_learning_matched": ":"}.get(key, "-")
            ax.errorbar(x, y, yerr=[y - lo, hi - y], color=color, linestyle=style, linewidth=2.0, marker="o",
                        markersize=5, capsize=2, label=label)
        _style(ax, cohort.replace("_", " "), "learning gain x100" if ax is np.atleast_1d(axes)[0] else "")
        ax.set_xticks(peaks)
        ax.set_xlabel("simulator learning peak (p at which practice teaches most)", fontsize=8.5, color=INK_2)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    _legend_top(fig, handles, labels, ncol=3)
    fig.suptitle("Ablation: where does learning peak?", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, f"{cfg['n_patients']} held-out patients per cohort, mean with 95% bootstrap CI. Dashed / "
             "dotted: PPO retrained at each peak on the simulation objective / the learning-only objective.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.8))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_pitch(cfg: dict, csv_path: Path, out: Path) -> None:
    """Slide figure: one cohort, a few policies, two panels (sessions abandoned; learning gain), large
    fonts, mean with 95% CI and the value printed on each bar. Our agent in the accent colour,
    baselines in neutral grey."""
    pc = cfg["pitch"]
    with open(csv_path) as f:
        rows = {(r["cohort"], r["policy"], r["metric"]): r for r in csv.DictReader(f)}
    keys = [p["key"] for p in pc["policies"]]
    labels = [p["label"] for p in pc["policies"]]
    colors = ["#b5b3ab"] * (len(keys) - 1) + [SERIES[0]]
    panels = [("quit", "Sessions abandoned", 100.0, "%", "{:.0f}%"),
              ("learning_gain", "Naming improvement per session", 100.0, "percentage points", "+{:.1f}")]
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), facecolor=SURFACE)
    for ax, (metric, title, sc, unit, fmt) in zip(axes, panels):
        m = np.array([sc * float(rows[(pc["cohort"], k, metric)]["mean"]) for k in keys])
        lo = np.array([sc * float(rows[(pc["cohort"], k, metric)]["ci_low"]) for k in keys])
        hi = np.array([sc * float(rows[(pc["cohort"], k, metric)]["ci_high"]) for k in keys])
        x = np.arange(len(keys))
        ax.bar(x, m, width=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK_2, elinewidth=1.4, capsize=5)
        for xi, (v, h) in enumerate(zip(m, hi)):
            ax.text(xi, h + 0.03 * max(hi), fmt.format(v), ha="center", va="bottom", fontsize=20, color=INK,
                    fontweight="bold" if xi == len(keys) - 1 else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=16, color=INK)
        ax.set_ylim(0, max(hi) * 1.25)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=19, color=INK, pad=12)
        ax.set_ylabel(unit, fontsize=13, color=INK_2)
        ax.tick_params(axis="y", colors=MUTED, labelsize=12)
        ax.tick_params(axis="x", length=0)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
    fig.suptitle("Severe aphasia (simulated patients)", x=0.01, ha="left", fontsize=22, color=INK)
    fig.text(0.01, 0.01, f"{cfg['n_patients']} held-out simulated patients; mean with 95% CI. Naming improvement = increase "
             "in the probability of naming a word without a hint, averaged over the word list. Simulation only, not a "
             "clinical result.", fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


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
