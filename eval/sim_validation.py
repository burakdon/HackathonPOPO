"""Simulator validation: what the simulated cohorts look like on the classic picture-naming effects.

  uv run python -m eval.sim_validation

Panels: (a) uncued accuracy vs word frequency, (b) immediate effect of each cue type, (c) accuracy
over a session from fatigue alone (learning switched off), (d) accuracy by tier per cohort. Published
calibration ranges (data/calibration_targets.yaml, if present) are drawn where they apply.
Writes results/simulator_validation.png and results/simulator_validation.csv.
"""
from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.config import load_clinical, load_yaml, resolve  # noqa: E402
from eval.plots import AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE, _style  # noqa: E402
from sim.config import load_sim  # noqa: E402
from sim.items import load_items  # noqa: E402
from sim.patient import PatientParams, SimPatient, sigmoid  # noqa: E402
from sim.population import make_patients  # noqa: E402

COHORTS = ["mild_anomic", "moderate_broca", "severe"]
CUES = ["first_letter", "semantic", "sentence_frame", "first_sound"]
N = 200


def cohort_patients(cfg, cl, bank, cohort, n=N):
    return [s.build(bank, cfg, cl)[0] for s in make_patients(cfg, cohort, "test", n, 0)]


def measures(cfg, cl, bank) -> dict:
    out = {"freq": {}, "cue": {}, "fatigue": {}, "tier": {}, "overall": {}}
    for c in COHORTS:
        pts = cohort_patients(cfg, cl, bank, c)
        P = np.array([p.retrieval_probabilities() for p in pts])          # patients x items, uncued, unfatigued
        out["freq"][c] = P.mean(0)
        out["overall"][c] = P.mean(1)                                       # per-patient accuracy over the bank
        out["tier"][c] = [float(P[:, bank.tier == t].mean()) for t in (1, 2, 3)]
        out["cue"][c] = {cue: float(np.mean([np.mean(sigmoid(p.ability - bank.difficulty + p.cue_gain[cue]))
                                             - np.mean(sigmoid(p.ability - bank.difficulty)) for p in pts]))
                         for cue in CUES}
        out.setdefault("cue_logodds", {})[c] = {cue: float(np.mean([p.cue_gain[cue] for p in pts])) for cue in CUES}
        curves = []
        for spec in make_patients(cfg, c, "test", N, 0):
            params = PatientParams(**{**spec.params.to_dict(), "learn_rate": 0.0})
            p = SimPatient(params, bank, cfg, cl, np.random.default_rng(spec.index))
            items = bank.indices_in_tier(2)
            curve = []
            for t in range(cfg.session.n_trials):
                curve.append(float(np.mean(sigmoid(p.ability - bank.difficulty - p.fatigue))))
                p.respond(int(items[t % len(items)]), "none")
            curves.append(curve)
        out["fatigue"][c] = np.array(curves).mean(0)
    return out


def plot(m: dict, bank, targets: dict, out) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), facecolor=SURFACE)
    ax = axes[0, 0]
    order = np.argsort(bank.frequency)
    bins = np.array_split(order, 6)
    for c, color in zip(COHORTS, SERIES):
        ax.scatter(bank.frequency, m["freq"][c], s=10, color=color, alpha=0.35, linewidths=0)
        ax.plot([bank.frequency[b].mean() for b in bins], [m["freq"][c][b].mean() for b in bins], color=color,
                linewidth=2, marker="o", markersize=5, label=c.replace("_", " "))
    _style(ax, "(a) Uncued accuracy vs word frequency\n" + targets.get("frequency_slope", ""), "p(correct), start of session")
    ax.set_xlabel("word frequency (Zipf = log10 per billion; higher = more common)", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")


    ax = axes[0, 1]
    w = 0.8 / len(COHORTS)
    for j, (c, color) in enumerate(zip(COHORTS, SERIES)):
        x = np.arange(len(CUES)) - 0.4 + w * (j + 0.5)
        ax.bar(x, [m["cue_logodds"][c][q] for q in CUES], width=w * 0.88, color=color, edgecolor=SURFACE,
               label=c.replace("_", " "))
    for q, rng in targets.get("cue_logodds_ci", {}).items():
        i = CUES.index(q)
        ax.add_patch(plt.Rectangle((i - 0.45, rng[0]), 0.9, rng[1] - rng[0], fill=False, edgecolor=INK,
                                   linestyle="--", linewidth=1.1,
                                   label="published 95% CI (Meteyard & Bose 2018)" if q == "first_sound" else None))
    ax.set_xticks(np.arange(len(CUES)))
    ax.set_xticklabels([q.replace("_", " ") for q in CUES], fontsize=8.5, color=INK_2)
    _style(ax, "(b) Immediate cue effect\npopulation mean calibrated to the published estimate", "gain over no cue (log-odds)")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    ax = axes[1, 0]
    for c, color in zip(COHORTS, SERIES):
        ax.plot(np.arange(1, len(m["fatigue"][c]) + 1), m["fatigue"][c], color=color, linewidth=2,
                label=c.replace("_", " "))
    _style(ax, "(c) Accuracy over a session: fatigue only (learning off)\n" + targets.get("fatigue", ""),
           "mean p(correct) over the word list")
    ax.set_xlabel("trial", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="center right")

    ax = axes[1, 1]
    for j, (c, color) in enumerate(zip(COHORTS, SERIES)):
        x = np.arange(3) - 0.4 + w * (j + 0.5)
        ax.bar(x, m["tier"][c], width=w * 0.88, color=color, edgecolor=SURFACE, label=c.replace("_", " "))
    if "accuracy_by_cohort" in targets:
        for j, (c, color) in enumerate(zip(COHORTS, SERIES)):
            if c in targets["accuracy_by_cohort"]:
                lo, hi = targets["accuracy_by_cohort"][c]
                ax.axhspan(lo, hi, xmin=0.02 + 0.32 * j, xmax=0.02 + 0.32 * j + 0.3, color=color, alpha=0.12)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["tier 1", "tier 2", "tier 3"], fontsize=9, color=INK_2)
    ax.set_ylim(0, 1)
    _style(ax, "(d) Uncued accuracy by tier and cohort\nordering mild > moderate > severe at every tier", "p(correct), start of session")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper right")
    fig.suptitle("Simulator validation: classic picture-naming effects in the simulated cohorts", x=0.01,
                 ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, f"{N} simulated held-out patients per cohort; placeholder item bank. Published values: "
             "data/calibration_targets.yaml (README, Simulator validity).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    cfg, cl = load_sim(), load_clinical()
    bank = load_items(cfg.items)
    tpath = resolve("data/calibration_targets.yaml")
    targets = load_yaml(tpath).get("figure", {}) if tpath.exists() else {}
    m = measures(cfg, cl, bank)
    plot(m, bank, targets, resolve("results/simulator_validation.png"))
    with open(resolve("results/simulator_validation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "cohort", "key", "value"])
        for c in COHORTS:
            w.writerow(["overall_accuracy_mean", c, "", float(np.mean(m["overall"][c]))])
            for t, v in zip((1, 2, 3), m["tier"][c]):
                w.writerow(["accuracy_by_tier", c, t, v])
            for q, v in m["cue"][c].items():
                w.writerow(["cue_effect", c, q, v])
            w.writerow(["fatigue_drop_40_trials", c, "", float(m["fatigue"][c][0] - m["fatigue"][c][-1])])
    print("wrote results/simulator_validation.png and .csv")


if __name__ == "__main__":
    main()
