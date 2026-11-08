"""Calibrate cue activation levels to cued-naming accuracy in the literature, and report what the
model then predicts about cues that it was not calibrated on.

Anchor (Nardo, Holland, Leff, Price & Crinion 2017, Brain 140:3039, Experiment 2, 18 chronic
patients, cues presented with the picture, before treatment): accuracy 0.60 with a noise control,
0.79 with the word's final phoneme, 0.88 with its initial phoneme, 0.93 with the whole word. In
rescue terms (share of otherwise-failed items named): final 0.475, initial 0.70, whole word 0.825.
The semantic cue has no comparable number; its rescue target 0.35 is an ASSUMPTION ranked below the
phonemic cues (Pease & Goodglass 1978 rank order; Python et al. 2021).

Method: a synthetic cohort from the fitted population prior, restricted to patients whose unaided
accuracy lies in 0.45-0.75 (around Nardo's 0.60), 1000 trials per patient with common random numbers;
one activation level per cue kind found by bisection so the cohort's mean rescue share matches.
Then, untouched by the fit: the rescue share by severity (Pease & Goodglass 1978: 49% of failed
pictures named after a phonemic cue, responsiveness lower in more severe patients) and by lesion
type for the semantic cue (Python et al. 2021: semantic cues ineffective in lexical-semantic anomia).

Run: uv run python -m scripts.calibrate_cues   (writes configs/cues.yaml and results/cue_calibration.md)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from patientmodel.cues import CueConfig
from patientmodel.dell_sp import CueInput, DellSPModel, load_dell_sp_config
from patientmodel.fit import load_or_build_grid
from patientmodel.population import PopulationSampler

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "cues.yaml"
REPORT = ROOT / "results" / "cue_calibration.md"
TARGETS = {"phonemic_initial": 0.70, "phonemic_final": 0.475, "whole_word": 0.825, "semantic": 0.35}
N_TRIALS = 1000


def make_input(kind: str, a: float) -> CueInput:
    if kind == "phonemic_initial":
        return CueInput(onset=a)
    if kind == "phonemic_final":
        return CueInput(coda=a)
    if kind == "semantic":
        return CueInput(semantic=a)
    return CueInput(onset=a, vowel=a, coda=a, word=a)


def p_correct(model: DellSPModel, s: float, p: float, seed: int, cue: CueInput | None = None) -> float:
    return model.simulate(s, p, N_TRIALS, seed=seed, cue=cue)["correct"]


def rescue(model, pats, base, kind, a) -> np.ndarray:
    cued = np.array([p_correct(model, pt.s, pt.p, 1000 + k, make_input(kind, a)) for k, pt in enumerate(pats)])
    return (cued - base) / np.maximum(1 - base, 1e-6)


def main() -> None:
    cfg, fit_cfg = load_dell_sp_config()
    model = DellSPModel(cfg)
    table = load_or_build_grid(model, fit_cfg["grid_step"], fit_cfg["trials_per_cell"], fit_cfg["seed"])
    sampler = PopulationSampler(table)
    pats = sampler.sample(400, seed=3)
    base = np.array([p_correct(model, pt.s, pt.p, 1000 + k) for k, pt in enumerate(pats)])
    calib_idx = [k for k in range(len(pats)) if 0.45 <= base[k] <= 0.75][:50]
    cal_pats = [pats[k] for k in calib_idx]; cal_base = base[calib_idx]

    levels: dict[str, float] = {}
    achieved: dict[str, float] = {}
    notes: dict[str, str] = {}
    for kind, target in TARGETS.items():
        if kind == "semantic":
            # No quantitative anchor: one extra semantic jolt (10 per feature, i.e. the cue carries as much
            # semantic input as the picture itself). ASSUMPTION; the rescue share it yields is reported.
            levels[kind] = float(cfg.semantic_jolt)
            notes[kind] = "set by assumption to one semantic jolt"
        else:
            # Rescue is not monotone in the level (a cue that also feeds competitors can hurt when large),
            # so scan a grid and take the level whose rescue share is closest to the target.
            grid = np.array([2.5, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130, 160, 200])
            r = np.array([float(np.mean(rescue(model, cal_pats, cal_base, kind, a))) for a in grid])
            k = int(np.argmin(np.abs(r - target)))
            levels[kind] = float(grid[k])
            if r.max() < target - 0.02:
                notes[kind] = f"target unreachable: the best the mechanism can do is {r.max():.2f} at level {grid[int(np.argmax(r))]:g}"
            else:
                notes[kind] = "fitted"
        achieved[kind] = float(np.mean(rescue(model, cal_pats, cal_base, kind, levels[kind])))

    # Predictions not used in the fit.
    bands = [("severe, unaided < 0.30", base < 0.30), ("0.30-0.50", (base >= 0.30) & (base < 0.50)),
             ("0.50-0.70", (base >= 0.50) & (base < 0.70)), ("mild, >= 0.70", base >= 0.70)]
    sev_rows = []
    for label, m in bands:
        idx = np.where(m)[0][:40]
        if len(idx) == 0:
            continue
        sub = [pats[k] for k in idx]; sb = base[idx]
        r_init = float(np.mean(rescue(model, sub, sb, "phonemic_initial", levels["phonemic_initial"])))
        r_sem = float(np.mean(rescue(model, sub, sb, "semantic", levels["semantic"])))
        r_word = float(np.mean(rescue(model, sub, sb, "whole_word", levels["whole_word"])))
        sev_rows.append(f"| {label} | {len(idx)} | {sb.mean():.2f} | {r_init:.2f} | {r_sem:.2f} | {r_word:.2f} |")
    lesion_rows = []
    lr = np.array([np.log(pt.p / pt.s) for pt in pats])
    for label, m in [("semantic lesion (p/s > 1.3)", lr > np.log(1.3)), ("balanced", np.abs(lr) <= np.log(1.3)),
                     ("phonological lesion (p/s < 1/1.3)", lr < -np.log(1.3))]:
        idx = np.where(m & (base > 0.2) & (base < 0.9))[0][:40]
        if len(idx) == 0:
            continue
        sub = [pats[k] for k in idx]; sb = base[idx]
        r_sem = float(np.mean(rescue(model, sub, sb, "semantic", levels["semantic"])))
        r_init = float(np.mean(rescue(model, sub, sb, "phonemic_initial", levels["phonemic_initial"])))
        lesion_rows.append(f"| {label} | {len(idx)} | {sb.mean():.2f} | {r_sem:.2f} | {r_init:.2f} |")

    raw = yaml.safe_load(open(CFG)) if CFG.exists() else {}
    raw["activation"] = {"phonemic_initial_onset": levels["phonemic_initial"], "phonemic_final_coda": levels["phonemic_final"],
                         "semantic_features": levels["semantic"], "whole_word_phoneme": levels["whole_word"],
                         "whole_word_lexical": levels["whole_word"]}
    raw.setdefault("effort", CueConfig().effort or {"phonemic_initial": 0.5, "phonemic_final": 0.4, "semantic": 0.3, "whole_word": 1.0})
    header = ("# Cue activation levels (activation units on the SP model's scale; semantic jolt is 10 per feature,\n"
              "# lexical jolt 100). Written by scripts/calibrate_cues.py; see results/cue_calibration.md.\n"
              "# effort: share of the retrieval the cue does for the patient; weights a cued success between an\n"
              "# unaided success and errorless study in the learning layer. ASSUMPTION (direction from Howard et al.\n"
              "# 1985 and Middleton et al. 2019; values not fitted).\n")
    CFG.write_text(header + yaml.safe_dump(raw, sort_keys=False))

    lines = ["<!-- Generated by: uv run python -m scripts.calibrate_cues. Do not edit by hand. -->",
             "# Cue calibration\n",
             f"Cohort: {len(pats)} synthetic patients from the fitted population prior; calibration on the {len(cal_pats)} with "
             f"unaided accuracy 0.45-0.75 (mean {cal_base.mean():.2f}), {N_TRIALS} trials each. Target rescue shares from Nardo et al. "
             "2017 (initial phoneme, final phoneme, whole word); semantic is an ASSUMPTION.\n",
             "| cue | activation level | target rescue share | achieved | note |", "|---|---|---|---|---|",
             *[f"| {k} | {levels[k]:g} | {TARGETS[k]:.3f}{' (assumed)' if k == 'semantic' else ''} | {achieved[k]:.3f} | {notes[k]} |" for k in TARGETS],
             "\nA rescue share is the fraction of items the patient would have failed that are named with the cue. "
             "The final-phoneme cue is the informative failure: in the model's six-word neighbourhood the two formal "
             "competitors share the target's coda, so a coda cue raises them as much as the target and its benefit "
             "peaks low and then reverses. Real final-phoneme cues (Nardo: 0.79 against 0.60 unaided) do better than "
             "that, which says real formal competitors share less of the word's ending than the model's do, or that a "
             "final sound carries information the model's three-slot phonology cannot represent.\n",
             "**Prediction 1: cue effect by severity** (not fitted). Pease & Goodglass 1978: phonemic cues named 49% of "
             "failed pictures, and responsiveness was lower in more severe patients.\n",
             "| unaided accuracy band | n | mean unaided | initial-phoneme rescue | semantic rescue | whole-word rescue |",
             "|---|---|---|---|---|---|", *sev_rows,
             "\n**Prediction 2: semantic cue by lesion type** (not fitted). Python et al. 2021: semantic cues were ineffective "
             "for lexical-semantic anomia and phonological cues worked regardless of profile.\n",
             "| lesion type | n | mean unaided | semantic rescue | initial-phoneme rescue |", "|---|---|---|---|---|", *lesion_rows,
             "\nHow to read a disagreement here: these two tables are what the SP model, with cues as external activation, "
             "says about cases it was not tuned on. Where they match the literature the mechanism is doing work; where "
             "they do not, the mechanism is wrong in a stated way."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
