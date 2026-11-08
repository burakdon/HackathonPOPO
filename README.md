# Aphasia Trainer

A home practice trainer for people with aphasia, the loss of language that follows many strokes. The
patient names pictures; an AI therapist chooses the difficulty and the hint (cue) for every trial so
the patient stays near 80 to 85 percent correct. The therapist is trained against a population of
simulated patients and runs live on a real person, and a speech-language pathologist (SLP) can
override any of its choices.

> **Practice tool used under an SLP.** It does not diagnose, replace therapy, or make treatment
> decisions. No real patient data is used anywhere. We do not claim it improves recovery in real
> patients.

## The system

| component | what it does |
|---|---|
| Patient app | Browser app: shows a picture, takes a spoken or tapped answer, shows the cue. Large type, one thing on screen. |
| Speech scorer | Transcribes the answer and labels it: correct, sound error, meaning error, or no response. |
| Therapist agent | Chooses the next difficulty tier and cue from the session so far (`agents/`, served by `service/`). |
| Patient simulator | A population of simulated patients used to train and evaluate the therapist (`sim/`). |
| Clinician dashboard | Session trace (difficulty, accuracy, error types, the agent's choices) with an override on every choice. |
| Expert-feedback loop | Clinician overrides are stored as demonstrations so the therapist can learn a clinician's preferences. |

Live loop: the patient names a picture, the scorer labels the answer, the therapist picks the next
picture's tier and cue, and the app shows it. Two learning signals feed the therapist: practice on
simulated patients, and demonstrations from clinician overrides.

## Status

The team objective is **better later naming without hints**. Sessions therefore include review
probes (an earlier practised word shown again with no cue), and every policy is evaluated on them.

Implemented and tested (170 tests, `uv run pytest`):
- Agent service (FastAPI) implementing the agreed State/Action interface, with one online-learning
  agent per session, clinician overrides logged for later training, CORS for a frontend on the local
  network, and `POST /next_trial`, which applies the shared item-selection and review-probe rule.
- Patient simulator: population sampling with a held-out test split, three evaluation cohorts,
  item difficulty from word frequency and length, patient-specific cue responsiveness, error types,
  fatigue with recovery, learning with a configurable peak, patient dropout after failure streaks,
  and review probes. **Calibrated against published picture-naming statistics** (14 group-level
  targets, all passing; see Simulator validity), with out-of-distribution and scorer-noise checks.
  The item bank is a placeholder.
- Therapist agents: MaskablePPO trained on the observable proxy reward (the live demo agent, chosen
  for visible adaptation to a healthy user), MaskablePPO trained on the simulation-only objective and
  the contextual Bayesian logistic Thompson-sampling bandit (live fallbacks), a Beta-Bernoulli
  Thompson bandit, and fixed-difficulty and staircase baselines.
- Trained agent: a Gymnasium environment with clinical action masks and a decomposed reward, and
  MaskablePPO trained on simulated patients with three rewards (observable proxy, proxy + review-probe
  bonus, privileged simulator-only objective), 3 seeds each, learning curves on held-out patients.
- Learning from clinician overrides: DITTO-style preference optimisation of the PPO policy from ~10
  overrides, with a before/after evaluation on held-out states.
- Evaluation harness: privileged oracle, 200 held-out patients per cohort, bootstrap CIs, paired
  differences and the two result figures, reproduced by one command (`eval/compare.py`); ablations of
  the learning-peak assumption, dropout and the reward (`eval/ablations.py`).

In progress or planned:
- Patient app and clinician dashboard (`frontend/`).
- Speech scoring and error classification.
- The prompted frontier-model baseline is implemented and tested with a mocked API; it runs as soon as
  an API key is available (`configs/eval_llm.yaml`).
- A clinically curated item bank and clinical validation of the simulator's parameters.

Explored, not part of the current scope: a language-model therapist trained with GRPO on Tinker
(prompt format, offline reward and dataset, and a 10-step smoke test are in `agents/llm_prompt.py`
and `train/grpo_*.py`; never run against the API).

## Quick start

```bash
uv sync                                         # Python 3.11 env from uv.lock
uv run pytest                                   # test suite
uv run uvicorn service.app:app --host 0.0.0.0 --port 8000   # agent service, reachable on the LAN
uv run python -m scripts.service_smoke --url http://localhost:8000   # 40-trial session over HTTP
uv run python -m train.fit_bandit_prior         # refit the bandit's empirical-Bayes prior (~2 s)
uv run python -m train.train_ppo --reward full          # MaskablePPO, 3 seeds (~3 min each on CPU) -> models/
uv run python -m train.train_ppo --reward full_probe    # proxy + review-probe bonus
uv run python -m train.train_ppo --reward privileged    # simulator-only objective (reference)
uv run python -m eval.plots                     # learning curves -> results/learning_curves.png
uv run python -m train.ditto_ppo                # DITTO-style tuning from overrides -> results/ditto_report.md
uv run python -m scripts.validate_live_ditto     # live override learning on fresh states
uv run python -m eval.calibration               # simulator vs published statistics (data/calibration_targets.yaml)
uv run python -m eval.sim_validation            # results/simulator_validation.png
uv run python -m eval.robustness                # out-of-distribution patients and scorer noise
uv run python -m scripts.judge_demo --url http://localhost:8000 --name live   # healthy-judge check
uv run python -m eval.compare --config configs/eval.yaml   # full evaluation -> results/
uv run python -m scripts.sanity_table --dropout on --out results/sanity_dropout_on.md
uv run python -m scripts.sanity_table --dropout off --out results/sanity_dropout_off.md
uv sync --extra grpo && uv run python -m train.grpo_smoke   # explored, out of scope; needs TINKER_API_KEY
```

**Frontend on another laptop:** run the service with `--host 0.0.0.0`, find this machine's LAN IP
(`ipconfig getifaddr en0` on macOS) and call `http://<that-ip>:8000`. CORS allows the Vite dev server
(port 5173) on localhost and on any private-network address (`cors_origins` / `cors_origin_regex` in
[`configs/service.yaml`](configs/service.yaml)). The macOS firewall may ask to allow incoming
connections for Python the first time.

## Repository layout

RL and agent code lives at the repository root; the frontend app goes in `frontend/`.

```
core/      shared schema (State/Action/Outcome), typed config loading, action masks
agents/    therapist policies behind one interface (agents/base.py)
service/   FastAPI agent service the frontend calls
sim/       simulated patient population
envs/      Gymnasium environment with action masks
train/     PPO training, bandit-prior fitting, language-model (GRPO) training
eval/      metrics and figures
models/    trained PPO policies (small, committed so results reproduce without retraining)
configs/   all tunable values (YAML)
scripts/   smoke tests and utilities
tests/     pytest suite
results/   committed CSV/PNG outputs for this README
docs/      starter scaffold kept for reference (docs/scaffold/rl_starter.py)
frontend/  patient app and clinician dashboard
```

## Branches

- `main`: stable and demo-ready only. Updated from `dev` at checkpoints, when the demo works end to end.
- `dev`: integration branch. Small fixes (CORS, typos, README) may be committed here directly.
- `feature/rl-agent`: all RL work (simulator, bandits, training, evaluation), branched from `dev`.

Merges happen only through pull requests on GitHub (`feature/rl-agent` → `dev`, and `dev` → `main`
at checkpoints), opened once the tests pass and a piece works. Nobody merges locally.

## Interfaces

The contract between the frontend, the scorer and the agent. Defined once in
[`core/schema.py`](core/schema.py) (pydantic); the service validates every request against it and
returns HTTP 422 with a precise error on any violation.

**Changes from the original contract (all additive and backward-compatible):**
1. `State` has an optional `session_id`. The service keeps one online-learning agent per session;
   requests without it share a single `default` session. **Please always send it** (one id per
   patient session, any string up to 128 chars).
2. `POST /update` takes an explicit `outcome` object (shape below).
3. Review probes: trials may carry `"probe": true`; `POST /update` accepts `"probe": true`; and
   `current_tier` means the tier of the last trial the **agent** chose (probes do not change it).
4. New endpoint `POST /next_trial` returns the complete next trial (word, image, tier, cue, probe)
   using the shared item-selection rule below, so the frontend does not have to implement it.
5. New endpoint `POST /learn_from_overrides` (live demo of learning from the clinician, below).

### State
```json
{"session_id": "patient-42-2026-11-07",
 "last_trials": [{"word": "cat", "tier": 2, "cue": "first_sound", "outcome": false,
                  "error_type": "sound", "response_time": 3.1}],
 "current_tier": 2, "trial_count": 14, "session_minutes": 6.5}
```
| field | type | rule |
|---|---|---|
| `session_id` | string, optional | 1 to 128 chars |
| `last_trials` | list | most recent last; up to 10 (longer lists are truncated to the last 10) |
| `last_trials[].tier` | int | 1..3 |
| `last_trials[].cue` | string | `none \| first_sound \| first_letter \| sentence_frame \| semantic` |
| `last_trials[].outcome` | bool | must be `true` iff `error_type == "correct"` |
| `last_trials[].error_type` | string | `correct \| sound \| meaning \| no_response` |
| `last_trials[].response_time` | float | seconds, >= 0 |
| `last_trials[].probe` | bool, optional | `true` for a review probe (default `false`) |
| `current_tier` | int | 1..3, the tier of the last trial the agent chose; review probes do not change it |
| `trial_count` | int | >= 0 |
| `session_minutes` | float | >= 0 |

Unknown extra fields are ignored, so adding e.g. a timestamp will not break anything.

### Action
```json
{"tier": 2, "cue": "first_sound", "rationale": "short text for the dashboard"}
```
`tier` in 1..3, `cue` as above. `rationale` is a human-readable explanation (may be null).
The agent never returns a tier more than one step from `current_tier`, and never a cue outside the
clinician-approved set in [`configs/clinical.yaml`](configs/clinical.yaml).

### Outcome
```json
{"correct": false, "error_type": "sound", "response_time": 3.1}
```
`correct` must be `true` iff `error_type == "correct"`.

### Item selection and review probes (the frontend must use the same rule)
The agent chooses only the tier and cue; the item is chosen by this rule (implemented in
[`sim/items.py`](sim/items.py) and used by the simulator and by `POST /next_trial`). Trials are
numbered k = 0, 1, 2, ... over the whole session, probes included.
1. **Review probe** if review probes are enabled, (k + 1) is a multiple of 5, and an eligible item
   exists. Eligible: practised earlier in this session as a normal trial, and not shown in the last
   3 trials. Pick the eligible item with the fewest probes so far this session; ties go to the item
   first practised earliest. Show it **with no cue**, at its own tier. Do not call `/act` for it.
2. **Otherwise a normal trial:** call `/act`, then pick an item in the returned tier that has not
   been shown this session; if all have been shown, the least recently shown (ties at random).
Parameters live in [`configs/sim.yaml`](configs/sim.yaml) (`review_probes`: `every` = 5,
`min_gap` = 3). The simplest integration is `POST /next_trial`, which applies this rule server-side.

### Endpoints
| method | path | body | returns |
|---|---|---|---|
| GET | `/health` | none | `{"status": "ok", "agent": "...", "active_sessions": n}` |
| POST | `/next_trial` | `State` | `{"word", "image_file", "tier", "cue", "probe", "rationale"}` |
| POST | `/act` | `State` | `Action` |
| POST | `/update` | `{"state": State, "action": Action, "outcome": Outcome, "probe": false}` | `{"status": "ok", "session_id": "...", "agent_updated": bool}` |
| POST | `/override` | `{"state": State, "agent_action": Action, "clinician_action": Action, "note": "optional"}` | `{"status": "stored", "session_id": "...", "n_overrides": n}` |
| POST | `/learn_from_overrides` | `{"session_id": "..."}` | `{"n_overrides", "n_skipped", "n_anchor_states", "seconds", "by_state_type": [{"state_type", "clinician_cue", "n_overrides", "n_eval_states", "p_before", "p_after"}], "other_states_unchanged"}` |

Session loop for the frontend:
1. `POST /next_trial` with the current `State` (or apply the rule above yourself and call `/act`
   for normal trials). Show the returned word with the returned cue.
2. If the clinician overrides a normal trial, `POST /override` with both actions and show the
   clinician's action.
3. After the patient answers, `POST /update` with the same `State`, the action **actually shown**
   (the clinician's if overridden), the `Outcome`, and `"probe"` as returned. Agents do not learn
   from probes, which they did not choose.
4. Append the trial (with `"probe"`) to `last_trials`; if it was not a probe, set `current_tier` to
   its tier. Repeat.

Overrides are stored as JSON lines in `runs/overrides.jsonl` (the input to `train/ditto_ppo.py --overrides`).

**Live learning from overrides (`POST /learn_from_overrides`, PPO agent only).** Runs one
DITTO-style round (see the DITTO section) on the overrides logged **in this session**, then swaps in
the tuned policy **for this session only** (other sessions and the base model are unchanged).
Anchors are this session's accepted choices plus the base policy's own decisions on 30 validation
patients, excluding the state types the clinician overrode. It takes well under a second on CPU
(~0.3 s for 10 overrides; the first call also collects the anchor pool, about 1 s). The response
reports, for each overridden state type (for example `"after a sound error"`), the mean probability
of choosing the clinician's cue in reference states of that type before and after, and the share of
other reference states whose top action is unchanged. Example on the live agent with 10 overrides
"after a sound error, use first_sound" collected from states the policy actually visits: `p_before`
0.35, `p_after` 0.87, `other_states_unchanged` 1.00 (0.5 s); the session then chose first_sound in
held-out sound-error states. Validation on fresh states: see the DITTO section. Errors: 409 if the live agent is not PPO, 400 if the session has no overrides.
Demo flow: play a few trials, override the cue a handful of times on the dashboard, call this
endpoint, and continue: the next similar states get the clinician's choice.
Interactive API docs: `http://localhost:8000/docs` while the service runs.

## Demo script

Start the service (`uv run uvicorn service.app:app --host 0.0.0.0 --port 8000`; `GET /health` should
report `"agent": "ppo"`). Use one `session_id` per demo, e.g. `"demo-judge"`. Each step below is one
or more API calls; the right column is what the clinician dashboard should show.

| step | API calls | dashboard shows |
|---|---|---|
| 1. Naming | `POST /next_trial` with the State (`trial_count` 0, `current_tier` 1, empty `last_trials`); the judge names the picture; `POST /update` with the outcome and `"probe": false`. Repeat. | The picture, the chosen tier and cue (`none`), the agent's `rationale`; the tier timeline and rolling accuracy with the 75-90% band. After 5-7 correct answers the tier climbs to 2, then 3. |
| 2. Cue after an error | The judge gives a wrong answer (or none) on purpose, 2-3 times in a row; keep calling `POST /next_trial` and `POST /update`. | The agent steps the tier down and offers a cue (e.g. first letter), with the rationale; error types appear in the breakdown. After the judge recovers, the tier climbs back. |
| 3. Review probe | Every 5th trial `POST /next_trial` returns `"probe": true` (a word practised earlier, `cue: "none"`); `POST /update` with `"probe": true`. | A "review, no hint" label on that trial; the current tier is unchanged; probe results are tallied as "recalled without a hint". |
| 4. Override | On a trial right after a sound error, the clinician presses override and picks the first-sound cue (same tier): `POST /override` with `state`, `agent_action`, `clinician_action`; show the clinician's cue; `POST /update` with the clinician's action. Repeat on a few such trials (10 give a strong effect; fewer give a smaller one). | The override is marked on the timeline; the override count increases. |
| 5. Learn from the clinician | `POST /learn_from_overrides` with `{"session_id": "demo-judge"}` (well under a second). | "After a sound error: P(first sound) about 0.35 -> about 0.9; other behaviour unchanged ~98-100%" from the response (`by_state_type`, `other_states_unchanged`). |
| 6. The agent now makes the clinician's choice | Continue the session; after the next sound error `POST /next_trial` returns the first-sound cue without an override. | The cue matches the clinician's earlier choice; no override was needed. |

Fallback if the live agent misbehaves: set `ppo_model: models/ppo_privileged_seed0.zip` or
`agent: logistic_bandit` in `configs/service.yaml` and restart (`/learn_from_overrides` needs a PPO
agent).

## Problem formulation

A **POMDP**. The patient's true state (retrieval ability per item, fatigue, responsiveness to each
cue type, learning rate) is hidden; the agent only observes trial outcomes through the State JSON.
An **episode** is one session of 40 trials; a **step** is one trial. The **action** is (tier 1..3) x
(cue: 5 types) = 15 discrete actions. Clinical constraints are **action masks**, not penalties: tier
moves at most one step per trial, and cues outside the clinician-approved set are never offered.

## Simulated patients (`sim/`)

`p(correct) = sigmoid(ability + item_learning[i] - difficulty[i] + cue_gain[cue] - fatigue)`, all on
the log-odds scale. Every parameter is in [`configs/sim.yaml`](configs/sim.yaml) with a comment that
either cites the paper behind the *direction* of the effect or says `ASSUMPTION`.

| component | model | grounding |
|---|---|---|
| item difficulty | `beta_f * (-z(frequency)) + beta_l * z(length)` | frequency and length effects on naming (Nickels & Howard 1995); frequency slope **calibrated** to 0.6 log-odds per log10 unit (published 0.41-0.88) |
| cues | `phonemic_weight[cue] * phon_resp + semantic_weight[cue] * sem_resp`, per-patient responsiveness | phonemic > semantic (Meteyard & Bose 2018); first-sound and semantic gains **calibrated** to their estimates (0.65 and about 0 log-odds) |
| error types | after a failure, P(sound, meaning, no response) proportional to (phon_impair, sem_impair, omission_bias) | simplified from the two-step model (Dell et al. 1997; Schwartz et al. 2006); error mix **calibrated** to Kittredge et al. 2008 and Dell, Martin & Schwartz 2007 |
| fatigue | grows per trial and faster during failure streaks, recovers a fraction each trial, capped | ASSUMPTION |
| learning | on a success, item-specific gain with E[gain] = p^a (1-p)^b, peaking at a configurable `peak`; weaker cues teach more; 5% spills over to general ability | item-specific gains with limited generalisation (Nickels 2002); the peak location is an ASSUMPTION; see the learning-peak ablation |
| response time | log-normal, slower for hard items and when fatigued | ASSUMPTION |
| dropout | after 3 failures in a row the patient may quit: 5% per trial, +5% per further failure | ASSUMPTION (conservative); results reported with dropout on and off |
| review probes | every 5th trial re-tests a practised word without a cue (rule in Interfaces); scored like any trial | measurement design, not a patient assumption |

**Population, not fixed profiles.** Training patients are drawn from broad parameter distributions
(domain randomisation). Three named presets (mild anomic, moderate Broca's, severe) are narrow
cohorts used only for evaluation. Patient *k* of a split comes from `SeedSequence([seed, split, cohort, k])`,
so the **train and test populations are disjoint** (tested), and every policy faces exactly the same
patients with the same random outcome stream (common random numbers, so comparisons are paired).

**Item bank.** Loaded from a CSV with columns `word,image_file,frequency,length,tier` (schema in
[`sim/items.py`](sim/items.py)). The committed bank [`data/items_placeholder.csv`](data/items_placeholder.csv)
is a **PLACEHOLDER**: 60 picturable nouns, frequencies from the `wordfreq` package (Zipf scale),
length in letters, tiers = difficulty terciles. It is to be replaced by a clinically curated bank. Within a tier,
items are shown unseen-first, then least recently seen; the app should use the same rule.

## Simulator validity

**What is calibrated.** `data/calibration_targets.yaml` lists 14 group-level statistics read in the
published sources (each with a verbatim quote and the exact measure we compare), and
`tests/test_calibration.py` asserts each one (`uv run python -m eval.calibration` prints the table).
Sources: Kittredge et al. 2008 (50 patients, PNT), Dell, Martin & Schwartz 2007 (65 patients,
per-patient PNT data), Meteyard & Bose 2018 (10 patients, cueing experiment), Mirman et al. 2024
(three samples, fatigue).

| target | published | simulator before | after |
|---|---|---|---|
| mean naming accuracy, mixed sample | 59% (range 3-98%) | 0.61 | 0.60 |
| spread of patient accuracy (5th-95th percentile width) | range 3-98% | 0.52 (too narrow) | 0.64 |
| semantic share of semantic + phonological + omission errors | 5 / 32 = 0.16 | 0.30 (off) | 0.17 |
| phonological share | 15 / 32 = 0.47 | 0.32 (off) | 0.46 |
| omission share | 12 / 32 = 0.375 | 0.38 | 0.38 |
| anomic accuracy, normalised as in Dell 2007 | 0.82 (SD 0.16) | 0.90 | 0.94 |
| Broca's accuracy, normalised | 0.69 (SD 0.24) | 0.72 | 0.73 |
| anomic: phonological / (phonological + semantic) errors | 0.59 | 0.30 (off) | 0.62 |
| Broca's: phonological / (phonological + semantic) errors | 0.70 | 0.79 | 0.79 |
| frequency effect, log-odds per log10 unit | 0.41 (Kittredge) to 0.88 (Mirman) | 1.25 (off) | 0.60 |
| first-sound cue gain, log-odds | 0.65 [0.48, 0.82] | 1.02 (off) | 0.65 |
| semantic cue gain, log-odds | -0.04 [-0.41, 0.32] | 0.57 (off) | 0.26 |
| between-patient SD of ability, logit | about 1.3 | 0.98 | 1.13 |
| accuracy falls over a session (direction) | yes, 2 of 3 samples | yes | yes |

Before calibration 7 of 14 targets passed. We changed only the parameters behind the 7 that were
off (population ability spread and error mix, the anomic error mix, the frequency weight, and the
phonemic and semantic cue channel magnitudes); all 14 now pass
([`results/calibration_report.md`](results/calibration_report.md)). Comparability caveats: the
published samples used the 175-item PNT (ours is a 60-item placeholder bank); the Dell 2007
proportions are normalised (omissions removed), so we normalise the same way; Meteyard & Bose gave
an associated word 750 ms *before* the picture, while our cues follow an error and our semantic cue
is a hint, so their semantic estimate is the closest verified number rather than an exact match;
their sample is small (n=10).

**Effect on the conclusions.** The full evaluation was re-run on the calibrated simulator with the
already-trained models (which were trained on the pre-calibration simulator and were not retrained,
so this is also a shift test). The main conclusions hold: on severe patients the PPO agents abandon
fewer sessions and achieve more probe lift than fixed difficulty and the staircase, with
non-overlapping 95% CIs. Magnitudes are smaller than before calibration because severe patients
now gain less from cues (sessions abandoned for the live agent 30% instead of 10%; probe lift 1.03
instead of 2.26, x100).

![Simulator validation](results/simulator_validation.png)

**What is still an assumption** (not calibrated, to be validated clinically): where learning peaks
and how much each success teaches (the learning-peak ablation shows the conclusions depend on it);
the cue discount on learning; the mix of phonemic and semantic components inside each cue; fatigue
magnitudes (only the direction is checked); dropout; response times; the three named cohorts'
parameters beyond the checks above (severe in particular has no verified published counterpart); the
item bank (placeholder); and the absence of any dependence of cue effectiveness on the previous
error type.

**Out-of-distribution patients** ([`results/robustness.md`](results/robustness.md),
`uv run python -m eval.robustness`). Three cohorts with parameters strictly outside the training
ranges, 200 patients each, mean [95% CI]:

| cohort (outside training on) | policy | sessions abandoned | learning gain x100 | probe lift x100 |
|---|---|---|---|---|
| very severe (ability) | fixed / staircase | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| | PPO proxy / PPO simulation objective | 0.97 [0.95, 0.99] / 0.98 [0.96, 0.99] | 0.00 / 0.00 | 0.01 / 0.00 |
| | oracle | 0.94 [0.91, 0.97] | 0.01 | 0.01 |
| fast fatigue (fatigue rate and recovery) | fixed / staircase | 0.47 [0.40, 0.54] / 0.27 [0.20, 0.33] | 1.55 / 2.27 | 2.55 / 2.70 |
| | PPO proxy / PPO simulation objective | 0.12 [0.08, 0.16] / 0.14 [0.09, 0.18] | 3.07 [2.92, 3.21] / 3.05 | 2.95 / 2.99 |
| | oracle | 0.07 [0.04, 0.11] | 3.42 | 3.14 |
| semantic responders (phonemic and semantic responsiveness) | fixed / staircase | 0.07 / 0.05 | 4.50 / 4.38 | 4.16 [3.95, 4.37] / 3.45 |
| | PPO proxy / PPO simulation objective | 0.03 / 0.03 | 4.92 / 5.45 [5.25, 5.66] | 3.53 / 3.98 |
| | oracle | 0.02 | 5.69 | 4.08 |

Patients more severe than any seen in training are beyond every policy, including the oracle: no
choice of tier or cue keeps them practising. For fast-fatiguing patients the agents stay close to
the oracle and keep abandonment at 12-14% versus 47% for fixed difficulty. For semantic responders
the simulation-objective PPO is second only to the oracle on learning gain, although it never saw
such patients; fixed tier 2 keeps the highest probe lift there, as on moderate patients.

**Scorer noise.** The observed outcome is flipped with probability 0.05 or 0.10 (evaluation only;
the patient's true responses and all metrics are unaffected). Paired against no noise on the same
patients, 10% noise changes learning gain by at most 0.23 points (x100) and probe lift by at most
0.32 points for any adaptive policy and cohort, and leaves the severe-cohort results unchanged within
their 95% CIs (full table in `results/robustness.md`). Fixed difficulty ignores outcomes and is
unaffected by construction.

## Agents (`agents/`)

All agents implement one interface (`act(state, mask) -> Action`, `update(state, action, outcome)`)
and decide from the State JSON only.

- **Fixed difficulty** (worksheets today) and **2-up/1-down staircase** (Levitt 1971).
- **Beta-Bernoulli Thompson bandit**: 15 independent arms, ported from the scaffold and bug-fixed.
- **Contextual Bayesian logistic Thompson bandit** (live fallback; learns online per session): `p(correct) = sigmoid(w . phi)`
  with 17 features (tier, cue, recent accuracy, fail streak, progress, response time, and last error
  type x cue family; the simulator does not yet model that interaction, see below). Posterior updated online with the closed-form Laplace/extended-Kalman step,
  with random-walk process noise so it tracks fatigue and learning. Weights are shared across
  actions, so one observation informs many actions.
- **Empirical-Bayes prior** for the contextual bandit, fitted on 300 simulated *training* patients
  (`train/fit_bandit_prior.py`): the per-patient weights' mean and shrunk covariance. This is the
  bandit's sim-to-real pretraining. It is inflated so a real person's data still dominates: in a test,
  a patient far from the population moves the prediction from 0.85 to 0.44 in 30 trials (truth 0.40).
- **Shared selection rule** (`agents/selection.py`): among actions whose predicted p is within 0.06
  of the 0.85 target, take the hardest tier, then the weakest cue; otherwise the closest to target.
  This fixes the tier-1 coasting of the first version of the bandit (37 of 40 trials on tier 1).
- **DITTO-style learning from clinician overrides** (`agents/ditto.py`, method after Shaikh et al.
  2025): see the section below.

## Trained agent: MaskablePPO (`envs/`, `train/train_ppo.py`)

**Environment.** One episode is one 40-trial session with a new simulated patient drawn from the
broad training population. The observation is built from the State JSON only (the last 8 trials as
one-hot tier, cue and error type plus response time, and the current tier, rolling accuracy, fail
streak and session progress), so the trained policy runs unchanged behind the service. The history
stack gives the policy memory in this POMDP. The action masks are the same clinical masks the
bandits use, including "cue only after an error". (sb3-contrib has no recurrent *and* masked PPO,
so we use MaskablePPO with a history stack rather than RecurrentPPO, which would turn masks into
penalties.)

**Rewards** ([`configs/reward.yaml`](configs/reward.yaml)). The default **proxy reward** uses
only terms observable on a real patient: +1 per trial with rolling accuracy in the band; -1 per trial while the failure streak is 3
or more; -0.1 when a correct uncued answer follows a correct answer with a heavy cue (cue probably
wasted); and, once at the end, +2 per tier climbed, paid only if the session spent at least half its
trials in band. Making the climb bonus terminal and conditional removes an exploit in the original
scaffold (per-step reward for climbing while in band, no cost for dropping out of band, so
oscillating tiers farmed reward). Tests check that oscillating tiers earns nothing per step and,
in the environment, that alternating tiers 1 and 2 scores less than holding tier 2 (mean return
13.8 vs 18.5 on validation patients). Two variants are trained for comparison:
- **proxy + probe**: the proxy plus +1 for every review probe answered correctly (still observable).
- **privileged** (reference only, not deployable): frustration penalty plus three simulator-only
  terms: the true control error |p - 0.85| on every agent trial, the dense increase in true
  retrieval over the item bank, and the probe lift (how much each probed word's uncued recall
  improved since the session began). Raw probe recall is deliberately not used here: probes revisit
  practised words, so a policy that practises only easy words gets easy probes. An earlier draft
  that rewarded the true p on probes indeed drove the agent to tier 1.

**Live agent.** (Judge traces below were recorded before the simulator calibration; the agents and the
service did not change, so the live behaviour is the same.) A live demo is played by a healthy person who answers almost everything correctly,
which is far outside the simulated patient population. `scripts/judge_demo.py` plays three scripted
"judges" through the HTTP API (`/next_trial` + `/update`, review probes included) against each
candidate agent, 30 trials each:

| judge | PPO, proxy reward | PPO, simulation objective | contextual bandit |
|---|---|---|---|
| always correct | climbs to tier 3 by trial 7 and stays (mean tier 2.54) | stays on tier 1 (mean 1.04) | wanders between tiers 1-3 (7 tier changes, mean 1.96) |
| ~90% correct | reaches tier 3 by trial 16, drops a tier with a cue after errors (mean 1.88) | stays on tier 1 | tiers 1-2, cues after errors (mean 1.75) |
| 3 errors in a row at trial 12 | tier 3 -> 2 with a first-letter cue -> 1, then back up to tier 3 within 8 trials | stays on tier 1, one cue | tier 2 -> 1, then oscillates 1-2 |

**We serve the proxy-reward PPO live** (`models/ppo_seed0.zip`, seed 0): it is the only agent whose
behaviour shows the intended adaptation to a healthy user (climb on success, back off with a cue
after errors, climb again). The simulation-objective PPO is better on the true objective in
simulation but stays on tier 1 for a healthy user, because in its training population tier 1 is
usually close to the target; it and the contextual bandit are configured fallbacks
(`configs/service.yaml`). Trade-off, stated plainly: in simulation the proxy-reward policy partly
games the 8-trial band (Results); it is chosen for legible live behaviour, not because it is the best
policy on the simulated objective. The policies do not update their weights during a session; they
react through their observation (the last 8 trials).

**Training.** 3 seeds x 1.5M steps per reward, CPU, about 3 minutes per run. Learning curves are
measured on 64 fixed validation patients disjoint from both the training and the test patients. All
variants converge within about 100k steps. The proxy-trained policies reach about 0.71 time in band
with a true control error near 0.14; the privileged policy holds the band less (about 0.52) but has
a much lower true control error (about 0.11) and higher learning gain and probe lift.

![PPO learning curves](results/learning_curves.png)

## Results

**Full evaluation** (`uv run python -m eval.compare --config configs/eval.yaml`, about 20 s) on the
calibrated simulator: 200 held-out test patients per cohort, 40 trials (every 5th a review probe),
dropout on. Every policy faces the same patients with the same random outcome stream; PPO is run
with each of its 3 training seeds and averaged per patient (per-seed results in
`results/comparison_seeds.csv`). The trained models (PPO, the bandit's fitted prior) were trained on
the pre-calibration simulator and are evaluated unchanged. Intervals are 95% percentile bootstrap
intervals over patients; differences between policies are **paired** (same patients) in
`results/paired_differences.csv`. Full tables, including % of oracle:
[`results/comparison.md`](results/comparison.md).

The **oracle** reads the patient's hidden state and picks, among allowed actions, the one whose true
p is near 0.85 (hardest tier, then weakest cue, within 0.06), under the same clinical masks. It is a
privileged *heuristic*, not a proven optimum: it is myopic and targets the band rather than
learning, so a policy can exceed it on learning gain or probe lift (fixed tier 2 does on moderate
patients).

Columns: time in band (observable); true control error |p - 0.85| (sim-only); learning gain over the
whole item bank (sim-only, x100); probe recall = unassisted recall on review probes (observable);
probe lift = improvement of the probed words' uncued recall since session start (sim-only, x100);
share of sessions abandoned. Bold = best non-oracle policy for that metric.

| cohort | policy | in band | \|p-0.85\| | learning gain | probe recall | probe lift | quit |
|---|---|---|---|---|---|---|---|
| mild anomic | fixed (tier 2) | 0.49 | **0.062 [0.059, 0.064]** | **3.71 [3.53, 3.87]** | 0.92 | **3.06 [2.90, 3.21]** | 0.01 [0.00, 0.01] |
| mild anomic | staircase | 0.77 | 0.094 [0.092, 0.095] | 3.62 [3.48, 3.77] | 0.88 | 2.81 [2.67, 2.96] | 0.01 [0.00, 0.03] |
| mild anomic | contextual bandit | 0.51 | 0.073 [0.071, 0.075] | 3.38 [3.22, 3.55] | 0.93 | 2.00 [1.88, 2.12] | 0.01 [0.00, 0.03] |
| mild anomic | PPO, proxy reward | **0.79** | 0.094 [0.093, 0.095] | 3.28 [3.14, 3.42] | 0.91 | 2.25 [2.13, 2.37] | **0.00 [0.00, 0.00]** |
| mild anomic | PPO, simulation objective | 0.43 | 0.078 [0.075, 0.080] | 3.18 [3.03, 3.35] | 0.93 | 2.13 [2.01, 2.26] | 0.00 [0.00, 0.01] |
| mild anomic | oracle (privileged) | 0.54 | 0.063 [0.060, 0.065] | 3.88 [3.72, 4.03] | 0.90 | 3.37 [3.22, 3.52] | 0.01 [0.00, 0.01] |
| moderate Broca's | fixed (tier 2) | 0.50 | 0.122 [0.114, 0.130] | 4.81 [4.59, 5.03] | 0.75 | **4.34 [4.16, 4.54]** | 0.04 [0.01, 0.06] |
| moderate Broca's | staircase | 0.57 | 0.143 [0.137, 0.149] | 4.59 [4.42, 4.76] | 0.75 | 3.55 [3.41, 3.68] | 0.04 [0.01, 0.07] |
| moderate Broca's | contextual bandit | 0.60 | 0.070 [0.065, 0.074] | 5.43 [5.22, 5.63] | 0.80 | 3.74 [3.59, 3.89] | 0.01 [0.00, 0.04] |
| moderate Broca's | PPO, proxy reward | **0.76** | 0.089 [0.086, 0.093] | 5.13 [4.96, 5.31] | 0.80 | 3.69 [3.55, 3.84] | **0.00 [0.00, 0.01]** |
| moderate Broca's | PPO, simulation objective | 0.62 | **0.057 [0.054, 0.061]** | **5.67 [5.47, 5.86]** | 0.82 | 4.10 [3.94, 4.25] | 0.00 [0.00, 0.01] |
| moderate Broca's | oracle (privileged) | 0.59 | 0.043 [0.041, 0.046] | 5.85 [5.64, 6.06] | 0.83 | 4.10 [3.94, 4.25] | 0.00 [0.00, 0.00] |
| severe | fixed (tier 2) | 0.02 | 0.533 [0.524, 0.541] | 0.08 [0.06, 0.10] | 0.25 | 0.15 [0.12, 0.18] | 0.79 [0.73, 0.84] |
| severe | staircase | 0.05 | 0.423 [0.414, 0.432] | 0.47 [0.41, 0.55] | 0.38 | 0.66 [0.58, 0.73] | 0.48 [0.41, 0.56] |
| severe | contextual bandit | 0.10 | 0.385 [0.375, 0.394] | 0.66 [0.58, 0.76] | 0.38 | 0.75 [0.66, 0.85] | 0.39 [0.32, 0.46] |
| severe | PPO, proxy reward | **0.13** | **0.354 [0.345, 0.363]** | **0.83 [0.72, 0.94]** | 0.42 | **1.03 [0.92, 1.14]** | **0.30 [0.24, 0.36]** |
| severe | PPO, simulation objective | 0.11 | 0.367 [0.357, 0.376] | 0.76 [0.66, 0.86] | 0.41 | 0.95 [0.86, 1.06] | 0.33 [0.27, 0.39] |
| severe | oracle (privileged) | 0.15 | 0.330 [0.321, 0.340] | 0.96 [0.85, 1.06] | 0.44 | 1.19 [1.07, 1.30] | 0.21 [0.16, 0.27] |

![Policy comparison](results/comparison.png)

A simplified slide version (severe cohort; fixed difficulty, staircase and the live agent; sessions
abandoned and naming improvement) is generated by the same command: [`results/pitch_severe.png`](results/pitch_severe.png).

![Rolling accuracy](results/rolling_accuracy.png)

What this shows, honestly:
- **Where difficulty matters most (severe patients), the adaptive policies beat fixed difficulty and
  the staircase.** Sessions abandoned: fixed 79%, staircase 48% [42, 56], contextual bandit 39%,
  proxy PPO 30% [24, 36], simulation-objective PPO 33%, oracle 22%. Probe lift: fixed 0.15, staircase
  0.66 [0.58, 0.73], proxy PPO 1.03 [0.92, 1.14]. Paired against fixed: proxy PPO probe lift +0.88
  [0.79, 0.97] and 49 points fewer abandoned sessions [42, 55]. Severe patients remain hard for every
  policy, including the oracle (rolling accuracy far below the band); one cause is the clinical rule
  "cue only after an error", whose value is to be validated clinically (`cue_only_after_error`).
- **On moderate patients the simulation-objective PPO is the best adaptive policy**: learning gain
  5.67 [5.48, 5.88] (oracle 5.85) and probe lift equal to the oracle's; paired against the
  contextual bandit +0.24 [0.18, 0.30] gain and +0.36 [0.30, 0.42] lift, against proxy PPO +0.54
  [0.48, 0.59] gain. On severe patients the two PPO rewards are within 0.1 points.
- **Proxy-trained PPO holds the band best but partly games it**: time in band 0.79 / 0.76 on mild /
  moderate patients, with a true control error 1.2-1.6 times the simulation-objective policy's (it
  steers the 8-trial rolling accuracy rather than true difficulty). Time in band is never reported
  alone.
- **On mild patients fixed tier 2 is close to the oracle on learning gain (3.71 vs 3.88)**, because
  for these patients tier 2 without a cue sits near the simulator's learning peak; adaptive policies
  pay for exploration and caution (see below).
- **Observed probe recall mostly measures how easy the practised words were**: it is high for
  policies that practise easy words (mild patients: contextual bandit and simulation-objective PPO
  0.93-0.94) while their probe lift is lower than fixed tier 2's. On a real patient probe recall is
  what can be measured, so it must be read together with the difficulty of practised words.

Earlier 50-patient runs with the Beta bandit and the proxy + probe reward (on the calibrated
simulator): [dropout on](results/sanity_dropout_on.md), [dropout off](results/sanity_dropout_off.md).

### Why fixed difficulty can win learning gain in the simulator

Fixed tier 2 is close to the oracle on mild patients (learning gain 3.71 vs 3.88) and has the highest
probe lift on moderate patients (4.34). We did not tune the simulator to reverse this. The mechanism
is visible in the "learn eff." column of the 50-patient tables (the mean expected learning per trial
implied by the learning rule at the true p; 1.0 = every trial at the peak with no cue): learning gain
tracks how much of the session is spent near the simulator's learning peak (p = 0.85) **without
cues** (successes with cues teach less, by assumption).

- For mild patients tier 2 gives p near the peak on every trial with no cue (learn eff. 0.81).
  Adaptive policies spend part of the session exploring, start on tier 1 where little is learned,
  and use cues after errors, all of which lower learning per trial.
- It is the *matching* that matters, not "fixed" (200 patients each): on mild patients fixed tier 1
  gets a learning gain of 2.63 and fixed tier 3 3.58; on moderate patients fixed tier 1 gets 5.95 (the
  best of any policy there) while fixed tier 3 gets 1.04 with 47% of sessions abandoned. Picking the
  right fixed tier for each patient requires exactly the knowledge an adaptive policy acquires;
  worksheets do not make that choice per patient.

What this **does** mean: in this simulator, with a single 40-trial session, a policy that sits at
the learning peak from the first trial is hard to beat, and exploration has a real cost. What it
**does not** mean: that fixed difficulty is a good strategy. The result depends on uncalibrated
assumptions (the location of the learning peak, the cue discount, item-specific learning, a single
session with no boredom or disengagement for items that are too easy) and reverses as soon as the
fixed tier does not match the patient (see the learning-peak ablation). A multi-session evaluation is
planned.

**Error-type x cue features.** The contextual bandit has six features crossing the last error type
(sound, meaning, no response) with the cue family (phonemic, semantic). **The simulator does not
model that interaction yet**: a patient's response to a cue does not depend on the error they just
made. Accordingly these weights fit to about 0 in the empirical-Bayes prior. They are kept because
real patients may show the dependency (for example, phonemic cues after sound errors), and they cost
little.

## Ablations (`eval/ablations.py`)

`uv run python -m eval.ablations --config configs/ablations.yaml` (about 2 minutes) reruns the
full-evaluation protocol (200 held-out patients per cohort, bootstrap CIs) under three changes.
Tables: [`results/ablations.md`](results/ablations.md).

**1. Where does learning peak?** The simulator's learning peak (the success rate at which practice
teaches most) is an assumption; 0.85 mirrors the 85% rule. Every policy is evaluated with the peak at
0.5, 0.7 and 0.85, and PPO is also retrained at each peak, once with the simulation objective (which
still contains the -|p - 0.85| control term) and once with a **learning-only** objective (no 85%
target at all).

![Learning-peak ablation](results/ablation_learning_peak.png)

- **The 85% target is load-bearing.** If learning actually peaked at 0.5, every policy that aims for
  85% would learn little with mild patients (oracle 0.39, contextual bandit 0.23, simulation-objective
  PPO trained at 0.85: 0.13, x100), while the harder-running staircase gets 1.08.
- **Optimising the true objective recovers most of the loss.** The learning-only PPO retrained at
  peak 0.5 reaches 1.91 [1.79, 2.03] on mild patients and 3.97 [3.79, 4.16] on moderate ones, the best
  of all policies there. At peak 0.85 it is close to the others (mild 3.44, moderate 5.55). This is
  the argument for learning the target from the true objective rather than fixing it at 85%: the
  agent then follows wherever learning actually peaks.
- **It is not uniformly best.** On severe patients at peak 0.5 the staircase (4.62) beats it (3.34),
  and at peak 0.7 fixed tier 2 is best on moderate patients (5.93), again because that tier happens
  to match. The retrained agents inherit every other simulator assumption, and, like all models in
  this repository, were trained on the pre-calibration simulator.

**2. Dropout on vs off.** Dropout matters only for severe patients: abandoned sessions with dropout
on are 79% for fixed tier 2, 48% for the staircase, 39% for the contextual bandit, 30-33% for PPO
and 21% for the oracle; learning gain changes by at most about 0.07 points (x100) when dropout is
switched off.

**3. Reward variants (PPO, 3 seeds each).** Band only, band + climb, full proxy and full proxy +
probe bonus are indistinguishable on every metric in every cohort (time in band 0.74-0.81 on mild and
moderate patients; learning gain within 0.15 points of each other in each cohort): the band term
drives the behaviour and the other observable terms change little. Only the simulation objective
changes it (lower control error, higher learning gain and probe lift on moderate patients, much less
time in band).

## Learning from clinician overrides: DITTO-style (`agents/ditto.py`, `train/ditto_ppo.py`)

The method follows DITTO (Shaikh et al., ICLR 2025), applied to the trained PPO policy instead of a
language model. Each override is a demonstration; it is preferred over every action the policy
itself samples in the same state, including samples from earlier rounds; the policy's masked action
log-probabilities are optimised with the DPO loss against a frozen reference (the original policy),
for 4 rounds. No LoRA is needed: the policy is a small MLP and only its actor is updated.

**Demo.** A simulated clinician with one consistent preference the policy does not follow: "after a
sound (phonological) error, give the first-sound cue" (keeping the policy's tier). 10 overrides are
taken from sessions of one set of validation patients; evaluation uses decision states from a
disjoint set of validation patients, plus full simulations on the three test cohorts. Real
overrides from the dashboard use the same pipeline (`--overrides runs/overrides.jsonl`).

**Result.** Plain DITTO-style tuning adopts the preference but **overgeneralises**: with only 10
demonstrations, nothing tells the shared network to leave other states alone, so it gives the
first-sound cue after *every* error and the policy's behaviour shifts (time in band on mild
patients 0.78 -> 0.52). We add **accepted-choice anchors**: on the dashboard, every agent choice the
clinician saw and did not override is an implicit acceptance, so a KL term keeps the policy close to
the reference on those states (weight and learning rate chosen by a small sweep scored only on the
override patients' states). Full report: [`results/ditto_report.md`](results/ditto_report.md).

| held-out states | n | top action = clinician's cue, before -> after: plain | with anchors | top action unchanged (anchors) |
|---|---|---|---|---|
| after a sound error (preference applies) | 424 | 0.36 -> 1.00 | **0.36 -> 0.89** | 0.45 |
| after other errors (cues allowed, preference does not apply) | 488 | 0.31 -> 0.99 | **0.31 -> 0.36** | 0.87 |
| all other states | 2700 | 0.00 -> 0.00 | 0.00 -> 0.00 | **0.99** |

(Calibrated simulator; before calibration the anchored adoption was 0.96.) With anchors, after 10
overrides the policy makes the clinician's choice in 89% of matching held-out states, leaves states
after other errors almost unchanged (0.31 -> 0.36), and keeps 99% of all other decisions identical; time in band, control error, learning gain and probe recall on the three test cohorts are
unchanged within noise. The anchors are our addition to DITTO and are only as good as the assumption
that a non-override means acceptance.

**Live endpoint, validated on fresh states.** The settings of `POST /learn_from_overrides` (one round,
300 gradient steps, learning rate 0.002) were chosen by looking at override and evaluation states from
validation patients 100+ of the moderate cohort. To check them honestly,
`scripts/validate_live_ditto.py` repeats the update 5 times with 10 overrides each from *different*
validation patients (1000+) and evaluates on states from yet other patients (2000+), for both PPO
models and two cohorts ([`results/live_ditto_validation.md`](results/live_ditto_validation.md)):

| model | cohort | P(clinician's cue) after a sound error, before -> after | top action = clinician's cue, before -> after | other state types, top action unchanged |
|---|---|---|---|---|
| PPO, proxy reward (live agent) | moderate Broca's | 0.35 -> 0.93 ± 0.04 | 0.38 -> 0.95 ± 0.04 | 0.97 |
| PPO, proxy reward (live agent) | population | 0.35 -> 0.90 ± 0.04 | 0.36 -> 0.93 ± 0.04 | 0.97 |
| PPO, simulation objective | moderate Broca's | 0.04 -> 0.89 ± 0.04 | 0.04 -> 0.90 ± 0.04 | 0.99 |
| PPO, simulation objective | population | 0.10 -> 0.80 ± 0.13 | 0.10 -> 0.84 ± 0.13 | 0.99 |

The effect holds on fresh states (calibrated simulator; top action switches in 84-95% of matching
states, ± over the 5 override sets). For the live agent 3% of other state types change their top
action, slightly more than the 0% seen on the states used for tuning. Each update takes well under a
second on CPU (0.3-0.6 s, depending on machine load).

## Known limitations
- With an 8-trial window, rolling accuracy only takes values k/8, so the [0.75, 0.90] band means
  exactly 6/8 or 7/8. Even a perfect controller holding p = 0.85 is in band only ~62% of the time
  (Binomial(8, 0.85)). This is why we also report true control error, and will report % of a privileged oracle.
- The simulator is a parametric model. 14 group-level statistics are calibrated against published
  data (Simulator validity); the learning rule, fatigue magnitudes, dropout and cue mixes remain
  assumptions to be validated clinically. All trained models (PPO, the bandit's fitted prior) were
  trained before the calibration and evaluated on the calibrated simulator without retraining;
  retraining on it is the next step (the live demo model was deliberately left unchanged).
- One action per trial applies the cue to the *next* item; real cueing happens within an item after
  a failed attempt. Kept for the agreed interface.

## Reproducibility
- Every random source takes a seed. Service sessions derive their seed from
  `(configs/bandit.yaml seed, session_id)` via crc32, so replaying the same requests reproduces the
  same actions (tested in `tests/test_service.py::test_full_session_replays_identically`).
- All tunable values live in `configs/*.yaml` and are validated at load time (`core/config.py`).

## Citations
Every paper cited in code or configs is listed here so it can be checked. Papers are cited only for
the **direction** of an effect; specific numeric values are marked `ASSUMPTION` in the configs.

| cited for | reference | where |
|---|---|---|
| 85% target accuracy (theoretical, not shown for aphasia) | Wilson, Shenhav, Straccia & Cohen (2019). The Eighty Five Percent Rule for optimal learning. *Nature Communications* 10, 4646. | `configs/clinical.yaml` |
| Phonemic cues have larger immediate effects than semantic cues; cue-effect and between-patient SD calibration targets | Meteyard & Bose (2018). What does a cue do? Comparing phonological and semantic cues for picture naming in aphasia. *JSLHR* 61(3), 658–674. | `configs/clinical.yaml`, `configs/sim.yaml`, `data/calibration_targets.yaml` |
| Discounted Thompson sampling for non-stationary bandits | Raj & Kalyani (2017). Taming non-stationary bandits: A Bayesian approach. arXiv:1707.09727. | `configs/bandit.yaml` |
| Word frequency and length predict naming accuracy in aphasia | Nickels & Howard (1995). Aphasic naming: What matters? *Neuropsychologia* 33(10), 1281–1303. | `configs/sim.yaml` (items) |
| Naming therapy gains are largely item-specific, limited generalisation | Nickels (2002). Therapy for naming disorders: Revisiting, revising, and reviewing. *Aphasiology* 16(10–11), 935–979. | `configs/sim.yaml` (learning) |
| Error types reflect semantic vs phonological impairment (two-step model) | Dell, Schwartz, Martin, Saffran & Gagnon (1997). Lexical access in aphasic and nonaphasic speakers. *Psychological Review* 104(4), 801–838. | `configs/sim.yaml` (errors) |
| Case-series evidence for the two-step model in picture naming | Schwartz, Dell, Martin, Gahl & Sobel (2006). A case-series test of the interactive two-step model of lexical access. *Journal of Memory and Language* 54(2), 228–264. | `configs/sim.yaml` (errors) |
| 2-up/1-down staircase converges near 70.7% correct | Levitt (1971). Transformed up-down methods in psychoacoustics. *JASA* 49(2B), 467–477. | `agents/baselines.py` |
| Thompson sampling with a Laplace-approximated logistic model | Chapelle & Li (2011). An empirical evaluation of Thompson sampling. *NeurIPS* 24. | `agents/logistic_bandit.py` |
| Word frequencies for the placeholder item bank | Speer (2022). wordfreq (software), v3. https://github.com/rspeer/wordfreq | `scripts/build_placeholder_items.py` |
| Proximal policy optimisation | Schulman, Wolski, Dhariwal, Radford & Klimov (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347. | `train/train_ppo.py` |
| Invalid-action masking in policy-gradient methods | Huang & Ontañón (2022). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. *FLAIRS-35*. | `envs/aphasia_env.py` |
| PPO implementation | Raffin et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR* 22(268). | `train/train_ppo.py` |
| Group-relative policy optimisation (GRPO), explored only | Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300. | `train/grpo_smoke.py` |
| Group-level picture-naming accuracy, error mix and frequency effect (calibration targets) | Kittredge, Dell, Verkuilen & Schwartz (2008). Where is the effect of frequency in word production? Insights from aphasic picture-naming errors. *Cognitive Neuropsychology* 25(4), 463-492. | `data/calibration_targets.yaml` |
| Per-patient PNT data by aphasia type (calibration targets) | Dell, Martin & Schwartz (2007). A case-series test of the interactive two-step model of lexical access: Predicting word repetition from picture naming. *Journal of Memory and Language* 56(4), 490-520. | `data/calibration_targets.yaml` |
| Within-session fatigue in naming (direction) | Mirman, Krason, Thothathiri & Middleton (2024). Effect of fatigue on word production in aphasia. *Proceedings of CogSci* 46. | `data/calibration_targets.yaml` |
| Learning from demonstrated feedback (DITTO), method followed for override learning | Shaikh et al. (2025). Show, Don't Tell: Aligning Language Models with Demonstrated Feedback. ICLR 2025. | `agents/ditto.py` |
| Direct preference optimisation (DPO loss) | Rafailov, Sharma, Mitchell, Ermon, Manning & Finn (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS* 36. | `agents/ditto.py` |
