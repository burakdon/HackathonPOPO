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

Implemented and tested (91 tests, `uv run pytest`):
- Agent service (FastAPI) implementing the agreed State/Action interface, with one online-learning
  agent per session, clinician overrides logged for later training, and CORS for a frontend on the
  local network.
- Patient simulator: population sampling with a held-out test split, three evaluation cohorts,
  item difficulty from word frequency and length, patient-specific cue responsiveness, error types,
  fatigue with recovery, and learning with a configurable peak. The item bank is a placeholder.
- Therapist agents: a contextual Bayesian logistic Thompson-sampling bandit with a prior fitted on
  simulated training patients (the live agent), a Beta-Bernoulli Thompson bandit, and fixed-difficulty
  and staircase baselines.
- Preliminary evaluation: four policies on three cohorts, 50 held-out patients each (see below).

In progress or planned:
- Patient app and clinician dashboard (`frontend/`).
- Speech scoring and error classification.
- Trained agent: reinforcement learning in the simulator (masked PPO) and a fine-tuned small
  language model.
- Full evaluation: privileged oracle, 200+ patients per cohort, several training seeds, bootstrap
  confidence intervals, and reward and learning-peak ablations.
- Expert alignment from clinician overrides.
- A clinically curated item bank and clinical validation of the simulator's parameters.

## Quick start

```bash
uv sync                                         # Python 3.11 env from uv.lock
uv run pytest                                   # test suite
uv run uvicorn service.app:app --host 0.0.0.0 --port 8000   # agent service, reachable on the LAN
uv run python -m scripts.service_smoke --url http://localhost:8000   # 40-trial session over HTTP
uv run python -m train.fit_bandit_prior         # refit the bandit's empirical-Bayes prior (~2 s)
uv run python -m scripts.sanity_table           # preliminary policy comparison (results/sanity_preliminary.md)
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
sim/       simulated patient population (Step 1b)
envs/ train/ eval/   Gymnasium env, training scripts, evaluation harness (later steps)
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

**Changes from the original contract (both additive and backward-compatible):**
1. `State` has an optional `session_id`. The service keeps one online-learning agent per session;
   requests without it share a single `default` session. **Please always send it** (one id per
   patient session, any string up to 128 chars).
2. `POST /update` takes an explicit `outcome` object (shape below).

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
| `current_tier` | int | 1..3, the tier of the most recent trial (the agent plans from this) |
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

### Endpoints
| method | path | body | returns |
|---|---|---|---|
| GET | `/health` | none | `{"status": "ok", "agent": "...", "active_sessions": n}` |
| POST | `/act` | `State` | `Action` |
| POST | `/update` | `{"state": State, "action": Action, "outcome": Outcome}` | `{"status": "ok", "session_id": "..."}` |
| POST | `/override` | `{"state": State, "agent_action": Action, "clinician_action": Action, "note": "optional"}` | `{"status": "stored", "session_id": "...", "n_overrides": n}` |

Session loop for the frontend:
1. `POST /act` with the current `State` and show the returned tier and cue.
2. If the clinician overrides, `POST /override` with both actions and show the clinician's action.
3. After the patient answers, `POST /update` with the same `State`, the action **actually shown**
   (the clinician's if overridden), and the `Outcome`.
4. Append the trial to `last_trials` and repeat.

Overrides are stored as JSON lines in `runs/overrides.jsonl` (the input to the DITTO step).
Interactive API docs: `http://localhost:8000/docs` while the service runs.

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
| item difficulty | `beta_f * (-z(frequency)) + beta_l * z(length)` | frequency and length effects on naming (Nickels & Howard 1995) |
| cues | `phonemic_weight[cue] * phon_resp + semantic_weight[cue] * sem_resp`, per-patient responsiveness | phonemic > semantic on average, with phonemic vs semantic responders (Meteyard & Bose 2018) |
| error types | after a failure, P(sound, meaning, no response) proportional to (phon_impair, sem_impair, omission_bias) | simplified from the two-step model (Dell et al. 1997; Schwartz et al. 2006) |
| fatigue | grows per trial and faster during failure streaks, recovers a fraction each trial, capped | ASSUMPTION |
| learning | on a success, item-specific gain with E[gain] = p^a (1-p)^b, peaking at a configurable `peak`; weaker cues teach more; 5% spills over to general ability | item-specific gains with limited generalisation (Nickels 2002); the peak location is an ASSUMPTION and is ablated in Step 3 |
| response time | log-normal, slower for hard items and when fatigued | ASSUMPTION |
| dropout (optional) | quit hazard rising after 3+ failures | ASSUMPTION, off by default |

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

## Agents (`agents/`)

All agents implement one interface (`act(state, mask) -> Action`, `update(state, action, outcome)`)
and decide from the State JSON only.

- **Fixed difficulty** (worksheets today) and **2-up/1-down staircase** (Levitt 1971).
- **Beta-Bernoulli Thompson bandit**: 15 independent arms, ported from the scaffold and bug-fixed.
- **Contextual Bayesian logistic Thompson bandit** (the live agent): `p(correct) = sigmoid(w . phi)`
  with 17 features (tier, cue, recent accuracy, fail streak, progress, response time, and last error
  type x cue family). Posterior updated online with the closed-form Laplace/extended-Kalman step,
  with random-walk process noise so it tracks fatigue and learning. Weights are shared across
  actions, so one observation informs many actions.
- **Empirical-Bayes prior** for the contextual bandit, fitted on 300 simulated *training* patients
  (`train/fit_bandit_prior.py`): the per-patient weights' mean and shrunk covariance. This is the
  bandit's sim-to-real pretraining. It is inflated so a real person's data still dominates: in a test,
  a patient far from the population moves the prediction from 0.85 to 0.44 in 30 trials (truth 0.40).
- **Shared selection rule** (`agents/selection.py`): among actions whose predicted p is within 0.06
  of the 0.85 target, take the hardest tier, then the weakest cue; otherwise the closest to target.
  This fixes the tier-1 coasting seen in Step 1a (37/40 trials on tier 1).

## Preliminary results (Step 1 sanity check)

50 held-out patients per cohort, 40 trials, mean ± 95% CI (normal approximation). Full evaluation
with the oracle, PPO, bootstrap CIs and 200+ sessions comes in Step 3. Columns: time in band (rolling
accuracy over 8 trials in [0.75, 0.90]), accuracy, mean tier, share of trials on tier 1, learning gain
(sim-only, x100), true control error |p_true - 0.85| (sim-only), frustration events (3-failure streaks).
Full table: [`results/sanity_preliminary.md`](results/sanity_preliminary.md).

| cohort | policy | in band | acc | mean tier | % tier 1 | gain x100 | \|p-0.85\| | frustration |
|---|---|---|---|---|---|---|---|---|
| mild anomic | fixed (tier 2) | 0.50 ± 0.06 | 0.90 | 2.00 | 0.00 | 3.33 ± 0.29 | 0.066 | 0.02 |
| mild anomic | staircase | **0.79 ± 0.03** | 0.79 | 2.29 | 0.15 | 2.98 ± 0.23 | 0.143 | 0.10 |
| mild anomic | Beta bandit | 0.41 ± 0.07 | 0.94 | 1.38 | 0.63 | 2.39 ± 0.24 | 0.095 | 0.00 |
| mild anomic | contextual bandit (fitted prior) | 0.52 ± 0.06 | 0.91 | 1.77 | 0.29 | 2.89 ± 0.27 | 0.085 | 0.00 |
| moderate Broca's | fixed (tier 2) | 0.52 ± 0.06 | 0.74 | 2.00 | 0.00 | 4.70 ± 0.39 | 0.109 | 0.64 |
| moderate Broca's | staircase | 0.53 ± 0.04 | 0.70 | 1.87 | 0.34 | 4.02 ± 0.27 | 0.169 | 0.28 |
| moderate Broca's | Beta bandit | 0.61 ± 0.05 | 0.84 | 1.28 | 0.72 | 4.75 ± 0.33 | **0.068** | 0.14 |
| moderate Broca's | contextual bandit (fitted prior) | **0.63 ± 0.05** | 0.83 | 1.42 | 0.60 | 4.80 ± 0.32 | 0.069 | 0.24 |
| severe | fixed (tier 2) | 0.02 ± 0.01 | 0.29 | 2.00 | 0.00 | 0.12 ± 0.05 | 0.566 | 3.80 |
| severe | staircase | 0.08 ± 0.02 | 0.49 | 1.25 | 0.77 | 1.32 ± 0.24 | 0.362 | 2.30 |
| severe | Beta bandit | 0.20 ± 0.05 | 0.55 | 1.30 | 0.71 | 1.67 ± 0.27 | 0.312 | 1.70 |
| severe | contextual bandit (fitted prior) | **0.27 ± 0.05** | 0.59 | 1.14 | 0.86 | **1.99 ± 0.32** | **0.263** | **1.38** |

What this shows, honestly:
- The contextual bandit has the lowest or near-lowest true control error in every cohort and the
  fewest frustration events among adaptive policies, and is best on time in band for moderate and
  severe. It does **not** win everywhere.
- **The staircase wins time in band on mild patients** (0.79 vs 0.52) while having the *worst*
  control error (0.143): oscillating between tiers keeps the 8-trial rolling accuracy inside the band
  without the true p ever being near 0.85. Time in band alone is a weak metric; we report it with
  control error and learning gain.
- **Fixed tier 2 has the highest learning gain on mild and moderate** patients: with only 20 items per
  tier, repeating the same tier gives more repetitions per item. Learning gain depends strongly on
  simulator assumptions (peak location, item-specific learning), which Step 3 ablates.
- **Severe patients are hard for every policy.** One cause is the clinical rule "cue only after an
  error" (inherited from the scaffold): their best action (tier 1 + first sound, p~0.84) cannot be
  repeated after a success. With the rule off, the contextual bandit's time in band rises from 0.56
  to 0.67 on mild and 0.20 to 0.28 on severe, but falls from 0.60 to 0.56 on moderate (hand prior).
  Whether to keep the rule is to be validated clinically (`cue_only_after_error`).
- The Beta bandit still spends 63% of mild sessions on tier 1 despite the progression rule: its
  discounting pulls unvisited arms back to the tier-1 prior mean 0.80, which sits inside the
  tolerance window. Shared weights in the contextual model avoid this.
- The six error-type x cue interaction weights fit to about 0: the simulator does not make cue
  effectiveness depend on the last error type, so these features have nothing to learn here. They
  are kept because a real patient may show the dependency.

## Known limitations
- With an 8-trial window, rolling accuracy only takes values k/8, so the [0.75, 0.90] band means
  exactly 6/8 or 7/8. Even a perfect controller holding p = 0.85 is in band only ~62% of the time
  (Binomial(8, 0.85)). This is why we also report true control error and, in Step 3, % of oracle.
- The simulator is a parametric model; magnitudes are assumptions pending clinical review. The
  bandit's fitted prior inherits these assumptions.
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
| Phonemic cues have larger immediate effects than semantic cues | Meteyard & Bose (2018). What does a cue do? Comparing phonological and semantic cues for picture naming in aphasia. *JSLHR* 61(3), 658–674. | `configs/clinical.yaml` |
| Discounted Thompson sampling for non-stationary bandits | Raj & Kalyani (2017). Taming non-stationary bandits: A Bayesian approach. arXiv:1707.09727. | `configs/bandit.yaml` |
| Word frequency and length predict naming accuracy in aphasia | Nickels & Howard (1995). Aphasic naming: What matters? *Neuropsychologia* 33(10), 1281–1303. | `configs/sim.yaml` (items) |
| Naming therapy gains are largely item-specific, limited generalisation | Nickels (2002). Therapy for naming disorders: Revisiting, revising, and reviewing. *Aphasiology* 16(10–11), 935–979. | `configs/sim.yaml` (learning) |
| Error types reflect semantic vs phonological impairment (two-step model) | Dell, Schwartz, Martin, Saffran & Gagnon (1997). Lexical access in aphasic and nonaphasic speakers. *Psychological Review* 104(4), 801–838. | `configs/sim.yaml` (errors) |
| Case-series evidence for the two-step model in picture naming | Schwartz, Dell, Martin, Gahl & Sobel (2006). A case-series test of the interactive two-step model of lexical access. *Journal of Memory and Language* 54(2), 228–264. | `configs/sim.yaml` (errors) |
| 2-up/1-down staircase converges near 70.7% correct | Levitt (1971). Transformed up-down methods in psychoacoustics. *JASA* 49(2B), 467–477. | `agents/baselines.py` |
| Thompson sampling with a Laplace-approximated logistic model | Chapelle & Li (2011). An empirical evaluation of Thompson sampling. *NeurIPS* 24. | `agents/logistic_bandit.py` |
| Word frequencies for the placeholder item bank | Speer (2022). wordfreq (software), v3. https://github.com/rspeer/wordfreq | `scripts/build_placeholder_items.py` |
| Learning from demonstrations (DITTO) | Shaikh et al. (2025). Show, Don't Tell: Aligning Language Models with Demonstrated Feedback. ICLR 2025. | planned (Step 5) |
