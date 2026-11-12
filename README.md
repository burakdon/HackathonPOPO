# NeuroCue

**A clinically grounded synthetic aphasia patient, a therapist policy trained against it, and a loop
that lets the clinician correct that policy.** Winner of the Health track and 3rd overall at the
Triangle AI Hackathon (November 2026).

Anomia therapy is a sequence of decisions: which word next, which cue, retry or move on, how to space
practice. Reinforcement learning could optimise those decisions, but it needs a patient to practise
on, and no simulator existed that was grounded in real naming errors and real retention data.
NeuroCue builds one. Every layer of the patient is fitted to published data and checked on held-out
data; the therapy environment, the trained policy, the clinician-feedback loop and the live console
all run on that same engine.

| | |
|---|---|
| Hosted demo | **https://burakdonbekci--neurocue-web.modal.run** (first visit after idle waits about a minute while the engine loads) |
| Video walkthrough | [results/demo/neurocue_demo.mp4](results/demo/neurocue_demo.mp4) (2.5 min, captioned) |
| Presentation notes | [docs/presentation_talking_points.md](docs/presentation_talking_points.md): every number with its source file |
| Engine internals | [patientmodel/README.md](patientmodel/README.md): module-by-module grounding |
| Data sources | [data/README.md](data/README.md): what each dataset is for, licences, what could not be obtained |
| Repository | https://github.com/burakdon/neurocue (`main`; the original scaffold is `legacy-main`) |

> **Prototype, not a clinical tool.** No claim of clinical effectiveness, diagnosis or assessment of
> impaired speech is made. No real patient is simulated: the patient is a model fitted to population
> data. The policy's results are the engine's current opinion under stated assumptions, not a verdict
> on any therapy. No real patient data is used in the demo.

## Contents

1. [Quick start](#1-quick-start)
2. [What is in the repository](#2-what-is-in-the-repository)
3. [The patient engine](#3-the-patient-engine-patientmodel)
4. [The therapy environment and the trained therapist](#4-the-therapy-environment-and-the-trained-therapist-neurocue)
5. [Learning from clinician corrections: DITTO with anchors](#5-learning-from-clinician-corrections-ditto-with-anchors)
6. [The console and the demo](#6-the-console-and-the-demo)
7. [Deployment](#7-deployment)
8. [Reproducing every result](#8-reproducing-every-result)
9. [Limits, stated](#9-limits-stated)
10. [Earlier generations kept in the repository](#10-earlier-generations-kept-in-the-repository)
11. [Data and licences](#11-data-and-licences)
12. [Branches and team workflow](#12-branches-and-team-workflow)
13. [References](#13-references)

## 1. Quick start

Everything the demo needs is committed: code, trained policies, picture norms and pictures, and the
caches (fitted model grid, lexicon, WordNet). One command:

```bash
./scripts/run_demo.sh
```

It installs `uv` if you do not have it, creates the Python 3.11 environment (first time: a few
minutes, mostly PyTorch), starts the server and opens http://localhost:8765. Use Chrome or Safari
with sound on for the spoken responses. Stop with Ctrl-C. Another port:
`NEUROCUE_PORT=9000 ./scripts/run_demo.sh`.

The test suite (250 tests, engine, environment, service and the earlier generations):

```bash
uv run python -m pytest
```

Not in the branch: the PSST transcripts and audio (TalkBank data) and the papers' PDFs. Neither is
needed to run the demo or the tests; `uv run python -m scripts.fetch_data` downloads the public
datasets for the fitting and validation scripts.

## 2. What is in the repository

The system is five layers. Each one consumes the one before it and is validated on its own.

| layer | code | what it does | fitted to / checked against |
|---|---|---|---|
| Data and labels | `scripts/code_psst_responses.py`, `scripts/learn_coder.py`, `data/derived/` | Codes real naming responses into error types; a clinician adjudicated the uncertain rows; a learned coder fills the rest | PSST corpus (Gale et al. 2022); 328 clinician-coded rows |
| Patient engine | `patientmodel/` | A mechanistic model of word retrieval, a population prior, real words and nonwords as errors, cues, learning and forgetting over days | Dell's published patients; PSST held-out participants; Middleton et al. 2019 retention data; Nardo et al. 2017 cue effects |
| Therapy environment | `neurocue/env.py`, `neurocue/baselines.py` | A Gymnasium course of four sessions on one engine patient, judged on retention one week and one month later; six clinical baselines | Middleton et al. 2019's protocol |
| Trained therapist | `neurocue/train_ppo.py`, `neurocue/evaluate.py`, `models/engine_ppo_*.zip` | MaskablePPO on the environment; paired evaluation against the baselines | 80 held-out synthetic patients |
| Clinician loop | `agents/ditto.py`, `neurocue/demo/server.py` | A clinician's correction becomes a demonstration; DITTO with anchors updates the policy where the rule applies and nowhere else | Held-out states: adoption and containment |
| Console | `neurocue/demo/` | FastAPI over the engine, an avatar whose mouth follows the phonemes produced, diagnosis, two-therapist course, teach-a-rule | Same code the validation ran |

Repository layout:

```
patientmodel/   the synthetic patient engine (Dell SP model, fitting, population, words, cues, learning)
neurocue/       therapy environment, baselines, PPO training and evaluation, the demo console (neurocue/demo/)
agents/         therapist policies behind one interface; agents/ditto.py is the DITTO-with-anchors update
core/           shared schema (State/Action/Outcome), typed config loading, action masks
service/        FastAPI agent service of the earlier generations (section 10)
sim/            the first-generation logistic patient simulator (section 10)
envs/           the first-generation Gymnasium environment (section 10)
train/          first-generation PPO, bandit-prior fitting, DITTO on PPO, GRPO exploration
training/       DITTO on a language model for the CueAgain practice app (section 10)
evaluation/     comparison of language-model decision policies (section 10)
speech/         local Whisper transcription for the practice app (section 10)
eval/           first-generation metrics, figures and ablations
scripts/        data download, PSST coding, coder learning, Dell refits, prior and cue calibration,
                validation, demo recording, deployment
configs/        every tunable value, YAML, validated at load time; assumptions are marked ASSUMPTION
data/           item banks, calibration tables, coded PSST rows (raw third-party data is not committed)
models/         trained policies, committed so results reproduce without retraining
results/        generated reports (markdown, CSV, PNG) that this README quotes; results/demo/ holds the video
deploy/         Modal app and the Hugging Face Space scaffold; Dockerfile at the root
frontend/       the CueAgain practice app and clinician view (section 10)
tests/          pytest suite
docs/           presentation notes; the original starter scaffold under docs/scaffold/
```

## 3. The patient engine (`patientmodel/`)

Two cognitive parameters and a severity level produce a patient who names pictures, makes the error
types real patients make in the proportions real patients make them, says actual words or nonwords,
is harder to understand on harder pictures, and, with the learning layer, gets better with practice
the way Middleton's patients did: more from effortful retrieval than from repetition, more from
spaced than massed practice, and forgetting over the following weeks. Everything is seeded and
reproducible; a 4,000-trial patient simulates in about 80 ms. The module table with per-module
grounding is in [patientmodel/README.md](patientmodel/README.md); this section follows the data
through the layers.

### 3.1 Real data: PSST

The Post-Stroke Speech Transcription corpus (Gale et al. 2022; TalkBank/AphasiaBank): 3,291 naming
utterances from 107 sessions of about 75 people with aphasia on the Boston Naming Test and Verb
Naming Test, with ARPAbet transcripts and WAB-R aphasia quotients. PSST labels correctness only, so
error *types* had to be built. The 1,074 BNT noun rows drive the model: 74 training-split
participants fit it, 33 validation/test participants check it and are never used for fitting.

### 3.2 Manual labelling and the learned response coder

`scripts/code_psst_responses.py` machine-codes every PSST response into Philadelphia Naming Test
style types (correct, semantic, formal, mixed, unrelated, nonword, fragment, multiword) with a
confidence, and writes the low-confidence rows out for a clinician
([data/derived/CODING_GUIDE.md](data/derived/CODING_GUIDE.md)). A speech-language pathologist on
the team adjudicated 278 uncertain noun rows and then a 50-row audit: 328 clinician-coded responses
in all. Her shorthand on the sheets: "y" means the production was effectively correct, "xxx" means
the speaker said a different word that is itself a test item.

The labels changed the model. On the uncertain rows the rule-based coder agreed with her only 32%
of the time, and only 3% of its "formal" codes survived (most were semantic errors or nonwords).
The audit checked 30 rows the machine had coded with high confidence: all 10 semantic calls held,
17 of 20 phonological-nonword calls stayed nonwords (8 as fragments), and 3 were productions she
judged effectively correct that PSST had scored wrong
([results/psst_coder_agreement.md](results/psst_coder_agreement.md)).

`scripts/learn_coder.py` trains a random forest on 42 features of each target/response pair
(normalised edit distance, MiniLM embedding cosine, WordNet similarity, length difference,
shared-position ratio, the machine's own code) on the 328 clinician rows. Cross-validated leaving
one target picture out, it is right 51% of the time on nine categories and 68% on the five
semantic-phonological error types, against 35% and 42% for the rules; on the 20 audited rows it had
coded it was right on 8. It then recoded the 91 remaining uncertain rows
([results/psst_coder_learned.md](results/psst_coder_learned.md)). Precedence downstream:
clinician > learned coder > rules.

### 3.3 The retrieval model: Dell's semantic-phonological model (`patientmodel/dell_sp.py`, `patientmodel/fit.py`)

The interactive two-step model of lexical access (Dell, Schwartz, Martin, Saffran and Gagnon 1997;
Foygel and Dell 2000; Dell et al. 2004, 2013) with two lesion weights, **s** (semantic to lexical)
and **p** (lexical to phonological). Six-word neighbourhood, 57 semantic units, 10 phonemes, decay
.6, two noise terms, 8 steps per stage; returns the six PNT response categories. The implementation
reproduces the proportions Dell published for his fitted patients to rmsd .002 to .011 and names 97%
correctly at normal weights (`tests/test_dell_sp.py`).

`patientmodel/fit.py` fits (s, p) to a patient's naming counts: a cached simulated probability grid keyed on the
full model configuration, Laplace-smoothed multinomial likelihood, Dell's rmsd, a grid posterior
with 95% intervals and a boundary flag. Refitting Dell's 2004 and 2013 patients from their published
proportions recovers his s and p to .002 on the 14 interior fits; the one near-normal patient sits
at the grid edge and is flagged (`scripts/fit_dell_sp.py`,
[data/calibration/dell2004_table2_sp_fits.csv](data/calibration/dell2004_table2_sp_fits.csv)).
The same fitter is the diagnosis shown in the console.

### 3.4 Population prior fitted to real patients (`patientmodel/population.py`)

A patient is sampled by AQ severity, a target accuracy from a logistic severity curve with
between-patient heterogeneity (fitted on the 74 training participants' BNT noun items, heterogeneity
estimated after removing binomial noise), and then an (s, p) cell consistent with that accuracy under
a lognormal prior on p/s. Profiles mild / moderate / severe / very severe.

Dell's Philadelphia prior (log p/s mean +0.5, sd 0.6) did not match PSST, so the prior was fitted by
multinomial likelihood on the training participants' coded error types
(`scripts/fit_population_prior.py`): **log p/s mean −0.2, sd 0.3** (deviance 164.8 against 279.5
for Dell's values; his prior is kept as `PopulationConfig.dell_philadelphia()`). On the 33 held-out
participants (154 codable errors) the fitted engine's nonword share is 0.56 against a real 0.53,
where Dell's prior gives 0.33 ([results/population_prior_fit.md](results/population_prior_fit.md)).

Held-out check of the severity mapping ([results/engine_validation.md](results/engine_validation.md)),
BNT items only, matched synthetic cohort, 10 replicates:

| AQ band | n real | real accuracy | synthetic accuracy |
|---|---|---|---|
| < 50 | 5 | 0.30 | 0.30 |
| 50 to 69 | 12 | 0.35 | 0.44 |
| 70 to 84 | 8 | 0.45 | 0.58 |
| 85+ | 8 | 0.65 | 0.66 |

With 15 items per participant the real values carry binomial noise of about 0.12. Correlation of
accuracy with AQ: real 0.52, synthetic 0.46. What the prior cannot fix is the word-error split: real
patients make semantic 0.37, formal 0.01, unrelated 0.07 of their errors; the engine makes 0.14,
0.17, 0.11 at any prior setting. That comes from the SP model's six-word neighbourhood (two formal
and two unrelated competitors against one semantic), a known approximation of the published model.
Item difficulty is not the cause (synthetic nonword share on the 13 BNT items moved only 0.02).

### 3.5 Words, pictures and how the patient talks (`patientmodel/items.py`, `patientmodel/lexicon.py`, `patientmodel/realize.py`, `patientmodel/utterance.py`, `patientmodel/patient.py`)

- **Pictures.** 520 International Picture Naming Project object pictures with US English norms and
  a difficulty index (healthy RT, frequency, length; name agreement deliberately excluded); 244
  drawings; 39 compound-name items flagged `pronounceable=False` (Bates et al. 2003; Szekely et al.
  2004).
- **Lexicon.** About 4,800 picturable nouns: concreteness ≥ 4 (Brysbaert et al. 2014), WordNet
  noun, CMUdict pronunciation, not on the profanity or sensitive-word lists; Zipf frequency
  (wordfreq); MiniLM embeddings; the PNT formal-relatedness criterion; English phonotactic legality.
- **Realisation.** A response category becomes an actual response: semantic neighbours need
  embedding similarity plus a WordNet taxonomic relation; formal neighbours meet the PNT criterion
  and are not semantic; mixed need both and a shared onset or ≤ 2 edits; unrelated are frequent
  nouns sharing nothing; nonwords are length-scaled, class-preserving phoneme edits that are not any
  English word and are phonotactically legal. Fallbacks are recorded in `fallback_from`.
- **Utterance frames.** Carrier phrases, fillers, false starts and trailing words ("uh … house",
  "okay that's a … home", "k- … crab") are sampled by severity band from counts mined off the PSST
  transcripts (`scripts/mine_utterance_frames.py` →
  `data/calibration/psst_utterance_frames.json`; frame rates 1% to 12% of responses, a lower bound
  since clips are trimmed). The response word itself is untouched. This drives the avatar's speech.
- **Item effects.** `Patient.name(item)` uses per-item effective weights
  `(s, p) · exp(−item_beta · difficulty)`; the mechanism and strength are assumptions. Durations are
  PSST clip lengths, never reported as latency.

### 3.6 Learning and forgetting over days (`patientmodel/learning.py`, `patientmodel/treatment.py`)

Item-level learning is ACT-R activation with per-trace decay that depends on activation at practice
(Pavlik and Anderson 2005), which produces the spacing effect mechanically rather than by rule, plus
trace weights for successful retrieval, failed retrieval with feedback, and errorless study (the
retrieval-practice effect). `TreatablePatient` keeps one memory per word; baseline retrievability
equals the snapshot's P(correct) for that item; `attempt`, `study`, `probe` and `session` schedule
practice over days. Errors on failed retrieval still come from the SP model conditioned on an error.

Fitted to Middleton, Rawson and Verkuilen 2019 by simulating their protocol (four people with
aphasia, 192 items each, retrieval practice versus repetition, spaced versus massed, tests at 7 and
28 days; 56 observed cells): trace weights **unaided success 1.00, failure followed by the word
0.59, study 0.58**. All 14 pooled condition × delay cells are within 0.09 (worst 0.08); the spacing
effect, the retrieval-practice advantage, the rise across sessions and the forgetting between one
week and one month are reproduced. Individual differences in the retrieval-practice effect are not
captured ([results/learning_fit.md](results/learning_fit.md)).

### 3.7 Cues as activation (`patientmodel/cues.py`)

A first sound (the target's onset unit), a last sound (the coda unit, shared with the formal
neighbours), a meaning cue (the target's semantic features) or the whole word (all three phoneme
units plus the word node, Dell 2013's dual-route repetition) enter the SP model as external
activation. `Cue.effort` is the share of the retrieval the cue does; the learning layer weights a
cued success between an unaided success and errorless study by it (direction from Howard et al.
1985 and Middleton et al. 2019; the values are assumptions, not fits).

Levels were calibrated on a synthetic cohort with unaided accuracy near Nardo et al. 2017's 0.60 so
that the share of otherwise-failed items rescued matches theirs
([results/cue_calibration.md](results/cue_calibration.md)):

| cue | rescue share, engine | Nardo et al. 2017 |
|---|---|---|
| initial phoneme | 0.70 | 0.70 |
| whole word | 0.82 | 0.825 |
| final phoneme | 0.24 (best reachable) | 0.475 |

The final-phoneme cue cannot be reproduced because both formal competitors in the six-word
neighbourhood share the target's coda, so a coda cue lifts them as much as the target: a stated
limit of the model's three-slot phonology. Two predictions the fit did not touch: initial-phoneme
rescue falls with severity (0.79 mild to 0.46 severe), matching Pease and Goodglass 1978's direction
and their 49% figure for severe patients; and the semantic cue helps semantic lesions most (0.42
against 0.15), the opposite of Python et al. 2021, so that mechanism is flagged as suspect and its
level is an assumption anyway.

## 4. The therapy environment and the trained therapist (`neurocue/`)

### 4.1 Formulation

A Gymnasium environment on the engine (`neurocue/env.py`, settings in
[configs/engine_env.yaml](configs/engine_env.yaml)). It keeps the repository's patterns from the
first generation (observation from observables only, hard action masks, MaskablePPO, paired
evaluation with common random numbers) and changes the problem to Middleton et al. 2019's design:

- **Episode** = a course: four sessions on days 0, 3, 7 and 10, 40 trials each, on 20 personally
  hard words (unaided P(correct) in [0.05, 0.55] before the course), judged by predicted unaided
  retention of those words on days 17 and 38.
- **Action** (16) = which word × which cue. Word rules: *due* (practised longest ago, maximal
  spacing), *repeat* (the word just practised, masked on a session's first trial), *weakest* (lowest
  observed accuracy), *strongest* (maintenance). Cues: none, first sound, meaning, whole word.
  Spacing, difficulty targeting and scaffolding are all the agent's to discover.
- **Observation** (136 floats, all recordable in a real session): the last 8 trials (cue, rule,
  outcome, response type) and summary counts (session and overall accuracy, share of words
  attempted, recency, share of words below 50% observed accuracy, position in the course). Hidden:
  lesion weights, memory traces, true probabilities. It is a POMDP.
- **Reward.** `privileged` is the change, after each trial, in the mean predicted unaided retention
  of the practised words on the first test day; it telescopes exactly to the episode objective, so
  it is dense without being a different objective. `observable` rewards unaided correct responses
  (+1) and cued ones less (+1 − effort), which a real session could compute. Policies trained on
  either are judged on retention.
- **Patients.** `PopulationSampler` with the fitted prior; patient k of a split is seeded from
  (root seed, split, k), so train, validation and test patients are disjoint and every policy meets
  the same patient with the same random stream.
- **Baselines** (`neurocue/baselines.py`): spaced retrieval without cues, massed practice, the
  clinic's cueing hierarchy on a spaced schedule, errorless (whole word every time), weakest-first
  with the hierarchy, random.

### 4.2 Training

MaskablePPO (Stable-Baselines3 and sb3-contrib; Schulman et al. 2017; Huang and Ontañón 2022;
Raffin et al. 2021), MLP policy, learning rate 3e-4, 160k steps, one seed, minutes on a laptop CPU:

```bash
uv run python -m neurocue.train_ppo --reward privileged      # -> models/engine_ppo_privileged_seed0.zip
uv run python -m neurocue.train_ppo --reward observable      # -> models/engine_ppo_observable_seed0.zip
```

### 4.3 Results on held-out patients

80 held-out test patients, population cohort, every policy treats the same patients
([results/engine_trainer.md](results/engine_trainer.md); 95% percentile bootstrap over patients;
paired differences against the cueing hierarchy). Pre-course retention of the chosen words: 0.391.

| policy | retention day 17 | gain day 17 | gain day 38 | paired gain vs hierarchy, day 17 | cued share | accuracy in practice |
|---|---|---|---|---|---|---|
| **trained policy** (privileged reward) | **0.706** | **+0.315** [+0.298, +0.332] | +0.252 | **+0.022** [+0.020, +0.026] | 0.01 | 0.69 |
| spaced retrieval, no cues | 0.705 | +0.315 [+0.297, +0.331] | +0.252 | +0.022 [+0.020, +0.025] | 0.00 | 0.69 |
| cueing hierarchy, spaced | 0.683 | +0.293 [+0.277, +0.307] | +0.232 | 0 (reference) | 0.26 | 0.73 |
| errorless | 0.646 | +0.256 [+0.242, +0.270] | +0.202 | −0.037 | 1.00 | 0.94 |
| random | 0.637 | +0.247 [+0.235, +0.258] | +0.194 | −0.046 | 0.75 | 0.84 |
| massed, no cues | 0.619 | +0.228 [+0.216, +0.240] | +0.176 | −0.064 | 0.00 | 0.78 |
| weakest-first, hierarchy | 0.609 | +0.218 [+0.203, +0.233] | +0.172 | −0.075 | 0.19 | 0.80 |

Reading it honestly. Nobody told the policy about spacing or retrieval practice; it rediscovered
spaced, uncued retrieval practice, which is the best clinical practice in the literature it was never
shown, and found nothing better. It beats the clinic's cueing hierarchy by a small, tight margin and
cues on 1% of trials. The learning layer was fitted to retrieval-practice data, so spaced beating
massed and retrieval beating errorless study are built in. What is not built in is how a policy
trades a cue-assisted success now against a weaker memory trace later, across words of different
strength, over four sessions. This is the engine's current opinion under its cue-effort
assumptions, not a verdict on cueing.

## 5. Learning from clinician corrections: DITTO with anchors

### 5.1 Method (`agents/ditto.py`)

DITTO (Shaikh et al., ICLR 2025) treats a demonstration as preferred over everything the policy
itself produces in the same situation and optimises that preference with the DPO loss (Rafailov et
al. 2023) against a frozen reference. Here the policy is the trained MaskablePPO actor, a small
network, so no LoRA is needed: each clinician correction is a demonstration, preferred over every
action the policy samples in that state, including samples from earlier rounds; the masked action
log-probabilities are optimised against a frozen copy of the original policy.

Plain DITTO over-generalises with a handful of demonstrations: nothing tells the shared network to
leave other states alone, so a rule such as "after a sound error, give the first sound" became "give
the first sound after every error" (leakage 0.38 → 1.00 on other error states). Our addition is
**anchors**: every agent choice the clinician saw and did not override is an implicit acceptance,
and a KL term keeps the policy close to the reference on those states. Weight and learning rate were
chosen by a small sweep scored only on the override patients' states.

### 5.2 Results

On the first-generation PPO policy, 10 overrides, evaluated on states from disjoint patients
([results/ditto_report.md](results/ditto_report.md)):

| held-out states | n | top action = clinician's cue, before → after: plain DITTO | with anchors | top action unchanged (anchors) |
|---|---|---|---|---|
| after a sound error (rule applies) | 273 | 0.35 → 1.00 | **0.35 → 0.96** | 0.38 |
| after other errors (rule does not apply) | 557 | 0.38 → 1.00 | **0.38 → 0.40** | 0.93 |
| all other states | 2,902 | 0.00 → 0.00 | 0.00 → 0.00 | **0.99** |

A fresh validation repeated the live update five times with 10 overrides each from different
validation patients and evaluated on yet other patients
([results/live_ditto_validation.md](results/live_ditto_validation.md)): the top action switched to
the clinician's cue in 89 to 98% of matching states, and 97 to 100% of other state types kept their
top action. Each update takes about 0.33 s on CPU.

On the NeuroCue policy (section 4), measured in the console: one demonstration teaches nothing
beyond its own state (0% adoption in unseen matching moments); four to six consistent
demonstrations teach the rule (76 to 100% adoption), with leakage to other error states and to
after-correct states reported alongside.

### 5.3 Why it is shown as adoption and containment, not as a score

Because the policy is already optimal under the fitted learning layer, a correction cannot raise
predicted retention on this engine. So the console judges a correction on two things only: adoption
where the clinician meant it, and containment everywhere else. That is customisability with full
transparency. The moment the engine's cue assumptions are replaced by cued-therapy retention data,
the same loop prices the corrections.

## 6. The console and the demo

The console comes in two forms that share one page (`neurocue/demo/static/index.html`):

- **Hosted, precomputed.** Six patients spread over severity and lesion type, each with an
  avatar: Marisol (23, AQ 62, balanced), Harold (74, AQ 34, phonological), Aisha (45, AQ 86,
  semantic), Kenji (67, AQ 48, semantic), Ruth (79, AQ 72, phonological) and Darnell (38, AQ 28,
  balanced). For each,
  `scripts/build_demo_patients.py` calls the console server's own endpoint functions and writes to
  `neurocue/demo/static/patients/`:
  - 100 naming responses in order;
  - the true lesion weights;
  - both four-session courses;
  - a DITTO result for every option the teach form offers (situation × word rule × cue, each
    taught from the original policy with six demonstrations).

  The diagnosis posterior is cheap, so the page computes it itself from the fitter's grid table
  (`table.json`). It matches `server.py` exactly. The page then needs no server compute at all.
  Rebuilding takes about 70 seconds:

  ```bash
  uv run python scripts/build_demo_patients.py
  ```

- **Local, live.** `neurocue/demo/server.py` is a FastAPI service over the engine. Its API
  (below) samples any patient (AQ, lesion type, seed) and runs DITTO live. `./scripts/run_demo.sh`
  starts it and serves the same page.

Every number is computed by the code the validation ran:

- the patient is `patientmodel.Patient` on the Dell model with the full word realizer;
- the diagnosis is the fitter's grid posterior;
- the retention curves are the fitted learning layer inside the therapy environment;
- the override is `agents/ditto.py` on the trained policy.

Each avatar is one parametric SVG face, varied by skin, hair, eyes, glasses and beard. Its mouth
follows the phonemes the patient produced, and browser speech synthesis voices the utterances.

The five steps on the page:

1. **Who are you listening to.** Pick one of the six patients.
2. **The patient names pictures.** Name one at a time with sound on, or many at once. The avatar
   mouths the phonemes the patient actually produced, nonwords and false starts included.
3. **Diagnose.** The posterior heat map over (s, p) comes from the answers alone. Reveal the true
   lesion and it sits inside the bright region. Switch from Harold (phonological) to Aisha
   (semantic) and the errors and the posterior move.
4. **Two therapists.** Two four-session courses run on identical copies of the patient: the
   clinic's cueing hierarchy against the trained policy. Both are judged on what the patient still
   names a month later. A strip shows which word and which cue each trial used.
5. **Where the physician comes in.** Play one correction, then teach a rule (for example: after a
   nonword, retry the word with its first sound). Two chips report adoption where the rule was
   meant and containment elsewhere.

API of the console:

| method | path | purpose |
|---|---|---|
| POST | `/api/patient` | sample a patient (AQ, lesion type, seed) → id and picture list |
| POST | `/api/patient/{id}/name` | name one picture: category, word, phonemes, utterance frame, duration |
| POST | `/api/patient/{id}/name_batch` | name n pictures at once |
| GET | `/api/patient/{id}/truth` | the true (s, p) |
| GET | `/api/image/{item_id}` | the IPNP drawing |
| POST | `/api/patient/{id}/course` | run the four-session course under two policies on identical copies |
| POST | `/api/patient/{id}/override` | teach a rule: DITTO with anchors on the policy, with adoption and leakage counts |

`neurocue/demo/record_demo.py` drives the page with Playwright and burns captions into the video in
`results/demo/`.

## 7. Deployment

The hosted demo runs on [Modal](https://modal.com) (serverless containers, free monthly credits, no
card). Because the hosted console is precomputed (section 6), `deploy/modal_app.py` is a slim
Python image with FastAPI serving `neurocue/demo/static/`. It has no torch and no engine, a quarter
CPU and 256 MB, and a cold start of a few seconds. Publishing or updating it:

```bash
uv tool install modal
```

```bash
modal setup
```

```bash
modal deploy deploy/modal_app.py
```

A deploy takes seconds. Since the site is static files, any static host (GitHub Pages, Netlify)
would also serve `neurocue/demo/static/` as it is. To redeploy on every push to `main`, add the
repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (from
https://modal.com/settings/tokens) on GitHub. The workflow in `.github/workflows/deploy-modal.yml`
then runs the same command.

The live server (any patient, live DITTO) also ships as a container. `Dockerfile` serves it on port 7860:

```bash
docker build -t neurocue-demo . && docker run --rm -p 7860:7860 neurocue-demo
```

`scripts/deploy_hf.sh` and `.github/workflows/deploy-hf-space.yml` publish that container to a
Hugging Face Space, which since late 2026 needs a PRO subscription for Docker Spaces.

## 8. Reproducing every result

All random sources take a seed; all tunable values live in `configs/*.yaml` and are validated at
load time (`core/config.py`); assumptions are marked `ASSUMPTION` in the configs. Each report in
`results/` names the command that generated it in its first line.

Data (the public sources; PSST needs the TalkBank archives, see [data/README.md](data/README.md)):

```bash
uv run python -m scripts.fetch_data
```

Labels and coder (section 3.2):

```bash
uv run python -m scripts.code_psst_responses        # machine codes + adjudication sheet -> data/derived/
uv run python -m scripts.merge_adjudication         # fold the clinician's sheets back in
uv run python -m scripts.learn_coder                # random forest, leave-one-picture-out -> results/psst_coder_learned.md
```

Engine fits and validation (sections 3.3 to 3.7):

```bash
uv run python -m scripts.fit_dell_sp                # refit Dell's 14 patients -> side-by-side s, p
uv run python -m scripts.fit_population_prior       # lesion prior on the training participants -> results/population_prior_fit.md
uv run python -m scripts.validate_engine            # held-out severity and error-mix check -> results/engine_validation.md
uv run python -m scripts.mine_utterance_frames      # carrier phrases and fillers by severity -> data/calibration/
uv run python -m patientmodel.learning_fit          # ACT-R layer on Middleton 2019 -> results/learning_fit.md
uv run python -m scripts.calibrate_cues             # cue activation levels on Nardo 2017 -> results/cue_calibration.md
```

Environment, trainer and the clinician loop (sections 4 and 5):

```bash
uv run python -m neurocue.evaluate --n 100                                        # baselines -> results/engine_trainer_baselines.md
uv run python -m neurocue.train_ppo --reward privileged                           # -> models/engine_ppo_privileged_seed0.zip
uv run python -m neurocue.evaluate --ppo models/engine_ppo_privileged_seed0.zip   # -> results/engine_trainer.md
uv run python -m train.ditto_ppo                                                  # DITTO with anchors on the first-generation policy -> results/ditto_report.md
uv run python -m scripts.validate_live_ditto                                      # five fresh override sets -> results/live_ditto_validation.md
```

Demo video:

```bash
uv run playwright install chromium
uv run python -m neurocue.demo.record_demo          # with the console running on :8765; writes results/demo/neurocue_demo.mp4
```

## 9. Limits, stated

- **Scope.** Single-word naming only. No sentences, discourse, comprehension, motor speech or
  conversation. Omissions and circumlocutions: the SP model's lexical-threshold variant is wired
  (`omission_threshold`) but not calibrated, and descriptions are not generated.
- **Error split.** The engine over-produces formal and unrelated errors and under-produces semantic
  ones because of the SP model's six-word neighbourhood (section 3.4). The nonword share and the
  severity curve are right; the word-error split is wrong in a known way.
- **Severity curve.** 74 participants with 15 noun items each; the heterogeneity term is an estimate
  after removing binomial noise; BNT items are harder than the PNT's. Item effects and the nonword
  edit rate are assumptions until item-level, category-coded patient data exist. The 39
  compound-name items (the hardest) are skipped by the realizer.
- **Learning layer.** Fitted to four participants with anomic aphasia (AQ 81 to 90) on personally
  unnameable proper nouns; transfer to common nouns and to severe patients is an assumption.
- **Cues.** Effort weights are assumptions; the final-phoneme cue is unreachable; the semantic-cue
  mechanism contradicts Python et al. 2021 and is flagged as suspect. No cued-therapy retention data
  has been compared yet.
- **Trainer.** One PPO seed. The advantage over the cueing hierarchy is real but small (+0.022). A
  correction cannot beat the already optimal policy on this engine, which is why DITTO is shown as
  adoption and containment, not as a retention gain.
- **Labels.** The 328 clinician labels come from one adjudicator.
- **Data not obtained.** MAPPD (item-level PNT responses for 240+ patients) is offline; the engine
  samples from Dell's published fits instead ([data/README.md](data/README.md)).

## 10. Earlier generations kept in the repository

NeuroCue grew out of two earlier builds in this repository. Their code, tests and results are kept
as they were, because the NeuroCue environment reuses their schema, masks, service patterns and the
DITTO-with-anchors update, and because their results are cited above. Do not mix their numbers with
the NeuroCue course results: they are different patients and different environments.

### 10.1 First generation: RL therapist on a logistic patient simulator (`sim/`, `envs/`, `train/`, `eval/`, `service/`)

**Problem.** One episode is one 40-trial session; the action is (difficulty tier 1..3) × (cue:
none, first sound, first letter, sentence frame, semantic) = 15 discrete actions; clinical
constraints are action masks, not penalties (tier moves at most one step per trial; cues only from
the clinician-approved set in [configs/clinical.yaml](configs/clinical.yaml); cue only after an
error). The target was 80 to 85% correct (Wilson et al. 2019's rule, theoretical, not shown for
aphasia) and the objective "better later naming without hints", measured by review probes (every
5th trial re-tests a practised word with no cue).

**Simulated patients** (`sim/`, [configs/sim.yaml](configs/sim.yaml)):
`p(correct) = sigmoid(ability + item_learning[i] − difficulty[i] + cue_gain[cue] − fatigue)` on the
log-odds scale. Item difficulty from frequency and length (Nickels and Howard 1995); cues with
per-patient phonemic and semantic responsiveness (Meteyard and Bose 2018); error types after a
failure proportional to phonological impairment, semantic impairment and omission bias (simplified
from Dell et al. 1997; Schwartz et al. 2006); fatigue with recovery; item-specific learning peaking
at a configurable success rate with 5% spill-over to general ability (Nickels 2002; the peak is an
assumption); log-normal response time; dropout after failure streaks. Training patients are drawn
from broad distributions (domain randomisation); three narrow cohorts (mild anomic, moderate
Broca's, severe) are for evaluation only; train and test populations are disjoint and every policy
faces the same patients with the same outcome stream. The item bank
([data/items_placeholder.csv](data/items_placeholder.csv)) is a placeholder of 60 nouns.

**Agents** (`agents/`), all behind `act(state, mask) → Action`, `update(state, action, outcome)`:
fixed difficulty; 2-up/1-down staircase (Levitt 1971); Beta-Bernoulli Thompson bandit; a
contextual Bayesian logistic Thompson bandit with 17 features, closed-form Laplace/extended-Kalman
posterior updates with random-walk process noise, and an empirical-Bayes prior fitted on 300
simulated training patients (`train/fit_bandit_prior.py`); a shared selection rule
(`agents/selection.py`: among actions within 0.06 of the 0.85 target, the hardest tier, then the
weakest cue); and MaskablePPO.

**MaskablePPO** (`envs/`, `train/train_ppo.py`). Observation from the State JSON only (the last 8
trials as one-hot tier, cue and error type plus response time, and the current tier, rolling
accuracy, fail streak and session progress), so the trained policy runs unchanged behind the
service. Three rewards ([configs/reward.yaml](configs/reward.yaml)): the **proxy** reward uses only
observable terms (+1 per trial with rolling accuracy in band; −1 per trial while the failure streak
is 3 or more; −0.1 for a probably wasted heavy cue; a terminal +2 per tier climbed, paid only if the
session spent at least half its trials in band, which removes a tier-oscillation exploit in the
original scaffold); **proxy + probe** adds +1 per correct review probe; **privileged** (reference
only) uses the true control error |p − 0.85|, the dense increase in true retrieval over the item
bank and the probe lift. 3 seeds × 1.5M steps per reward, about 3 minutes each on CPU; all variants
converge within about 100k steps ([results/learning_curves.png](results/learning_curves.png)).

**Results** (`uv run python -m eval.compare --config configs/eval.yaml`; 200 held-out patients per
cohort, 40 trials, dropout on, PPO averaged over its 3 seeds, 95% bootstrap intervals, paired
differences in [results/paired_differences.csv](results/paired_differences.csv); full tables in
[results/comparison.md](results/comparison.md)). The oracle reads the hidden state and picks the
allowed action whose true p is nearest 0.85; it is a privileged heuristic, not an optimum.

| cohort | policy | in band | \|p − 0.85\| | learning gain ×100 | probe recall | probe lift ×100 | quit |
|---|---|---|---|---|---|---|---|
| mild anomic | fixed (tier 2) | 0.46 | **0.066** | **3.53** | 0.92 | **2.84** | 0.01 |
| mild anomic | staircase | 0.81 | 0.137 | 3.06 | 0.86 | 2.29 | 0.01 |
| mild anomic | contextual bandit | 0.48 | 0.088 | 3.00 | 0.93 | 1.61 | 0.01 |
| mild anomic | PPO, proxy reward | **0.86** | 0.129 | 2.61 | 0.90 | 1.71 | **0.00** |
| mild anomic | PPO, simulation objective | 0.40 | 0.087 | 2.77 | 0.94 | 1.74 | 0.00 |
| mild anomic | oracle (privileged) | 0.47 | 0.068 | 3.53 | 0.92 | 2.84 | 0.01 |
| moderate Broca's | fixed (tier 2) | 0.52 | 0.110 | 4.56 | 0.75 | **4.40** | 0.02 |
| moderate Broca's | staircase | 0.59 | 0.155 | 3.99 | 0.77 | 3.02 | 0.04 |
| moderate Broca's | contextual bandit | 0.62 | 0.074 | 4.68 | 0.83 | 3.14 | 0.01 |
| moderate Broca's | PPO, proxy reward | **0.83** | 0.112 | 4.20 | 0.81 | 2.95 | **0.00** |
| moderate Broca's | PPO, simulation objective | 0.62 | **0.065** | **4.77** | 0.84 | 3.43 | 0.00 |
| moderate Broca's | oracle (privileged) | 0.59 | 0.048 | 5.07 | 0.87 | 3.33 | 0.00 |
| severe | fixed (tier 2) | 0.02 | 0.516 | 0.11 | 0.26 | 0.22 | 0.77 |
| severe | staircase | 0.11 | 0.342 | 1.27 | 0.48 | 1.49 | 0.29 |
| severe | contextual bandit | 0.24 | 0.284 | 1.71 | 0.49 | 1.78 | 0.17 |
| severe | PPO, proxy reward | **0.33** | **0.240** | **2.08** | 0.55 | **2.26** | **0.10** |
| severe | PPO, simulation objective | 0.29 | 0.251 | 2.04 | 0.53 | 2.20 | 0.12 |
| severe | oracle (privileged) | 0.37 | 0.214 | 2.27 | 0.58 | 2.51 | 0.07 |

![Policy comparison](results/comparison.png)

What it showed: where difficulty matters most (severe patients) every adaptive policy beat fixed
difficulty by a wide, tight margin (paired learning gain +1.9 to +2.0 points for PPO, abandoned
sessions from 77% to 10 to 17%; PPO reached about 90% of the oracle's learning gain there). PPO
trained on the simulation objective was the most consistent adaptive policy. Proxy-trained PPO held
the band best but partly gamed it (true control error 1.5 to 1.7 times the simulation-objective
policy's), so time in band is never reported alone. On mild patients fixed tier 2 equalled the
oracle because tier 2 without a cue already sat at the simulator's learning peak: it is the
*matching* that matters (fixed tier 1 gets 1.72 and fixed tier 3 gets 2.36 on mild patients; on
moderate patients fixed tier 3 abandons 82% of sessions), and picking the right fixed tier per
patient requires exactly the knowledge an adaptive policy acquires. Observed probe recall mostly
measured how easy the practised words were. Severe patients stayed hard for every policy, partly
because the "cue only after an error" rule forbids repeating their best action after a success.

**Ablations** (`eval/ablations.py`, [results/ablations.md](results/ablations.md),
[results/ablation_learning_peak.png](results/ablation_learning_peak.png)). The 85% target is
load-bearing: with the learning peak moved to 0.5, every policy aiming at 85% learns little on mild
patients (oracle 0.23) while the staircase gets 1.22; a learning-only PPO retrained at peak 0.5
recovers most of the loss (2.23 mild, 3.00 moderate, best of all policies there), which is the
argument for learning the target from the true objective rather than fixing it at 85%. Dropout
matters only for severe patients. The four observable reward variants are indistinguishable; only
the simulation objective changes behaviour.

**Live behaviour.** `scripts/judge_demo.py` plays three scripted healthy "judges" through the HTTP
API. The proxy-reward PPO (`models/ppo_seed0.zip`) was served live because it is the only agent
whose behaviour shows the intended adaptation to a healthy user (climb to tier 3 on success, back
off with a cue after errors, climb again); the simulation-objective PPO stays on tier 1 for a
healthy user and the contextual bandit wanders. A trade-off, stated plainly: chosen for legible
live behaviour, not because it is the best policy on the simulated objective.

**Agent service and interface** (`service/`, [core/schema.py](core/schema.py)). A FastAPI service
with one online-learning agent per `session_id`, validating every request against the pydantic
schema (HTTP 422 on violation). `State` carries the last 10 trials (tier, cue, outcome, error type
in {correct, sound, meaning, no_response}, response time, probe flag), the current tier, trial count
and session minutes; `Action` is `{tier, cue, rationale}`; `Outcome` is
`{correct, error_type, response_time}`. Endpoints: `GET /health`; `POST /next_trial` (applies the
shared item-selection and review-probe rule from `sim/items.py`: a probe every 5th trial on an
eligible practised item, otherwise an unseen item in the returned tier); `POST /act`;
`POST /update` (idempotent by `attempt_id`); `POST /override` (stored as JSON lines in
`runs/overrides.jsonl`); `POST /learn_from_overrides` (one DITTO-with-anchors round on this
session's overrides, well under a second on CPU, swapped in for this session only; returns, per
overridden state type, the probability of the clinician's cue before and after and the share of
other states whose top action is unchanged). Service sessions derive their seed from the config seed
and the session id, so replaying the same requests reproduces the same actions
(`tests/test_service.py::test_full_session_replays_identically`). Interactive docs at `/docs`.

```bash
uv run uvicorn service.app:app --host 0.0.0.0 --port 8000
uv run python -m scripts.service_smoke --url http://localhost:8000     # 40-trial session over HTTP
uv run python -m scripts.judge_demo --url http://localhost:8000 --name live
uv run python -m train.fit_bandit_prior
uv run python -m train.train_ppo --reward full          # 3 seeds -> models/ppo_seed*.zip
uv run python -m train.train_ppo --reward full_probe
uv run python -m train.train_ppo --reward privileged
uv run python -m eval.plots
uv run python -m eval.compare --config configs/eval.yaml
uv run python -m eval.ablations --config configs/ablations.yaml
uv run python -m scripts.sanity_table --dropout on --out results/sanity_dropout_on.md
```

Known limitations of this generation: with an 8-trial window rolling accuracy takes values k/8, so
even a perfect controller at p = 0.85 is in band only about 62% of the time, which is why true
control error is reported; the simulator is parametric and its magnitudes are assumptions; one
action per trial applies the cue to the *next* item, whereas real cueing happens within an item.
Explored but out of scope: a language-model therapist trained with GRPO on Tinker
(`agents/llm_prompt.py`, `train/grpo_*.py`, `uv sync --extra grpo`; never run against the API), and
a prompted frontier-model baseline (`configs/eval_llm.yaml`, tested with a mocked API).

### 10.2 Second generation: CueAgain, a practice app with a language-model decision policy (`frontend/`, `training/`, `evaluation/`, `speech/`)

**What it is.** Picture-naming practice with speech input. The patient sees a picture and names it
aloud; a pretrained speech recogniser proposes a transcript; a helper confirms or corrects the
result; a decision model chooses what happens next: retry the same picture with a predefined hint,
or move on to a new picture at a chosen difficulty. A clinician view captures preferred decisions.
The proof-of-concept question was whether DITTO can make a small language model's exercise-selection
decisions follow a demonstrator's preferences in situations it has not seen. The demonstrations are
team-authored mock preferences written from a rule
([data/demonstrations/README.md](data/demonstrations/README.md)), not clinician-validated treatment
knowledge.

**Components.** Patient screen (large picture, one main action, user-initiated microphone, explicit
recording state, hint, next picture, pause). Speech: local `faster-whisper` `base.en` on CPU, no API
key; WebM/Opus and MP4 browser recordings decoded with PyAV; silent, undecodable, timed-out or
failed audio returns an explicit status and is never scored. Helper confirmation: editable
transcript, suggested match only, correct / sound / meaning / no response / incorrect of unclear
type / not scored. Attempt state keyed by `session_id`, `exercise_id`, `presentation_id`,
`attempt_id`, idempotent (repeat = same result; different content = 409; stale picture = 409).
Picture bank: 24 items ([data/items.json](data/items.json)) with predefined hints and Twemoji images
(CC BY 4.0, [data/ATTRIBUTION.md](data/ATTRIBUTION.md); MultiPic could not be downloaded
non-interactively). Decision models: the Beta bandit, unadapted Qwen3-1.7B, few-shot prompted
Qwen3-1.7B, an SFT-only adapter and the DITTO adapter, all validated against the valid options, with
invalid, unparseable or timed-out output falling back to the bandit and saying so. Clinician view
with the latest decision, its source and fallback reason, alternatives in plain words, "Use my
choice" and "Save as a good example"; a Models & results view to activate a version and read
training runs. Hard constraints (at most 3 attempts per picture, tier moves of at most one step,
approved hints only; [configs/practice.yaml](configs/practice.yaml)) are enforced on every model's
output at the service boundary.

**How the decision model is trained** (`training/ditto.py`, after `SALT-NLP/demonstrated-feedback`).
All code paths render contexts and parse actions through `core/practice.py`, so training, evaluation
and serving see identical text; the model answers with one option such as
`{"op":"retry","hint":"first_sound"}` or `{"op":"next","tier":2}`; Qwen3 thinking is disabled.
(1) SFT: LoRA adapter (r 16, q/k/v/o) on the demonstrations' answer tokens with an early stop on the
loss. (2) DITTO: a copy of the SFT adapter is trained with the fixed SFT adapter as the DPO
reference. (3) Each of 3 rounds: sample 6 answers per training context at temperature 1; build
expert (demonstration ≻ this round's sample), replay (demonstration ≻ earlier rounds' samples) and
intermodel (later round ≻ earlier round) comparisons; draw 70 / 20 / 10% of n_prompts × 3 pairs; 15
DPO steps (β 0.1, lr 5e-5). Differences from the reference implementation, each recorded in the run
manifest: pairs drawn once per round rather than per batch; pairs whose two sides are the same
action, or whose rejected side is an action the demonstrator also accepts, are discarded rather than
trained on; the DPO loss is implemented directly with reference log-probabilities computed once per
round, so TRL is not a dependency; learning rate and β are larger than the reference's because the
answers are about 10 tokens and each round has 15 steps, chosen before evaluation.

**Result** (run `demoA-qwen3-1.7b-r1`, 20 training demonstrations, settings fixed before
evaluation; [results/ditto_llm_test.md](results/ditto_llm_test.md),
[results/ditto_llm_dev.md](results/ditto_llm_dev.md)). "Agrees" = the final action is one the
demonstrator accepts.

| model | test (10 situations) | dev (5) | valid output (test) | median latency |
|---|---|---|---|---|
| Qwen3-1.7B, task instruction only | 2/10 | 1/5 | 10/10 | 0.34 s |
| same + 20 demonstrations in the prompt | 5/10 | 2/5 | 10/10 | 2.7 s |
| SFT adapter only | 3/10 | 1/5 | 10/10 | 0.38 s |
| DITTO after round 1 / round 2 | 3/10 / 4/10 | 1/5 / 0/5 | 6/10 / 10/10 | 0.36 s |
| **DITTO, final (round 3)** | **6/10** | **1/5** | 10/10 | 0.38 s |
| bandit baseline | 4/10 | 1/5 | n/a | <1 ms |

The DITTO adapter agreed with the demonstrator more often than the unadapted model, the SFT-only
adapter and the bandit, and slightly more than few-shot prompting at a seventh of its latency, with
successive rounds moving 3 → 4 → 6. It is not robust: on the 5 development situations every model
scores 0 to 2, and with 10 and 5 situations one or two decisions change the ranking. The adapter
learned "move on" better than "retry with the matched hint" (it missed every retry-with-hint
situation), and a hand-coded rule would score 10/10; the question was whether a language model picks
the preference up from 20 demonstrations. Compute: the whole pipeline ran on an Apple M4 Pro GPU via
MPS (training 323 s, 4.1 GB peak, inference about 0.4 s per decision); no cloud was used.

```bash
uv sync --extra speech                                          # local speech recognition
uv sync --extra llm --extra speech                              # + language-model decisions and training
uv run --extra llm --extra speech uvicorn service.app:app --port 8000     # open http://localhost:8000/app/
uv run --extra llm python -m training.smoke --model Qwen/Qwen3-0.6B
uv run python -m training.build_demonstrations
uv run --extra llm python -m training.ditto --run-id demoA-qwen3-1.7b-r1
uv run --extra llm python -m evaluation.compare --run demoA-qwen3-1.7b-r1 --split test --rounds --markdown results/ditto_llm_test.md
```

`uv sync` without the extras removes them, so keep passing the extras you use. The default decision
model for new sessions is `default_model` in `configs/practice.yaml`; the base model and pinned
revision are `llm.base_model` / `llm.revision` there. Runs, adapters, evaluations, sessions and
captured examples live in `runs/` (gitignored); the adapter of the run above is 25 MB.

Practice API (additive to the agent service): `GET /items`; `POST /transcribe`;
`GET /speech/status`; `POST /practice/sessions`, `GET /practice/sessions[/{id}]`;
`POST /practice/sessions/{id}/attempts`, `/manual`, `/override`, `/model`, `/pause`;
`POST/GET /demonstrations`; `GET/POST /models`, `/models/active`; `GET /runs[/{run_id}]`. Not done:
training from the browser (it is a CLI; saved clinician examples can be added with
`--extra-demos runs/demonstrations/captured.jsonl`); the microphone path was exercised with
synthesized speech and typed answers only; Whisper `base.en` is not evaluated on aphasic speech;
sessions are in memory with JSON snapshots in `runs/practice/`.

## 11. Data and licences

Raw third-party data lives under `data/raw/` and is not committed; `uv run python -m scripts.fetch_data`
re-downloads what is publicly reachable. Details and status per source in
[data/README.md](data/README.md).

| source | used for | terms |
|---|---|---|
| PSST (Gale et al. 2022; AphasiaBank/TalkBank) | severity curve, error-type prior, utterance frames, coder labels | AphasiaBank / PSST terms; cite Gale et al. 2022; transcripts and audio never committed |
| Middleton, Rawson and Verkuilen 2019 (OSF `cwq75`) | learning-layer fit | no licence stated; cite the paper; not redistributed |
| International Picture Naming Project (Bates et al. 2003; Szekely et al. 2004) | 520 pictures, norms, difficulty index | pictures described as freeware; cite the papers |
| Brysbaert, Warriner and Kuperman 2014 | concreteness for the lexicon | publisher supplement |
| Dell et al. 2004, Table 2; Dell et al. 2013, Table 1 | refit targets, Philadelphia prior | published tables, transcribed to `data/calibration/` |
| Nardo et al. 2017; Pease and Goodglass 1978 | cue calibration and checks | published figures |
| CMUdict, WordNet (via NLTK), wordfreq (Speer 2022), all-MiniLM-L6-v2 | pronunciations, semantics, frequency, embeddings | their own licences |
| Twemoji 15.1.0 | the 24 CueAgain pictures | CC BY 4.0 ([data/ATTRIBUTION.md](data/ATTRIBUTION.md)) |

The clinician's adjudication sheets are summarised in `data/derived/`; the PSST transcripts they
quote are not committed.

## 12. Branches and team workflow

- `main` (default): NeuroCue, everything needed to run the demo and reproduce the results.
- `legacy-main`: the original CueAgain scaffold that `main` grew from, kept for reference.
- `dev`: integration branch of the CueAgain app; `feature/rl-agent`: the first-generation RL work;
  `feature/cueagain-ditto`, `feature/ditto-training`, `kuba/clinician-real-data` and personal
  branches: teammates' work during the hackathon.

Merges into `dev` went through pull requests on GitHub once the tests passed. On `main`, pushes
trigger the Modal redeploy when the secrets are set (section 7).

## 13. References

Papers are cited for the mechanism or the direction of an effect; where a numeric value is fitted the
results file is named, and where it is assumed the config marks it `ASSUMPTION`.

**Lexical access and aphasic naming**

- Dell, G. S., Schwartz, M. F., Martin, N., Saffran, E. M. and Gagnon, D. A. (1997). Lexical access
  in aphasic and nonaphasic speakers. *Psychological Review* 104(4), 801–838.
- Foygel, D. and Dell, G. S. (2000). Models of impaired lexical access in speech production.
  *Journal of Memory and Language* 43, 182–216.
- Dell, G. S., Lawler, E. N., Harris, H. D. and Gordon, J. K. (2004). Models of errors of omission
  in aphasic naming. *Cognitive Neuropsychology* 21(2–4), 125–145 (Table 2 SP-model fits, the refit
  targets in `data/calibration/`).
- Dell, G. S., Schwartz, M. F., Nozari, N., Faseyitan, O. and Coslett, H. B. (2013). Voxel-based
  lesion-parameter mapping. *Cognition* 128(3), 380–396 (dual-route repetition, Table 1 example).
- Schwartz, M. F., Dell, G. S., Martin, N., Gahl, S. and Sobel, P. (2006). A case-series test of the
  interactive two-step model of lexical access. *Journal of Memory and Language* 54(2), 228–264.
- Walker, G. M. (2016). WebFit: Bayesian fitting of the SP model.
- Mirman, D. et al. (2010). The Moss Aphasia Psycholinguistics Project Database (MAPPD).
- Nickels, L. and Howard, D. (1995). Aphasic naming: What matters? *Neuropsychologia* 33(10),
  1281–1303.
- Nickels, L. (2002). Therapy for naming disorders: Revisiting, revising, and reviewing.
  *Aphasiology* 16(10–11), 935–979.

**Data**

- Gale, R. et al. (2022). Post-Stroke Speech Transcription (PSST) challenge corpus; AphasiaBank,
  TalkBank.
- Bates, E. et al. (2003). Timed picture naming in seven languages. *Psychonomic Bulletin & Review*
  10(2), 344–380. Szekely, A. et al. (2004). A new on-line resource for psycholinguistic studies.
  *Journal of Memory and Language* 51(2), 247–250.
- Brysbaert, M., Warriner, A. B. and Kuperman, V. (2014). Concreteness ratings for 40 thousand
  generally known English word lemmas. *Behavior Research Methods* 46, 904–911.
- Speer, R. (2022). wordfreq, v3. https://github.com/rspeer/wordfreq

**Learning, spacing and cues**

- Middleton, E. L., Rawson, K. A. and Verkuilen, J. (2019). Retrieval practice and spacing effects
  in multi-session treatment of naming impairment in aphasia. *Cortex* 119, 386–400. Middleton,
  E. L. et al. (2016). Towards a theory of learning for naming rehabilitation. *JSLHR* 59.
- Pavlik, P. I. and Anderson, J. R. (2005). Practice and forgetting effects on vocabulary memory:
  An activation-based model of the spacing effect. *Cognitive Science* 29(4), 559–586.
- Nardo, D. et al. (2017). Less is more: neural mechanisms underlying anomia treatment in chronic
  aphasic patients. *Brain* 140(11), 3039–3054.
- Pease, D. M. and Goodglass, H. (1978). The effects of cuing on picture naming in aphasia.
  *Cortex* 14(2), 178–189.
- Howard, D., Patterson, K., Franklin, S., Orchard-Lisle, V. and Morton, J. (1985). The facilitation
  of picture naming in aphasia. *Cognitive Neuropsychology* 2(1), 49–80.
- Python, G. et al. (2021). Phonological and semantic cue effects by type of lexical-semantic
  impairment (cited for the direction the engine's semantic cue contradicts).
- Meteyard, L. and Bose, A. (2018). What does a cue do? Comparing phonological and semantic cues for
  picture naming in aphasia. *JSLHR* 61(3), 658–674.
- Wilson, R. C., Shenhav, A., Straccia, M. and Cohen, J. D. (2019). The Eighty Five Percent Rule for
  optimal learning. *Nature Communications* 10, 4646 (first generation's target; theoretical, not
  shown for aphasia).
- Levitt, H. (1971). Transformed up-down methods in psychoacoustics. *JASA* 49(2B), 467–477.

**Learning algorithms**

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal Policy
  Optimization Algorithms. arXiv:1707.06347.
- Huang, S. and Ontañón, S. (2022). A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms. *FLAIRS-35*.
- Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations.
  *JMLR* 22(268). Gymnasium; sb3-contrib (MaskablePPO).
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D. and Finn, C. (2023). Direct
  Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS* 36.
- Shaikh, O. et al. (2025). Show, Don't Tell: Aligning Language Models with Demonstrated Feedback.
  *ICLR 2025*.
- Chapelle, O. and Li, L. (2011). An empirical evaluation of Thompson sampling. *NeurIPS* 24.
  Raj, V. and Kalyani, S. (2017). Taming non-stationary bandits: A Bayesian approach.
  arXiv:1707.09727.
- Shao, Z. et al. (2024). DeepSeekMath (GRPO). arXiv:2402.03300 (explored only).

**Tools**: scikit-learn, NLTK, sentence-transformers, FastAPI, uv, Playwright, Modal, faster-whisper,
transformers, peft.
