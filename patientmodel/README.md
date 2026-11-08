# patientmodel: the synthetic aphasia patient engine

A mechanistic simulator of single-word retrieval in aphasia. Two cognitive parameters and a
severity level produce a patient who names pictures, makes the error types real patients make in
the proportions real patients make them, says actual words or nonwords, is harder to understand on
harder pictures, and, with the learning layer, gets better with practice the way Middleton's
patients did: more from effortful retrieval than from repetition, more from spaced than massed
practice, and forgetting over the following weeks. Everything is seeded and reproducible; a 4,000-trial
patient simulates in about 80 ms.

An independent code review (2026-11-07) of the first version found real-word "nonwords", a latency
model built on clip length, a circular severity validation, over-loose semantic and mixed criteria
and non-noun error vocabulary; all are fixed in this version and guarded by tests.

The trainer (`agents/`, `envs/`, `service/`) is the engine's first application. The engine's own
deliverables are synthetic patients, cohorts and datasets, validated against real data.

## Modules

| module | what it is | grounded on |
|---|---|---|
| `dell_sp.py` | The semantic-phonological interactive two-step model of lexical access (Foygel & Dell 2000). Parameters s (semantic-lexical weight) and p (lexical-phonological weight); six-word neighbourhood, 57 semantic units, 10 phonemes, decay .6, two noise terms, 8 steps per stage. Returns the six PNT response categories. | Dell et al. 2004 (architecture, constants); Walker 2016 (lexicon). Reproduces the SP-model proportions Dell published for 7 fitted patients to rmsd .002-.011 and names 97% correctly at normal weights (`tests/test_dell_sp.py`, tolerance .015). |
| `fit.py` | Fits (s, p) to a patient's naming counts: cached simulated probability grid keyed on the full model config, Laplace-smoothed multinomial likelihood, Dell's rmsd, grid posterior with 95% intervals, a boundary flag. Counts must be naming responses only. | Refitting Dell's 2004/2013 patients from their published proportions with the correct effective N recovers his s and p to .002 on the 14 interior fits; the one near-normal patient sits at the grid edge and is flagged (`scripts/fit_dell_sp.py`). |
| `population.py` | Samples patients: AQ severity, a target accuracy from a logistic severity curve with between-patient heterogeneity, then an (s, p) cell consistent with it under a lognormal prior on p/s. Profiles mild / moderate / severe / very severe. | Severity curve fitted on the PSST noun items of the 74 training-split participants; heterogeneity after removing binomial noise; p/s prior fitted to the training participants' error mix and checked held out (Dell's prior kept as a named profile). |
| `utterance.py` | How a response is spoken: samples the carrier phrases, fillers, false starts and trailing words real patients put around the response word ("uh … house", "okay that's a … home", "k- … crab"), by severity band, from counts mined off PSST transcripts (`scripts/mine_utterance_frames.py` → `data/calibration/psst_utterance_frames.json`). Drives the demo avatar's speech. The response word itself is untouched. | Frame rates by band measured on PSST BNT (1% to 12% of responses; a lower bound, since clips are trimmed). |
| `cues.py` | Cues as external activation in the SP model: a first sound (target's onset unit), a last sound (coda unit, shared with the formal neighbours), a meaning cue (target's semantic features), or the whole word (all three phoneme units plus the word node, Dell 2013's dual-route repetition). `Cue.effort` is the share of the retrieval the cue does, which the learning layer uses to weight a cued success between an unaided success and errorless study. `TreatablePatient.attempt(item, t, cue)` and `cue_hierarchy()`. | Activation levels calibrated to cued-naming accuracy in Nardo et al. 2017 (initial phoneme and whole word fitted exactly; final phoneme unreachable, see below); semantic level and effort shares are ASSUMPTIONS stated in `configs/cues.yaml`. |
| `items.py` | 520 IPNP object pictures with norms and a difficulty index (healthy RT, frequency, length; name agreement deliberately excluded); 244 drawings; 39 compound-name items flagged `pronounceable=False`. | Bates et al. 2003; Szekely et al. 2004. |
| `lexicon.py` | About 4,800 picturable nouns: concreteness >= 4, WordNet noun, CMUdict pronunciation, not on the profanity or sensitive-word lists; Zipf frequency; MiniLM embeddings; the full CMUdict pronunciation set; the PNT formal-relatedness criterion; English phonotactic legality. | CMUdict; Brysbaert et al. 2014; WordNet; wordfreq; all-MiniLM-L6-v2. |
| `realize.py` | Turns a response category into an actual response for a target: semantic neighbours need embedding similarity plus a WordNet taxonomic relation (or very high similarity); formal neighbours meet the PNT criterion and are not semantic; mixed need both and shared onset or <= 2 edits; unrelated are frequent nouns sharing nothing; nonwords are length-scaled, class-preserving phoneme edits that are not any English word and are phonotactically legal. Fallbacks when a category has no candidate are recorded in `fallback_from`. | Qualitative; the nonword edit rate is an ASSUMPTION pending category-coded PSST data. |
| `learning.py` | Item-level learning and forgetting: ACT-R activation with per-trace decay that depends on activation at practice (Pavlik & Anderson 2005), which produces the spacing effect mechanically, plus trace weights for successful retrieval, failed retrieval with feedback, and errorless study (the retrieval-practice effect). | Fitted to Middleton, Rawson & Verkuilen 2019 item-level data by simulating their protocol (`learning_fit.py`, `results/learning_fit.md`): all 14 pooled condition x delay cells within 0.09, spacing and retrieval-practice effects and 7-to-28-day forgetting reproduced. Individual differences in the retrieval-practice effect are not captured. |
| `treatment.py` | `TreatablePatient`: the snapshot patient plus one memory per word. Baseline retrievability per word equals the snapshot's P(correct) for that item; `attempt` (retrieval practice with feedback), `study` (errorless), `probe` (no trace), and `session` schedule practice over days. Errors on failed retrieval still come from the SP model conditioned on an error. | Cues are not modelled yet (next layer). |
| `patient.py` | `Patient.name(item)`: per-item effective weights `(s, p) * exp(-item_beta * difficulty)`, category from the SP model, word from the realizer, and an utterance-duration proxy. Distinct patients get distinct random streams; diagnostics use a separate stream. | Item-effect mechanism and strength are ASSUMPTIONS. Durations are PSST clip lengths, not response latency, and are never reported as latency. |

## Validation so far (`results/engine_validation.md`)

This is a held-out consistency check of the severity mapping, not a validation of the error
structure. On the 33 PSST validation/test participants (not used to fit the curve), noun items only,
a matched synthetic cohort reproduces real accuracy to within 0.02 in the lowest and highest severity
bands and overshoots by 0.08-0.10 in the two middle bands (n = 12 and 8 participants, 15 items
each). Synthetic accuracy correlates 0.43 with severity against 0.52 for the real participants.
PSST codes correctness only. `scripts/code_psst_responses.py` machine-codes its transcripts into
PNT-style types with a confidence; a clinician on the team adjudicated the 278 uncertain noun rows
and then a 50-row audit (`data/derived/CODING_GUIDE.md`; agreement in
`results/psst_coder_agreement.md`). The audit checked 30 rows the machine had coded with high
confidence: all 10 semantic calls held, and 17 of 20 phonological-nonword calls stayed nonwords
(8 as fragments), while 3 were productions she judged effectively correct that PSST scored wrong.
The 328 adjudicated rows are ground truth for `scripts/learn_coder.py`, a random forest on 42
features of the target/response pair that codes the remaining 91 uncertain noun rows; by
leave-one-target-out cross-validation it is right 51% of the time on nine categories and 68% on
the five SP error types, against 35% and 42% for the rules; on the 20 audited rows it had coded, it
was right on 8 (`results/psst_coder_learned.md`). Precedence downstream: clinician > learned >
rules. Table 2c of `results/engine_validation.md` compares error-only mixes on that basis. The lesion
prior was then fitted to the 74 training-split participants' mix (`scripts/fit_population_prior.py`,
`results/population_prior_fit.md`): log p/s mean -0.2, sd 0.3, against the +0.5, 0.6 borrowed from
Dell's Philadelphia patients, which is kept as `PopulationConfig.dell_philadelphia()`. On the 33
held-out participants (154 codable errors) the fitted engine's nonword share is 0.56 against a real
0.53, where Dell's prior gave 0.30. What the prior cannot fix is the word-error split: real patients
make semantic 0.37, formal 0.01, unrelated 0.07 of their errors; the engine makes 0.14, 0.17, 0.11
at any prior setting. That comes from the SP model's six-word neighbourhood (two formal and two
unrelated competitors against one semantic), a known approximation of the published model. Item
difficulty is not the cause (synthetic nonword share on the 13 BNT items moved only 0.02). Accuracy
by severity is unchanged by the prior, as it should be, since the sampler matches accuracy
separately.

### Cues (`results/cue_calibration.md`)

Cue activation levels were fitted on a synthetic cohort with unaided accuracy near Nardo et al.
2017's 0.60 so that the share of otherwise-failed items rescued matches theirs: initial phoneme 0.70
and whole word 0.82 are reproduced. The final-phoneme cue cannot be: its best rescue share is 0.24
against 0.475 observed, because in the SP model's six-word neighbourhood both formal competitors
share the target's coda, so a coda cue lifts them as much as the target. That is a stated limit of
the model's three-slot phonology, not a tuning failure. Two predictions the fit did not touch: the
initial-phoneme rescue share falls with severity, 0.79 in mild to 0.46 in severe patients, matching
Pease & Goodglass 1978's direction and their 49% figure for the severe band; and the semantic cue
helps patients with a semantic lesion most (0.42 against 0.15 for phonological lesions), which is
the opposite of what Python et al. 2021 reported for lexical-semantic anomia, so the semantic cue
mechanism is suspect and its level is an assumption anyway. In the learning layer a cued success
lays a trace weighted between an unaided success and errorless study by the cue's effort share;
the direction follows Howard et al. 1985 and Middleton et al. 2019, the values are not fitted, and
no cued-therapy retention data has been compared yet.

## Known limitations

- Single-word naming only. No sentences, discourse, comprehension, motor speech or conversation.
- Omissions and circumlocutions: the SP model's lexical-threshold variant is wired
  (`omission_threshold`) but not calibrated, and descriptions are not generated.
- The severity curve rests on 74 participants with 15 noun items each; its heterogeneity term is
  an estimate after removing binomial noise. BNT items are harder than the PNT's.
- Item effects and nonword edit rates are assumptions until item-level and category-coded patient
  data are available. The 39 compound-name items (the hardest) are skipped by the realizer.
- The learning layer is fitted to four participants with anomic aphasia (AQ 81-90) on personally
  unnameable proper nouns; its transfer to common nouns and to severe patients is an assumption.
- Cues (phonemic, semantic) are not modelled: a cue should raise retrieval probability on the trial
  and lower the weight of the trace it leaves. That is the next layer.

## Usage

```python
import numpy as np
from patientmodel import DellSPModel, load_dell_sp_config
from patientmodel.fit import load_or_build_grid
from patientmodel.items import load_item_bank
from patientmodel.lexicon import build_lexicon
from patientmodel.patient import Patient
from patientmodel.population import PopulationSampler
from patientmodel.realize import Realizer

cfg, fit_cfg = load_dell_sp_config()
model = DellSPModel(cfg)
items = load_item_bank()
realizer = Realizer(build_lexicon(extra_words=[i.word for i in items if i.pronounceable]))
grid = load_or_build_grid(model, fit_cfg["grid_step"], fit_cfg["trials_per_cell"], fit_cfg["seed"])
sampled = PopulationSampler(grid).from_profile("moderate", 1, seed=0)[0]
patient = Patient(sampled.patient_id, sampled.s, sampled.p, model, realizer, aq=sampled.aq)
for r in patient.naming_test([i for i in items if i.pronounceable][:10]):
    print(r.item.word, "->", r.category, r.word, f"{r.duration_s:.1f}s")
```
