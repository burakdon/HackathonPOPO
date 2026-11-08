# patientmodel: the synthetic aphasia patient engine

A mechanistic simulator of single-word retrieval in aphasia. Two cognitive parameters and a
severity level produce a patient who names pictures, makes the error types real patients make in
the proportions real patients make them, says actual words or nonwords, takes a realistic time, and
is harder to understand on harder pictures. Everything is seeded and reproducible; a 4,000-trial
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
| `population.py` | Samples patients: AQ severity, a target accuracy from a logistic severity curve with between-patient heterogeneity, then an (s, p) cell consistent with it under a lognormal prior on p/s. Profiles mild / moderate / severe / very severe. | Severity curve fitted on the PSST noun items of the 74 training-split participants; heterogeneity after removing binomial noise; p/s prior from Dell's fitted patients (flagged: a selected sample). |
| `items.py` | 520 IPNP object pictures with norms and a difficulty index (healthy RT, frequency, length; name agreement deliberately excluded); 244 drawings; 39 compound-name items flagged `pronounceable=False`. | Bates et al. 2003; Szekely et al. 2004. |
| `lexicon.py` | About 4,800 picturable nouns: concreteness >= 4, WordNet noun, CMUdict pronunciation, not on the profanity or sensitive-word lists; Zipf frequency; MiniLM embeddings; the full CMUdict pronunciation set; the PNT formal-relatedness criterion; English phonotactic legality. | CMUdict; Brysbaert et al. 2014; WordNet; wordfreq; all-MiniLM-L6-v2. |
| `realize.py` | Turns a response category into an actual response for a target: semantic neighbours need embedding similarity plus a WordNet taxonomic relation (or very high similarity); formal neighbours meet the PNT criterion and are not semantic; mixed need both and shared onset or <= 2 edits; unrelated are frequent nouns sharing nothing; nonwords are length-scaled, class-preserving phoneme edits that are not any English word and are phonotactically legal. Fallbacks when a category has no candidate are recorded in `fallback_from`. | Qualitative; the nonword edit rate is an ASSUMPTION pending category-coded PSST data. |
| `patient.py` | `Patient.name(item)`: per-item effective weights `(s, p) * exp(-item_beta * difficulty)`, category from the SP model, word from the realizer, and an utterance-duration proxy. Distinct patients get distinct random streams; diagnostics use a separate stream. | Item-effect mechanism and strength are ASSUMPTIONS. Durations are PSST clip lengths, not response latency, and are never reported as latency. |

## Validation so far (`results/engine_validation.md`)

This is a held-out consistency check of the severity mapping, not a validation of the error
structure. On the 33 PSST validation/test participants (not used to fit the curve), noun items only,
a matched synthetic cohort reproduces real accuracy to within 0.02 in the lowest and highest severity
bands and overshoots by 0.08-0.10 in the two middle bands (n = 12 and 8 participants, 15 items
each). Synthetic accuracy correlates 0.43 with severity against 0.52 for the real participants.
PSST codes correctness only, so the error-type mix cannot be validated against it until its
transcripts are coded into the six PNT categories, which is the next validation task.

## Known limitations

- Single-word naming only. No sentences, discourse, comprehension, motor speech or conversation.
- Omissions and circumlocutions: the SP model's lexical-threshold variant is wired
  (`omission_threshold`) but not calibrated, and descriptions are not generated.
- The severity curve rests on 74 participants with 15 noun items each; its heterogeneity term is
  an estimate after removing binomial noise. BNT items are harder than the PNT's.
- Item effects and nonword edit rates are assumptions until item-level and category-coded patient
  data are available. The 39 compound-name items (the hardest) are skipped by the realizer.
- No learning yet: the patient is a snapshot. Practice-driven change over days is the next layer.

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
