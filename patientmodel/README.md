# patientmodel: the synthetic aphasia patient engine

A mechanistic simulator of single-word retrieval in aphasia. Two cognitive parameters and a
severity level produce a patient who names pictures, makes the error types real patients make in
the proportions real patients make them, says actual words or nonwords, takes a realistic time, and
is harder to understand on harder pictures. Everything is seeded and reproducible; a 4,000-trial
patient simulates in about 80 ms.

The trainer (`agents/`, `envs/`, `service/`) is the engine's first application. The engine's own
deliverables are synthetic patients, cohorts and datasets, validated against real data.

## Modules

| module | what it is | grounded on |
|---|---|---|
| `dell_sp.py` | The semantic-phonological interactive two-step model of lexical access (Foygel & Dell 2000). Parameters s (semantic-lexical weight) and p (lexical-phonological weight); six-word neighbourhood, 57 semantic units, 10 phonemes, decay .6, two noise terms, 8 steps per stage. Returns the six PNT response categories. | Dell et al. 2004 (architecture, constants); Walker 2016 (lexicon). Reproduces the SP-model proportions Dell published for 8 fitted patients to rmsd .005 and names 97% correctly at normal weights (`tests/test_dell_sp.py`). |
| `fit.py` | Fits (s, p) to a patient's response counts: cached simulated probability grid, maximum likelihood, Dell's rmsd, grid posterior with 95% intervals. | Refitting Dell's 2004/2013 patients from their published proportions recovers his s and p to about .003 (`scripts/fit_dell_sp.py`). |
| `population.py` | Samples patients: AQ severity, then a target accuracy from a logistic severity curve with between-patient heterogeneity, then an (s, p) cell consistent with it under a lognormal prior on p/s. Profiles mild / moderate / severe / very severe. | Severity curve and heterogeneity from the 107 PSST participants; p/s prior from Dell's fitted patients. |
| `items.py` | 520 IPNP object pictures with measured norms and a difficulty index (name agreement, RT, frequency, length); 244 drawings. | Bates et al. 2003; Szekely et al. 2004. |
| `lexicon.py` | Vocabulary of about 5,800 concrete words with ARPAbet pronunciations, Zipf frequency and sentence-embedding vectors; PNT formal-relatedness criterion; phoneme edit distance. | CMUdict; Brysbaert et al. 2014 concreteness; wordfreq; all-MiniLM-L6-v2. |
| `realize.py` | Turns a response category into an actual response for a target: semantic neighbours by embedding similarity, formal neighbours by the PNT criterion, mixed = both, unrelated = frequent concrete word sharing nothing, nonwords = length-scaled within-class phoneme edits kept pronounceable. Candidates weighted by relatedness and frequency. | Qualitative; the nonword edit rate is an ASSUMPTION pending category-coded PSST data. |
| `patient.py` | `Patient.name(item)`: per-item effective weights `(s, p) * exp(-item_beta * difficulty)`, category from the SP model, word from the realizer, latency from a log-normal located at the PSST medians. `naming_test(items)` runs a test. | Item-effect strength `item_beta` is an ASSUMPTION to calibrate. |

## Validation so far (`results/engine_validation.md`)

A synthetic cohort matched to the PSST participants' severity scores and item counts reproduces
their accuracy by severity band within 0.03 in three of four bands (the most severe band is too
accurate by about 0.09), with a correlation of 0.91 between synthetic accuracy and severity (real: 0.81).

## Known limitations

- Single-word naming only. No sentences, discourse, comprehension, motor speech or conversation.
- Omissions and circumlocutions: the SP model's lexical-threshold variant is wired (`omission_threshold`)
  but not calibrated, and descriptions are not generated.
- The error vocabulary uses concreteness as a proxy for "picturable noun"; some non-nouns slip in.
- Item effects and nonword edit rates are assumptions until PSST responses are category-coded.
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
realizer = Realizer(build_lexicon(extra_words=[i.word for i in items]))
grid = load_or_build_grid(model, fit_cfg["grid_step"], fit_cfg["trials_per_cell"], fit_cfg["seed"])
sampled = PopulationSampler(grid).from_profile("moderate", 1, seed=0)[0]
patient = Patient(sampled.patient_id, sampled.s, sampled.p, model, realizer, aq=sampled.aq)
for r in patient.naming_test(items[:10]):
    print(r.item.word, "->", r.category, r.word, f"{r.latency_s:.1f}s")
```
