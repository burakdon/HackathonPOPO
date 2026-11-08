# Real data: Philadelphia Naming Test (PNT)

Three files ground the data-based simulator (`configs/sim_pnt.yaml`) in real people with aphasia.

| file | rows | contents |
|---|---|---|
| `pnt_items.csv` | 175 | The 175 PNT target words with their **measured** 1PL (Rasch) item difficulty `irt_difficulty_t` on the T-scale (common discrimination 0.182 per T-point). `frequency` (Zipf, from the `wordfreq` package), `length` (letters) and `tier` (difficulty terciles) are added by us. `image_file` is a placeholder path: **the PNT pictures are copyrighted test materials and are not included.** |
| `pnt_thetas.csv` | 335 | PNT naming-ability scores (T-scale) of **335 people with aphasia**, described by the source app as coming from the Moss Aphasia Psycholinguistics Project Database (MAPPD) and an R03 study sample. No identifiers. |
| `mappd_naming_recovery.csv` | 115 | PNT results of **23 people with aphasia** (6 anomic, 9 conduction, 8 Wernicke's) at the start of a treatment trial (`TestTime` 0) and at later test points: proportion correct and proportions of semantic, mixed, formal, unrelated and nonword errors. Anonymous IDs. Used to fit the demo personas (docs/PERSONAS.md). |

`mappd_naming_recovery.csv` is the `NamingRecovery` dataset of the gazer R package
(https://github.com/jgeller112/gazer, `data/NamingRecovery.rda`, GPL-3), documented there as coming
from the Moss Aphasia Psycholinguistics Project Database (mappd.org). Exported with pyreadr. Its
later test points are real recovery trajectories during treatment, a possible check on the
simulator's learning rule (not done yet).

## Where the numbers come from

- Item difficulties and the ability scale: Fergadiotis, Kellough & Hula (2015). Item response theory
  modeling of the Philadelphia Naming Test. *JSLHR* 58(3), 865-877.
  Dataset: https://asha.figshare.com/articles/dataset/Item_Response_Theory_of_Philadelphia_Naming_Test_Fergadiotis_et_al_2015_/6170306
- Computer-adaptive PNT: Hula, Kellough & Fergadiotis (2015), and Hula et al. (2020).
- PNT: Roach, Schwartz, Martin, Grewal & Brecher (1996). The Philadelphia Naming Test: Scoring and rationale.
- MAPPD: Mirman, Strauss, Brecher, Walker, Sobel, Dell & Schwartz (2010). A large, searchable,
  web-based database of aphasic performance on picture naming and other tests of cognitive function.
  *Cognitive Neuropsychology* 27(6), 495-504.

Extracted from the internal data of the open-source PNT computer-adaptive testing app
(https://github.com/rbcavanaugh/pnt, `R/sysdata.rda`, objects `items` and `thetas`) with
`scripts/export_pnt_data.py` (needs `pyreadr`, not a project dependency). That app is licensed
AGPL-3.0. We use only numeric research values (item difficulties and anonymous ability scores),
not code; **before any public or commercial release, confirm the terms with the authors** or
re-derive the values from the original publications and datasets above.

## What is and is not grounded

Grounded in data: which words, how hard each word is (measured on real patients), and how naming
ability is spread across real people with aphasia (the training population).
Not yet grounded: how a patient's difficulty splits between word selection and sound encoding
(`profile_beta`), cue effects, fatigue and learning. MAPPD also has per-patient error-type
counts (semantic, formal, nonword...), which could fit the split; access needs registration.
