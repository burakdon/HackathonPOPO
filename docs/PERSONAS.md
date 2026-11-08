# Demo patients: Maria and James

Two **composite personas** (not real people). Each one is **fitted to the average Philadelphia Naming
Test (PNT) results of a group of real patients** in `data/mappd_naming_recovery.csv` (23 people with
aphasia from the Moss Aphasia Psycholinguistics Project Database, PNT at the start of a treatment
trial; see `data/PNT_SOURCE.md`). Simulator: `configs/sim_pnt.yaml`, presets `demo_maria` and
`demo_james`. Check and agent results: `uv run python -m scripts.demo_patients` ->
[`results/demo_patients.md`](../results/demo_patients.md).

How the fit works: the two-step model gives each persona a meaning -> word strength (s) and a
word -> sounds strength (p). s and p are chosen so that, over the 175 real PNT items with their
measured difficulty, the persona's accuracy and the share of errors that are sound-based (formal +
nonword) vs wrong-word (semantic + mixed + unrelated) equal the real group's averages.

## Maria, 62: conduction aphasia, "the word is there, the sounds come out wrong"

- **Fitted to 9 real patients with conduction aphasia:** 53% correct on the PNT; 73% of errors are
  sound-based ("lat" or "mat" for "cat"). The persona matches both.
- **Naming ability:** PNT about 47, the **32nd percentile of 335 real people with aphasia**.
- **What helps (model):** the first-sound cue (+0.26 on tier-2 items) and first letter (+0.16);
  semantic cues barely help (+0.03).
- **In simulation (50 sessions):** with fixed difficulty she quits **26%** of sessions and is in the
  75-90% band 19% of the time; with our agent she quits **2%**, is in band 42% of the time, and her
  learning gain more than doubles (2.96 vs 1.31 x100).
- Invented details (persona only): retired teacher; personal words are her grandchildren's names,
  garden plants, classroom objects; tires after 20 to 25 minutes.

## James, 58: anomic aphasia, "I know it... it's the thing you..."

- **Fitted to 6 real patients with anomic aphasia:** 66% correct; 58% of errors sound-based and 42%
  wrong-word. The persona matches both.
- **Naming ability:** PNT about 50, the **44th percentile** of the same 335 real people.
- **In simulation:** our agent keeps him in band more (64% vs 51%) with slightly higher learning
  gain (4.27 vs 4.00 x100); the gap is smaller than for Maria because fixed tier 2 already suits him.
- **Clinician demo moment:** after James's wrong-word errors the agent mostly gives the first-sound
  cue (46%). Many clinicians would give a meaning-based cue there (semantic feature analysis is a
  standard approach for anomia). That is a clinical preference the reward cannot express, which is
  exactly what learning from the clinician (DITTO) is for: she overrides a few of those choices, and
  the agent adopts her approach for James.
- Invented details: former bus driver; his daughter helps him practise.

## Grounded vs assumed

| part | source |
|---|---|
| Accuracy and error mix (sound-based vs wrong-word) | Real group means: 9 conduction and 6 anomic patients (MAPPD, via data/mappd_naming_recovery.csv) |
| Placement among people with aphasia | Real PNT ability scores of 335 people (data/pnt_thetas.csv) |
| Words and their difficulty | The 175 PNT words with difficulty measured on real patients (Fergadiotis et al. 2015) |
| Structure linking errors to cues | Two-step model of word retrieval (Dell et al. 1997; Foygel & Dell 2000) |
| How much each cue helps, omissions, fatigue, learning | ASSUMPTIONS, to be reviewed by the clinical lead |
| Life details | Invented |

Limits: group averages over 9 and 6 patients, not individuals; no recordings are used anywhere; the
real data does not code omissions separately, so the personas have few no-responses (5%).

Say in the pitch: "Maria and James are composites fitted to real patients' naming test results:
Maria to nine people with conduction aphasia, James to six with anomic aphasia."
