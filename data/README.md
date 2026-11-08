# Data for the patient model

Raw third-party data lives under `data/raw/` and is **not committed** (licenses differ and the
archives are large). `uv run python -m scripts.fetch_data` re-downloads everything that is publicly
reachable; the two gated sources need the credentials described below.

## What each source is for

| layer | source | status | folder |
|---|---|---|---|
| Patient snapshot model (Dell SP model: who the patient is, which errors they make) | MAPPD, item-level Philadelphia Naming Test responses for 240+ patients (Mirman et al. 2010) | **Gated.** mappd.org is down (expired certificate). As of Aug 2025 the site read "contact Dan Mirman (dan@danmirman.org) for access." Email sent? ___ | `data/raw/mappd/` |
| Fallback for the snapshot model | Published S/P weight distributions and fitting method: Foygel & Dell 2000; Dell et al. 2004; Dell et al. 2013; Walker 2016 (WebFit, Bayesian fit) | PDFs for Dell 2004, Dell 2013, Mirman 2010, Castro 2025 downloaded. Foygel & Dell 2000 and Walker 2016 are paywalled / blocked to scripts: fetch by hand via the library. | `data/raw/refs/` |
| Learning model (how practice changes retrieval over days) | Middleton, Rawson & Verkuilen 2019, *Cortex*: item-level retention and follow-up accuracy for 4 PWA, retrieval practice vs repetition, spaced (lag 24) vs massed, 192 items each | **Downloaded** from OSF `cwq75` (no license stated on OSF; cite the paper, do not redistribute). | `data/raw/middleton2019_osf/` |
| Learning model, second anchor | Middleton et al. 2016, *JSLHR*: retrieval practice and spacing (lags 1/5/15/30), retention at 1 day and 1 week | Paper text read; per-condition accuracies are in supplements S5/S6 on the publisher site. No open data. | — |
| Item bank and difficulty norms | International Picture Naming Project (Bates et al. 2003, Szekely et al. 2004): 520 object drawings, US English norms (name agreement, RT, no-response rate, frequency, syllables, visual complexity) | **Downloaded.** `USpno.xls` (520 x 27), `7lgpno.xls` (7 languages), variable docs, 244 original PNG drawings. Pictures are described as freeware; cite the papers. | `data/raw/ipnp/` |
| Speech scorer training and evaluation; realism of simulated attempts; demo clips | PSST challenge data (Gale et al. 2022): 3,291 utterances of BNT/VNT naming from AphasiaBank with ARPAbet transcripts, correctness, WAB-R AQ | **Gated.** Needs AphasiaBank membership, then `uv run python -m psstdata` prompts for the PSST username/password and downloads ~437 MB to `~/psst-data`. | `~/psst-data` |

## Middleton 2019 files (OSF cwq75)

- `week.txt`: 768 rows, retention test one week after training. Columns `Test, Subject (P1-P4), Training (repetition|retrievalpractice), Spacing (Massed|Spaced), ItemNumber, ResponseAccuracy (0/1)`.
- `month.txt`: 768 rows, same columns, follow-up test one month after training. CR line endings.
- `spacingbysessioninteraction.txt`: 1,152 rows, within-training accuracy on retrieval-practice items by `Day (1-3), Subject, ItemNumber, Spacing`.
- `analysis_code.docx`: the authors' Stata code.

This is the only public item-level dataset we have found on how practice schedule changes naming
over days in aphasia. It anchors the learning model's spacing and retrieval-practice effects; it is
four participants, so it constrains effect direction and rough size, not individual dynamics.

## IPNP US English norms (`USpno.xls`, sheet `database`)

Key columns: `picnum`, `ename` (dominant name), `elex1` (% producing the target name), `elex13`
(% target + variants + synonyms), `erttar` (mean RT to target, ms), `enoname` (% no response),
`estatu` (H statistic, name agreement), `lnefreq` (log frequency), `esyll`, `echar`, `ecomplx`
(visual complexity), `ecdi` (age of acquisition proxy). Full descriptions in `7lgpnovariables.html`.

## Credentials to obtain

1. **AphasiaBank membership** (https://aphasia.talkbank.org/). Unlocks PSST and the wider naming
   data. One person applies; credentials go in `~/.config/psstdata/settings.json`, never in the repo.
2. **MAPPD access**: email Dan Mirman. Item-level PNT responses are what the per-patient Dell fits
   need; without them we sample patients from published parameter distributions.

## Calibration

`data/calibration/dell2004_table2_sp_fits.csv` holds the observed PNT proportions and the (s, p)
Dell's own fitter chose for 14 patients (Dell et al. 2004, Table 2, SP-ind rows) plus the Dell et al.
(2013) Table 1 example. `uv run python -m scripts.fit_dell_sp` fits our implementation to the same
proportions and prints both side by side; mean |delta s| and |delta p| are about 0.003 on a 0.002 grid.
The simulated probability grid is cached under `data/cache/` (gitignored, rebuilt in about 80 s).
