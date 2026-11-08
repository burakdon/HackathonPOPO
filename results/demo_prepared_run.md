# Prepared per-patient run for the demo: demo-ana-r1

Made with `scripts/prepare_demo_run.py` through the real endpoints (after `scripts/seed_demo_sessions.py`), on an
Apple M5 laptop. Note: "When meaning hints have repeatedly failed, try a first-letter hint before moving to another
picture." Rule (entered by hand; no API key for drafting): meaning hint already shown and 2+ failed tries -> first-
letter hint. The shared adapter `ditto:sft200-g2-s0` chose the first-sound hint in all 11 tickable seeded decisions.

Training (`training/patient_ditto.py`, from the shared adapter): 5 ticked examples (x-0001 to x-0005, repeated 4
times) + 30 anchors; SFT 30 steps, 3 gentle DITTO rounds; best stage on the training rows: round 2 (agreement
sft 49/50, round1 48, round2 50, round3 50). Total 411 s including the preview (383 s training).

Preview (service/preview.py, real model choices):

| | new adapter | current (shared) | total |
|---|---|---|---|
| follows_your_examples (trained on these) | 5 | 0 | 5 |
| follows_your_rules_new_situations (generated from the rule, other pictures) | 5 | 0 | 10 |
| unchanged_elsewhere (held-out test situations, same choice as current) | 9 same | | 10 |

Real demo situations (meaning hint failed twice, tries left; options: first sound, first letter, sentence):

| decisions | base Qwen3-1.7B | shared adapter | patient adapter demo-ana-r1 |
|---|---|---|---|
| 5 trained on | first sound 4, sentence 1 | first sound 5 | **first letter 5** |
| 6 never trained on (other seeded decisions) | first sound 5, sentence 1 | first sound 6 | **first letter 6** |

Latency per decision on the laptop: about 0.3-0.4 s for all three. The 6 unseen decisions are real seeded situations
of the same kind (other pictures or sessions), not new patients; the generated situations of the preview are the
harder check (5/10). One example among 5 and one rule: an exploratory demo, not a clinical result.
