# Sabri_Branch: closing the DITTO loop

Branched from `dev` at `380c48a`. `main` is untouched. All 137 tests pass (`uv run pytest`).

## What this branch adds

| change | files | why |
|---|---|---|
| Live DITTO loop: serve PPO, train from dashboard overrides, hot-swap | `service/live_ditto.py`, `service/app.py`, `configs/service.yaml` | Before, the service ran a bandit and DITTO only ran offline on simulated overrides, so the demo could not show "she overrides, it learns". |
| `/ditto/status`, `/ditto/train`, `/ditto/reset` endpoints | `service/app.py` | One button on the dashboard: "Learn from my overrides". |
| Accepted-choice log (anchors) from real clinician behaviour, only for `"reviewed": true` trials | `core/schema.py`, `service/app.py` | Anchors were simulated only. Unsupervised home practice must never count as approval. |
| Paper-faithful DITTO variant (SFT init, SFT reference, 70/20/10 online/replay/intermodel) | `agents/ditto.py`, `configs/ditto.yaml` | The existing variant left out SFT initialisation and intermodel comparisons. The legacy variant is unchanged and still the default offline. |
| SFT (behaviour-cloning) baseline + sweep over 1 to 10 overrides, 3 rules, 3 seeds | `train/ditto_sweep.py`, `results/ditto_sweep.*` | Judges will ask "why not just fine-tune on the overrides?" The chart answers it. |
| Bug fix: overrides the policy cannot produce (cue after a correct answer) are skipped and reported | `train/ditto_ppo.py` | Real overrides were masked without the cue-only-after-error gate the policy uses. |
| Demo rehearsal script | `scripts/ditto_live_demo.py` | Runs the whole before / teach / train / after loop over HTTP. |

## Results to put on the slide

- With the same overrides, anchors and compute, **DITTO beats plain fine-tuning (SFT)** at every
  n from 2 up: 0.69 to 0.71 vs 0.59 adoption at 3 overrides, 0.93 to 0.95 vs 0.83 at 10.
- Live, over HTTP: after a sound error the agent gives the first-sound cue in **34% -> 63% (3
  overrides), 76% (5), 96% (10)** of held-out decisions; training takes about 2 s.
- 99% of unrelated decisions are unchanged; changes after *other* errors stay at 4 to 13%.

## For the frontend (dashboard)

1. Start the service with `AGENT=ppo uv run uvicorn service.app:app --host 0.0.0.0 --port 8000`.
2. While the clinician is watching, send `"reviewed": true` on every `POST /update`.
3. Override button: unchanged (`POST /override`, then show and `/update` the clinician's action).
4. Add a "Learn from my overrides" button -> `POST /ditto/train`. Show `agreement_before ->
   agreement_after` and the `per_override` list (base choice vs tuned choice vs clinician).
5. Optional "Reset" button -> `POST /ditto/reset`. `GET /health` shows `ppo` or `ppo+ditto`.

## Demo script for the DITTO minute

1. The agent runs on a teammate. After a sound error it picks, e.g., a semantic cue.
2. The clinician overrides 3 to 5 times ("after a sound error I give the first sound"), then stops.
3. Press "Learn from my overrides" (2 s). Show the per-override table flipping to her choice.
4. Keep going: after the next sound error the agent picks the first-sound cue on its own.
5. If live is risky, run `uv run python -m scripts.ditto_live_demo --inprocess --overrides 5` and show its output.

## Honest caveats (keep these in the README and the Q&A)

- These are simulated clinicians with one consistent rule. A real SLP's preferences are noisier.
- With 3 overrides the effect is clear but partial (63%); 5 to 10 is more reliable.
- The anchors' "not overridden = accepted" assumption only holds while she is watching.
- The two DITTO variants differ by less than one standard deviation at 3 overrides; do not claim
  the paper variant is better there.
