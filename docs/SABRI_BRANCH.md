# Sabri_Branch

Up to date with `dev` (includes PR #5). `main` is untouched. 157 tests pass, 1 skipped (`uv run pytest`).

## History
The first commit built a live DITTO loop in the service. Meanwhile `dev` gained its own, already
validated live loop (`POST /learn_from_overrides`, `service/live_ditto.py`). The merge keeps **dev's
live loop unchanged** and drops the overlapping service code from this branch. What remains is
additive and does not touch the service:

| addition | files |
|---|---|
| Paper-faithful DITTO variant (SFT init + SFT reference, 70/20/10 online/replay/intermodel) and an SFT behaviour-cloning baseline. Default `ditto()` behaviour is unchanged. | `agents/ditto.py`, `configs/ditto.yaml` |
| Sweep: SFT vs both DITTO variants, 1 to 10 overrides, 3 clinician rules, 3 seeds | `train/ditto_sweep.py`, `results/ditto_sweep.{md,csv,png}` |
| Fix: offline overrides use the cue-only-after-error gate; impossible overrides are skipped and reported | `train/ditto_ppo.py` |
| Tests for the above | `tests/test_ditto_paper.py` |

## For the slides
- "Why not just fine-tune on the overrides?" DITTO beats behaviour cloning at every n from 2 up
  (0.69 to 0.71 vs 0.59 adoption at 3 overrides; 0.93 to 0.95 vs 0.83 at 10). Chart: `results/ditto_sweep.png`.
- "How many overrides?" About 70% adoption at 3, about 80% at 5, above 90% at 10. Plan the demo for 5+.

## Open points for the team
- The live endpoint counts every agent choice in the session that was not overridden as accepted.
  That holds while the clinician watches the whole session. If home practice ever shares a session id
  with a supervised session, unsupervised choices would be treated as approval; a `reviewed` flag on
  `/update` would fix it (the first commit on this branch had one).
- The paper variant could be offered on `/learn_from_overrides` (pass `mixture=` and `sft_steps=`).
  It was not validated with the live endpoint's settings, so it is not switched on.
