# Loading the DITTO adapter

A LoRA adapter that makes a small language model choose practice decisions (retry with a hint, or move
on to a new picture at a tier) the way demonstrator A does. Trained with `training/ditto.py` (SFT,
then 3 rounds of DPO with expert / replay / intermodel comparisons) on the 20 training situations in
`data/demonstrations/demonstrator_a_v1.jsonl`. The demonstrations are team-authored fictional
scenarios pending human review, not clinician-validated treatment knowledge.

## What to load

| | |
|---|---|
| base model | `Qwen/Qwen3-1.7B` |
| base revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (pinned in `configs/practice.yaml`) |
| run id | `20261107-214025` |
| adapter (committed) | `models/ditto_llm/20261107-214025/` (final DITTO round, 25.7 MB) |
| also in the folder | `manifest.json` (config, data hash, losses, per-round sample and pair counts), `eval_test.json` (held-out check below). The SFT and earlier-round adapters stay in the gitignored `runs/ditto/20261107-214025/` |
| LoRA | r 16, alpha 32, dropout 0, on `q_proj k_proj v_proj o_proj` |
| prompt template | Qwen3 chat template, `add_generation_prompt=True`, `enable_thinking=False` |
| decoding | greedy, `max_new_tokens` 24 |

Library versions used for training and for the example below (the `llm` extra in `pyproject.toml`):
Python 3.11.16, torch 2.15.0, transformers 5.22.2, peft 0.21.2, accelerate 1.15.0. Trained on Apple
MPS (M5, 24 GB), bfloat16, in 270 s (4.5 min; SFT stopped after 6 steps, then 3 DITTO rounds of 15 DPO steps), peak MPS memory 4.1 GB.

```bash
uv sync --extra llm
```

## Example: patient_state in, `{"action_id"}` out

`training/team_format.py` converts the team's `patient_state.json` into the exact prompt the adapter
was trained on, and maps the model's answer back to one of the state's `available_actions`.

```python
import json

from agents.llm_runtime import LLMRuntime
from core.practice import load_practice
from training.team_format import model_messages, to_action_id

state = json.load(open("patient_state.json"))     # the team's agreed format
pcfg = load_practice()                             # base model, revision, decoding settings
rt = LLMRuntime(pcfg.llm)
rt.ensure_adapter("ditto", "models/ditto_llm/20261107-214025")

msgs = model_messages(state, pcfg.history_in_prompt)
text, seconds = rt.generate(msgs, adapter="ditto")   # adapter=None runs the unadapted base model
print(to_action_id(text, state))                      # {"action_id": "act-n2"} for the input below
```

On the example input below this prints `{"action_id": "act-n2"}` (new picture, same tier) with both
the adapter and the base model, in about 0.4 s. The demonstrator's rule D would retry with the first
sound (`act-fs`): see the held-out check for how often the adapter agrees.

`to_action_id` returns `None` when the output is not one of the available actions; the caller must
then fall back (the app falls back to the bandit baseline, see `agents/llm_policy.py`).

Without the repo's runtime, plain transformers + peft give the same result:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, rev = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
tok = AutoTokenizer.from_pretrained(base, revision=rev)
model = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "models/ditto_llm/20261107-214025").eval()
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=24, do_sample=False)
text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
```

## Example input

```json
{"patient_id": "p-001", "session_id": "s-001", "state_version": 3,
 "current_exercise": {"exercise_id": "ex-banana", "target_word": "banana", "difficulty_tier": 2},
 "recent_attempts": [
   {"attempt_id": "t1", "exercise_id": "ex-cat", "transcript": "cat", "result": "correct", "hint_used": null},
   {"attempt_id": "t2", "exercise_id": "ex-banana", "transcript": "bandana", "result": "sound", "hint_used": "none"}],
 "clinician_guidance": [],
 "available_actions": [
   {"action_id": "act-fs", "operation": "retry_current", "exercise_id": "ex-banana", "hint_type": "first_sound"},
   {"action_id": "act-sem", "operation": "retry_current", "exercise_id": "ex-banana", "hint_type": "semantic"},
   {"action_id": "act-n1", "operation": "next_exercise", "exercise_id": "ex-dog", "hint_type": null},
   {"action_id": "act-n2", "operation": "next_exercise", "exercise_id": "ex-chair", "hint_type": null}]}
```

Accepted values: `operation` is `retry_current` or `next_exercise`; `hint_type` / `hint_used` is
`none` (or null), `first_sound`, `first_letter`, `sentence_frame` or `semantic`; `result` is `correct`,
`sound`, `meaning`, `no_response` or `incorrect` (error type not identified). Synonyms such as
`phonemic` and `semantic` are mapped in `training/team_format.py`.

## Held-out check

`uv run --extra llm python -m evaluation.compare --run 20261107-214025` on the 10 reserved test
situations (never used in training or in prompts); counts out of 10:

| model | valid output | constraint violations | agrees with demonstrator | median latency |
|---|---|---|---|---|
| base Qwen3-1.7B | 10 | 0 | 2 | 444 ms |
| base + 20 demonstrations in the prompt | 10 | 0 | 5 | 1922 ms |
| SFT adapter | 10 | 0 | 3 | 445 ms |
| **DITTO adapter (this one)** | 10 | 0 | 4 | 446 ms |
| bandit baseline | n/a | n/a | 4 | 0 ms |

The adapter always produces a valid action and moves agreement from 2/10 (base) to 4/10, but it
does not beat the few-shot prompted base (5/10), which is about 4x slower. It gets the "correct
answer -> harder / same tier" rules mostly right and misses most "retry with the matched hint"
rules (D, E-alt) and "two failures -> easier" (F). Ten situations are a small exploratory check of
mock preferences, not a statistical or clinical validation.

## Training on the team's demonstrations

```bash
uv run python -m training.team_format demo1.json demo2.json --states saved_states/ --out runs/demonstrations/team.jsonl
uv run --extra llm python -m training.ditto --extra-demos runs/demonstrations/team.jsonl
```

`saved_states/` holds one `<state_snapshot_id>.json` patient_state per demonstration. Only
`scope: "global"` demonstrations are added to training. The preferred action and the clinician note
are stored with the row but never enter the model input.

## Field mapping and gaps

| team format | trainer format |
|---|---|
| `current_exercise.exercise_id / target_word / difficulty_tier` | `context.current.exercise_id / word / tier` |
| `recent_attempts[].exercise_id`, `hint_used` | `context.history[].exercise_id`, `hint` |
| `recent_attempts[].result` | `context.history[].correct` + `error` |
| attempts and hints on the current exercise | `context.current.attempts`, `hints_shown` (derived) |
| `available_actions[]` | `options` (canonical `{"op":"retry","hint":...}` / `{"op":"next","tier":...}`) |
| `preferred_action_id` (+ joined state) | `preferred` |
| `clinician_note`, `scope` | `note`, `scope` |
| `patient_id`, `state_snapshot_id`, `provenance` | `provenance` (not shown to the model) |

Missing in the team format (filled as stated):
- **response time** per attempt: the demonstrator's rule A/B depends on it (at most 4 s = one tier
  harder). Optional extra field `response_time_s`; when absent, 5 s is used, so a missing time never
  causes a tier increase.
- **session minutes**: rule G (20 min or more) depends on it. Optional extra field `session_minutes`;
  default 0.
- **tier and word of attempts on other pictures**, and the **tier of a `next_exercise` action**: looked
  up in `data/items.json` by `exercise_id`; an unknown next exercise is an error.
- **accepted actions** (used only by evaluation): set to the preferred action.
- **split, family_id, rule**: `train`, `state_snapshot_id`, none.

Not used by the trainer (kept in provenance only): `patient_id`, `session_id`, `state_version`,
`attempt_id`, `transcript`, `clinician_guidance`.

Ambiguity: two `next_exercise` actions at the same tier look identical to the model (it chooses a
tier, not a picture); the first one listed is returned.
