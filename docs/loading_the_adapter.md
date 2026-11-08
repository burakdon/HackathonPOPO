# Loading the DITTO adapter

The delivered shared adapter makes Qwen3-1.7B choose practice decisions (retry the picture with a hint,
or move to a new picture at a level) the way demonstrator A does. It was trained on Colab with
`notebooks/ditto_colab_round2.ipynb`: SFT for 200 steps, then gentle DITTO (setting g2), with the best
stage chosen on the dev split before any test evaluation. The demonstrations are fictional scenarios
generated from demonstrator A's written strategy, pending human review; they are not clinician-validated
treatment knowledge.

## What to load

| | |
|---|---|
| base model | `Qwen/Qwen3-1.7B` |
| base revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| adapter | `models/ditto_llm/sft200-g2-s0/` (LoRA r 16, alpha 32, q/k/v/o projections, 25.7 MB) |
| model version id | `ditto:sft200-g2-s0` (the default in `configs/practice.yaml`) |
| decoding | **option scoring**: the model scores every available action and the most likely one is taken, so the output is always valid |
| prompt | **format v2** (`core/practice.py`, `render_context(..., fmt="v2")`): options labelled in words, levels relative to the current picture |
| chat template | Qwen3, `add_generation_prompt=True`, `enable_thinking=False` |

The adapter only works with prompt format v2 and option scoring. `configs/practice.yaml` sets both
(`decoding: score`, `prompt_format: v2`), and the model registry refuses an adapter whose recorded prompt
format differs from the config. The earlier adapters in `models/ditto_llm/` (`20261107-214025`,
`v4_score-s0`) are kept for reference; they use prompt format v1.

Library versions (the `llm` extra in `pyproject.toml`): Python 3.11, torch 2.15.0, transformers 5.22.2,
peft 0.21.2, accelerate 1.15.0. Trained on an A100 (bfloat16) in 2.4 minutes.

```bash
uv sync --extra llm
```

## patient_state in, `{"action_id"}` out

```python
import json

from agents.llm_runtime import LLMRuntime
from core.practice import load_practice
from training.team_format import decide_action_id

pcfg = load_practice()                       # base model, revision, decoding: score, prompt_format: v2
rt = LLMRuntime(pcfg.llm)
rt.ensure_adapter("shared", "models/ditto_llm/sft200-g2-s0")

state = json.load(open("patient_state.json"))     # the team's agreed format
print(decide_action_id(rt, state, "shared", pcfg.history_in_prompt, fmt=pcfg.llm.prompt_format))
# {"action_id": "..."}: always one of state["available_actions"]; adapter=None scores with the base model
```

Inside the app, the same model is `ModelRegistry(pcfg).policy("ditto:sft200-g2-s0")`, whose `decide(ctx,
options, fallback)` returns a `Decision`. The registry finds delivered adapters in `models/ditto_llm/`
(`llm.models_dir`) as well as training runs in `runs/ditto/`.

Measured on an Apple M5 laptop (MPS, bfloat16): base model and adapter load in about 4 s once the model is
in the Hugging Face cache; a decision takes 0.25 s with 2-3 available actions and about 1 s with 7. On an
A100 a decision takes about 80 ms.

## Results (test split, 55 situations, evaluated once; 3 seeds)

From `results/ditto_colab_r2_test.md`. Agreement with demonstrator A, 95% bootstrap intervals over
situations; SFT and DITTO are means over 3 seeds.

| model | agreement on test |
|---|---|
| base Qwen3-1.7B | 16/55 (29%, CI 18-42%) |
| base + 20 demonstrations in the prompt | 21/55 (38%, CI 25-51%) |
| base + 128 demonstrations in the prompt | 15/55 (27%, CI 16-38%) |
| bandit baseline | 15/55 (27%, CI 16-40%) |
| SFT (200 steps) | 42.3/55 (77%, CI 68-85%) |
| **DITTO (delivered)** | 41.3/55 (75%, CI 65-84%) |

- DITTO vs base: +46 points (CI +32 to +59). DITTO vs 20-demonstration prompting: +37 (CI +18 to +56).
- **DITTO vs SFT: a tie** (-1.8 points, CI -7.3 to +4.2). DITTO was chosen because it led on dev (29/33 vs
  25/33), but that lead was the best of 9 candidate stages on 33 situations and did not carry over to test.
- Weak rules on test: F ("still wrong after two hints -> easier", 33%) and G ("error after 20 minutes ->
  easier", 27%), where few-shot prompting does better.
- Exploratory check on mock preferences: not a statistical or clinical validation.

## Field mapping between the team format and the trainer

`training/team_format.py` converts `patient_state.json` / `demonstration.json` to and from the trainer's
format. Missing in the team format and filled as stated: response time per attempt (default 5 s, so a
missing time never makes the model pick a harder picture), session minutes (default 0; rule G depends on
it), and the tier of attempts on other pictures and of "next" actions (looked up in `data/items.json`).
Not used by the model: `patient_id`, `session_id`, `state_version`, `attempt_id`, `transcript`,
`clinician_guidance`. Please send `response_time_s` and `session_minutes` when available.
