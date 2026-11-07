"""10-step GRPO smoke test on Tinker: does the loop run, and does reward move?

Loop per step (Tinker cookbook pattern, group-relative advantages):
  1. get a sampling client for the current LoRA weights
  2. for each prompt, sample `group_size` completions (with per-token logprobs)
  3. reward each completion with the expected band-tracking reward (train/grpo_data.py)
  4. advantage = reward - mean reward of its group; skip groups with no spread
  5. one Datum per completion (prompt positions get advantage 0), forward_backward with the
     importance-sampling loss, then optim_step
Logs mean reward, format-valid rate and chosen-action stats per step to runs/grpo/smoke.csv.

Requires TINKER_API_KEY in .env (never committed). Without it the script stops before any API call.
Usage:  uv sync --extra grpo && uv run python -m train.grpo_smoke [--steps 10]
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from agents.llm_prompt import messages, parse_action
from core.config import load_clinical, load_yaml, resolve
from core.schema import State
from train.grpo_data import build_dataset, reward


def load_dotenv(path=".env") -> None:
    """Minimal .env reader (KEY=VALUE lines) so the key never has to be typed on the command line."""
    p = resolve(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    cfg = load_yaml("configs/grpo.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=cfg["smoke_steps"])
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set (add it to .env). Nothing was sent to Tinker.")

    import tinker
    from tinker import types

    clinical = load_clinical()
    data = build_dataset(cfg["dataset_size"], cfg["dataset_seed"], clinical=clinical)
    rng = np.random.default_rng(cfg["seed"])
    print(f"dataset: {len(data)} states from simulated training patients")

    service = tinker.ServiceClient()
    trainer = service.create_lora_training_client(base_model=cfg["base_model"], rank=cfg["lora_rank"], seed=cfg["seed"])
    tok = trainer.get_tokenizer()
    sp = types.SamplingParams(max_tokens=cfg["max_tokens"], temperature=cfg["temperature"], seed=cfg["seed"])
    adam = types.AdamParams(learning_rate=cfg["learning_rate"], beta1=0.9, beta2=0.95, eps=1e-8)

    def encode(example: dict) -> list[int]:
        state = State(**example["state"])
        text = tok.apply_chat_template(messages(state, np.array(example["mask"])), add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
        return tok.encode(text, add_special_tokens=False)

    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    for step in range(args.steps):
        t0 = time.time()
        sampler = trainer.save_weights_and_get_sampling_client()
        batch = [data[i] for i in rng.choice(len(data), cfg["prompts_per_step"], replace=False)]
        prompts = [encode(ex) for ex in batch]
        futures = [sampler.sample(prompt=types.ModelInput.from_ints(p), num_samples=cfg["group_size"],
                                  sampling_params=sp) for p in prompts]
        datums, rewards, valid = [], [], []
        for ex, p, fut in zip(batch, prompts, futures):
            seqs = fut.result().sequences
            texts = [tok.decode(s.tokens, skip_special_tokens=True) for s in seqs]
            rs = [reward(t, ex, clinical, cfg["cue_cost"], cfg["tier_bonus"], cfg["format_penalty"]) for t in texts]
            rewards += rs
            valid += [parse_action(t) is not None for t in texts]
            adv = np.array(rs) - np.mean(rs)
            if np.allclose(adv, 0.0):
                continue
            for s, a in zip(seqs, adv):
                full = p + list(s.tokens)
                n_prefix = len(p) - 1
                datums.append(types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=full[:-1]),
                    loss_fn_inputs={"target_tokens": full[1:],
                                    "logprobs": [0.0] * n_prefix + list(s.logprobs),
                                    "advantages": [0.0] * n_prefix + [float(a)] * len(s.tokens)}))
        if datums:
            fb = trainer.forward_backward(datums, loss_fn=cfg["loss_fn"])
            opt = trainer.optim_step(adam)
            fb.result()
            opt.result()
        row = {"step": step, "mean_reward": float(np.mean(rewards)), "valid_rate": float(np.mean(valid)),
               "n_datums": len(datums), "seconds": round(time.time() - t0, 1)}
        log_rows.append(row)
        print(row, flush=True)
    with open(out_dir / "smoke.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        w.writeheader()
        w.writerows(log_rows)
    print(f"saved {out_dir / 'smoke.csv'}")


if __name__ == "__main__":
    main()
