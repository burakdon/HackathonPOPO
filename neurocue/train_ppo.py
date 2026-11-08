"""Train MaskablePPO on the engine therapy environment.

  uv run python -m neurocue.train_ppo [--seeds 0] [--timesteps 160000] [--reward privileged|observable]

Each reset draws a new training-split patient. Returns are undiscounted (gamma 1): the privileged
reward telescopes to the course's retention gain, so the return is the objective itself. A validation
curve on fixed val-split patients is written next to the model.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from neurocue.env import EngineTherapyEnv, load_engine_env_config

ROOT = Path(__file__).resolve().parents[1]


def evaluate(model: MaskablePPO, env: EngineTherapyEnv, n: int) -> dict[str, float]:
    gains, cued = [], []
    for k in range(n):
        obs, _ = env.reset(options={"patient_index": k})
        done = False
        while not done:
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _, done, _, info = env.step(int(a))
        gains.append(info["gain"][env.cfg.test_days[0]])
        cued.append(np.mean([h["cue"] is not None for h in info["history"]]))
    return {"gain_test1": float(np.mean(gains)), "cued_share": float(np.mean(cued))}


class ValCurve(BaseCallback):
    def __init__(self, env, every, n, path):
        super().__init__(); self.env, self.every, self.n, self.path = env, every, n, path
        self.next_eval, self.rows = 0, []

    def _record(self):
        m = evaluate(self.model, self.env, self.n); m["timesteps"] = self.num_timesteps
        self.rows.append(m)
        print(f"  [{self.num_timesteps:>8}] val gain day {int(self.env.cfg.test_days[0])}: {m['gain_test1']:+.3f}  cued {m['cued_share']:.2f}", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self._record(); self.next_eval += self.every
        return True

    def _on_training_end(self) -> None:
        self._record()
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0])); w.writeheader(); w.writerows(self.rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--timesteps", type=int, default=160_000)
    ap.add_argument("--reward", default="privileged")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--eval-patients", type=int, default=12)
    args = ap.parse_args()
    cfg = load_engine_env_config(); cfg.reward = args.reward
    out_dir = ROOT / "models"; out_dir.mkdir(exist_ok=True)
    runs = ROOT / "runs" / "engine_ppo"; runs.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        torch.set_num_threads(1); torch.manual_seed(seed)
        venv = make_vec_env(lambda: EngineTherapyEnv(cfg, split="train", root_seed=seed), n_envs=args.n_envs, seed=seed,
                            vec_env_cls=DummyVecEnv)
        model = MaskablePPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, learning_rate=3e-4, n_steps=320,
                            batch_size=640, n_epochs=8, gamma=1.0, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                            policy_kwargs={"net_arch": [128, 128]})
        val_env = EngineTherapyEnv(cfg, split="val", root_seed=0)
        tag = f"engine_ppo_{args.reward}_seed{seed}"
        cb = ValCurve(val_env, args.eval_every, args.eval_patients, runs / f"{tag}_val.csv")
        t0 = time.time(); print(f"seed {seed}: {args.timesteps} steps, reward={args.reward}", flush=True)
        model.learn(total_timesteps=args.timesteps, callback=cb)
        model.save(str(out_dir / f"{tag}.zip"))
        print(f"seed {seed}: done in {time.time() - t0:.0f}s -> models/{tag}.zip", flush=True)


if __name__ == "__main__":
    main()
