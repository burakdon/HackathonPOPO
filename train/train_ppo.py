"""Train MaskablePPO in the simulator, one run per seed, with held-out learning curves.

Training patients come from the broad population (train split); every reset draws a new patient.
Every `eval_freq` steps the deterministic policy is evaluated on fixed validation patients (val
split) and the proxy return, time in band, true control error and learning gain are recorded. The
test split is never touched here. TensorBoard logs go to runs/ppo/seed_<k>.

Usage:  uv run python -m train.train_ppo [--seeds 0 1 2] [--timesteps N] [--reward full]
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from core.config import load_yaml, resolve
from core.reward import load_reward
from envs.aphasia_env import AphasiaEnv


def evaluate_policy_on(model: MaskablePPO, env: AphasiaEnv, n_patients: int) -> dict[str, float]:
    """Deterministic policy on patients 0..n-1 of the env's cohort/split."""
    rows = []
    for k in range(n_patients):
        obs, _ = env.reset(options={"patient_index": k})
        done, ret = False, 0.0
        while not done:
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, r, done, _, info = env.step(int(a))
            ret += r
        trace = info["result"].trace
        rows.append({"return": ret, "time_in_band": info["time_in_band"], "learning_gain": info["learning_gain"],
                     "control_error": float(np.mean([abs(t["p_true"] - env.clinical.target) for t in trace])),
                     "mean_tier": float(np.mean([t["tier"] for t in trace]))})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


class ValidationCurve(BaseCallback):
    def __init__(self, env: AphasiaEnv, every: int, n_patients: int, path) -> None:
        super().__init__()
        self.env, self.every, self.n, self.path = env, every, n_patients, path
        self.next_eval = 0
        self.rows: list[dict] = []

    def _record(self) -> None:
        m = evaluate_policy_on(self.model, self.env, self.n)
        m["timesteps"] = self.num_timesteps
        self.rows.append(m)
        for k, v in m.items():
            if k != "timesteps":
                self.logger.record(f"val/{k}", v)
        print(f"  [{self.num_timesteps:>8}] val return {m['return']:6.2f}  in band {m['time_in_band']:.2f}  "
              f"|p-.85| {m['control_error']:.3f}  gain {100 * m['learning_gain']:.2f}  mean tier {m['mean_tier']:.2f}",
              flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self._record()
            self.next_eval += self.every
        return True

    def _on_training_end(self) -> None:
        self._record()
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)


def train_one(seed: int, cfg: dict, timesteps: int, reward_variant: str, tag: str) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    reward_cfg = load_reward(variant=reward_variant)
    venv = make_vec_env(lambda: AphasiaEnv(cohort="population", split="train", root_seed=seed, reward_cfg=reward_cfg),
                        n_envs=cfg["n_envs"], seed=seed, vec_env_cls=DummyVecEnv)
    out = resolve(cfg["output_dir"]) / f"{tag}seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    model = MaskablePPO(
        "MlpPolicy", venv, seed=seed, device="cpu", verbose=0, tensorboard_log=str(out),
        learning_rate=cfg["learning_rate"], n_steps=cfg["n_steps"], batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"], gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"], ent_coef=cfg["ent_coef"],
        policy_kwargs={"net_arch": list(cfg["net_arch"])},
    )
    val_env = AphasiaEnv(cohort="population", split="val", root_seed=0, reward_cfg=reward_cfg)
    curve = ValidationCurve(val_env, cfg["eval_freq"], cfg["eval_patients"], out / "val_curve.csv")
    t0 = time.time()
    print(f"seed {seed}: training {timesteps} steps (reward={reward_variant})", flush=True)
    model.learn(total_timesteps=timesteps, callback=curve, tb_log_name="tb")
    models = resolve(cfg["models_dir"])
    models.mkdir(parents=True, exist_ok=True)
    # Exclude the TensorBoard path: it is absolute (machine-specific) and the models are committed.
    model.save(str(models / f"ppo_{tag}seed{seed}.zip"), exclude=["tensorboard_log"])
    print(f"seed {seed}: done in {time.time() - t0:.0f}s -> {models / f'ppo_{tag}seed{seed}.zip'}", flush=True)


def main() -> None:
    cfg = load_yaml("configs/ppo.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg["seeds"])
    ap.add_argument("--timesteps", type=int, default=cfg["total_timesteps"])
    ap.add_argument("--reward", default=load_reward().variant, help="reward variant (ablation)")
    args = ap.parse_args()
    tag = "" if args.reward == load_reward().variant else f"{args.reward}_"
    for s in args.seeds:
        train_one(s, cfg, args.timesteps, args.reward, tag)


if __name__ == "__main__":
    main()
