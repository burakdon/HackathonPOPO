"""DITTO-style preference optimisation of a MaskablePPO policy from a few clinician overrides.

Method (after Shaikh et al., ICLR 2025, "Show, Don't Tell: Aligning Language Models with
Demonstrated Feedback"), applied to a discrete-action policy instead of a language model:
  - demonstrations D = {(state, clinician action)};
  - each round, sample actions from the current policy in every demonstration state; every
    demonstration is preferred over every sampled action that differs from it, including samples
    kept from earlier rounds (comparisons across policy versions);
  - minimise the DPO loss on the policy's masked action log-probabilities against a frozen
    reference policy (the original PPO policy), which keeps the update anchored with few examples:
        -log sigmoid(beta * [(log pi(a_w|s) - log ref(a_w|s)) - (log pi(a_l|s) - log ref(a_l|s))])
Only the actor (policy MLP + action head) is updated; the value head is untouched.

Accepted-choice anchors (our addition, optional): with a small policy network and ~10
demonstrations, the DPO update alone also changes the policy in states the clinician did not
correct (it overgeneralises; see README). On the dashboard, every agent choice the clinician saw and
did NOT override is an implicit acceptance, so we add anchor_weight * KL(ref || pi) on those states.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sb3_contrib import MaskablePPO


@dataclass
class Demo:
    obs: np.ndarray          # observation vector (core/observation.py)
    mask: np.ndarray         # allowed actions in that state
    action: int              # clinician's action (must be allowed)


def _log_probs(policy, obs: torch.Tensor, masks: np.ndarray) -> torch.Tensor:
    """(n, 15) masked log-probabilities of every action."""
    dist = policy.get_distribution(obs, action_masks=masks)
    return dist.distribution.logits.log_softmax(dim=-1)


def action_probs(model: MaskablePPO, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        lp = _log_probs(model.policy, torch.as_tensor(obs, dtype=torch.float32), masks)
    return lp.exp().numpy()


def ditto(model: MaskablePPO, demos: list[Demo], rounds: int, samples_per_state: int, beta: float,
          lr: float, steps_per_round: int, seed: int, anchors: list[tuple[np.ndarray, np.ndarray]] | None = None,
          anchor_weight: float = 0.0) -> tuple[MaskablePPO, list[dict]]:
    """Return a DITTO-tuned copy of `model` (the input model is the frozen reference) and a log."""
    for d in demos:
        if not d.mask[d.action]:
            raise ValueError("a demonstration action is masked in its own state")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ref = model.policy
    tuned = copy.deepcopy(model)
    pol = tuned.policy
    params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    obs = torch.as_tensor(np.stack([d.obs for d in demos]), dtype=torch.float32)
    masks = np.stack([d.mask for d in demos])
    winners = torch.as_tensor([d.action for d in demos])
    with torch.no_grad():
        ref_lp = _log_probs(ref, obs, masks)
    use_anchors = bool(anchors) and anchor_weight > 0
    if use_anchors:
        a_obs = torch.as_tensor(np.stack([o for o, _ in anchors]), dtype=torch.float32)
        a_masks = np.stack([m for _, m in anchors])
        with torch.no_grad():
            a_ref = _log_probs(ref, a_obs, a_masks)
            a_ref_p = a_ref.exp()
    rejected: list[list[int]] = [[] for _ in demos]       # accumulates samples across rounds
    log = []
    for r in range(rounds):
        with torch.no_grad():
            probs = _log_probs(pol, obs, masks).exp().numpy()
        for i, d in enumerate(demos):
            p = probs[i] / probs[i].sum()
            for a in rng.choice(len(p), size=samples_per_state, p=p):
                if int(a) != d.action:
                    rejected[i].append(int(a))
        pairs = [(i, a) for i, rs in enumerate(rejected) for a in rs]
        if not pairs:
            log.append({"round": r, "pairs": 0, "loss": 0.0,
                        "demo_prob": float(probs[np.arange(len(demos)), winners.numpy()].mean())})
            continue
        idx = torch.as_tensor([i for i, _ in pairs])
        losers = torch.as_tensor([a for _, a in pairs])
        for _ in range(steps_per_round):
            lp = _log_probs(pol, obs, masks)
            w_ratio = (lp - ref_lp)[idx, winners[idx]]
            l_ratio = (lp - ref_lp)[idx, losers]
            loss = -torch.nn.functional.logsigmoid(beta * (w_ratio - l_ratio)).mean()
            if use_anchors:
                a_lp = _log_probs(pol, a_obs, a_masks)
                finite = torch.isfinite(a_ref)                     # masked actions have -inf log-prob
                kl = torch.where(finite, a_ref_p * (a_ref - a_lp), torch.zeros_like(a_ref)).sum(-1).mean()
                loss = loss + anchor_weight * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            demo_prob = _log_probs(pol, obs, masks).exp()[torch.arange(len(demos)), winners].mean()
        log.append({"round": r, "pairs": len(pairs), "loss": loss.item(), "demo_prob": demo_prob.item()})
    return tuned, log
