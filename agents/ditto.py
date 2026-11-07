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

Two variants are implemented:
  - legacy (mixture=None, sft_steps=0, the default): every comparison "demonstration > sampled
    action" accumulated so far is used uniformly, and the DPO reference is the original policy.
  - paper (mixture={...}): Algorithm 1 of Shaikh et al. more closely.
      1. Initialise with a few SFT (behaviour-cloning) steps on the demonstrations; the frozen DPO
         reference is this SFT policy pi_0 (the paper's choice), not the original policy.
      2. Each round t, sample M actions per demonstration state from pi_t. The induced ranking is
         demos > D_t > D_{t-1} > ... > D_0 (later policies are assumed better than earlier ones).
      3. Each gradient step draws a batch mixing three comparison types (paper: 70/20/10):
           online      demo > sample of the current policy    (D_E > D_t)
           replay      demo > sample of an earlier policy     (D_E > D_{i<t})
           intermodel  later sample > earlier sample          (D_{i<=t} > D_{j<i})
         Intermodel pairs never use the demonstrated action as the loser (in a discrete action
         space a policy can sample exactly the demonstration; pushing it down would be wrong).
    Anchors, if used, stay anchored to the ORIGINAL policy (what the clinician accepted), not to the
    SFT policy.
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


def _anchor_kl(pol, a_obs: torch.Tensor, a_masks: np.ndarray, a_ref: torch.Tensor,
               a_ref_p: torch.Tensor) -> torch.Tensor:
    """Mean KL(ref || pi) over anchor states; masked actions (log-prob -inf) contribute 0."""
    a_lp = _log_probs(pol, a_obs, a_masks)
    finite = torch.isfinite(a_ref)
    return torch.where(finite, a_ref_p * (a_ref - a_lp), torch.zeros_like(a_ref)).sum(-1).mean()


def _actor_params(pol) -> list:
    return list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())


def _prepare_anchors(ref, anchors):
    a_obs = torch.as_tensor(np.stack([o for o, _ in anchors]), dtype=torch.float32)
    a_masks = np.stack([m for _, m in anchors])
    with torch.no_grad():
        a_ref = _log_probs(ref, a_obs, a_masks)
    return a_obs, a_masks, a_ref, a_ref.exp()


def behaviour_cloning(model: MaskablePPO, demos: list["Demo"], steps: int, lr: float, seed: int,
                      anchors: list[tuple[np.ndarray, np.ndarray]] | None = None,
                      anchor_weight: float = 0.0) -> MaskablePPO:
    """SFT baseline (the paper's main baseline): maximise log pi(demo action | state) for `steps`
    gradient steps, optionally with the same accepted-choice anchors as DITTO."""
    torch.manual_seed(seed)
    tuned = copy.deepcopy(model)
    pol = tuned.policy
    opt = torch.optim.Adam(_actor_params(pol), lr=lr)
    obs = torch.as_tensor(np.stack([d.obs for d in demos]), dtype=torch.float32)
    masks = np.stack([d.mask for d in demos])
    winners = torch.as_tensor([d.action for d in demos])
    use_anchors = bool(anchors) and anchor_weight > 0
    if use_anchors:
        anc = _prepare_anchors(model.policy, anchors)
    for _ in range(steps):
        loss = -_log_probs(pol, obs, masks)[torch.arange(len(demos)), winners].mean()
        if use_anchors:
            loss = loss + anchor_weight * _anchor_kl(pol, *anc)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return tuned


def action_probs(model: MaskablePPO, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        lp = _log_probs(model.policy, torch.as_tensor(obs, dtype=torch.float32), masks)
    return lp.exp().numpy()


def ditto(model: MaskablePPO, demos: list[Demo], rounds: int, samples_per_state: int, beta: float,
          lr: float, steps_per_round: int, seed: int, anchors: list[tuple[np.ndarray, np.ndarray]] | None = None,
          anchor_weight: float = 0.0, sft_steps: int = 0, mixture: dict[str, float] | None = None,
          batch_size: int = 64) -> tuple[MaskablePPO, list[dict]]:
    """Return a DITTO-tuned copy of `model` (the input model is never modified) and a log.

    mixture=None (default) runs the legacy variant; a dict with keys online/replay/intermodel runs
    the paper variant (see the module docstring)."""
    for d in demos:
        if not d.mask[d.action]:
            raise ValueError("a demonstration action is masked in its own state")
    if not demos:
        raise ValueError("DITTO needs at least one demonstration")
    if mixture is not None or sft_steps > 0:
        return _ditto_paper(model, demos, rounds, samples_per_state, beta, lr, steps_per_round, seed,
                            anchors, anchor_weight, sft_steps, mixture or {"online": 1.0}, batch_size)
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


def _ditto_paper(model, demos, rounds, samples_per_state, beta, lr, steps_per_round, seed, anchors,
                 anchor_weight, sft_steps, mixture, batch_size):
    unknown = set(mixture) - {"online", "replay", "intermodel"}
    if unknown or any(v < 0 for v in mixture.values()) or sum(mixture.values()) <= 0:
        raise ValueError(f"bad mixture {mixture}")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    use_anchors = bool(anchors) and anchor_weight > 0
    anc = _prepare_anchors(model.policy, anchors) if use_anchors else None   # anchored to the ORIGINAL policy

    # 1. SFT initialisation; the frozen DPO reference is pi_0 = the SFT policy.
    tuned = behaviour_cloning(model, demos, sft_steps, lr, seed, anchors, anchor_weight) if sft_steps > 0 \
        else copy.deepcopy(model)
    pol = tuned.policy
    # Copy via state_dict: a policy that just took gradient steps can hold non-leaf tensors
    # (the cached action distribution), which torch refuses to deepcopy.
    ref = copy.deepcopy(model.policy)
    ref.load_state_dict(pol.state_dict())
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.Adam(_actor_params(pol), lr=lr)

    n = len(demos)
    obs = torch.as_tensor(np.stack([d.obs for d in demos]), dtype=torch.float32)
    masks = np.stack([d.mask for d in demos])
    demo_a = np.array([d.action for d in demos])
    with torch.no_grad():
        ref_lp = _log_probs(ref, obs, masks)
    samples: list[list[np.ndarray]] = []       # samples[t][i] = actions sampled from pi_t in demo state i
    log = []
    for t in range(rounds):
        with torch.no_grad():
            probs = _log_probs(pol, obs, masks).exp().numpy()
        samples.append([rng.choice(probs.shape[1], size=samples_per_state, p=probs[i] / probs[i].sum())
                        for i in range(n)])
        pools = {"online": [(i, demo_a[i], a) for i in range(n) for a in samples[t][i] if a != demo_a[i]],
                 "replay": [(i, demo_a[i], a) for j in range(t) for i in range(n) for a in samples[j][i]
                            if a != demo_a[i]],
                 "intermodel": [(i, aw, al) for hi in range(t + 1) for lo in range(hi) for i in range(n)
                                for aw in samples[hi][i] for al in samples[lo][i] if aw != al and al != demo_a[i]]}
        weights = {k: mixture.get(k, 0.0) if pools[k] else 0.0 for k in pools}
        total = sum(weights.values())
        entry = {"round": t, **{f"pairs_{k}": len(v) for k, v in pools.items()}}
        if total == 0:                      # the policy already only samples the demonstrations
            entry.update(loss=0.0, demo_prob=float(probs[np.arange(n), demo_a].mean()))
            log.append(entry)
            continue
        counts = {k: int(round(batch_size * w / total)) for k, w in weights.items()}
        for _ in range(steps_per_round):
            batch = [pools[k][j] for k, c in counts.items() if c for j in rng.integers(len(pools[k]), size=c)]
            idx = torch.as_tensor([b[0] for b in batch])
            win = torch.as_tensor([int(b[1]) for b in batch])
            los = torch.as_tensor([int(b[2]) for b in batch])
            ratio = _log_probs(pol, obs, masks) - ref_lp
            loss = -torch.nn.functional.logsigmoid(beta * (ratio[idx, win] - ratio[idx, los])).mean()
            if use_anchors:
                loss = loss + anchor_weight * _anchor_kl(pol, *anc)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            demo_prob = _log_probs(pol, obs, masks).exp()[torch.arange(n), torch.as_tensor(demo_a)].mean()
        entry.update(loss=loss.item(), demo_prob=demo_prob.item(), batch=counts)
        log.append(entry)
    return tuned, log
