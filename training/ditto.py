"""DITTO (Shaikh et al., ICLR 2025) for practice decisions: adapt a small language model so its
exercise/hint decisions follow a demonstrator's preferences.

    uv run --extra llm python -m training.ditto                        # full run -> runs/ditto/<run_id>/
    uv run --extra llm python -m training.ditto --smoke                # tiny run on Qwen3-0.6B

Follows the authors' implementation (github.com/SALT-NLP/demonstrated-feedback, scripts/run_ditto.py,
scripts/dataset_utils.py):
  1. SFT: a LoRA adapter "sft" is trained on the demonstrations (answer tokens only), with an early
     stop on the loss.
  2. A second adapter "ditto" starts as a copy of "sft". The reference policy for DPO is the fixed
     "sft" adapter (the reference passes ref_adapter_name="sft").
  3. Each round t: sample `samples_per_prompt` completions per training context from the current
     "ditto" policy and store them in a cache keyed by round. Build comparisons:
       expert     demonstration > sample from round t
       replay     demonstration > sample from an earlier round (< t)
       intermodel sample from round a > sample from round b, for b < a
     and draw them in the reference proportions (0.7 / 0.2 / 0.1 of n_prompts * rescale_batch).
     Run DPO (sigmoid loss) for `steps_per_round` steps. Save the round's adapter.
  The intermodel comparisons rest on DITTO's assumption that each round's policy is closer to the
  demonstrator than earlier rounds' policies, so its outputs are ranked above theirs.

Differences from the reference (documented in the README):
  - Pairs are drawn once per round instead of once per batch inside the collator; rounds play the
    role of the reference's resample_rate steps.
  - Structured actions make exact duplicates common, so pairs whose two sides are the same action,
    or whose rejected side is an action the demonstrator accepts, are discarded instead of being
    trained on (the reference compares free text and does not need this filter).
  - The DPO loss is implemented here (same sigmoid objective as TRL's DPOTrainer) instead of
    through TRL, to keep the fixed SFT reference adapter and per-round pairs explicit.
  - Reference log-probabilities are computed once per round with the "sft" adapter (it is fixed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from agents.llm_runtime import TEMPLATE_SETTINGS, chat_prompt, pick_device
from core.config import load_clinical, resolve
from core.practice import DecisionContext, load_practice, messages, parse_action


# --------------------------------------------------------------------------------------------
# Data

def load_examples(path: str | Path, split: str | None = "train") -> list[dict]:
    rows = [json.loads(line) for line in resolve(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if split is None or r.get("split") == split]


def examples_hash(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()).hexdigest()[:16]


def example_messages(row: dict, history_in_prompt: int, few_shot: list[tuple[str, str]] | None = None,
                     fmt: str = "v1"):
    ctx = DecisionContext(**row["context"])
    opts = [parse_action(o) for o in row["options"]]
    return messages(ctx, opts, history_in_prompt, few_shot, fmt)


def equivalent(a: str, b: str) -> bool:
    """Same decision regardless of wording (both parse to the same action)."""
    pa, pb = parse_action(a), parse_action(b)
    return pa is not None and pb is not None and pa.key() == pb.key()


def is_accepted(text: str, accepted: list[str]) -> bool:
    p = parse_action(text)
    return p is not None and p.key() in accepted


def build_pairs(cache: dict[int, dict[str, list[str]]], rows: list[dict], t: int, cfg: dict,
                rng: random.Random) -> tuple[list[tuple[str, str, str, str]], dict]:
    """(example_id, kind, chosen_text, rejected_text) comparisons for round t, plus counts."""
    pools: dict[str, list] = {"expert": [], "replay": [], "intermodel": []}
    dropped = {"rejected_is_accepted": 0, "equivalent_pair": 0}
    for r in rows:
        eid, demo, acc = r["example_id"], r["preferred"], r["accepted"]
        for step in range(t, -1, -1):
            for s in cache.get(step, {}).get(eid, []):
                kind = "expert" if step == t else "replay"
                if is_accepted(s, acc):
                    dropped["rejected_is_accepted"] += 1
                    continue
                pools[kind].append((eid, kind, demo, s))
            for older in range(step - 1, -1, -1):
                for a in cache.get(step, {}).get(eid, []):
                    for b in cache.get(older, {}).get(eid, []):
                        if a == b or equivalent(a, b):
                            dropped["equivalent_pair"] += 1
                            continue
                        if is_accepted(b, acc):        # never rank an accepted action as inferior
                            dropped["rejected_is_accepted"] += 1
                            continue
                        pools["intermodel"].append((eid, "intermodel", a, b))
    n_super = len(rows) * cfg["rescale_batch"]
    fracs = {"expert": cfg["frac_expert"], "replay": cfg["frac_replay"], "intermodel": cfg["frac_intermodel"]}
    chosen = []
    for kind, frac in fracs.items():
        chosen += rng.sample(pools[kind], min(len(pools[kind]), round(n_super * frac)))
    rng.shuffle(chosen)
    stats = {"available": {k: len(v) for k, v in pools.items()},
             "used": {k: sum(1 for p in chosen if p[1] == k) for k in pools},
             "dropped": dropped, "target_total": n_super}
    return chosen, stats


# --------------------------------------------------------------------------------------------
# Model helpers (torch imported lazily so this module can be imported for tests without the extra)

def copy_adapter_weights(model, src: str, dst: str) -> None:
    """As in the reference: initialise the 'ditto' adapter from the 'sft' adapter."""
    for module in model.modules():
        if hasattr(module, "lora_A") and src in module.lora_A and dst in module.lora_A:
            module.lora_A[dst].load_state_dict(module.lora_A[src].state_dict())
            module.lora_B[dst].load_state_dict(module.lora_B[src].state_dict())


def save_adapter(model, name: str, out: Path) -> Path:
    import shutil
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp), selected_adapters=[name])
    src = tmp / name if (tmp / name).exists() else tmp      # PEFT puts non-default adapters in a subfolder
    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(src), str(out))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


class Encoder:
    def __init__(self, tokenizer, device: str) -> None:
        self.tok, self.device = tokenizer, device
        self.end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def prompt_ids(self, msgs) -> list[int]:
        return self.tok(chat_prompt(self.tok, msgs), add_special_tokens=False)["input_ids"]

    def answer_ids(self, text: str) -> list[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return ids if ids and ids[-1] == self.end else ids + [self.end]

    def batch(self, items: list[tuple[list[int], list[int]]]):
        """Left-padded input_ids, attention mask, position ids, a mask over answer tokens, and K: the
        answer tokens all sit in the last K positions, so only K positions' logits are ever computed
        (full-vocabulary logits for every prompt token would need gigabytes)."""
        import torch
        n = max(len(p) + len(a) for p, a in items)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.end
        ids = torch.full((len(items), n), pad, dtype=torch.long)
        attn = torch.zeros((len(items), n), dtype=torch.long)
        amask = torch.zeros((len(items), n), dtype=torch.bool)
        for i, (p, a) in enumerate(items):
            seq = p + a
            ids[i, n - len(seq):] = torch.tensor(seq)
            attn[i, n - len(seq):] = 1
            amask[i, n - len(a):] = True
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        k = max(len(a) for _, a in items) + 1
        return ids.to(self.device), attn.to(self.device), pos.to(self.device), amask.to(self.device), k


def answer_token_logps(model, enc: Encoder, items):
    """Per-token log-probabilities of the answer tokens (B x K-1) and their mask."""
    import torch
    ids, attn, pos, amask, k = enc.batch(items)
    logits = model(input_ids=ids, attention_mask=attn, position_ids=pos, logits_to_keep=k).logits[:, :-1].float()
    targets = ids[:, -k + 1:]
    lp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return lp, amask[:, -k + 1:]


def answer_logps(model, enc: Encoder, items) -> "torch.Tensor":
    """Summed log-probability of each answer given its prompt."""
    lp, m = answer_token_logps(model, enc, items)
    return (lp * m).sum(-1)


def sample_options(model, enc: Encoder, prompt: list[int], options: list[str], n: int, temperature: float,
                   gen) -> list[str]:
    """n alternatives drawn from the policy's distribution over the valid options (softmax of their
    summed log-probabilities / temperature), for decoding = "score": the alternatives are then always
    valid actions, matching how that policy decides at inference."""
    import torch
    with torch.inference_mode():
        lp = answer_logps(model, enc, [(prompt, enc.answer_ids(o)) for o in options]).float().cpu()
    probs = torch.softmax(lp / temperature, dim=0)
    return [options[i] for i in torch.multinomial(probs, n, replacement=True, generator=gen).tolist()]


def sample(model, enc: Encoder, prompt: list[int], n: int, cfg: dict, max_new: int) -> list[str]:
    import torch
    ids = torch.tensor([prompt], device=enc.device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                             temperature=cfg["temperature"], top_k=cfg["top_k"], top_p=1.0,
                             num_return_sequences=n, max_new_tokens=max_new,
                             pad_token_id=enc.end, eos_token_id=enc.end)
    return [enc.tok.decode(o[len(prompt):], skip_special_tokens=True).strip() for o in out]


def option_agreement(model, enc: Encoder, rows: list[dict], prompts: dict[str, list[int]]) -> int:
    """How many rows' most likely valid option (summed log-probability, as decoding = "score") is
    among the row's accepted actions. Used to evaluate the dev split after every stage."""
    import torch
    hits = 0
    with torch.inference_mode():
        for r in rows:
            lp = answer_logps(model, enc, [(prompts[r["example_id"]], enc.answer_ids(o)) for o in r["options"]])
            hits += r["options"][int(lp.argmax())] in r["accepted"]
    return hits


def select_stage(agreement: dict[str, int]) -> str:
    """Best stage by dev agreement; ties go to the earlier stage ("sft" before "round1" ...), so DITTO
    rounds are kept only when they improve on dev."""
    order = list(agreement)
    return max(order, key=lambda k: (agreement[k], -order.index(k)))


# --------------------------------------------------------------------------------------------
# Training run

def run(cfg_path: str, run_id: str | None, smoke: bool, extra_demos: str | None, prepare_rows=None) -> Path:
    """`prepare_rows(model, enc, fmt, rows, history_in_prompt) -> (rows, info)`, if given, is called once the
    model (and init_adapter, as the active "sft" adapter) is loaded and returns the rows to train on; `info` is
    stored in the manifest. Per-patient training uses it to add anchors computed with the loaded model."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = yaml.safe_load(resolve(cfg_path).read_text())
    pcfg = load_practice(cfg["practice_config"])
    clinical = load_clinical()
    base, revision = pcfg.llm.base_model, pcfg.llm.revision
    if smoke:
        base, revision = "Qwen/Qwen3-0.6B", None
        cfg["sft"]["max_steps"], cfg["ditto"]["rounds"], cfg["ditto"]["steps_per_round"] = 4, 2, 3
        cfg["ditto"]["samples_per_prompt"] = 3
    seed = cfg["seed"]
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    rng = random.Random(seed)

    rows = load_examples(cfg["demonstrations"], "train")
    if extra_demos:      # captured clinician examples (global scope only; other scopes are not mixed in)
        extra = [r for r in load_examples(extra_demos, None) if r.get("scope") == "global"]
        rows += [{**r, "split": "train"} for r in extra]
    if smoke:
        rows = rows[:6]
    if any(r["split"] != "train" for r in rows):
        raise ValueError("only training demonstrations may be used for training")

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + ("-smoke" if smoke else "")
    out = resolve(pcfg.llm.runs_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id, "method": "DITTO (SFT + iterative DPO with expert/replay/intermodel comparisons)",
        "status": "running", "started": datetime.now(timezone.utc).isoformat(), "smoke": smoke,
        "base_model": base, "base_revision": revision, "template": TEMPLATE_SETTINGS,
        "dtype": pcfg.llm.dtype, "decoding": pcfg.llm.decoding, "init_adapter": cfg.get("init_adapter"),
        "prompt_format": pcfg.llm.prompt_format,
        "history_in_prompt": pcfg.history_in_prompt,
        "demonstrations": {"path": cfg["demonstrations"], "extra_path": extra_demos,
                           "example_ids": [r["example_id"] for r in rows], "hash": examples_hash(rows),
                           "n": len(rows)},
        "config": cfg, "seed": seed, "device": pick_device(pcfg.llm.device), "rounds": [],
        "constraints": {"approved_cues": clinical.approved_cues, "max_tier_step": clinical.max_tier_step,
                        "max_attempts_per_exercise": pcfg.max_attempts_per_exercise},
    }
    mpath = out / "manifest.json"

    def write_manifest():
        mpath.write_text(json.dumps(manifest, indent=1))

    write_manifest()
    try:
        t_start = time.time()
        dev = manifest["device"]
        tok = AutoTokenizer.from_pretrained(base, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(base, revision=revision,
                                                     dtype=getattr(torch, pcfg.llm.dtype)).to(dev)
        model.config.use_cache = False
        enc = Encoder(tok, dev)
        lcfg = cfg["lora"]
        lora = LoraConfig(r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
                          target_modules=lcfg["target_modules"], task_type="CAUSAL_LM")
        if cfg.get("init_adapter"):
            # Per-patient run: start from the shared adapter (same LoRA shape), which is also the
            # starting point of the brief SFT stage and hence of the DPO reference.
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(resolve(cfg["init_adapter"])), adapter_name="sft",
                                              is_trainable=True)
        else:
            model = get_peft_model(model, lora, adapter_name="sft")
        fmt = pcfg.llm.prompt_format
        if prepare_rows is not None:
            model.set_adapter("sft")
            rows, info = prepare_rows(model, enc, fmt, rows, pcfg.history_in_prompt)
            model.train()
            manifest["demonstrations"].update({"example_ids": [r["example_id"] for r in rows],
                                               "hash": examples_hash(rows), "n": len(rows)})
            manifest["prepared_rows"] = info
            write_manifest()
        prompts = {r["example_id"]: enc.prompt_ids(example_messages(r, pcfg.history_in_prompt, fmt=fmt)) for r in rows}
        demo_ids = {r["example_id"]: enc.answer_ids(r["preferred"]) for r in rows}
        manifest["max_prompt_tokens"] = max(len(p) for p in prompts.values())

        # ---- Stage 1: SFT
        s = cfg["sft"]
        model.set_adapter("sft")
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=s["learning_rate"])
        sft_log = []
        for step in range(s["max_steps"]):
            batch = rng.sample(rows, min(s["batch_size"], len(rows)))
            items = [(prompts[r["example_id"]], demo_ids[r["example_id"]]) for r in batch]
            lp, m = answer_token_logps(model, enc, items)
            loss = -(lp * m).sum() / m.sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            sft_log.append(round(loss.item(), 4))
            if s.get("stop_loss") and loss.item() < s["stop_loss"]:       # 0 or null: fixed number of steps
                break
        save_adapter(model, "sft", out / "sft")
        manifest["sft"] = {"steps": len(sft_log), "loss": sft_log, "adapter": "sft"}
        write_manifest()
        print(f"SFT: {len(sft_log)} steps, last loss {sft_log[-1]}")

        # Optional selection on a held-out split (cfg["select_on"], e.g. "dev"): agreement after SFT
        # (stage "sft") and after every DITTO round; the best stage becomes the run's final adapter.
        # select_on: "train" selects on the training rows themselves (a per-patient run has no held-out split).
        sel_rows = (rows if cfg.get("select_on") == "train"
                    else load_examples(cfg["demonstrations"], cfg["select_on"]) if cfg.get("select_on") else [])
        if cfg.get("select_on") != "train" and {r["example_id"] for r in sel_rows} & {r["example_id"] for r in rows}:
            raise ValueError("selection rows overlap the training rows")
        sel_prompts = {r["example_id"]: enc.prompt_ids(example_messages(r, pcfg.history_in_prompt, fmt=fmt))
                       for r in sel_rows}
        agreement: dict[str, int] = {}

        def evaluate_stage(name: str) -> None:
            if not sel_rows:
                return
            model.eval()
            agreement[name] = option_agreement(model, enc, sel_rows, sel_prompts)
            model.train()
            print(f"{cfg['select_on']} agreement after {name}: {agreement[name]}/{len(sel_rows)}", flush=True)

        evaluate_stage("sft")

        # ---- Stage 2: DITTO rounds
        d = cfg["ditto"]
        model.add_adapter("ditto", lora)
        copy_adapter_weights(model, "sft", "ditto")
        cache: dict[int, dict[str, list[str]]] = {}
        options = {r["example_id"]: r["options"] for r in rows}
        gen = torch.Generator().manual_seed(seed)
        for t in range(d["rounds"]):
            model.set_adapter("ditto")
            model.eval()
            model.config.use_cache = True
            if pcfg.llm.decoding == "score":
                cache[t] = {eid: sample_options(model, enc, prompts[eid], options[eid], d["samples_per_prompt"],
                                                d["temperature"], gen) for eid in prompts}
            else:
                cache[t] = {eid: sample(model, enc, prompts[eid], d["samples_per_prompt"], d, pcfg.llm.max_new_tokens)
                            for eid in prompts}
            model.config.use_cache = False
            acc = {r["example_id"]: r["accepted"] for r in rows}
            n_samples = sum(len(v) for v in cache[t].values())
            n_match = sum(is_accepted(x, acc[e]) for e, v in cache[t].items() for x in v)
            n_invalid = sum(parse_action(x) is None for v in cache[t].values() for x in v)
            (out / f"samples_round{t + 1}.json").write_text(json.dumps(cache[t], indent=1))
            pairs, pstats = build_pairs(cache, rows, t, d, rng)
            (out / f"pairs_round{t + 1}.jsonl").write_text(
                "".join(json.dumps({"example_id": e, "kind": k, "chosen": c, "rejected": r}) + "\n"
                        for e, k, c, r in pairs))
            rinfo = {"round": t + 1, "samples": n_samples, "samples_matching_demonstrator": n_match,
                     "samples_unparseable": n_invalid, "pairs": pstats, "steps": 0, "loss": [],
                     "reward_accuracy": []}
            print(f"round {t + 1}: {n_match}/{n_samples} samples already match the demonstrator, "
                  f"{n_invalid} unparseable; pairs used {pstats['used']} of available {pstats['available']}")
            if pairs:
                # Reference log-probs with the fixed SFT adapter, once per round.
                model.set_adapter("sft")
                ref = []
                with torch.no_grad():
                    for i in range(0, len(pairs), d["batch_size"]):
                        chunk = pairs[i:i + d["batch_size"]]
                        items = ([(prompts[e], enc.answer_ids(c)) for e, _, c, _ in chunk]
                                 + [(prompts[e], enc.answer_ids(r)) for e, _, _, r in chunk])
                        lp = answer_logps(model, enc, items)
                        ref += list(zip(lp[:len(chunk)].tolist(), lp[len(chunk):].tolist()))
                model.set_adapter("ditto")
                for n, p in model.named_parameters():
                    p.requires_grad = ".ditto." in n
                model.train()
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=d["learning_rate"])
                order, ptr = list(range(len(pairs))), 0
                for step in range(d["steps_per_round"]):
                    idx = []
                    for _ in range(min(d["batch_size"], len(order))):
                        if ptr >= len(order):
                            rng.shuffle(order)
                            ptr = 0
                        idx.append(order[ptr])
                        ptr += 1
                    chunk = [pairs[i] for i in idx]
                    items = ([(prompts[e], enc.answer_ids(c)) for e, _, c, _ in chunk]
                             + [(prompts[e], enc.answer_ids(r)) for e, _, _, r in chunk])
                    lp_tok, lp_mask = answer_token_logps(model, enc, items)
                    lp = (lp_tok * lp_mask).sum(-1)
                    pc, pr = lp[:len(chunk)], lp[len(chunk):]
                    rc = torch.tensor([ref[i][0] for i in idx], device=dev)
                    rr = torch.tensor([ref[i][1] for i in idx], device=dev)
                    margin = d["beta"] * ((pc - rc) - (pr - rr))
                    loss = -torch.nn.functional.logsigmoid(margin).mean()
                    if d.get("rpo_alpha"):
                        # RPO-style term: per-token NLL of the demonstrated action (expert and replay
                        # pairs, whose chosen side is the demonstration), so DPO cannot drift away
                        # from what SFT learned.
                        demo = torch.tensor([k != "intermodel" for _, k, _, _ in chunk], device=dev)
                        if demo.any():
                            nll = -(pc / lp_mask[:len(chunk)].sum(-1))[demo].mean()
                            loss = loss + d["rpo_alpha"] * nll
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    rinfo["loss"].append(round(loss.item(), 4))
                    rinfo["reward_accuracy"].append(round((margin > 0).float().mean().item(), 3))
                    rinfo["steps"] += 1
            save_adapter(model, "ditto", out / f"round{t + 1}")
            if dev == "mps":
                torch.mps.empty_cache()
            rinfo["adapter"] = f"round{t + 1}"
            model.set_adapter("ditto")
            evaluate_stage(f"round{t + 1}")
            if agreement:
                rinfo[f"{cfg['select_on']}_agreement"] = agreement[f"round{t + 1}"]
            manifest["rounds"].append(rinfo)
            write_manifest()
        manifest["final_adapter"] = f"round{d['rounds']}"
        if agreement:
            chosen = select_stage(agreement)
            manifest["selection"] = {"on": cfg["select_on"], "n": len(sel_rows), "agreement": agreement,
                                     "selected": chosen, "last_round": f"round{d['rounds']}",
                                     "rule": "best dev agreement; ties go to the earlier stage"}
            manifest["final_adapter"] = chosen
            print(f"selected {chosen} on {cfg['select_on']}: {agreement}", flush=True)
        manifest["status"] = "completed"
        manifest["seconds"] = round(time.time() - t_start, 1)
        if dev == "mps":
            manifest["peak_mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    except Exception as e:      # keep the manifest honest about failures
        manifest["status"] = "failed"
        manifest["error"] = f"{type(e).__name__}: {e}"
        manifest["traceback"] = traceback.format_exc()[-4000:]
        write_manifest()
        raise
    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    write_manifest()
    print(f"run {run_id} completed in {manifest['seconds']} s -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/ditto_llm.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run on Qwen3-0.6B to test the pipeline")
    ap.add_argument("--extra-demos", default=None,
                    help="captured demonstrations (JSONL, same format) to add to training; global scope only")
    args = ap.parse_args()
    run(args.config, args.run_id, args.smoke, args.extra_demos)


if __name__ == "__main__":
    main()
