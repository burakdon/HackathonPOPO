"""Infrastructure smoke test: load a base model, generate, train a LoRA adapter for a few steps,
save it, and reload it in a fresh process. Proves the GPU stack before a real run.

    uv run --extra llm python -m training.smoke --model Qwen/Qwen3-0.6B
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time

import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--reload", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    msgs = [{"role": "user", "content": 'Reply with {"op":"next","tier":2} only.'}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
    print(f"device={dev} load_s={time.time() - t0:.1f}")
    if args.reload:
        model = PeftModel.from_pretrained(model, args.reload)
    ids = tok(prompt, return_tensors="pt").to(dev)
    t0 = time.time()
    out = model.generate(**ids, max_new_tokens=16, do_sample=False)
    text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"generate_s={time.time() - t0:.2f} text={text!r}")
    if args.reload:
        print("RELOAD_OK")
        return
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    batch = tok([prompt + '{"op":"next","tier":2}<|im_end|>'] * 4, return_tensors="pt").to(dev)
    t0 = time.time()
    for s in range(args.steps):
        loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"step {s} loss={loss.item():.3f}")
    if dev == "mps":
        print(f"mps_allocated_gb={torch.mps.driver_allocated_memory() / 1e9:.2f}")
    print(f"train_s_per_step={(time.time() - t0) / args.steps:.2f}")
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        print(json.dumps(sorted(__import__("os").listdir(d))))
        r = subprocess.run([sys.executable, "-m", "training.smoke", "--model", args.model, "--reload", d],
                           capture_output=True, text=True)
        print(r.stdout[-400:], r.stderr[-400:] if r.returncode else "")
        assert "RELOAD_OK" in r.stdout, "adapter did not reload in a fresh process"
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
