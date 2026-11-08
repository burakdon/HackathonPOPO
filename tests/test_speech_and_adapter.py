"""Speech provider statuses (needs the `speech` extra) and adapter save/reload in a fresh process
(needs the `llm` extra and the Qwen3 tokenizer in the local Hugging Face cache; no download)."""
import io
import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest


def test_whisper_reports_garbage_and_silence_without_running_the_model():
    pytest.importorskip("faster_whisper")
    av = pytest.importorskip("av")
    from speech.whisper_local import WhisperLocalProvider
    p = WhisperLocalProvider()
    assert p.transcribe(b"", None).status == "empty"
    assert p.transcribe(b"definitely not audio", "audio/webm").status == "failed"
    # 1 s of digital silence encoded as WebM/Opus (what Chrome's MediaRecorder sends)
    buf = io.BytesIO()
    out = av.open(buf, "w", format="webm")
    st = out.add_stream("libopus", rate=48000)
    st.layout = "mono"
    fr = av.AudioFrame.from_ndarray(np.zeros((1, 48000), dtype=np.int16), format="s16", layout="mono")
    fr.sample_rate = 48000
    for pkt in st.encode(fr):
        out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()
    r = p.transcribe(buf.getvalue(), "audio/webm;codecs=opus")
    assert r.status == "empty" and "silent" in r.error and p._model is None      # model never loaded


def _tokenizer_cached() -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tokenizer_cached(), reason="needs the llm extra and a cached Qwen3 tokenizer")
def test_adapter_saved_by_training_reloads_in_a_fresh_process_and_drives_a_decision(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from training.ditto import save_adapter
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    base = tmp_path / "tiny-base"
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=2048)
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(base)
    tok.save_pretrained(base)
    model = Qwen3ForCausalLM.from_pretrained(base)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"), adapter_name="sft")
    model.add_adapter("ditto", LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    save_adapter(model, "ditto", tmp_path / "round1")
    assert (tmp_path / "round1" / "adapter_config.json").exists()
    script = textwrap.dedent(f"""
        import json
        from agents.llm_policy import BanditDecisionPolicy, LLMDecisionPolicy
        from agents.llm_runtime import LLMRuntime
        from core.config import load_beta_bandit, load_clinical
        from core.practice import LLMConfig, DecisionContext, valid_options
        rows = [json.loads(l) for l in open("data/demonstrations/demonstrator_a_v1.jsonl")]
        r = next(x for x in rows if x["split"] == "test")
        rt = LLMRuntime(LLMConfig(base_model={str(base)!r}, device="cpu", dtype="float32", max_new_tokens=8,
                                  timeout_s=60, runs_dir="runs/ditto"))
        rt.ensure_adapter("ditto-test", {str(tmp_path / 'round1')!r})
        ctx = DecisionContext(**r["context"])
        cl = load_clinical()
        opts = valid_options(ctx, cl.approved_cues, cl.max_tier_step, 3)
        pol = LLMDecisionPolicy(rt, "ditto:test", "ditto-test", {{}}, 4, 60)
        d = pol.decide(ctx, opts, BanditDecisionPolicy(load_beta_bandit(), cl, seed=0))
        print(json.dumps({{"adapters": list(rt.adapters), "source": d.source, "requested": d.requested_model,
                          "action": d.action.key(), "valid_option": d.action.key() in r["options"]}}))
    """)
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["adapters"] == ["ditto-test"] and out["requested"] == "ditto:test"
    # A random tiny model emits junk: the boundary must catch it and fall back, never pass it through.
    assert out["valid_option"]
    assert out["source"] in ("ditto:test", "bandit")
