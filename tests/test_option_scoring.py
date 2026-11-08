"""decoding = "score": choose the most likely valid option (inference) and sample DITTO alternatives
from the distribution over valid options (training)."""
import pytest

from agents.llm_policy import LLMDecisionPolicy
from core.practice import DecisionContext, LLMConfig, format_action, next_ex, retry

CTX = DecisionContext(
    current={"exercise_id": "ex-banana", "word": "banana", "tier": 2, "hints_shown": [], "attempts": 1},
    history=[{"exercise_id": "ex-banana", "word": "banana", "tier": 2, "hint": "none", "correct": False,
              "error": "sound", "response_time_s": 5.0}],
    session_minutes=5)
OPTIONS = [retry("first_sound"), retry("semantic"), next_ex(1), next_ex(2)]


class ScoringRuntime:
    cfg = LLMConfig(base_model="x", max_new_tokens=8, timeout_s=5, runs_dir="runs/ditto", decoding="score")

    def score(self, msgs, candidates, adapter=None):
        self.seen = candidates
        return [-3.0, -0.5, -2.0, -1.0], 0.01

    def generate(self, msgs, adapter=None):
        raise AssertionError("score mode must not generate")


def test_score_mode_picks_the_most_likely_valid_option():
    rt = ScoringRuntime()
    pol = LLMDecisionPolicy(rt, "base", None, {}, 4, 5.0)
    text, _ = pol.generate(CTX, OPTIONS)
    assert rt.seen == [format_action(o) for o in OPTIONS]
    assert text == format_action(retry("semantic"))


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception:
        return None


@pytest.mark.skipif(_tokenizer() is None, reason="needs the llm extra and a cached Qwen3 tokenizer")
def test_runtime_scores_match_training_logps_and_option_samples_are_valid_and_seeded(tmp_path):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from agents.llm_runtime import LLMRuntime
    from core.practice import messages
    from training.ditto import Encoder, answer_logps, sample_options
    tok = _tokenizer()
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=2048)
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    rt = LLMRuntime(LLMConfig(base_model=str(tmp_path), device="cpu", dtype="float32", max_new_tokens=8,
                              timeout_s=10, runs_dir="runs/ditto", decoding="score"))
    msgs = messages(CTX, OPTIONS, 4)
    keys = [format_action(o) for o in OPTIONS]
    lps, _ = rt.score(msgs, keys)
    enc = Encoder(rt.tokenizer, "cpu")
    ref = answer_logps(rt.model, enc, [(enc.prompt_ids(msgs), enc.answer_ids(k)) for k in keys])
    assert lps == pytest.approx(ref.tolist(), abs=1e-4)
    draw = lambda: sample_options(rt.model, enc, enc.prompt_ids(msgs), keys, 12, 1.0, torch.Generator().manual_seed(3))
    a = draw()
    assert len(a) == 12 and set(a) <= set(keys) and a == draw()


def test_stage_selection_prefers_the_earlier_stage_on_ties():
    from training.ditto import select_stage
    assert select_stage({"sft": 20, "round1": 22, "round2": 22, "round3": 18}) == "round1"
    assert select_stage({"sft": 22, "round1": 22, "round2": 21}) == "sft"
    assert select_stage({"sft": 10, "round1": 9, "round2": 8}) == "sft"


@pytest.mark.skipif(_tokenizer() is None, reason="needs the llm extra and a cached Qwen3 tokenizer")
def test_one_option_at_a_time_scoring_equals_batched_scoring(tmp_path, monkeypatch):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    import agents.llm_runtime as llm_runtime
    from core.practice import messages
    tok = _tokenizer()
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=2048)
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    rt = llm_runtime.LLMRuntime(LLMConfig(base_model=str(tmp_path), device="cpu", dtype="float32", max_new_tokens=8,
                                          timeout_s=10, runs_dir="runs/ditto", decoding="score"))
    msgs, keys = messages(CTX, OPTIONS, 4, fmt="v2"), [format_action(o) for o in OPTIONS]
    batched, _ = rt.score(msgs, keys)
    monkeypatch.setattr(llm_runtime, "MAX_SCORE_BATCH_TOKENS", 1)
    single, _ = rt.score(msgs, keys)
    assert single == pytest.approx(batched, abs=1e-4)
