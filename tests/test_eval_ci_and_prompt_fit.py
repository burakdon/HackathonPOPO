"""Bootstrap intervals for agreement and the prompted_all context fitting."""
import pytest

from agents.registry import fit_few_shot
from core.practice import TASK_INSTRUCTION
from evaluation.compare import bootstrap_ci, fmt_agree


def test_bootstrap_interval_contains_the_estimate_is_seeded_and_degenerate_when_constant():
    hits = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1] * 5
    lo, hi = bootstrap_ci(hits)
    assert lo < sum(hits) / len(hits) < hi and 0 <= lo and hi <= 1
    assert bootstrap_ci(hits) == (lo, hi)
    assert bootstrap_ci([1] * 20) == (1.0, 1.0) and bootstrap_ci([0] * 20) == (0.0, 0.0)
    # More situations, narrower interval.
    lo2, hi2 = bootstrap_ci(hits * 4)
    assert hi2 - lo2 < hi - lo


def test_per_situation_means_over_seeds_are_accepted():
    lo, hi = bootstrap_ci([1, 2 / 3, 1 / 3, 0, 1])
    assert 0 <= lo < 0.6 < hi <= 1


def test_agreement_formatting():
    assert fmt_agree(26, 50, (0.38, 0.66)) == "26/50 (52%, CI 38-66%)"
    assert fmt_agree(4.33, 10, (0.2, 0.7)).startswith("4.33/10 (43%")


class WordTokenizer:
    """One token per whitespace-separated word; the chat template just joins the contents."""

    def apply_chat_template(self, msgs, tokenize=False, **kw):
        return " ".join(m["content"] for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_fit_few_shot_takes_the_longest_prefix_within_the_budget():
    tok = WordTokenizer()
    shots = [("a b c d", "x")] * 50                     # 5 words per demonstration
    base = len(tok(tok.apply_chat_template([{"content": TASK_INSTRUCTION}]))["input_ids"])
    fitted, n = fit_few_shot(shots, tok, window=base + 5 * 12 + 100, reserve=100)
    assert len(fitted) == 12 and n == base + 60
    assert len(fit_few_shot(shots, tok, window=10 ** 6)[0]) == 50
    assert fit_few_shot(shots, tok, window=base + 4 + 100, reserve=100)[0] == []
