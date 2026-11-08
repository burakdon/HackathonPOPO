"""v3: v2 training rows unchanged, new larger dev/test splits, v1 held-out rows kept as dev_v1/test_v1."""
import json
from collections import Counter

from core.config import load_clinical, resolve
from core.practice import DecisionContext, parse_action, render_context, valid_options
from training.build_demonstrations import OUT_V2, OUT_V3, build_v3, demonstrator_a
from training.ditto import examples_hash, load_examples


def load(path):
    return [json.loads(line) for line in resolve(path).read_text().splitlines()]


def render(x):
    return render_context(DecisionContext(**x["context"]), [parse_action(o) for o in x["options"]], 4)


def current(rows):
    return {x["context"]["current"]["word"] for x in rows}


def test_committed_file_matches_the_generator():
    assert load(OUT_V3) == build_v3()


def test_training_rows_are_exactly_v2s_and_v1_held_out_rows_are_kept_relabelled():
    v2, v3 = load(OUT_V2), load(OUT_V3)
    assert load_examples(OUT_V3, "train") == load_examples(OUT_V2, "train")
    assert examples_hash(load_examples(OUT_V3, "train")) == examples_hash(load_examples(OUT_V2, "train"))
    for old in ("dev", "test"):
        assert [{**x, "split": old} for x in v3 if x["split"] == f"{old}_v1"] == [x for x in v2 if x["split"] == old]


def test_new_held_out_splits_are_larger_cover_every_rule_and_never_reuse_training_pictures():
    v3 = load(OUT_V3)
    split = lambda s: [x for x in v3 if x["split"] == s]
    dev, test, train = split("dev"), split("test"), split("train")
    assert len(dev) >= 30 and len(test) >= 50
    assert min(Counter(x["rule"] for x in dev).values()) >= 3 and len(Counter(x["rule"] for x in dev)) == 11
    assert min(Counter(x["rule"] for x in test).values()) >= 5 and len(Counter(x["rule"] for x in test)) == 11
    assert not current(test) & current(train) and not current(dev) & current(train)
    assert not current(dev) & current(test)
    assert all(x["family_id"][0] in "HT" for x in dev + test)


def test_no_situation_duplicates_another_and_labels_follow_the_strategy():
    clinical = load_clinical()
    v3 = load(OUT_V3)
    renders = [render(x) for x in v3]
    assert len(set(renders)) == len(renders)
    assert len({x["example_id"] for x in v3}) == len(v3)
    for x in v3:
        ctx = DecisionContext(**x["context"])
        opts = valid_options(ctx, clinical.approved_cues, clinical.max_tier_step, 3)
        assert demonstrator_a(ctx, opts)[1] == x["accepted"]
        assert x["provenance"]["review_status"] == "pending human review"
        if x["split"] in ("dev", "test"):
            assert "generated from the written strategy" in x["provenance"]["source"]
