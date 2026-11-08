"""v2 demonstration set: v1 unchanged plus generated training families; dev/test untouched."""
import json
from collections import Counter

from core.config import load_clinical, resolve
from core.practice import DecisionContext, parse_action, render_context, valid_options
from training.build_demonstrations import OUT, OUT_V2, build_v2, demonstrator_a


def load(path):
    return [json.loads(line) for line in resolve(path).read_text().splitlines()]


def render(x):
    return render_context(DecisionContext(**x["context"]), [parse_action(o) for o in x["options"]], 4)


def test_committed_file_matches_the_generator():
    assert load(OUT_V2) == build_v2()


def test_dev_and_test_rows_are_identical_to_v1_and_v1_training_rows_are_kept():
    v1, v2 = load(OUT), load(OUT_V2)
    assert v2[:len(v1)] == v1
    assert [x for x in v2 if x["split"] != "train"] == [x for x in v1 if x["split"] != "train"]


def test_generated_situations_never_duplicate_or_reword_dev_or_test():
    v2 = load(OUT_V2)
    gen = [x for x in v2 if x["family_id"].startswith("G")]
    held = [x for x in v2 if x["split"] != "train"]
    assert len(gen) >= 90 and all(x["split"] == "train" for x in gen)
    held_words = {x["context"]["current"]["word"] for x in held}
    assert not held_words & {x["context"]["current"]["word"] for x in gen}
    renders = [render(x) for x in v2 if x["split"] == "train"]
    assert len(set(renders)) == len(renders)                     # no duplicate training situations
    assert not set(renders) & {render(x) for x in held}
    assert len({x["example_id"] for x in v2}) == len({x["family_id"] for x in v2}) == len(v2)


def test_every_rule_is_covered_several_times_and_labels_follow_the_strategy():
    clinical = load_clinical()
    v2 = load(OUT_V2)
    per_rule = Counter(x["rule"] for x in v2 if x["split"] == "train")
    assert len(per_rule) == 11 and min(per_rule.values()) >= 8
    for x in v2:
        ctx = DecisionContext(**x["context"])
        opts = valid_options(ctx, clinical.approved_cues, clinical.max_tier_step, 3)
        assert x["options"] == [o.key() for o in opts]
        assert demonstrator_a(ctx, opts)[1] == x["accepted"]
        assert x["provenance"]["review_status"] == "pending human review"
    assert all("generated from the written strategy" in x["provenance"]["source"]
               for x in v2 if x["family_id"].startswith("G"))
