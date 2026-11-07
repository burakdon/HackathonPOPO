"""Demonstration dataset: fixed splits, no leakage of test situations, consistent preferences."""
import json

from agents.registry import few_shot_examples
from core.config import load_clinical, resolve
from core.practice import DecisionContext, load_items, parse_action, render_context, valid_options
from training.build_demonstrations import OUT, build, demonstrator_a
from training.ditto import build_pairs, load_examples


def rows():
    return [json.loads(line) for line in resolve(OUT).read_text().splitlines()]


def test_committed_file_matches_the_generator():
    assert rows() == build()


def test_split_sizes_and_unique_ids():
    r = rows()
    assert {s: sum(x["split"] == s for x in r) for s in ("train", "dev", "test")} == {"train": 20, "dev": 5, "test": 10}
    assert len({x["example_id"] for x in r}) == len(r)
    assert len({x["family_id"] for x in r}) == len(r)          # one situation per family: no variants


def test_test_pictures_never_current_in_train_or_dev():
    r = rows()
    test_words = {x["context"]["current"]["word"] for x in r if x["split"] == "test"}
    other = {x["context"]["current"]["word"] for x in r if x["split"] != "test"}
    assert not test_words & other


def test_no_test_context_duplicates_a_training_context():
    r = rows()
    h = 4
    render = lambda x: render_context(DecisionContext(**x["context"]), [parse_action(o) for o in x["options"]], h)
    train = {render(x) for x in r if x["split"] == "train"}
    assert not any(render(x) in train for x in r if x["split"] == "test")


def test_preferences_are_valid_options_and_follow_the_written_strategy():
    clinical = load_clinical()
    for x in rows():
        ctx = DecisionContext(**x["context"])
        opts = valid_options(ctx, clinical.approved_cues, clinical.max_tier_step, 3)
        assert x["options"] == [o.key() for o in opts]
        assert x["preferred"] == x["accepted"][0] and set(x["accepted"]) <= set(x["options"])
        assert demonstrator_a(ctx, opts)[1] == x["accepted"]
        assert x["provenance"]["review_status"] == "pending human review"


def test_items_referenced_exist():
    bank = load_items()
    for x in rows():
        assert x["context"]["current"]["exercise_id"] in bank.by_id


def test_training_and_prompted_baseline_use_only_training_situations():
    train = load_examples(OUT, "train")
    assert len(train) == 20 and all(x["split"] == "train" for x in train)
    shots = few_shot_examples(OUT, 4)
    test_prompts = {render_context(DecisionContext(**x["context"]), [parse_action(o) for o in x["options"]], 4)
                    for x in rows() if x["split"] != "train"}
    assert len(shots) == 20 and not {c for c, _ in shots} & test_prompts


def test_pairs_never_reject_an_accepted_action_or_an_equivalent_one():
    import random
    train = [x for x in rows() if x["rule"] == "E-alt" and x["split"] == "train"]
    eid = train[0]["example_id"]
    acc = train[0]["accepted"]
    alt = acc[1]                                          # the also-accepted alternative
    respaced = alt.replace(",", ", ")                     # same action, different wording
    cache = {0: {eid: [alt, respaced, '{"op":"next","tier":1}', "nonsense"]},
             1: {eid: [alt, '{"op":"next","tier":1}']}}
    cfg = {"rescale_batch": 50, "frac_expert": 0.7, "frac_replay": 0.2, "frac_intermodel": 0.1}
    pairs, stats = build_pairs(cache, train, 1, cfg, random.Random(0))
    for _, kind, chosen, rejected in pairs:
        p = parse_action(rejected)
        assert p is None or p.key() not in acc
        assert chosen != rejected
    kinds = {k for _, k, _, _ in pairs}
    assert kinds == {"expert", "replay", "intermodel"}
    assert stats["dropped"]["rejected_is_accepted"] > 0
