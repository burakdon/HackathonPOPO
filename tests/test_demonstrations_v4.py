"""v4: contract options everywhere, labels unchanged, targeted training additions; prompt format v2."""
import json
from collections import Counter

from core.config import load_clinical, resolve
from core.practice import DecisionContext, contract_options, format_action, parse_action, render_context
from training.build_demonstrations import OUT_V3, OUT_V4, SPECS, build_v4


def load(path):
    return [json.loads(line) for line in resolve(path).read_text().splitlines()]


def test_committed_file_matches_the_generator():
    assert load(OUT_V4) == build_v4()


def test_same_situations_and_labels_as_v3_with_contract_options():
    c = load_clinical()
    v3 = {r["example_id"]: r for r in load(OUT_V3)}
    v4 = load(OUT_V4)
    for r in v4:
        ctx = DecisionContext(**r["context"])
        assert r["options"] == [format_action(o) for o in contract_options(ctx, c.approved_cues, c.max_tier_step, 3)]
        assert r["preferred"] in r["options"] and set(r["accepted"]) <= set(r["options"])
        if r["example_id"] in v3:
            old = v3[r["example_id"]]
            assert (r["context"], r["preferred"], r["accepted"], r["rule"], r["split"]) == \
                (old["context"], old["preferred"], old["accepted"], old["rule"], old["split"])
    assert {r["example_id"] for r in v4} >= set(v3)


def test_additions_are_training_only_on_training_pictures_and_never_duplicate():
    v4 = load(OUT_V4)
    add = [r for r in v4 if r["family_id"].startswith("X")]
    held = {r["context"]["current"]["word"] for r in v4 if r["split"] != "train"}
    assert len(add) == 42 and all(r["split"] == "train" for r in add)
    assert not {r["context"]["current"]["word"] for r in add} & held
    renders = [render_context(DecisionContext(**r["context"]), [parse_action(o) for o in r["options"]], 4) for r in v4]
    assert len(set(renders)) == len(renders)
    rules = Counter(r["rule"] for r in v4 if r["split"] == "train")
    assert rules["G"] >= 30 and min(rules.values()) >= 10
    g_minutes = sorted(r["context"]["session_minutes"] for r in add if r["rule"] == "G")
    assert sum(20 <= m < 24 for m in g_minutes) >= 8                    # near the 20-minute boundary
    assert all(r["context"]["session_minutes"] < 20 for r in add if r["family_id"] in
               {f"X{n:03d}" for n in range(21, 33)})                     # contrast set stays below it


def test_prompt_v2_labels_options_relatively_and_names_hints_like_errors():
    r = next(x for x in load(OUT_V4) if x["rule"] == "F" and x["context"]["current"]["tier"] == 3)
    text = render_context(DecisionContext(**r["context"]), [parse_action(o) for o in r["options"]], 4, "v2")
    assert '{"op":"next","tier":2} = new picture, one level easier (tier 2)' in text
    assert '{"op":"next","tier":3} = new picture, same level (tier 3)' in text
    d = next(x for x in load(OUT_V4) if x["rule"] == "D-meaning")
    text = render_context(DecisionContext(**d["context"]), [parse_action(o) for o in d["options"]], 4, "v2")
    assert "incorrect (meaning error)" in text and "= same picture again with the meaning hint" in text
    assert "Session time:" in text
    # v1 rendering is unchanged by default
    assert " = " not in render_context(DecisionContext(**d["context"]), [parse_action(o) for o in d["options"]], 4)
