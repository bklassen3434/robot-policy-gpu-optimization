"""Unit tests for the voice instruction parser (Path B).

Pure-Python parsing, no mic/model/torch — the correctness signal for the voice
layer is that free-form transcripts map onto the exact canonical instruction that
conditions SmolVLA (verb synonyms normalized, unknown/verbless input rejected).
"""

from __future__ import annotations

from robopolicy.realbot.config import load_config
from robopolicy.realbot.voice import normalize_instruction

OBJECTS = ["red block", "green block", "blue block"]
SYNONYMS = ["pick up", "grab", "get", "take", "pick"]
TEMPLATE = "pick up the {object}"


def norm(text):
    return normalize_instruction(text, OBJECTS, SYNONYMS, TEMPLATE)


def test_canonical_passthrough():
    r = norm("pick up the red block")
    assert r is not None
    assert r.canonical == "pick up the red block"
    assert r.target == "red block"


def test_synonyms_normalize_to_pick_up():
    for verb in ["grab", "get", "take", "pick"]:
        r = norm(f"{verb} the blue block")
        assert r is not None, verb
        assert r.canonical == "pick up the blue block"


def test_case_and_punctuation_insensitive():
    r = norm("PICK UP THE GREEN BLOCK!!")
    assert r is not None
    assert r.canonical == "pick up the green block"


def test_verbless_mention_rejected():
    assert norm("the red block is on the table") is None


def test_unknown_object_rejected():
    assert norm("pick up the yellow duck") is None


def test_empty_input_rejected():
    assert norm("") is None
    assert norm("   ") is None


def test_longest_object_name_wins():
    # a bare "block" mention with a color present resolves to the colored object,
    # not a spurious partial match.
    r = norm("please grab the green block now")
    assert r is not None
    assert r.target == "green block"


def test_config_objects_match_parser_contract():
    # the shipped config's objects/synonyms/template all parse round-trip.
    cfg = load_config()
    task = cfg["task"]
    objs = task["objects"]
    syn = task["verb_synonyms"]
    tmpl = task["instruction_template"]
    for obj in objs:
        r = normalize_instruction(f"grab the {obj}", objs, syn, tmpl)
        assert r is not None
        assert r.canonical == tmpl.format(object=obj)
