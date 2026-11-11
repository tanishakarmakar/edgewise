import pytest

from edgewise.evals.scorers import (
    extract_json,
    score_json_fields,
    score_number,
    score_qa_f1,
    score_text_match,
    score_tool_call,
    value_matches,
)
from edgewise.schema import EvalItem

TOOLS = [
    {
        "name": "navigate_to",
        "parameters": {
            "properties": {
                "destination": {"type": "string"},
                "avoid_tolls": {"type": "boolean", "default": False},
            }
        },
    }
]


def item(expected, tools=None):
    return EvalItem(id="x", prompt="p", expected=expected, tools=tools)


# ------------------------------------------------------------------ helpers


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        'Sure! ```json\n{"a": 1}\n```',
        '<tool_call>{"a": 1}</tool_call>',
        'Here {not json} then {"a": 1}',
    ],
)
def test_extract_json_finds_object(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_none_when_missing():
    assert extract_json("no json here [1, 2]") is None


@pytest.mark.parametrize(
    "pred, gold",
    [
        ("AI504", "AI 504"),
        ("lo-fi beats", {"$contains": "lofi"}),
        ("Bengaluru", {"$any": ["Bangalore", "Bengaluru"]}),
        ("18,450", 18450),
        ("₹18450", 18450),
        (75.0, 75),
        (None, None),
        ("null", None),
        ("true", True),
        ({"a": "X", "b": 2}, {"a": "x"}),
    ],
)
def test_value_matches(pred, gold):
    assert value_matches(pred, gold)


@pytest.mark.parametrize(
    "pred, gold",
    [("Delhi", "Mumbai"), (True, 1), (1, True), ("abc", 3), (None, "x"), ("7", 7.5)],
)
def test_value_mismatches(pred, gold):
    assert not value_matches(pred, gold)


# ------------------------------------------------------------------ tool calls

NAV = {"name": "navigate_to", "arguments": {"destination": {"$contains": "airport"}}}


def test_tool_call_correct():
    out = '{"name": "navigate_to", "arguments": {"destination": "Nearest airport"}}'
    assert score_tool_call(out, item(NAV, TOOLS)).value == 1


def test_tool_call_accepts_openai_style_and_string_arguments():
    out = '{"function": {"name": "navigate_to", "arguments": "{\\"destination\\": \\"airport\\"}"}}'
    assert score_tool_call(out, item(NAV, TOOLS)).value == 1


def test_tool_call_extra_default_argument_is_ok():
    out = '{"name": "navigate_to", "arguments": {"destination": "airport", "avoid_tolls": false}}'
    assert score_tool_call(out, item(NAV, TOOLS)).value == 1


def test_tool_call_extra_non_default_argument_fails():
    out = '{"name": "navigate_to", "arguments": {"destination": "airport", "avoid_tolls": true}}'
    s = score_tool_call(out, item(NAV, TOOLS))
    assert (s.value, s.valid) == (0, True)
    assert "avoid_tolls" in s.detail


def test_tool_call_wrong_name():
    s = score_tool_call('{"name": "play_media", "arguments": {}}', item(NAV, TOOLS))
    assert (s.value, s.valid) == (0, True)


def test_tool_call_invalid_output():
    s = score_tool_call("I'll navigate you there!", item(NAV, TOOLS))
    assert (s.value, s.valid) == (0, False)


# ------------------------------------------------------------------ json extraction

GOLD = {"name": "Ananya Rao", "city": "Bangalore", "phone": None}


def test_json_fields_perfect():
    out = '{"name": "Ananya Rao", "city": "Bangalore", "phone": null}'
    assert score_json_fields(out, item(GOLD)).value == 1


def test_json_fields_partial_f1():
    out = '{"name": "Ananya Rao", "city": "Pune", "phone": null, "extra": 1}'
    s = score_json_fields(out, item(GOLD))
    precision, recall = 2 / 4, 2 / 3
    assert s.value == pytest.approx(2 * precision * recall / (precision + recall))
    assert "city" in s.detail


def test_json_fields_invalid():
    assert score_json_fields("name: Ananya", item(GOLD)).valid is False


# ------------------------------------------------------------------ QA and numbers


def test_qa_f1_ignores_articles_and_case():
    assert score_qa_f1("The Bay of Bengal.", item(["the Bay of Bengal"])).value == 1


def test_qa_f1_partial_credit_and_best_gold():
    assert score_qa_f1("POSIX", item(["POSIX operating systems", "POSIX"])).value == 1
    assert 0 < score_qa_f1("Bay", item(["Bay of Bengal"])).value < 1


def test_qa_full_sentence_containing_the_answer_gets_full_credit():
    out = "The Ganges empties into the Bay of Bengal."
    assert score_qa_f1(out, item(["the Bay of Bengal"])).value == 1
    out = "The three factors are severity, exposure, and controllability of a hazard."
    assert score_qa_f1(out, item(["severity, exposure and controllability"])).value == 1


def test_qa_wrong_sentence_gets_no_full_credit():
    out = "The Ganges empties into the Arabian Sea."
    assert score_qa_f1(out, item(["the Bay of Bengal"])).value < 0.5


def test_text_match_exact_ignoring_fences_and_case():
    gold = "def f(x):\n    return x + 1"
    assert score_text_match("```python\ndef f(x):\n    return x + 1\n```", item(gold)).value == 1
    assert score_text_match("The cat sat.", item("the cat sat")).value == 1


def test_text_match_partial_credit_for_small_differences():
    gold = "codes older than sixty days are archived automatically"
    s = score_text_match("codes older than ninety days are archived automatically", item(gold))
    assert 0.8 < s.value < 1
    assert score_text_match("", item(gold)).valid is False


def test_number_prefers_marked_answer():
    out = "180 / 2.5 = 72, so about 72.\nAnswer: 72"
    assert score_number(out, item(72)).value == 1
    assert score_number("We get 3 then 5. Answer: 41", item(41)).value == 1


def test_number_falls_back_to_last_number_and_handles_commas():
    assert score_number("The total is 1,000 tokens", item(1000)).value == 1


def test_number_wrong_and_missing():
    assert score_number("Answer: 40", item(41)).value == 0
    assert score_number("I don't know", item(41)).valid is False
