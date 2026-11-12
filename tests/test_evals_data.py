"""The eval data and the scorers must agree: a perfect answer to any item scores 1.0."""

import json

import pytest

from edgewise.evals import TASKS


def concrete(value):
    """Turn expected values with matchers into one concrete acceptable value."""
    if isinstance(value, dict) and "$contains" in value:
        return value["$contains"]
    if isinstance(value, dict) and "$any" in value:
        return value["$any"][0]
    if isinstance(value, dict):
        return {k: concrete(v) for k, v in value.items()}
    return value


def perfect_answer(task_id: str, item) -> str:
    if task_id in ("tool_calling", "json_extraction"):
        return json.dumps(concrete(item.expected))
    if task_id == "grounded_qa":
        return item.expected[0]
    if task_id == "short_reasoning":
        return f"Working it out step by step. Answer: {item.expected}"
    return item.expected


ALL_ITEMS = [(task_id, item) for task_id, task in TASKS.items() for item in task.load_items()]


@pytest.mark.parametrize("task_id, item", ALL_ITEMS, ids=[item.id for _, item in ALL_ITEMS])
def test_perfect_answer_scores_one(task_id, item):
    score = TASKS[task_id].score(perfect_answer(task_id, item), item)
    assert score.valid and score.value == 1, score.detail


@pytest.mark.parametrize("task_id, item", ALL_ITEMS, ids=[item.id for _, item in ALL_ITEMS])
def test_unhelpful_answer_does_not_score_one(task_id, item):
    assert TASKS[task_id].score("I'm not sure, sorry.", item).value < 1


def test_item_counts_and_unique_ids():
    counts = {task_id: len(task.load_items()) for task_id, task in TASKS.items()}
    assert counts == {
        "tool_calling": 20,
        "json_extraction": 16,
        "grounded_qa": 16,
        "short_reasoning": 20,
        "text_edit": 12,
    }
    ids = [item.id for _, item in ALL_ITEMS]
    assert len(set(ids)) == len(ids)


def test_tool_calls_only_use_declared_tools_and_arguments():
    for item in TASKS["tool_calling"].load_items():
        tools = {t["name"]: t["parameters"] for t in item.tools}
        params = tools[item.expected["name"]]
        assert set(item.expected["arguments"]) <= set(params["properties"])
        assert set(params["required"]) <= set(item.expected["arguments"])
