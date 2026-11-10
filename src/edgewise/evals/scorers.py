"""Scorers: turn a model's raw text output into a Score (see docs/SPEC.md §6.4).

Expected values in eval data may use two matchers where exact strings would be unfair:
  {"$contains": "airport"}         the prediction contains this text
  {"$any": ["Bangalore", "Bengaluru"]}   the prediction matches any option
Text comparison ignores case, spaces and punctuation ("AI 504" == "ai504", "lo-fi" == "lofi").
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any

from edgewise.schema import EvalItem, Score

_DECODER = json.JSONDecoder()
_NUMBER = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"


# ------------------------------------------------------------------ parsing helpers


def extract_json(text: str) -> Any | None:
    """Return the first JSON object found in `text` (tolerates code fences and tags around it)."""
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = _DECODER.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _norm(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value).lower())


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[,\s₹$€]", "", value)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def value_matches(pred: Any, gold: Any) -> bool:
    if isinstance(gold, dict) and "$contains" in gold:
        return pred is not None and _norm(gold["$contains"]) in _norm(pred)
    if isinstance(gold, dict) and "$any" in gold:
        return any(value_matches(pred, option) for option in gold["$any"])
    if gold is None:
        return pred is None or _norm(pred) in {"", "null", "none", "na"}
    if isinstance(gold, bool):
        if isinstance(pred, str):
            return _norm(pred) == str(gold).lower()
        return pred is gold
    if isinstance(gold, int | float):
        number = _to_number(pred)
        return number is not None and abs(number - gold) <= 1e-6 * max(1.0, abs(gold))
    if isinstance(gold, dict):
        return isinstance(pred, dict) and all(
            value_matches(pred.get(k), v) for k, v in gold.items()
        )
    if isinstance(gold, list):
        return (
            isinstance(pred, list)
            and len(pred) == len(gold)
            and all(value_matches(p, g) for p, g in zip(pred, gold, strict=True))
        )
    return pred is not None and _norm(pred) == _norm(gold)


# ------------------------------------------------------------------ scorers


def score_tool_call(output: str, item: EvalItem) -> Score:
    """1 if the right function is called with the right arguments, else 0."""
    obj = extract_json(output)
    if isinstance(obj, dict) and isinstance(obj.get("function"), dict):
        obj = obj["function"]
    if not isinstance(obj, dict) or "name" not in obj:
        return Score(value=0, valid=False, detail="no tool call found")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        args = extract_json(args) or {}
    if not isinstance(args, dict):
        return Score(value=0, valid=False, detail="arguments are not an object")

    expected = item.expected
    if obj["name"] != expected["name"]:
        return Score(
            value=0, valid=True, detail=f"called {obj['name']}, expected {expected['name']}"
        )

    for key, gold in expected["arguments"].items():
        if not value_matches(args.get(key), gold):
            return Score(value=0, valid=True, detail=f"argument {key}={args.get(key)!r} is wrong")

    # Extra arguments are fine only if they just restate the parameter's default.
    defaults = _tool_defaults(item, expected["name"])
    for key in args.keys() - expected["arguments"].keys():
        if key not in defaults or not value_matches(args[key], defaults[key]):
            return Score(value=0, valid=True, detail=f"unexpected argument {key}={args[key]!r}")
    return Score(value=1, valid=True)


def _tool_defaults(item: EvalItem, name: str) -> dict[str, Any]:
    for tool in item.tools or []:
        if tool["name"] == name:
            props = tool.get("parameters", {}).get("properties", {})
            return {k: p["default"] for k, p in props.items() if "default" in p}
    return {}


def score_json_fields(output: str, item: EvalItem) -> Score:
    """Field-level F1 between the extracted JSON object and the expected fields."""
    obj = extract_json(output)
    if not isinstance(obj, dict):
        return Score(value=0, valid=False, detail="no JSON object found")
    gold: dict[str, Any] = item.expected
    correct = [k for k, v in gold.items() if k in obj and value_matches(obj[k], v)]
    if not correct:
        return Score(value=0, valid=True, detail="no field correct")
    precision = len(correct) / len(obj)
    recall = len(correct) / len(gold)
    wrong = sorted(set(gold) - set(correct))
    return Score(
        value=2 * precision * recall / (precision + recall),
        valid=True,
        detail=f"wrong/missing: {', '.join(wrong)}" if wrong else "",
    )


def _squad_tokens(text: str) -> list[str]:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def _token_f1(pred: str, gold: str) -> float:
    p, g = _squad_tokens(pred), _squad_tokens(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def score_qa_f1(output: str, item: EvalItem) -> Score:
    """SQuAD-style token F1 against the best-matching acceptable answer."""
    answer = output.strip().splitlines()[0] if output.strip() else ""
    golds = item.expected if isinstance(item.expected, list) else [item.expected]
    best = max(_token_f1(answer, g) for g in golds)
    return Score(value=best, valid=bool(answer), detail="" if best == 1 else f"answered {answer!r}")


def score_number(output: str, item: EvalItem) -> Score:
    """Exact match on the final number ("Answer: <n>" if present, else the last number)."""
    marked = re.findall(rf"answer\s*[:=]\s*\**\s*[₹$]?\s*({_NUMBER})", output, re.IGNORECASE)
    numbers = marked or re.findall(_NUMBER, output)
    if not numbers:
        return Score(value=0, valid=False, detail="no number found")
    got = _to_number(numbers[-1])
    ok = value_matches(got, item.expected)
    return Score(value=float(ok), valid=True, detail="" if ok else f"answered {numbers[-1]}")


SCORERS = {
    "tool_call": score_tool_call,
    "json_fields": score_json_fields,
    "qa_f1": score_qa_f1,
    "number": score_number,
}
