"""Eval tasks: how each task turns an item into a prompt, and how it is scored."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from edgewise.evals.scorers import SCORERS
from edgewise.schema import EvalItem, Message, Score

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "evals" / "data"


@dataclass(frozen=True)
class Task:
    id: str
    description: str
    scorer: str
    max_tokens: int
    build: Callable[[EvalItem], list[Message]]

    def score(self, output: str, item: EvalItem) -> Score:
        return SCORERS[self.scorer](output, item)

    def load_items(self, data_dir: Path = DEFAULT_DATA_DIR) -> list[EvalItem]:
        path = data_dir / f"{self.id}.jsonl"
        with path.open() as f:
            return [EvalItem(**json.loads(line)) for line in f if line.strip()]


def _tool_calling(item: EvalItem) -> list[Message]:
    tools = json.dumps(item.tools, indent=1)
    system = (
        "You are an in-car assistant. You can call exactly one of these tools:\n"
        f"{tools}\n\n"
        'Respond with ONLY a JSON object: {"name": <tool name>, "arguments": {...}}. '
        "No other text."
    )
    return [Message(role="system", content=system), Message(role="user", content=item.prompt)]


def _json_extraction(item: EvalItem) -> list[Message]:
    system = (
        "Extract information from the text. Reply with ONLY a JSON object containing exactly "
        "the requested fields. Use null for anything the text does not state."
    )
    user = f"Text:\n{item.context}\n\n{item.prompt}"
    return [Message(role="system", content=system), Message(role="user", content=user)]


def _grounded_qa(item: EvalItem) -> list[Message]:
    system = (
        "Answer the question using only the passage. "
        "Reply with just the answer, in as few words as possible."
    )
    user = f"Passage:\n{item.context}\n\nQuestion: {item.prompt}"
    return [Message(role="system", content=system), Message(role="user", content=user)]


def _short_reasoning(item: EvalItem) -> list[Message]:
    system = "Solve the problem. Think briefly, then end with a final line 'Answer: <number>'."
    return [Message(role="system", content=system), Message(role="user", content=item.prompt)]


def _text_edit(item: EvalItem) -> list[Message]:
    system = (
        "Apply the requested edit to the text. "
        "Reply with ONLY the full edited text, changing nothing else."
    )
    user = f"Text:\n{item.context}\n\nEdit: {item.prompt}"
    return [Message(role="system", content=system), Message(role="user", content=user)]


TASKS: dict[str, Task] = {
    t.id: t
    for t in [
        Task("tool_calling", "Pick the right tool and arguments", "tool_call", 128, _tool_calling),
        Task(
            "json_extraction",
            "Fill requested fields from text",
            "json_fields",
            160,
            _json_extraction,
        ),
        Task("grounded_qa", "Answer from a given passage", "qa_f1", 48, _grounded_qa),
        Task("short_reasoning", "Arithmetic word problems", "number", 256, _short_reasoning),
        Task(
            "text_edit",
            "Apply a small edit and return the full text",
            "text_match",
            200,
            _text_edit,
        ),
    ]
}


def get_task(task_id: str) -> Task:
    try:
        return TASKS[task_id]
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}. Known tasks: {', '.join(TASKS)}") from None
