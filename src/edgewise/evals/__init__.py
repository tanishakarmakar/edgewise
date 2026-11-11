"""Task-based eval suite: datasets live in evals/data/<task>.jsonl at the repo root."""

from edgewise.evals.tasks import TASKS, Task, get_task

__all__ = ["TASKS", "Task", "get_task"]
