# Eval data

One JSONL file per task. Each line is an `EvalItem` (see `src/edgewise/schema.py`).

| Task | Items | Scorer | Source |
|---|---:|---|---|
| `tool_calling` | 20 | Right tool + right arguments (1/0) | Hand-written (in-car assistant tools) |
| `json_extraction` | 16 | Field-level F1 | Hand-written |
| `grounded_qa` | 16 | Contains an acceptable answer, else token F1 | Hand-written passages |
| `short_reasoning` | 20 | Exact match on the final number | Hand-written word problems |
| `text_edit` | 12 | Exact match, else token F1 | Hand-written (apply one edit, return full text) |

84 items in total. That is enough to build and compare with, but still small: one item moves a task score by 0.05 to 0.08, so differences of a few points between models are noise. Adding public datasets (where their license allows) is the next step; sources and licenses will be listed here.

`tests/test_evals_data.py` checks every item against its scorer: a perfect answer must score 1.0 and an unhelpful one must not.

## Matchers in `expected`

Exact strings would penalize answers that are correct but phrased differently, so expected values may use:

- `{"$contains": "airport"}`: the prediction contains this text
- `{"$any": ["Bangalore", "Bengaluru"]}`: the prediction matches any option

Text comparison ignores case, spaces and punctuation.
