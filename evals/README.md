# Eval data

One JSONL file per task. Each line is an `EvalItem` (see `src/edgewise/schema.py`).

| Task | Items | Scorer | Source |
|---|---|---|---|
| `tool_calling` | 10 | Right tool + right arguments (1/0) | Hand-written (in-car assistant tools) |
| `json_extraction` | 8 | Field-level F1 | Hand-written |
| `grounded_qa` | 8 | SQuAD-style token F1 | Hand-written passages |
| `short_reasoning` | 8 | Exact match on the final number | Hand-written word problems |

These are **seed sets**, so the bench pipeline can be built and tested end to end. They are too small to support confident conclusions: a single item is worth 10–12.5 points. Milestone 1 grows each task toward 50–100 items, adding public datasets where their license allows; sources and licenses will be listed here.

## Matchers in `expected`

Exact strings would penalize answers that are correct but phrased differently, so expected values may use:

- `{"$contains": "airport"}`: the prediction contains this text
- `{"$any": ["Bangalore", "Bengaluru"]}`: the prediction matches any option

Text comparison ignores case, spaces and punctuation.
