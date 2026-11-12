# Edgewise

**Run the right LLM, the right way, on the hardware you have.**

Edgewise is a toolkit for running large language models on everyday devices (laptops, edge boxes, in-vehicle computers) and reaching for a bigger model only when it is actually needed. It measures real models on your own machine and uses those measurements to answer three questions:

| | Question | Command |
|---|---|---|
| **Advisor** | Which model variant is the best fit for this device and this task? | `edgewise advise` |
| **Accelerator** | How much faster can it generate without changing its output? | `edgewise bench --decoding …` |
| **Router** | Which requests can stay on the device, and which need a stronger model? | `edgewise route`, `edgewise serve` |

![edgewise models ls](docs/img/models-ls.svg)

## Status

All five parts are implemented and covered by unit tests. What is still missing is the full set of measurements.

| Part | Code | Verified on real models |
|---|---|---|
| Bench (speed, memory, task quality) | ✅ | ✅ one model benchmarked end to end |
| Accelerator (speculative + prompt-lookup decoding) | ✅ | 🟡 outputs match standard decoding; speed numbers not final |
| Advisor | ✅ | ⏳ needs the full benchmark run |
| Router + local API | ✅ | ⏳ tested with stand-in models only |
| Reports and charts | ✅ | ⏳ charts checked with synthetic data only |

**Nothing in this README is made up.** The few numbers below were measured on the reference device and are marked preliminary. Where a result is missing, it has not been measured yet.

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and, for running models, an Apple Silicon Mac.

```bash
git clone https://github.com/tanishakarmakar/edgewise.git
cd edgewise
uv sync --extra mlx                 # add --extra cloud --extra serve for the router's cloud tier and API

uv run edgewise info                # this device and its memory budget for models
uv run edgewise models ls           # catalog, estimated memory, what fits
uv run edgewise models pull qwen2.5-0.5b-instruct@mlx-4bit
uv run edgewise bench -m qwen2.5-0.5b-instruct@mlx-4bit --all-tasks
```

Then, once a few models are benchmarked:

```bash
uv run edgewise advise --task tool_calling --memory 4GB --min-score 0.8
uv run edgewise bench -m qwen2.5-3b-instruct@mlx-4bit --all-tasks --decoding prompt_lookup
uv run edgewise route "Summarise this for me: ..." --remote none
uv run edgewise serve               # OpenAI-compatible API on http://127.0.0.1:8080
uv run edgewise report              # tables + charts in results/published/
```

`scripts/bench_all.sh` benchmarks the whole catalog, one model at a time, smallest first.

> Benchmarks load multi-gigabyte models. On an 8 GB machine, close other apps first: Edgewise warns when the system is already swapping and tags those results as unreliable.

## How it works

Everything is built on one foundation: a **backend interface** (MLX today), a **task-based eval suite**, a **bench runner** and a **results store** (DuckDB). The three features only ever read measurements from that store.

### Bench

For one model, the runner checks the memory estimate against the device budget, loads the model, warms it up, measures raw speed at three prompt lengths (time to first token, prefill and decode tokens/s, inter-token latency, peak memory; median and interquartile range over repeats), scores it on the eval tasks, and unloads it even if something fails. Every run records the hardware, library versions, git commit, power source and swap usage.

Eval tasks (hand-written seed sets, 6 to 10 items each; small, so treat scores as rough):

| Task | What the model must do | Scoring |
|---|---|---|
| `tool_calling` | Pick the right tool and arguments from a tool list | Exact call match |
| `json_extraction` | Fill requested fields from text | Field-level F1 |
| `grounded_qa` | Answer from a given passage | Contains the answer, else token F1 |
| `short_reasoning` | Arithmetic word problems | Final number |
| `text_edit` | Apply one edit and return the full text | Exact match, else token F1 |

### Advisor

`advise` takes a task, a memory budget and an optional quality floor. It drops variants that do not fit (using the measured peak when there is one), drops those below the floor, keeps the ones on the score/speed Pareto frontier, and picks by preference (`balanced`, `speed` or `quality`). It explains the choice against the runner-up and flags weak evidence (few test items, battery power, memory pressure). Variants that have not been benchmarked are listed with the command to benchmark them and are never ranked.

### Accelerator

Speculative decoding with greedy verification: something cheap guesses the next few tokens, and the model checks all of them in a single forward pass and keeps the longest prefix it agrees with. Every emitted token is the model's own choice, so the output is the same as normal decoding.

- **Draft model**: a 0.5B model guesses for a 1.5B or 3B model with the same tokenizer.
- **Prompt lookup**: if the last few tokens already appeared in the prompt, guess what followed them there. No extra model, useful when the output copies the input (editing, extraction, RAG).
- **Cost-aware guess length**: verifying guesses is not free. On the reference device a pass over 9 tokens costs about 3× a pass over 1 (see below), so guessing only pays when guesses land. Edgewise measures that cost curve per model, tracks how often guesses are accepted, and picks the guess length with the best expected tokens per millisecond, including "don't guess".

The engine is written against a small model interface, so 42 tests prove with fake models that accelerated output equals plain greedy output for every proposer and policy.

### Router

For each request, in order:

1. **Hard rules** that nothing overrides: requests marked private, requests containing personal data (email, phone, card, national ID patterns), and requests made offline stay on the device.
2. **Cascade**: answer locally, then escalate only if the answer is empty, is not valid JSON when JSON was asked for, or was generated with low confidence (mean token log-probability below a threshold).
3. If escalation fails, the local answer is returned instead of an error.

The escalation tier is either a larger on-device model or the Claude API; cost is tracked per request, and prompts are never written to the route log. `edgewise report --local A --strong B` replays the cascade at every threshold from stored benchmark results, so the threshold is chosen from data instead of by guess. When there are enough eval items, it is chosen on one half and reported on the other.

`edgewise serve` exposes the router as `POST /v1/chat/completions` (JSON and SSE), so OpenAI-compatible clients work unchanged. Each response carries an `edgewise` object with the route, the reason and the cost.

## Preliminary measurements

Reference device: **Apple A18 Pro, 8 GB RAM**. These were taken on battery power with other apps open, so they are indicative only and will be replaced by a clean run.

**Qwen 2.5 0.5B, MLX 4-bit, standard decoding**

| Prompt tokens | First token | Decode speed | Peak memory |
|---:|---:|---:|---:|
| 157 | 134 ms | 85 tokens/s | 382 MB |
| 1,053 | 945 ms | 81 tokens/s | 662 MB |
| 4,125 | 4,375 ms | 61 tokens/s | 1,105 MB |

Task scores for the same model: grounded QA 1.00, JSON extraction 0.97, tool calling 0.60, short reasoning 0.38 (8 to 10 items per task, so one item is worth 0.10 to 0.13).

**Cost of verifying several tokens in one pass (Qwen 2.5 3B, MLX 4-bit)**

| Tokens in the pass | 1 | 2 | 5 | 9 | 17 |
|---|---:|---:|---:|---:|---:|
| Time vs. one token | 1.0× | 1.04× | 1.8× | 3.1× | 4.1× |

This is why the Accelerator's guess length is cost-aware rather than fixed.

**Two things already learned**

- The memory estimate (weights + KV cache) is too low for long prompts: 0.63 GB estimated vs. 1.1 GB measured at 4K tokens. Prefill needs working memory the estimate does not model yet.
- In early runs, prompt lookup roughly doubled decode speed on the text-editing task with identical output, and did not help on free-form reasoning. Those runs were disturbed by memory pressure, so the exact figures are not reported here.

## Development

```bash
uv sync --all-extras
uv run pytest            # ~190 tests, a couple of seconds, no model downloads
uv run ruff check . && uv run ruff format --check .
```

Tests use a fake backend with a virtual clock and a stand-in API client, so the whole suite runs without models, a GPU or network access.

```
src/edgewise/
  cli.py  config.py  schema.py  hardware.py  catalog.py  store.py
  backends/     mlx, cloud, fake
  bench/        runner, metrics
  evals/        tasks, scorers            (data in evals/data/)
  accelerator/  engine, mlx_model
  advisor/      recommend
  router/       core, rules, evaluate, build
  server/       app
  report/       build, svg
catalog/models.toml   scripts/   docs/
```

Design notes: [technical spec](docs/SPEC.md) · [architecture and flow diagrams](docs/architecture.html) (open locally in a browser).

## Known gaps

- Only one of the ten catalog models has been benchmarked; Advisor and Router results depend on the rest.
- Eval sets are small seed sets; they need to grow before score differences mean much.
- The memory estimate needs a prefill term (see above).
- MLX is the only local backend. llama.cpp (GGUF) is planned.
- The Router's streaming sends the answer after the routing decision, not token by token.

## License

[MIT](LICENSE)
