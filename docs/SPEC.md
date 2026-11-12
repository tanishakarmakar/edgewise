# Edgewise: Technical Specification

| | |
|---|---|
| **Status** | v0.2: implemented; see §17 for where the build differs from this design |
| **Author** | Tanisha Karmakar |
| **Last updated** | 2026-11-12 |
| **Flow diagrams** | [`architecture.html`](architecture.html) (open in a browser) |

---

## 1. Summary

Edgewise is a toolkit for running LLMs on everyday hardware. It answers three questions, each with measured evidence:

1. **Advisor**: *Which model variant should I run on this device for this task?*
2. **Accelerator**: *How do I make it generate faster without changing its output?*
3. **Router**: *Which requests can be answered locally, and which need a cloud model?*

All three are built on one shared foundation: a **backend abstraction** (MLX, llama.cpp, cloud), a **task-based eval suite**, a **benchmark runner**, and a **results store**.

The reference device is an **8 GB Apple Silicon MacBook**, a deliberately constrained target.

---

## 2. Goals and non-goals

### Goals
- **G1. Measured, reproducible claims.** Every number in the README comes from a run that can be repeated with one command, and every run records the hardware, software versions and git commit.
- **G2. Task quality, not just perplexity.** Model variants are judged on the tasks people actually use them for: tool calls, structured output, grounded QA, short reasoning.
- **G3. Works on 8 GB RAM.** All features run on the reference device with models of 3B parameters or fewer at 4-bit.
- **G4. Lossless acceleration.** Speculative decoding under greedy decoding must produce *exactly* the same tokens as normal decoding. This is verified by tests.
- **G5. Local-first routing.** The router keeps requests local by default, escalates only when needed, and never sends requests marked private to the cloud.
- **G6. Usable as a tool.** A clean CLI plus an OpenAI-compatible local HTTP endpoint, so other apps can use Edgewise without code changes.

### Non-goals (v1)
- Training or fine-tuning models.
- Writing custom GPU kernels or a new inference engine. Edgewise orchestrates and measures existing engines (MLX, llama.cpp) and implements decoding *algorithms* on top of them.
- Multi-GPU or datacenter serving.
- Multimodal (image/audio) models.
- Production multi-tenant deployment (auth, quotas, horizontal scaling).

---

## 3. Constraints

| Constraint | Value | Consequence |
|---|---|---|
| RAM | 8 GB unified memory | Model weights + KV cache must fit in about 4.5 GB (see §6.3). One model pair loaded at a time. |
| Disk | 256 GB SSD | Model cache capped at about 25 GB; old variants evicted manually via `edgewise models rm`. |
| Compute | Apple Silicon (A18 Pro / M-series) | MLX is the primary backend; llama.cpp (Metal build) is secondary. |
| No NVIDIA GPU | | vLLM, TensorRT-LLM, CUDA kernels are out of scope. |
| Network | Slow/unreliable at times | Downloads are resumable and cached; the router must handle offline mode. |
| Budget | Personal | Cloud calls are capped per run by a configurable spend limit. |

---

## 4. Architecture overview

Edgewise is a layered system. Higher layers only depend on lower ones.

```
┌──────────────────────────────────────────────────────────────────────┐
│ Interfaces     CLI (typer + rich)        Local API (FastAPI, OpenAI- │
│                                          compatible /v1/chat/...)    │
├──────────────────────────────────────────────────────────────────────┤
│ Features       Advisor           Accelerator           Router        │
├──────────────────────────────────────────────────────────────────────┤
│ Foundation     Bench runner · Eval suite · Model catalog ·           │
│                Hardware profiler · Results store · Config            │
├──────────────────────────────────────────────────────────────────────┤
│ Backends       MLX (in-process)   llama.cpp (llama-server over HTTP) │
│                Cloud (Anthropic API)        Fake (tests)             │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.1 Component summary

| Component | Responsibility | Depends on |
|---|---|---|
| **Hardware profiler** | Detect chip, RAM, free disk; compute the memory budget for models | — |
| **Config** | Load `edgewise.toml` (paths, spend limits, price table, router policy) | — |
| **Model catalog** | Describe every model variant (family, size, quantization, backend, file location, architecture numbers needed for memory estimates) | Config |
| **Backends** | Load a model, generate tokens (streaming), report timings, token log-probs and memory | Catalog |
| **Eval suite** | Task datasets + scorers that turn a model output into a quality score | — |
| **Bench runner** | Run (model × task × settings), collect performance + quality, write results | Backends, Eval suite, Results store |
| **Results store** | Persist runs, measurements and scores (DuckDB) | — |
| **Advisor** | Filter variants that fit the device, compute the quality/speed Pareto frontier, recommend | Results store, Catalog, Hardware |
| **Accelerator** | Speculative, prompt-lookup and adaptive decoding on top of a backend | Backends |
| **Router** | Decide local vs. cloud per request; cascade; offline and privacy rules; cost tracking | Backends, Eval suite scorers, Config |
| **Reporting** | Charts and tables (Plotly) from the results store; `results/published/` snapshots | Results store |
| **Interfaces** | CLI commands and the local HTTP API | Everything above |

---

## 5. Core data model

Defined as Pydantic models in `edgewise/schema.py`. These are the contracts between components.

```python
class ModelVariant(BaseModel):
    id: str                       # "qwen2.5-1.5b-instruct@mlx-4bit"
    family: str                   # "qwen2.5"
    params_b: float               # 1.5
    quant: str                    # "fp16" | "q8_0" | "q5_k_m" | "q4_k_m" | "mlx-4bit" | ...
    bits_per_weight: float        # effective, used for memory estimates
    backend: Literal["mlx", "llamacpp", "cloud"]
    source: str                   # Hugging Face repo id or local path
    tokenizer_family: str         # draft/target pairs must match
    n_layers: int
    n_kv_heads: int
    head_dim: int
    context_max: int

class Task(BaseModel):
    id: str                       # "tool_calling"
    description: str
    scorer: str                   # registry key, e.g. "tool_call_match"
    items_path: Path              # JSONL file

class EvalItem(BaseModel):
    id: str
    task_id: str
    messages: list[Message]
    tools: list[ToolSpec] | None = None
    json_schema: dict | None = None
    expected: Any                 # gold answer / gold tool call / reference text
    tags: list[str] = []          # e.g. ["long_context", "contains_pii"]

class GenerationSettings(BaseModel):
    max_tokens: int = 256
    temperature: float = 0.0      # greedy by default for reproducibility
    seed: int = 0
    decoding: Literal["standard", "speculative", "prompt_lookup", "adaptive"] = "standard"
    draft_model_id: str | None = None
    num_draft_tokens: int = 4

class GenerationResult(BaseModel):
    text: str
    tokens: list[int]
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float                # time to first token
    decode_tps: float             # completion tokens / decode time
    prefill_tps: float
    itl_ms: list[float]           # inter-token latencies
    peak_memory_mb: float
    mean_logprob: float | None    # used by the router's confidence signal
    accepted_draft_tokens: int | None = None
    proposed_draft_tokens: int | None = None

class RunManifest(BaseModel):
    run_id: str
    started_at: datetime
    git_sha: str
    edgewise_version: str
    hardware: HardwareInfo
    backend_versions: dict[str, str]   # {"mlx": "...", "mlx-lm": "...", "llama.cpp": "b...."}
    on_ac_power: bool
    notes: str = ""
```

---

## 6. Foundation

### 6.1 Backends

All backends implement one protocol:

```python
class Backend(Protocol):
    name: str
    def load(self, variant: ModelVariant) -> LoadedModel: ...
    def generate(self, model: LoadedModel, messages: list[Message],
                 settings: GenerationSettings) -> Iterator[TokenEvent]: ...
    def unload(self, model: LoadedModel) -> None: ...
    def versions(self) -> dict[str, str]: ...
```

`generate` streams `TokenEvent(token_id, text, logprob, t_ns)`. The bench runner builds the `GenerationResult` from the stream, so **timing is measured the same way for every backend**.

| Backend | How | Notes |
|---|---|---|
| **MLX** | In-process via `mlx-lm` (`load`, `stream_generate`) | Primary backend. Gives token-level control needed by the Accelerator. Peak memory from MLX's memory API. |
| **llama.cpp** | Prebuilt `llama-server` binary, called over HTTP (`/completion` with streaming) | No compiling Python bindings (see commit `50c41d5`). Server timing fields cross-check our own timings. Memory = server process RSS. |
| **Cloud** | Anthropic SDK | Used by the Router and as an LLM judge. Usage tokens → cost via the price table in config. Never used for local benchmarks. |
| **Fake** | Deterministic scripted tokens with configurable delays | Makes all logic unit-testable without models. |

### 6.2 Model catalog

- File: `catalog/models.toml`, version-controlled.
- Initial candidate set (all ≤ 3B so they fit on the reference device):

| Family | Sizes | Variants |
|---|---|---|
| Qwen 2.5 Instruct | 0.5B, 1.5B, 3B | MLX 4-bit, MLX 8-bit; GGUF Q4_K_M, Q5_K_M, Q8_0 |
| Llama 3.2 Instruct | 1B, 3B | MLX 4-bit, MLX 8-bit; GGUF Q4_K_M, Q8_0 |

  The 0.5B/1B models double as **draft models** for speculative decoding (same tokenizer family as the larger models). The list is a starting point and may change after milestone 1 if newer small models perform better.
- `edgewise models pull <id>` downloads and caches with resume. `edgewise models ls` shows size on disk and whether each variant fits in memory.

### 6.3 Memory model

Used by the Advisor to rule out variants that won't fit before running anything.

```
weights_gb   ≈ params_b × bits_per_weight / 8
kv_cache_gb  ≈ 2 × n_layers × n_kv_heads × head_dim × context_tokens × kv_bytes / 1e9
overhead_gb  ≈ 0.3                      # runtime buffers; calibrated in milestone 1
required_gb  = weights_gb + kv_cache_gb + overhead_gb (+ draft model if speculative)

fits  ⇔  required_gb ≤ model_budget_gb = RAM − system_reserve (3.5 GB default)
```

Milestone 1 compares this estimate with measured peak memory and records the error. Variants within 10% of the budget are marked "tight".

### 6.4 Eval suite

Each task is a JSONL file of `EvalItem`s plus a scorer. Datasets are small (50–100 items per task) so a full run fits in minutes on the reference device, and are drawn from public sources with compatible licenses (sources and licenses listed in `evals/README.md`).

| Task | What it tests | Scorer | Score |
|---|---|---|---|
| `tool_calling` | Pick the right function and arguments from a tool list | Parse the call → compare function name + arguments (order-insensitive, type-normalized) | 1 / 0 per item; also report "valid call" rate |
| `json_extraction` | Fill a JSON schema from text | JSON parses + validates against schema + field-level match | Field F1 |
| `grounded_qa` | Answer from a provided passage (RAG-style) | Normalized exact match / token F1 vs. gold | F1 |
| `short_reasoning` | Grade-school math word problems | Extract final number, exact match | Accuracy |
| `chat` | Open-ended helpfulness | LLM judge (cloud) with a fixed rubric; judge agreement checked on a 30-item hand-labeled subset | 1–5 |
| `long_context` | Find a fact in a long prompt (2K–8K tokens) | Exact match | Accuracy; also stresses KV memory |

Rules:
- Greedy decoding (`temperature=0`) for all quality runs.
- **Relative quality** = variant score ÷ score of the best-precision variant of the same model (e.g. Q4 vs. 8-bit). This is the main number the Advisor uses.
- The LLM judge is only used where no automatic scorer exists, and its cost counts against the spend limit.

### 6.5 Bench runner

**Command:** `edgewise bench --model <id> [--task <id>...] [--decoding speculative --draft <id>]`

**Procedure per (model, settings):**
1. Write the `RunManifest` (hardware, versions, git sha, AC-power status).
2. Load the model and record load time and memory after load.
3. **Warm-up:** 3 generations, discarded.
4. **Performance pass:** a fixed set of prompts at three prompt lengths (128 / 1K / 4K tokens), each repeated 5×. Record TTFT, prefill tokens/s, decode tokens/s, inter-token latency (P50/P95), peak memory.
5. **Quality pass:** every item of the selected tasks, once (greedy is deterministic).
6. Unload; write everything to the results store.

**Measurement hygiene** (documented in the README and checked where possible):
- Plugged into power; the runner warns if on battery.
- A 10-second cool-down between models to reduce thermal throttling effects; report the spread across repeats, not only the median.
- Other heavy apps closed; the runner warns if free memory at start is below the model's estimate.

**Metrics glossary:**

| Metric | Definition |
|---|---|
| TTFT | Time from request start to the first generated token |
| Prefill tokens/s | Prompt tokens ÷ time to process the prompt |
| Decode tokens/s | Generated tokens (after the first) ÷ time spent generating them |
| ITL P50/P95 | Median / 95th-percentile time between consecutive generated tokens |
| Peak memory | Highest memory used during the run (MLX memory API or process RSS) |
| Relative quality | Score ÷ best-precision variant's score on the same task |

### 6.6 Results store

DuckDB file at `results/edgewise.duckdb` (git-ignored). Curated snapshots exported to `results/published/` (committed) for the README.

```
runs(run_id PK, started_at, git_sha, version, hardware JSON, backend_versions JSON, on_ac_power)
perf(run_id, model_id, decoding, draft_model_id, prompt_len, repeat,
     ttft_ms, prefill_tps, decode_tps, itl_p50_ms, itl_p95_ms, peak_memory_mb,
     accepted_draft_tokens, proposed_draft_tokens)
quality(run_id, model_id, task_id, item_id, score, valid_output, latency_ms, output_text)
route_log(request_id, ts, policy, decision, reason, local_model_id, cloud_model_id,
          confidence, escalated, prompt_tokens, completion_tokens, cost_usd, latency_ms, quality_score)
```

---

## 7. Feature: Advisor

**Command:** `edgewise advise --task tool_calling [--memory 4GB] [--min-quality 0.9] [--prefer speed|quality|balanced]`

**Pipeline:**
1. **Candidates:** all catalog variants.
2. **Feasibility filter:** drop variants whose estimated `required_gb` (§6.3) exceeds the memory budget (device budget by default, or `--memory`).
3. **Evidence lookup:** read measured quality and performance from the results store. Variants with no measurements are listed as "not benchmarked" with the command to benchmark them; they are never ranked on guesses.
4. **Constraint filter:** drop variants below `--min-quality` (relative quality on the chosen task).
5. **Pareto frontier:** keep variants that no other variant beats on *both* quality and decode speed.
6. **Pick:** from the frontier, choose by preference:
   - `speed`: fastest variant meeting the quality floor
   - `quality`: highest quality that fits
   - `balanced` (default): maximize `quality_rel × normalized_speed`
7. **Explain:** print the choice, the runner-up and *why* (e.g. "Q4 is 1.6× faster but drops tool-call accuracy from 0.92 to 0.71").

**Outputs:** a Rich table in the terminal, `--json` for scripts, and a Pareto chart (`edgewise report advisor`) for the README.

**Key questions the Advisor's results should answer** (findings for the write-up):
- Does 4-bit hurt structured tasks (tool calls, JSON) more than chat?
- Is a bigger model at 4-bit better than a smaller model at 8-bit for the same memory?
- How accurate is the memory estimate vs. measured peak memory?

---

## 8. Feature: Accelerator

Speeds up generation by verifying several tokens per forward pass of the main ("target") model. Implemented on the **MLX backend**, where Edgewise controls the decoding loop.

### 8.1 Methods

| Method | How draft tokens are produced | Best for |
|---|---|---|
| **Standard** | — (baseline) | — |
| **Speculative (draft model)** | A small model of the same tokenizer family (e.g. 0.5B for a 1.5B/3B target) proposes `k` tokens | General text |
| **Prompt lookup** | Find the last n-gram of the output in the prompt; propose the tokens that followed it there | RAG, summarization, code editing (output copies input) |
| **Adaptive** | Either of the above, but `k` adjusts per step based on recent acceptance rate | Mixed workloads |

### 8.2 Algorithm (greedy verification)

```
loop until max_tokens or EOS:
    draft = propose(k)                         # k tokens from draft model or prompt lookup
    logits = target.forward(context + draft)   # ONE forward pass scores all k positions
    accept the longest prefix of draft where draft[i] == argmax(logits[i])
    append one extra token from the target at the first mismatch (or after all k)
    roll back draft/target KV caches to the accepted length
```

Because every kept token equals the target model's own greedy choice, **the output is identical to standard decoding**. This is enforced by a test that compares outputs token-for-token on all eval prompts (G4). Sampling-based verification (temperature > 0) is a stretch goal.

### 8.3 Adaptive `k`

- Track the acceptance rate over the last N steps (exponential moving average).
- If acceptance is high, increase `k` (up to 8); if low, decrease it (down to 1). If acceptance stays near zero, fall back to standard decoding for a cooldown period.
- The policy parameters are configurable and tuned on a held-out prompt set, not on the reported benchmark prompts.

### 8.4 Measurements
- Speedup = decode tokens/s ÷ standard decode tokens/s, **per task**.
- Acceptance rate, mean accepted tokens per step.
- Extra memory used by the draft model (it must still fit in the budget).
- Comparison with mlx-lm's built-in speculative decoding as a sanity check: our implementation should be in the same range.

---

## 9. Feature: Router

Decides, per request, whether to answer with the local model or a cloud model.

### 9.1 Request flow

```
request
  → hard rules
       • private=true or PII detected   → LOCAL ONLY (never cloud)
       • offline                         → LOCAL ONLY
       • forced cloud by caller          → CLOUD
  → policy (configurable)
       • local-only | cloud-only         (baselines)
       • classifier: predict "local will succeed" before generating
       • cascade:    generate locally → check confidence → escalate if low
  → response + route_log entry (decision, reason, cost, latency)
```

### 9.2 Signals

| Signal | Source | Used by |
|---|---|---|
| Prompt features | Length, task type, presence of tools/schema, embedding of the prompt | Classifier |
| Mean token log-prob | Local generation (`GenerationResult.mean_logprob`) | Cascade |
| Output validity | JSON parses / tool call valid against schema | Cascade |
| PII detection | Regex patterns (emails, phone numbers, IDs) + optional small NER model | Hard rules |
| Connectivity | Cached reachability check of the cloud endpoint | Hard rules |

### 9.3 Classifier training
- Labels come from the eval suite: for each item, *did the local model reach the cloud model's score?*
- Model: logistic regression on prompt embeddings (small local embedding model) + simple features. Trained with scikit-learn, saved to `models/router.joblib`.
- Train/test split by item, so reported routing results are on items the classifier never saw.

### 9.4 Cost tracking
- Price per million input/output tokens configured in `edgewise.toml` (not hard-coded, since prices change).
- Every cloud call logs tokens and cost. A per-run spend limit stops the run when exceeded.

### 9.5 Evaluation
Compare policies on the same held-out items:

| Policy | Metrics |
|---|---|
| Local-only | quality, latency, cost = 0 |
| Cloud-only | quality, latency, cost |
| Classifier | % local, quality vs. cloud-only, cost saved, latency |
| Cascade | same, plus extra latency from escalated requests |

Result: a **cost vs. quality curve** across confidence thresholds, and the chosen operating point.

---

## 10. Interfaces

### 10.1 CLI

| Command | Purpose | Milestone |
|---|---|---|
| `edgewise info` | Device hardware and memory budget | ✅ 0 |
| `edgewise models ls / pull / rm` | Manage the local model cache | 1 |
| `edgewise bench` | Benchmark performance + quality | 1 |
| `edgewise advise` | Recommend a variant | 2 |
| `edgewise bench --decoding speculative\|prompt_lookup\|adaptive` | Benchmark acceleration | 3 |
| `edgewise route "<prompt>"` | Route one request and show the decision | 4 |
| `edgewise serve` | Start the local OpenAI-compatible API | 4 |
| `edgewise report <advisor\|accelerator\|router>` | Generate charts/tables from results | 2–5 |

All commands support `--json` for machine-readable output.

### 10.2 Local API (`edgewise serve`)
- FastAPI app on `127.0.0.1:8080` (localhost only).
- `POST /v1/chat/completions`: OpenAI-compatible request/response, streaming via SSE. The router chooses the backend. Extra response fields under `edgewise`: `{route, reason, model_id, cost_usd}`.
- `GET /v1/models`, `GET /healthz`, `GET /metrics` (request counts, % local, cost so far).

---

## 11. Configuration

`edgewise.toml` (user-level at `~/.config/edgewise/`, overridable per project):

```toml
[paths]
model_cache = "~/.cache/edgewise/models"
results_db  = "results/edgewise.duckdb"

[hardware]
system_reserve_gb = 3.5

[cloud]
provider   = "anthropic"
model      = "<cloud model id>"
spend_limit_usd_per_run = 2.00
# API key read from the ANTHROPIC_API_KEY environment variable, never stored in config

[cloud.prices_per_mtok]
input  = 0.0   # fill in from the provider's current price page
output = 0.0

[router]
policy = "cascade"            # local-only | cloud-only | classifier | cascade
local_model = "qwen2.5-1.5b-instruct@mlx-4bit"
cascade_min_mean_logprob = -0.8
```

---

## 12. Repository layout

```
edgewise/
├── src/edgewise/
│   ├── cli.py                # typer app
│   ├── config.py
│   ├── schema.py             # Pydantic contracts (§5)
│   ├── hardware.py           # ✅ exists
│   ├── catalog.py
│   ├── backends/   {base.py, mlx.py, llamacpp.py, cloud.py, fake.py}
│   ├── evals/      {registry.py, scorers.py, judge.py}
│   ├── bench/      {runner.py, metrics.py}
│   ├── store.py              # DuckDB access
│   ├── advisor/    {feasibility.py, pareto.py, recommend.py}
│   ├── accelerator/{speculative.py, lookup.py, adaptive.py}
│   ├── router/     {rules.py, pii.py, classifier.py, cascade.py, cost.py}
│   ├── server/     {app.py}
│   └── report/     {charts.py}
├── catalog/models.toml
├── evals/data/<task>.jsonl
├── results/published/        # committed snapshots for the README
├── docs/{SPEC.md, architecture.html}
└── tests/  {unit/, integration/}   # integration = needs real models, marked slow
```

---

## 13. Testing strategy

| Level | What | How |
|---|---|---|
| Unit | Scorers, memory model, Pareto logic, routing rules, PII detection, cost math, adaptive-k policy | pytest + `FakeBackend`; run on every commit |
| Correctness | Speculative/prompt-lookup output == standard output (greedy), token for token | Integration test on a 0.5B model; marked `slow` |
| Integration | Each backend loads a tiny model and streams tokens | `pytest -m slow`, run manually / before releases |
| Regression | Published results reproduce within tolerance (±10% perf, exact quality for greedy) | `edgewise bench --check results/published/...` |
| Static | Lint + format + type check | ruff, pyright |
| CI | Unit tests + lint on GitHub Actions (macOS runner) | No model downloads in CI |

---

## 14. Milestones and acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| 0 | Foundation | ✅ CLI skeleton, `edgewise info`, repo, license |
| 1 | Bench | MLX + llama.cpp backends stream tokens; `bench` writes perf + quality for ≥ 4 variants; memory-estimate error measured; 3 tasks with scorers |
| 2 | Advisor | All catalog variants benchmarked on all tasks; `advise` returns a recommendation with explanation; Pareto chart in README |
| 3 | Accelerator | Speculative + prompt lookup + adaptive implemented; lossless test passes; per-task speedup table in README |
| 4 | Router | Rules + classifier + cascade; `route` and `serve` work; cost-vs-quality curve; offline demo |
| 5 | Write-up | README with GIFs, published results, blog post with findings and limitations |

**Status on 2026-11-12.** The code for milestones 1 to 4 and the report generator exists and is unit-tested (fake backends, no models needed). Still open: benchmarking the whole catalog (1 of 10 variants done), calibrating the memory model, exercising the Router and local API on real models, the llama.cpp backend, the routing classifier, larger eval sets, and the write-up.

---

## 15. Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| 8 GB is too tight for 3B + draft model + OS | Some combos can't run | Memory model rules them out up front; document as a finding |
| Thermal throttling skews benchmarks | Noisy numbers | Warm-up, cool-down, repeats, report spread |
| Slow network for model downloads | Lost time | Resumable cached downloads; start with the smallest variants |
| Small eval sets → noisy quality scores | Wrong conclusions | Report confidence intervals (bootstrap); avoid claiming tiny differences |
| LLM-judge bias | Unreliable chat scores | Use only where needed; check agreement vs. hand labels |
| Library APIs change (mlx-lm, llama.cpp) | Breakage | Pin versions in `uv.lock` and record them in each run manifest |
| Cloud cost overruns | Money | Per-run spend limit; cache cloud responses by prompt hash |

**Open questions** (to settle in milestone 1):
1. Exact eval datasets and licenses per task.
2. Whether llama.cpp's server exposes enough timing detail, or we measure only from the client side.
3. Which cloud model to use as router fallback and as judge (choose from current models at the time, cost vs. quality).

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **Quantization** | Storing model weights in fewer bits (e.g. 4 instead of 16) to save memory and speed up inference, at some quality cost |
| **GGUF** | llama.cpp's model file format; names like Q4_K_M describe the quantization scheme |
| **MLX** | Apple's array framework for Apple Silicon; `mlx-lm` runs LLMs on it |
| **KV cache** | Stored attention keys/values for past tokens, so they aren't recomputed; grows with context length |
| **Prefill / decode** | Processing the prompt (parallel, compute-bound) vs. generating tokens one by one (memory-bound) |
| **Speculative decoding** | A fast draft proposes tokens; the main model verifies several at once |
| **Pareto frontier** | The set of options where you can't improve one metric without worsening another |
| **Cascade** | Try the cheap model first; escalate to the expensive one only when unsure |

---

## 17. Implementation notes: where the build differs from this design

Sections 1 to 16 are the original design. These are the places the implementation departed from it, and why.

| Area | Design | Built | Why |
|---|---|---|---|
| Decoding modes (§5, §8) | `decoding` = standard / speculative / prompt_lookup / adaptive | `decoding` = standard / speculative / prompt_lookup, plus a separate `k_policy` = fixed / heuristic / cost_aware | "Adaptive" is a property of how many tokens to guess, not a fourth way of guessing; it applies to both proposers. |
| Guess-length policy (§8.3) | Raise or lower `k` from the recent acceptance rate | **Cost-aware**: measure the cost of a forward pass over 1 to 17 tokens per model, estimate the per-token acceptance probability online, and choose the `k` with the best expected tokens per millisecond (possibly 0). The +2/−1 heuristic is kept as a baseline. | Measured on the reference device, a pass over 9 tokens costs about 3× a pass over 1. A rule that ignores this cost makes decoding slower whenever acceptance is moderate. |
| Speed of accelerated decoding (§6.5, §8.4) | Compare on the raw speed pass | Compare on the eval task prompts; the raw speed pass is skipped for accelerated modes | The raw pass uses repeated filler text, which prompt lookup can copy almost entirely. That would overstate its benefit. |
| Memory pinning (§8) | Not considered | The accelerated loop pins model memory, as mlx-lm's own loop does | Without it, macOS pages weights out under memory pressure and single passes stall for seconds. |
| Eval tasks (§6.4) | 6 tasks incl. `chat` (LLM judge) and `long_context` | 5 tasks: the first four plus `text_edit`; no `chat` or `long_context` yet | `text_edit` is prompt lookup's natural workload and needs no judge. The judged task was deferred to keep every score automatic and free. |
| QA scoring (§6.4) | Token F1 | Full credit if the answer contains an acceptable answer, else token F1 | Plain F1 marked correct full-sentence answers down for their extra words. |
| Advisor quality floor (§7) | `--min-quality` on score relative to the best-precision variant | `--min-score` on the absolute task score; the relative score is shown as a column | An absolute floor is what a user actually needs ("at least 0.8 on tool calling"); the relative number still shows what quantization cost. |
| Advisor memory check (§7) | Estimate only | Measured peak when the variant has been benchmarked, estimate otherwise | The estimate is known to be low for long prompts (§6.3). |
| Router escalation tier (§9) | Cloud model | Any backend: the Claude API **or** a larger on-device model | Lets the cascade be built, tested and demonstrated fully offline, and makes "small model first, bigger model if unsure" a single mechanism. |
| Router policies (§9.1) | local-only, cloud-only, classifier, cascade | local-only, remote-only, cascade. **No classifier yet.** | A classifier trained on 84 eval items would be noise. It needs larger eval sets first. |
| Cascade evaluation (§9.5) | Run each policy live on held-out items | Replay the cascade at every threshold from stored benchmark results (`evaluate_cascade`) | Costs nothing to run and makes the threshold a measured choice. With at least 10 items per half, the threshold is chosen on one half and reported on the other; with fewer, the report says the figure is an upper bound. |
| Router failure handling (§9) | Not specified | If escalation fails (network, credentials, rate limit, refusal), return the local answer with the reason | An answer the user can judge is better than an error. |
| Streaming API (§10.2) | Token streaming via SSE | SSE in the OpenAI format, sent after the routing decision | The cascade must see the whole local answer before deciding whether to keep it. |
| API thread model (§10.2) | Not specified | Models are loaded and run on one worker thread | MLX state belongs to the thread that created it; one generation at a time is also all an 8 GB device can do. |
| Results schema (§6.6) | As listed | Adds decoding configuration, per-item timings, mean log-probability, draft acceptance, power source and swap usage; the route log stores no prompt or answer text | Needed for per-task speed-ups, the cascade replay, and for flagging results measured on battery or under memory pressure. |
| llama.cpp backend (§6.1) | Milestone 1 | Not built | `llama-cpp-python` ships as source only (a long compile); the plan is to drive the prebuilt `llama-server` over HTTP. MLX covers the reference device. |
| Charts (§4, §7) | Plotly | Hand-written SVG | No plotting dependency, renders in a GitHub README, follows light/dark mode. |
| Run hygiene (§6.5) | Warn on battery | Also warn, and tag the run, when the system is already using more than 1 GB of swap | Results measured while the machine swaps are not comparable; found the hard way on the 8 GB reference device. |

### Lessons recorded during the build

- **Verification is not free on this hardware.** The usual assumption behind speculative decoding (checking `k` tokens costs about the same as generating one) does not hold on the A18 Pro: cost grows to roughly 1.8× at 5 tokens and 3× at 9. The best guess length is therefore small unless guesses are almost always right.
- **The memory model needs a prefill term.** Weights + KV cache underestimates peak memory for long prompts (0.63 GB estimated vs. 1.1 GB measured at 4K tokens on the 0.5B model).
- **A project under an iCloud-synced folder breaks editable installs.** iCloud marks files inside dot-folders as hidden, and Python 3.12 skips hidden `.pth` files, so the package intermittently became unimportable. The virtual environment now lives outside the synced folder.
