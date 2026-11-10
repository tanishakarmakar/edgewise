# Edgewise

**Run the right LLM, the right way, on the hardware you have.**

Edgewise is a toolkit for running large language models on everyday devices (laptops, edge boxes, in-vehicle computers) and using the cloud only when it is actually needed.

> 🚧 **Status: early development.** The CLI skeleton and hardware detection work today. Benchmarks and results will be published here as each milestone lands. No numbers in this README are made up: if a result isn't here yet, it hasn't been measured yet.

## Why

Running every request through a large cloud model is slow, expensive, and impossible offline. Small local models are fast and private, but:

- **Which version should I run?** Every model ships in many quantized variants (Q4, Q5, Q8, MLX 4-bit…). Perplexity doesn't tell you whether a variant still does *your* task (tool calls, JSON, RAG answers) well.
- **Can it be faster?** Token-by-token generation is memory-bound; speculative decoding can speed it up, but how much depends on the task.
- **When is local not enough?** Some requests really need a bigger model. Knowing *which ones* is the hard part.

## What it does

| Module | Question it answers | Command |
|---|---|---|
| **Advisor** | Which quantized model gives the best quality per GB and per second on this device, for this task? | `edgewise advise` |
| **Accelerator** | How much faster can generation get with speculative and prompt-lookup decoding, and when does it hurt? | `edgewise bench --speculative` |
| **Router** | Which requests can stay local, and which should escalate to the cloud? (offline-aware, privacy-aware, cost-tracked) | `edgewise route` |

All three share one **task-based eval suite**, so every claim is backed by a measurement.

📐 Design: [technical spec](docs/SPEC.md) · [architecture & flow diagrams](docs/architecture.html) (open locally in a browser)

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tanishakarmakar/edgewise.git
cd edgewise
uv sync
uv run edgewise info
```

Optional backends:

```bash
uv sync --extra mlx        # Apple Silicon (MLX)
uv sync --extra cloud      # cloud fallback for the router
```

llama.cpp (GGUF models) support is coming in milestone 1.

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| 0. Foundation | Project setup, CLI, hardware detection | ✅ Done |
| 1. Bench | Run models via MLX and llama.cpp; measure TTFT, tokens/s, peak memory; task eval sets | ⏳ Next |
| 2. Advisor | Benchmark quantized variants on real tasks; quality-vs-speed Pareto chart; `advise` command | Planned |
| 3. Accelerator | Speculative decoding, prompt-lookup decoding, adaptive speculation length | Planned |
| 4. Router | Local/cloud routing, cascades, offline mode, privacy rules, cost tracking | Planned |
| 5. Write-up | Results, charts, demo GIFs, blog post | Planned |

## Reference device

Development and benchmarks run on an **8 GB Apple Silicon MacBook**, a deliberately constrained device. If it works well here, it works on most laptops.

## License

[MIT](LICENSE)
