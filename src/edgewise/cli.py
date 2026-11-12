"""Command-line entry point: `edgewise <command>`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from edgewise import __version__
from edgewise.catalog import Catalog, estimate_memory
from edgewise.hardware import detect

app = typer.Typer(
    help="Run the right LLM, the right way, on the hardware you have.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Manage the model catalog and local downloads.", no_args_is_help=True)
app.add_typer(models_app, name="models")
console = Console()

FIT_STYLE = {"fits": "green", "tight": "yellow", "too big": "red"}


@app.command()
def version() -> None:
    """Show the installed version."""
    console.print(f"edgewise {__version__}")


@app.command()
def info() -> None:
    """Show this device's hardware and how much memory is available for models."""
    hw = detect()
    table = Table(title="Device", show_header=False, title_justify="left")
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Chip", hw.chip)
    table.add_row("OS / arch", f"{hw.os} / {hw.machine}")
    table.add_row("CPU cores", str(hw.cpu_cores))
    table.add_row("RAM", f"{hw.ram_gb} GB")
    table.add_row("Model memory budget", f"~{hw.model_budget_gb} GB")
    table.add_row("Free disk", f"{hw.disk_free_gb} GB")
    table.add_row("Python", hw.python)
    table.add_row("MLX backend", "supported" if hw.apple_silicon else "not available")
    console.print(table)


# ------------------------------------------------------------------ models


def _downloaded_sizes() -> dict[str, float]:
    """Hugging Face repo id -> GB on disk, for repos already in the local cache."""
    try:
        from huggingface_hub import scan_cache_dir

        return {r.repo_id: r.size_on_disk / 1e9 for r in scan_cache_dir().repos}
    except Exception:  # no cache yet, or an unreadable one
        return {}


@models_app.command("ls")
def models_ls(
    context: int = typer.Option(4096, help="Context length used for the memory estimate."),
) -> None:
    """List catalog models, estimated memory, and whether each fits on this device."""
    budget = detect().model_budget_gb
    downloaded = _downloaded_sizes()
    table = Table(title=f"Models (estimated memory at {context} tokens, budget ~{budget} GB)")
    table.add_column("Model", no_wrap=True)  # ids get copy-pasted into other commands
    # Quantization and backend are part of the id, so they get no columns of their own:
    # the table has to fit an 80-column terminal without cutting ids short.
    for col in ("Params", "Memory", "Fit", "On disk"):
        table.add_column(col, no_wrap=True)
    for v in Catalog.load():
        est = estimate_memory(v, context_tokens=context)
        fit = est.fit(budget)
        size = downloaded.get(v.source)
        table.add_row(
            v.id,
            f"{v.params_b:g}B",
            f"{est.total_gb:.2f} GB",
            f"[{FIT_STYLE[fit]}]{fit}[/]",
            f"{size:.2f} GB" if size is not None else "—",
        )
    console.print(table)


@models_app.command("pull")
def models_pull(model_id: str = typer.Argument(..., help="Catalog model id.")) -> None:
    """Download a model into the local Hugging Face cache."""
    from huggingface_hub import snapshot_download

    variant = Catalog.load().get(model_id)
    console.print(f"downloading [bold]{variant.source}[/bold] …")
    path = snapshot_download(
        variant.source, allow_patterns=[variant.file] if variant.file else None
    )
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
    console.print(f"[green]done[/green]: {size:.2f} GB at {path}")


# ------------------------------------------------------------------ bench


@app.command()
def bench(
    model: str = typer.Option(..., "--model", "-m", help="Catalog model id."),
    task: list[str] = typer.Option(
        [], "--task", "-t", help="Eval task(s) to score. Repeatable. Omit for performance only."
    ),
    all_tasks: bool = typer.Option(False, "--all-tasks", help="Score every eval task."),
    decoding: str = typer.Option(
        "standard", help="standard | speculative (needs --draft) | prompt_lookup"
    ),
    draft: str = typer.Option(None, help="Draft model id for speculative decoding."),
    draft_tokens: int = typer.Option(4, help="Tokens to guess per step (fixed policy)."),
    k_policy: str = typer.Option(
        "cost_aware", help="How many tokens to guess: fixed | heuristic | cost_aware"
    ),
    prompt_lengths: str = typer.Option(
        None,
        help="Comma-separated prompt lengths for the raw speed pass. "
        "Default: 128,1024,4096 for standard decoding, none for accelerated decoding.",
    ),
    repeats: int = typer.Option(5, help="Repeats per prompt length."),
    warmup: int = typer.Option(3, help="Warm-up generations (discarded)."),
    max_tokens: int = typer.Option(128, help="Tokens to generate in the performance pass."),
    cooldown: float = typer.Option(10.0, help="Seconds to wait after unloading."),
    force: bool = typer.Option(False, help="Run even if the memory estimate says it won't fit."),
    db: Path = typer.Option(Path("results/edgewise.duckdb"), help="Results database."),
    json_out: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Measure speed, memory and task quality of a model on this device."""
    from edgewise.backends import get_backend
    from edgewise.bench.runner import BenchConfig, BenchError, run_bench
    from edgewise.evals import TASKS
    from edgewise.store import Store

    catalog = Catalog.load()
    try:
        variant = catalog.get(model)
    except KeyError as e:
        console.print(f"[red]error:[/red] {e.args[0]}")
        raise typer.Exit(code=1) from e
    if prompt_lengths is None:
        prompt_lengths = "128,1024,4096" if decoding == "standard" else ""
    config = BenchConfig(
        tasks=list(TASKS) if all_tasks else task,
        prompt_lengths=[int(n) for n in prompt_lengths.split(",") if n.strip()],
        repeats=repeats,
        warmup=warmup,
        perf_max_tokens=max_tokens,
        cooldown_s=cooldown,
        force=force,
        decoding=decoding,
        num_draft_tokens=draft_tokens,
        k_policy=k_policy,
    )
    log = (lambda msg: None) if json_out else (lambda msg: console.print(f"[dim]{msg}[/dim]"))
    with Store(db) as store:
        try:
            report = run_bench(
                variant,
                get_backend(variant.backend),
                config,
                store,
                log,
                draft=catalog.get(draft) if draft else None,
            )
        except (BenchError, ValueError, KeyError) as e:
            console.print(f"[red]error:[/red] {e.args[0] if isinstance(e, KeyError) else e}")
            raise typer.Exit(code=1) from e

    if json_out:
        print(report.model_dump_json(indent=2))
        return
    _print_bench_report(report, db)


def _print_bench_report(report, db: Path) -> None:
    label = report.model_id
    if report.decoding != "standard":
        label += f" · {report.decoding}"
        if report.draft_model_id:
            label += f" (draft {report.draft_model_id})"

    if report.perf:
        # "\\[" stops Rich from reading the bracketed range as a style tag.
        perf = Table(title=f"{label}: speed (median \\[p25–p75])", title_justify="left")
        for col in ("Prompt", "TTFT ms", "Prefill tok/s", "Decode tok/s", "ITL p95 ms", "Peak MB"):
            perf.add_column(col, justify="right")
        for p in report.perf:
            perf.add_row(
                str(p.prompt_tokens),
                str(p.ttft_ms),
                str(p.prefill_tps),
                str(p.decode_tps),
                str(p.itl_p95_ms),
                f"{p.peak_memory_mb:.0f}",
            )
        console.print(perf)

    if report.quality:
        quality = Table(title=f"{label}: tasks", title_justify="left")
        columns = ["Task", "Items", "Score", "Valid", "Decode tok/s", "Latency", "Peak MB"]
        accelerated = any(q.acceptance_rate is not None for q in report.quality)
        if accelerated:
            columns.insert(5, "Guesses accepted")
        for col in columns:
            quality.add_column(col, justify="right")
        for q in report.quality:
            row = [
                q.task_id,
                str(q.n),
                f"{q.mean_score:.2f}",
                f"{q.valid_rate:.0%}",
                f"{q.decode_tps:.1f}",
                f"{q.mean_latency_ms:.0f} ms",
                f"{q.peak_memory_mb:.0f}",
            ]
            if accelerated:
                rate = q.acceptance_rate
                row.insert(5, f"{rate:.0%}" if rate is not None else "—")
            quality.add_row(*row)
        console.print(quality)

    console.print(
        f"load {report.load_s:.1f}s · memory after load {report.memory_after_load_mb:.0f} MB · "
        f"estimate {report.est_memory_gb:.2f} GB · run [bold]{report.run_id}[/bold] → {db}"
    )


def _parse_gb(text: str) -> float:
    """ "4", "4GB", "4.5 gb" or "512MB" -> gigabytes."""
    cleaned = text.strip().lower().replace(" ", "")
    if cleaned.endswith("mb"):
        return float(cleaned[:-2]) / 1000
    return float(cleaned.removesuffix("gb").removesuffix("g"))


STATUS_STYLE = {
    "recommended": "bold green",
    "runner-up": "green",
    "candidate": "",
    "dominated": "dim",
    "below quality floor": "yellow",
    "too big": "red",
    "not benchmarked": "dim",
}


@app.command()
def advise(
    task: str = typer.Option(..., "--task", "-t", help="Eval task the model will be used for."),
    memory: str = typer.Option(None, help="Memory budget, e.g. 4GB. Default: this device's."),
    min_score: float = typer.Option(0.0, help="Lowest acceptable task score (0-1)."),
    prefer: str = typer.Option("balanced", help="balanced | speed | quality"),
    db: Path = typer.Option(Path("results/edgewise.duckdb"), help="Results database."),
    json_out: bool = typer.Option(False, "--json", help="Print the recommendation as JSON."),
) -> None:
    """Recommend the best model variant for a task and memory budget, from measured results."""
    from edgewise.advisor import advise as run_advise
    from edgewise.evals import get_task
    from edgewise.store import Store

    try:
        get_task(task)
        budget = _parse_gb(memory) if memory else detect().model_budget_gb
        if prefer not in ("balanced", "speed", "quality"):
            raise ValueError(f"--prefer must be balanced, speed or quality, not {prefer!r}")
    except (KeyError, ValueError) as e:
        console.print(f"[red]error:[/red] {e.args[0]}")
        raise typer.Exit(code=1) from e

    with Store(db) as store:
        rec = run_advise(Catalog.load(), store, task, budget, min_score, prefer)

    if json_out:
        print(rec.model_dump_json(indent=2))
        return

    table = Table(
        title=f"Advice for {task} · budget {budget:.1f} GB · prefer {prefer}", title_justify="left"
    )
    table.add_column("Model", no_wrap=True)
    for col in ("Score", "vs hi-prec", "Tokens/s", "TTFT ms", "Memory GB"):
        table.add_column(col, justify="right")
    table.add_column("Status")
    for row in rec.rows:
        e = row.evidence
        style = STATUS_STYLE[row.status]
        table.add_row(
            row.model_id,
            f"{e.score:.2f}" if e else "—",
            f"{row.vs_full_precision:.0%}" if row.vs_full_precision is not None else "—",
            f"{e.decode_tps:.0f}" if e else "—",
            f"{e.ttft_ms:.0f}" if e else "—",
            f"{row.memory_gb:.2f}" + ("" if row.memory_measured else " (est.)"),
            f"[{style}]{row.status}[/]" if style else row.status,
        )
    console.print(table)
    if rec.choice:
        console.print(f"[bold green]→ {rec.choice.model_id}[/bold green]")
    for reason in rec.reasons:
        console.print(f"  {reason}")
    if rec.choice is None:
        raise typer.Exit(code=1)


def _load_router_config(config_path, local, remote, policy, min_logprob):
    from edgewise.config import load_config

    config = load_config(config_path)
    updates = {
        "local_model": local,
        "remote_model": remote,
        "policy": policy,
        "min_logprob": min_logprob,
    }
    config.router = config.router.model_copy(
        update={k: v for k, v in updates.items() if v is not None}
    )
    return type(config).model_validate(config.model_dump())  # re-check overridden values


@app.command()
def route(
    prompt: str = typer.Argument(..., help="The user message."),
    system: str = typer.Option(None, help="Optional system prompt."),
    private: bool = typer.Option(False, help="Never send this request off the device."),
    json_answer: bool = typer.Option(False, "--expect-json", help="The answer must be JSON."),
    force: str = typer.Option(None, help="local | remote: skip the cascade."),
    local: str = typer.Option(None, help="Local model id (default: from config)."),
    remote: str = typer.Option(None, help="cloud | a larger catalog model id | none."),
    policy: str = typer.Option(None, help="cascade | local-only | remote-only"),
    min_logprob: float = typer.Option(None, help="Escalate below this confidence."),
    max_tokens: int = typer.Option(256, help="Maximum tokens in the answer."),
    config_path: Path = typer.Option(None, "--config", help="Path to edgewise.toml."),
    db: Path = typer.Option(Path("results/edgewise.duckdb"), help="Results database."),
    json_out: bool = typer.Option(False, "--json", help="Print the decision as JSON."),
) -> None:
    """Answer on this device when possible; escalate to a stronger model when needed."""
    from rich.panel import Panel

    from edgewise.router import RouteRequest
    from edgewise.router.build import build_router
    from edgewise.schema import Message
    from edgewise.store import Store

    messages = [Message(role="system", content=system)] if system else []
    messages.append(Message(role="user", content=prompt))
    try:
        config = _load_router_config(config_path, local, remote, policy, min_logprob)
        request = RouteRequest(
            messages=messages,
            max_tokens=max_tokens,
            private=private,
            force=force,
            expect="json" if json_answer else "text",
        )
        with Store(db) as store:
            router = build_router(config, Catalog.load(), store)
            decision = router.route(request)
    except (KeyError, ValueError, FileNotFoundError) as e:
        console.print(f"[red]error:[/red] {e.args[0] if isinstance(e, KeyError) else e}")
        raise typer.Exit(code=1) from e

    if json_out:
        print(decision.model_dump_json(indent=2))
        return
    where = (
        "[green]on device[/green]" if decision.route == "local" else "[yellow]escalated[/yellow]"
    )
    console.print(Panel(decision.text.strip() or "(empty answer)", title=decision.model_id))
    console.print(f"{where} · {decision.reason}")
    console.print(
        f"[dim]{decision.completion_tokens} tokens · {decision.latency_ms:.0f} ms · "
        f"cost ${decision.cost_usd:.5f}[/dim]"
    )


@app.command()
def report(
    out: Path = typer.Option(Path("results/published"), help="Directory for REPORT.md and charts."),
    local: str = typer.Option(None, help="Local model id for the routing section."),
    strong: str = typer.Option(None, help="Stronger model id the cascade escalates to."),
    memory: str = typer.Option(None, help="Memory budget, e.g. 4GB. Default: this device's."),
    db: Path = typer.Option(Path("results/edgewise.duckdb"), help="Results database."),
) -> None:
    """Write tables and charts from the measured results (for the README)."""
    from edgewise.report.build import build_report, write_report
    from edgewise.store import Store

    budget = _parse_gb(memory) if memory else detect().model_budget_gb
    with Store(db) as store:
        result = build_report(store, Catalog.load(), budget, local, strong)
    for path in write_report(result, out):
        console.print(f"wrote {path}")
    if not result.sections:
        console.print("[yellow]no benchmark results yet: run `edgewise bench` first[/yellow]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Address to listen on (localhost by default)."),
    port: int = typer.Option(8080, help="Port to listen on."),
    local: str = typer.Option(None, help="Local model id (default: from config)."),
    remote: str = typer.Option(None, help="cloud | a larger catalog model id | none."),
    policy: str = typer.Option(None, help="cascade | local-only | remote-only"),
    min_logprob: float = typer.Option(None, help="Escalate below this confidence."),
    config_path: Path = typer.Option(None, "--config", help="Path to edgewise.toml."),
    db: Path = typer.Option(Path("results/edgewise.duckdb"), help="Results database."),
) -> None:
    """Start the demo page and a local OpenAI-compatible API, both backed by the router."""
    try:
        import uvicorn

        from edgewise.server import create_app
    except ImportError as e:
        console.print("[red]error:[/red] `edgewise serve` needs: uv sync --extra serve")
        raise typer.Exit(code=1) from e
    from edgewise.router.build import build_router
    from edgewise.store import Store

    try:
        config = _load_router_config(config_path, local, remote, policy, min_logprob)
        catalog = Catalog.load()
        catalog.get(config.router.local_model)
    except (KeyError, ValueError, FileNotFoundError) as e:
        console.print(f"[red]error:[/red] {e.args[0] if isinstance(e, KeyError) else e}")
        raise typer.Exit(code=1) from e

    console.print(
        f"local [bold]{config.router.local_model}[/bold] · escalation "
        f"[bold]{config.router.remote_model}[/bold] · policy {config.router.policy}"
    )
    console.print(f"demo page  http://{host}:{port}/")
    console.print(f"API        POST http://{host}:{port}/v1/chat/completions")
    # The store and the models are created on the server's worker thread (see create_app).
    app_ = create_app(lambda: build_router(config, catalog, Store(db)), catalog)
    uvicorn.run(app_, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
