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


def _not_ready(command: str, milestone: str) -> None:
    console.print(f"[yellow]`edgewise {command}` is planned for {milestone}.[/yellow]")
    console.print("See the roadmap: https://github.com/tanishakarmakar/edgewise#roadmap")
    raise typer.Exit(code=1)


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
    table = Table(title=f"Models (memory at {context} tokens, budget ~{budget} GB)")
    table.add_column("Model", no_wrap=True)  # ids get copy-pasted into other commands
    for col in ("Params", "Quant", "Backend", "Est. memory", "Fit", "Downloaded"):
        table.add_column(col)
    for v in Catalog.load():
        est = estimate_memory(v, context_tokens=context)
        fit = est.fit(budget)
        size = downloaded.get(v.source)
        table.add_row(
            v.id,
            f"{v.params_b:g}B",
            v.quant,
            v.backend,
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
    prompt_lengths: str = typer.Option("128,1024,4096", help="Comma-separated prompt lengths."),
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

    variant = Catalog.load().get(model)
    config = BenchConfig(
        tasks=list(TASKS) if all_tasks else task,
        prompt_lengths=[int(n) for n in prompt_lengths.split(",") if n.strip()],
        repeats=repeats,
        warmup=warmup,
        perf_max_tokens=max_tokens,
        cooldown_s=cooldown,
        force=force,
    )
    log = (lambda msg: None) if json_out else (lambda msg: console.print(f"[dim]{msg}[/dim]"))
    with Store(db) as store:
        try:
            report = run_bench(variant, get_backend(variant.backend), config, store, log)
        except BenchError as e:
            console.print(f"[red]error:[/red] {e}")
            raise typer.Exit(code=1) from e

    if json_out:
        print(report.model_dump_json(indent=2))
        return

    perf = Table(title=f"{report.model_id}: performance (median [p25–p75])", title_justify="left")
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
        quality = Table(title="Quality", title_justify="left")
        for col in ("Task", "Items", "Score", "Valid output", "Mean latency"):
            quality.add_column(col, justify="right")
        for q in report.quality:
            quality.add_row(
                q.task_id,
                str(q.n),
                f"{q.mean_score:.2f}",
                f"{q.valid_rate:.0%}",
                f"{q.mean_latency_ms:.0f} ms",
            )
        console.print(quality)

    console.print(
        f"load {report.load_s:.1f}s · memory after load {report.memory_after_load_mb:.0f} MB · "
        f"estimate {report.est_memory_gb:.2f} GB · run [bold]{report.run_id}[/bold] → {db}"
    )


@app.command()
def advise() -> None:
    """Recommend the best quantized model for a memory budget and task."""
    _not_ready("advise", "milestone 2")


@app.command()
def route() -> None:
    """Answer locally when possible and escalate to a cloud model when needed."""
    _not_ready("route", "milestone 4")


if __name__ == "__main__":
    app()
