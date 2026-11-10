"""Command-line entry point: `edgewise <command>`."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from edgewise import __version__
from edgewise.hardware import detect

app = typer.Typer(
    help="Run the right LLM, the right way, on the hardware you have.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


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


@app.command()
def bench() -> None:
    """Measure speed, memory and task quality of a model on this device."""
    _not_ready("bench", "milestone 1")


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
