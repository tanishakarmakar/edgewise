"""Regenerate the terminal screenshots in docs/img/ (light commands only: no model is loaded).

Usage: uv run python scripts/screenshots.py
"""

from pathlib import Path

from rich.console import Console

from edgewise import cli

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"


def capture(name: str, command: str, run) -> None:
    console = Console(record=True, width=96, force_terminal=True, color_system="truecolor")
    cli.console = console
    console.print(f"[bold]$ {command}[/bold]")
    run()
    console.save_svg(str(OUT / f"{name}.svg"), title=command)
    print(f"wrote {OUT / f'{name}.svg'}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    capture("info", "edgewise info", cli.info)
    capture("models-ls", "edgewise models ls", lambda: cli.models_ls(context=4096))
