"""User configuration: `edgewise.toml` (see docs/SPEC.md §11).

Looked up in order: an explicit path, ./edgewise.toml, ~/.config/edgewise/edgewise.toml.
Missing files and missing keys fall back to the defaults below. API credentials are never
read from this file; the cloud backend takes them from the environment.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SEARCH_PATHS = (Path("edgewise.toml"), Path.home() / ".config" / "edgewise" / "edgewise.toml")


class CloudConfig(BaseModel):
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # USD per million tokens. Defaults are this model's list price on 2026-11-03; prices
    # change, so check the provider's pricing page and override here.
    input_price_per_mtok: float = 4.0
    output_price_per_mtok: float = 20.0


class RouterConfig(BaseModel):
    local_model: str = "qwen2.5-1.5b-instruct@mlx-4bit"
    # "cloud", the id of a larger catalog model to escalate to on-device, or "none".
    remote_model: str = "cloud"
    policy: Literal["cascade", "local-only", "remote-only"] = "cascade"
    # Escalate when the local answer's mean token log-probability is below this.
    # A starting point only: calibrate it with `edgewise report router`.
    min_logprob: float = -0.8


class Config(BaseModel):
    router: RouterConfig = Field(default_factory=RouterConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)


def load_config(path: Path | None = None) -> Config:
    if path is not None and not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    for candidate in (path,) if path else SEARCH_PATHS:
        if candidate.exists():
            with candidate.open("rb") as f:
                return Config(**tomllib.load(f))
    return Config()
