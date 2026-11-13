"""Data contracts shared by every component (see docs/SPEC.md §5)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from edgewise.hardware import HardwareInfo

BackendName = Literal["mlx", "llamacpp", "cloud", "openai", "fake"]
Decoding = Literal["standard", "speculative", "prompt_lookup"]
KPolicy = Literal["fixed", "heuristic", "cost_aware"]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ModelVariant(BaseModel):
    """One concrete, runnable model file: a model family + size + quantization + backend."""

    id: str
    family: str
    params_b: float
    quant: str
    bits_per_weight: float
    backend: BackendName
    source: str  # Hugging Face repo id
    file: str | None = None  # single file within the repo (GGUF); None = whole repo (MLX)
    tokenizer_family: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    context_max: int


class GenerationSettings(BaseModel):
    max_tokens: int = 256
    temperature: float = 0.0  # greedy by default, so quality runs are reproducible
    seed: int = 0
    decoding: Decoding = "standard"
    num_draft_tokens: int = 4  # guesses per step (starting value for the heuristic policy)
    k_policy: KPolicy = "fixed"  # how the number of guesses per step is chosen


class TokenEvent(BaseModel):
    """One generated token, timestamped by the backend's clock."""

    token_id: int
    text: str
    t_ns: int
    logprob: float | None = None


class GenerationResult(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    prefill_tps: float
    decode_tps: float
    itl_ms: list[float] = Field(default_factory=list)
    peak_memory_mb: float
    mean_logprob: float | None = None
    decode_ms: float = 0.0  # first token -> last token
    proposed_draft_tokens: int = 0
    accepted_draft_tokens: int = 0


class EvalItem(BaseModel):
    id: str
    prompt: str
    expected: Any
    context: str | None = None  # passage for grounded QA / extraction
    tools: list[dict[str, Any]] | None = None
    tags: list[str] = Field(default_factory=list)


class Score(BaseModel):
    value: float  # 0..1
    valid: bool  # output was well-formed (parsable JSON, a number, ...)
    detail: str = ""


class RunManifest(BaseModel):
    run_id: str
    started_at: datetime
    git_sha: str
    edgewise_version: str
    hardware: HardwareInfo
    backend_versions: dict[str, str]
    on_ac_power: bool | None
    swap_used_gb: float | None = None
    notes: str = ""
