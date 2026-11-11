"""Turn a timestamped token stream into performance metrics (see docs/SPEC.md §6.5).

All backends go through the same code here, so their numbers are directly comparable.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence

from pydantic import BaseModel

from edgewise.backends.base import TokenStream
from edgewise.schema import GenerationResult

NS_PER_MS = 1e6
NS_PER_S = 1e9


def collect(stream: TokenStream, start_ns: int, peak_memory_mb: float = 0.0) -> GenerationResult:
    """Consume `stream` and summarize it. `start_ns` is when the request was issued."""
    events = list(stream.events)
    text = "".join(e.text for e in events)
    if not events:
        return GenerationResult(
            text="",
            prompt_tokens=stream.prompt_tokens,
            completion_tokens=0,
            ttft_ms=0.0,
            prefill_tps=0.0,
            decode_tps=0.0,
            peak_memory_mb=peak_memory_mb,
        )

    times = [e.t_ns for e in events]
    ttft_ns = times[0] - start_ns
    itl_ns = [b - a for a, b in zip(times, times[1:], strict=False)]
    decode_ns = times[-1] - times[0]
    logprobs = [e.logprob for e in events if e.logprob is not None]

    return GenerationResult(
        text=text,
        prompt_tokens=stream.prompt_tokens,
        completion_tokens=len(events),
        ttft_ms=ttft_ns / NS_PER_MS,
        # TTFT includes producing the first token, so this slightly underestimates pure prefill
        # speed; the error shrinks as prompts get longer.
        prefill_tps=stream.prompt_tokens / (ttft_ns / NS_PER_S) if ttft_ns > 0 else 0.0,
        # The first token is part of TTFT; decode speed counts the tokens generated after it.
        decode_tps=len(itl_ns) / (decode_ns / NS_PER_S) if decode_ns > 0 else 0.0,
        itl_ms=[d / NS_PER_MS for d in itl_ns],
        peak_memory_mb=peak_memory_mb,
        mean_logprob=statistics.fmean(logprobs) if logprobs else None,
        decode_ms=decode_ns / NS_PER_MS,
        proposed_draft_tokens=stream.stats.get("proposed", 0),
        accepted_draft_tokens=stream.stats.get("accepted", 0),
    )


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


class Spread(BaseModel):
    """Median with interquartile range: robust to the odd slow repeat (thermal, background apps)."""

    median: float
    p25: float
    p75: float
    n: int

    @classmethod
    def of(cls, values: Iterable[float]) -> Spread:
        vals = list(values)
        return cls(
            median=percentile(vals, 50),
            p25=percentile(vals, 25),
            p75=percentile(vals, 75),
            n=len(vals),
        )

    def __str__(self) -> str:
        return f"{self.median:.1f} [{self.p25:.1f}–{self.p75:.1f}]"
