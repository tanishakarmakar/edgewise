"""The Router: answer on the device when possible, escalate when needed (docs/SPEC.md §9).

Order of decisions for every request:
  1. Hard rules. Private requests, requests containing personal data, and requests made while
     offline (or with no escalation model) stay local. Nothing overrides these.
  2. Explicit choice by the caller or policy (local-only / remote-only).
  3. Cascade. Answer locally, then escalate only if the local answer looks unreliable:
     empty, not valid JSON when JSON was asked for, or generated with low confidence.
If the escalation model fails, the local answer is returned rather than an error.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from edgewise.backends.base import Backend, LoadedModel
from edgewise.bench.metrics import collect
from edgewise.evals.scorers import extract_json
from edgewise.router.rules import detect_pii
from edgewise.schema import GenerationResult, GenerationSettings, Message
from edgewise.store import Store

Policy = Literal["cascade", "local-only", "remote-only"]
Route = Literal["local", "remote"]
# Called with (which tier is generating, the new token's text) while an answer is produced.
TokenListener = Callable[[Route, str], None]


class EscalationFailed(RuntimeError):
    pass


@dataclass
class Tier:
    """A model the router can send requests to."""

    backend: Backend
    model: LoadedModel
    input_price: float = 0.0  # USD per million input tokens (0 for on-device models)
    output_price: float = 0.0
    needs_network: bool = False
    settings: dict = field(default_factory=dict)  # extra GenerationSettings, e.g. decoding

    @property
    def model_id(self) -> str:
        return self.model.variant.id


def _tap(events, route: Route, on_token: TokenListener):
    for event in events:
        on_token(route, event.text)
        yield event


class RouteRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 256
    private: bool = False  # never leaves the device
    force: Route | None = None
    expect: Literal["text", "json"] = "text"


class RouteDecision(BaseModel):
    request_id: str
    route: Route
    reason: str
    escalated: bool = False  # a local answer was generated first and then replaced
    model_id: str
    text: str
    confidence: float | None = None  # mean token log-probability of the local answer
    pii: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class RouterStats(BaseModel):
    requests: int = 0
    local: int = 0
    remote: int = 0
    escalated: int = 0
    kept_private: int = 0
    cost_usd: float = 0.0

    @property
    def local_share(self) -> float:
        return self.local / self.requests if self.requests else 0.0


class Router:
    def __init__(
        self,
        local: Tier,
        remote: Tier | None = None,
        policy: Policy = "cascade",
        min_logprob: float = -0.8,
        is_online: Callable[[], bool] = lambda: True,
        store: Store | None = None,
    ):
        self.local = local
        self.remote = remote
        self.policy = policy
        self.min_logprob = min_logprob
        self.is_online = is_online
        self.store = store
        self.stats = RouterStats()

    # ------------------------------------------------------------------ decision

    def route(self, request: RouteRequest, on_token: TokenListener | None = None) -> RouteDecision:
        """`on_token` lets a caller watch the answer being generated. In a cascade it sees the
        local answer first and then, if that one is not kept, the escalated answer."""
        pii = detect_pii(" ".join(m.content for m in request.messages))
        stay_local = self._must_stay_local(request, pii)

        def local(reason: str) -> RouteDecision:
            return self._answer(self.local, request, "local", reason, on_token)

        if stay_local:
            decision = local(stay_local)
        elif request.force == "local":
            decision = local("forced local by the caller")
        elif request.force == "remote":
            decision = self._remote_or_local(request, "forced remote by the caller", on_token)
        elif self.policy == "local-only":
            decision = local("policy: local-only")
        elif self.policy == "remote-only":
            decision = self._remote_or_local(request, "policy: remote-only", on_token)
        else:
            decision = self._cascade(request, on_token)

        decision.pii = pii
        self._record(decision, request, kept_private=bool(request.private or pii))
        return decision

    def _must_stay_local(self, request: RouteRequest, pii: list[str]) -> str | None:
        if request.private:
            return "marked private: kept on device"
        if pii:
            return f"contains {', '.join(pii)}: kept on device"
        if self.remote is None:
            return "no escalation model configured"
        if self.remote.needs_network and not self.is_online():
            return "offline: answered on device"
        return None

    def _cascade(self, request: RouteRequest, on_token: TokenListener | None) -> RouteDecision:
        local = self._answer(self.local, request, "local", "", on_token)
        doubt = self._doubt(local, request)
        if doubt is None:
            confidence = "" if local.confidence is None else f" (confidence {local.confidence:.2f})"
            local.reason = f"local answer accepted{confidence}"
            return local
        try:
            remote = self._answer(self.remote, request, "remote", f"escalated: {doubt}", on_token)
        except EscalationFailed as e:
            local.reason = f"{doubt}, but escalation failed ({e}): returning the local answer"
            return local
        remote.escalated = True
        remote.confidence = local.confidence
        remote.latency_ms += local.latency_ms  # the user waited for both
        return remote

    def _remote_or_local(
        self, request: RouteRequest, reason: str, on_token: TokenListener | None
    ) -> RouteDecision:
        try:
            return self._answer(self.remote, request, "remote", reason, on_token)
        except EscalationFailed as e:
            reason = f"escalation failed ({e}): answered on device"
            return self._answer(self.local, request, "local", reason, on_token)

    def _doubt(self, local: RouteDecision, request: RouteRequest) -> str | None:
        """Why the local answer should not be trusted, or None if it looks fine."""
        if not local.text.strip():
            return "local answer was empty"
        if request.expect == "json" and extract_json(local.text) is None:
            return "local answer was not valid JSON"
        if local.confidence is not None and local.confidence < self.min_logprob:
            return f"low confidence ({local.confidence:.2f} < {self.min_logprob:.2f})"
        return None

    # ------------------------------------------------------------------ generation

    def _answer(
        self,
        tier: Tier,
        request: RouteRequest,
        route: Route,
        reason: str,
        on_token: TokenListener | None = None,
    ) -> RouteDecision:
        settings = GenerationSettings(max_tokens=request.max_tokens, **tier.settings)
        start = tier.backend.now_ns()
        try:
            stream = tier.backend.generate(tier.model, request.messages, settings)
            if on_token is not None:
                stream.events = _tap(stream.events, route, on_token)
            result: GenerationResult = collect(stream, start)
        except Exception as e:
            if route == "remote":
                raise EscalationFailed(str(e) or type(e).__name__) from e
            raise
        # Cloud backends only learn token counts at the end of the stream.
        prompt_tokens = stream.stats.get("input_tokens", result.prompt_tokens)
        completion_tokens = stream.stats.get("output_tokens", result.completion_tokens)
        return RouteDecision(
            request_id=uuid.uuid4().hex[:12],
            route=route,
            reason=reason,
            model_id=stream.stats.get("served_by") or tier.model_id,
            text=result.text,
            confidence=result.mean_logprob,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=(prompt_tokens * tier.input_price + completion_tokens * tier.output_price)
            / 1e6,
            latency_ms=(tier.backend.now_ns() - start) / 1e6,
        )

    # ------------------------------------------------------------------ bookkeeping

    def _record(self, d: RouteDecision, request: RouteRequest, kept_private: bool) -> None:
        self.stats.requests += 1
        self.stats.local += d.route == "local"
        self.stats.remote += d.route == "remote"
        self.stats.escalated += d.escalated
        self.stats.kept_private += kept_private
        self.stats.cost_usd += d.cost_usd
        if self.store is not None:
            self.store.add_route(
                {
                    "request_id": d.request_id,
                    "ts": datetime.now(),
                    "policy": self.policy,
                    "route": d.route,
                    "reason": d.reason,
                    "escalated": d.escalated,
                    "model_id": d.model_id,
                    "confidence": d.confidence,
                    "private": request.private,
                    "pii": ", ".join(d.pii),
                    "prompt_tokens": d.prompt_tokens,
                    "completion_tokens": d.completion_tokens,
                    "cost_usd": d.cost_usd,
                    "latency_ms": d.latency_ms,
                }
            )
