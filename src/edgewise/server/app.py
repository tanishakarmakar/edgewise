"""Local HTTP API in the OpenAI chat-completions format, backed by the Router.

Any app that can talk to an OpenAI-compatible endpoint can point at this server and get
local-first answers. Each response carries an extra `edgewise` object explaining the route.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from edgewise.catalog import Catalog
from edgewise.evals import TASKS
from edgewise.router import RouteDecision, Router, RouteRequest
from edgewise.schema import Message
from edgewise.server import demo

STATIC = Path(__file__).parent / "static"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[dict[str, Any]]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(part.get("text", "") for part in self.content if part.get("type") == "text")


class EdgewiseOptions(BaseModel):
    private: bool = False
    force: Literal["local", "remote"] | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None  # accepted for compatibility; the router picks the model
    max_tokens: int = Field(default=256, ge=1, le=8192)
    stream: bool = False
    response_format: dict[str, Any] | None = None
    edgewise: EdgewiseOptions = Field(default_factory=EdgewiseOptions)


class CompareRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=200, ge=1, le=512)


def _route_request(body: ChatRequest) -> RouteRequest:
    wants_json = (body.response_format or {}).get("type") in ("json_object", "json_schema")
    return RouteRequest(
        messages=[Message(role=m.role, content=m.text()) for m in body.messages],
        max_tokens=body.max_tokens,
        private=body.edgewise.private,
        force=body.edgewise.force,
        expect="json" if wants_json else "text",
    )


def create_app(router: Router | Callable[[], Router], catalog: Catalog | None = None) -> FastAPI:
    """`router` may be a factory; it is then called at startup on the worker thread, so models
    are loaded on the same thread that later runs them."""
    # MLX state belongs to the thread that created it, and one generation at a time is all an
    # 8 GB device can do anyway: load and run everything on a single worker thread.
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edgewise-router")
    state: dict[str, Router] = {}
    if isinstance(router, Router):
        state["router"] = router

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if "router" not in state:
            state["router"] = await asyncio.get_running_loop().run_in_executor(worker, router)
        yield
        worker.shutdown(wait=False)

    app = FastAPI(title="Edgewise", description="Local-first LLM routing", lifespan=lifespan)
    catalog = catalog or Catalog.load()

    async def on_worker(fn, *args):
        """Models and the results database live on the worker thread; run `fn` there."""
        return await asyncio.get_running_loop().run_in_executor(worker, fn, *args)

    def stream_from_worker(job: Callable[[demo.Emit], None]) -> StreamingResponse:
        """Run `job(emit)` on the worker and relay every emitted event as server-sent events."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(event: dict[str, Any] | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def run() -> None:
            try:
                job(emit)
            except Exception as e:
                emit({"type": "error", "message": str(e) or type(e).__name__})
            finally:
                emit(None)

        loop.run_in_executor(worker, run)

        async def events():
            while (event := await queue.get()) is not None:
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    # ------------------------------------------------------------------ demo page

    app.mount("/ui", StaticFiles(directory=STATIC), name="ui")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/overview")
    async def overview() -> dict[str, Any]:
        return await on_worker(demo.overview, state["router"], catalog)

    @app.get("/api/advise")
    async def advise_endpoint(
        task: str,
        memory_gb: float = Query(gt=0, le=512),
        min_score: float = Query(default=0.0, ge=0, le=1),
        prefer: Literal["balanced", "speed", "quality"] = "balanced",
    ) -> dict[str, Any]:
        if task not in TASKS:
            raise HTTPException(status_code=404, detail=f"unknown task {task!r}")
        try:
            return await on_worker(
                demo.advice, state["router"], catalog, task, memory_gb, min_score, prefer
            )
        except LookupError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    @app.post("/api/route")
    async def route_live(body: ChatRequest) -> StreamingResponse:
        """The same decision as /v1/chat/completions, with tokens relayed as they are
        generated, so a page can show the local answer and then the escalation."""
        request = _route_request(body)

        def job(emit: demo.Emit) -> None:
            decision = state["router"].route(
                request,
                on_token=lambda route, text: emit({"type": "token", "route": route, "text": text}),
            )
            emit(
                {
                    "type": "decision",
                    **_route_info(decision),
                    "model": decision.model_id,
                    "text": decision.text,
                    "prompt_tokens": decision.prompt_tokens,
                    "completion_tokens": decision.completion_tokens,
                    "stats": {
                        **state["router"].stats.model_dump(),
                        "local_share": state["router"].stats.local_share,
                    },
                }
            )

        return stream_from_worker(job)

    @app.post("/api/compare")
    async def compare(body: CompareRequest) -> StreamingResponse:
        messages = [Message(role=m.role, content=m.text()) for m in body.messages]
        return stream_from_worker(
            lambda emit: demo.run_comparison(state["router"].local, messages, body.max_tokens, emit)
        )

    # ------------------------------------------------------------------ OpenAI-style API

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        r = state["router"]
        tiers = [r.local] + ([r.remote] if r.remote else [])
        return {
            "object": "list",
            "data": [{"id": t.model_id, "object": "model", "owned_by": "edgewise"} for t in tiers],
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        stats = state["router"].stats
        return {**stats.model_dump(), "local_share": stats.local_share}

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest):
        request = _route_request(body)
        loop = asyncio.get_running_loop()
        try:
            decision = await loop.run_in_executor(worker, state["router"].route, request)
        except Exception as e:  # the local model itself failed; nothing to fall back to
            raise HTTPException(status_code=500, detail=f"generation failed: {e}") from e

        if body.stream:
            return StreamingResponse(_sse(decision), media_type="text/event-stream")
        return _completion(decision)

    return app


def _route_info(d: RouteDecision) -> dict[str, Any]:
    return {
        "route": d.route,
        "reason": d.reason,
        "escalated": d.escalated,
        "confidence": d.confidence,
        "pii": d.pii,
        "cost_usd": d.cost_usd,
        "latency_ms": round(d.latency_ms, 1),
    }


def _envelope(d: RouteDecision, kind: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{d.request_id}",
        "object": kind,
        "created": int(time.time()),
        "model": d.model_id,
    }


def _completion(d: RouteDecision) -> dict[str, Any]:
    return {
        **_envelope(d, "chat.completion"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": d.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": d.prompt_tokens,
            "completion_tokens": d.completion_tokens,
            "total_tokens": d.prompt_tokens + d.completion_tokens,
        },
        "edgewise": _route_info(d),
    }


def _sse(d: RouteDecision, chunk_chars: int = 24):
    """Server-sent events in the OpenAI streaming format.

    The cascade has to see the whole local answer before deciding whether to keep it, so
    chunks are sent once the decision is made rather than token by token.
    """

    def event(delta: dict[str, Any], finish: str | None = None, extra: dict | None = None) -> str:
        payload = {
            **_envelope(d, "chat.completion.chunk"),
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            **(extra or {}),
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield event({"role": "assistant"})
    for i in range(0, len(d.text), chunk_chars):
        yield event({"content": d.text[i : i + chunk_chars]})
    yield event({}, finish="stop", extra={"edgewise": _route_info(d)})
    yield "data: [DONE]\n\n"
