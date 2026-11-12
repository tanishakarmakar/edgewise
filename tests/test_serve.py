"""Config, router assembly, the local HTTP API, and the route command."""

import json

import pytest
from conftest import record_run
from typer.testing import CliRunner

from edgewise.backends.fake import FakeBackend
from edgewise.catalog import Catalog
from edgewise.cli import app as cli
from edgewise.config import Config, load_config
from edgewise.router import Router, Tier
from edgewise.router.build import build_router

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from edgewise.server import create_app  # noqa: E402

SMALL, BIG = "qwen2.5-0.5b-instruct@mlx-4bit", "qwen2.5-3b-instruct@mlx-4bit"


def fake_tier(answer, logprob=-0.1, model_id=SMALL, **kwargs):
    backend = FakeBackend(responder=lambda messages: answer, logprob=logprob)
    return Tier(backend, backend.load(Catalog.load().get(model_id)), **kwargs)


# ------------------------------------------------------------------ config


def test_config_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_config()
    assert config.router.policy == "cascade" and config.cloud.model == "claude-opus-5-5"


def test_config_file_overrides_defaults(tmp_path):
    path = tmp_path / "edgewise.toml"
    path.write_text(
        '[router]\nlocal_model = "x"\nremote_model = "none"\nmin_logprob = -0.5\n'
        "[cloud]\ninput_price_per_mtok = 1.5\n"
    )
    config = load_config(path)
    assert config.router.local_model == "x" and config.router.min_logprob == -0.5
    assert config.cloud.input_price_per_mtok == 1.5 and config.cloud.output_price_per_mtok == 20.0


def test_config_rejects_bad_values_and_missing_file(tmp_path):
    path = tmp_path / "edgewise.toml"
    path.write_text('[router]\npolicy = "always-cloud"\n')
    with pytest.raises(ValueError):
        load_config(path)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


# ------------------------------------------------------------------ router assembly


def config_with(**router):
    return Config.model_validate({"router": router})


def test_build_local_only_router():
    backends = []

    def backend_for(name):
        backends.append(FakeBackend())
        return backends[-1]

    router = build_router(
        config_with(local_model=SMALL, remote_model="none"), Catalog.load(), budget_gb=4.5,
        backend_for=backend_for,
    )  # fmt: skip
    assert router.remote is None and backends[0].loaded == [SMALL]


def test_build_with_larger_local_model_as_escalation():
    backend = FakeBackend()
    router = build_router(
        config_with(local_model=SMALL, remote_model=BIG, min_logprob=-0.3), Catalog.load(),
        budget_gb=4.5, backend_for=lambda name: backend,
    )  # fmt: skip
    assert router.remote.model_id == BIG and not router.remote.needs_network
    assert router.remote.input_price == 0 and router.min_logprob == -0.3
    assert backend.loaded == [SMALL, BIG]


def test_build_refuses_models_that_do_not_fit_before_loading():
    backend = FakeBackend()
    with pytest.raises(ValueError, match="memory budget"):
        build_router(
            config_with(local_model=SMALL, remote_model=BIG), Catalog.load(), budget_gb=1.0,
            backend_for=lambda name: backend,
        )  # fmt: skip
    assert backend.loaded == []


def test_build_with_cloud_escalation_uses_configured_prices():
    pytest.importorskip("anthropic")
    config = Config.model_validate(
        {"router": {"local_model": SMALL}, "cloud": {"input_price_per_mtok": 2.5}}
    )
    router = build_router(
        config, Catalog.load(), budget_gb=4.5, backend_for=lambda n: FakeBackend()
    )
    assert router.remote.model_id == "cloud:claude-opus-5-5"
    assert router.remote.needs_network and router.remote.input_price == 2.5


# ------------------------------------------------------------------ HTTP API


@pytest.fixture
def client():
    router = Router(
        fake_tier("local answer", logprob=-0.2),
        fake_tier("remote answer", model_id=BIG, input_price=4.0, output_price=20.0),
        min_logprob=-0.8,
    )
    return TestClient(create_app(router)), router


def post(client, **body):
    body.setdefault("messages", [{"role": "user", "content": "Capital of France?"}])
    return client.post("/v1/chat/completions", json=body)


def test_chat_completion_shape(client):
    http, _ = client
    data = post(http).json()
    assert data["object"] == "chat.completion" and data["model"] == SMALL
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "local answer"}
    assert data["usage"]["total_tokens"] == 3 + 2
    assert data["edgewise"]["route"] == "local" and data["edgewise"]["cost_usd"] == 0


def test_low_confidence_request_is_escalated(client):
    http, router = client
    router.min_logprob = 0.0
    data = post(http).json()
    assert data["model"] == BIG and data["choices"][0]["message"]["content"] == "remote answer"
    assert data["edgewise"]["escalated"] is True and data["edgewise"]["cost_usd"] > 0


def test_private_flag_and_pii_stay_local(client):
    http, router = client
    router.min_logprob = 0.0
    assert post(http, edgewise={"private": True}).json()["edgewise"]["route"] == "local"
    pii = post(http, messages=[{"role": "user", "content": "my email is a@b.co"}]).json()
    assert pii["edgewise"]["route"] == "local" and pii["edgewise"]["pii"] == ["email address"]


def test_json_response_format_escalates_invalid_json(client):
    http, _ = client
    data = post(http, response_format={"type": "json_object"}).json()
    assert data["edgewise"]["route"] == "remote" and "JSON" in data["edgewise"]["reason"]


def test_content_parts_are_accepted(client):
    http, _ = client
    parts = [{"type": "text", "text": "Capital "}, {"type": "text", "text": "of France?"}]
    assert post(http, messages=[{"role": "user", "content": parts}]).status_code == 200


def test_streaming_sends_openai_style_events(client):
    http, _ = client
    lines = [x for x in post(http, stream=True).text.split("\n\n") if x]
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(x.removeprefix("data: ")) for x in lines[:-1]]
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert text == "local answer"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["edgewise"]["route"] == "local"


def test_validation_and_support_endpoints(client):
    http, _ = client
    assert http.post("/v1/chat/completions", json={"messages": []}).status_code == 422
    assert http.get("/healthz").json() == {"status": "ok"}
    assert [m["id"] for m in http.get("/v1/models").json()["data"]] == [SMALL, BIG]
    post(http)
    metrics = http.get("/metrics").json()
    assert metrics["requests"] == 1 and metrics["local_share"] == 1.0


def test_router_factory_runs_at_startup():
    built = []

    def factory():
        built.append(1)
        return Router(fake_tier("hello"))

    with TestClient(create_app(factory)) as http:
        assert post(http).json()["choices"][0]["message"]["content"] == "hello"
    assert built == [1]


# ------------------------------------------------------------------ CLI


def test_cli_route(tmp_path, monkeypatch):
    import edgewise.router.build as build

    seen = {}

    def fake_build(config, catalog, store=None, **kwargs):
        seen["config"] = config
        return Router(fake_tier("Paris"), store=store)

    monkeypatch.setattr(build, "build_router", fake_build)
    runner = CliRunner()
    args = ["route", "Capital of France?", "--db", str(tmp_path / "r.duckdb"), "--remote", "none"]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0 and "Paris" in result.stdout and "on device" in result.stdout
    assert seen["config"].router.remote_model == "none"

    as_json = runner.invoke(cli, [*args, "--json", "--private"])
    assert json.loads(as_json.stdout)["route"] == "local"

    bad = runner.invoke(cli, [*args, "--policy", "sometimes"])
    assert bad.exit_code == 1


# ------------------------------------------------------------------ demo page


def sse(response):
    return [json.loads(x.removeprefix("data: ")) for x in response.text.split("\n\n") if x]


@pytest.fixture
def demo_client(store):
    record_run(store, SMALL, scores=[1] * 19 + [0], tps=120, peak_mb=400)
    record_run(store, BIG, scores=[1] * 20, tps=24, peak_mb=2100)
    router = Router(
        fake_tier("the local answer", logprob=-0.2),
        fake_tier("the remote answer", model_id=BIG),
        min_logprob=-0.8,
        store=store,
    )
    return TestClient(create_app(router)), router


def test_demo_page_and_assets_are_served(demo_client):
    http, _ = demo_client
    page = http.get("/")
    assert page.status_code == 200 and "Edgewise" in page.text
    for asset in ("app.css", "app.js"):
        assert http.get(f"/ui/{asset}").status_code == 200
        assert f"/ui/{asset}" in page.text


def test_demo_page_loads_nothing_from_the_network(demo_client):
    # An on-device tool's own page must work offline: no CDN fonts, scripts or styles.
    http, _ = demo_client
    for path in ("/", "/ui/app.css", "/ui/app.js"):
        text = http.get(path).text
        assert "https://" not in text and "http://" not in text.replace("http://www.w3.org", "")


def test_overview_describes_device_router_and_tasks(demo_client):
    http, _ = demo_client
    data = http.get("/api/overview").json()
    assert data["router"]["local"] == {"id": SMALL, "label": "qwen2.5-0.5b 4bit", "on_device": True}
    assert data["router"]["remote"]["id"] == BIG and data["router"]["min_logprob"] == -0.8
    assert [t["id"] for t in data["tasks"]][0] == "tool_calling"
    assert data["benchmarked"] == 2 and data["device"]["ram_gb"] > 0


def test_advise_endpoint_returns_rows_with_labels(demo_client):
    http, _ = demo_client
    data = http.get("/api/advise", params={"task": "tool_calling", "memory_gb": 4.5}).json()
    assert data["choice"]["model_id"] == SMALL and data["choice"]["label"] == "qwen2.5-0.5b 4bit"
    assert all("label" in row for row in data["rows"])
    # A tighter budget rules the 3B model out.
    tight = http.get("/api/advise", params={"task": "tool_calling", "memory_gb": 1.0}).json()
    assert {r["model_id"]: r["status"] for r in tight["rows"]}[BIG] == "too big"


def test_advise_endpoint_rejects_unknown_task_and_missing_store(demo_client, client):
    http, _ = demo_client
    assert http.get("/api/advise", params={"task": "nope", "memory_gb": 4}).status_code == 404
    no_store, _ = client
    response = no_store.get("/api/advise", params={"task": "tool_calling", "memory_gb": 4})
    assert response.status_code == 503


def test_live_route_streams_tokens_then_the_decision(demo_client):
    http, router = demo_client
    body = {"messages": [{"role": "user", "content": "Capital of France?"}]}
    events = sse(http.post("/api/route", json=body))
    tokens = [e for e in events if e["type"] == "token"]
    assert "".join(e["text"] for e in tokens) == "the local answer"
    assert {e["route"] for e in tokens} == {"local"}
    decision = events[-1]
    assert decision["type"] == "decision" and decision["route"] == "local"
    assert decision["text"] == "the local answer" and decision["stats"]["requests"] == 1

    # When the local answer is not trusted, the page sees it first and then the escalation.
    router.min_logprob = 0.0
    events = sse(http.post("/api/route", json=body))
    routes = [e["route"] for e in events if e["type"] == "token"]
    assert routes == ["local"] * 3 + ["remote"] * 3
    assert events[-1]["escalated"] is True and events[-1]["text"] == "the remote answer"


def test_compare_runs_both_decodings_and_reports_identity(demo_client):
    http, router = demo_client
    body = {"messages": [{"role": "user", "content": "Fix the typo"}], "max_tokens": 50}
    events = sse(http.post("/api/compare", json=body))
    lanes = [e["lane"] for e in events if e["type"] == "start"]
    assert lanes == ["standard", "prompt_lookup"]
    done = {e["lane"]: e for e in events if e["type"] == "done"}
    assert done["standard"]["tokens"] == 3 and done["standard"]["decode_tps"] == 100.0
    assert router.local.backend.last_settings.decoding == "prompt_lookup"
    # One warm-up generation the first time, then one per lane; no warm-up the second time.
    assert router.local.backend.generations == 3
    http.post("/api/compare", json=body)
    assert router.local.backend.generations == 5
    result = events[-1]
    assert result == {"type": "result", "identical": True, "speedup": 1.0}


def test_compare_reports_a_failing_backend_as_an_event(demo_client):
    http, router = demo_client

    def broken(*args, **kwargs):
        raise RuntimeError("model fell over")

    router.local.backend.generate = broken
    events = sse(http.post("/api/compare", json={"messages": [{"role": "user", "content": "x"}]}))
    assert events == [{"type": "error", "message": "model fell over"}]
