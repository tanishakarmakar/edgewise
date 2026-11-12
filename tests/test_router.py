import pytest
from conftest import record_run

from edgewise.backends.fake import FakeBackend
from edgewise.catalog import Catalog
from edgewise.router import (
    Connectivity,
    Router,
    RouteRequest,
    Tier,
    detect_pii,
    evaluate_cascade,
)
from edgewise.schema import Message

SMALL, BIG = "qwen2.5-0.5b-instruct@mlx-4bit", "qwen2.5-3b-instruct@mlx-4bit"


def tier(answer="local answer", logprob=-0.1, model_id=SMALL, fail=False, **kwargs):
    def respond(messages):
        if fail:
            raise ConnectionError("network down")
        return answer

    backend = FakeBackend(responder=respond, logprob=logprob)
    return Tier(backend=backend, model=backend.load(Catalog.load().get(model_id)), **kwargs)


def remote_tier(answer="remote answer", **kwargs):
    kwargs.setdefault("input_price", 4.0)
    kwargs.setdefault("output_price", 20.0)
    return tier(answer=answer, model_id=BIG, **kwargs)


def ask(text="What is the capital of France?", **kwargs):
    return RouteRequest(messages=[Message(role="user", content=text)], **kwargs)


# ------------------------------------------------------------------ PII detection


@pytest.mark.parametrize(
    "text, kind",
    [
        ("mail me at priya.s@example.com", "email address"),
        ("call +91 98450 12345 tomorrow", "phone number"),
        ("my card is 4111 1111 1111 1111", "card number"),
        ("Aadhaar 2345 6789 0123", "Aadhaar number"),
        ("PAN ABCDE1234F", "PAN"),
        ("SSN 078-05-1120", "US social security number"),
    ],
)
def test_detects_pii(text, kind):
    assert kind in detect_pii(text)


@pytest.mark.parametrize(
    "text",
    [
        "Set the temperature to 22 degrees",
        "Odometer reads 48,210 km, next service at 58,000 km",
        "Order 1234567890123456 shipped",  # 16 digits but fails the card checksum
        "Released in 1986, payload 64 bytes",
    ],
)
def test_no_false_alarm_on_ordinary_numbers(text):
    assert detect_pii(text) == []


# ------------------------------------------------------------------ hard rules


def test_private_request_never_leaves_the_device():
    remote = remote_tier()
    router = Router(tier(logprob=-5.0), remote, policy="remote-only")
    d = router.route(ask(private=True, force="remote"))
    assert d.route == "local" and "private" in d.reason
    assert remote.backend.generations == 0


def test_pii_keeps_request_local_even_when_confidence_is_low():
    remote = remote_tier()
    d = Router(tier(logprob=-5.0), remote).route(ask("email the report to ravi@example.com"))
    assert d.route == "local" and d.pii == ["email address"]
    assert remote.backend.generations == 0


def test_offline_answers_locally():
    remote = remote_tier(needs_network=True)
    d = Router(tier(logprob=-5.0), remote, is_online=lambda: False).route(ask())
    assert d.route == "local" and "offline" in d.reason
    assert remote.backend.generations == 0


def test_no_remote_configured():
    d = Router(tier(logprob=-5.0)).route(ask())
    assert d.route == "local" and "no escalation model" in d.reason


# ------------------------------------------------------------------ cascade


def test_confident_local_answer_is_kept():
    remote = remote_tier()
    d = Router(tier(logprob=-0.2), remote, min_logprob=-0.8).route(ask())
    assert (d.route, d.escalated, d.text) == ("local", False, "local answer")
    assert d.cost_usd == 0 and remote.backend.generations == 0


def test_low_confidence_escalates_and_costs_money():
    d = Router(tier(logprob=-1.5), remote_tier(), min_logprob=-0.8).route(ask())
    assert (d.route, d.escalated, d.text) == ("remote", True, "remote answer")
    assert "low confidence" in d.reason and d.confidence == pytest.approx(-1.5)
    # 6 prompt words × $4/M + 2 answer words × $20/M
    assert d.cost_usd == pytest.approx((6 * 4 + 2 * 20) / 1e6)


def test_invalid_json_escalates_even_when_confident():
    d = Router(tier("sure, here you go", logprob=-0.05), remote_tier('{"ok": true}')).route(
        ask(expect="json")
    )
    assert d.route == "remote" and "not valid JSON" in d.reason


def test_valid_json_stays_local():
    d = Router(tier('{"city": "Paris"}'), remote_tier()).route(ask(expect="json"))
    assert d.route == "local"


def test_escalation_failure_falls_back_to_local_answer():
    d = Router(tier(logprob=-2.0), remote_tier(fail=True)).route(ask())
    assert d.route == "local" and d.text == "local answer"
    assert "escalation failed" in d.reason and "network down" in d.reason


def test_escalated_latency_includes_both_models():
    router = Router(tier(logprob=-2.0), remote_tier())
    local_only = Router(tier(logprob=-2.0)).route(ask())
    assert router.route(ask()).latency_ms > local_only.latency_ms


# ------------------------------------------------------------------ policies and forcing


def test_policies():
    assert (
        Router(tier(logprob=-5.0), remote_tier(), policy="local-only").route(ask()).route == "local"
    )
    assert Router(tier(), remote_tier(), policy="remote-only").route(ask()).route == "remote"


def test_force_overrides_policy_but_not_privacy():
    router = Router(tier(), remote_tier(), policy="local-only")
    assert router.route(ask(force="remote")).route == "remote"
    assert router.route(ask(force="remote", private=True)).route == "local"


def test_remote_only_falls_back_when_remote_fails():
    d = Router(tier(), remote_tier(fail=True), policy="remote-only").route(ask())
    assert d.route == "local" and "escalation failed" in d.reason


# ------------------------------------------------------------------ bookkeeping


def test_stats_and_route_log(store):
    router = Router(tier(logprob=-0.2), remote_tier(), store=store)
    router.route(ask())
    router.route(ask(private=True))
    router.min_logprob = 0.0  # everything now looks unconfident
    router.route(ask())
    s = router.stats
    assert (s.requests, s.local, s.remote, s.escalated, s.kept_private) == (3, 2, 1, 1, 1)
    assert s.local_share == pytest.approx(2 / 3) and s.cost_usd > 0
    log = store.query("SELECT route, private, cost_usd FROM route_log ORDER BY ts")
    assert [r["route"] for r in log] == ["local", "local", "remote"]
    assert "France" not in str(store.query("SELECT * FROM route_log"))  # prompts are not logged


def test_connectivity_is_cached(monkeypatch):
    probes = []
    check = Connectivity(ttl_s=60)
    monkeypatch.setattr(check, "_probe", lambda: probes.append(1) or True)
    assert check() and check() and check()
    assert len(probes) == 1


# ------------------------------------------------------------------ offline cascade evaluation


@pytest.fixture
def benchmarked(store):
    # The local model is right when confident and wrong when unsure; the strong one is always right.
    record_run(store, SMALL, scores=[1, 1, 1, 0, 0], logprobs=[-0.1, -0.2, -0.3, -1.0, -2.0])
    record_run(store, BIG, scores=[1, 1, 1, 1, 1], tokens=101, prompt_tokens=200)
    return store


def test_cascade_evaluation_finds_the_cheapest_good_threshold(benchmarked):
    ev = evaluate_cascade(benchmarked, SMALL, BIG, input_price=4.0, output_price=20.0)
    assert ev.n_items == 5
    assert ev.local_only_score == pytest.approx(0.6) and ev.strong_only_score == 1.0
    # Escalating just the two unconfident items already matches the strong model.
    best = ev.recommended
    assert best.score == 1.0 and best.local_share == pytest.approx(0.6)
    assert -1.0 < best.threshold < -0.3
    per_request = (200 * 4 + 101 * 20) / 1e6
    assert ev.strong_only_cost_per_1k_usd == pytest.approx(per_request * 1000)
    assert best.cost_per_1k_usd == pytest.approx(per_request * 1000 * 2 / 5)


def test_cascade_curve_runs_from_all_local_to_all_remote(benchmarked):
    points = evaluate_cascade(benchmarked, SMALL, BIG).points
    assert points[0].local_share == 1.0 and points[0].score == pytest.approx(0.6)
    assert points[-1].local_share == 0.0 and points[-1].score == 1.0
    assert [p.local_share for p in points] == sorted((p.local_share for p in points), reverse=True)


def test_cascade_evaluation_needs_shared_items(store):
    record_run(store, SMALL, task_id="tool_calling")
    record_run(store, BIG, task_id="grounded_qa")
    with pytest.raises(ValueError, match="edgewise bench"):
        evaluate_cascade(store, SMALL, BIG)


def test_cascade_evaluation_no_threshold_reaches_target(store):
    record_run(store, SMALL, scores=[0, 0], logprobs=[-0.1, -0.1])  # confidently wrong
    record_run(store, BIG, scores=[1, 1])
    ev = evaluate_cascade(store, SMALL, BIG)
    # Both items share one confidence, so the cascade is all-or-nothing.
    assert [p.score for p in ev.points] == [0.0, 1.0]
    assert ev.recommended.local_share == 0.0


# ------------------------------------------------------------------ held-out evaluation


def big_benchmark(store, n=40):
    """n items: the local model is right exactly when its confidence is above -0.5."""
    logprobs = [-0.1 - (i % 10) * 0.1 for i in range(n)]  # -0.1 .. -1.0
    record_run(store, SMALL, scores=[1 if lp > -0.5 else 0 for lp in logprobs], logprobs=logprobs)
    record_run(store, BIG, scores=[1] * n)


def test_threshold_is_chosen_on_one_half_and_reported_on_the_other(store):
    big_benchmark(store)
    ev = evaluate_cascade(store, SMALL, BIG, target=1.0)
    held = ev.heldout
    assert held is not None and 10 <= held.n_items <= 30
    assert held.n_items < ev.n_items
    # The confidence signal is clean here, so the threshold found on one half holds on the other.
    assert -0.5 < ev.recommended.threshold < -0.4
    assert held.point.score == 1.0 and held.point.threshold == ev.recommended.threshold
    assert held.local_only_score < 1.0 and held.strong_only_score == 1.0


def test_split_is_stable_across_runs(store):
    big_benchmark(store)
    first = evaluate_cascade(store, SMALL, BIG).heldout
    assert first == evaluate_cascade(store, SMALL, BIG).heldout


def test_too_few_items_means_no_held_out_estimate(benchmarked):
    ev = evaluate_cascade(benchmarked, SMALL, BIG)
    assert ev.heldout is None and ev.recommended is not None


def test_held_out_score_can_be_lower_than_the_fitted_one(store):
    # Confidence carries no information about correctness: whatever looks good on the fitting
    # half is luck, and the held-out half shows it.
    import random

    rng = random.Random(7)
    n = 60
    record_run(
        store, SMALL, scores=[rng.choice([0, 1]) for _ in range(n)],
        logprobs=[-rng.random() for _ in range(n)],
    )  # fmt: skip
    record_run(store, BIG, scores=[1] * n)
    ev = evaluate_cascade(store, SMALL, BIG, target=0.8)
    assert ev.heldout is not None
    assert ev.heldout.point.score <= ev.recommended.score + 0.15
