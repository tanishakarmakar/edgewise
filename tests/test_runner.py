import pytest

from edgewise.backends.fake import FakeBackend
from edgewise.bench.runner import BenchConfig, BenchError, run_bench
from edgewise.catalog import Catalog
from edgewise.evals.tasks import TASKS
from edgewise.store import Store


def oracle_responder():
    """Answers every eval item correctly, to check the pipeline end to end."""
    answers = {}
    for item in TASKS["grounded_qa"].load_items():
        answers[item.prompt] = item.expected[0]
    for item in TASKS["short_reasoning"].load_items():
        answers[item.prompt] = f"Thinking... Answer: {item.expected}"

    def respond(messages):
        user = messages[-1].content
        for prompt, answer in answers.items():
            if prompt in user:
                return answer
        return "word " * 200

    return respond


@pytest.fixture
def setup(tmp_path):
    variant = Catalog.load().get("qwen2.5-0.5b-instruct@mlx-4bit")
    store = Store(tmp_path / "results.duckdb")
    config = BenchConfig(
        tasks=["grounded_qa", "short_reasoning"],
        prompt_lengths=[64, 256],
        repeats=3,
        warmup=1,
        perf_max_tokens=32,
        cooldown_s=0,
        budget_gb=4.0,
    )
    yield variant, store, config
    store.close()


def test_bench_end_to_end_with_fake_backend(setup):
    variant, store, config = setup
    backend = FakeBackend(responder=oracle_responder(), decode_ns_per_token=20_000_000)
    report = run_bench(variant, backend, config, store)

    # Performance: one summary per prompt length, decode speed = 1 token / 20 ms = 50 tok/s.
    assert [p.prompt_len for p in report.perf] == [64, 256]
    assert report.perf[0].decode_tps.median == pytest.approx(50)
    assert report.perf[1].ttft_ms.median > report.perf[0].ttft_ms.median  # longer prefill

    # Quality: the oracle gets everything right.
    assert {q.task_id: q.mean_score for q in report.quality} == {
        "grounded_qa": 1.0,
        "short_reasoning": 1.0,
    }

    # Everything persisted; the model was unloaded.
    assert store.count("runs", report.run_id) == 1
    assert store.count("perf", report.run_id) == 2 * 3
    assert store.count("quality", report.run_id) == 8 + 8
    assert backend.loaded == []


def test_bench_refuses_model_over_budget(setup):
    variant, store, config = setup
    backend = FakeBackend()
    with pytest.raises(BenchError, match="budget"):
        run_bench(variant, backend, config.model_copy(update={"budget_gb": 0.1}), store)
    assert backend.generations == 0


def test_bench_force_runs_anyway_and_warns(setup):
    variant, store, config = setup
    config = config.model_copy(update={"budget_gb": 0.1, "force": True, "tasks": []})
    report = run_bench(variant, FakeBackend(), config, store)
    assert any("tight" in w for w in report.warnings)


def test_speculative_needs_a_draft_and_counts_its_memory(setup):
    variant, store, config = setup
    spec = config.model_copy(update={"decoding": "speculative", "prompt_lengths": []})
    with pytest.raises(BenchError, match="draft"):
        run_bench(variant, FakeBackend(), spec, store)

    draft = Catalog.load().get("qwen2.5-0.5b-instruct@mlx-8bit")
    backend = FakeBackend(responder=oracle_responder())
    report = run_bench(variant, backend, spec, store, draft=draft)
    alone = run_bench(variant, FakeBackend(responder=oracle_responder()), config, store)
    assert report.est_memory_gb > alone.est_memory_gb
    assert report.draft_model_id == draft.id
    assert backend.last_settings.decoding == "speculative"
    assert backend.loaded == []  # target and draft both unloaded
    assert report.perf == []  # accelerated decodings are measured on task prompts
    row = store.query("SELECT decoding, draft_model_id FROM runs WHERE run_id = ?", [report.run_id])
    assert row == [{"decoding": "speculative", "draft_model_id": draft.id}]


def test_unloads_even_if_generation_fails(setup):
    variant, store, config = setup

    def explode(messages):
        raise RuntimeError("boom")

    backend = FakeBackend(responder=explode)
    with pytest.raises(RuntimeError):
        run_bench(variant, backend, config, store)
    assert backend.loaded == []


def test_memory_pressure_is_warned_about_and_recorded(setup, monkeypatch):
    import edgewise.bench.runner as runner

    variant, store, config = setup
    monkeypatch.setattr(runner, "swap_used_gb", lambda: 6.1)
    report = run_bench(variant, FakeBackend(), config.model_copy(update={"tasks": []}), store)
    assert any("6.1 GB of swap" in w for w in report.warnings)
    row = store.query("SELECT swap_used_gb FROM runs WHERE run_id = ?", [report.run_id])
    assert row == [{"swap_used_gb": 6.1}]


def test_eval_data_is_consistent():
    for task in TASKS.values():
        items = task.load_items()
        assert len(items) >= 6
        assert len({i.id for i in items}) == len(items)
        for item in items:
            assert task.build(item)[-1].role == "user"
    for item in TASKS["tool_calling"].load_items():
        names = {t["name"] for t in item.tools}
        assert item.expected["name"] in names
