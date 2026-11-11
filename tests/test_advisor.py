import pytest
from conftest import record_run
from typer.testing import CliRunner

from edgewise.advisor import Evidence, advise, pareto_frontier
from edgewise.catalog import Catalog
from edgewise.cli import _parse_gb, app

Q05_4, Q05_8 = "qwen2.5-0.5b-instruct@mlx-4bit", "qwen2.5-0.5b-instruct@mlx-8bit"
Q15_4, Q15_8 = "qwen2.5-1.5b-instruct@mlx-4bit", "qwen2.5-1.5b-instruct@mlx-8bit"
Q3_4, Q3_8 = "qwen2.5-3b-instruct@mlx-4bit", "qwen2.5-3b-instruct@mlx-8bit"


@pytest.fixture(scope="module")
def catalog():
    return Catalog.load()


@pytest.fixture
def benchmarked(store):
    """A plausible spread: bigger models score higher and run slower; 8-bit is slower than 4-bit."""
    record_run(store, Q05_4, scores=[1, 1, 0, 0, 0], tps=120, peak_mb=400)  # 0.4
    record_run(store, Q05_8, scores=[1, 1, 1, 0, 0], tps=90, peak_mb=700)  # 0.6
    record_run(store, Q15_4, scores=[1, 1, 1, 1, 0], tps=45, peak_mb=1100)  # 0.8
    record_run(store, Q15_8, scores=[1, 1, 1, 1, 0], tps=30, peak_mb=1900)  # 0.8, slower: dominated
    record_run(store, Q3_4, scores=[1, 1, 1, 1, 1], tps=24, peak_mb=2100)  # 1.0
    record_run(store, Q3_8, scores=[1, 1, 1, 1, 1], tps=14, peak_mb=3900)  # 1.0, slower: dominated
    return store


def status(rec):
    return {r.model_id: r.status for r in rec.rows}


def ev(model_id, score, tps):
    return Evidence(
        model_id=model_id, run_id="r", on_ac_power=True, n_items=5, score=score, valid_rate=1,
        decode_tps=tps, ttft_ms=1, peak_memory_mb=1,
    )  # fmt: skip


def test_pareto_frontier_drops_dominated_points():
    points = [ev("fast", 0.5, 100), ev("good", 0.9, 20), ev("worse", 0.5, 50), ev("mid", 0.7, 60)]
    assert pareto_frontier(points) == {"fast", "good", "mid"}


def test_pareto_keeps_ties():
    assert pareto_frontier([ev("a", 0.5, 50), ev("b", 0.5, 50)]) == {"a", "b"}


def test_balanced_picks_best_quality_times_speed(catalog, benchmarked):
    rec = advise(catalog, benchmarked, "tool_calling", budget_gb=4.5)
    # balanced = score × speed/fastest: 0.5b-8bit → 0.6×0.75 = 0.45 beats 0.5b-4bit → 0.40
    assert rec.choice.model_id == Q05_8
    s = status(rec)
    assert s[Q15_8] == "dominated" and s[Q3_8] == "dominated"
    assert rec.runner_up is not None


def test_prefer_quality_and_speed(catalog, benchmarked):
    assert (
        advise(catalog, benchmarked, "tool_calling", 4.5, prefer="quality").choice.model_id == Q3_4
    )
    assert (
        advise(catalog, benchmarked, "tool_calling", 4.5, prefer="speed").choice.model_id == Q05_4
    )


def test_quality_floor_removes_weak_models(catalog, benchmarked):
    rec = advise(catalog, benchmarked, "tool_calling", 4.5, min_score=0.75, prefer="speed")
    assert rec.choice.model_id == Q15_4  # fastest model that scores at least 0.75
    assert status(rec)[Q05_4] == "below quality floor"


def test_memory_budget_uses_measured_peak(catalog, benchmarked):
    rec = advise(catalog, benchmarked, "tool_calling", budget_gb=2.0, prefer="quality")
    assert rec.choice.model_id == Q15_4  # the 3B models peak above 2 GB
    assert status(rec)[Q3_4] == "too big"
    row = next(r for r in rec.rows if r.model_id == Q3_4)
    assert row.memory_measured and row.memory_gb == pytest.approx(2.1)


def test_unbenchmarked_models_are_listed_but_never_ranked(catalog, benchmarked):
    rec = advise(catalog, benchmarked, "tool_calling", 4.5)
    llama = [r for r in rec.rows if r.model_id.startswith("llama")]
    assert llama and all(r.status in ("not benchmarked", "too big") for r in llama)
    assert all(not r.memory_measured for r in llama)


def test_no_data_explains_what_to_run(catalog, store):
    rec = advise(catalog, store, "tool_calling", 4.5)
    assert rec.choice is None
    assert "edgewise bench -m" in rec.reasons[0]


def test_score_relative_to_higher_precision(catalog, benchmarked):
    rec = advise(catalog, benchmarked, "tool_calling", 4.5)
    rows = {r.model_id: r for r in rec.rows}
    assert rows[Q05_4].vs_full_precision == pytest.approx(0.4 / 0.6)
    assert rows[Q05_8].vs_full_precision is None  # it is the reference


def test_uses_latest_run_and_ignores_accelerated_runs(catalog, store):
    record_run(store, Q05_4, scores=[0, 0], tps=10)
    record_run(store, Q05_4, scores=[1, 1], tps=100)  # newer
    record_run(store, Q05_4, scores=[1, 1], tps=500, decoding="prompt_lookup")
    rec = advise(catalog, store, "tool_calling", 4.5)
    assert rec.choice.evidence.score == 1 and rec.choice.evidence.decode_tps == pytest.approx(100)


def test_reasons_flag_small_samples_and_battery(catalog, store):
    record_run(store, Q05_4, scores=[1, 0], on_ac_power=False)
    text = " ".join(advise(catalog, store, "tool_calling", 4.5).reasons)
    assert "only 2 test items" in text and "battery" in text


def test_parse_gb():
    assert _parse_gb("4GB") == 4 and _parse_gb("4.5 gb") == 4.5
    assert _parse_gb("512MB") == pytest.approx(0.512) and _parse_gb("3") == 3


def test_cli_advise(tmp_path, catalog):
    from edgewise.store import Store

    db = tmp_path / "r.duckdb"
    with Store(db) as s:
        record_run(s, Q15_4, scores=[1, 1, 1, 0], tps=45, peak_mb=1100)
    runner = CliRunner()
    result = runner.invoke(
        app, ["advise", "-t", "tool_calling", "--memory", "4GB", "--db", str(db)]
    )
    assert result.exit_code == 0 and f"→ {Q15_4}" in result.stdout
    assert runner.invoke(app, ["advise", "-t", "nope", "--db", str(db)]).exit_code == 1
    empty = runner.invoke(app, ["advise", "-t", "grounded_qa", "--db", str(db)])
    assert empty.exit_code == 1 and "edgewise bench" in empty.stdout
