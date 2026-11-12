import xml.etree.ElementTree as ET

import pytest
from conftest import record_perf, record_run
from typer.testing import CliRunner

from edgewise.catalog import Catalog
from edgewise.cli import app
from edgewise.report import svg
from edgewise.report.build import build_report, write_report

Q05, Q15, Q3 = (f"qwen2.5-{size}-instruct@mlx-4bit" for size in ("0.5b", "1.5b", "3b"))


@pytest.fixture
def populated(store):
    for model, scores, tps, peak in [
        (Q05, [1, 0, 0, 0], 120, 400),
        (Q15, [1, 1, 1, 0], 45, 1100),
        (Q3, [1, 1, 1, 1], 24, 2100),
    ]:
        run = record_run(
            store, model, scores=scores, tps=tps, peak_mb=peak,
            logprobs=[-0.1, -0.4, -0.9, -1.6], outputs=["a", "b", "c", "d"],
        )  # fmt: skip
        record_perf(store, run, model, {128: tps, 1024: tps * 0.9}, peak_mb=peak)
        record_run(store, model, task_id="text_edit", scores=[1, 1], tps=tps, peak_mb=peak,
                   outputs=["x", "y"])  # fmt: skip
    record_run(store, Q3, task_id="text_edit", scores=[1, 1], tps=48, decoding="prompt_lookup",
               k_policy="cost_aware", outputs=["x", "y"])  # fmt: skip
    record_run(store, Q3, task_id="text_edit", scores=[1, 1], tps=20, decoding="speculative",
               draft_model_id=Q05, k_policy="cost_aware", outputs=["x", "DIFFERENT"])  # fmt: skip
    return store


def parse(svg_text):
    return ET.fromstring(svg_text)  # raises if the SVG is not well-formed XML


def test_full_report(populated):
    r = build_report(populated, Catalog.load(), 4.5, local_model=Q05, strong_model=Q3)
    assert {"speed", "quality", "accelerator", "router", "tradeoff:tool_calling"} <= set(r.sections)
    md = r.markdown
    assert "| qwen2.5-3b 4bit | 157 |" in md  # speed table: 128 + 29 template tokens
    assert "tool_calling (n=4)" in md and "| qwen2.5-1.5b 4bit | 0.75 |" in md
    assert "2.00×" in md and "2/2" in md  # prompt lookup: twice as fast, identical outputs
    assert "0.83×" in md and "1/2" in md  # speculative: slower here, one output differed
    assert "upper bound" in md  # 4 shared items: too few to hold any out
    for name, text in r.charts.items():
        assert name.endswith(".svg") and parse(text).tag.endswith("svg")
        assert f"]({name})" in md
    assert {"cascade.svg", "tradeoff-tool_calling.svg"} <= set(r.charts)
    assert any(name.startswith("speedup-") for name in r.charts)


def test_memory_estimate_error_is_reported(populated):
    md = build_report(populated, Catalog.load(), 4.5).markdown
    assert "Memory estimate vs. measured peak" in md and "%" in md


def test_empty_store_says_what_to_run(store):
    r = build_report(store, Catalog.load(), 4.5)
    assert r.sections == [] and r.charts == {} and "edgewise bench" in r.markdown


def test_router_section_skipped_without_shared_items(populated):
    r = build_report(populated, Catalog.load(), 4.5, local_model=Q05, strong_model="nope")
    assert "router" not in r.sections


def test_write_report_and_cli(populated, tmp_path):
    r = build_report(populated, Catalog.load(), 4.5)
    paths = write_report(r, tmp_path / "out")
    assert (tmp_path / "out" / "REPORT.md").read_text() == r.markdown
    assert len(paths) == 1 + len(r.charts)


def test_cli_report(tmp_path):
    from edgewise.store import Store

    db = tmp_path / "cli.duckdb"
    with Store(db) as s:
        for model, scores, tps in [(Q05, [1, 0], 120), (Q3, [1, 1], 24)]:
            record_run(s, model, scores=scores, tps=tps)
    out = tmp_path / "published"
    result = CliRunner().invoke(app, ["report", "--db", str(db), "--out", str(out)])
    assert result.exit_code == 0
    assert (out / "REPORT.md").exists() and (out / "tradeoff-tool_calling.svg").exists()


# ------------------------------------------------------------------ chart building blocks


def test_nice_ticks_are_round_and_cover_the_range():
    assert svg.nice_ticks(0, 100) == [0, 20, 40, 60, 80, 100]
    assert svg.nice_ticks(0.5, 2.0, 5) == [0.5, 1.0, 1.5, 2.0]
    ticks = svg.nice_ticks(0, 126)
    assert ticks[0] == 0 and ticks[-1] >= 126


def test_short_name():
    assert svg.short_name("qwen2.5-1.5b-instruct@mlx-4bit") == "qwen2.5-1.5b 4bit"


def test_tradeoff_chart_labels_only_the_frontier():
    points = [(Q05, 120, 0.4, True), (Q15, 45, 0.8, True), (Q3, 24, 1.0, True),
              ("llama3.2-1b-instruct@mlx-4bit", 40, 0.5, False)]  # fmt: skip
    text = svg.tradeoff_chart(points, "tool_calling", recommended=Q15)
    parse(text)
    assert ">qwen2.5-1.5b 4bit (recommended)<" in text
    assert ">llama3.2-1b 4bit<" not in text  # dominated: tooltip and table only
    assert "llama3.2-1b 4bit: score 0.50" in text


def test_speedup_chart_handles_slowdowns_and_caps_series():
    data = {"text_edit": {"a": 2.0, "b": 0.7, "c": 1.2, "d": 3.0}}
    text = svg.speedup_chart(data, ["a", "b", "c", "d"], Q3)
    parse(text)
    assert "2.00×" in text and "0.70×" in text and "3.00×" not in text


def test_cascade_chart_marks_the_recommended_point():
    text = svg.cascade_chart([(1.0, 0.6), (0.6, 0.95), (0.0, 1.0)], (0.6, 0.95), Q05, Q3)
    parse(text)
    assert "recommended: 60% local, score 0.95" in text
    assert "everything local: 0.60" in text and "everything escalated: 1.00" in text


def test_crowded_labels_are_dropped_not_stacked():
    # Eight frontier points almost on top of each other: only some labels can fit.
    points = [(f"model-{i}@mlx-4bit", 50 + i * 0.5, 0.5 + i * 0.002, True) for i in range(8)]
    text = svg.tradeoff_chart(points, "tool_calling")
    labels = [line for line in text.splitlines() if line.startswith("<text") and "model-" in line]
    assert 1 <= len(labels) < 8
    assert text.count("<circle") == 8 + 2  # every point is still drawn (plus 2 legend dots)


def test_report_uses_held_out_items_when_there_are_enough(store):
    logprobs = [-0.1 - (i % 10) * 0.1 for i in range(40)]
    record_run(store, Q05, scores=[1 if lp > -0.5 else 0 for lp in logprobs], logprobs=logprobs)
    record_run(store, Q3, scores=[1] * 40)
    md = build_report(store, Catalog.load(), 4.5, local_model=Q05, strong_model=Q3).markdown
    assert "played no part in choosing it" in md and "upper bound" not in md
