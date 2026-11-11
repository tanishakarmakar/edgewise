"""Build the published report: Markdown tables + SVG charts from the results store.

Everything here is derived from measurements. A section with no data is left out, with a note
on how to produce it, rather than filled with placeholders.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from edgewise.advisor import advise
from edgewise.catalog import Catalog
from edgewise.evals import TASKS
from edgewise.report import svg
from edgewise.router import evaluate_cascade
from edgewise.store import Store

# Latest standard-decoding run per model that has a raw speed pass.
SPEED_SQL = """
WITH latest AS (
    SELECT p.model_id, max(r.started_at) AS started_at
    FROM perf p JOIN runs r USING (run_id)
    WHERE r.decoding = 'standard'
    GROUP BY p.model_id
)
SELECT p.model_id, p.prompt_len, r.on_ac_power, r.swap_used_gb, r.est_memory_gb,
       median(p.prompt_tokens) AS prompt_tokens,
       median(p.ttft_ms) AS ttft_ms,
       median(p.decode_tps) AS decode_tps,
       max(p.peak_memory_mb) AS peak_memory_mb
FROM perf p
JOIN runs r USING (run_id)
JOIN latest l ON l.model_id = p.model_id AND l.started_at = r.started_at
GROUP BY p.model_id, p.prompt_len, r.on_ac_power, r.swap_used_gb, r.est_memory_gb
ORDER BY p.model_id, p.prompt_len
"""

# Latest run per (model, task, decoding configuration), with per-task speed.
TASK_SQL = """
WITH latest AS (
    SELECT q.model_id, q.task_id, r.decoding, r.draft_model_id, r.k_policy,
           max(r.started_at) AS started_at
    FROM quality q JOIN runs r USING (run_id)
    GROUP BY ALL
)
SELECT q.model_id, q.task_id, r.decoding, r.draft_model_id, r.k_policy, r.run_id,
       count(*) AS n_items,
       avg(q.score) AS score,
       sum(greatest(q.completion_tokens - 1, 0)) / nullif(sum(q.decode_ms) / 1000, 0) AS decode_tps,
       sum(q.accepted_draft_tokens) / nullif(sum(q.proposed_draft_tokens), 0) AS acceptance
FROM quality q
JOIN runs r USING (run_id)
JOIN latest l ON l.model_id = q.model_id AND l.task_id = q.task_id
    AND l.decoding = r.decoding AND l.started_at = r.started_at
GROUP BY ALL
"""

# How many items produced exactly the same text in two runs.
IDENTICAL_SQL = """
SELECT count(*) AS n, sum(CAST(a.output_text = b.output_text AS INTEGER)) AS same
FROM quality a JOIN quality b USING (task_id, item_id)
WHERE a.run_id = ? AND b.run_id = ?
"""


class Report(BaseModel):
    markdown: str
    charts: dict[str, str]  # file name -> SVG text
    sections: list[str]


def _table(headers: list[str], rows: list[list[str]], align_right_from: int = 1) -> str:
    sep = ["---" if i < align_right_from else "---:" for i in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _decoding_label(row: dict) -> str:
    if row["decoding"] == "speculative":
        return f"speculative (draft {svg.short_name(row['draft_model_id'])})"
    return row["decoding"].replace("_", " ")


def build_report(
    store: Store,
    catalog: Catalog,
    budget_gb: float,
    local_model: str | None = None,
    strong_model: str | None = None,
) -> Report:
    parts: list[str] = ["# Edgewise results", ""]
    charts: dict[str, str] = {}
    sections: list[str] = []

    speed = store.query(SPEED_SQL)
    task_rows = store.query(TASK_SQL)
    standard = [r for r in task_rows if r["decoding"] == "standard"]
    battery = any(r["on_ac_power"] is False for r in speed)
    pressured = any((r["swap_used_gb"] or 0) > 1.0 for r in speed)

    # ---------------------------------------------------------------- speed and memory
    if speed:
        sections.append("speed")
        parts += ["## Speed and memory", ""]
        if battery:
            parts += [
                "> Some of these runs were measured on battery power, which can understate speed.",
                "",
            ]
        if pressured:
            parts += [
                "> Some of these runs were measured while the system was swapping memory. "
                "Treat their speed as unreliable and re-run with other apps closed.",
                "",
            ]
        parts.append(
            _table(
                ["Model", "Prompt tokens", "First token (ms)", "Tokens/s", "Peak memory (MB)"],
                [
                    [
                        svg.short_name(r["model_id"]),
                        f"{r['prompt_tokens']:.0f}",
                        f"{r['ttft_ms']:.0f}",
                        f"{r['decode_tps']:.1f}",
                        f"{r['peak_memory_mb']:.0f}",
                    ]
                    for r in speed
                ],
            )
        )
        peaks: dict[str, tuple[float, float]] = {}
        for r in speed:
            est, peak = r["est_memory_gb"], r["peak_memory_mb"] / 1000
            if r["model_id"] not in peaks or peak > peaks[r["model_id"]][1]:
                peaks[r["model_id"]] = (est, peak)
        parts += ["", "### Memory estimate vs. measured peak", ""]
        parts.append(
            _table(
                ["Model", "Estimated (GB)", "Measured peak (GB)", "Error"],
                [
                    [svg.short_name(m), f"{est:.2f}", f"{peak:.2f}", f"{(est - peak) / peak:+.0%}"]
                    for m, (est, peak) in peaks.items()
                ],
            )
        )
        parts.append("")

    # ---------------------------------------------------------------- quality
    if standard:
        sections.append("quality")
        tasks = [t for t in TASKS if any(r["task_id"] == t for r in standard)]
        scores = {(r["model_id"], r["task_id"]): r for r in standard}
        models = list(dict.fromkeys(v.id for v in catalog if any(k[0] == v.id for k in scores)))
        n_items = {t: max(r["n_items"] for r in standard if r["task_id"] == t) for t in tasks}
        parts += ["## Task quality", "", "Mean score per task (0 to 1), standard decoding.", ""]
        parts.append(
            _table(
                ["Model"] + [f"{t} (n={n_items[t]})" for t in tasks],
                [
                    [svg.short_name(m)]
                    + [f"{scores[m, t]['score']:.2f}" if (m, t) in scores else "—" for t in tasks]
                    for m in models
                ],
            )
        )
        parts += [
            "",
            "Item counts are small: one item moves a score by "
            f"{1 / max(n_items.values()):.2f} to {1 / min(n_items.values()):.2f}.",
            "",
        ]

        # One trade-off chart per task that has at least two measured models.
        for task in tasks:
            rec = advise(catalog, store, task, budget_gb)
            measured = [r for r in rec.rows if r.evidence is not None and r.status != "too big"]
            if len(measured) < 2:
                continue
            name = f"tradeoff-{task}.svg"
            charts[name] = svg.tradeoff_chart(
                [
                    (r.model_id, r.evidence.decode_tps, r.evidence.score, r.on_frontier)
                    for r in measured
                ],
                task,
                recommended=rec.choice.model_id if rec.choice else None,
            )
            parts += [f"![Quality vs. speed for {task}]({name})", ""]
            sections.append(f"tradeoff:{task}")

    # ---------------------------------------------------------------- accelerator
    accelerated = [r for r in task_rows if r["decoding"] != "standard"]
    baseline = {(r["model_id"], r["task_id"]): r for r in standard}
    if accelerated:
        rows, by_model = [], {}
        for r in accelerated:
            base = baseline.get((r["model_id"], r["task_id"]))
            if base is None or not base["decode_tps"] or not r["decode_tps"]:
                continue
            same = store.query(IDENTICAL_SQL, [r["run_id"], base["run_id"]])[0]
            speedup = r["decode_tps"] / base["decode_tps"]
            label = _decoding_label(r)
            by_model.setdefault(r["model_id"], {}).setdefault(r["task_id"], {})[label] = speedup
            rows.append(
                [
                    svg.short_name(r["model_id"]),
                    label,
                    r["task_id"],
                    f"{base['decode_tps']:.1f}",
                    f"{r['decode_tps']:.1f}",
                    f"{speedup:.2f}×",
                    f"{r['acceptance']:.0%}" if r["acceptance"] is not None else "—",
                    f"{same['same']}/{same['n']}",
                ]
            )
        if rows:
            sections.append("accelerator")
            parts += [
                "## Accelerated decoding",
                "",
                "Same model, same prompts. “Identical outputs” counts items whose text matches "
                "standard decoding exactly.",
                "",
                _table(
                    ["Model", "Decoding", "Task", "Standard tok/s", "Accelerated tok/s",
                     "Speed-up", "Guesses accepted", "Identical outputs"],
                    sorted(rows),
                    align_right_from=3,
                ),
                "",
            ]  # fmt: skip
            for model_id, speedups in by_model.items():
                series = list(dict.fromkeys(name for t in speedups.values() for name in t))
                name = f"speedup-{model_id.replace('@', '-')}.svg"
                charts[name] = svg.speedup_chart(speedups, series, model_id)
                parts += [f"![Decoding speed-up for {svg.short_name(model_id)}]({name})", ""]

    # ---------------------------------------------------------------- router
    if local_model and strong_model:
        try:
            ev = evaluate_cascade(store, local_model, strong_model)
        except ValueError:
            ev = None
        if ev is not None:
            sections.append("router")
            charts["cascade.svg"] = svg.cascade_chart(
                [(p.local_share, p.score) for p in ev.points],
                (ev.recommended.local_share, ev.recommended.score) if ev.recommended else None,
                local_model,
                strong_model,
            )
            parts += [
                "## Local-first routing",
                "",
                f"Replayed from stored results on {ev.n_items} items "
                f"({', '.join(ev.tasks)}): {svg.short_name(local_model)} answers first and "
                f"escalates to {svg.short_name(strong_model)} when its confidence is low.",
                "",
                _table(
                    ["Policy", "Answered on device", "Mean score"],
                    [
                        ["Local only", "100%", f"{ev.local_only_score:.2f}"],
                        ["Escalate everything", "0%", f"{ev.strong_only_score:.2f}"],
                    ]
                    + (
                        [
                            [
                                f"Cascade (escalate below {ev.recommended.threshold:.2f})",
                                f"{ev.recommended.local_share:.0%}",
                                f"{ev.recommended.score:.2f}",
                            ]
                        ]
                        if ev.recommended
                        else []
                    ),
                ),
                "",
                "The threshold was chosen on the same items it is reported on, so this is an "
                "upper bound; a held-out set is needed for a fair estimate.",
                "",
                "![Cascade quality vs. share answered locally](cascade.svg)",
                "",
            ]

    if not sections:
        parts += [
            "No benchmark results yet. Run, for example:",
            "",
            "```bash",
            "edgewise bench -m qwen2.5-0.5b-instruct@mlx-4bit --all-tasks",
            "```",
            "",
        ]
    return Report(markdown="\n".join(parts), charts=charts, sections=sections)


def write_report(report: Report, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "REPORT.md"]
    written[0].write_text(report.markdown)
    for name, text in report.charts.items():
        (out_dir / name).write_text(text)
        written.append(out_dir / name)
    return written
