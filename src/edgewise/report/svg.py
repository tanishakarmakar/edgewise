"""Static SVG charts for the README, written by hand so reports need no plotting library.

Each chart is one self-contained file that follows the viewer's light/dark setting. Colors are
the first slots of a categorical palette validated for color-vision deficiency in both modes;
text always uses ink colors, never a series color. Values that are not labeled on a chart are
in the tables of the same report.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from xml.sax.saxutils import escape

WIDTH = 720
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
CHAR_W = 6.3  # average glyph width at 12px, for label placement

STYLE = f"""
:root {{
  --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --deemph: #c3c2b7;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --deemph: #52514e;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  }}
}}
text {{ font-family: {FONT}; font-size: 12px; fill: var(--ink2); }}
.title {{ font-size: 15px; font-weight: 600; fill: var(--ink); }}
.strong {{ fill: var(--ink); font-weight: 600; }}
.muted {{ fill: var(--muted); }}
.tick {{ fill: var(--muted); font-variant-numeric: tabular-nums; }}
.bg {{ fill: var(--surface); }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.axis {{ stroke: var(--axis); stroke-width: 1; }}
.line {{ fill: none; stroke: var(--s1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.ring {{ stroke: var(--surface); stroke-width: 2; }}
.s1 {{ fill: var(--s1); }} .s2 {{ fill: var(--s2); }} .s3 {{ fill: var(--s3); }}
.deemph {{ fill: var(--deemph); }}
"""


def _doc(height: int, title: str, subtitle: str, body: list[str], description: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
            f'width="{WIDTH}" height="{height}" role="img">',
            f"<title>{escape(title)}</title>",
            f"<desc>{escape(description)}</desc>",
            f"<style>{STYLE}</style>",
            f'<rect class="bg" width="{WIDTH}" height="{height}" rx="8"/>',
            f'<text class="title" x="24" y="30">{escape(title)}</text>',
            f'<text x="24" y="50">{escape(subtitle)}</text>',
            *body,
            "</svg>\n",
        ]
    )


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round tick values covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * magnitude >= raw), default=10) * magnitude
    first = math.floor(lo / step) * step
    ticks = []
    value = first
    while value < hi + step * 0.999:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _fmt(value: float) -> str:
    return f"{value:g}"


def short_name(model_id: str) -> str:
    """qwen2.5-1.5b-instruct@mlx-4bit -> qwen2.5-1.5b 4bit"""
    name, _, quant = model_id.partition("@")
    return f"{name.replace('-instruct', '')} {quant.replace('mlx-', '')}".strip()


# ------------------------------------------------------------------ quality vs speed


Box = tuple[float, float, float, float]  # x0, y0, x1, y1


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _place_label(x: float, y: float, text: str, taken: list[Box], bounds: Box):
    """Find a free spot for a point's label: (x, y, anchor), or None if every spot is taken.

    A label that doesn't fit is dropped rather than stacked on another one; the value stays
    available in the tooltip and the report table.
    """
    w, h = len(text) * CHAR_W, 14
    candidates = [
        (x + 10, y - 8, "start"),
        (x + 10, y + 18, "start"),
        (x - 10, y - 8, "end"),
        (x - 10, y + 18, "end"),
        (x, y - 14, "middle"),
        (x, y + 26, "middle"),
    ]
    for lx, ly, anchor in candidates:
        x0 = {"start": lx, "end": lx - w, "middle": lx - w / 2}[anchor]
        box = (x0, ly - h + 3, x0 + w, ly + 3)
        inside = (
            bounds[0] <= box[0]
            and box[2] <= bounds[2]
            and bounds[1] <= box[1]
            and box[3] <= bounds[3]
        )
        if inside and not any(_overlaps(box, other) for other in taken):
            taken.append(box)
            return lx, ly, anchor
    return None


def tradeoff_chart(
    points: Sequence[tuple[str, float, float, bool]], task_id: str, recommended: str | None = None
) -> str:
    """Scatter of score against speed. points: (model_id, tokens_per_s, score, on_frontier).

    Emphasis rather than one color per model: the models worth considering (the Pareto
    frontier) are colored and labeled; dominated ones recede to gray.
    """
    left, right, top, bottom = 56, 28, 116, 52
    height = 430
    plot_w, plot_h = WIDTH - left - right, height - top - bottom
    x_ticks = nice_ticks(0, max(p[1] for p in points) * 1.05)
    y_lo = min(0.0, *(p[2] for p in points))
    y_ticks = nice_ticks(y_lo, 1.0)
    x_min, x_max, y_min, y_max = x_ticks[0], x_ticks[-1], y_ticks[0], y_ticks[-1]

    def sx(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v: float) -> float:
        return top + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    # Legend: always present for two groups; the second entry starts after the first one's text.
    first = "Worth considering (nothing else is both faster and better)"
    second_x = 42 + len(first) * CHAR_W + 24
    body = [
        f'<circle class="s1" cx="30" cy="70" r="5"/><text x="42" y="74">{first}</text>',
        f'<circle class="deemph" cx="{second_x:.0f}" cy="70" r="5"/>'
        f'<text x="{second_x + 12:.0f}" y="74">Beaten by another model</text>',
    ]
    for t in y_ticks:
        body.append(
            f'<line class="grid" x1="{left}" x2="{WIDTH - right}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
        )
        body.append(
            f'<text class="tick" x="{left - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{_fmt(t)}</text>'
        )
    for t in x_ticks:
        body.append(
            f'<text class="tick" x="{sx(t):.1f}" y="{top + plot_h + 18}" text-anchor="middle">{_fmt(t)}</text>'
        )
    body.append(
        f'<line class="axis" x1="{left}" x2="{WIDTH - right}" y1="{top + plot_h}" y2="{top + plot_h}"/>'
    )
    body.append(
        f'<text class="muted" x="{left + plot_w / 2:.0f}" y="{height - 12}" text-anchor="middle">Speed (tokens per second)</text>'
    )
    body.append(f'<text class="muted" x="{left - 40}" y="{top - 22}">Task score</text>')

    dominated = [p for p in points if not p[3]]
    frontier = sorted((p for p in points if p[3]), key=lambda p: p[1])
    for model_id, tps, score, _ in dominated:
        body.append(
            f'<circle class="deemph ring" cx="{sx(tps):.1f}" cy="{sy(score):.1f}" r="5">'
            f"<title>{escape(short_name(model_id))}: score {score:.2f}, {tps:.0f} tokens/s</title></circle>"
        )
    if len(frontier) > 1:
        path = " ".join(f"{sx(p[1]):.1f},{sy(p[2]):.1f}" for p in frontier)
        body.append(f'<polyline class="line" style="stroke-opacity:.35" points="{path}"/>')
    for model_id, tps, score, _ in frontier:
        body.append(
            f'<circle class="s1 ring" cx="{sx(tps):.1f}" cy="{sy(score):.1f}" r="6">'
            f"<title>{escape(short_name(model_id))}: score {score:.2f}, {tps:.0f} tokens/s</title></circle>"
        )

    # Labels: every dot is an obstacle; the recommended model gets first pick of the free spots.
    taken: list[Box] = [(sx(p[1]) - 8, sy(p[2]) - 8, sx(p[1]) + 8, sy(p[2]) + 8) for p in points]
    taken.append((left - 40, top - 34, left + 30, top - 18))  # the "Task score" axis title
    # Labels may use the head-room above the plot and the right margin, but not the tick column.
    bounds = (left, top - 32, WIDTH - 8, top + plot_h - 2)
    for model_id, tps, score, _ in sorted(frontier, key=lambda p: (p[0] != recommended, -p[2])):
        is_pick = model_id == recommended
        text = short_name(model_id) + (" (recommended)" if is_pick else "")
        spot = _place_label(sx(tps), sy(score), text, taken, bounds)
        if spot is not None:
            lx, ly, anchor = spot
            cls = ' class="strong"' if is_pick else ""
            body.append(
                f'<text{cls} x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">{escape(text)}</text>'
            )

    return _doc(
        height,
        f"Quality vs. speed: {task_id}",
        "Each dot is one model variant measured on this device. Up and to the right is better.",
        body,
        f"Scatter plot of task score against decoding speed for {len(points)} model variants "
        f"on the {task_id} task. {len(frontier)} are on the Pareto frontier.",
    )


# ------------------------------------------------------------------ speed-up bars


def _hbar(x0: float, x1: float, y: float, h: float, r: float = 4) -> str:
    """Horizontal bar from baseline x0 to x1: square at the baseline, rounded at the data end."""
    r = min(r, abs(x1 - x0), h / 2)
    d = 1 if x1 >= x0 else -1
    return (
        f"M{x0:.1f},{y:.1f} H{x1 - d * r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
        f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - d * r:.1f},{y + h:.1f} H{x0:.1f} Z"
    )


def speedup_chart(
    speedups: dict[str, dict[str, float]], series: Sequence[str], model_id: str
) -> str:
    """Bars growing from the 1x baseline. speedups[task][series_name] = speed ÷ standard speed.

    At most three series (the palette slots validated for every pairing).
    """
    series = list(series)[:3]
    tasks = list(speedups)
    bar_h, gap, row_pad = 14, 2, 18
    row_h = len(series) * (bar_h + gap) - gap + row_pad
    left, right, top = 150, 64, 96
    height = top + len(tasks) * row_h + 44
    plot_w = WIDTH - left - right
    values = [v for t in tasks for v in speedups[t].values()]
    ticks = nice_ticks(min(0.5, *values), max(2.0, *values), 5)
    lo, hi = ticks[0], ticks[-1]

    def sx(v: float) -> float:
        return left + (v - lo) / (hi - lo) * plot_w

    body = []
    x = 24
    for i, name in enumerate(series):  # legend
        body.append(f'<rect class="s{i + 1}" x="{x}" y="64" width="12" height="12" rx="2"/>')
        body.append(f'<text x="{x + 18}" y="74">{escape(name)}</text>')
        x += 18 + len(name) * CHAR_W + 24
    for t in ticks:
        cls = "axis" if t == 1 else "grid"
        body.append(
            f'<line class="{cls}" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 6}" y2="{height - 40}"/>'
        )
        label = "1× (standard)" if t == 1 else f"{_fmt(t)}×"
        body.append(
            f'<text class="tick" x="{sx(t):.1f}" y="{height - 22}" text-anchor="middle">{label}</text>'
        )

    for row, task in enumerate(tasks):
        y0 = top + row * row_h
        block_h = len(series) * (bar_h + gap) - gap
        body.append(
            f'<text x="{left - 12}" y="{y0 + block_h / 2 + 4:.1f}" text-anchor="end">{escape(task)}</text>'
        )
        for i, name in enumerate(series):
            value = speedups[task].get(name)
            if value is None:
                continue
            y = y0 + i * (bar_h + gap)
            body.append(
                f'<path class="s{i + 1}" d="{_hbar(sx(1), sx(value), y, bar_h)}">'
                f"<title>{escape(task)}, {escape(name)}: {value:.2f}× standard speed</title></path>"
            )
            # A slower bar grows left towards the task names, so its value goes on the
            # empty side of the baseline instead of at the bar's end.
            anchor, lx = "start", max(sx(value), sx(1)) + 6
            body.append(
                f'<text class="tick" x="{lx:.1f}" y="{y + bar_h - 3:.1f}" text-anchor="{anchor}">{value:.2f}×</text>'
            )

    return _doc(
        height,
        f"Decoding speed-up: {short_name(model_id)}",
        "Decode speed relative to standard decoding, same model and prompts. Bars left of 1× are slower.",
        body,
        f"Bar chart of decoding speed relative to standard decoding for {len(tasks)} tasks "
        f"and {len(series)} accelerated decoding modes.",
    )


# ------------------------------------------------------------------ cascade curve


def cascade_chart(
    points: Sequence[tuple[float, float]],
    recommended: tuple[float, float] | None,
    local_model: str,
    strong_model: str,
) -> str:
    """Line of cascade score against the share of requests answered on the device.

    points: (local_share 0..1, score), any order.
    """
    left, right, top, bottom = 56, 36, 104, 52
    height = 400
    plot_w, plot_h = WIDTH - left - right, height - top - bottom
    pts = sorted(points)
    y_ticks = nice_ticks(min(p[1] for p in pts), max(1.0, max(p[1] for p in pts)), 4)
    y_min, y_max = y_ticks[0], y_ticks[-1]

    def sx(v: float) -> float:
        return left + v * plot_w

    def sy(v: float) -> float:
        return top + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    body = []
    for t in y_ticks:
        body.append(
            f'<line class="grid" x1="{left}" x2="{WIDTH - right}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
        )
        body.append(
            f'<text class="tick" x="{left - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{_fmt(t)}</text>'
        )
    for share in (0, 0.25, 0.5, 0.75, 1):
        body.append(
            f'<text class="tick" x="{sx(share):.1f}" y="{top + plot_h + 18}" text-anchor="middle">{share:.0%}</text>'
        )
    body.append(
        f'<line class="axis" x1="{left}" x2="{WIDTH - right}" y1="{top + plot_h}" y2="{top + plot_h}"/>'
    )
    body.append(
        f'<text class="muted" x="{left + plot_w / 2:.0f}" y="{height - 12}" text-anchor="middle">Share of requests answered on the device</text>'
    )
    body.append(f'<text class="muted" x="{left - 40}" y="{top - 30}">Mean task score</text>')

    path = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in pts)
    body.append(f'<polyline class="line" points="{path}"/>')

    def end_label(share: float, value: float) -> str:
        where = {0.0: "everything escalated", 1.0: "everything local"}.get(
            share, f"{share:.0%} local"
        )
        return f"{where}: {value:.2f}"

    for (share, value), anchor, text in (
        (pts[0], "start", end_label(*pts[0])),
        (pts[-1], "end", end_label(*pts[-1])),
    ):
        if recommended is not None and share == recommended[0]:
            continue  # the recommended label below already names this point
        body.append(f'<circle class="s1 ring" cx="{sx(share):.1f}" cy="{sy(value):.1f}" r="5"/>')
        dx = 10 if anchor == "start" else -10
        body.append(
            f'<text x="{sx(share) + dx:.1f}" y="{sy(value) + 20:.1f}" text-anchor="{anchor}">{escape(text)}</text>'
        )
    if recommended is not None:
        share, value = recommended
        text = f"recommended: {share:.0%} local, score {value:.2f}"
        anchor, dx = ("end", -10) if share > 0.5 else ("start", 10)
        x_text = sx(share) + dx
        x_far = x_text + (-1 if anchor == "end" else 1) * len(text) * 7.6
        x_lo, x_hi = sorted((x_text, x_far))

        def line_y(x: float) -> float:
            for (s0, v0), (s1, v1) in zip(pts, pts[1:], strict=False):
                if sx(s0) <= x <= sx(s1):
                    t = (x - sx(s0)) / (sx(s1) - sx(s0)) if s1 > s0 else 0.0
                    return sy(v0) + t * (sy(v1) - sy(v0))
            return sy(pts[-1][1])

        def clearance(baseline: float) -> float:
            # Distance from the middle of the text to the line, over the text's width.
            xs = [x_lo + (x_hi - x_lo) * i / 20 for i in range(21)]
            return min(abs(line_y(x) - (baseline - 5)) for x in xs)

        # Above the marker or below it: whichever keeps the text further from the line.
        y_text = max((sy(value) - 14, sy(value) + 26), key=clearance)
        body.append(f'<circle class="s1 ring" cx="{sx(share):.1f}" cy="{sy(value):.1f}" r="7"/>')
        body.append(
            f'<text class="strong" x="{x_text:.1f}" y="{y_text:.1f}" text-anchor="{anchor}">{escape(text)}</text>'
        )

    return _doc(
        height,
        "Local-first routing: quality vs. how much stays on the device",
        f"{short_name(local_model)} answers first; unsure answers escalate to {short_name(strong_model)}.",
        body,
        "Line chart of mean task score against the share of requests answered locally, "
        "as the escalation confidence threshold varies.",
    )
