// Edgewise demo page. Plain JavaScript, no dependencies, nothing loaded from the network.
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHtml = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const VIEWS = ["router", "advisor", "accelerator"];
const CHECK = '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M3.5 8.5 6.5 11.5 12.5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const LOCK = '<svg viewBox="0 0 16 16" width="13" height="13"><rect x="3.5" y="7" width="9" height="6.5" rx="1.5" fill="currentColor"/><path d="M5.5 7V5.5a2.5 2.5 0 0 1 5 0V7" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>';
const UP = '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M8 12.5v-9M4.5 7 8 3.5 11.5 7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

let overview = null;

// ------------------------------------------------------------------ plumbing

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (e) { /* not JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

// POST a JSON body and yield each server-sent event as it arrives.
async function* events(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`the server answered ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      if (chunk.startsWith("data: ")) yield JSON.parse(chunk.slice(6));
    }
  }
}

function showBanner(message) {
  const banner = $("#banner");
  banner.textContent = message;
  banner.hidden = !message;
}

// Model answers often use **bold**; show it as bold instead of asterisks. Everything else is
// left as written, and the text is escaped first.
const formatted = (text) => escapeHtml(text).replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>");

const seconds = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
const logprob = (value) => value.toFixed(2).replace("-", "−");

// ------------------------------------------------------------------ views and theme

function showView(name) {
  if (!VIEWS.includes(name)) name = "router";
  for (const view of VIEWS) $(`#view-${view}`).hidden = view !== name;
  for (const tab of $$(".tabs button")) tab.setAttribute("aria-selected", String(tab.dataset.view === name));
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
  if (name === "advisor") loadAdvice();
}

function setUpChrome() {
  for (const tab of $$(".tabs button")) tab.addEventListener("click", () => showView(tab.dataset.view));
  window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
  document.addEventListener("keydown", (e) => {
    const typing = /^(TEXTAREA|INPUT)$/.test(document.activeElement.tagName);
    if (!typing && !e.metaKey && !e.ctrlKey && "123".includes(e.key)) showView(VIEWS[Number(e.key) - 1]);
  });
  $("#theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("edgewise-theme", next); } catch (e) { /* private window */ }
  });
}

// Segmented controls: one selected button per group.
function segmented(container, options, selected, onChange) {
  if (options) {
    container.innerHTML = options
      .map((o) => `<button role="radio" data-value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</button>`)
      .join("");
  }
  const select = (value) => {
    for (const button of $$("button", container)) {
      button.setAttribute("aria-checked", String(button.dataset.value === value));
    }
  };
  select(selected);
  container.addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    select(button.dataset.value);
    onChange(button.dataset.value);
  });
}

// ------------------------------------------------------------------ router

const SAMPLES = [
  { label: "Quick fact", prompt: "What is the boiling point of water at sea level, in Celsius? Answer in one sentence." },
  { label: "Open-ended explanation", prompt: "Explain in two sentences why the sky looks blue during the day." },
  {
    label: "Contains personal data",
    prompt: "Draft a two-line reply to priya.sharma@example.com confirming the review meeting on Friday at 3 pm.",
  },
  {
    label: "Structured output",
    json: true,
    prompt: 'Extract the fields from this text as JSON with keys "part", "quantity" and "due": "Please send 40 units of bracket BR-220 to the Pune plant by 14 March."',
  },
];

const routerState = { busy: false, history: [] };

function setStep(name, state, sub) {
  const step = $(`.step[data-step="${name}"]`);
  if (state) step.dataset.state = state;
  else delete step.dataset.state;
  delete step.dataset.went;
  const glyph = { pass: CHECK, answered: CHECK, stop: LOCK, doubt: UP, escalated: CHECK }[state];
  $(".dot", step).innerHTML = glyph || String(["rules", "local", "gate", "remote"].indexOf(name) + 1);
  if (sub !== undefined) $(".sub", step).textContent = sub;
}

// Colour the connectors the request travelled along: from each named step to the next one.
function went(...names) {
  for (const name of names) $(`.step[data-step="${name}"]`).dataset.went = "true";
}

function resetSteps() {
  const r = overview.router;
  setStep("rules", null, "privacy and offline checks");
  setStep("local", null, r.local.label);
  setStep("gate", null, `keep the answer if confidence ≥ ${logprob(r.min_logprob)}`);
  setStep("remote", null, r.remote ? `${r.remote.label}${r.remote.on_device ? "" : " (cloud)"}` : "not configured");
}

function setBadge(kind, text) {
  const badge = $("#route-badge");
  badge.dataset.kind = kind;
  $("span", badge).textContent = text;
}

function updateTiles(stats) {
  $("#tile-requests").textContent = stats.requests;
  $("#tile-local").textContent = stats.requests ? `${Math.round(stats.local_share * 100)}%` : "–";
  $("#tile-escalated").textContent = stats.escalated;
  $("#tile-private").textContent = stats.kept_private;
  $("#tile-cost").textContent = stats.cost_usd ? `$${stats.cost_usd.toFixed(4)}` : "$0";
}

async function sendRequest() {
  const prompt = $("#prompt").value.trim();
  if (!prompt || routerState.busy) return;
  routerState.busy = true;
  $("#send").disabled = true;

  resetSteps();
  setStep("rules", "active");
  setBadge("busy", "Checking the rules");
  const answer = $("#answer-text");
  answer.textContent = "";
  answer.classList.add("caret");
  $("#answer-model").textContent = "";
  for (const id of ["#draft", "#meter", "#answer-meta", "#answer-reason"]) $(id).hidden = true;

  const text = { local: "", remote: "" };
  let generating = null;
  const body = {
    messages: [{ role: "user", content: prompt }],
    max_tokens: 320,
    edgewise: { private: $("#opt-private").checked },
  };
  if ($("#opt-json").checked) body.response_format = { type: "json_object" };

  try {
    for await (const event of events("/api/route", body)) {
      if (event.type === "token") {
        if (event.route !== generating) {
          const before = generating;
          generating = event.route;
          if (generating === "local") {
            setStep("rules", "pass");
            went("rules");
            setStep("local", "active");
            setBadge("busy", "The small model is answering");
          } else {
            setStep("rules", "pass");
            // A local answer came first and was not kept, or the request went straight up.
            setStep("local", before ? "pass" : "skip");
            setStep("gate", before ? "doubt" : "skip", before ? "not confident enough" : undefined);
            setStep("remote", "active");
            went("rules", "local", "gate");
            setBadge("busy", "Escalated: the bigger model is answering");
            if (before) {
              $("#draft-text").textContent = text.local;
              $("#draft").hidden = false;
              $("#draft").open = false;
            }
            answer.textContent = "";
          }
        }
        text[event.route] += event.text;
        answer.innerHTML = formatted(text[event.route]);
      } else if (event.type === "decision") {
        finishRequest(event, prompt);
      } else if (event.type === "error") {
        throw new Error(event.message);
      }
    }
  } catch (error) {
    setBadge("error", "Something went wrong");
    answer.textContent = `The request failed: ${error.message}`;
    resetSteps();
  } finally {
    answer.classList.remove("caret");
    routerState.busy = false;
    $("#send").disabled = false;
  }
}

function finishRequest(d, prompt) {
  const threshold = overview.router.min_logprob;
  const keptByRule = /kept on device|no escalation model|offline/.test(d.reason);
  const failedUp = /escalation failed/.test(d.reason);
  const compare = d.confidence === null ? "" : `${logprob(d.confidence)} ${d.confidence < threshold ? "<" : "≥"} ${logprob(threshold)}`;

  if (d.route === "local") {
    setStep("local", "answered");
    if (keptByRule) {
      setStep("rules", "stop", d.reason.replace(/: kept on device$/, ""));
      setStep("gate", "skip");
      setStep("remote", "skip");
      went("rules");
    } else if (failedUp) {
      setStep("rules", "pass");
      setStep("gate", "doubt", compare || "not confident enough");
      setStep("remote", "skip", "could not be reached");
      went("rules", "local", "gate");
    } else {
      setStep("rules", "pass");
      const checked = /accepted/.test(d.reason);
      setStep("gate", checked ? "pass" : "skip", checked && compare ? compare : undefined);
      setStep("remote", "skip");
      went("rules", ...(checked ? ["local"] : []));
    }
    setBadge("local", "Answered on this device");
  } else {
    setStep("rules", "pass");
    if (d.escalated) {
      setStep("local", "pass");
      setStep("gate", "doubt", compare || d.reason.replace(/^escalated: /, ""));
    }
    setStep("remote", "escalated");
    went("rules", "local", "gate");
    setBadge("remote", d.escalated ? "Escalated to the bigger model" : "Answered by the bigger model");
  }

  $("#answer-text").innerHTML = d.text ? formatted(d.text) : '<span class="placeholder">(empty answer)</span>';
  $("#answer-model").textContent = d.model;
  const reason = d.reason.replace(/-(\d)/g, "−$1");
  $("#answer-reason").textContent = reason.charAt(0).toUpperCase() + reason.slice(1) + ".";
  $("#answer-reason").hidden = false;

  if (d.confidence !== null) {
    // Map log-probability −1.5 … 0 onto the track; anything lower sits at the left edge.
    const position = (value) => `${Math.max(0, Math.min(1, (value + 1.5) / 1.5)) * 100}%`;
    // A hard rule or an explicit choice decides before confidence is ever looked at.
    const used = d.escalated || /accepted|escalation failed/.test(d.reason);
    $("#meter-value").textContent = logprob(d.confidence);
    $("#meter-threshold").textContent = used ? `escalate below ${logprob(threshold)}` : "not used for this request";
    $("#meter-fill").style.width = position(d.confidence);
    $("#meter-fill").dataset.low = String(d.confidence < threshold);
    $("#meter-fill").dataset.unused = String(!used);
    $("#meter-mark").style.left = position(threshold);
    $("#meter").hidden = false;
  }

  $("#meta-time").textContent = seconds(d.latency_ms);
  $("#meta-tokens").textContent = `${d.prompt_tokens} in · ${d.completion_tokens} out`;
  $("#meta-cost").textContent = d.cost_usd ? `$${d.cost_usd.toFixed(4)}` : "$0";
  $("#answer-meta").hidden = false;
  updateTiles(d.stats);

  routerState.history.unshift({ prompt, route: d.route, reason: d.reason, ms: d.latency_ms });
  $("#history").innerHTML = routerState.history
    .slice(0, 8)
    .map(
      (h) =>
        `<li data-route="${h.route}"><i></i><span class="what">${escapeHtml(h.prompt)}</span>` +
        `<span class="why">${escapeHtml(h.reason)} · ${seconds(h.ms)}</span></li>`,
    )
    .join("");
  $("#history-card").hidden = false;
}

function setUpRouter() {
  $("#router-samples").innerHTML = SAMPLES.map((s, i) => `<button class="chip" data-i="${i}">${s.label}</button>`).join("");
  $("#router-samples").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    const sample = SAMPLES[Number(chip.dataset.i)];
    $("#prompt").value = sample.prompt;
    $("#opt-json").checked = Boolean(sample.json);
    $("#prompt").focus();
  });
  $("#send").addEventListener("click", sendRequest);
  $("#prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) sendRequest();
  });
  resetSteps();
  updateTiles(overview.stats);
}

// ------------------------------------------------------------------ advisor

const advisor = { task: null, memory: 4.5, floor: 0, prefer: "balanced", timer: null, request: 0 };

function setUpAdvisor() {
  advisor.task = overview.tasks[0].id;
  advisor.memory = Math.min(8, Math.max(1, Math.round(overview.device.budget_gb * 2) / 2));
  segmented(
    $("#adv-task"),
    overview.tasks.map((t) => ({ value: t.id, label: t.id.replace(/_/g, " ") })),
    advisor.task,
    (value) => { advisor.task = value; loadAdvice(); },
  );
  segmented($("#adv-prefer"), null, advisor.prefer, (value) => { advisor.prefer = value; loadAdvice(); });

  const memory = $("#adv-mem");
  memory.value = advisor.memory;
  const showMemory = () => { $("#adv-mem-out").textContent = `${Number(memory.value).toFixed(1)} GB`; };
  memory.addEventListener("input", () => { advisor.memory = Number(memory.value); showMemory(); loadAdviceSoon(); });
  showMemory();

  const floor = $("#adv-floor");
  floor.addEventListener("input", () => {
    advisor.floor = Number(floor.value);
    $("#adv-floor-out").textContent = advisor.floor ? advisor.floor.toFixed(2) : "none";
    loadAdviceSoon();
  });
}

function loadAdviceSoon() {
  clearTimeout(advisor.timer);
  advisor.timer = setTimeout(loadAdvice, 120);
}

async function loadAdvice() {
  if (!overview) return;
  const mine = ++advisor.request;
  const query = new URLSearchParams({
    task: advisor.task,
    memory_gb: advisor.memory,
    min_score: advisor.floor,
    prefer: advisor.prefer,
  });
  try {
    const rec = await getJSON(`/api/advise?${query}`);
    if (mine !== advisor.request) return; // a newer request is already on its way
    drawChart(rec);
    drawPick(rec);
    drawTable(rec);
  } catch (error) {
    $("#adv-chart").innerHTML = `<text class="empty" x="380" y="200" text-anchor="middle">${escapeHtml(error.message)}</text>`;
    $("#adv-pick").innerHTML = "";
    $("#adv-table tbody").innerHTML = "";
  }
}

function pointKind(row) {
  if (row.status === "too big" || row.status === "below quality floor") return "out";
  return row.on_frontier ? "frontier" : "beaten";
}

function drawChart(rec) {
  const W = 760, H = 400, left = 46, right = 24, top = 34, bottom = 46;
  const rows = rec.rows.filter((r) => r.evidence);
  const svg = $("#adv-chart");
  if (!rows.length) {
    svg.innerHTML =
      `<text class="empty" x="${W / 2}" y="${H / 2 - 10}" text-anchor="middle">No model has been benchmarked on this task yet.</text>` +
      `<text x="${W / 2}" y="${H / 2 + 16}" text-anchor="middle">Run: edgewise bench -m &lt;model&gt; -t ${escapeHtml(rec.task_id)}</text>`;
    return;
  }

  const fastest = Math.max(...rows.map((r) => r.evidence.decode_tps));
  const step = [5, 10, 20, 25, 50, 100, 200].find((s) => fastest / s <= 6) || 500;
  const xMax = Math.ceil((fastest * 1.06) / step) * step;
  // The score axis starts a little below the weakest model instead of at zero, so the
  // differences that matter are readable; its tick labels say where it starts.
  const lowest = Math.min(...rows.map((r) => r.evidence.score), rec.min_score || 1);
  const yMin = Math.max(0, Math.floor((lowest - 0.08) * 5) / 5);
  const yStep = 1 - yMin > 0.61 ? 0.2 : 0.1;
  const x = (tps) => left + (tps / xMax) * (W - left - right);
  const y = (score) => top + ((1 - score) / (1 - yMin)) * (H - top - bottom);

  const parts = [];
  for (let score = yMin; score < 1.001; score += yStep) {
    const base = score === yMin;
    parts.push(`<line class="${base ? "axis" : "gridline"}" x1="${left}" x2="${W - right}" y1="${y(score)}" y2="${y(score)}"/>`);
    parts.push(`<text x="${left - 10}" y="${y(score) + 4}" text-anchor="end">${score.toFixed(1)}</text>`);
  }
  for (let tps = 0; tps <= xMax; tps += step) {
    parts.push(`<text x="${x(tps)}" y="${H - bottom + 20}" text-anchor="middle">${tps}</text>`);
  }
  parts.push(`<text x="${(left + W - right) / 2}" y="${H - 6}" text-anchor="middle">Speed (tokens per second)</text>`);
  parts.push(`<text x="${left - 34}" y="${top - 18}">Task score</text>`);

  if (rec.min_score > 0) {
    parts.push(`<line class="floor" x1="${left}" x2="${W - right}" y1="${y(rec.min_score)}" y2="${y(rec.min_score)}"/>`);
    parts.push(`<text class="floor-label" x="${W - right}" y="${y(rec.min_score) - 6}" text-anchor="end">minimum score ${rec.min_score.toFixed(2)}</text>`);
  }

  const frontier = rows.filter((r) => pointKind(r) === "frontier").sort((a, b) => a.evidence.decode_tps - b.evidence.decode_tps);
  if (frontier.length > 1) {
    parts.push(`<polyline class="frontier" points="${frontier.map((r) => `${x(r.evidence.decode_tps)},${y(r.evidence.score)}`).join(" ")}"/>`);
  }

  // Marks: ruled-out and beaten first, the frontier on top.
  const order = { out: 0, beaten: 1, frontier: 2 };
  const sorted = [...rows].sort((a, b) => order[pointKind(a)] - order[pointKind(b)]);
  const chosen = rec.choice ? rec.choice.model_id : null;
  for (const row of sorted) {
    const cx = x(row.evidence.decode_tps), cy = y(row.evidence.score), kind = pointKind(row);
    if (row.model_id === chosen) parts.push(`<circle class="halo" cx="${cx}" cy="${cy}" r="12"/>`);
    parts.push(`<circle class="pt pt-${kind}" cx="${cx}" cy="${cy}" r="${kind === "frontier" ? 6.5 : 5}"/>`);
  }

  // Direct labels for the models worth considering; everything else is in the tooltip and table.
  // Each label tries four spots around its point and takes the one that collides least with
  // other labels, other points and the frontier line.
  const dots = rows.map((r) => ({ x: x(r.evidence.decode_tps), y: y(r.evidence.score) }));
  const lineSamples = [];
  for (let i = 1; i < frontier.length; i++) {
    const a = frontier[i - 1].evidence, b = frontier[i].evidence;
    for (let t = 0; t <= 1; t += 0.04) {
      lineSamples.push({ x: x(a.decode_tps + (b.decode_tps - a.decode_tps) * t), y: y(a.score + (b.score - a.score) * t) });
    }
  }
  const inside = (p, box, pad) => p.x > box.x0 - pad && p.x < box.x1 + pad && p.y > box.y0 - pad && p.y < box.y1 + pad;
  // The axis title sits above the plot on the left; labels may use the rest of that strip.
  const placed = [{ x0: 0, x1: left + 44, y0: 0, y1: top - 10 }];
  for (const row of frontier.slice().sort((a, b) => b.evidence.score - a.evidence.score)) {
    const cx = x(row.evidence.decode_tps), cy = y(row.evidence.score);
    const strong = row.model_id === chosen;
    const text = strong ? `${row.label} · recommended` : row.label;
    const width = text.length * (strong ? 7.4 : 6.9);
    const gap = strong ? 16 : 12;
    const spots = [
      { anchor: "start", tx: cx + gap, ty: cy - 8 }, // right, slightly above
      { anchor: "start", tx: cx + gap, ty: cy + 18 }, // right, below
      { anchor: "end", tx: cx - gap, ty: cy + 18 }, // left, below
      { anchor: "end", tx: cx - gap, ty: cy - 8 }, // left, above
      { anchor: "middle", tx: cx, ty: cy - 18 }, // straight above
    ];
    let best = null;
    for (const spot of spots) {
      const x0 = spot.anchor === "start" ? spot.tx : spot.anchor === "end" ? spot.tx - width : spot.tx - width / 2;
      const box = { x0, x1: x0 + width, y0: spot.ty - 12, y1: spot.ty + 4 };
      let cost = 0;
      if (box.x0 < left || box.x1 > W - right || box.y0 < 2 || box.y1 > H - bottom) cost += 100;
      cost += 50 * placed.filter((p) => box.x0 < p.x1 && p.x0 < box.x1 && box.y0 < p.y1 && p.y0 < box.y1).length;
      cost += 8 * dots.filter((d) => (d.x !== cx || d.y !== cy) && inside(d, box, 5)).length;
      cost += 3 * lineSamples.filter((p) => inside(p, box, 3)).length;
      if (!best || cost < best.cost) best = { ...spot, box, cost };
    }
    placed.push(best.box);
    parts.push(`<text class="${strong ? "label-strong" : "label"}" x="${best.tx}" y="${best.ty}" text-anchor="${best.anchor}">${escapeHtml(text)}</text>`);
  }

  for (const row of rows) {
    parts.push(`<circle class="hit" data-model="${escapeHtml(row.model_id)}" cx="${x(row.evidence.decode_tps)}" cy="${y(row.evidence.score)}" r="15"/>`);
  }
  svg.innerHTML = parts.join("");

  const tip = $("#adv-tip");
  for (const hit of $$(".hit", svg)) {
    const row = rows.find((r) => r.model_id === hit.dataset.model);
    hit.addEventListener("pointerenter", () => {
      const e = row.evidence;
      tip.innerHTML =
        `<b>${escapeHtml(row.label)}</b>` +
        `<div><span>Score</span><span>${e.score.toFixed(2)}</span></div>` +
        `<div><span>Speed</span><span>${e.decode_tps.toFixed(0)} tokens/s</span></div>` +
        `<div><span>First token</span><span>${seconds(e.ttft_ms)}</span></div>` +
        `<div><span>Peak memory</span><span>${row.memory_gb.toFixed(1)} GB</span></div>` +
        `<em>${escapeHtml(row.status)} · ${e.n_items} test items</em>`;
      const box = hit.getBoundingClientRect(), wrap = svg.parentElement.getBoundingClientRect();
      tip.style.left = `${box.left + box.width / 2 - wrap.left}px`;
      tip.style.top = `${box.top + box.height / 2 - wrap.top}px`;
      tip.hidden = false;
    });
    hit.addEventListener("pointerleave", () => { tip.hidden = true; });
  }
}

function drawPick(rec) {
  const pick = $("#adv-pick");
  const task = rec.task_id.replace(/_/g, " ");
  if (!rec.choice) {
    pick.innerHTML =
      `<span class="eyebrow">No recommendation for ${escapeHtml(task)}</span>` +
      `<ul>${rec.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`;
    return;
  }
  const e = rec.choice.evidence;
  const flags = [`--task ${rec.task_id}`, `--memory ${rec.budget_gb}GB`];
  if (rec.min_score > 0) flags.push(`--min-score ${rec.min_score}`);
  if (rec.prefer !== "balanced") flags.push(`--prefer ${rec.prefer}`);
  pick.innerHTML =
    `<span class="eyebrow">Recommended for ${escapeHtml(task)}</span>` +
    `<h2>${escapeHtml(rec.choice.label)}</h2>` +
    `<div class="stats">` +
    `<div><span>Score</span><b>${e.score.toFixed(2)}</b></div>` +
    `<div><span>Tokens/s</span><b>${e.decode_tps.toFixed(0)}</b></div>` +
    `<div><span>Memory</span><b>${rec.choice.memory_gb.toFixed(1)} GB</b></div>` +
    `</div>` +
    `<ul>${rec.reasons.slice(1).map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>` +
    `<pre>edgewise advise \\\n  ${escapeHtml(flags.join(" \\\n  "))}</pre>`;
}

function drawTable(rec) {
  $("#adv-table tbody").innerHTML = rec.rows
    .map((row) => {
      const e = row.evidence;
      const dim = ["too big", "below quality floor", "not benchmarked", "dominated"].includes(row.status);
      const memory = `${row.memory_gb.toFixed(1)} GB${row.memory_measured ? "" : " est."}`;
      return (
        `<tr data-dim="${dim}"><td>${escapeHtml(row.label)}</td>` +
        `<td class="num">${e ? e.score.toFixed(2) : "–"}</td>` +
        `<td class="num">${e ? e.decode_tps.toFixed(0) : "–"}</td>` +
        `<td class="num">${e ? seconds(e.ttft_ms) : "–"}</td>` +
        `<td class="num">${memory}</td>` +
        `<td><span class="status" data-status="${escapeHtml(row.status)}"><i></i>${escapeHtml(row.status)}</span></td></tr>`
      );
    })
    .join("");
}

// ------------------------------------------------------------------ accelerator

const PRESETS = [
  {
    value: "edit",
    label: "Edit a paragraph",
    system: "Apply the requested edit to the text. Reply with ONLY the full edited text, changing nothing else.",
    user:
      "Text:\nThe quarterly service for vehicle KA-05-MN-4821 is booked for Tuesday at the Whitefield workshop. " +
      "The technician will replace the cabin air filter, rotate all four tyres and check the brake pads. " +
      "Please bring the service booklet and both sets of keys. The work should take about three hours, " +
      "and a courtesy shuttle to the metro station leaves every thirty minutes.\n\n" +
      "Edit: Change the day from Tuesday to Thursday.",
    max: 220,
  },
  {
    value: "extract",
    label: "Extract fields",
    system: "Extract the requested fields from the text. Reply with ONLY a JSON object.",
    user:
      "Text:\nInvoice INV-2024-0917 from Meridian Components Pvt Ltd, dated 12 September. " +
      "40 units of mounting bracket BR-220 at 185 rupees each, delivered to the Pune assembly plant. " +
      "Payment is due within 45 days of the invoice date.\n\n" +
      "Fields: invoice_number, vendor, part, quantity, unit_price, delivery_location, payment_terms",
    max: 180,
  },
  {
    value: "reason",
    label: "Free-form reasoning",
    system: "Solve the problem. Think briefly, then end with a final line 'Answer: <number>'.",
    user:
      "A warehouse has 18 shelves. Each shelf holds 24 boxes, and each box holds 12 sensors. " +
      "A truck takes away 5 full shelves. How many sensors are left in the warehouse?",
    max: 220,
  },
];

const accelerator = { preset: PRESETS[0], busy: false, runs: {}, replay: 0 };
const laneEl = (lane) => $(`.lane[data-lane="${lane}"]`);

function setUpAccelerator() {
  segmented($("#acc-presets"), PRESETS, accelerator.preset.value, (value) => {
    accelerator.preset = PRESETS.find((p) => p.value === value);
    showPreset();
  });
  showPreset();
  $("#acc-model").textContent = `Model: ${overview.router.local.label}`;
  $("#acc-run").addEventListener("click", runComparison);
  $("#acc-replay").addEventListener("click", replay);
}

function showPreset() {
  $("#acc-prompt").value = accelerator.preset.user;
  $("#acc-system").textContent = `Instruction: ${accelerator.preset.system}`;
}

function appendPass(output, tokens) {
  // All but the last token of a pass were guesses the model accepted.
  tokens.forEach((token, i) => {
    if (i < tokens.length - 1) {
      const span = document.createElement("span");
      span.className = "guess";
      span.textContent = token;
      output.append(span);
    } else {
      output.append(token);
    }
  });
  output.scrollTop = output.scrollHeight;
}

function clearLane(lane, status) {
  const el = laneEl(lane);
  $(".output", el).textContent = "";
  $(".speed", el).textContent = "–";
  $(".lane-status", el).textContent = status;
  delete el.dataset.done;
  delete el.dataset.busy;
}

function laneSummary(run) {
  const done = run.done;
  const guessed = done.accepted ? ` · ${Math.round((done.accepted / done.tokens) * 100)}% of tokens were accepted guesses` : "";
  return `${done.tokens} tokens in ${seconds(done.total_ms)} · first token after ${seconds(done.ttft_ms)}${guessed}`;
}

async function runComparison() {
  if (accelerator.busy) return;
  accelerator.busy = true;
  accelerator.replay++; // stops a replay that is still playing
  $("#acc-run").disabled = true;
  $("#acc-result").hidden = true;
  accelerator.runs = {};
  clearLane("standard", "Waiting to start");
  clearLane("prompt_lookup", "Runs after standard decoding finishes");

  const body = {
    messages: [
      { role: "system", content: accelerator.preset.system },
      { role: "user", content: $("#acc-prompt").value },
    ],
    max_tokens: accelerator.preset.max,
  };
  try {
    for await (const event of events("/api/compare", body)) {
      if (event.type === "error") throw new Error(event.message);
      if (event.type === "result") { showResult(event); continue; }
      const el = laneEl(event.lane);
      if (event.type === "start") {
        accelerator.runs[event.lane] = { passes: [], count: 0 };
        el.dataset.busy = "true";
        $(".lane-status", el).textContent = `Reading the prompt (${event.prompt_tokens} tokens)`;
      } else if (event.type === "tokens") {
        const run = accelerator.runs[event.lane];
        run.passes.push(event);
        run.count += event.tokens.length;
        appendPass($(".output", el), event.tokens);
        $(".lane-status", el).textContent = `Generating · ${run.count} tokens · ${seconds(event.t_ms)}`;
      } else if (event.type === "done") {
        accelerator.runs[event.lane].done = event;
        delete el.dataset.busy;
        el.dataset.done = "true";
        $(".speed", el).textContent = event.decode_tps.toFixed(1);
        $(".lane-status", el).textContent = laneSummary(accelerator.runs[event.lane]);
      }
    }
  } catch (error) {
    for (const lane of ["standard", "prompt_lookup"]) delete laneEl(lane).dataset.busy;
    $(".lane-status", laneEl("standard")).textContent = `The comparison failed: ${error.message}`;
  } finally {
    accelerator.busy = false;
    $("#acc-run").disabled = false;
  }
}

function showResult(result) {
  const fast = accelerator.runs.prompt_lookup.done, slow = accelerator.runs.standard.done;
  const gain = result.speedup !== null && result.speedup >= 1.05;
  $("#acc-result").dataset.gain = String(gain);
  $("#acc-speedup").textContent = result.speedup === null ? "–" : `${result.speedup.toFixed(2)}×`;
  $("#acc-speedup-label").textContent = "decode speed vs. standard";

  const facts = [];
  facts.push(
    result.identical
      ? { kind: "good", text: "Output identical to standard decoding" }
      : { kind: "warn", text: "Output differs: a near-tie between two tokens broke the other way" },
  );
  if (fast.proposed) {
    facts.push({ text: `${fast.accepted} of ${fast.proposed} guesses accepted` });
  }
  facts.push({ text: `Whole request: ${seconds(fast.total_ms)} vs. ${seconds(slow.total_ms)}` });
  if (!gain) {
    facts.push({ text: "Little to guess here: the answer does not repeat the prompt" });
  }
  $("#acc-facts").innerHTML = facts
    .map((f) => `<span class="fact" data-kind="${f.kind || ""}">${escapeHtml(f.text)}</span>`)
    .join("");
  $("#acc-result").hidden = false;
}

// Play both recorded runs back together, each at the pace it was measured at.
function replay() {
  const runs = accelerator.runs;
  if (!runs.standard || !runs.prompt_lookup || accelerator.busy) return;
  const mine = ++accelerator.replay;
  const lanes = ["standard", "prompt_lookup"].map((lane) => {
    const el = laneEl(lane);
    $(".output", el).textContent = "";
    el.dataset.busy = "true";
    return { lane, el, run: runs[lane], next: 0, count: 0 };
  });
  const started = performance.now();

  const frame = (now) => {
    if (mine !== accelerator.replay) return;
    const elapsed = now - started;
    let playing = false;
    for (const l of lanes) {
      while (l.next < l.run.passes.length && l.run.passes[l.next].t_ms <= elapsed) {
        const pass = l.run.passes[l.next++];
        l.count += pass.tokens.length;
        appendPass($(".output", l.el), pass.tokens);
      }
      if (l.next < l.run.passes.length) {
        playing = true;
        $(".lane-status", l.el).textContent = `Replaying · ${l.count} tokens · ${seconds(elapsed)}`;
      } else if (l.el.dataset.busy) {
        delete l.el.dataset.busy;
        $(".lane-status", l.el).textContent = laneSummary(l.run);
      }
    }
    if (playing) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ start

async function start() {
  setUpChrome();
  try {
    overview = await getJSON("/api/overview");
  } catch (error) {
    showBanner(`Could not reach the Edgewise server: ${error.message}`);
    return;
  }
  const d = overview.device;
  $("#version").textContent = `v${overview.version}`;
  $("#device").innerHTML = `<b>${escapeHtml(d.chip)}</b> · ${d.ram_gb} GB · ${d.budget_gb} GB for models`;
  if (!overview.benchmarked) {
    showBanner("No benchmark results yet, so the Advisor has nothing to show. Run: edgewise bench -m <model> --all-tasks");
  }
  setUpRouter();
  setUpAdvisor();
  setUpAccelerator();
  showView(location.hash.slice(1) || "router");
}

start();
