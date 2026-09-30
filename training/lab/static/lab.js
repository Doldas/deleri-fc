/* Match Lab front end.
 *
 * Draws a replay produced by the authoritative engine (see lab_server.py). The
 * pitch is 60 x 40 m per RULES.md, the goal is 6 m wide and centred on each
 * goal line, and there are five players a side: one goalkeeper, four outfield.
 * Team-a defends x=0 and attacks +x; the replay is already in absolute
 * coordinates, so nothing is mirrored here.
 */

const PITCH_L = 60;      // metres, goal line to goal line
const PITCH_W = 40;      // metres, touchline to touchline
const GOAL_W = 6;        // metres
const GOAL_D = 2.0;      // drawn depth, cosmetic
const PEN_D = 9.0;       // penalty area depth, cosmetic
const PEN_HALF = 13.0;   // penalty area half-width, cosmetic
const SIX_D = 3.5;       // six-yard box depth, cosmetic
const SIX_HALF = 8.0;    // six-yard box half-width, cosmetic
const CIRCLE_R = 6.0;    // centre circle radius, cosmetic

const el = (id) => document.getElementById(id);
const canvas = el("pitch");
const ctx = canvas.getContext("2d");

// ---------------------------------------------------------------- kit colours
//
// The two sides must never render the same colour. The bundled CPUs ship no
// identity at all (the server reports colors: null for them, because there is
// nothing to read out of a compiled team), so a naive `primary || fallback`
// handed the away side My Team FC's own blue and both teams came out identical.
// resolveKits() keeps each team's real primary when it is already distinct and
// otherwise falls back, in order, to that team's secondary and then to a spare
// palette. Perceptual distance is used rather than string equality so a navy
// opponent against a blue home kit is treated as the clash it looks like.

const KIT_ALTERNATES = ["#dc2626", "#f59e0b", "#16a34a", "#7c3aed",
                        "#0891b2", "#db2777", "#16a34a", "#0f766e"];
// Calibrated against the real pool: of the 51 opponents, one ships a primary
// byte-identical to ours and six more sit within 150, so anything at or below
// ~190 is treated as a clash and gets swapped. 35 opponents keep their real kit.
const KIT_CONTRAST_MIN = 190;
const HOME_FALLBACK = "#2563eb";

function hexToRgb(hex) {
  if (typeof hex !== "string") return null;
  const match = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!match) return null;
  const n = parseInt(match[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// Approximate perceptual distance (redmean). 0 = identical, ~765 = extreme.
function colorDistance(a, b) {
  const x = hexToRgb(a), y = hexToRgb(b);
  if (!x || !y) return Infinity;   // unknown colour: never claim a clash
  const rbar = (x[0] + y[0]) / 2;
  const dr = x[0] - y[0], dg = x[1] - y[1], db = x[2] - y[2];
  return Math.sqrt((2 + rbar / 256) * dr * dr + 4 * dg * dg
                   + (2 + (255 - rbar) / 256) * db * db);
}

function resolveKits(meta) {
  const home = (meta && meta.home) || {};
  const away = (meta && meta.away) || {};
  // The engine's replay meta reports the kit flat; /api/opponents nests it
  // under `colors`. Accept either so a kit is never silently dropped.
  const awayColors = away.colors || {};
  const homeColors = home.colors || {};
  const a = home.primary || homeColors.primary || HOME_FALLBACK;
  const awayPrimary = away.primary || awayColors.primary || null;
  const awaySecondary = away.secondary || awayColors.secondary || null;

  // Preference order: the opponent's real primary, then its secondary (what
  // the team actually wears as trim), then spare colours for teams that ship
  // no identity at all.
  const candidates = [awayPrimary, awaySecondary, ...KIT_ALTERNATES]
    .filter((c, i, arr) => c && arr.indexOf(c) === i);

  let b = candidates[0] || KIT_ALTERNATES[0];
  for (const c of candidates) {
    if (colorDistance(a, c) >= KIT_CONTRAST_MIN) { b = c; break; }
  }
  return { a, b, aAlt: home.secondary || homeColors.secondary || a,
           bAlt: awaySecondary || b };
}

function kits() {
  return state.kits || resolveKits(state.match && state.match.meta);
}

const state = {
  opponents: [],
  match: null,      // trimmed replay
  frame: 0,
  playing: false,
  speed: 1,
  lastTick: 0,
  carry: 0,          // fractional frame carried between ticks
  kits: null,        // resolved home/away colours, guaranteed distinct
  poll: null,
};

// Role abbreviations (2 letters) for display on players
// Handles engine replay IDs like "team-a:gk", "team-a:d1", "team-b:f1" etc.
const ROLE_ABBR = {
  // Our team (team-a) replay IDs
  "team-a:gk": "GK",
  "team-a:d1": "DF",
  "team-a:d2": "DF",
  "team-a:f1": "ST",
  "team-a:f2": "ST",
  // Opponent team (team-b) replay IDs
  "team-b:gk": "GK",
  "team-b:d1": "DF",
  "team-b:d2": "DF",
  "team-b:f1": "ST",
  "team-b:f2": "ST",
  // Short IDs from observation (our team)
  gk: "GK",
  def: "DF",
  cb: "DF",
  df: "DF",
  db: "DF",
  dm: "DF",
  left: "WL",
  right: "WR",
  lw: "WL",
  rw: "WR",
  wb: "WB",
  lwb: "WL",
  rwb: "WR",
  st: "ST",
  cf: "ST",
  fw: "ST",
  am: "AM",
  cm: "CM",
  lm: "LM",
  rm: "RM",
  wm: "WM",
  mm: "CM",
  // Short IDs from observation (opponent team with t_ prefix)
  tgk: "GK",
  tdef: "DF",
  tcb: "DF",
  tdf: "DF",
  tdb: "DF",
  tdm: "DF",
  tleft: "WL",
  tright: "WR",
  tlw: "WL",
  trw: "WR",
  twb: "WB",
  tlwb: "WL",
  trwb: "WR",
  tst: "ST",
  tcf: "ST",
  tfw: "ST",
  tam: "AM",
  tcm: "CM",
  tlm: "LM",
  trm: "RM",
  twm: "WM",
  tmm: "CM",
};

function roleAbbr(pid) {
  if (!pid) return "??";
  const key = pid.toLowerCase();
  // Try exact match first (handles engine replay IDs like "team-a:gk")
  if (ROLE_ABBR[key]) return ROLE_ABBR[key];
  // Try stripping team prefix (handles "team-a:gk" -> "gk")
  const parts = key.split(":");
  if (parts.length === 2 && ROLE_ABBR[parts[1]]) return ROLE_ABBR[parts[1]];
  // Try stripping t_ prefix (handles "tgk" -> "gk")
  if (key.startsWith("t") && ROLE_ABBR[key.slice(1)]) return ROLE_ABBR[key.slice(1)];
  // Fallback: first 2 chars
  return key.slice(0, 2).toUpperCase();
}

// Legend items for the role key
const ROLE_LEGEND = [
  { abbr: "GK", name: "Goalkeeper" },
  { abbr: "DF", name: "Defender" },
  { abbr: "WL", name: "Winger Left" },
  { abbr: "WR", name: "Winger Right" },
  { abbr: "ST", name: "Striker" },
  { abbr: "AM", name: "Attacking Midfielder" },
  { abbr: "CM", name: "Central Midfielder" },
  { abbr: "LM", name: "Left Midfielder" },
  { abbr: "RM", name: "Right Midfielder" },
  { abbr: "WB", name: "Wing-Back" },
];

let view = { scale: 1, ox: 0, oy: 0 };

function layout() {
  // Fit the pitch into the canvas with a small surround so goals and shadows
  // are not clipped, and keep a 1:1.5 aspect (60:40 plus goal depth).
  const padX = 26, padY = 22;
  const w = canvas.width, h = canvas.height;
  const usableW = w - padX * 2;
  const usableH = h - padY * 2;
  const scale = Math.min(usableW / (PITCH_L + GOAL_D * 2), usableH / PITCH_W);
  view.scale = scale;
  view.ox = padX + (usableW - PITCH_L * scale) / 2 + GOAL_D * scale;
  view.oy = padY + (usableH - PITCH_W * scale) / 2;
}

const sx = (x) => view.ox + x * view.scale;
const sy = (y) => view.oy + y * view.scale;

// ---------------------------------------------------------------- painting

function drawPitch() {
  const w = PITCH_L * view.scale, h = PITCH_W * view.scale;

  ctx.fillStyle = "#0a0d12";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Mown stripes: eight bands along the length, alternating tone.
  const css = getComputedStyle(document.documentElement);
  const bands = 8;
  for (let i = 0; i < bands; i++) {
    ctx.fillStyle = css.getPropertyValue(i % 2 ? "--grass-b" : "--grass-a").trim()
      || "#2f6b3a";
    ctx.fillRect(sx(i * PITCH_L / bands), sy(0), (PITCH_L / bands) * view.scale, h);
  }

  // Goals, drawn behind the lines.
  ctx.fillStyle = "rgba(255,255,255,0.14)";
  const gy0 = sy((PITCH_W - GOAL_W) / 2), gy1 = sy((PITCH_W + GOAL_W) / 2);
  ctx.fillRect(sx(0) - GOAL_D * view.scale, gy0, GOAL_D * view.scale, gy1 - gy0);
  ctx.fillRect(sx(PITCH_L), gy0, GOAL_D * view.scale, gy1 - gy0);

  ctx.strokeStyle = css.getPropertyValue("--line-white").trim()
    || "rgba(255,255,255,0.82)";
  ctx.lineWidth = Math.max(1.5, 0.12 * view.scale);
  ctx.strokeRect(sx(0), sy(0), w, h);

  // Halfway line and centre circle.
  ctx.beginPath();
  ctx.moveTo(sx(PITCH_L / 2), sy(0));
  ctx.lineTo(sx(PITCH_L / 2), sy(PITCH_W));
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(sx(PITCH_L / 2), sy(PITCH_W / 2), CIRCLE_R * view.scale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(sx(PITCH_L / 2), sy(PITCH_W / 2), 0.4 * view.scale, 0, Math.PI * 2);
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fill();

  // Boxes, mirrored.
  for (const side of [0, 1]) {
    const x = side ? sx(PITCH_L) : sx(0);
    const dir = side ? -1 : 1;
    ctx.strokeRect(
      Math.min(x, x + dir * PEN_D * view.scale), sy(PITCH_W / 2 - PEN_HALF),
      PEN_D * view.scale, 2 * PEN_HALF * view.scale);
    ctx.strokeRect(
      Math.min(x, x + dir * SIX_D * view.scale), sy(PITCH_W / 2 - SIX_HALF),
      SIX_D * view.scale, 2 * SIX_HALF * view.scale);
  }

  // Corner arcs.
  for (const [cx, cy, a0] of [[0, 0, 0], [PITCH_L, 0, Math.PI / 2],
                              [PITCH_L, PITCH_W, Math.PI], [0, PITCH_W, 1.5 * Math.PI]]) {
    ctx.beginPath();
    ctx.arc(sx(cx), sy(cy), 1 * view.scale, a0, a0 + Math.PI / 2);
    ctx.stroke();
  }
}

function drawFrame() {
  drawPitch();
  const m = state.match;
  if (!m) return;
  const f = m.frames;
  const i = Math.max(0, Math.min(state.frame, f.phase.length - 1));
  const per = 5;  // players per side

  // Ball shadow, then ball.
  const bx = f.ball[i * 2], by = f.ball[i * 2 + 1];
  const held = f.owner[i] >= 0;
  ctx.beginPath();
  ctx.ellipse(sx(bx), sy(by) + 0.35 * view.scale, 0.42 * view.scale,
              0.18 * view.scale, 0, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.fill();

  // Players: away first so the home side draws on top at restarts.
  const kit = kits();
  for (const side of ["b", "a"]) {
    const flat = f[side], facing = f["facing" + side.toUpperCase()];
    const primary = side === "a" ? kit.a : kit.b;
    const ids = m.meta.order[side];
    for (let p = 0; p < per; p++) {
      const o = i * per * 3 + p * 3;
      const x = flat[o], y = flat[o + 1], canAct = flat[o + 2] > 0.5;
      const gk = /gk/.test(ids[p] || "");
      const isBallHolder = held && f.owner[i] === (side === "a" ? 0 : 1) &&
        dist(x, y, bx, by) < 1.6;
      drawPlayer(x, y, facing[i * per + p], primary, gk, canAct, isBallHolder, ids[p]);
    }
  }

  ctx.beginPath();
  ctx.arc(sx(bx), sy(by), (held ? 0.32 : 0.26) * view.scale, 0, Math.PI * 2);
  ctx.fillStyle = held ? "#f0b429" : "#ffffff";
  ctx.fill();
  ctx.lineWidth = Math.max(1, 0.06 * view.scale);
  ctx.strokeStyle = "rgba(0,0,0,0.5)";
  ctx.stroke();
}

const dist = (x1, y1, x2, y2) => Math.hypot(x1 - x2, y1 - y2);

function drawPlayer(x, y, facing, primary, gk, canAct, isHolder, pid) {
  const px = sx(x), py = sy(y);
  const r = (gk ? 0.95 : 0.8) * view.scale;

  ctx.beginPath();
  ctx.ellipse(px, py + 0.3 * view.scale, r, r * 0.4, 0, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.32)";
  ctx.fill();

  // Facing tick: which way this player is looking.
  if (Number.isFinite(facing)) {
    const len = r * 1.9;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + Math.cos(facing) * len, py + Math.sin(facing) * len);
    ctx.lineWidth = Math.max(1, 0.13 * view.scale);
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fillStyle = primary;
  ctx.fill();
  ctx.lineWidth = Math.max(1, 0.1 * view.scale);
  ctx.strokeStyle = gk ? "#58a6ff" : "rgba(255,255,255,0.75)";
  ctx.stroke();

  if (!canAct) {
    // canAct = false: mid-knockdown, so show it rather than let the player
    // look like they simply chose to stand still.
    ctx.beginPath();
    ctx.arc(px, py, r + 0.22 * view.scale, 0, Math.PI * 2);
    ctx.setLineDash([0.3 * view.scale, 0.3 * view.scale]);
    ctx.lineWidth = Math.max(1, 0.12 * view.scale);
    ctx.strokeStyle = "#f85149";
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (isHolder) {
    ctx.beginPath();
    ctx.arc(px, py, r + 0.42 * view.scale, 0, Math.PI * 2);
    ctx.lineWidth = Math.max(1, 0.1 * view.scale);
    ctx.strokeStyle = "rgba(240,180,41,0.9)";
    ctx.stroke();
  }

  // Role abbreviation label (2 letters) inside the player circle with high contrast
  if (pid) {
    const label = roleAbbr(pid);
    // Choose text color based on kit primary color luminance for best contrast
    const kitRgb = hexToRgb(primary);
    const luminance = kitRgb ? (0.299 * kitRgb[0] + 0.587 * kitRgb[1] + 0.114 * kitRgb[2]) : 128;
    const textColor = luminance > 140 ? "#000" : "#fff";
    const strokeColor = luminance > 140 ? "#fff" : "#000";

    const fontSize = Math.max(8, 0.35 * view.scale);
    ctx.font = `bold ${fontSize}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = Math.max(2, 0.15 * view.scale);
    ctx.strokeStyle = strokeColor;
    ctx.strokeText(label, px, py);
    ctx.fillStyle = textColor;
    ctx.fillText(label, px, py);
  }
}

// ---------------------------------------------------------------- chrome

function paintChrome() {
  const m = state.match;
  if (!m) return;
  const f = m.frames;
  const i = Math.max(0, Math.min(state.frame, f.phase.length - 1));

  el("score-a").textContent = f.scoreA[i];
  el("score-b").textContent = f.scoreB[i];
  el("clock").textContent = `${Math.max(0, f.time[i]).toFixed(0)}s ${f.phase[i]}`;
  el("tcode").textContent = `${i + 1} / ${f.phase.length}`;
  const kit = kits();
  el("home-chip").style.background = kit.a;
  el("away-chip").style.background = kit.b;
  el("home-name").textContent = m.meta.home.name || "home";
  el("away-name").textContent = m.meta.away.name || "away";
}

function paintGoalMarkers() {
  const scrub = el("scrub");
  // Rebuild the datalist-free marker overlay as absolutely positioned ticks.
  let host = scrub.parentElement.querySelector(".goalmarks");
  if (host) host.remove();
  const m = state.match;
  if (!m || !m.meta.goals.length) return;
  host = document.createElement("div");
  host.className = "goalmarks";
  const kit = kits();
  const total = Math.max(1, m.meta.frames - 1);
  for (const g of m.meta.goals) {
    const tick = document.createElement("i");
    tick.style.left = `${(g.frame / total) * 100}%`;
    tick.style.background = g.side === "a" ? kit.a : kit.b;
    host.appendChild(tick);
  }
  scrub.parentElement.appendChild(host);
}

// ---------------------------------------------------------------- playback

function setFrame(i) {
  const m = state.match;
  if (!m) return;
  const max = m.frames.phase.length - 1;
  state.frame = Math.max(0, Math.min(i, max));
  el("scrub").value = state.frame;
  drawFrame();
  paintChrome();
}

function play() {
  if (!state.match) return;
  if (state.frame >= state.match.frames.phase.length - 1) state.frame = 0;
  state.playing = true;
  state.carry = 0;
  state.lastTick = performance.now();
  el("playpause").innerHTML = "&#10074;&#10074;";
  requestAnimationFrame(tick);
}

function pause() {
  state.playing = false;
  el("playpause").innerHTML = "&#9654;";
}

function tick(now) {
  if (!state.playing) return;
  const dt = (now - state.lastTick) / 1000;
  state.lastTick = now;
  const m = state.match;
  // The replay is 20 Hz / step; play it back at engine speed times the factor.
  const framesPerSecond = 20 / m.meta.step;
  // Carry the fractional frame across ticks, otherwise at low speeds or on
  // high-refresh displays the sub-frame remainder is dropped every tick and
  // playback stalls on the current frame.
  state.carry += dt * framesPerSecond * state.speed;
  const whole = Math.floor(state.carry);
  if (whole > 0) {
    state.carry -= whole;
    const next = state.frame + whole;
    if (next >= m.frames.phase.length - 1) {
      setFrame(m.frames.phase.length - 1);
      pause();
      return;
    }
    setFrame(next);
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- data

async function loadOpponents() {
  let data;
  try {
    const res = await fetch("/api/opponents");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    // Never fail silently: an empty dropdown with no error is indistinguishable
    // from "no opponents exist", which is exactly the bug this guards against.
    const select = el("opponent");
    select.innerHTML = "";
    const opt = document.createElement("option");
    opt.textContent = `failed to load opponents: ${err.message}`;
    opt.disabled = true;
    select.appendChild(opt);
    console.error("loadOpponents failed", err);
    return;
  }
  state.opponents = data.opponents || [];
  const select = el("opponent");
  select.innerHTML = "";
  let group = null;
  let og = null;
  for (const o of state.opponents) {
    if (o.group !== group) {
      group = o.group;
      og = document.createElement("optgroup");
      og.label = group === "Bundled CPUs" ? group : group.replace(/_/g, " ");
      select.appendChild(og);
    }
    const opt = document.createElement("option");
    opt.value = o.id;
    const tag = o.shortName ? ` [${o.shortName}]` : "";
    opt.textContent = `${o.name}${tag}`;
    og.appendChild(opt);
  }
  if (!select.options.length) {
    const none = document.createElement("option");
    none.textContent = "no opponents found";
    none.disabled = true;
    select.appendChild(none);
  }
  updateHint();
}

function updateHint() {
  const o = state.opponents.find((x) => x.id === el("opponent").value);
  const hint = el("opponent-hint");
  if (!o) { hint.innerHTML = "&nbsp;"; return; }
  const bits = [];
  if (o.generated) {
    bits.push(`<span class="tag">generated</span>${o.family}${o.variant ? " / " + o.variant : ""}`);
  } else {
    bits.push(`<span class="tag">bundled</span>compiled CPU`);
  }
  if (o.colors && o.colors.primary) {
    bits.push(`<span class="tag" style="background:${o.colors.primary};color:#fff;border-color:transparent">kit</span>${o.colors.primary}`);
  }
  hint.innerHTML = bits.join(" ");
}

async function startMatch() {
  const opponent = el("opponent").value;
  const duration = el("duration").value;
  pause();
  el("kickoff").disabled = true;
  el("kickoff").textContent = "Running…";
  showOverlay("Building and running the match…", "The engine is deterministic, so this is the same match every time for this opponent and duration.", "busy");

  try {
    const res = await fetch("/api/match", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ opponent, duration }),
    });
    const started = await res.json();
    if (!res.ok) throw new Error(started.error || `HTTP ${res.status}`);
    const result = await pollJob(started.job);
    state.match = result;
    state.kits = resolveKits(result.meta);
    el("scrub").max = result.frames.phase.length - 1;
    paintReadout(result.meta);
    paintGoalMarkers();
    setFrame(0);
    hideOverlay();
    play();
  } catch (err) {
    showOverlay("Could not run the match", String(err.message || err), "error");
  } finally {
    el("kickoff").disabled = false;
    el("kickoff").textContent = "Play match";
  }
}

async function pollJob(jobId) {
  // A Docker build plus an engine run is tens of seconds, so poll rather than
  // hold one long request open.
  for (;;) {
    const res = await fetch(`/api/job/${jobId}`);
    const job = await res.json();
    if (job.state === "error") throw new Error(job.error || "engine run failed");
    if (job.state === "done") return job.result;
    await new Promise((r) => setTimeout(r, 1200));
  }
}

function paintReadout(meta) {
  const s = meta.summary || {};
  el("r-poss").textContent = s.possession != null ? `${s.possession}%` : "—";
  el("r-shots").textContent = s.shots != null ? s.shots : "—";
  el("r-missed").textContent = s.missed != null ? s.missed : "—";
  el("r-seed").textContent = meta.seed || "—";
  el("r-wall").textContent = meta.wallTime != null ? `${meta.wallTime}s` : "—";
}

function showOverlay(title, body, cls) {
  const o = el("overlay");
  o.hidden = false;
  el("overlay-title").textContent = title;
  const b = el("overlay-body");
  b.textContent = body;
  b.className = "overlay-body" + (cls ? " " + cls : "");
}

function hideOverlay() { el("overlay").hidden = true; }

// ---------------------------------------------------------------- wiring

function init() {
  layout();
  window.addEventListener("resize", () => { layout(); drawFrame(); });

  el("kickoff").addEventListener("click", startMatch);
  el("opponent").addEventListener("change", updateHint);
  el("playpause").addEventListener("click", () =>
    state.playing ? pause() : play());
  el("stepfwd").addEventListener("click", () => { pause(); setFrame(state.frame + 1); });
  el("stepback").addEventListener("click", () => { pause(); setFrame(state.frame - 1); });
  el("scrub").addEventListener("input", (e) => { pause(); setFrame(+e.target.value); });
  el("speed").addEventListener("change", (e) => { state.speed = +e.target.value; });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.code === "Space") {
      e.preventDefault();
      state.playing ? pause() : play();
    } else if (e.code === "ArrowRight") {
      pause(); setFrame(state.frame + (e.shiftKey ? 10 : 1));
    } else if (e.code === "ArrowLeft") {
      pause(); setFrame(state.frame - (e.shiftKey ? 10 : 1));
    }
  });

  drawFrame();
  loadOpponents();
}

document.addEventListener("DOMContentLoaded", init);
