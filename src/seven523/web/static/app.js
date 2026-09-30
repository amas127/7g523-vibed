/* 7g523-web — the 牌桌 page.
 *
 * The server exposes the real engine and the real placement session as JSON;
 * this file is presentation plus one fetch helper.  No framework, no build
 * step: open the page the server prints and play. */

"use strict";

const SUIT_BY_LABEL = { "♦": 0, "♣": 1, "♥": 2, "♠": 3 };

const S = {
  cfg: null,
  snap: null,
  screen: "home",
  setup: {
    opponent: "random", seat: 0, seed: "", games: 10, usePrior: true, search: null,
    twinPairs: 30, twinSeed: "",
  },
  lastStart: null,
  pendingSuit: null, // { actionId }
  selected: [], // card labels tapped in hand
  hoverAction: null,
  busy: false,
  showLogA: false,
  counterOpen: false,
};

/* -- helpers ---------------------------------------------------------------- */

const $ = (sel, root) => (root || document).querySelector(sel);

function esc(value) {
  return String(value == null ? "" : value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function fmt(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits == null ? 0 : digits);
}

async function api(path, body) {
  const options = body === undefined
    ? undefined
    : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      };
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({ error: "服务器返回了非 JSON" }));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

let toastTimer = null;
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3600);
}

function round2(value) { return Math.round(Number(value)); }

function defaultSearch(search) {
  const out = {};
  const options = (search && search.options) || {};
  Object.keys(options).forEach((key) => { out[key] = options[key].default; });
  return out;
}

function sameSearch(a, b) {
  return Boolean(a && b) &&
    Number(a.trunc_ply) === Number(b.trunc_ply) &&
    Number(a.rollout_k) === Number(b.rollout_k);
}

function searchLabel(search) {
  if (!search) return "";
  const t = Number(search.trunc_ply) === 0 ? "全量" : `t=${search.trunc_ply}`;
  return `${t} · K=${search.rollout_k}`;
}

function armLabel(twin, arm) {
  const arms = (twin && twin.arms) || {};
  const entry = arm === "search" ? arms.search : arms.raw;
  if (entry && entry.label) return entry.label;
  return arm === "search" ? "搜索" : "raw";
}

function fmtSigned(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  return (number > 0 ? "+" : "") + number.toFixed(digits == null ? 1 : digits);
}

function signCounts(sign) {
  if (!sign) return "—";
  return `好 ${sign.improved || 0} / 平 ${sign.same || 0} / 差 ${sign.worsened || 0}`;
}

function fmtCi(interval) {
  if (!interval) return "";
  return `[${fmtSigned(interval[0])}, ${fmtSigned(interval[1])}]`;
}

function resultZh(result) {
  return result === "win" ? "胜" : result === "loss" ? "负" : "平";
}

/* -- card primitives -------------------------------------------------------- */

function cardHTML(card, opts) {
  opts = opts || {};
  const cls = ["card"];
  if (card.red) cls.push("red");
  if (card.joker) cls.push("joker");
  if (opts.size) cls.push(opts.size);
  if (opts.highlight) cls.push("hl");
  if (opts.selected) cls.push("sel");
  const suit = card.suit == null ? "★" : card.suit;
  return (
    `<div class="${cls.join(" ")}" data-label="${esc(card.label)}" data-card="${esc(card.label)}" title="${esc(card.label)}">` +
    `<span class="card-corner"><span class="card-rank">${esc(card.rank)}</span>` +
    `<span class="card-suit">${suit}</span></span>` +
    `<span class="card-center">${suit}</span>` +
    (card.points ? `<span class="card-points">${card.points}</span>` : "") +
    `</div>`
  );
}

function cardsRow(cards, size) {
  return cards.map((card) => cardHTML(card, { size })).join("");
}

function handHTML(cards, highlight, extraClass) {
  const hl = new Set(highlight || []);
  const sel = new Set(S.selected || []);
  return (
    `<div class="hand ${extraClass || ""}">` +
    cards
      .map((card) =>
        cardHTML(card, {
          highlight: hl.has(card.label) || sel.has(card.label),
          selected: sel.has(card.label),
        })
      )
      .join("") +
    `</div>`
  );
}

function actionFor(snap, actionId) {
  if (!snap || !snap.legal) return null;
  return snap.legal.find((a) => a.action_id === actionId) || null;
}

function highlightLabels(snap) {
  const pending = S.pendingSuit && actionFor(snap, S.pendingSuit.actionId);
  const hovered = S.hoverAction != null && actionFor(snap, S.hoverAction);
  const chosen = pending || hovered;
  const labels = chosen ? chosen.cards.map((c) => c.label) : [];
  return labels.concat(S.selected || []);
}

function lastPlayOf(snap, seat) {
  const plays = (snap.plays || []).filter((p) => p.seat === seat);
  return plays.length ? plays[plays.length - 1] : null;
}

function starterSeat(snap) {
  // 亮牌定先：the seat holding the smallest revealed card (card_key order).
  const revealed = snap.revealed || [];
  if (!revealed.length) return null;
  const key = (entry) => {
    const card = entry.card;
    const suit = card.suit == null ? -1 : SUIT_BY_LABEL[card.suit];
    return card.order * 10 + suit;
  };
  return revealed.reduce((best, entry) => (key(entry) < key(best) ? entry : best), revealed[0]).seat;
}

function firstTrickOver(snap) {
  // The reveal is only the opening ceremony: hide it the moment trick 1 ends,
  // including while its winner is still deciding what to lead next.
  if (typeof snap.tricks_completed === "number") return snap.tricks_completed >= 1;
  // Compatibility with a server process started before tricks_completed existed
  // (static files hot-reload, the Python process does not).  The trick event is
  // the public marker; a second 开墩 also implies trick 1 is over.
  if ((snap.log || []).some((entry) => entry.type === "trick")) return true;
  return (snap.plays || []).filter((play) => play.opens_trick).length >= 2;
}

function revealSlot(snap, seat) {
  if (firstTrickOver(snap)) return "";
  const entry = (snap.revealed || []).find((r) => r.seat === seat);
  if (!entry) return "";
  const played = new Set(
    (snap.plays || []).flatMap((play) => (play.cards || []).map((card) => card.label))
  );
  const isPlayed = played.has(entry.card.label);
  const first = starterSeat(snap) === seat;
  const caption = first ? "亮牌 · 先手" : "亮牌";
  return (
    `<div class="reveal-slot ${isPlayed ? "played" : ""} ${first ? "first" : ""}" title="开局亮牌（最小牌）：${esc(entry.card.label)}">` +
    cardHTML(entry.card) +
    `<span class="reveal-cap">${caption}${isPlayed ? " · 已出" : ""}</span>` +
    `</div>`
  );
}

/* -- shared game pieces ----------------------------------------------------- */

function scoreChips(snap) {
  return (snap.scores || [])
    .map(
      (score, seat) =>
        `<span class="badge score-chip ${seat === snap.human_seat ? "gold" : ""}">` +
        `${esc(snap.names[seat] || "座位" + seat)} <b>${score}</b></span>`
    )
    .join("");
}

function modeBar(snap) {
  const twin = snap.mode === "twin" ? (snap.twin || {}) : null;
  const current = twin && twin.current;
  const modeLabel =
    snap.mode === "twin"
      ? `deal-twin · pair ${current ? current.pair + 1 : "—"}/${twin.pairs_total || "—"}`
      : snap.mode === "placement"
        ? `定级 ${snap.game_index}/${snap.games_total}`
        : "自由对战";
  const opp = snap.opponent;
  return (
    `<div class="modebar">` +
    `<span class="badge gold">${modeLabel}</span>` +
    (opp ? `<span class="badge">对手 ${esc(opp.id)} · μ${round2(opp.mu)}</span>` : "") +
    (twin && current
      ? `<span class="badge ${current.arm === "search" ? "red" : ""}">当前臂 ${esc(armLabel(twin, current.arm))}</span>`
      : "") +
    (snap.search ? `<span class="badge">搜索 ${esc(searchLabel(snap.search))}</span>` : "") +
    `<span class="badge">种子 ${snap.seed == null ? "—" : snap.seed}</span>` +
    `<span class="spacer"></span>` +
    scoreChips(snap) +
    `<button class="btn ghost" data-nav="quit">结束本局</button>` +
    `</div>`
  );
}

function seatBox(snap, seat) {
  const count = (snap.hand_counts || [])[seat] || 0;
  const last = lastPlayOf(snap, seat);
  const backs = Array.from({ length: Math.min(count, 7) }, () => `<div class="card back"></div>`).join("");
  return (
    `<div class="seat-box">` +
    `<div class="seat-score"><b>${snap.scores[seat]}</b><small>分</small></div>` +
    `<div>` +
    `<div class="who">${esc(snap.names[seat] || "座位" + seat)}${seat === snap.human_seat ? "（你）" : ""}</div>` +
    `<div class="meta">手牌 ${count}` +
    (last ? ` · 上一手 ${esc(last.label)}` : "") +
    `</div>` +
    `</div>` +
    revealSlot(snap, seat) +
    `<div class="oppo-cards">${backs}</div>` +
    `</div>`
  );
}

function drawPile(snap) {
  return (
    `<div class="draw-pile">` +
    `<div class="card back"></div><div class="card back"></div><div class="card back"></div>` +
    `<div class="count">底牌 ${snap.draw_count}</div>` +
    `</div>`
  );
}

function logLines(snap, limit) {
  const log = (snap.log || []).slice(-(limit || 40));
  if (!log.length) return `<small>等待第一步…</small>`;
  return log
    .map((entry) => {
      if (entry.type === "trick") {
        const refill = (entry.refilled || [])
          .map((seat) => esc(snap.names[seat] || "座位" + seat))
          .join("、");
        return (
          `<div class="line trick">◆ ${esc(snap.names[entry.winner] || "座位" + entry.winner)} 收 ${entry.points} 分` +
          (entry.dug ? " · 撬底！" : "") +
          (refill ? ` · 补牌 ${refill}` : "") +
          `</div>`
        );
      }
      return (
        `<div class="line"><span class="seat">${esc(snap.names[entry.seat] || "座位" + entry.seat)}</span>` +
        `<span>${esc(entry.text)}</span></div>`
      );
    })
    .join("");
}

function placementPanel(snap) {
  const p = snap.placement;
  if (!p) return "";
  const done = p.games_played || 0;
  const total = p.games_total || 0;
  const est = p.estimate;
  const pct = total ? Math.round((100 * done) / total) : 0;
  const rows = (p.records || [])
    .slice(-6)
    .map(
      (r) =>
        `<div class="record-row"><span>#${r.index + 1} ${esc(r.opponent_id)} · ${r.seat === 0 ? "0 号位" : "1 号位"}</span>` +
        `<span class="${r.result === "win" ? "w" : r.result === "loss" ? "l" : "d"}">` +
        `${r.scores[r.seat]}:${r.scores[1 - r.seat]} ${r.result === "win" ? "胜" : r.result === "loss" ? "负" : "平"}</span></div>`
    )
    .join("");
  return (
    `<section class="panel progress-panel">` +
    `<h2>定级进度</h2>` +
    `<div class="metric"><span>已完成</span><b>${done}/${total}</b></div>` +
    `<div class="bar"><i style="width:${pct}%"></i></div>` +
    (est
      ? `<div class="metric"><span>当前估计</span><b>${fmt(est.mu)}</b></div>` +
        `<div class="metric"><span>95% CI</span><b>± ${fmt(est.ci_half_width)}</b></div>` +
        `<div class="metric"><span>最近档</span><b>${esc((est.nearest_level || {}).id || "—")} (${fmt((est.nearest_level || {}).mu)})</b></div>` +
        `<div class="metric"><span>状态</span><b>${est.provisional ? "临时 provisional" : "已达 ±50"}</b></div>`
      : `<small>第一局结束后给出两通道估计。</small>`) +
    (rows ? `<div class="record-list">${rows}</div>` : "") +
    `</section>`
  );
}

function twinPanel(snap) {
  const t = snap.twin;
  if (!t) return "";
  const interim = t.interim || {};
  const pct = t.games_total ? Math.round((100 * t.games_played) / t.games_total) : 0;
  const current = t.current;
  return (
    `<section class="panel progress-panel">` +
    `<h2>deal-twin 进度</h2>` +
    `<div class="metric"><span>完成 pair</span><b>${t.pairs_complete || 0}/${t.pairs_total || 0}</b></div>` +
    `<div class="bar"><i style="width:${pct}%"></i></div>` +
    `<div class="metric"><span>当前臂</span><b>${current ? esc(armLabel(t, current.arm)) : "—"}</b></div>` +
    `<div class="metric"><span>Δ 分差</span><b>${fmtSigned(interim.margin_points_delta)}</b></div>` +
    `<div class="metric"><span>Δ 胜率</span><b>${fmtSigned(interim.winrate_delta, 2)}</b></div>` +
    `<div class="metric"><span>符号（Δ>0）</span><b>${signCounts(interim.sign)}</b></div>` +
    (t.pairs_incomplete
      ? `<div class="metric"><span>未完成 pair</span><b>${t.pairs_incomplete}</b></div>`
      : "") +
    `<small>Δ = 对搜索臂 − 对 raw 臂；只在完整 pair 上计算（每 pair 同 seed 同座位）。</small>` +
    `</section>`
  );
}

function renderCounter(snap) {
  const counter = snap.counter;
  if (!counter) return "";
  const open = Boolean(S.counterOpen);
  const hotRanks = new Set(["7", "大王", "小王"]);
  const rows = (counter.rows || [])
    .map((row) => {
      const hot = row.points > 0 || hotRanks.has(row.rank);
      const gone = row.unseen === 0;
      const points = row.points && row.unseen ? `<small> ${row.unseen * row.points}分</small>` : "";
      return (
        `<tr class="${gone ? "gone" : ""} ${hot ? "hot" : ""}">` +
        `<td class="rank">${esc(row.rank)}</td>` +
        `<td>${row.mine || "·"}</td>` +
        `<td>${row.played || "·"}</td>` +
        `<td><b>${row.unseen}</b>${points}</td>` +
        `</tr>`
      );
    })
    .join("");
  return (
    `<section class="panel counter ${open ? "open" : ""}">` +
    `<button class="counter-head" data-nav="toggle-counter-table" aria-expanded="${open}" title="${open ? "收起记牌器" : "展开记牌器"}">` +
    `<span class="counter-title">记牌器</span>` +
    `<span class="counter-summary">` +
    `<span class="badge gold">未见 ${counter.unseen_cards} 张</span>` +
    `<span class="badge ${counter.unseen_points ? "red" : "green"}">分牌 ${counter.unseen_points} 分</span>` +
    `</span>` +
    `<span class="counter-caret">${open ? "▾" : "▸"}</span>` +
    `</button>` +
    (open
      ? `<table><thead><tr><th>牌</th><th>我</th><th>已出</th><th>余</th></tr></thead><tbody>${rows}</tbody></table>` +
        `<small class="counter-note">余 = 底牌堆 + 对手手牌（含未打出的亮牌）；已出 = 公开出牌日志</small>`
      : "") +
    `</section>`
  );
}

function matchSelection(snap) {
  const selected = S.selected || [];
  if (!selected.length || !snap || !snap.legal) return null;
  const selSet = new Set(selected);
  for (const action of snap.legal) {
    if (action.kind_key === "pass") continue;
    const labels = action.cards.map((c) => c.label);
    if (labels.length === selected.length && labels.every((l) => selSet.has(l))) {
      return { action, suit: suitFromSelection(action, selected) };
    }
  }
  // Suit-swapped realisation: same rank multiset, top card of a different suit.
  const hand = snap.your_hand || [];
  const selRanks = selected
    .map((label) => (hand.find((c) => c.label === label) || {}).rank)
    .filter(Boolean)
    .sort();
  for (const action of snap.legal) {
    if (action.kind_key === "pass") continue;
    const ranks = action.cards.map((c) => c.rank).sort();
    if (ranks.length === selRanks.length && ranks.every((r, i) => r === selRanks[i])) {
      return { action, suit: suitFromSelection(action, selected) };
    }
  }
  return null;
}

function suitFromSelection(action, selected) {
  if (!action.suits || action.suits.length <= 1) return null;
  const hand = S.snap.your_hand || [];
  const top = selected
    .map((label) => hand.find((c) => c.label === label))
    .find((card) => card && card.order === action.top_order);
  if (!top) return null;
  const value = SUIT_BY_LABEL[top.label[0]];
  if (value == null) return null;
  return action.suits.some((s) => s.value === value) ? value : null;
}

function selectionBar(snap) {
  if (!S.selected || !S.selected.length) return "";
  const hand = snap.your_hand || [];
  const cards = S.selected.map((label) => hand.find((c) => c.label === label)).filter(Boolean);
  const match = matchSelection(snap);
  return (
    `<div class="selection-bar">` +
    `<span class="sel-cards">${cardsRow(cards, "mini")}</span>` +
    (match
      ? `<span class="sel-ok">可出：${esc(match.action.label)}</span>` +
        `<button class="btn primary" data-nav="play-selection">出牌</button>`
      : `<span class="sel-bad">这组牌不是合法牌型（或压不过当前顶牌）</span>`) +
    `<button class="btn ghost" data-nav="clear-selection">清空</button>` +
    `</div>`
  );
}

function passAction(snap) {
  return (snap.legal || []).find((a) => a.kind_key === "pass") || null;
}

function pendingSuitBar(snap) {
  if (!S.pendingSuit) return "";
  const action = actionFor(snap, S.pendingSuit.actionId);
  if (!action || !action.suits.length) return "";
  const buttons = action.suits
    .map(
      (suit) =>
        `<button class="suit-btn ${"♥♦".includes(suit.label) ? "red" : ""}" data-suit="${suit.value}" title="顶牌用 ${suit.label}">${suit.label}</button>`
    )
    .join("");
  return (
    `<div class="suit-bar"><span>选择顶牌花色：</span>${buttons}` +
    `<small>默认最强（${esc(action.suits[0].label)}）</small>` +
    `<button class="btn ghost" data-cancel-suit>取消</button></div>`
  );
}

/* -- the table -------------------------------------------------------------- */

function trickBlock(snap) {
  // The incumbent is already shown as 当前顶牌; show only the earlier plays
  // of this trick here so the same physical card never appears twice.
  const incumbent = new Set(((snap.incumbent || {}).cards || []).map((card) => card.label));
  const cards = (snap.trick_cards || []).filter((card) => !incumbent.has(card.label));
  if (!cards.length) return "";
  return (
    `<div class="felt-block">` +
    `<div class="felt-title">本墩弃牌 · ${snap.trick_points} 分</div>` +
    `<div class="trick-cards">${cardsRow(cards, "small")}</div>` +
    `</div>`
  );
}

function incumbentBlock(snap) {
  const inc = snap.incumbent;
  if (!inc) {
    return (
      `<div class="felt-block">` +
      `<div class="felt-title">当前顶牌</div>` +
      `<div class="turn-hint">${snap.current === snap.human_seat ? "等你领出" : "等待对手领出…"}</div>` +
      `</div>`
    );
  }
  return (
    `<div class="felt-block">` +
    `<div class="felt-title">当前顶牌 · ${esc(snap.names[inc.seat] || "")}</div>` +
    `<div class="trick-cards">${cardsRow(inc.cards, "small")}</div>` +
    `<div class="turn-hint">${esc(inc.label)}</div>` +
    `</div>`
  );
}

function actionStrip(snap) {
  const chips = (snap.legal || []).map((action, index) => {
    const isPass = action.kind_key === "pass";
    const inner = isPass
      ? `<span>过</span>`
      : `<span class="chip-cards">${cardsRow(action.cards, "mini")}</span>` +
        `<span class="chip-label">${esc(action.label)}</span>`;
    return (
      `<button class="action-chip ${isPass ? "pass" : ""} ${action.bomb ? "bomb" : ""}"` +
      ` data-play="${action.action_id}" title="快捷键 ${index + 1}">${inner}</button>`
    );
  });
  return `<div class="action-strip">${chips.join("")}</div>`;
}

function renderTable(snap) {
  const oppSeat = 1 - snap.human_seat;
  return (
    `<div class="va">` +
    `<div class="va-top">${modeBar(snap)}</div>` +
    `<div class="va-board">` +
    `<div class="va-opp">${seatBox(snap, oppSeat)}</div>` +
    `<div class="va-felt ${S.showLogA ? "with-log" : ""}">` +
    `<button class="btn ghost felt-toggle" data-nav="toggle-log">${S.showLogA ? "收起日志" : "日志"}</button>` +
    `<div class="felt-center">` +
    drawPile(snap) +
    trickBlock(snap) +
    incumbentBlock(snap) +
    `</div>` +
    (S.showLogA
      ? `<div class="felt-log"><div class="felt-title">动作日志</div>${logLines(snap, 40)}</div>`
      : "") +
    `</div>` +
    `<div class="va-side">${placementPanel(snap)}${twinPanel(snap)}${renderCounter(snap)}</div>` +
    `</div>` +
    `<div class="va-bottom">` +
    `<div class="turn-hint ${snap.phase === "human" ? "yours" : ""}">` +
    (snap.phase === "human" ? "轮到你出牌 — 从下面选一手" : "等待中…") +
    `</div>` +
    pendingSuitBar(snap) +
    `<div class="va-hand-row">` +
    `<div class="va-mystat"><span>你</span><b>${snap.scores[snap.human_seat]}</b><small>分</small></div>` +
    `<div class="va-myhand">${handHTML(snap.your_hand, highlightLabels(snap))}</div>` +
    revealSlot(snap, snap.human_seat) +
    `</div>` +
    selectionBar(snap) +
    `<div class="va-actions">${actionStrip(snap)}</div>` +
    `</div>` +
    `</div>`
  );
}

/* -- screens: home / setup --------------------------------------------------- */

function searchControls(search) {
  const options = search.options || {};
  const t = options.trunc_ply || {};
  const k = options.rollout_k || {};
  const current = S.setup.search || defaultSearch(search);
  const presets = (search.presets || [])
    .map((preset, index) =>
      `<button data-search-preset="${index}" class="${sameSearch(preset, current) ? "on" : ""}">${esc(preset.label)}</button>`
    )
    .join("");
  return (
    `<div class="field"><label>搜索深度（逐局可选）</label>` +
    `<div class="search-row">` +
    `<input type="number" id="search-t" min="${t.min}" max="${t.max}" value="${esc(current.trunc_ply)}" title="截断 t（0 = 全量）">` +
    `<input type="number" id="search-k" min="${k.min}" max="${k.max}" value="${esc(current.rollout_k)}" title="K 个隐藏世界">` +
    `</div>` +
    (presets ? `<div class="seg">${presets}</div>` : "") +
    `<small>左 = t（${esc(t.note || "")}），右 = K（${esc(k.note || "")}）。</small></div>`
  );
}

function renderHome() {
  const search = S.cfg && S.cfg.search;
  const twin = S.cfg && S.cfg.twin;
  const plugin = S.cfg && S.cfg.plugin;
  return (
    `<div class="screen"><div class="wrap">` +
    `<h1>7鬼523 · 牌桌</h1>` +
    `<div class="sub">7g523-elo 的卡牌交互前端草稿 — 连的是真实引擎与真实定级管线。` +
    `点手牌或牌型按钮出牌；右栏记牌器只用公开信息推算。</div>` +
    (plugin && plugin.error
      ? `<div class="hint-note bad">插件加载失败，已回退 raw-only：${esc(plugin.error)}` +
        `<br>用 <code>--plugin PATH</code> 显式指定，或检查 <code>${esc(plugin.source || "")}</code>。</div>`
      : "") +
    (search
      ? `<div class="hint-note">搜索已启用：${esc(search.label)}。` +
        `自由对战中的 ckpt 对手由搜索包装，设置页可<b>逐局选择 t / K</b>；` +
        `定级使用 manifest 的<b>单一对手池</b>（raw 档 + 搜索 rung 同表调度，` +
        `搜索 rung 的固定配置来自 manifest 的 <code>search_config</code>）。</div>`
      : "") +
    (twin
      ? `<div class="hint-note">deal-twin 已启用：${esc(twin.label || "")}。` +
        `每个 deal 打两局（同 seed 同座位），跨 pair 座位/顺序交替；` +
        `<b>≥${twin.min_pairs || 30} 对</b>才有分辨力（会话可另选 2..100 偶数），` +
        `结果只落 <code>traces/twins/</code>。</div>`
      : `<div class="hint-note">deal-twin 未启用：用 ` +
        `<code>uv run --group train python runs/o4lite-search/web_twin.py</code> 启动。</div>`) +
    `<div class="mode-cards">` +
    `<button class="mode-card" data-nav="place_setup">` +
    `<h2>定级模式</h2>` +
    `<p>manifest 单一对手池（锚点 + raw 档 + 可构建的搜索 rung），自适应选档 + 双通道估计，` +
    `产物与 <code>7g523-elo</code> 一致；搜索 rung 行不进轨迹先验并留有记录。</p>` +
    `</button>` +
    (twin
      ? `<button class="mode-card" data-nav="twin_setup">` +
        `<h2>deal-twin 对比</h2>` +
        `<p>同一副牌、同一个你：一次对 raw、一次对搜索（t=5 K=32 C=6），两局同 seed 同座位。</p>` +
        `<p>输出配对 Δ（对搜索 − 对 raw）与 pair-cluster CI；不进评分，只落 <code>traces/twins/</code>。</p>` +
        `</button>`
      : "") +
    `<button class="mode-card" data-nav="free_setup">` +
    `<h2>自由对战</h2>` +
    `<p>从当前 10 级池任选对手（random 锚 + lvl1–lvl4 + 顶部平台簇），指定座位直接开打。</p>` +
    `<p>每局都是可回放的真实 trace，落到 <code>traces/web/</code>。</p>` +
    `</button>` +
    `</div>` +
    `<div class="hint-note">对手强度是 RandomBot=0 的 probit-MLE 绝对表口径（manifest 契约值）。` +
    `本页是 throwaway 原型：无测试、无鉴权、单会话。</div>` +
    `</div></div>`
  );
}

function renderFreeSetup() {
  const s = S.setup;
  const opponents = (S.cfg && S.cfg.opponents) || [];
  const torch = Boolean(S.cfg && S.cfg.torch);
  const search = S.cfg && S.cfg.search;
  const cards = opponents
    .map((opponent) => {
      const unavailable = !opponent.anchor && !torch;
      const tag = opponent.anchor
        ? "RandomBot（0 基准）"
        : !torch
          ? "需要 torch（--group train）"
          : search
            ? "搜索包装"
            : "checkpoint 梯级";
      return (
        `<button class="opp-card ${s.opponent === opponent.id ? "selected" : ""}" data-opp="${esc(opponent.id)}" ${unavailable ? "disabled" : ""}>` +
        `<div class="id">${esc(opponent.id)}</div>` +
        `<div class="mu">μ ${round2(opponent.mu)}${opponent.anchor ? " · 锚" : ""}</div>` +
        `<div class="tag">${tag}</div>` +
        `</button>`
      );
    })
    .join("");
  return (
    `<div class="screen"><div class="wrap">` +
    `<h1>自由对战</h1>` +
    `<div class="sub">选一个对手、指定座位，直接开打。每局都记录真实 trace（可用 7g523-play --replay 校验）。</div>` +
    (search
      ? `<div class="hint-note">搜索已启用：${esc(search.label)}。` +
        `ckpt 对手由 O4-lite 搜索包装（random 锚不包装）· ${esc(search.note || "")}</div>`
      : "") +
    (search ? searchControls(search) : "") +
    (search
      ? ""
      : `<div class="hint-note">当前服务进程未启用搜索包装（没有逐局 t/K）。` +
        `用 <code>uv run --group train python runs/o4lite-search/web_search.py</code> ` +
        `启动即可启用。</div>`) +
    `<div class="setup-grid">` +
    `<div><div class="field"><label>对手（${opponents.length} 级）</label><div class="opp-grid">${cards}</div>` +
    (torch ? "" : `<div class="hint-note">本进程没有 torch，ckpt 对手已禁用；用 <code>uv run --group train python …</code> 启动即可启用。</div>`) +
    `</div></div>` +
    `<div>` +
    `<div class="field"><label>座位</label>` +
    `<div class="seg">` +
    `<button data-seat="0" class="${s.seat === 0 ? "on" : ""}">0 号位</button>` +
    `<button data-seat="1" class="${s.seat === 1 ? "on" : ""}">1 号位</button>` +
    `</div><small>座位只影响谁先手（先手由亮牌定），与强弱无关。</small></div>` +
    `<div class="field"><label>随机种子（可留空）</label>` +
    `<input type="text" id="seed-input" value="${esc(s.seed)}" placeholder="例如 42"></div>` +
    `<div class="actions">` +
    `<button class="btn primary" data-nav="start_free" ${S.busy ? "disabled" : ""}>开始对局</button>` +
    `<button class="btn ghost" data-nav="home">返回</button>` +
    `</div>` +
    `</div>` +
    `</div>` +
    `</div></div>`
  );
}

function renderPlaceSetup() {
  const s = S.setup;
  const prior = S.cfg && S.cfg.prior;
  const gameButtons = (values) =>
    values
      .map((n) => `<button data-games="${n}" class="${s.games === n ? "on" : ""}">${n} 局</button>`)
      .join("");
  return (
    `<div class="screen"><div class="wrap">` +
    `<h1>定级模式</h1>` +
    `<div class="sub">manifest 单一对手池：锚点 + raw 档 + 可构建的搜索 rung，同一联合拟合尺度。` +
    `产物与 <code>7g523-elo</code> 一致（<code>traces/sessions/</code>）。</div>` +
    `<div class="setup-grid">` +
    `<div>` +
    `<div class="field"><label>局数（偶数，5/5 座位轮换）</label><div class="seg">${gameButtons([2, 4, 6, 10])}</div></div>` +
    `<div class="field"><label>估计通道</label>` +
    `<div class="seg">` +
    `<button data-prior="1" class="${s.usePrior ? "on" : ""}" ${prior ? "" : "disabled"}>轨迹先验 + 结果（推荐）</button>` +
    `<button data-prior="0" class="${s.usePrior ? "" : "on"}">只用结果（冷启动）</button>` +
    `</div>` +
    (prior
      ? `<small>先验 ${esc(prior.path)} · v${prior.version}${prior.kind ? " · " + esc(prior.kind) : ""}</small>`
      : `<small>服务端以 --no-trace-prior 启动，只能冷启动。</small>`) +
    `<div><small>${S.cfg && S.cfg.torch ? "本进程 torch 可用：定级池的 ckpt 对手会真实加载。" : "本进程无 torch：ckpt 对手不可用。"}</small></div>` +
    `</div>` +
    `<div>` +
    `<div class="field"><label>随机种子（可留空）</label>` +
    `<input type="text" id="seed-input" value="${esc(s.seed)}" placeholder="例如 42"></div>` +
    `<div class="actions">` +
    `<button class="btn primary" data-nav="start_placement" ${S.busy ? "disabled" : ""}>开始定级</button>` +
    `<button class="btn ghost" data-nav="home">返回</button>` +
    `</div>` +
    `<div class="hint-note">` +
    `<b>流程</b>：前 2 局 Thompson 探索（防先验偏），之后按 Fisher 信息选最难分辨的档；` +
    `每局结束更新两通道估计，CI ≤ 50 可提前收尾。<br>` +
    `<b>搜索 rung</b>：按 manifest 的 <code>search_config</code> 固定测量身份；` +
    `该局的行不会进入轨迹先验（ADR-0013），报告里会有 <code>prior_off_reason</code> 记录。<br>` +
    `<b>注意</b>：10 局只承诺点估计 + 诚实 CI（历史研究 RMSE ≈ 54–72），provisional 是正常状态。` +
    `</div>` +
    `</div>` +
    `</div>` +
    `</div></div>`
  );
}

/* -- screens: results -------------------------------------------------------- */

function renderTwinSetup() {
  const twin = S.cfg && S.cfg.twin;
  if (!twin) return renderHome();
  const s = S.setup;
  const raw = twin.raw || {};
  const search = twin.search || {};
  const params = search.params || {};
  const minPairs = twin.min_pairs || 30;
  const options = twin.pairs_options || [10, 20, 30, 50];
  const pairs = options
    .map(
      (n) =>
        `<button data-twin-pairs="${n}" class="${s.twinPairs === n ? "on" : ""}">${n} 对</button>`
    )
    .join("");
  return (
    `<div class="screen"><div class="wrap">` +
    `<h1>deal-twin 对比</h1>` +
    `<div class="sub">同一副牌、同一个你：每个 deal 打两局（同 seed、同座位），一次对 raw、一次对搜索。` +
    `Δ = 对搜索臂分差 − 对 raw 臂分差；座位与先后顺序跨 pair 交替。</div>` +
    `<div class="setup-grid">` +
    `<div>` +
    `<div class="field"><label>raw 臂（对照）</label>` +
    `<div class="hint-note"><b>${esc(raw.label || raw.id || "raw")}</b><br>` +
    `<code>${esc(raw.spec || "")}</code></div></div>` +
    `<div class="field"><label>搜索臂</label>` +
    `<div class="hint-note"><b>${esc(search.label || search.id || "search")}</b><br>` +
    `<code>${esc(search.identity || search.base_spec || "")}</code><br>` +
    `t=${esc(params.trunc_ply)} · K=${esc(params.rollout_k)} · C=${esc(params.max_candidates)} · ` +
    `value <code>${esc(search.value_ckpt || "")}</code></div></div>` +
    `</div>` +
    `<div>` +
    `<div class="field"><label>pair 数（每 pair 两局，2..100 偶数）</label><div class="seg">${pairs}</div>` +
    `<input type="number" id="twin-pairs-input" min="2" max="100" step="2" value="${esc(s.twinPairs)}">` +
    (s.twinPairs < minPairs
      ? `<small class="bad">只有 ${s.twinPairs} 对（<${minPairs}）：点估计/CI 只作流程验证。</small>`
      : `<small>≥${minPairs} 对（${minPairs * 2} 局）才有分辨力；默认 ${twin.default_pairs || 30}。</small>`) +
    `</div>` +
    `<div class="field"><label>调度种子（可留空）</label>` +
    `<input type="text" id="twin-seed-input" value="${esc(s.twinSeed)}" placeholder="例如 7"></div>` +
    `<div class="actions">` +
    `<button class="btn primary" data-nav="start_twin" ${S.busy ? "disabled" : ""}>开始 twin 会话</button>` +
    `<button class="btn ghost" data-nav="home">返回</button>` +
    `</div>` +
    `<div class="hint-note">` +
    `<b>读数口径</b>：只用完整 pair，pair-cluster bootstrap（B=${esc(twin.bootstrap || 4000)}）出 CI；` +
    `open-label、单 bank、CRN 记忆等 caveats 见结算页。<br>` +
    `<b>产物</b>：<code>traces/twins/</code>（serve）或 <code>--out</code>（simulate）；` +
    `不进 manifest / prior / 评分。</div>` +
    `</div>` +
    `</div>` +
    `</div></div>`
  );
}

function maybeProbabilisticTag(est) {
  if (!est) return "";
  return est.provisional
    ? `<span class="badge red">临时 provisional</span>`
    : `<span class="badge green">已达 ±50</span>`;
}

function renderResult() {
  const snap = S.snap || {};
  const result = snap.result || {};
  const aborted = snap.phase === "aborted";
  const scores = snap.scores || [];
  return (
    `<div class="screen"><div class="wrap"><div class="result-card">` +
    `<div class="big-outcome">${aborted ? "本局已中止" : esc(result.outcome || "对局结束")}</div>` +
    `<div class="scores">${scores.map((score, seat) => `<div>${esc(snap.names[seat])} <b>${score}</b></div>`).join("")}</div>` +
    `<dl class="kv">` +
    `<dt>对手</dt><dd>${esc((snap.opponent || {}).id)} (μ ${round2((snap.opponent || {}).mu)})</dd>` +
    `<dt>你坐</dt><dd>${snap.human_seat} 号位</dd>` +
    `<dt>种子</dt><dd>${snap.seed == null ? "—" : snap.seed}</dd>` +
    (snap.search ? `<dt>搜索</dt><dd>${esc(searchLabel(snap.search))}</dd>` : "") +
    (result.steps ? `<dt>步数</dt><dd>${result.steps}</dd>` : "") +
    (result.trace_path || snap.trace_path ? `<dt>轨迹</dt><dd><code>${esc(result.trace_path || snap.trace_path)}</code></dd>` : "") +
    `</dl>` +
    `<div class="actions">` +
    `<button class="btn primary" data-nav="rematch">再打一局</button>` +
    `<button class="btn" data-nav="free_setup">换对手</button>` +
    `<button class="btn ghost" data-nav="home">返回首页</button>` +
    `</div></div></div></div>`
  );
}

function renderRound() {
  const snap = S.snap || {};
  if (snap.mode === "twin") return renderTwinRound();
  const p = snap.placement || {};
  const records = p.records || [];
  const last = records[records.length - 1] || {};
  const est = p.estimate;
  const label = last.result === "win" ? "你赢了" : last.result === "loss" ? "你输了" : "平局";
  const rows = records
    .map(
      (r) =>
        `<tr><td>#${r.index + 1}</td><td>${esc(r.opponent_id)} (μ${round2(r.opponent_mu)})</td>` +
        `<td>${r.seat} 号位</td><td>${r.scores[r.seat]}:${r.scores[1 - r.seat]}</td>` +
        `<td>${r.result === "win" ? "胜" : r.result === "loss" ? "负" : "平"}</td></tr>`
    )
    .join("");
  return (
    `<div class="screen"><div class="wrap"><div class="result-card">` +
    `<div class="big-outcome">第 ${(last.index || 0) + 1}/${p.games_total} 局 · ${label}</div>` +
    `<div class="scores"><div>你 <b>${(last.scores || [])[last.seat]}</b></div><div>${esc(last.opponent_id)} <b>${(last.scores || [])[1 - last.seat]}</b></div></div>` +
    `<div class="modebar">` +
    `<span class="badge gold">当前估计 ${est ? fmt(est.mu) + " ± " + fmt(est.ci_half_width) : "—"}</span>` +
    (est && est.nearest_level ? `<span class="badge">最近档 ${esc(est.nearest_level.id)} (${fmt(est.nearest_level.mu)})</span>` : "") +
    maybeProbabilisticTag(est) +
    `</div>` +
    `<div class="bar"><i style="width:${p.games_total ? Math.round((100 * p.games_played) / p.games_total) : 0}%"></i></div>` +
    `<table class="games"><thead><tr><th>局</th><th>对手</th><th>座位</th><th>比分</th><th></th></tr></thead><tbody>${rows}</tbody></table>` +
    `<div class="actions">` +
    `<button class="btn primary" data-nav="continue">继续下一局</button>` +
    `<button class="btn" data-nav="quit">提前结束并出报告</button>` +
    `</div>` +
    `<div class="hint-note">每局结束都写入 trace 并更新 OpenSkill 后验；继续 = 按同一调度打下一条。</div>` +
    `</div></div></div>`
  );
}

function renderTwinRound() {
  const snap = S.snap || {};
  const t = snap.twin || {};
  const last = t.last || {};
  const interim = t.interim || {};
  const lastPair = interim.last_pair;
  const label = resultZh(last.result) === "胜" ? "你赢了" : last.result === "loss" ? "你输了" : "平局";
  const pairTable = lastPair
    ? `<table class="games"><thead><tr><th>pair ${lastPair.pair + 1} 臂</th><th>分差（你−对手）</th><th>结果</th><th>Δ分差</th></tr></thead><tbody>` +
      `<tr><td>raw</td><td>${fmtSigned(lastPair.margin_raw, 0)}</td><td>${resultZh(lastPair.result_raw)}</td>` +
      `<td rowspan="2"><b>${fmtSigned(lastPair.d_margin, 0)}</b></td></tr>` +
      `<tr><td>${esc(armLabel(t, "search"))}</td><td>${fmtSigned(lastPair.margin_search, 0)}</td><td>${resultZh(lastPair.result_search)}</td></tr>` +
      `</tbody></table>`
    : "";
  return (
    `<div class="screen"><div class="wrap"><div class="result-card">` +
    `<div class="big-outcome">第 ${t.games_played || 0}/${t.games_total || 0} 局 · ${label}</div>` +
    `<div class="scores">` +
    (last.scores || [])
      .map((score, seat) => `<div>${esc((snap.names || [])[seat] || "座位" + seat)} <b>${score}</b></div>`)
      .join("") +
    `</div>` +
    `<div class="modebar">` +
    `<span class="badge gold">pair ${last.pair != null ? last.pair + 1 : "—"} · ${esc(armLabel(t, last.arm))}</span>` +
    `<span class="badge">完成 pair ${t.pairs_complete || 0}/${t.pairs_total || 0}</span>` +
    `<span class="badge">Δ 分差 ${fmtSigned(interim.margin_points_delta)}</span>` +
    `<span class="badge">Δ 胜率 ${fmtSigned(interim.winrate_delta, 2)}</span>` +
    `</div>` +
    pairTable +
    `<div class="actions">` +
    `<button class="btn primary" data-nav="continue">继续下一局</button>` +
    `<button class="btn" data-nav="quit">提前结束并出报告</button>` +
    `</div>` +
    `<div class="hint-note">同一 pair 内两局的初始手牌与座位完全一致（硬 CRN）；第二局打完给出本 pair 的 Δ。</div>` +
    `</div></div></div>`
  );
}

function renderTwinSummary() {
  const snap = S.snap || {};
  const t = snap.twin || {};
  const report = t.report;
  if (!report) {
    return (
      `<div class="screen"><div class="wrap"><div class="result-card">` +
      `<div class="big-outcome">未完成任何 pair</div>` +
      `<div class="actions"><button class="btn primary" data-nav="twin_setup">重新开始</button>` +
      `<button class="btn ghost" data-nav="home">返回首页</button></div></div></div></div>`
    );
  }
  const delta = report.delta || {};
  const counts = report.counts || {};
  const margin = delta.margin_points || {};
  const winrate = delta.winrate || {};
  const elo = delta.elo || {};
  const sign = delta.sign || {};
  const swing = delta.outcome_swing || {};
  const resolution = report.resolution || {};
  const arms = report.arms || {};
  const caveats = (report.caveats || []).map((line) => `<li>${esc(line)}</li>`).join("");
  const okBadge = resolution.ok
    ? `<span class="badge green">≥${resolution.min_pairs} 对：可读</span>`
    : `<span class="badge red">仅流程验证</span>`;
  return (
    `<div class="screen"><div class="wrap"><div class="result-card">` +
    `<div class="big-outcome">deal-twin 报告${resolution.ok ? "" : "（样本不足）"}</div>` +
    `<div class="ci-big">Δ 分差 ${fmtSigned(margin.value)} ${esc(fmtCi(margin.ci95))}` +
    `<small> 分（对搜索 − 对 raw，95% CI）</small></div>` +
    `<div class="modebar">` +
    okBadge +
    `<span class="badge">完成 pair ${counts.pairs_complete || 0}/${counts.pairs_total || 0}</span>` +
    `<span class="badge ${counts.pairs_incomplete ? "red" : ""}">未完成 ${counts.pairs_incomplete || 0}</span>` +
    `<span class="badge">中途放弃 ${counts.abandoned_games || 0}</span>` +
    `</div>` +
    `<dl class="kv">` +
    `<dt>Δ 胜率（期望得分）</dt><dd>${fmtSigned(winrate.value, 3)} ${esc(fmtCi(winrate.ci95))}</dd>` +
    `<dt>raw / search 胜率</dt><dd>${fmt(winrate.raw_winrate, 2)} / ${fmt(winrate.search_winrate, 2)}</dd>` +
    `<dt>Δ Elo（描述）</dt><dd>${fmtSigned(elo.value, 1)} ${esc(fmtCi(elo.ci95))}</dd>` +
    `<dt>pair 符号</dt><dd>${signCounts(sign)} · p=${fmt(sign.p_value, 4)}</dd>` +
    `<dt>胜负翻转</dt><dd>更好 ${swing.better || 0} / 不变 ${swing.same || 0} / 更差 ${swing.worse || 0}</dd>` +
    `<dt>MDE80</dt><dd>${fmt(resolution.mde80_points)} 分</dd>` +
    `<dt>raw 臂</dt><dd><code>${esc((arms.raw || {}).spec || "")}</code></dd>` +
    `<dt>search 臂</dt><dd><code>${esc((arms.search || {}).identity || "")}</code><br>` +
    `<small>${esc(JSON.stringify((arms.search || {}).params || {}))}</small></dd>` +
    `<dt>会话目录</dt><dd><code>${esc((report.session || {}).directory || t.session_dir || "")}</code></dd>` +
    `<dt>报告文件</dt><dd><code>${esc((report.session || {}).report || "")}</code></dd>` +
    `</dl>` +
    `<div class="hint-note"><b>读数提醒</b>：${esc(resolution.note || "")}<ul>${caveats}</ul></div>` +
    `<div class="actions">` +
    `<button class="btn primary" data-nav="twin_setup">再测一次</button>` +
    `<button class="btn ghost" data-nav="home">返回首页</button>` +
    `</div>` +
    `</div></div></div>`
  );
}

function renderPlacementSummary() {
  const snap = S.snap || {};
  const p = snap.placement || {};
  const report = p.report;
  if (!report) {
    return (
      `<div class="screen"><div class="wrap"><div class="result-card">` +
      `<div class="big-outcome">未完成任何一局</div>` +
      `<div class="actions"><button class="btn primary" data-nav="place_setup">重新开始</button>` +
      `<button class="btn ghost" data-nav="home">返回首页</button></div></div></div></div>`
    );
  }
  const human = report.human;
  const nearest = report.nearest_level || {};
  const weights = (report.channels || {}).weights || {};
  const bandLabel =
    report.band === "placed" ? "已定级（±50 内）" : report.band === "provisional" ? "临时 provisional" : "粗略 coarse";
  const rows = (report.plan && report.plan.played ? report.plan.played : [])
    .map(
      (r) =>
        `<tr><td>#${r.index + 1}</td><td>${esc(r.opponent_id)}</td><td>${r.seat}</td>` +
        `<td>${r.scores[r.seat]}:${r.scores[1 - r.seat]}</td><td>${r.result === "win" ? "胜" : r.result === "loss" ? "负" : "平"}</td>` +
        `<td>${r.posterior_mu == null ? "—" : fmt(r.posterior_mu)}</td></tr>`
    )
    .join("");
  return (
    `<div class="screen"><div class="wrap"><div class="result-card">` +
    `<div class="big-outcome">定级报告</div>` +
    `<div class="ci-big">${fmt(human.mu)} ± ${fmt(human.ci_half_width)}<small> （95% CI，${human.n} 局）</small></div>` +
    `<div class="modebar">` +
    `<span class="badge gold">${bandLabel}</span>` +
    `<span class="badge">最近档 ${esc(nearest.id || "—")} (${fmt(nearest.mu)})</span>` +
    `<span class="badge">停止原因 ${esc((report.stop || {}).reason || "—")}</span>` +
    `</div>` +
    `<dl class="kv">` +
    `<dt>轨迹通道权重</dt><dd>${weights.trace == null ? "—" : fmt(weights.trace, 2)}</dd>` +
    `<dt>结果通道权重</dt><dd>${weights.result == null ? "—" : fmt(weights.result, 2)}</dd>` +
    `<dt>会话目录</dt><dd><code>${esc((report.session || {}).directory || p.session_dir || "")}</code></dd>` +
    `<dt>报告文件</dt><dd><code>${esc((report.session || {}).report || "")}</code></dd>` +
    `</dl>` +
    `<table class="games"><thead><tr><th>局</th><th>对手</th><th>座位</th><th>比分</th><th></th><th>后验 μ</th></tr></thead><tbody>${rows}</tbody></table>` +
    `<div class="actions">` +
    `<button class="btn primary" data-nav="place_setup">再测一次</button>` +
    `<button class="btn" data-nav="free_setup">去自由对战</button>` +
    `<button class="btn ghost" data-nav="home">返回首页</button>` +
    `</div>` +
    `<div class="hint-note">provisional 很正常：研究结论是 10 局 RMSE ≈ 54–72、95% CI ±100–133；` +
    `要收进 ±50 需继续对局（加局数或重复会话）。</div>` +
    `</div></div></div>`
  );
}

function renderError() {
  const snap = S.snap || {};
  return (
    `<div class="screen"><div class="wrap"><div class="result-card">` +
    `<div class="big-outcome">出错了</div>` +
    `<p>${esc(snap.error || "未知错误")}</p>` +
    `<div class="actions"><button class="btn primary" data-nav="home">返回首页</button></div>` +
    `</div></div></div>`
  );
}

/* -- root render ------------------------------------------------------------- */

function screenHTML() {
  const snap = S.snap;
  switch (S.screen) {
    case "free_setup": return renderFreeSetup();
    case "place_setup": return renderPlaceSetup();
    case "twin_setup": return renderTwinSetup();
    case "game": return renderTable(snap);
    case "result": return renderResult();
    case "round": return renderRound();
    case "placement_summary": return renderPlacementSummary();
    case "twin_summary": return renderTwinSummary();
    case "error": return renderError();
    default: return renderHome();
  }
}

function render() {
  const app = $("#app");
  app.innerHTML = screenHTML() + (S.busy ? `<div class="busy">对手思考中…</div>` : "");
  applyHighlight();
}

function applyHighlight() {
  const snap = S.snap;
  if (!snap || !snap.your_hand) return;
  const labels = new Set(highlightLabels(snap));
  document.querySelectorAll("#app .hand .card").forEach((el) => {
    el.classList.toggle("hl", labels.has(el.dataset.label));
  });
}

/* -- actions ----------------------------------------------------------------- */

function apply(snap) {
  if (!snap) return;
  S.snap = snap;
  S.pendingSuit = null;
  S.hoverAction = null;
  S.selected = [];
  const phase = snap.phase;
  if (phase === "human") S.screen = "game";
  else if (phase === "game_over" || phase === "aborted") S.screen = "result";
  else if (phase === "round_over") S.screen = "round";
  else if (phase === "session_over") {
    S.screen = snap.mode === "twin" ? "twin_summary" : "placement_summary";
  }
  else if (phase === "error") S.screen = "error";
  else S.screen = "home";
}

function chooseAction(actionId) {
  const snap = S.snap;
  if (!snap || snap.phase !== "human" || S.busy) return;
  const action = actionFor(snap, actionId);
  if (!action) return;
  if (action.kind_key === "pass" || action.suits.length <= 1) {
    playAction(action.action_id, null);
    return;
  }
  S.pendingSuit = { actionId: action.action_id };
  S.selected = [];
  render();
}

async function playAction(actionId, suit) {
  if (S.busy) return;
  S.busy = true;
  S.pendingSuit = null;
  S.selected = [];
  render();
  try {
    apply(await api("/api/action", { action_id: actionId, suit }));
  } catch (error) {
    toast(error.message);
  }
  S.busy = false;
  render();
}

async function startFree() {
  const s = S.setup;
  const seed = s.seed === "" ? null : s.seed;
  S.busy = true;
  render();
  try {
    const payload = { mode: "free", opponent_id: s.opponent, seat: s.seat, seed };
    if (S.cfg && S.cfg.search && s.search) payload.search = s.search;
    apply(await api("/api/start", payload));
    S.lastStart = { mode: "free", opponent_id: s.opponent, seat: s.seat, seed, search: s.search };
  } catch (error) {
    toast(error.message);
  }
  S.busy = false;
  render();
}

async function startPlacement() {
  const s = S.setup;
  S.busy = true;
  render();
  try {
    const seed = s.seed === "" ? null : s.seed;
    const payload = {
      mode: "placement",
      games: s.games,
      use_prior: s.usePrior,
      seed,
    };
    apply(await api("/api/start", payload));
  } catch (error) {
    toast(error.message);
  }
  S.busy = false;
  render();
}

async function startTwin() {
  const s = S.setup;
  S.busy = true;
  render();
  try {
    apply(await api("/api/start", {
      mode: "twin",
      pairs: s.twinPairs,
      seed: s.twinSeed === "" ? null : s.twinSeed,
    }));
  } catch (error) {
    toast(error.message);
  }
  S.busy = false;
  render();
}

async function postControl(path) {
  S.busy = true;
  render();
  try {
    apply(await api(path, {}));
  } catch (error) {
    toast(error.message);
  }
  S.busy = false;
  render();
}

/* -- events ------------------------------------------------------------------ */

function bindEvents() {
  const app = $("#app");

  app.addEventListener("click", (event) => {
    const cardEl = event.target.closest("[data-card]");
    if (cardEl && cardEl.closest(".hand") && !cardEl.closest("[data-play]")) {
      const label = cardEl.dataset.card;
      const index = S.selected.indexOf(label);
      if (index >= 0) S.selected.splice(index, 1);
      else S.selected.push(label);
      render();
      return;
    }
    const playEl = event.target.closest("[data-play]");
    if (playEl) {
      chooseAction(Number(playEl.dataset.play));
      return;
    }
    const suitEl = event.target.closest("[data-suit]");
    if (suitEl && S.pendingSuit) {
      playAction(S.pendingSuit.actionId, Number(suitEl.dataset.suit));
      return;
    }
    if (event.target.closest("[data-cancel-suit]")) {
      S.pendingSuit = null;
      render();
      return;
    }
    const oppEl = event.target.closest("[data-opp]");
    if (oppEl) {
      S.setup.opponent = oppEl.dataset.opp;
      render();
      return;
    }
    const seatEl = event.target.closest("[data-seat]");
    if (seatEl) {
      S.setup.seat = Number(seatEl.dataset.seat);
      render();
      return;
    }
    const presetEl = event.target.closest("[data-search-preset]");
    if (presetEl && S.cfg && S.cfg.search) {
      const preset = (S.cfg.search.presets || [])[Number(presetEl.dataset.searchPreset)] || {};
      S.setup.search = { trunc_ply: preset.trunc_ply, rollout_k: preset.rollout_k };
      render();
      return;
    }
    const gamesEl = event.target.closest("[data-games]");
    if (gamesEl) {
      S.setup.games = Number(gamesEl.dataset.games);
      render();
      return;
    }
    const twinPairsEl = event.target.closest("[data-twin-pairs]");
    if (twinPairsEl) {
      S.setup.twinPairs = Number(twinPairsEl.dataset.twinPairs);
      render();
      return;
    }
    const priorEl = event.target.closest("[data-prior]");
    if (priorEl) {
      S.setup.usePrior = priorEl.dataset.prior === "1";
      render();
      return;
    }
    const navEl = event.target.closest("[data-nav]");
    if (navEl) {
      onNav(navEl.dataset.nav);
    }
  });

  app.addEventListener("mouseover", (event) => {
    const playEl = event.target.closest("[data-play]");
    if (!playEl) return;
    const actionId = Number(playEl.dataset.play);
    if (S.hoverAction === actionId) return;
    S.hoverAction = actionId;
    applyHighlight();
  });

  app.addEventListener("mouseout", (event) => {
    const playEl = event.target.closest("[data-play]");
    if (!playEl) return;
    S.hoverAction = null;
    applyHighlight();
  });

  app.addEventListener("input", (event) => {
    if (event.target.id === "seed-input") S.setup.seed = event.target.value;
    if (event.target.id === "twin-seed-input") S.setup.twinSeed = event.target.value;
    if (event.target.id === "twin-pairs-input") S.setup.twinPairs = Number(event.target.value);
    if (S.cfg && S.cfg.search && S.setup.search) {
      if (event.target.id === "search-t" && event.target.value !== "") {
        S.setup.search.trunc_ply = Number(event.target.value);
      }
      if (event.target.id === "search-k" && event.target.value !== "") {
        S.setup.search.rollout_k = Number(event.target.value);
      }
    }
  });

  document.addEventListener("keydown", (event) => {
    const tag = (event.target.tagName || "").toLowerCase();
    const typing = tag === "input" || tag === "textarea" || event.target.isContentEditable;
    const snap = S.snap;
    if (typing || S.screen !== "game" || !snap || snap.phase !== "human" || S.busy) return;
    if (/^[1-9]$/.test(event.key)) {
      const action = (snap.legal || [])[Number(event.key) - 1];
      if (action) {
        event.preventDefault();
        chooseAction(action.action_id);
      }
    } else if (event.key === "p" || event.key === "P" || event.key === "0") {
      const pass = passAction(snap);
      if (pass) {
        event.preventDefault();
        chooseAction(pass.action_id);
      }
    } else if (event.key === "Escape" && S.pendingSuit) {
      S.pendingSuit = null;
      render();
    }
  });
}

function onNav(nav) {
  switch (nav) {
    case "home":
      S.screen = "home";
      render();
      break;
    case "free_setup":
      S.screen = "free_setup";
      render();
      break;
    case "place_setup":
      S.screen = "place_setup";
      render();
      break;
    case "twin_setup":
      S.screen = "twin_setup";
      render();
      break;
    case "start_free":
      startFree();
      break;
    case "start_placement":
      startPlacement();
      break;
    case "start_twin":
      startTwin();
      break;
    case "rematch":
      if (S.lastStart) {
        S.setup.opponent = S.lastStart.opponent_id;
        S.setup.seat = S.lastStart.seat;
        if (S.lastStart.search) S.setup.search = { ...S.lastStart.search };
      }
      startFree();
      break;
    case "continue":
      postControl("/api/continue");
      break;
    case "quit":
      postControl("/api/quit");
      break;
    case "clear-selection":
      S.selected = [];
      render();
      break;
    case "play-selection": {
      const match = matchSelection(S.snap);
      if (match) playAction(match.action.action_id, match.suit);
      break;
    }
    case "toggle-log":
      S.showLogA = !S.showLogA;
      render();
      break;
    case "toggle-counter-table":
      S.counterOpen = !S.counterOpen;
      render();
      break;
    default:
      break;
  }
}

/* -- boot -------------------------------------------------------------------- */

async function boot() {
  bindEvents();
  try {
    S.cfg = await api("/api/config");
  } catch (error) {
    toast("无法连接原型服务器：" + error.message);
  }
  if (S.cfg) {
    if (Number(S.cfg.proto_version || 0) < 2) {
      toast("服务端进程较旧（缺 tricks_completed）：请重启 7g523-web；已启用兼容模式");
    }
    if (S.cfg.twin && Number(S.cfg.proto_version || 0) < 4) {
      S.cfg.twin = null;
      toast("服务端进程较旧（无 twin 支持）：请重启 web_twin.py；已隐藏 twin 模式");
    }
    if (S.cfg.plugin && Number(S.cfg.proto_version || 0) < 6) {
      S.cfg.plugin = null;
      toast("服务端进程较旧（无 v6 插件能力）：请重启 7g523-web；已隐藏插件信息");
    }
    const first = (S.cfg.opponents || [])[0];
    S.setup.opponent = first ? first.id : "random";
    S.setup.usePrior = Boolean(S.cfg.prior);
    if (S.cfg.search) S.setup.search = defaultSearch(S.cfg.search);
    if (S.cfg.twin) S.setup.twinPairs = Number(S.cfg.twin.default_pairs || 30);
  }
  render();
}

boot();
