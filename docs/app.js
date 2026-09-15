const D = window.DEMO_DATA;
const SVG_NS = "http://www.w3.org/2000/svg";

const fmt = (x) => x.toFixed(3);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v);
  }
  for (const c of children) if (c != null) el.append(c);
  return el;
}

function s(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

/* ---------- 요약 ---------- */

function renderSummary() {
  const t = D.train;
  document.getElementById("lede").textContent =
    `DeepSpeech2(${D.model.params_m}M 파라미터)를 PyTorch로 직접 구현해 공개 한국어 데이터 ` +
    `${Math.round(t.zeroth.minutes / 6) / 10}시간으로 사전학습하고, 제 목소리 녹음 ${t.my_voice.minutes}분 ` +
    `(${t.my_voice.count}문장)으로 파인튜닝했습니다. 같은 문장을 두 모델이 어떻게 받아쓰는지 들어보고 비교해 보세요.`;

  const tiles = [
    { key: "my_voice", label: "내 목소리 CER", note: `학습에 쓰지 않은 테스트 ${D.sets.my_voice.count}문장` },
    { key: "zeroth", label: "다른 사람 목소리 CER", note: `Zeroth 테스트 ${D.sets.zeroth.count}문장 · 화자 10명` },
  ];
  const root = document.getElementById("summary");
  for (const t of tiles) {
    const m = D.sets[t.key].metrics;
    const a = m.base.cer, b = m.mine.cer;
    const better = b < a;
    const change = Math.round((Math.abs(b - a) / a) * 100);
    root.append(
      h("div", { class: "tile" },
        h("div", { class: "tile-label", text: t.label }),
        h("div", { class: "tile-value" },
          h("span", { class: "from", text: fmt(a) }),
          h("span", { class: "arrow", text: " → " }),
          h("span", { text: fmt(b) })),
        h("div", { class: `tile-delta ${better ? "good" : "bad"}`,
                   text: `${better ? "▼" : "▲"} ${change}% ${better ? "개선" : "악화"} (사전학습 → 파인튜닝)` }),
        h("div", { class: "tile-note", text: t.note })));
  }

  const table = document.getElementById("metrics");
  table.append(h("thead", {}, h("tr", {},
    ...["테스트셋", "사전학습 CER", "파인튜닝 CER", "사전학습 WER", "파인튜닝 WER"].map((x) => h("th", { text: x })))));
  const tbody = h("tbody");
  for (const t of tiles) {
    const m = D.sets[t.key].metrics;
    tbody.append(h("tr", {},
      h("td", { text: t.label.replace(" CER", "") }),
      ...[m.base.cer, m.mine.cer, m.base.wer, m.mine.wer].map((x) => h("td", { text: fmt(x) }))));
  }
  table.append(tbody);
}

/* ---------- 파인튜닝의 가치 ---------- */

function renderValue() {
  const my = D.sets.my_voice.metrics, zeroth = D.sets.zeroth.metrics;
  const change = (a, b) => Math.round((Math.abs(b - a) / a) * 100);
  // 사전학습 대비 파인튜닝으로 가장 많이 좋아진 문장을 예시로 쓴다
  const example = D.sets.my_voice.samples
    .map((s) => ({ s, gain: align(s.ref, s.base).cer - align(s.ref, s.mine).cer }))
    .sort((a, b) => b.gain - a.gain)[0]?.s;

  const points = [
    [
      "효과는 실제로 있었습니다.",
      ` 같은 모델이 녹음 ${D.train.my_voice.minutes}분(${D.train.my_voice.count}문장)만으로 내 목소리 CER이 ` +
      `${fmt(my.base.cer)} → ${fmt(my.mine.cer)}로 ${change(my.base.cer, my.mine.cer)}% 줄었습니다. ` +
      "음성인식에서는 이걸 화자 적응(speaker adaptation)이라고 부르며, 실제로 쓰이는 기법입니다.",
      example && h("span", { class: "example" },
        "예: ", h("q", { text: example.ref }), " → 사전학습 ", h("q", { text: example.base }),
        " / 파인튜닝 ", h("q", { text: example.mine })),
    ],
    ["실제로는 이런 곳에 씁니다.",
      " 발음이 독특하거나 사투리가 있는 사람, 구음장애가 있는 사람의 음성인식, 그리고 회사 용어처럼 특정 분야 단어를 잘 알아듣게 할 때입니다."],
    ["대가도 숫자로 보였습니다.",
      ` 다른 사람 목소리는 ${fmt(zeroth.base.cer)} → ${fmt(zeroth.mine.cer)}로 ${change(zeroth.base.cer, zeroth.mine.cer)}% 나빠졌습니다. ` +
      "적은 데이터로 파인튜닝하면 원래 알던 것을 잊는 망각(catastrophic forgetting)이 그대로 나타난 것이고, 학습용으로는 오히려 좋은 관찰 거리입니다."],
    ["DS2 부품을 한 번에 조립해 본 결과물입니다.",
      " how-ai-works에서 따로 다룬 행 합성곱, CTC 빔 서치, RNN 드롭아웃 같은 DS2 구성 요소가 실제 학습 파이프라인 안에서 어떻게 맞물리는지 직접 돌려 볼 수 있습니다."],
  ];
  document.getElementById("value").append(...points.map(([title, body, extra]) =>
    h("li", {}, h("strong", { text: title }), body, extra ?? null)));
}

/* ---------- 샘플 ---------- */

// 공백을 뺀 글자끼리 편집거리 정렬을 하고, 예측 문장의 어느 글자가 틀렸는지 표시한다.
function align(ref, hyp) {
  const r = [...ref.replace(/ /g, "")];
  const hypChars = [...hyp];
  const hc = [], hi = [];
  hypChars.forEach((c, i) => { if (c !== " ") { hc.push(c); hi.push(i); } });
  const n = r.length, m = hc.length;
  const d = Array.from({ length: n + 1 }, (_, i) => {
    const row = new Array(m + 1).fill(0);
    row[0] = i;
    return row;
  });
  for (let j = 0; j <= m; j++) d[0][j] = j;
  for (let i = 1; i <= n; i++)
    for (let j = 1; j <= m; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] === hc[j - 1] ? 0 : 1));

  const wrong = new Set();
  const deletedAfter = new Map(); // 예측 문장의 글자 index(-1 = 맨 앞) 뒤에 빠진 글자 수
  let i = n, j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + (r[i - 1] === hc[j - 1] ? 0 : 1)) {
      if (r[i - 1] !== hc[j - 1]) wrong.add(hi[j - 1]);
      i--; j--;
    } else if (j > 0 && d[i][j] === d[i][j - 1] + 1) {
      wrong.add(hi[j - 1]);
      j--;
    } else {
      const k = j > 0 ? hi[j - 1] : -1;
      deletedAfter.set(k, (deletedAfter.get(k) || 0) + 1);
      i--;
    }
  }
  return { chars: hypChars, wrong, deletedAfter, cer: d[n][m] / Math.max(n, 1) };
}

function markedText(a) {
  const frag = document.createDocumentFragment();
  const caret = (count) => h("span", { class: "del", title: `빠진 글자 ${count}개`, text: "‸" });
  if (a.deletedAfter.has(-1)) frag.append(caret(a.deletedAfter.get(-1)));
  let run = "";
  const flush = () => { if (run) { frag.append(run); run = ""; } };
  a.chars.forEach((c, i) => {
    if (a.wrong.has(i)) { flush(); frag.append(h("mark", { class: "err", text: c })); }
    else run += c;
    if (a.deletedAfter.has(i)) { flush(); frag.append(caret(a.deletedAfter.get(i))); }
  });
  flush();
  return frag;
}

function sampleCard(sample, index) {
  const dl = h("dl");
  dl.append(h("dt", {}, h("span", { text: "정답" })), h("dd", { class: "ref", text: sample.ref }));
  for (const [key, label] of [["base", "사전학습"], ["mine", "파인튜닝"]]) {
    const hyp = sample[key];
    const a = align(sample.ref, hyp);
    dl.append(h("dt", {}, h("span", { text: label }), h("span", { class: "cer", text: `CER ${a.cer.toFixed(2)}` })));
    dl.append(hyp ? h("dd", {}, markedText(a)) : h("dd", { class: "empty", text: "(인식 결과 없음)" }));
  }
  return h("article", { class: "sample" },
    h("div", { class: "sample-head" },
      h("span", { class: "sample-no", text: `#${index + 1}` }),
      h("audio", { controls: "", preload: "none", src: sample.audio }),
      h("span", { class: "dur", text: `${sample.duration.toFixed(1)}초` })),
    dl);
}

function renderSamples(key) {
  const set = D.sets[key];
  const note = key === "my_voice"
    ? `학습에 쓰지 않은 테스트 문장 ${set.samples.length}개 전부입니다. 사전학습 모델은 제 목소리를 한 번도 들은 적이 없고, 파인튜닝 모델은 다른 문장 ${D.train.my_voice.count}개로 제 목소리에 적응했습니다.`
    : `공개 데이터 테스트셋 앞쪽에서 10초 이하 문장 ${set.samples.length}개를 순서대로 가져왔습니다(골라내지 않음). 파인튜닝 후 다른 사람 목소리가 나빠지는 '망각'을 볼 수 있습니다. 위 수치는 ${set.count}문장 전체 기준입니다.`;
  document.getElementById("samples-note").textContent = note;
  document.getElementById("samples").replaceChildren(...set.samples.map(sampleCard));
}

function setupTabs() {
  const buttons = [...document.querySelectorAll("#tabs button")];
  const select = (btn) => {
    for (const b of buttons) b.setAttribute("aria-selected", String(b === btn));
    document.querySelectorAll("audio").forEach((a) => a.pause());
    renderSamples(btn.dataset.set);
  };
  buttons.forEach((btn, i) => {
    btn.addEventListener("click", () => select(btn));
    btn.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = buttons[(i + (e.key === "ArrowRight" ? 1 : buttons.length - 1)) % buttons.length];
      next.focus();
      select(next);
    });
  });
  renderSamples("my_voice");
}

// 한 번에 오디오 하나만 재생
document.addEventListener("play", (e) => {
  document.querySelectorAll("audio").forEach((a) => { if (a !== e.target) a.pause(); });
}, true);

/* ---------- 학습 곡선 ---------- */

function lineChart(figure, points, { title, sub, labels }) {
  const W = 480, H = 240, m = { t: 20, r: 48, b: 34, l: 44 };
  const xs = points.map((p) => p.epoch);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const yMax = Math.ceil(Math.max(...points.map((p) => p.cer)) * 10) / 10;
  const yStep = yMax <= 0.4 ? 0.05 : 0.2;
  const X = (e) => m.l + ((e - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (v) => m.t + (1 - v / yMax) * (H - m.t - m.b);

  figure.append(h("figcaption", {},
    h("div", { class: "title", text: title }),
    h("div", { class: "sub", text: `${sub} · 가로축 epoch` })));

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", tabindex: "0", role: "img",
                         "aria-label": `${title}. 방향키로 epoch별 값을 볼 수 있습니다.` });

  for (let v = 0; v <= yMax + 1e-9; v += yStep) {
    svg.append(s("line", { class: v === 0 ? "axis" : "grid", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
    const t = s("text", { class: "tick", x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" });
    t.textContent = v.toFixed(yStep < 0.1 ? 2 : 1);
    svg.append(t);
  }
  for (let e = Math.ceil(x0 / 5) * 5; e <= x1; e += 5) {
    const t = s("text", { class: "tick", x: X(e), y: H - m.b + 18, "text-anchor": "middle" });
    t.textContent = e;
    svg.append(t);
  }

  const d = points.map((p, i) => `${i ? "L" : "M"}${X(p.epoch).toFixed(1)},${Y(p.cer).toFixed(1)}`).join("");
  svg.append(s("path", { class: "area", d: `${d}L${X(x1)},${Y(0)}L${X(x0)},${Y(0)}Z` }));
  svg.append(s("path", { class: "line", d }));

  for (const lab of labels) {
    const p = points.find((q) => q.epoch === lab.epoch);
    svg.append(s("circle", { class: "dot", cx: X(p.epoch), cy: Y(p.cer), r: 4 }));
    const t = s("text", { class: "label", x: X(p.epoch) + (lab.dx ?? 0), y: Y(p.cer) + (lab.dy ?? 0),
                          "text-anchor": lab.anchor ?? "middle" });
    t.textContent = lab.text(p);
    svg.append(t);
  }

  // hover / keyboard
  const cross = s("line", { class: "cross", y1: m.t, y2: H - m.b, visibility: "hidden" });
  const hoverDot = s("circle", { class: "dot", r: 5, visibility: "hidden" });
  const hit = s("rect", { class: "hit", x: m.l, y: 0, width: W - m.l - m.r, height: H - m.b });
  svg.append(cross, hoverDot, hit);
  const tip = h("div", { class: "tooltip", hidden: "" });

  let current = -1;
  const show = (i) => {
    current = i;
    const p = points[i];
    const px = X(p.epoch), py = Y(p.cer);
    for (const el of [cross]) { el.setAttribute("x1", px); el.setAttribute("x2", px); el.setAttribute("visibility", "visible"); }
    hoverDot.setAttribute("cx", px);
    hoverDot.setAttribute("cy", py);
    hoverDot.setAttribute("visibility", "visible");
    tip.textContent = `epoch ${p.epoch} · CER ${fmt(p.cer)}`;
    const box = svg.getBoundingClientRect(), fig = figure.getBoundingClientRect();
    tip.style.left = `${box.left - fig.left + (px / W) * box.width}px`;
    tip.style.top = `${box.top - fig.top + (py / H) * box.height}px`;
    tip.hidden = false;
  };
  const hide = () => {
    current = -1;
    cross.setAttribute("visibility", "hidden");
    hoverDot.setAttribute("visibility", "hidden");
    tip.hidden = true;
  };
  svg.addEventListener("pointermove", (e) => {
    const box = svg.getBoundingClientRect();
    const ex = ((e.clientX - box.left) / box.width) * W;
    let best = 0;
    points.forEach((p, i) => { if (Math.abs(X(p.epoch) - ex) < Math.abs(X(points[best].epoch) - ex)) best = i; });
    show(best);
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") show(Math.min(points.length - 1, current + 1));
    else if (e.key === "ArrowLeft") show(current < 0 ? points.length - 1 : Math.max(0, current - 1));
    else return;
    e.preventDefault();
  });

  const table = h("table", {},
    h("thead", {}, h("tr", {}, h("th", { text: "epoch" }), h("th", { text: "valid CER" }))),
    h("tbody", {}, ...points.map((p) => h("tr", {}, h("td", { text: p.epoch }), h("td", { text: fmt(p.cer) })))));
  figure.append(svg, tip, h("details", { class: "table-view" }, h("summary", { text: "표로 보기" }), table));
}

function renderCharts() {
  const base = D.curves.base, mine = D.curves.mine;
  const bestMine = mine.reduce((a, b) => (b.cer < a.cer ? b : a));
  lineChart(document.getElementById("chart-base"), base, {
    title: "사전학습: 다른 사람 목소리 CER",
    sub: `Zeroth-Korean ${D.train.zeroth.count.toLocaleString()}문장, ${base.length} epoch`,
    labels: [
      { epoch: base[0].epoch, text: (p) => fmt(p.cer), anchor: "start", dx: 10, dy: 4 },
      { epoch: base[base.length - 1].epoch, text: (p) => fmt(p.cer), anchor: "end", dy: -12 },
    ],
  });
  lineChart(document.getElementById("chart-mine"), mine, {
    title: "파인튜닝: 내 목소리 CER",
    sub: `내 녹음 ${D.train.my_voice.count}문장, epoch 0 = 파인튜닝 전`,
    labels: [
      { epoch: 0, text: (p) => `파인튜닝 전 ${fmt(p.cer)}`, anchor: "start", dy: -12 },
      { epoch: bestMine.epoch, text: (p) => `최저 ${fmt(p.cer)}`, anchor: "middle", dy: 22 },
    ],
  });
}

/* ---------- 설명 ---------- */

function renderHow() {
  const t = D.train, mine = D.curves.mine;
  const last = mine[mine.length - 1];
  const bestMine = mine.reduce((a, b) => (b.cer < a.cer ? b : a));
  const steps = [
    ["모델", `Conv2d 2층 → 양방향 GRU ${D.model.rnn_layers}층(hidden ${D.model.rnn_hidden}) → CTC. 입력은 20ms 창의 선형 스펙트로그램, 출력은 한글 자모 67개 + 공백.`],
    ["사전학습", `Zeroth-Korean ${t.zeroth.count.toLocaleString()}문장(${Math.round(t.zeroth.minutes / 6) / 10}시간, 화자 105명)으로 ${D.curves.base.length} epoch. RTX 3070에서 약 1시간 40분.`],
    ["녹음", `화면에 뜬 문장을 읽어 ${t.my_voice.count + D.sets.my_voice.count}문장을 녹음하고, ${D.sets.my_voice.count}문장은 테스트용으로 따로 떼어 둠.`],
    ["파인튜닝", `사전학습 모델에서 시작해 conv 층만 고정하고 학습률 5e-5로 ${last.epoch} epoch. 몇 분이면 끝남.`],
    ["디코딩", "프레임별로 가장 확률이 높은 토큰을 고르는 greedy CTC. 언어모델은 쓰지 않음."],
  ];
  document.getElementById("pipeline").append(...steps.map(([k, v]) => h("li", {}, h("strong", { text: k }), ` — ${v}`)));

  const caveats = [
    `파인튜닝 모델은 테스트 ${D.sets.my_voice.count}문장에서 CER이 가장 낮았던 epoch ${bestMine.epoch}을 골랐기 때문에 ${fmt(bestMine.cer)}은 약간 낙관적인 수치입니다. 고르지 않은 마지막 epoch ${last.epoch}도 ${fmt(last.cer)}였습니다.`,
    `테스트가 ${D.sets.my_voice.count}문장뿐이라 문장 몇 개에 따라 수치가 크게 흔들립니다.`,
    "언어모델이 없어서 발음은 맞아도 맞춤법이 틀리는 경우가 많습니다 (예: '공익' → '공릭').",
  ];
  document.getElementById("caveats").append(...caveats.map((c) => h("li", { text: c })));
}

renderSummary();
renderValue();
setupTabs();
renderCharts();
renderHow();
