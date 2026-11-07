// Browser test page: YAMNet (TensorFlow.js) + exported head + the same event engine as Python.
// No audio is uploaded anywhere; only model files are fetched.
import * as E from "./engine.js";

const $ = (id) => document.getElementById(id);
const HUB_URL = "https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1";
const NAMES = { scream: "Scream", groan_whimper: "Groan / whimper", impact: "Impact" };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COLOR = (l) => css(`--${l}`) || "#17324d";

const S = {
  yamnet: null, yamnetSource: "", classNames: [],
  baseline: null, head: null, scorer: null, params: {}, baseParams: {},
  src: null,            // {name, duration, emb[], scores[], probs[], live}
  events: [], gtRows: null,
  span: 0, viewStart: 0,
  live: null,
};

// ------------------------------------------------------------------ status
function setStatus(state, text) { const s = $("status"); s.dataset.state = state; s.textContent = text; }
function meta(id, text, err = false) { const m = $(id); m.textContent = text; m.classList.toggle("err", err); }

// ------------------------------------------------------------------ YAMNet
async function yamnetInfer(samples) {
  const outs = tf.tidy(() => {
    const r = S.yamnet.predict(tf.tensor1d(samples));
    return Array.isArray(r) ? r : [r];
  });
  let sc = null, em = null;
  for (const t of outs) {
    const last = t.shape[t.shape.length - 1];
    if (t.shape.length === 2 && last === 521) sc = t;
    else if (t.shape.length === 2 && last === 1024) em = t;
  }
  if (!sc || !em) { outs.forEach((t) => t.dispose()); throw new Error("These model files don't produce YAMNet outputs (521 scores and 1024-value embeddings per frame)."); }
  const [s, e] = await Promise.all([sc.data(), em.data()]);
  const T = sc.shape[0];
  outs.forEach((t) => t.dispose());
  const scores = [], emb = [];
  for (let t = 0; t < T; t++) { scores.push(s.subarray(t * 521, (t + 1) * 521)); emb.push(e.subarray(t * 1024, (t + 1) * 1024)); }
  return { scores, emb };
}

async function useYamnet(model, source) {
  S.yamnet = model;
  await yamnetInfer(new Float32Array(E.FRAME_SAMPLES));   // warm up + check outputs
  S.yamnetSource = source;
  setStatus("ready", `YAMNet ready (${tf.getBackend()})`);
  meta("yamnetMeta", `Loaded from ${source}.`);
  $("micBtn").disabled = false;
}

async function initYamnet() {
  setStatus("busy", "Loading YAMNet…");
  const sources = [["this site", "./models/yamnet/model.json", {}], ["TensorFlow Hub", HUB_URL, { fromTFHub: true }]];
  let lastErr = null;
  for (const [name, url, opts] of sources) {
    try { await useYamnet(await tf.loadGraphModel(url, opts), name); return; }
    catch (e) { lastErr = e; }
  }
  setStatus("error", "YAMNet not loaded");
  meta("yamnetMeta", `Couldn't download YAMNet (${lastErr?.message || "network error"}). Use local model files below.`, true);
}

$("yamnetFiles").addEventListener("change", async (ev) => {
  const files = [...ev.target.files];
  const json = files.find((f) => f.name.endsWith(".json"));
  if (!json) { meta("yamnetMeta", "Include model.json in the selection.", true); return; }
  setStatus("busy", "Loading YAMNet…");
  try {
    const bins = files.filter((f) => f !== json);
    await useYamnet(await tf.loadGraphModel(tf.io.browserFiles([json, ...bins])), "your files");
    if (S.src && !S.src.live && S.src.samples) await analyseSamples();
  } catch (e) {
    setStatus("error", "YAMNet not loaded");
    meta("yamnetMeta", `Couldn't load those files: ${e.message}`, true);
  }
});

// ------------------------------------------------------------------ classifier
function useScorer(sc) {
  S.scorer = sc;
  S.baseParams = JSON.parse(JSON.stringify(sc.params));
  S.params = JSON.parse(JSON.stringify(sc.params));
  buildSliders();
  if (S.src && S.src.emb?.length) { rescore(); }
}

function buildSliders() {
  const box = $("sliders");
  box.innerHTML = "";
  for (const l of S.scorer.labels) {
    const id = `thr-${l}`;
    const wrap = document.createElement("div");
    wrap.className = "slider";
    wrap.innerHTML = `<label for="${id}"><span class="swatch" style="background:${COLOR(l)}"></span>${NAMES[l] || l}</label>
      <output for="${id}" id="${id}-out"></output>
      <input id="${id}" type="range" min="0.02" max="0.98" step="0.01">`;
    box.appendChild(wrap);
    const input = wrap.querySelector("input");
    input.value = S.params[l].on_threshold;
    wrap.querySelector("output").textContent = (+input.value).toFixed(2);
    input.addEventListener("input", () => {
      const base = S.baseParams[l], v = +input.value;
      S.params[l].on_threshold = v;
      S.params[l].off_threshold = v * (base.off_threshold / base.on_threshold);
      wrap.querySelector("output").textContent = v.toFixed(2);
      recompute();
    });
  }
}

$("resetThr").addEventListener("click", () => { useScorer(S.scorer); recompute(); });

document.querySelectorAll('input[name="scorer"]').forEach((r) => r.addEventListener("change", () => {
  if (r.checked) useScorer(r.value === "head" ? S.head : S.baseline);
}));

$("headFile").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  try {
    S.head = E.headScorer(JSON.parse(await f.text()));
    const i = S.head.info, parts = [`${f.name}: ${i.head_kind || "head"}`];
    if (i.train_files) parts.push(`trained on ${i.train_files} files`);
    if (i.target_fa_per_hour != null) parts.push(`thresholds tuned for at most ${i.target_fa_per_hour} false alerts per hour`);
    meta("headMeta", parts.join(", ") + ".");
    $("headRadio").disabled = false;
    $("headRadio").checked = true;
    useScorer(S.head);
  } catch (e) {
    meta("headMeta", `Couldn't read that file: ${e.message}`, true);
  }
});

// ------------------------------------------------------------------ audio files
async function decode16k(buf) {
  try {
    const ab = await new OfflineAudioContext(1, 1, E.SAMPLE_RATE).decodeAudioData(buf.slice(0));
    return mono(ab);
  } catch {
    const ac = new AudioContext();
    const ab = await ac.decodeAudioData(buf.slice(0));
    ac.close();
    const off = new OfflineAudioContext(1, Math.ceil(ab.duration * E.SAMPLE_RATE), E.SAMPLE_RATE);
    const node = off.createBufferSource();
    node.buffer = ab; node.connect(off.destination); node.start();
    return (await off.startRendering()).getChannelData(0).slice();
  }
}

function mono(ab) {
  const n = ab.length, out = new Float32Array(n);
  for (let c = 0; c < ab.numberOfChannels; c++) {
    const d = ab.getChannelData(c);
    for (let i = 0; i < n; i++) out[i] += d[i] / ab.numberOfChannels;
  }
  for (let i = 0; i < n; i++) out[i] = Math.max(-1, Math.min(1, out[i]));
  return out;
}

async function analyseFile(file) {
  stopMic();
  meta("audioMeta", `Reading ${file.name}…`);
  let samples;
  try { samples = await decode16k(await file.arrayBuffer()); }
  catch { meta("audioMeta", `This browser can't decode ${file.name}. Try a WAV file.`, true); return; }
  const player = $("player");
  if (player.src) URL.revokeObjectURL(player.src);
  player.src = URL.createObjectURL(file);
  S.src = { name: file.name, duration: samples.length / E.SAMPLE_RATE, samples, emb: [], scores: [], probs: [], live: false };
  S.viewStart = 0;
  $("empty").hidden = true;
  await analyseSamples();
}

async function analyseSamples() {
  const src = S.src;
  if (!S.yamnet) { meta("audioMeta", "Waiting for YAMNet to load. Results will appear once it is ready.", true); draw(); return; }
  const x = src.samples, n = x.length;
  const total = n <= E.FRAME_SAMPLES ? 1 : 1 + Math.ceil((n - E.FRAME_SAMPLES) / E.FRAME_HOP);
  const CH = 64, bar = $("progress");
  bar.style.display = "block";
  src.emb = []; src.scores = [];
  const t0 = performance.now();
  setStatus("busy", "Analysing…");
  for (let f0 = 0; f0 < total; f0 += CH) {
    const a = f0 * E.FRAME_HOP, b = Math.min(n, a + (CH - 1) * E.FRAME_HOP + E.FRAME_SAMPLES);
    const r = await yamnetInfer(x.subarray(a, b));
    src.emb.push(...r.emb); src.scores.push(...r.scores);
    bar.firstElementChild.style.width = `${Math.min(100, ((f0 + CH) / total) * 100)}%`;
    meta("audioMeta", `Analysing ${src.name}: ${Math.min(total, f0 + CH)} of ${total} frames`);
    await tf.nextFrame();
    if (S.src !== src) return;   // another file was chosen meanwhile
  }
  bar.style.display = "none";
  setStatus("ready", `YAMNet ready (${tf.getBackend()})`);
  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  meta("audioMeta", `${src.name}: ${fmtTime(src.duration)} analysed in ${secs} s.`);
  rescore();
}

function rescore() {
  const src = S.src;
  if (!src || src.live) return;
  src.probs = S.scorer.score(src.emb, src.scores);
  recompute();
}

function recompute() {
  if (!S.src) return;
  S.events = E.runEngine(S.src.probs, S.scorer.labels, S.params);
  renderEvents();
  renderGT();
  draw();
  ["dlCsv", "dlJson", "dlFrames"].forEach((id) => ($(id).disabled = !S.src.probs.length));
}

$("audioFile").addEventListener("change", (ev) => ev.target.files[0] && analyseFile(ev.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) analyseFile(f); });

// ------------------------------------------------------------------ microphone
const TAP = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const c = inputs[0] && inputs[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; }
}
registerProcessor("tap", Tap);`;

function makeResampler(inRate) {
  // windowed-sinc low-pass, then linear interpolation down to 16 kHz (streaming)
  const N = 63, fc = 7200 / inRate, h = new Float32Array(N);
  let sum = 0;
  for (let k = 0; k < N; k++) {
    const m = k - (N - 1) / 2;
    const sinc = m === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * m) / (Math.PI * m);
    const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * k) / (N - 1)) + 0.08 * Math.cos((4 * Math.PI * k) / (N - 1));
    h[k] = sinc * w; sum += h[k];
  }
  for (let k = 0; k < N; k++) h[k] /= sum;
  const step = inRate / E.SAMPLE_RATE;
  let hist = new Float32Array(N - 1), prev = 0, p = 0;
  return (x) => {
    const ext = new Float32Array(hist.length + x.length);
    ext.set(hist); ext.set(x, hist.length);
    const z = new Float32Array(x.length + 1);
    z[0] = prev;
    for (let i = 0; i < x.length; i++) {
      let acc = 0;
      for (let k = 0; k < N; k++) acc += h[k] * ext[i + N - 1 - k];
      z[i + 1] = acc;
    }
    hist = ext.slice(ext.length - (N - 1));
    const out = [];
    const L = z.length;
    while (p < L - 1) { const i = Math.floor(p), f = p - i; out.push(z[i] * (1 - f) + z[i + 1] * f); p += step; }
    p -= L - 1; prev = z[L - 1];
    return Float32Array.from(out);
  };
}

async function startMic() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) { meta("audioMeta", `Microphone not available: ${e.message}. Check the browser's permission for this site.`, true); return; }
  let ctx, source, resample = null;
  try { ctx = new AudioContext({ sampleRate: E.SAMPLE_RATE }); source = ctx.createMediaStreamSource(stream); }
  catch { ctx?.close(); ctx = new AudioContext(); source = ctx.createMediaStreamSource(stream); }
  if (ctx.sampleRate !== E.SAMPLE_RATE) resample = makeResampler(ctx.sampleRate);
  const url = URL.createObjectURL(new Blob([TAP], { type: "application/javascript" }));
  await ctx.audioWorklet.addModule(url);
  const node = new AudioWorkletNode(ctx, "tap");
  const mute = ctx.createGain(); mute.gain.value = 0;
  source.connect(node); node.connect(mute); mute.connect(ctx.destination);
  const player = $("player");
  player.pause(); player.removeAttribute("src"); player.load();
  S.src = { name: "Microphone", duration: 0, emb: [], scores: [], probs: [], live: true };
  S.live = { ctx, stream, node, pending: [], buf: new Float32Array(0), consumed: 0, next: 0, busy: false, embHist: [], lastMs: 0 };
  node.port.onmessage = (e) => { S.live.pending.push(resample ? resample(e.data) : e.data); pump(); };
  $("micBtn").textContent = "Stop microphone"; $("micBtn").classList.add("recording");
  $("empty").hidden = true;
  document.querySelectorAll('input[name="scorer"]').forEach((r) => (r.disabled = true));
  setSpan(60);
  meta("audioMeta", `Listening at ${ctx.sampleRate} Hz${resample ? ", resampled to 16 kHz" : ""}. Nothing is recorded or sent.`);
}

async function pump() {
  const L = S.live;
  if (!L || L.busy) return;
  L.busy = true;
  try {
    if (L.pending.length) {
      const add = L.pending.reduce((a, c) => a + c.length, 0);
      const nb = new Float32Array(L.buf.length + add);
      nb.set(L.buf); let o = L.buf.length;
      for (const c of L.pending) { nb.set(c, o); o += c.length; }
      L.buf = nb; L.pending = [];
    }
    while (S.live === L && L.consumed + L.buf.length >= L.next + E.FRAME_SAMPLES) {
      const a = L.next - L.consumed, t0 = performance.now();
      const r = await yamnetInfer(L.buf.slice(a, a + E.FRAME_SAMPLES));
      L.embHist.push(r.emb[0]);
      while (L.embHist.length > S.scorer.contextFrames + 1) L.embHist.shift();
      const hist = [...L.embHist];
      while (hist.length < S.scorer.contextFrames + 1) hist.unshift(hist[0]);
      const row = S.scorer.score(hist, hist.map(() => r.scores[0])).at(-1);
      L.lastMs = performance.now() - t0;
      S.src.probs.push(row);
      L.next += E.FRAME_HOP;
      const drop = Math.min(L.buf.length, L.next - L.consumed);
      L.buf = L.buf.slice(drop); L.consumed += drop;
      S.src.duration = E.frameAvailable(S.src.probs.length - 1);
      if (S.span) S.viewStart = Math.max(0, S.src.duration - S.span);
      recompute();
      meta("audioMeta", `Listening: ${fmtTime(S.src.duration)}, last frame took ${L.lastMs.toFixed(0)} ms. Nothing is recorded or sent.`);
    }
  } finally { L.busy = false; }
  if (S.live === L && L.pending.length) pump();
}

function stopMic() {
  const L = S.live;
  if (!L) return;
  S.live = null;
  L.node.port.onmessage = null;
  L.stream.getTracks().forEach((t) => t.stop());
  L.ctx.close();
  $("micBtn").textContent = "Start microphone"; $("micBtn").classList.remove("recording");
  document.querySelectorAll('input[name="scorer"]').forEach((r) => (r.disabled = r.value === "head" && !S.head));
  if (S.src) meta("audioMeta", `Microphone stopped after ${fmtTime(S.src.duration)}.`);
}

$("micBtn").addEventListener("click", () => (S.live ? stopMic() : startMic()));

// ------------------------------------------------------------------ annotations
$("gtFile").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  try {
    const rows = E.parseCSV(await f.text());
    if (!rows.length || !("onset" in rows[0]) || !("label" in rows[0])) throw new Error("expected columns file,onset,offset,label");
    S.gtRows = rows.map((r) => ({ file: r.file || "", onset: +r.onset, offset: +r.offset, label: r.label }));
    meta("gtMeta", `${S.gtRows.length} annotated events loaded from ${f.name}.`);
    renderGT(); draw();
  } catch (e) { meta("gtMeta", `Couldn't read that CSV: ${e.message}`, true); }
});

const base = (p) => p.split(/[\\/]/).pop();
function currentGT() {
  if (!S.gtRows || !S.src || S.src.live) return null;
  const rows = S.gtRows.filter((r) => !r.file || base(r.file) === S.src.name);
  return rows.filter((r) => E.LABELS.includes(r.label));
}

function renderGT() {
  const box = $("gtSummary"), gt = currentGT();
  if (!gt) return;
  if (!gt.length && !S.gtRows.some((r) => base(r.file) === S.src.name)) {
    box.innerHTML = `<h3>Against your annotations</h3><p class="meta">No rows for “${esc(S.src.name)}” in the annotations file. The file column must match this file's name.</p>`;
    return;
  }
  const m = E.matchEvents(gt, S.events);
  let html = `<h3>Against your annotations</h3><dl>`;
  for (const l of S.scorer.labels) {
    const idx = gt.map((g, i) => i).filter((i) => gt[i].label === l);
    const found = idx.filter((i) => m.found[i]);
    const delays = found.map((i) => m.delays[i]).sort((a, b) => a - b);
    const fa = m.falseAlerts.filter((d) => d.label === l).length;
    const med = delays.length ? delays[Math.floor((delays.length - 1) / 2)].toFixed(2) + " s" : "–";
    html += `<dt><span class="tag"><span class="swatch" style="background:${COLOR(l)}"></span>${NAMES[l]}</span></dt>
      <dd>${idx.length ? `${found.length} of ${idx.length} found` : "none annotated"}, ${fa} false alert${fa === 1 ? "" : "s"}, raised after ${med}</dd>`;
  }
  html += `</dl><p class="meta">Short files give rough numbers; use the Python <code>evaluate</code> command for false alerts per hour on held-out recordings.</p>`;
  box.innerHTML = html;
}

// ------------------------------------------------------------------ events table
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
function fmtTime(t) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return m ? `${m}:${s.toFixed(1).padStart(4, "0")}` : `${s.toFixed(1)} s`;
}

function renderEvents() {
  const tb = $("events");
  $("eventsTitle").textContent = `Events (${S.events.length})`;
  if (!S.events.length) {
    tb.innerHTML = `<tr><td colspan="5" class="meta">${S.src?.probs.length ? "No events above the current thresholds." : "Events appear here once a sound is analysed."}</td></tr>`;
    return;
  }
  const list = S.src.live ? [...S.events].reverse() : S.events;
  tb.innerHTML = list.map((e, i) => `<tr tabindex="0" data-t="${e.onset_s}">
    <td><span class="tag"><span class="swatch" style="background:${COLOR(e.label)}"></span>${NAMES[e.label] || e.label}</span></td>
    <td class="num">${fmtTime(e.onset_s)}</td><td class="num">${fmtTime(e.offset_s)}</td>
    <td class="num">${fmtTime(e.trigger_s)}</td><td class="num">${e.peak_score.toFixed(2)}</td></tr>`).join("");
}

$("events").addEventListener("click", (ev) => seekRow(ev.target.closest("tr")));
$("events").addEventListener("keydown", (ev) => { if (ev.key === "Enter") seekRow(ev.target.closest("tr")); });
function seekRow(tr) {
  if (!tr || tr.dataset.t == null || !S.src || S.src.live) return;
  const t = +tr.dataset.t;
  if (S.span) S.viewStart = Math.max(0, Math.min(S.src.duration - S.span, t - S.span * 0.25));
  const p = $("player");
  if (p.src) p.currentTime = Math.max(0, t - 0.5);
  syncPan(); draw();
}

// ------------------------------------------------------------------ downloads
function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
const stem = () => (S.src?.name || "audio").replace(/\.[^.]+$/, "");
$("dlCsv").addEventListener("click", () => download(`${stem()}_events.csv`, E.toCSV(S.events.map(E.roundEvent), E.EVENT_FIELDS), "text/csv"));
$("dlJson").addEventListener("click", () => download(`${stem()}_events.json`, JSON.stringify({
  note: "Outputs are acoustic event detections (scream-like, groan/whimper-like, impact-like sounds). They are not confirmed emergencies or falls.",
  file: S.src.name, duration_s: +S.src.duration.toFixed(3), scorer: S.scorer.kind, yamnet_source: S.yamnetSource,
  created_utc: new Date().toISOString(), event_params: S.params, events: S.events.map(E.roundEvent),
}, null, 2), "application/json"));
$("dlFrames").addEventListener("click", () => {
  const labels = S.scorer.labels;
  const rows = S.src.probs.map((p, t) => ({ frame: t, start_s: E.frameStart(t).toFixed(3), ...Object.fromEntries(labels.map((l, k) => [l, p[k].toFixed(4)])) }));
  download(`${stem()}_frames.csv`, E.toCSV(rows, ["frame", "start_s", ...labels]), "text/csv");
});

// ------------------------------------------------------------------ timeline
function viewRange() {
  const d = Math.max(S.src?.duration || 0, 1);
  if (!S.span || S.span >= d) return [0, S.src?.live ? Math.max(d, S.span || 60) : d];
  const s = Math.max(0, Math.min(S.viewStart, d - S.span));
  return [s, s + S.span];
}

function setSpan(span) {
  S.span = span;
  document.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.span === span)));
  syncPan(); draw();
}
document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => setSpan(+b.dataset.span)));

function syncPan() {
  const pan = $("pan"), d = S.src?.duration || 0;
  const show = S.span && d > S.span && !S.src?.live;
  pan.hidden = !show;
  if (show) pan.value = String(S.viewStart / (d - S.span));
}
$("pan").addEventListener("input", (e) => { S.viewStart = +e.target.value * (S.src.duration - S.span); draw(); });

function niceStep(span, px) {
  const target = span / Math.max(1, px / 90);
  return [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600].find((s) => s >= target) || 3600;
}

function draw() {
  const cv = $("chart"), dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = 330;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.height = `${H}px`;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = css("--sheet"); g.fillRect(0, 0, W, H);
  if (!S.src) return;
  const labels = S.scorer.labels, [v0, v1] = viewRange();
  const X = (t) => ((t - v0) / (v1 - v0)) * W;
  const laneH = 96, axisY = labels.length * laneH;
  const step = niceStep(v1 - v0, W);
  const grid = css("--grid"), strong = css("--grid-strong"), ink = css("--ink"), soft = css("--ink-soft");

  // time grid
  g.lineWidth = 1;
  for (let t = Math.ceil(v0 / step) * step; t <= v1; t += step) {
    g.strokeStyle = grid; g.beginPath(); g.moveTo(Math.round(X(t)) + 0.5, 0); g.lineTo(Math.round(X(t)) + 0.5, axisY); g.stroke();
    g.fillStyle = soft; g.font = "12px 'Atkinson Hyperlegible', sans-serif"; g.textAlign = "center";
    g.fillText(fmtTime(t), Math.min(W - 24, Math.max(24, X(t))), axisY + 18);
  }
  const gt = currentGT() || [];
  const probs = S.src.probs;
  const f0 = Math.max(0, Math.floor(v0 / E.FRAME_HOP_S) - 2), f1 = Math.min(probs.length, Math.ceil(v1 / E.FRAME_HOP_S) + 2);

  labels.forEach((l, k) => {
    const y0 = k * laneH, top = y0 + 30, bot = y0 + laneH - 16, Y = (p) => bot - p * (bot - top);
    const col = COLOR(l);
    g.strokeStyle = grid;
    for (const p of [0.25, 0.5, 0.75, 1]) { g.beginPath(); g.moveTo(0, Math.round(Y(p)) + 0.5); g.lineTo(W, Math.round(Y(p)) + 0.5); g.stroke(); }
    g.strokeStyle = strong; g.beginPath(); g.moveTo(0, y0 + laneH - 0.5); g.lineTo(W, y0 + laneH - 0.5); g.stroke();

    // probability curve (frame centres)
    if (f1 > f0) {
      g.beginPath();
      for (let t = f0; t < f1; t++) { const x = X(E.frameStart(t) + E.FRAME_WIN_S / 2), y = Y(probs[t][k]); t === f0 ? g.moveTo(x, y) : g.lineTo(x, y); }
      g.strokeStyle = col; g.lineWidth = 1.75; g.stroke();
      g.lineTo(X(E.frameStart(f1 - 1) + E.FRAME_WIN_S / 2), bot); g.lineTo(X(E.frameStart(f0) + E.FRAME_WIN_S / 2), bot); g.closePath();
      g.globalAlpha = 0.13; g.fillStyle = col; g.fill(); g.globalAlpha = 1;
    }
    // threshold
    g.setLineDash([5, 4]); g.strokeStyle = col; g.lineWidth = 1.25;
    g.beginPath(); g.moveTo(0, Y(S.params[l].on_threshold)); g.lineTo(W, Y(S.params[l].on_threshold)); g.stroke(); g.setLineDash([]);
    // ground truth (hatched)
    for (const e of gt.filter((e) => e.label === l)) {
      const a = X(e.onset), b = Math.max(a + 3, X(e.offset));
      if (b < 0 || a > W) continue;
      g.save(); g.beginPath(); g.rect(a, bot + 4, b - a, 8); g.clip();
      g.strokeStyle = ink; g.lineWidth = 1.5;
      for (let x = a - 12; x < b + 12; x += 5) { g.beginPath(); g.moveTo(x, bot + 14); g.lineTo(x + 10, bot + 2); g.stroke(); }
      g.restore(); g.strokeStyle = ink; g.strokeRect(a + 0.5, bot + 4.5, b - a - 1, 7);
    }
    // detections + raise markers
    for (const e of S.events.filter((e) => e.label === l)) {
      const a = X(e.onset_s), b = Math.max(a + 3, X(e.offset_s));
      if (b < 0 || a > W) continue;
      g.fillStyle = col; g.fillRect(a, y0 + 14, b - a, 9);
      const tx = X(e.trigger_s);
      g.fillStyle = ink; g.beginPath(); g.moveTo(tx - 6, y0 + 3); g.lineTo(tx + 6, y0 + 3); g.lineTo(tx, y0 + 12); g.closePath(); g.fill();
    }
    // lane label
    g.font = "bold 13px 'Atkinson Hyperlegible', sans-serif"; g.textAlign = "left";
    const name = NAMES[l] || l, tw = g.measureText(name).width;
    g.fillStyle = "rgba(255,255,255,0.88)"; g.fillRect(6, top + 2, tw + 26, 20);
    g.fillStyle = col; g.fillRect(12, top + 8, 9, 9);
    g.fillStyle = ink; g.fillText(name, 26, top + 17);
  });

  // playhead
  const p = $("player");
  if (!S.src.live && p.src && p.currentTime > 0) {
    const x = X(p.currentTime);
    g.strokeStyle = ink; g.lineWidth = 2; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, axisY); g.stroke();
  }
}

$("chart").addEventListener("click", (ev) => {
  if (!S.src || S.src.live) return;
  const [v0, v1] = viewRange(), r = ev.currentTarget.getBoundingClientRect();
  const t = v0 + ((ev.clientX - r.left) / r.width) * (v1 - v0);
  const p = $("player");
  if (p.src) p.currentTime = Math.max(0, Math.min(S.src.duration, t));
  draw();
});

let raf = 0;
function follow() {
  const p = $("player");
  if (S.span && S.src && !S.src.live) {
    const [v0, v1] = viewRange();
    if (p.currentTime > v1 || p.currentTime < v0) { S.viewStart = Math.max(0, p.currentTime - S.span * 0.1); syncPan(); }
  }
  draw();
  if (!p.paused) raf = requestAnimationFrame(follow);
}
$("player").addEventListener("play", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(follow); });
$("player").addEventListener("seeked", draw);
new ResizeObserver(() => draw()).observe($("chartbox"));

// ------------------------------------------------------------------ start
async function init() {
  try {
    const rows = E.parseCSV(await (await fetch("./yamnet_class_map.csv")).text());
    S.classNames = rows.map((r) => r.display_name);
  } catch { S.classNames = []; }
  S.baseline = E.baselineScorer(S.classNames);
  useScorer(S.baseline);
  draw();
  if (window.__AE_SKIP_AUTOLOAD__) { setStatus("busy", "Waiting for model"); return; }
  await initYamnet();
}

// hook for automated tests: inject any object with a tf.js-style predict()
window.acousticEvents = { useYamnet, state: S };
init();
