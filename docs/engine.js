// Pure logic shared by the page and the Node parity test. Mirrors the Python
// package (acoustic_events/config.py, events.py, scorers.py, metrics.py).

export const SAMPLE_RATE = 16000;
export const FRAME_HOP = 7680;          // 0.48 s
export const FRAME_SAMPLES = 15600;     // 0.975 s, one YAMNet patch
export const FRAME_HOP_S = FRAME_HOP / SAMPLE_RATE;
export const FRAME_WIN_S = 0.96;
export const FRAME_AVAILABLE_S = FRAME_SAMPLES / SAMPLE_RATE;
export const OFF_RATIO = 0.6;
export const COLLAR_S = 1.0;
export const LABELS = ["scream", "groan_whimper", "impact"];

export const DEFAULT_EVENT_PARAMS = {
  scream: { smooth_frames: 1, min_frames: 1, release_frames: 1 },
  groan_whimper: { smooth_frames: 2, min_frames: 1, release_frames: 2 },
  impact: { smooth_frames: 1, min_frames: 1, release_frames: 1 },
};
export const BASELINE_CLASS_MAP = {
  scream: ["Screaming"],
  groan_whimper: ["Groan", "Whimper", "Wail, moan"],
  impact: ["Thump, thud", "Bang", "Slam", "Smash, crash", "Breaking", "Shatter"],
};
export const BASELINE_THRESHOLDS = { scream: 0.3, groan_whimper: 0.2, impact: 0.3 };

export const frameStart = (t) => t * FRAME_HOP_S;
export const frameEnd = (t) => t * FRAME_HOP_S + FRAME_WIN_S;
export const frameAvailable = (t) => t * FRAME_HOP_S + FRAME_AVAILABLE_S;

export function makeParams(label, on, overrides = {}) {
  const p = { on_threshold: on, off_threshold: null, ...DEFAULT_EVENT_PARAMS[label], ...overrides };
  if (p.off_threshold == null) p.off_threshold = p.on_threshold * OFF_RATIO;
  return p;
}

// ---------------------------------------------------------------- scorers
export function baselineScorer(classNames) {
  const idx = new Map(classNames.map((n, i) => [n, i]));
  const cols = LABELS.map((l) => BASELINE_CLASS_MAP[l].filter((n) => idx.has(n)).map((n) => idx.get(n)));
  return {
    kind: "baseline",
    labels: [...LABELS],
    contextFrames: 0,
    params: Object.fromEntries(LABELS.map((l) => [l, makeParams(l, BASELINE_THRESHOLDS[l])])),
    // scores: array of Float32Array(521) per frame -> array of [K] per frame
    score: (emb, scores) => scores.map((row) => cols.map((c) => Math.max(...c.map((i) => row[i])))),
  };
}

export function headScorer(doc) {
  if (doc.format !== "acoustic-events-head/1") throw new Error("Not a head file from `export-web`.");
  const layers = doc.layers.map((L) => ({ rows: L.shape[0], cols: L.shape[1], W: Float32Array.from(L.W), b: Float32Array.from(L.b) }));
  const mean = Float32Array.from(doc.mean), scale = Float32Array.from(doc.scale);
  const c = doc.context_frames;
  const params = {};
  for (const l of doc.labels) params[l] = { ...doc.params[l] };
  const forward = (x) => {
    let h = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) h[i] = (x[i] - mean[i]) / scale[i];
    layers.forEach((L, li) => {
      const o = new Float32Array(L.cols);
      for (let j = 0; j < L.cols; j++) o[j] = L.b[j];
      for (let i = 0; i < L.rows; i++) {
        const hi = h[i];
        if (hi === 0) continue;
        const base = i * L.cols;
        for (let j = 0; j < L.cols; j++) o[j] += hi * L.W[base + j];
      }
      if (li < layers.length - 1) for (let j = 0; j < o.length; j++) o[j] = Math.max(0, o[j]);
      h = o;
    });
    return Array.from(h, (z) => 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, z)))));
  };
  return {
    kind: "head",
    labels: [...doc.labels],
    contextFrames: c,
    params,
    info: doc.info || {},
    // emb: array of Float32Array(1024) per frame; causal context, edge-padded
    score: (emb) => emb.map((_, t) => {
      const parts = [];
      for (let lag = 0; lag <= c; lag++) parts.push(emb[Math.max(0, t - lag)]);
      const x = new Float32Array(parts.length * emb[0].length);
      parts.forEach((p, k) => x.set(p, k * p.length));
      return forward(x);
    }),
  };
}

// ---------------------------------------------------------------- event engine
class Track {
  constructor(label, p) { this.label = label; this.p = p; this.hist = []; this.active = false; }
  event() {
    const s = this.scores;
    return {
      label: this.label,
      onset_s: frameStart(this.start), offset_s: frameEnd(this.last),
      trigger_s: this.trigger,
      peak_score: Math.max(...s), mean_score: s.reduce((a, b) => a + b, 0) / s.length,
      n_frames: s.length, start_frame: this.start, end_frame: this.last,
    };
  }
  update(t, prob) {
    const p = this.p, out = [];
    this.hist.push(prob);
    if (this.hist.length > p.smooth_frames) this.hist.shift();
    const s = this.hist.reduce((a, b) => a + b, 0) / this.hist.length;
    if (!this.active) {
      if (s >= p.on_threshold) {
        Object.assign(this, { active: true, confirmed: false, start: t, last: t, below: 0, scores: [s] });
        if (p.min_frames <= 1) { this.confirmed = true; this.trigger = frameAvailable(t); out.push(["start", this.event()]); }
      }
      return out;
    }
    if (s >= p.off_threshold) {
      this.last = t; this.below = 0; this.scores.push(s);
      if (!this.confirmed && this.scores.length >= p.min_frames) {
        this.confirmed = true; this.trigger = frameAvailable(t); out.push(["start", this.event()]);
      }
    } else if (++this.below >= p.release_frames) {
      if (this.confirmed) out.push(["end", this.event()]);
      this.active = false;
    }
    return out;
  }
  flush() { const o = this.active && this.confirmed ? [["end", this.event()]] : []; this.active = false; return o; }
}

export class EventEngine {
  constructor(labels, params) { this.tracks = labels.map((l) => new Track(l, params[l])); this.t = 0; }
  update(probs) { const out = this.tracks.flatMap((tr, k) => tr.update(this.t, probs[k])); this.t++; return out; }
  flush() { return this.tracks.flatMap((tr) => tr.flush()); }
}

export function runEngine(probs, labels, params) {
  const eng = new EventEngine(labels, params), ev = [];
  for (const row of probs) for (const [k, e] of eng.update(row)) if (k === "end") ev.push(e);
  for (const [, e] of eng.flush()) ev.push(e);
  return ev.sort((a, b) => a.onset_s - b.onset_s || a.label.localeCompare(b.label));
}

// ---------------------------------------------------------------- ground-truth check
export function matchEvents(gt, dets, collar = COLLAR_S) {
  // gt: [{label, onset, offset}], dets: events. Per label, same rule as Python metrics.
  const res = { found: [], falseAlerts: [], delays: [] };
  const used = new Set();
  for (const g of gt) {
    let best = null;
    dets.forEach((d, i) => {
      if (d.label === g.label && d.onset_s < g.offset + collar && d.offset_s > g.onset - collar) {
        used.add(i);
        const dl = d.trigger_s - g.onset;
        if (best === null || dl < best) best = dl;
      }
    });
    res.found.push(best !== null);
    res.delays.push(best);
  }
  dets.forEach((d, i) => { if (!used.has(i)) res.falseAlerts.push(d); });
  return res;
}

// ---------------------------------------------------------------- small CSV helpers
export function parseCSV(text) {
  const rows = [];
  let row = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') q = false;
      else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); cell = "";
      if (row.some((c) => c !== "")) rows.push(row);
      row = [];
    } else cell += ch;
  }
  row.push(cell);
  if (row.some((c) => c !== "")) rows.push(row);
  const head = rows.shift().map((h) => h.trim());
  return rows.map((r) => Object.fromEntries(head.map((h, i) => [h, (r[i] ?? "").trim()])));
}

export function toCSV(rows, fields) {
  const esc = (v) => { const s = v == null ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return [fields.join(","), ...rows.map((r) => fields.map((f) => esc(r[f])).join(","))].join("\n") + "\n";
}

export const EVENT_FIELDS = ["label", "onset_s", "offset_s", "trigger_s", "peak_score", "mean_score", "n_frames", "start_frame", "end_frame"];

export function roundEvent(e) {
  const r3 = (v) => Math.round(v * 1000) / 1000, r4 = (v) => Math.round(v * 10000) / 10000;
  return { ...e, onset_s: r3(e.onset_s), offset_s: r3(e.offset_s), trigger_s: r3(e.trigger_s), peak_score: r4(e.peak_score), mean_score: r4(e.mean_score) };
}
