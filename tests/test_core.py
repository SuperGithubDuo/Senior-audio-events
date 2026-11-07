"""Fast tests that need no TensorFlow (a numpy fake backend mimics YAMNet framing)."""

import numpy as np
import pytest

from acoustic_events.config import FRAME_HOP, FRAME_SAMPLES, LABELS, frame_available
from acoustic_events.dataset import Recording, frame_labels
from acoustic_events.events import EventParams, run_engine
from acoustic_events.live import StreamProcessor
from acoustic_events.metrics import evaluate, match, poisson_rate_ci, wilson
from acoustic_events.scorers import HeadScorer, stack_context
from acoustic_events.split import leakage, make_split


class FakeBackend:
    """Same padding/framing as YAMNet; each frame depends only on its own window."""
    fingerprint = "fake"
    class_names = ["x"] * 521

    def infer(self, w):
        w = np.asarray(w, np.float32)
        n = max(len(w), FRAME_SAMPLES)
        extra = n - FRAME_SAMPLES
        n += (-extra) % FRAME_HOP
        w = np.pad(w, (0, n - len(w)))
        T = 1 + (n - FRAME_SAMPLES) // FRAME_HOP
        win = np.stack([w[t * FRAME_HOP:t * FRAME_HOP + FRAME_SAMPLES] for t in range(T)])
        rms = np.sqrt((win ** 2).mean(1, keepdims=True))
        emb = np.concatenate([rms, np.abs(win).max(1, keepdims=True), np.zeros((T, 6))], 1)
        return np.zeros((T, 521), np.float32), emb.astype(np.float32)


def tiny_head(context=1):
    D = 8 * (1 + context)
    W = np.zeros((D, 3), np.float32)
    W[0, :] = 40.0                    # louder -> higher prob for every label
    return HeadScorer(LABELS, np.zeros(D), np.ones(D), [(W, np.full(3, -4.0, np.float32))], context,
                      {l: EventParams.for_label(l, 0.5) for l in LABELS})


# --------------------------------------------------------------- event engine
def test_engine_hysteresis_and_trigger_time():
    p = np.array([0.1, 0.6, 0.4, 0.35, 0.1, 0.1, 0.9, 0.1])[:, None]
    ev = run_engine(p, ["impact"], {"impact": EventParams(on_threshold=0.5, off_threshold=0.3)})
    assert [(e.start_frame, e.end_frame) for e in ev] == [(1, 3), (6, 6)]
    assert ev[0].trigger_s == pytest.approx(frame_available(1))


def test_engine_min_frames_delays_trigger_and_drops_blips():
    p = np.array([0.9, 0.1, 0.9, 0.9, 0.9, 0.1])[:, None]
    prm = {"scream": EventParams(on_threshold=0.5, min_frames=2)}
    ev = run_engine(p, ["scream"], prm)
    assert len(ev) == 1 and ev[0].start_frame == 2
    assert ev[0].trigger_s == pytest.approx(frame_available(3))


def test_engine_release_merges_gaps():
    p = np.array([0.9, 0.1, 0.9, 0.1, 0.1])[:, None]
    ev = run_engine(p, ["g"], {"g": EventParams(on_threshold=0.5, release_frames=2)})
    assert len(ev) == 1 and (ev[0].start_frame, ev[0].end_frame) == (0, 2)


def test_engine_multilabel_overlap():
    p = np.array([[0.9, 0.0, 0.9], [0.9, 0.0, 0.1], [0.1, 0.0, 0.1]])
    prm = {l: EventParams(on_threshold=0.5) for l in LABELS}
    ev = run_engine(p, list(LABELS), prm)
    assert sorted(e.label for e in ev) == ["impact", "scream"]


# --------------------------------------------------------------- frame labels
def test_frame_labels_short_impact_and_ignore_band():
    r = Recording("f", None, "r", events=[(1.00, 1.05, "impact"), (5.0, 7.0, "scream")])
    Y, M = frame_labels(20, r)
    k_imp, k_scr = LABELS.index("impact"), LABELS.index("scream")
    assert Y[:, k_imp].sum() >= 1                        # 50 ms impact still labelled
    # frame 9 starts 4.32 s, ends 5.28 s -> overlap 0.28 >= 0.25 -> positive
    assert Y[9, k_scr] == 1
    # frame 8 (3.84-4.80) has no overlap -> clean negative
    assert Y[8, k_scr] == 0 and M[8, k_scr] == 1
    r2 = Recording("f", None, "r", events=[(4.70, 6.0, "scream")])
    Y2, M2 = frame_labels(20, r2)
    assert Y2[8, k_scr] == 0 and M2[8, k_scr] == 0     # 0.10 s overlap -> ignored


# --------------------------------------------------------------- metrics
class E:
    def __init__(self, label, on, off, trig):
        self.label, self.onset_s, self.offset_s, self.trigger_s = label, on, off, trig

    def to_dict(self):
        return {"label": self.label, "onset_s": self.onset_s, "offset_s": self.offset_s}


def test_match_recall_fa_delay():
    m = match([(10.0, 11.0), (30.0, 31.0)], [E("s", 9.6, 11.5, 10.6), E("s", 50, 51, 51), E("s", 10.5, 11.5, 11.0)])
    assert m["found"] == [True, False]
    assert m["delays"][0] == pytest.approx(0.6)
    assert len(m["false_alerts"]) == 1 and m["duplicates"] == 1


def test_evaluate_rates_and_any_view():
    items = [{"file": "a", "duration_s": 1800.0, "gt": {"scream": [(10, 11)], "impact": [(10.2, 10.3)]},
              "det": [E("impact", 9.6, 11.0, 10.5), E("scream", 100, 101, 101)]}]
    rep = evaluate(items, ["scream", "impact"])
    assert rep["per_label"]["scream"]["recall"] == 0
    assert rep["per_label"]["scream"]["false_alerts_per_hour"] == pytest.approx(2.0)
    assert rep["per_label"]["impact"]["recall"] == 1
    assert rep["any_label"]["recall"] == 1 and rep["any_label"]["false_alerts"] == 1


def test_confidence_intervals():
    lo, hi = wilson(9, 10)
    assert lo < 0.9 < hi
    lo, hi = poisson_rate_ci(0, 10.0)
    assert lo == 0 and 0.36 < hi < 0.38               # 3.69 / 10 h


# --------------------------------------------------------------- split
def test_split_separates_speakers_rooms_recordings():
    recs = []
    for room in range(8):
        for take in range(3):
            recs.append(Recording(f"r{room}_{take}.wav", None, f"rec{room}_{take // 2}",
                                  speaker_id=f"s{room}", room_id=f"room{room}",
                                  events=[(1, 2, LABELS[take])]))
        recs.append(Recording(f"r{room}_bg.wav", None, f"bg{room}", room_id=f"room{room}"))
    split_of, rep = make_split(recs, seed=1)
    assert rep["enforced_separation"] == ["recording_id", "speaker_id", "room_id"]
    assert not any(leakage(recs, split_of).values())
    assert set(split_of) == {"train", "val", "test"}


def test_split_relaxes_when_single_room():
    recs = [Recording(f"f{i}.wav", None, f"rec{i}", speaker_id=f"s{i % 8}", room_id="only",
                      events=[(1, 2, LABELS[i % 3])]) for i in range(24)]
    split_of, rep = make_split(recs, seed=0)
    assert "room_id" not in rep["enforced_separation"]
    lk = leakage(recs, split_of)
    assert not lk["recording_id"] and not lk["speaker_id"] and lk["room_id"]
    assert any("room_id" in w for w in rep["warnings"])


# --------------------------------------------------------------- head + live parity
def test_head_save_load_roundtrip(tmp_path):
    h = tiny_head()
    h.meta = {"train_files": []}
    h.save(tmp_path)
    h2 = HeadScorer.load(tmp_path)
    x = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    assert np.allclose(h.score(x), h2.score(x))
    assert h2.params["impact"].on_threshold == 0.5


def test_stack_context_causal():
    e = np.arange(4)[:, None].astype(float)
    s = stack_context(e, 2)
    assert s[:, 0].tolist() == [0, 1, 2, 3]
    assert s[:, 1].tolist() == [0, 0, 1, 2] and s[:, 2].tolist() == [0, 0, 0, 1]


def test_stream_matches_offline():
    from acoustic_events.detect import detect_waveform

    rng = np.random.default_rng(3)
    x = 0.005 * rng.normal(size=16000 * 30).astype(np.float32)
    for s in (5.0, 12.3, 20.0):
        a = int(s * 16000)
        x[a:a + 8000] += 0.5 * rng.normal(size=8000)
    be, head = FakeBackend(), tiny_head()
    off_events, _ = detect_waveform(x, be, head)
    proc = StreamProcessor(be, head)
    live = []
    for i in range(0, len(x), 1234):                 # odd chunk size on purpose
        live += [e for k, e, _ in proc.push(x[i:i + 1234]) if k == "end"]
    live += [e for k, e, _ in proc.finish()]
    key = lambda e: (e.label, e.start_frame, e.end_frame, round(e.trigger_s, 3), round(e.peak_score, 4))
    # offline may add a zero-padded final frame the stream never sees
    off = sorted(key(e) for e in off_events if e.offset_s < 29.0)
    on = sorted(key(e) for e in live if e.offset_s < 29.0)
    assert off == on and len(off) >= 3
