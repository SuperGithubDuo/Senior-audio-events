"""WAV file -> acoustic events with timestamps and scores."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio import load_wav
from .config import DISCLAIMER, SAMPLE_RATE, frame_start
from .events import Event, run_engine

EVENT_FIELDS = ["label", "onset_s", "offset_s", "trigger_s", "peak_score",
                "mean_score", "n_frames", "start_frame", "end_frame"]


def score_waveform(x: np.ndarray, backend, scorer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, emb = backend.infer(x)
    return scorer.score(emb, scores), emb, scores


def detect_waveform(x: np.ndarray, backend, scorer, params=None) -> tuple[list[Event], np.ndarray]:
    probs, _, _ = score_waveform(x, backend, scorer)
    events = run_engine(probs, scorer.labels, params or scorer.params)
    return events, probs


def detect_file(path: str | Path, backend, scorer, params=None):
    x = load_wav(path)
    events, probs = detect_waveform(x, backend, scorer, params)
    return events, probs, len(x) / SAMPLE_RATE


def export_events(events: list[Event], out_path: str | Path, info: dict) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        doc = {"note": DISCLAIMER, **info, "events": [e.to_dict() for e in events]}
        out.write_text(json.dumps(doc, indent=2))
        return
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        w.writeheader()
        for e in events:
            w.writerow(e.to_dict())


def export_frame_scores(probs: np.ndarray, labels: list[str], out_path: str | Path) -> None:
    with Path(out_path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "start_s", *labels])
        for t, row in enumerate(probs):
            w.writerow([t, round(frame_start(t), 3), *(f"{p:.4f}" for p in row)])


def run_info(path, duration_s, backend, scorer, params) -> dict:
    return {
        "file": str(path),
        "duration_s": round(duration_s, 3),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorer": scorer.kind,
        "yamnet_fingerprint": backend.fingerprint,
        "event_params": {l: p.to_dict() for l, p in params.items()},
        "time_fields": {
            "onset_s/offset_s": "event extent in the file (frame resolution 0.48 s)",
            "trigger_s": "file time at which a live stream would have raised this event",
        },
    }
