"""Causal, per-label event engine shared by offline detection, evaluation and live.

Each label is tracked independently, so several events (e.g. an impact during
a scream) can be active at once. The engine only looks at the current and past
frames, which makes offline results identical to what the live stream would
have produced, including the time at which each event would have been alerted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import (
    DEFAULT_EVENT_PARAMS,
    OFF_RATIO,
    frame_available,
    frame_end,
    frame_start,
)


@dataclass
class EventParams:
    on_threshold: float
    off_threshold: float | None = None
    smooth_frames: int = 1
    min_frames: int = 1
    release_frames: int = 1

    def __post_init__(self):
        if self.off_threshold is None:
            self.off_threshold = self.on_threshold * OFF_RATIO
        if not 0 < self.off_threshold <= self.on_threshold:
            raise ValueError("need 0 < off_threshold <= on_threshold")
        for k in ("smooth_frames", "min_frames", "release_frames"):
            if getattr(self, k) < 1:
                raise ValueError(f"{k} must be >= 1")

    @classmethod
    def for_label(cls, label: str, on_threshold: float, **overrides) -> "EventParams":
        base = dict(DEFAULT_EVENT_PARAMS.get(label, {}))
        base.update(overrides)
        return cls(on_threshold=on_threshold, **base)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Event:
    label: str
    onset_s: float          # start of first active frame
    offset_s: float         # end of last active frame
    trigger_s: float        # when the event was confirmed (audio time; alert time)
    peak_score: float
    mean_score: float
    n_frames: int
    start_frame: int
    end_frame: int

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("onset_s", "offset_s", "trigger_s"):
            d[k] = round(d[k], 3)
        for k in ("peak_score", "mean_score"):
            d[k] = round(d[k], 4)
        return d


@dataclass
class _Track:
    params: EventParams
    label: str
    hist: deque = field(default_factory=deque)
    active: bool = False
    confirmed: bool = False
    start: int = 0
    last_active: int = 0
    below: int = 0
    trigger_s: float = 0.0
    scores: list = field(default_factory=list)

    def _event(self) -> Event:
        s = np.asarray(self.scores)
        return Event(
            label=self.label,
            onset_s=frame_start(self.start),
            offset_s=frame_end(self.last_active),
            trigger_s=self.trigger_s,
            peak_score=float(s.max()),
            mean_score=float(s.mean()),
            n_frames=len(s),
            start_frame=self.start,
            end_frame=self.last_active,
        )

    def update(self, t: int, p: float) -> list[tuple[str, Event]]:
        prm = self.params
        self.hist.append(float(p))
        if len(self.hist) > prm.smooth_frames:
            self.hist.popleft()
        s = sum(self.hist) / len(self.hist)
        out: list[tuple[str, Event]] = []
        if not self.active:
            if s >= prm.on_threshold:
                self.active, self.confirmed = True, False
                self.start = self.last_active = t
                self.below, self.scores = 0, [s]
                if prm.min_frames <= 1:
                    self.confirmed, self.trigger_s = True, frame_available(t)
                    out.append(("start", self._event()))
            return out
        if s >= prm.off_threshold:
            self.last_active, self.below = t, 0
            self.scores.append(s)
            if not self.confirmed and len(self.scores) >= prm.min_frames:
                self.confirmed, self.trigger_s = True, frame_available(t)
                out.append(("start", self._event()))
        else:
            self.below += 1
            if self.below >= prm.release_frames:
                if self.confirmed:
                    out.append(("end", self._event()))
                self.active = False
        return out

    def flush(self) -> list[tuple[str, Event]]:
        out = [("end", self._event())] if self.active and self.confirmed else []
        self.active = False
        return out


class EventEngine:
    """Feed one frame of per-label probabilities at a time."""

    def __init__(self, labels: list[str], params: dict[str, EventParams]):
        self.labels = list(labels)
        self.tracks = [_Track(params[l], l) for l in self.labels]
        self.t = 0

    def update(self, probs: np.ndarray) -> list[tuple[str, Event]]:
        out = []
        for tr, p in zip(self.tracks, probs):
            out.extend(tr.update(self.t, p))
        self.t += 1
        return out

    def flush(self) -> list[tuple[str, Event]]:
        out = []
        for tr in self.tracks:
            out.extend(tr.flush())
        return out


def run_engine(probs: np.ndarray, labels: list[str], params: dict[str, EventParams]) -> list[Event]:
    """Offline helper: probs (T, K) -> completed events sorted by onset."""
    eng = EventEngine(labels, params)
    events = []
    for row in np.atleast_2d(probs):
        events.extend(e for kind, e in eng.update(row) if kind == "end")
    events.extend(e for _, e in eng.flush())
    return sorted(events, key=lambda e: (e.onset_s, e.label))
