"""Group-aware train/val/test split.

Files are linked when they share a recording_id, speaker_id or room_id; each
connected group goes entirely into one split, so no original recording,
speaker or room appears on both sides. If that leaves too few groups (e.g. all
data from one room), constraints are relaxed in order room -> speaker, never
recording, and the report says exactly what still leaks.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import LABELS
from .dataset import Recording

SPLITS = ("train", "val", "test")
KEYS = ("recording_id", "speaker_id", "room_id")


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _groups(recs: list[Recording], keys) -> list[list[int]]:
    dsu = _DSU(len(recs))
    for key in keys:
        first: dict[str, int] = {}
        for i, r in enumerate(recs):
            v = getattr(r, key)
            if v:
                if v in first:
                    dsu.union(i, first[v])
                else:
                    first[v] = i
    comp = defaultdict(list)
    for i in range(len(recs)):
        comp[dsu.find(i)].append(i)
    return list(comp.values())


def _weights(recs, durations) -> np.ndarray:
    """Per file: [duration, #events per label] used for balancing."""
    w = np.zeros((len(recs), 1 + len(LABELS)))
    for i, r in enumerate(recs):
        w[i, 0] = durations.get(r.file, 1.0)
        for k, lab in enumerate(LABELS):
            w[i, 1 + k] = len(r.target_events(lab))
    return w


def _assign(groups, w, fractions, rng) -> list[str]:
    total = w.sum(axis=0) + 1e-9
    target = np.outer(fractions, total)            # (3, F)
    have = np.zeros_like(target)
    order = rng.permutation(len(groups))
    order = sorted(order, key=lambda g: -w[groups[g]].sum(axis=0)[0])
    out = [""] * len(w)
    for g in order:
        gw = w[groups[g]].sum(axis=0)
        # choose the split whose normalised deficit benefits most
        gain = [np.sum(np.minimum(gw, np.maximum(target[s] - have[s], 0)) / total) for s in range(3)]
        s = int(np.argmax(gain)) if max(gain) > 0 else int(np.argmin((have / target.clip(1e-9)).mean(1)))
        have[s] += gw
        for i in groups[g]:
            out[i] = SPLITS[s]
    return out


def leakage(recs: list[Recording], split_of: list[str]) -> dict:
    rep = {}
    for key in KEYS:
        where = defaultdict(set)
        for r, s in zip(recs, split_of):
            v = getattr(r, key)
            if v:
                where[v].add(s)
        rep[key] = {v: sorted(s) for v, s in where.items() if len(s) > 1}
    return rep


def make_split(recs: list[Recording], durations: dict[str, float] | None = None,
               fractions=(0.6, 0.2, 0.2), seed: int = 0, min_groups: int = 6) -> tuple[list[str], dict]:
    durations = durations or {}
    rng = np.random.default_rng(seed)
    w = _weights(recs, durations)
    fixed = [r.split for r in recs]
    if any(fixed):
        if not all(s in SPLITS for s in fixed):
            raise ValueError("metadata 'split' column must be train/val/test for every row")
        split_of, used = fixed, ["(fixed by metadata)"]
    else:
        tried = [("recording_id", "speaker_id", "room_id"),
                 ("recording_id", "speaker_id"),
                 ("recording_id",)]
        for keys in tried:
            groups = _groups(recs, keys)
            biggest = max(w[g, 0].sum() for g in groups) / w[:, 0].sum()
            if len(groups) >= min_groups and biggest <= 0.5:
                break
        split_of, used = _assign(groups, w, np.asarray(fractions), rng), list(keys)
    summary = {}
    for s in SPLITS:
        idx = [i for i, x in enumerate(split_of) if x == s]
        summary[s] = {
            "files": len(idx),
            "hours": round(float(w[idx, 0].sum()) / 3600, 3),
            "events": {lab: int(w[idx, 1 + k].sum()) for k, lab in enumerate(LABELS)},
        }
    leaks = leakage(recs, split_of)
    report = {"enforced_separation": used, "splits": summary, "leakage": leaks, "warnings": []}
    if leaks["recording_id"]:
        report["warnings"].append("original recordings span splits: results will be optimistic")
    for key in ("speaker_id", "room_id"):
        if leaks[key]:
            report["warnings"].append(f"{len(leaks[key])} {key}(s) appear in more than one split")
    for s in SPLITS:
        for lab, n in summary[s]["events"].items():
            if n == 0:
                report["warnings"].append(f"{s} split has no '{lab}' events")
    return split_of, report


def write_splits(recs, split_of, path: str | Path) -> None:
    with Path(path).open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "split", "recording_id", "speaker_id", "room_id"])
        for r, s in zip(recs, split_of):
            wr.writerow([r.file, s, r.recording_id, r.speaker_id, r.room_id])
