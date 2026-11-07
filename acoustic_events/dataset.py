"""Dataset description, frame labels and an embedding cache.

metadata.csv (one row per WAV; every listed file is treated as fully
annotated, so a file with no annotation rows is pure background):
    file, recording_id, speaker_id, room_id[, split]
  - file: path relative to the metadata CSV (or --audio-root)
  - recording_id: the original recording the file was cut from. Clips cut from
    the same session must share it. Defaults to the file stem.
  - speaker_id / room_id: may be blank when unknown or not applicable
    (e.g. impacts have no speaker).

annotations.csv:
    file, onset, offset, label        (seconds; label in config.LABELS,
                                       other labels are kept but ignored)
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import load_wav
from .config import (
    FRAME_HOP_S,
    FRAME_WIN_S,
    LABELS,
    POS_EVENT_FRAC,
    POS_OVERLAP_S,
    SAMPLE_RATE,
)

UNKNOWN = {"", "unknown", "none", "na", "n/a", "-"}


@dataclass
class Recording:
    file: str
    path: Path
    recording_id: str
    speaker_id: str = ""
    room_id: str = ""
    split: str = ""
    events: list = field(default_factory=list)   # (onset, offset, label)

    def target_events(self, label: str) -> list[tuple[float, float]]:
        return [(on, off) for on, off, lab in self.events if lab == label]


def _clean(v) -> str:
    v = (v or "").strip()
    return "" if v.lower() in UNKNOWN else v


def load_dataset(metadata_csv: str, annotations_csv: str | None = None,
                 audio_root: str | None = None, splits_csv: str | None = None) -> list[Recording]:
    meta_path = Path(metadata_csv)
    root = Path(audio_root) if audio_root else meta_path.parent
    recs: dict[str, Recording] = {}
    with meta_path.open(newline="") as f:
        for row in csv.DictReader(f):
            fn = row["file"].strip()
            if fn in recs:
                raise ValueError(f"duplicate file in metadata: {fn}")
            recs[fn] = Recording(
                file=fn,
                path=root / fn,
                recording_id=_clean(row.get("recording_id")) or Path(fn).stem,
                speaker_id=_clean(row.get("speaker_id")),
                room_id=_clean(row.get("room_id")),
                split=_clean(row.get("split")).lower(),
            )
    if annotations_csv:
        ignored = set()
        with open(annotations_csv, newline="") as f:
            for row in csv.DictReader(f):
                fn = row["file"].strip()
                if fn not in recs:
                    raise ValueError(f"annotation for file not in metadata: {fn}")
                on, off, lab = float(row["onset"]), float(row["offset"]), row["label"].strip()
                if off <= on:
                    raise ValueError(f"{fn}: offset <= onset ({on}, {off})")
                if lab not in LABELS:
                    ignored.add(lab)
                recs[fn].events.append((on, off, lab))
        if ignored:
            print(f"note: non-target labels treated as background: {sorted(ignored)}")
    if splits_csv:
        with open(splits_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["file"] in recs:
                    recs[row["file"]].split = row["split"].strip().lower()
    return list(recs.values())


def frame_labels(n_frames: int, rec: Recording, labels=LABELS) -> tuple[np.ndarray, np.ndarray]:
    """Y (T, K) in {0,1}; M (T, K) mask, 0 = ambiguous partial overlap (ignored)."""
    Y = np.zeros((n_frames, len(labels)), np.float32)
    M = np.ones((n_frames, len(labels)), np.float32)
    starts = np.arange(n_frames) * FRAME_HOP_S
    ends = starts + FRAME_WIN_S
    for k, lab in enumerate(labels):
        for on, off in rec.target_events(lab):
            ov = np.clip(np.minimum(ends, off) - np.maximum(starts, on), 0, None)
            need = min(POS_OVERLAP_S, POS_EVENT_FRAC * (off - on))
            pos = ov >= need
            Y[pos, k] = 1.0
            M[(ov > 0) & ~pos, k] = 0.0
        M[Y[:, k] == 1, k] = 1.0
    return Y, M


# ---------------------------------------------------------------- augmentation
def augment(x: np.ndarray, rng: np.random.Generator, noise_pool: list[np.ndarray]) -> np.ndarray:
    """Random gain plus background from other *training* files at 5-25 dB SNR."""
    y = x * (10 ** (rng.uniform(-12, 6) / 20))
    if noise_pool:
        n = noise_pool[rng.integers(len(noise_pool))]
        if len(n) < len(y):
            n = np.tile(n, int(np.ceil(len(y) / len(n))))
        start = rng.integers(0, len(n) - len(y) + 1)
        n = n[start:start + len(y)]
        ps, pn = np.mean(y ** 2) + 1e-10, np.mean(n ** 2) + 1e-10
        snr = rng.uniform(5, 25)
        y = y + n * np.sqrt(ps / (pn * 10 ** (snr / 10)))
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


# ---------------------------------------------------------------- cache
class EmbeddingCache:
    """Caches YAMNet scores/embeddings per (file content, model, augmentation)."""

    def __init__(self, backend, cache_dir: str | Path = ".cache/embeddings"):
        self.backend = backend
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, path: Path, tag: str) -> Path:
        st = path.stat()
        s = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{self.backend.fingerprint}|{tag}"
        return self.dir / (hashlib.sha1(s.encode()).hexdigest() + ".npz")

    def get(self, rec: Recording, tag: str = "", waveform_fn=None) -> dict:
        p = self._key(rec.path, tag)
        if p.exists():
            z = np.load(p, allow_pickle=False)
            return {k: z[k] for k in z.files} | {"duration_s": float(z["duration_s"])}
        x = load_wav(rec.path)
        if waveform_fn is not None:
            x = waveform_fn(x)
        scores, emb = self.backend.infer(x)
        out = {"scores": scores.astype(np.float32), "emb": emb.astype(np.float32),
               "duration_s": np.float64(len(x) / SAMPLE_RATE)}
        np.savez(p, **out)
        out["duration_s"] = float(out["duration_s"])
        return out


def manifest(recs: list[Recording]) -> list[dict]:
    return [{"file": r.file, "recording_id": r.recording_id,
             "speaker_id": r.speaker_id, "room_id": r.room_id} for r in recs]


def dump_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
