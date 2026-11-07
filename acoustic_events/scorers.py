"""Turn YAMNet outputs into per-label probabilities.

* BaselineScorer: pretrained YAMNet only (max over mapped AudioSet classes).
* HeadScorer: custom transfer-learning head trained on YAMNet embeddings.

Both expose .labels, .context_frames, .params (EventParams per label) and
.score(embeddings, yamnet_scores) -> (T, K).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import BASELINE_CLASS_MAP, BASELINE_THRESHOLDS, LABELS
from .events import EventParams


class BaselineScorer:
    kind = "baseline"
    context_frames = 0

    def __init__(self, class_names: list[str], class_map=None, thresholds=None):
        class_map = class_map or BASELINE_CLASS_MAP
        thresholds = thresholds or BASELINE_THRESHOLDS
        self.labels = list(LABELS)
        idx = {n: i for i, n in enumerate(class_names)}
        self.columns = []
        for lab in self.labels:
            cols = [idx[n] for n in class_map[lab] if n in idx]
            missing = [n for n in class_map[lab] if n not in idx]
            if missing:
                print(f"warning: YAMNet class map lacks {missing} for '{lab}'")
            if not cols:
                raise ValueError(f"no YAMNet classes found for label '{lab}'")
            self.columns.append(cols)
        self.params = {l: EventParams.for_label(l, thresholds[l]) for l in self.labels}
        self.meta = {"class_map": {l: list(class_map[l]) for l in self.labels}}

    def score(self, embeddings: np.ndarray, yamnet_scores: np.ndarray) -> np.ndarray:
        return np.stack([yamnet_scores[:, c].max(axis=1) for c in self.columns], axis=1)


def stack_context(emb: np.ndarray, context_frames: int) -> np.ndarray:
    """Causal context: [e_t, e_{t-1}, ..., e_{t-c}], edge-padded at the start."""
    if context_frames <= 0:
        return emb
    parts = [emb]
    for lag in range(1, context_frames + 1):
        shifted = np.concatenate([np.repeat(emb[:1], lag, axis=0), emb[:-lag]], axis=0)
        parts.append(shifted[: len(emb)])
    return np.concatenate(parts, axis=1)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class HeadScorer:
    """Small MLP / logistic head stored as plain arrays (npz + json, no pickle)."""

    kind = "head"

    def __init__(self, labels, mean, scale, layers, context_frames, params, meta=None):
        self.labels = list(labels)
        self.mean = np.asarray(mean, np.float32)
        self.scale = np.asarray(scale, np.float32)
        self.layers = [(np.asarray(W, np.float32), np.asarray(b, np.float32)) for W, b in layers]
        self.context_frames = int(context_frames)
        self.params = params
        self.meta = meta or {}

    def predict_features(self, x: np.ndarray) -> np.ndarray:
        h = (x - self.mean) / self.scale
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return _sigmoid(h)

    def score(self, embeddings: np.ndarray, yamnet_scores: np.ndarray | None = None) -> np.ndarray:
        return self.predict_features(stack_context(embeddings, self.context_frames))

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        arrays = {"mean": self.mean, "scale": self.scale}
        for i, (W, b) in enumerate(self.layers):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        np.savez(out / "head.npz", **arrays)
        cfg = {
            "labels": self.labels,
            "n_layers": len(self.layers),
            "context_frames": self.context_frames,
            "params": {l: p.to_dict() for l, p in self.params.items()},
            "meta": self.meta,
        }
        (out / "head.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, model_dir: str | Path) -> "HeadScorer":
        d = Path(model_dir)
        cfg = json.loads((d / "head.json").read_text())
        z = np.load(d / "head.npz", allow_pickle=False)
        layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(cfg["n_layers"])]
        params = {l: EventParams(**p) for l, p in cfg["params"].items()}
        return cls(cfg["labels"], z["mean"], z["scale"], layers,
                   cfg["context_frames"], params, cfg.get("meta", {}))


def load_scorer(model: str | None, class_names: list[str]):
    """model=None or 'baseline' -> pretrained mapping, else a trained head dir."""
    if model in (None, "", "baseline"):
        return BaselineScorer(class_names)
    return HeadScorer.load(model)
