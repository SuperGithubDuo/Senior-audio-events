"""Export a trained head to JSON for the browser test page (docs/).

Only weights, thresholds and counts are written; file names from the training
manifest are left out, since they can identify residents or rooms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scorers import HeadScorer

FORMAT = "acoustic-events-head/1"


def export_web(model_dir: str, out_path: str) -> dict:
    h = HeadScorer.load(model_dir)
    layers = [{"shape": list(W.shape), "W": np.round(W.ravel(), 7).tolist(),
               "b": np.round(b, 7).tolist()} for W, b in h.layers]
    meta = h.meta
    doc = {
        "format": FORMAT,
        "labels": h.labels,
        "context_frames": h.context_frames,
        "mean": np.round(h.mean, 7).tolist(),
        "scale": np.round(h.scale, 7).tolist(),
        "layers": layers,
        "params": {l: p.to_dict() for l, p in h.params.items()},
        "info": {
            "head_kind": meta.get("head_kind"),
            "yamnet_fingerprint": meta.get("yamnet_fingerprint"),
            "target_fa_per_hour": meta.get("target_fa_per_hour"),
            "train_files": len(meta.get("train_files", [])),
            "val_files": len(meta.get("val_files", [])),
            "val_hours": meta.get("val_hours"),
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, separators=(",", ":")))
    return doc
