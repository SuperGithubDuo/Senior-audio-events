"""YAMNet backend, always loaded from a local SavedModel directory.

Downloading happens only in `fetch_model` (one time, explicit). Inference never
touches the network.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from .config import DEFAULT_YAMNET_DIR, YAMNET_HUB_URL

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def resolve_model_dir(model_dir: str | None) -> Path:
    d = Path(model_dir or os.environ.get("YAMNET_DIR") or DEFAULT_YAMNET_DIR)
    if not (d / "saved_model.pb").exists():
        raise FileNotFoundError(
            f"No YAMNet SavedModel at '{d}'. Run once (needs internet):\n"
            f"  python -m acoustic_events fetch-model --out {d}\n"
            "or, on an offline machine, copy a downloaded archive and run:\n"
            f"  python -m acoustic_events fetch-model --from-archive yamnet.tar.gz --out {d}"
        )
    return d


class YamnetBackend:
    """Wraps the TF Hub YAMNet SavedModel: waveform -> (scores, embeddings)."""

    def __init__(self, model_dir: str | None = None):
        import tensorflow as tf

        self._tf = tf
        self.model_dir = resolve_model_dir(model_dir)
        self.model = tf.saved_model.load(str(self.model_dir))
        class_map = self.model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map) as f:
            self.class_names = [row["display_name"] for row in csv.DictReader(f)]
        h = hashlib.sha1()
        h.update((self.model_dir / "saved_model.pb").read_bytes())
        var_idx = self.model_dir / "variables" / "variables.index"
        if var_idx.exists():
            h.update(var_idx.read_bytes())
        self.fingerprint = h.hexdigest()[:16]

    def infer(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """waveform: float32 mono 16 kHz. Returns scores (T, 521), embeddings (T, 1024)."""
        w = self._tf.convert_to_tensor(np.asarray(waveform, dtype=np.float32))
        scores, embeddings, _ = self.model(w)
        return scores.numpy(), embeddings.numpy()


def fetch_model(out_dir: str, from_archive: str | None = None) -> Path:
    """Place YAMNet in out_dir as a plain SavedModel (one-time step)."""
    out = Path(out_dir)
    if (out / "saved_model.pb").exists():
        print(f"YAMNet already present at {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        if from_archive:
            with tarfile.open(from_archive) as tar:
                tar.extractall(tmp, filter="data")
            src = next(Path(tmp).rglob("saved_model.pb")).parent
        else:
            os.environ["TFHUB_CACHE_DIR"] = tmp
            import tensorflow_hub as hub

            src = Path(hub.resolve(YAMNET_HUB_URL))
        shutil.copytree(src, out)
    print(f"YAMNet saved to {out}; all later runs load it locally.")
    return out
