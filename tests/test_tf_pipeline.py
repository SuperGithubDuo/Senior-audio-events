"""End-to-end with a YAMNet-format SavedModel (real, or the test stand-in).

Skipped unless $YAMNET_DIR points at a SavedModel.
"""

import os
from pathlib import Path

import numpy as np
import pytest

MODEL = os.environ.get("YAMNET_DIR", "models/yamnet")
pytestmark = pytest.mark.skipif(not Path(MODEL, "saved_model.pb").exists(),
                                reason="no YAMNet SavedModel available")


@pytest.fixture(scope="module")
def backend():
    from acoustic_events.backend import YamnetBackend
    return YamnetBackend(MODEL)


def test_framing_matches_live_windows(backend):
    from acoustic_events.config import FRAME_HOP, FRAME_SAMPLES

    x = np.random.default_rng(0).normal(0, 0.1, 16000 * 6).astype(np.float32)
    _, emb = backend.infer(x)
    for t in (0, 3, 7):
        _, e1 = backend.infer(x[t * FRAME_HOP:t * FRAME_HOP + FRAME_SAMPLES])
        assert e1.shape[0] == 1
        assert np.allclose(emb[t], e1[0], atol=1e-4)


def test_baseline_scorer_maps_classes(backend):
    from acoustic_events.scorers import BaselineScorer

    sc = BaselineScorer(backend.class_names)
    scores, emb = backend.infer(np.zeros(16000 * 3, np.float32))
    assert sc.score(emb, scores).shape == (len(scores), 3)


def test_short_audio_yields_one_frame(backend):
    scores, emb = backend.infer(np.zeros(4000, np.float32))
    assert scores.shape[0] == 1 and emb.shape == (1, 1024)
