"""Shared constants and defaults.

Everything time-related follows YAMNet's own framing so that offline files,
training, evaluation and the live microphone all see identical frames:

    frame t covers samples [t * FRAME_HOP, t * FRAME_HOP + FRAME_SAMPLES)

FRAME_SAMPLES is 15600 (0.975 s), the minimum YAMNet needs for one patch
(0.96 s patch + one 25 ms STFT window - one 10 ms hop). A frame's scores can
only exist once its last sample has arrived, so frame t becomes *available* at
t * FRAME_HOP_S + FRAME_AVAILABLE_S. Detection delay is measured from that.
"""

from __future__ import annotations

LABELS: tuple[str, ...] = ("scream", "groan_whimper", "impact")

SAMPLE_RATE = 16000
FRAME_HOP = 7680            # 0.48 s
FRAME_SAMPLES = 15600       # 0.975 s
FRAME_HOP_S = FRAME_HOP / SAMPLE_RATE
FRAME_WIN_S = 0.96          # nominal patch length used for event extents
FRAME_AVAILABLE_S = FRAME_SAMPLES / SAMPLE_RATE

DEFAULT_YAMNET_DIR = "models/yamnet"
YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"

# Pretrained baseline (before any custom training): each target label is the
# max score over these AudioSet classes (YAMNet display names). "Knock",
# "Yell" and "Crying, sobbing" are left out on purpose: they are common in
# homes (doors, TV, conversation) and inflate false alerts. Edit if needed.
BASELINE_CLASS_MAP: dict[str, tuple[str, ...]] = {
    "scream": ("Screaming",),
    "groan_whimper": ("Groan", "Whimper", "Wail, moan"),
    "impact": ("Thump, thud", "Bang", "Slam", "Smash, crash", "Breaking", "Shatter"),
}

# Event engine defaults per label. Thresholds for a trained head are tuned on
# the validation split; these are starting points only.
# smooth_frames: causal moving average length (adds delay, reduces flicker)
# min_frames:    frames above threshold before an event is confirmed/alerted
# release_frames: frames below off_threshold before an event ends (merges gaps)
DEFAULT_EVENT_PARAMS: dict[str, dict] = {
    "scream":        {"smooth_frames": 1, "min_frames": 1, "release_frames": 1},
    "groan_whimper": {"smooth_frames": 2, "min_frames": 1, "release_frames": 2},
    "impact":        {"smooth_frames": 1, "min_frames": 1, "release_frames": 1},
}
BASELINE_THRESHOLDS = {"scream": 0.30, "groan_whimper": 0.20, "impact": 0.30}
HEAD_DEFAULT_THRESHOLD = 0.5
OFF_RATIO = 0.6             # off_threshold = on_threshold * OFF_RATIO

# Frame labelling from annotations: a frame is positive for a label when the
# event overlaps the frame window by >= min(POS_OVERLAP_S, POS_EVENT_FRAC *
# event duration). Smaller non-zero overlaps are ignored during training.
POS_OVERLAP_S = 0.25
POS_EVENT_FRAC = 0.5

# Evaluation: a detection counts for a ground-truth event if their extents
# overlap after widening the ground truth by these collars (seconds).
COLLAR_BEFORE_S = 1.0
COLLAR_AFTER_S = 1.0

DISCLAIMER = (
    "Outputs are acoustic event detections (scream-like, groan/whimper-like, "
    "impact-like sounds). They are not confirmed emergencies or falls."
)


def frame_start(t: int) -> float:
    return t * FRAME_HOP_S


def frame_end(t: int) -> float:
    return t * FRAME_HOP_S + FRAME_WIN_S


def frame_available(t: int) -> float:
    return t * FRAME_HOP_S + FRAME_AVAILABLE_S
