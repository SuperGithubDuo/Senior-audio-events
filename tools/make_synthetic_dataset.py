"""TEST ONLY: generate a small synthetic dataset in the expected layout.

Synthetic screams/groans/impacts are crude signal models; they validate that
the pipeline runs end to end, not that it works on real people or rooms.

usage: python tools/make_synthetic_dataset.py --out data/synthetic
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, lfilter

SR = 16000


def env(n, a=0.05, r=0.1):
    e = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na)
    e[-nr:] = np.minimum(e[-nr:], np.linspace(1, 0, nr))
    return e


def harmonic(f0_track, n_harm, tilt):
    ph = 2 * np.pi * np.cumsum(f0_track) / SR
    return sum((tilt ** h) * np.sin((h + 1) * ph) for h in range(n_harm))


def scream(rng, pitch):
    n = int(rng.uniform(0.6, 2.0) * SR)
    t = np.arange(n) / SR
    f0 = pitch * 4 * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(4, 7) * t)) * np.linspace(1, rng.uniform(0.85, 1.15), n)
    x = harmonic(f0, 8, 0.7) + 0.2 * rng.normal(size=n)
    return x * env(n) / 3


def groan(rng, pitch):
    n = int(rng.uniform(0.8, 2.5) * SR)
    t = np.arange(n) / SR
    f0 = pitch * (1 + 0.03 * np.sin(2 * np.pi * 3 * t)) * np.linspace(1.1, 0.85, n)
    am = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(2, 5) * t)
    x = harmonic(f0, 15, 0.8) * am
    x = lfilter([1.0], [1.0, -0.9], x)
    return x * env(n, 0.15, 0.3) / np.max(np.abs(x) + 1e-9) * 0.5


def impact(rng):
    n = int(rng.uniform(0.15, 0.4) * SR)
    t = np.arange(n) / SR
    body = rng.normal(size=n) * np.exp(-t / rng.uniform(0.02, 0.08))
    thud = np.sin(2 * np.pi * rng.uniform(50, 120) * t) * np.exp(-t / 0.06)
    return 0.9 * (0.6 * body + thud) / 1.6


def babble(rng, n):
    """Speech-like distractor: formant-filtered pulses with syllable rhythm."""
    t = np.arange(n) / SR
    f0 = rng.uniform(100, 220) * (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t))
    x = harmonic(f0, 20, 0.85)
    syl = (np.sin(2 * np.pi * rng.uniform(3, 5) * t) > 0.2).astype(float)
    x = lfilter([1.0], [1.0, -0.95], x) * syl
    return 0.05 * x / (np.std(x) + 1e-9)


def room_ir(rng, rt60):
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.normal(size=n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    return ir / np.sqrt(np.sum(ir ** 2))


def make_file(rng, dur, room, pitch, n_events, kinds):
    n = int(dur * SR)
    x = room["noise_level"] * lfilter([1.0], [1.0, -room["noise_color"]], rng.normal(size=n))
    x += 0.003 * np.sin(2 * np.pi * room["hum"] * np.arange(n) / SR)
    if rng.random() < 0.7:                       # TV / conversation in the background
        a = rng.integers(0, n // 2)
        b = min(n, a + rng.integers(SR * 5, SR * 20))
        x[a:b] += babble(rng, b - a)
    for _ in range(rng.integers(0, 3)):          # soft door clicks (hard negatives)
        p = rng.integers(0, n - 800)
        x[p:p + 800] += 0.05 * rng.normal(size=800) * np.exp(-np.arange(800) / 80)
    events = []
    for _ in range(n_events):
        kind = kinds[rng.integers(len(kinds))]
        s = {"scream": lambda: scream(rng, pitch), "groan_whimper": lambda: groan(rng, pitch),
             "impact": lambda: impact(rng)}[kind]()
        s = fftconvolve(s, room["ir"])[: len(s) + len(room["ir"]) // 3]
        s *= 10 ** (rng.uniform(-8, 0) / 20)
        p = rng.integers(SR, n - len(s) - SR)
        x[p:p + len(s)] += s
        events.append((round(p / SR, 3), round((p + len(s)) / SR, 3), kind))
        if kind == "scream" and rng.random() < 0.3:   # co-occurring impact
            im = fftconvolve(impact(rng), room["ir"])
            q = p + rng.integers(0, max(1, len(s) - len(im)))
            x[q:q + len(im)] += im[: n - q]
            events.append((round(q / SR, 3), round((q + len(im)) / SR, 3), "impact"))
    x = np.clip(x, -1, 1)
    return (x * 32767).astype(np.int16), events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rooms", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    meta, ann = [], []
    for r in range(a.rooms):
        room = {"ir": room_ir(rng, rng.uniform(0.2, 0.7)), "noise_level": rng.uniform(0.002, 0.01),
                "noise_color": rng.uniform(0.5, 0.98), "hum": rng.choice([50.0, 60.0])}
        spk = f"spk{r:02d}"
        pitch = rng.uniform(90, 200)
        for i in range(4):
            fn = f"audio/room{r:02d}_{spk}_take{i}.wav"
            pcm, evs = make_file(rng, rng.uniform(30, 50), room, pitch, rng.integers(3, 7),
                                 ["scream", "groan_whimper", "impact"])
            wavfile.write(out / fn, SR, pcm)
            meta.append([fn, f"rec_r{r}_s{i}", spk, f"room{r:02d}"])
            ann += [[fn, *e] for e in evs]
        fn = f"audio/room{r:02d}_background.wav"
        pcm, _ = make_file(rng, 180, room, pitch, 0, ["impact"])
        wavfile.write(out / fn, SR, pcm)
        meta.append([fn, f"rec_r{r}_bg", "", f"room{r:02d}"])
    with (out / "metadata.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["file", "recording_id", "speaker_id", "room_id"], *meta])
    with (out / "annotations.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["file", "onset", "offset", "label"], *ann])
    print(f"wrote {len(meta)} files, {len(ann)} events to {out}")


if __name__ == "__main__":
    main()
