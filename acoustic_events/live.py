"""Live microphone input (local only; no audio leaves the machine).

The stream is cut into exactly the windows YAMNet uses offline (15600 samples
every 7680), so a recorded WAV run through `--simulate` produces the same
events as `detect` on that file. By default no audio is written to disk.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio import load_wav, write_wav
from .config import DISCLAIMER, FRAME_HOP, FRAME_SAMPLES, SAMPLE_RATE
from .events import EventEngine


class StreamProcessor:
    """Push arbitrary-size 16 kHz chunks; get (kind, Event, info) as frames complete."""

    def __init__(self, backend, scorer, params=None, keep_audio_s: float = 0.0):
        self.backend, self.scorer = backend, scorer
        self.engine = EventEngine(scorer.labels, params or scorer.params)
        self.buf = np.zeros(0, np.float32)
        self.consumed = 0                     # samples dropped from the front of buf
        self.next_start = 0                   # absolute sample index of next frame
        self.emb_hist = deque(maxlen=scorer.context_frames + 1)
        self.keep = int(keep_audio_s * SAMPLE_RATE)
        self.last_probs = None

    def push(self, chunk: np.ndarray) -> list[tuple[str, object, dict]]:
        self.buf = np.concatenate([self.buf, np.asarray(chunk, np.float32).ravel()])
        out = []
        while self.consumed + len(self.buf) >= self.next_start + FRAME_SAMPLES:
            a = self.next_start - self.consumed
            win = self.buf[a:a + FRAME_SAMPLES]
            t0 = time.perf_counter()
            scores, emb = self.backend.infer(win)
            self.emb_hist.append(emb[:1])
            hist = np.concatenate(list(self.emb_hist))
            if len(hist) < self.emb_hist.maxlen:   # same edge padding as offline
                hist = np.concatenate([np.repeat(hist[:1], self.emb_hist.maxlen - len(hist), 0), hist])
            probs = self.scorer.score(hist, np.repeat(scores[:1], len(hist), 0))[-1]
            compute_ms = (time.perf_counter() - t0) * 1000
            self.last_probs = probs
            for kind, ev in self.engine.update(probs):
                out.append((kind, ev, {"compute_ms": round(compute_ms, 1)}))
            self.next_start += FRAME_HOP
            drop = max(0, self.next_start - self.consumed - self.keep)
            drop = min(drop, len(self.buf))
            self.buf, self.consumed = self.buf[drop:], self.consumed + drop
        return out

    def finish(self):
        return [(k, e, {}) for k, e in self.engine.flush()]

    def audio_between(self, start_s: float, end_s: float) -> np.ndarray:
        a = max(0, int(start_s * SAMPLE_RATE) - self.consumed)
        b = max(0, int(end_s * SAMPLE_RATE) - self.consumed)
        return self.buf[a:b]


class AlertLog:
    def __init__(self, path: str | None, quiet=False):
        self.f = open(path, "a", buffering=1) if path else None
        self.quiet = quiet

    def write(self, kind, ev, info, stream_t0):
        rec = {"wall_time": datetime.now().isoformat(timespec="milliseconds"),
               "type": f"event_{kind}", **ev.to_dict(), **info}
        if self.f:
            self.f.write(json.dumps(rec) + "\n")
        if not self.quiet:
            if kind == "start":
                print(f"[{rec['wall_time']}] {ev.label.upper():<14} detected  "
                      f"score={ev.peak_score:.2f} (stream t={ev.trigger_s:.2f}s, "
                      f"compute {info.get('compute_ms', 0)} ms)", flush=True)
            else:
                print(f"[{rec['wall_time']}] {ev.label:<14} ended     "
                      f"{ev.onset_s:.2f}-{ev.offset_s:.2f}s peak={ev.peak_score:.2f}", flush=True)
        return rec


def simulate(wav: str, backend, scorer, log_path=None, realtime=False, chunk_s=0.1, quiet=False):
    """Feed a WAV through the live path (for testing without a microphone)."""
    x = load_wav(wav)
    proc, log = StreamProcessor(backend, scorer), AlertLog(log_path, quiet)
    step = int(chunk_s * SAMPLE_RATE)
    t0 = time.time()
    records = []
    for i in range(0, len(x), step):
        for k, e, info in proc.push(x[i:i + step]):
            records.append(log.write(k, e, info, t0))
        if realtime:
            time.sleep(max(0.0, t0 + (i + step) / SAMPLE_RATE - time.time()))
    for k, e, info in proc.finish():
        records.append(log.write(k, e, info, t0))
    return records


def run_microphone(backend, scorer, device=None, log_path="logs/events.jsonl",
                   save_clips: str | None = None, block_s=0.1, quiet=False):
    import sounddevice as sd

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    keep = 12.0 if save_clips else 0.0
    proc, log = StreamProcessor(backend, scorer, keep_audio_s=keep), AlertLog(log_path, quiet)
    q: queue.Queue = queue.Queue(maxsize=200)
    stop = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"audio status: {status}", file=sys.stderr)
        try:
            q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            print("warning: processing is falling behind; dropping audio", file=sys.stderr)

    if save_clips:
        Path(save_clips).mkdir(parents=True, exist_ok=True)
    print(DISCLAIMER)
    print(f"Listening on {'default device' if device is None else device} at {SAMPLE_RATE} Hz. "
          f"Log: {log_path}. Ctrl+C to stop.")
    stream_t0 = time.time()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=int(block_s * SAMPLE_RATE), device=device, callback=callback):
        try:
            while not stop.is_set():
                chunk = q.get()
                for k, e, info in proc.push(chunk):
                    log.write(k, e, info, stream_t0)
                    if save_clips and k == "end":
                        clip = proc.audio_between(e.onset_s - 1.0, e.offset_s + 1.0)
                        name = f"{datetime.now():%Y%m%d-%H%M%S}_{e.label}.wav"
                        write_wav(Path(save_clips) / name, clip)
        except KeyboardInterrupt:
            pass
    for k, e, info in proc.finish():
        log.write(k, e, info, stream_t0)
