"""Drive docs/index.html in headless Chrome with a fake YAMNet; compare with Python.

Skipped unless Playwright + a Chrome binary and a local copy of tf.min.js
(npm install @tensorflow/tfjs@4.22.0 into $TFJS_DIR or ./node_modules) exist.
"""

import functools
import http.server
import json
import os
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest

from acoustic_events.audio import write_wav
from acoustic_events.config import FRAME_HOP, FRAME_SAMPLES, LABELS
from acoustic_events.detect import detect_waveform
from acoustic_events.events import EventParams
from acoustic_events.scorers import HeadScorer
from acoustic_events.webexport import export_web

ROOT = Path(__file__).resolve().parents[1]
TFJS = Path(os.environ.get("TFJS_DIR", ROOT / "node_modules")) / "@tensorflow/tfjs/dist/tf.min.js"
CHROME = os.environ.get("CHROME", "/usr/bin/google-chrome")
pw = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.skipif(not (TFJS.exists() and Path(CHROME).exists()), reason="needs tf.min.js and Chrome")

FAKE_MODEL_JS = """() => {
  const fake = { predict: (w) => tf.tidy(() => {
    const n = w.shape[0]; let N = Math.max(n, 15600); const extra = N - 15600;
    N += (7680 - (extra % 7680)) % 7680;
    const fr = tf.signal.frame(tf.pad(w, [[0, N - n]]), 15600, 7680);
    const T = fr.shape[0];
    const emb = tf.concat([fr.square().mean(1, true).sqrt(), fr.abs().max(1, true), tf.zeros([T, 1022])], 1);
    return [tf.zeros([T, 521]), emb, tf.zeros([4, 64])];
  })};
  return window.acousticEvents.useYamnet(fake, "test stand-in");
}"""


class Fake1024:
    fingerprint = "fake"

    def infer(self, w):
        n = max(len(w), FRAME_SAMPLES)
        n += (-(n - FRAME_SAMPLES)) % FRAME_HOP
        w = np.pad(np.asarray(w, np.float32), (0, n - len(w)))
        T = 1 + (n - FRAME_SAMPLES) // FRAME_HOP
        win = np.stack([w[t * FRAME_HOP:t * FRAME_HOP + FRAME_SAMPLES] for t in range(T)])
        emb = np.zeros((T, 1024), np.float32)
        emb[:, 0] = np.sqrt((win ** 2).mean(1))
        emb[:, 1] = np.abs(win).max(1)
        return np.zeros((T, 521), np.float32), emb


@pytest.fixture(scope="module")
def server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "docs"))
    handler.log_message = lambda *a: None
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_page_matches_python(server, tmp_path):
    rng = np.random.default_rng(3)
    x = 0.005 * rng.normal(size=16000 * 40).astype(np.float32)
    for s in (5.0, 12.3, 20.0, 31.0):
        a = int(s * 16000)
        x[a:a + 8000] += 0.4 * rng.normal(size=8000)
    wav = tmp_path / "bursts.wav"
    write_wav(wav, x)
    from acoustic_events.audio import load_wav
    x = load_wav(wav)                                  # use the int16-quantised signal

    W = np.zeros((2048, 3), np.float32)
    W[0, :] = 40.0
    head = HeadScorer(LABELS, np.zeros(2048), np.ones(2048), [(W, np.full(3, -4.0, np.float32))], 1,
                      {l: EventParams.for_label(l, 0.5) for l in LABELS}, {"head_kind": "logreg"})
    head.save(tmp_path / "head")
    export_web(str(tmp_path / "head"), str(tmp_path / "head_web.json"))
    expected, _ = detect_waveform(x, Fake1024(), head)
    (tmp_path / "ann.csv").write_text("file,onset,offset,label\nbursts.wav,5.0,5.5,impact\nbursts.wav,25,26,scream\n")

    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1360, "height": 1100})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("https://cdn.jsdelivr.net/**", lambda r: r.fulfill(path=str(TFJS), content_type="application/javascript"))
        page.route("https://fonts.googleapis.com/**", lambda r: r.abort())
        page.add_init_script("window.__AE_SKIP_AUTOLOAD__ = true;")
        page.goto(server + "/index.html")
        page.wait_for_function("window.acousticEvents && window.acousticEvents.state.baseline")
        page.evaluate(FAKE_MODEL_JS)
        page.set_input_files("#headFile", str(tmp_path / "head_web.json"))
        page.set_input_files("#audioFile", str(wav))
        page.wait_for_function("window.acousticEvents.state.src && window.acousticEvents.state.src.probs.length > 0")
        page.set_input_files("#gtFile", str(tmp_path / "ann.csv"))
        page.wait_for_timeout(300)
        got = page.evaluate("window.acousticEvents.state.events")
        rows = page.locator("#events tr").count()
        summary = page.locator("#gtSummary").inner_text()
        page.screenshot(path=str(ROOT / "docs" / "screenshot.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 900})
        page.wait_for_timeout(200)
        page.screenshot(path=str(tmp_path / "mobile.png"), full_page=True)
        mobile_overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1")
        browser.close()

    assert not errors, errors
    key = lambda e: (e["label"], e["start_frame"], e["end_frame"], round(e["trigger_s"], 3))
    assert sorted(map(key, got)) == sorted(key(e.to_dict()) for e in expected)
    assert rows == len(expected) and len(expected) >= 12
    assert "1 of 1 found" in summary and "0 of 1 found" in summary
    assert not mobile_overflow
