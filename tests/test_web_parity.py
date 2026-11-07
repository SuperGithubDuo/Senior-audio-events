"""The browser page (docs/engine.js) must score and segment exactly like Python."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from acoustic_events.config import LABELS
from acoustic_events.events import EventParams, run_engine
from acoustic_events.scorers import HeadScorer
from acoustic_events.webexport import export_web

ROOT = Path(__file__).resolve().parents[1]
NODE_SCRIPT = r"""
import { headScorer, runEngine } from "%s";
import fs from "node:fs";
const fx = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const head = headScorer(fx.head);
const emb = fx.emb.map((r) => Float32Array.from(r));
const probs = head.score(emb);
const events = runEngine(probs, head.labels, head.params);
console.log(JSON.stringify({ probs, events }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("kind", ["logreg", "mlp"])
def test_engine_js_matches_python(tmp_path, kind):
    rng = np.random.default_rng(5)
    D, c = 16, 1
    if kind == "logreg":
        layers = [(rng.normal(0, 0.6, (D * (1 + c), 3)), rng.normal(0, 0.5, 3))]
    else:
        layers = [(rng.normal(0, 0.5, (D * (1 + c), 12)), rng.normal(0, 0.2, 12)),
                  (rng.normal(0, 0.8, (12, 3)), rng.normal(0, 0.2, 3))]
    params = {"scream": EventParams.for_label("scream", 0.55),
              "groan_whimper": EventParams.for_label("groan_whimper", 0.5),
              "impact": EventParams.for_label("impact", 0.6, min_frames=2)}
    head = HeadScorer(LABELS, rng.normal(0, 0.1, D * 2), rng.uniform(0.5, 2, D * 2),
                      layers, c, params, {"head_kind": kind})
    head.save(tmp_path / "head")
    doc = export_web(str(tmp_path / "head"), str(tmp_path / "head_web.json"))
    head = HeadScorer.load(tmp_path / "head")
    emb = rng.normal(0, 1, (300, D)).astype(np.float32)
    fixture = tmp_path / "fx.json"
    fixture.write_text(json.dumps({"head": doc, "emb": emb.tolist()}))
    script = tmp_path / "run.mjs"
    script.write_text(NODE_SCRIPT % (ROOT / "docs" / "engine.js").as_uri())
    out = json.loads(subprocess.run(["node", str(script), str(fixture)], check=True,
                                    capture_output=True, text=True).stdout)
    py_probs = head.score(emb)
    assert np.allclose(np.array(out["probs"]), py_probs, atol=1e-5)
    py_events = run_engine(py_probs, LABELS, head.params)
    key = lambda e: (e["label"], e["start_frame"], e["end_frame"], round(e["trigger_s"], 3))
    assert sorted(map(key, out["events"])) == sorted(key(e.to_dict()) for e in py_events)
    assert len(py_events) > 5
