# Local acoustic event detection on YAMNet

Detects three kinds of **acoustic events** — `scream`, `groan_whimper`, `impact` —
in WAV files or from a live microphone, using pretrained YAMNet plus a small
transfer-learning head trained on your own recordings.

> These are acoustic event detections, **not** confirmed emergencies or falls.
> An "impact" is a thud-like sound; it may or may not be a person falling.
> Any care workflow built on top should treat a detection as a prompt to check,
> and should be validated with residents, staff and consent in place.

Labels are independent (multi-label): a scream and an impact can be active at
the same time, and each gets its own event.

All audio processing is local. The only network access is the explicit,
one-time `fetch-model` step; after that YAMNet loads from `models/yamnet`.
Nothing is uploaded, and live mode writes no audio unless you pass `--save-clips`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m acoustic_events fetch-model          # one time, downloads YAMNet to models/yamnet
python -m pytest -q                            # unit tests (TF tests run when the model is present)
```

Air-gapped machine: download the YAMNet TF2 SavedModel archive elsewhere and run
`python -m acoustic_events fetch-model --from-archive yamnet.tar.gz`.

## 1. Detect events in a WAV file

```bash
# pretrained YAMNet only (AudioSet classes mapped to the three labels)
python -m acoustic_events detect recording.wav --out events.csv

# with a trained head, JSON output and per-frame scores
python -m acoustic_events detect recording.wav --model models/head --out events.json --frame-scores frames.csv
```

Any sample rate / channel count is accepted (converted to 16 kHz mono).
Each event row has:

| field | meaning |
|---|---|
| `label` | scream, groan_whimper, impact |
| `onset_s`, `offset_s` | event extent in the file (0.48 s frame resolution) |
| `trigger_s` | file time at which a live stream would have raised it |
| `peak_score`, `mean_score` | head (or YAMNet) probability |
| `n_frames`, `start_frame`, `end_frame` | YAMNet frames involved |

The pretrained mapping lives in `acoustic_events/config.py` (`BASELINE_CLASS_MAP`):
scream ← Screaming; groan_whimper ← Groan, Whimper, Wail/moan;
impact ← Thump/thud, Bang, Slam, Smash/crash, Breaking, Shatter. It is a
starting point; expect a trained head to do much better in your rooms.
Override a threshold with `--threshold impact=0.4`.

## 2. Prepare data

`metadata.csv` — one row per WAV. Every listed file is assumed fully
annotated, so a file with no annotation rows is pure background.

```
file,recording_id,speaker_id,room_id
audio/apt3_bath_take1.wav,sess_014,spk07,apt3_bath
audio/apt3_bath_bg_evening.wav,sess_015,,apt3_bath
```

* `recording_id`: the original recording a file was cut from. Clips from one
  session must share it — this is what keeps near-duplicate audio out of test.
* `speaker_id`, `room_id`: blank when unknown or not applicable (impacts).
* optional `split` column (train/val/test) to fix the split yourself.

`annotations.csv` — `file,onset,offset,label` in seconds. Other labels
(e.g. `door`, `tv`) are allowed and treated as background, which is a handy way
to document hard negatives.

## 3. Split by recording, speaker and room

```bash
python -m acoustic_events split --metadata data/metadata.csv --annotations data/annotations.csv \
    --out data/splits.csv --report data/splits_report.json
```

Files sharing a recording, speaker **or** room are grouped and each group goes
wholly into train, val or test (≈60/20/20 by duration, balanced for events of
each label). If that's impossible — e.g. everything was recorded in one room —
it relaxes room, then speaker (never recording), and the report lists what
still crosses splits under `enforced_separation`, `leakage` and `warnings`.

## 4. Train the head

```bash
python -m acoustic_events train --metadata data/metadata.csv --annotations data/annotations.csv \
    --splits data/splits.csv --out models/head --target-fa-per-hour 0.5 --augment 2
```

* YAMNet stays frozen; its 1024-d embeddings (plus `--context 1` previous
  frame, causal) feed one sigmoid output per label.
  `--head logreg` (default, class-balanced) or `--head mlp`.
* Frame labels: a 0.96 s frame is positive when it overlaps an event by
  ≥ min(0.25 s, half the event); smaller partial overlaps are ignored rather
  than called negative, so short impacts are still learned.
* `--augment N` adds N copies of each training file with random gain and
  background mixed in from other **training** background files (5–25 dB SNR).
* Thresholds are tuned per label on **val**: highest event recall with
  false alerts/hour ≤ `--target-fa-per-hour`. Test is never read.
* The head is stored as `head.npz` + `head.json` (no pickle) along with the
  train/val manifest used for leakage checks.

## 5. Evaluate on unseen recordings

```bash
python -m acoustic_events evaluate --metadata data/metadata.csv --annotations data/annotations.csv \
    --splits data/splits.csv --model models/head --out reports/test
# or --model baseline to score pretrained YAMNet on the same test set
```

Refuses to run if any test file was in the head's training/validation set,
and reports shared speakers/rooms. Writes `report.md`, `report.json`,
`ground_truth_events.csv` (hit/miss + delay per event), `false_alerts.csv`
(for listening review) and `detections.csv`.

Metrics, per label and for an "any label" alert stream:

* **Event recall** — share of annotated events with an overlapping detection
  (±1 s collar), with a 95% Wilson interval.
* **False alerts per hour** — detections overlapping no annotated event of
  that label, divided by hours evaluated, with a 95% Poisson interval. Repeated
  detections on an already-found event are counted as duplicates, not FAs.
  `false_alerts_on_other_label_events` shows confusions between labels.
* **Detection delay** — `trigger_s − onset` for detected events (median, p90,
  max). It includes YAMNet's ~0.98 s window; live mode logs compute time too.
* Per-room breakdown, frame-level average precision, and a threshold sweep
  (recall vs FA/h) per label.

FA/h is only as good as the background audio in the test set. Hold out several
hours of realistic, unscripted audio per room (TV, cooking, conversation,
vacuuming, dishes, doors, coughing, snoring, pets); otherwise the number will
be optimistic and the intervals wide.

## 6. Live microphone

```bash
python -m acoustic_events devices                                   # list inputs
python -m acoustic_events live --model models/head --log logs/events.jsonl [--device 2]
python -m acoustic_events live --model models/head --simulate test.wav [--realtime]
```

The stream is cut into exactly YAMNet's offline windows (0.975 s every 0.48 s),
so `--simulate file.wav` reproduces `detect file.wav`. Each event logs an
`event_start` line when raised and `event_end` when it finishes, with scores
and `compute_ms`. `--save-clips DIR` keeps a few seconds around each event
locally for review — off by default; only enable with residents' consent.

From Python:

```python
from acoustic_events.backend import YamnetBackend
from acoustic_events.scorers import HeadScorer
from acoustic_events.live import StreamProcessor

proc = StreamProcessor(YamnetBackend(), HeadScorer.load("models/head"))
for kind, event, info in proc.push(chunk_16k_float32):
    ...   # kind is "start" or "end"
```

## 7. Browser test page

`docs/` is a static page (hosted with GitHub Pages) for trying the detector
without installing anything. It runs YAMNet with TensorFlow.js in the
browser: **audio never leaves the tester's device**; only model files are
downloaded (from TensorFlow Hub, or from the site itself, see `docs/models/README.md`).

```bash
python -m acoustic_events export-web --model models/head --out head_web.json
```

Then open the page, load `head_web.json` under "Classifier" and drop a WAV file
or start the microphone. It shows per-label scores over time, detected events
with the moment each would be raised, threshold sliders, CSV/JSON export, and
an optional comparison against an annotations CSV. The exported JSON holds
weights and thresholds only, not training file names.

The page uses the same framing, head maths and event engine as Python
(`tests/test_web_parity.py`, `tests/test_web_ui.py` check this). TF.js and
TensorFlow YAMNet are separate conversions of the same weights and the browser
resamples non-16 kHz files itself, so expect small score differences; report
numbers from `evaluate`, not the page.

Local preview: `python -m http.server -d docs 8000` and open http://localhost:8000.

## Tuning knobs

`acoustic_events/config.py`: per-label `smooth_frames`, `min_frames`
(frames needed to raise an event — fewer FAs, more delay), `release_frames`
(merges gaps), the off/on hysteresis ratio, labelling overlap and evaluation
collars.

## Data collection notes

* Record older adults' voices where possible; YAMNet and most public screams
  skew younger. Groans and whimpers are quiet — capture them at realistic
  distances, not close-miked.
* Impacts: drop realistic objects and fall mats/dummies on the actual floor
  types (carpet, tile, vinyl), in each room, at several distances.
* Collect hard negatives deliberately: TV shows with screaming, door slams,
  dropped pans, chairs, laughter, coughing fits.
* Keep each recording session's clips under one `recording_id`.

## Layout

```
acoustic_events/   config, audio, backend (YAMNet), scorers (baseline/head), events
                   (causal engine), detect, dataset, split, train, metrics, evaluate, live
tests/             unit tests (no TF), TF pipeline tests (need a model), browser parity tests
docs/              GitHub Pages test page (index.html, app.js, engine.js)
tools/             make_synthetic_dataset.py, make_standin_yamnet.py — TEST ONLY
examples/          metadata / annotations templates
```

The tools are for exercising the code without real data or model weights: the
stand-in reproduces YAMNet's framing and interface with random weights, and the
synthetic sounds are crude. Neither says anything about real-world accuracy.
