"""Evaluate a scorer on unseen recordings (default: the 'test' split)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import DISCLAIMER, LABELS
from .dataset import EmbeddingCache, dump_json, frame_labels
from .events import EventParams
from .metrics import evaluate
from .train import items_for


def leakage_check(eval_recs, scorer) -> dict:
    """Compare evaluation files with the head's training/validation manifest."""
    seen = scorer.meta.get("train_files", []) + scorer.meta.get("val_files", [])
    out = {"files": [], "recording_id": [], "speaker_id": [], "room_id": []}
    if not seen:
        return out
    sets = {k: {m[k] for m in seen if m.get(k)} for k in ("file", "recording_id", "speaker_id", "room_id")}
    for r in eval_recs:
        if r.file in sets["file"]:
            out["files"].append(r.file)
        for k in ("recording_id", "speaker_id", "room_id"):
            v = getattr(r, k)
            if v and v in sets[k] and v not in out[k]:
                out[k].append(v)
    return out


def _frame_ap(recs, probs_list, labels):
    from sklearn.metrics import average_precision_score

    ap = {}
    ys, ms = zip(*(frame_labels(len(p), r, labels) for r, p in zip(recs, probs_list)))
    Y, M, P = np.concatenate(ys), np.concatenate(ms), np.concatenate(probs_list)
    for k, lab in enumerate(labels):
        use = M[:, k] > 0
        ap[lab] = (round(float(average_precision_score(Y[use, k], P[use, k])), 4)
                   if Y[use, k].sum() > 0 else None)
    return ap


def _write_csv(rows, path):
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_evaluation(recs, backend, scorer, split="test", out_dir="reports/eval",
                   allow_leakage=False, sweep=True, cache_dir=".cache/embeddings") -> dict:
    eval_recs = [r for r in recs if r.split == split] if split != "all" else list(recs)
    if not eval_recs:
        raise ValueError(f"no files in split '{split}'")
    leaks = leakage_check(eval_recs, scorer)
    if leaks["files"] and not allow_leakage:
        raise SystemExit(f"refusing to evaluate: {len(leaks['files'])} file(s) were used in "
                         f"training/validation, e.g. {leaks['files'][:3]} (use --allow-leakage to override)")
    fp = scorer.meta.get("yamnet_fingerprint")
    if fp and fp != backend.fingerprint:
        print("warning: head was trained with a different YAMNet model file")

    cache = EmbeddingCache(backend, cache_dir)
    labels = scorer.labels
    probs_list, durs = [], []
    for r in eval_recs:
        d = cache.get(r)
        probs_list.append(scorer.score(d["emb"], d["scores"]))
        durs.append(d["duration_s"])

    items = items_for(eval_recs, probs_list, durs, scorer.params, labels)
    res = evaluate(items, labels)

    by_room = defaultdict(list)
    for it in items:
        by_room[it["room_id"] or "(unknown)"].append(it)
    rooms = {room: {k: v for k, v in evaluate(its, labels, keep_details=False).items()
                    if k in ("hours", "files", "per_label", "any_label")}
             for room, its in sorted(by_room.items())}

    curves = {}
    if sweep:
        for k, lab in enumerate(labels):
            rows = []
            for th in np.round(np.arange(0.05, 0.951, 0.05), 3):
                prm = {lab: EventParams.for_label(lab, float(th), **{
                    f: getattr(scorer.params[lab], f) for f in ("smooth_frames", "min_frames", "release_frames")})}
                its = items_for(eval_recs, [p[:, [k]] for p in probs_list], durs, prm, [lab])
                r = evaluate(its, [lab], keep_details=False)["per_label"][lab]
                rows.append({"threshold": float(th), "recall": r["recall"],
                             "fa_per_hour": r["false_alerts_per_hour"],
                             "median_delay_s": r["detection_delay"]["median_s"]})
            curves[lab] = rows

    report = {
        "note": DISCLAIMER,
        "split": split, "scorer": scorer.kind,
        "hours": res["hours"], "files": res["files"],
        "event_params": {l: scorer.params[l].to_dict() for l in labels},
        "per_label": res["per_label"], "any_label": res["any_label"],
        "frame_average_precision": _frame_ap(eval_recs, probs_list, labels),
        "by_room": rooms,
        "leakage_vs_training": leaks,
        "operating_curves": curves,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(report, out / "report.json")
    _write_csv(res["gt_rows"], out / "ground_truth_events.csv")
    _write_csv(res["fa_rows"], out / "false_alerts.csv")
    _write_csv([{"file": it["file"], **e.to_dict()} for it in items for e in it["det"]],
               out / "detections.csv")
    (out / "report.md").write_text(render_markdown(report))
    return report


def _fmt(v, nd=3):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render_markdown(rep: dict) -> str:
    L = [f"# Evaluation: split '{rep['split']}' ({rep['scorer']})", "",
         f"_{rep['note']}_", "",
         f"Audio evaluated: **{rep['hours']:.2f} h** in {rep['files']} files.", ""]
    if rep["hours"] < 5:
        L += ["> False alerts per hour from less than ~5 h of audio are imprecise; see the "
              "95% intervals. Add long realistic background recordings to the test set.", ""]
    L += ["| label | events | recall (95% CI) | false alerts | FA/h (95% CI) | delay median / p90 / max (s) | frame AP |",
          "|---|---|---|---|---|---|---|"]
    rows = list(rep["per_label"].items()) + [("**any label**", rep["any_label"])]
    for lab, m in rows:
        ci, fci, d = m["recall_ci95"], m["false_alerts_per_hour_ci95"], m["detection_delay"]
        ap = rep["frame_average_precision"].get(lab)
        L.append(f"| {lab} | {m['gt_events']} | {_fmt(m['recall'])} ({_fmt(ci[0], 2)}–{_fmt(ci[1], 2)}) "
                 f"| {m['false_alerts']} | {_fmt(m['false_alerts_per_hour'], 2)} ({_fmt(fci[0], 2)}–{_fmt(fci[1], 2)}) "
                 f"| {_fmt(d['median_s'], 2)} / {_fmt(d['p90_s'], 2)} / {_fmt(d['max_s'], 2)} | {_fmt(ap)} |")
    L += ["", "Delay = time from sound onset until the event is raised, including the "
          "~0.98 s YAMNet window; live runs add compute latency (logged separately).", "",
          "## By room", "", "| room | hours | any-label recall | any-label FA/h |", "|---|---|---|---|"]
    for room, r in rep["by_room"].items():
        L.append(f"| {room} | {r['hours']:.2f} | {_fmt(r['any_label']['recall'])} | "
                 f"{_fmt(r['any_label']['false_alerts_per_hour'], 2)} |")
    lk = rep["leakage_vs_training"]
    if any(lk.values()):
        L += ["", "## Leakage warning", "",
              "Some evaluation data shares identities with training/validation data: "
              + "; ".join(f"{k}: {len(v)}" for k, v in lk.items() if v)]
    return "\n".join(L) + "\n"
