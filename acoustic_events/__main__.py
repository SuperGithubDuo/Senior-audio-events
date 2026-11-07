"""Command line: python -m acoustic_events <command> ..."""

from __future__ import annotations

import argparse
import json
import sys

from .config import LABELS


def _backend(args):
    from .backend import YamnetBackend
    return YamnetBackend(args.yamnet)


def _scorer(args, backend):
    from .events import EventParams
    from .scorers import load_scorer

    sc = load_scorer(args.model, backend.class_names)
    for spec in args.threshold or []:
        lab, val = spec.split("=")
        if lab not in sc.params:
            raise SystemExit(f"unknown label in --threshold: {lab}")
        p = sc.params[lab].to_dict()
        p.update(on_threshold=float(val), off_threshold=None)
        sc.params[lab] = EventParams(**p)
    return sc


def cmd_fetch(args):
    from .backend import fetch_model
    fetch_model(args.out, args.from_archive)


def cmd_detect(args):
    from .detect import detect_file, export_events, export_frame_scores, run_info

    backend = _backend(args)
    scorer = _scorer(args, backend)
    events, probs, dur = detect_file(args.wav, backend, scorer)
    info = run_info(args.wav, dur, backend, scorer, scorer.params)
    export_events(events, args.out, info)
    if args.frame_scores:
        export_frame_scores(probs, scorer.labels, args.frame_scores)
    print(f"{len(events)} event(s) in {dur:.1f} s -> {args.out}")
    for e in events:
        print(f"  {e.label:<14} {e.onset_s:7.2f}-{e.offset_s:7.2f}s  peak={e.peak_score:.2f}  "
              f"raised at {e.trigger_s:.2f}s")


def cmd_split(args):
    from .audio import read_audio
    from .dataset import load_dataset
    from .split import make_split, write_splits

    recs = load_dataset(args.metadata, args.annotations, args.audio_root)
    durs = {}
    for r in recs:
        try:
            import soundfile as sf
            durs[r.file] = sf.info(str(r.path)).duration
        except Exception:
            data, sr = read_audio(r.path)
            durs[r.file] = len(data) / sr
    fr = tuple(float(x) for x in args.fractions.split(","))
    split_of, report = make_split(recs, durs, fr, args.seed)
    write_splits(recs, split_of, args.out)
    print(json.dumps({k: report[k] for k in ("enforced_separation", "splits", "warnings")}, indent=2))
    if args.report:
        from .dataset import dump_json
        dump_json(report, args.report)


def cmd_train(args):
    from .dataset import load_dataset
    from .train import train

    recs = load_dataset(args.metadata, args.annotations, args.audio_root, args.splits)
    head = train(recs, _backend(args), args.out, args.head, args.context, args.C,
                 args.target_fa_per_hour, args.augment, args.seed, args.cache)
    print(f"saved head to {args.out}; validation-tuned thresholds: {head.meta['thresholds']}")


def cmd_evaluate(args):
    from .dataset import load_dataset
    from .evaluate import run_evaluation

    recs = load_dataset(args.metadata, args.annotations, args.audio_root, args.splits)
    backend = _backend(args)
    rep = run_evaluation(recs, backend, _scorer(args, backend), args.split, args.out,
                         args.allow_leakage, not args.no_sweep, args.cache)
    print(open(f"{args.out}/report.md").read())


def cmd_live(args):
    from .live import run_microphone, simulate

    backend = _backend(args)
    scorer = _scorer(args, backend)
    if args.simulate:
        simulate(args.simulate, backend, scorer, args.log, args.realtime)
    else:
        run_microphone(backend, scorer, args.device, args.log, args.save_clips)


def cmd_export_web(args):
    from .webexport import export_web
    export_web(args.model, args.out)
    print(f"wrote {args.out}; load it in the test page under 'Classifier'")


def cmd_devices(args):
    import sounddevice as sd
    print(sd.query_devices())


def main(argv=None):
    p = argparse.ArgumentParser(prog="acoustic_events", description=(
        f"Local YAMNet-based detection of acoustic events: {', '.join(LABELS)}. "
        "Detections are acoustic events, not confirmed emergencies or falls."))
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, model=True):
        sp.add_argument("--yamnet", help="local YAMNet SavedModel dir (default models/yamnet or $YAMNET_DIR)")
        if model:
            sp.add_argument("--model", default="baseline",
                            help="trained head dir, or 'baseline' for pretrained YAMNet classes")
            sp.add_argument("--threshold", action="append", metavar="LABEL=VALUE",
                            help="override an on-threshold, e.g. --threshold impact=0.6")

    def data(sp, splits=True):
        sp.add_argument("--metadata", required=True)
        sp.add_argument("--annotations", required=True)
        sp.add_argument("--audio-root")
        if splits:
            sp.add_argument("--splits", required=True, help="CSV from the split command")
            sp.add_argument("--cache", default=".cache/embeddings")

    s = sub.add_parser("fetch-model", help="one-time: place YAMNet locally")
    s.add_argument("--out", default="models/yamnet")
    s.add_argument("--from-archive", help="tar.gz downloaded elsewhere (offline install)")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("detect", help="WAV file -> events CSV/JSON")
    s.add_argument("wav")
    s.add_argument("--out", default="events.csv", help=".csv or .json")
    s.add_argument("--frame-scores", help="optional CSV of per-frame probabilities")
    common(s)
    s.set_defaults(fn=cmd_detect)

    s = sub.add_parser("split", help="group-aware train/val/test split")
    data(s, splits=False)
    s.add_argument("--out", default="splits.csv")
    s.add_argument("--report", default="splits_report.json")
    s.add_argument("--fractions", default="0.6,0.2,0.2")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("train", help="train a head on YAMNet embeddings")
    data(s)
    common(s, model=False)
    s.add_argument("--out", default="models/head")
    s.add_argument("--head", choices=["logreg", "mlp"], default="logreg")
    s.add_argument("--context", type=int, default=1, help="past frames stacked as context")
    s.add_argument("--C", type=float, default=0.5, help="logreg inverse regularisation")
    s.add_argument("--target-fa-per-hour", type=float, default=0.5,
                   help="per-label false-alert budget used to pick thresholds on val")
    s.add_argument("--augment", type=int, default=0, help="augmented copies per training file")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="event recall, false alerts/h, delay on unseen files")
    data(s)
    common(s)
    s.add_argument("--split", default="test", help="train/val/test/all")
    s.add_argument("--out", default="reports/eval")
    s.add_argument("--allow-leakage", action="store_true")
    s.add_argument("--no-sweep", action="store_true")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("live", help="microphone input (or --simulate a WAV)")
    common(s)
    s.add_argument("--device", help="input device index or name (see `devices`)")
    s.add_argument("--log", default="logs/events.jsonl")
    s.add_argument("--save-clips", help="dir to save event audio locally (off by default)")
    s.add_argument("--simulate", help="WAV to stream through the live path instead of a mic")
    s.add_argument("--realtime", action="store_true", help="with --simulate, pace at real time")
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("export-web", help="trained head -> JSON for the browser test page")
    s.add_argument("--model", required=True, help="trained head dir")
    s.add_argument("--out", default="head_web.json")
    s.set_defaults(fn=cmd_export_web)

    s = sub.add_parser("devices", help="list audio input devices")
    s.set_defaults(fn=cmd_devices)

    args = p.parse_args(argv)
    if getattr(args, "device", None) and str(args.device).isdigit():
        args.device = int(args.device)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
