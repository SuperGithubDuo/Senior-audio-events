"""Transfer learning: a small classifier head on frozen YAMNet embeddings.

* trained on the 'train' split only (optionally with gain/noise augmentation
  that mixes in other *training* files),
* thresholds tuned per label on the 'val' split to maximise event recall
  subject to a false-alert-per-hour budget,
* the 'test' split is never read here.
"""

from __future__ import annotations

import numpy as np

from .config import HEAD_DEFAULT_THRESHOLD, LABELS
from .dataset import EmbeddingCache, Recording, augment, frame_labels, manifest
from .events import EventParams, run_engine
from .metrics import evaluate
from .scorers import HeadScorer, stack_context


def _collect(recs, cache, context, aug_copies=0, seed=0):
    rng = np.random.default_rng(seed)
    noise_pool = []
    if aug_copies:
        from .audio import load_wav
        noise_pool = [load_wav(r.path) for r in recs
                      if not any(lab in LABELS for _, _, lab in r.events)][:50]
    X, Y, M, G = [], [], [], []
    for gi, r in enumerate(recs):
        variants = [("", None)]
        for c in range(aug_copies):
            sub = np.random.default_rng(rng.integers(1 << 31))
            variants.append((f"aug{seed}-{c}", lambda x, s=sub: augment(x, s, noise_pool)))
        for tag, fn in variants:
            d = cache.get(r, tag, fn)
            y, m = frame_labels(len(d["emb"]), r)
            X.append(stack_context(d["emb"], context))
            Y.append(y)
            M.append(m)
            G.append(np.full(len(y), gi))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(M), np.concatenate(G)


def fit_head(X, Y, M, kind="logreg", C=0.5, hidden=256, seed=0):
    from sklearn.linear_model import LogisticRegression

    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-6
    Z = (X - mean) / scale
    K = Y.shape[1]
    if kind == "logreg":
        W = np.zeros((X.shape[1], K), np.float32)
        b = np.zeros(K, np.float32)
        for k in range(K):
            use = M[:, k] > 0
            y = Y[use, k]
            if y.sum() == 0 or y.sum() == len(y):
                raise ValueError(f"label '{LABELS[k]}' needs positive and negative training frames")
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
            clf.fit(Z[use], y)
            W[:, k], b[k] = clf.coef_[0], clf.intercept_[0]
        return mean, scale, [(W, b)]
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        use = M.min(axis=1) > 0
        Zu, Yu = Z[use], Y[use]
        pos = np.where(Yu.max(axis=1) > 0)[0]
        neg = np.where(Yu.max(axis=1) == 0)[0]
        rng = np.random.default_rng(seed)
        reps = max(1, int(0.5 * len(neg) / max(1, len(pos))))   # ~1:2 pos:neg
        idx = np.concatenate([neg, np.tile(pos, reps)])
        rng.shuffle(idx)
        clf = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=1e-3, early_stopping=True,
                            max_iter=300, random_state=seed)
        clf.fit(Zu[idx], Yu[idx].astype(int))
        layers = list(zip(clf.coefs_, clf.intercepts_))
        return mean, scale, layers
    raise ValueError(f"unknown head kind {kind}")


def items_for(recs: list[Recording], probs_list, durations, params, labels=LABELS):
    items = []
    for r, probs, dur in zip(recs, probs_list, durations):
        items.append({"file": r.file, "duration_s": dur, "room_id": r.room_id,
                      "gt": {l: r.target_events(l) for l in labels},
                      "det": run_engine(probs, list(labels), params)})
    return items


def tune_thresholds(head: HeadScorer, val_recs, cache, target_fa_per_hour: float, labels=LABELS):
    """Per label: highest recall with FA/h <= target on validation (ties -> fewer FAs)."""
    probs_list, durs = [], []
    for r in val_recs:
        d = cache.get(r)
        probs_list.append(head.score(d["emb"]))
        durs.append(d["duration_s"])
    hours = sum(durs) / 3600
    grid = np.round(np.arange(0.05, 0.951, 0.025), 3)
    chosen, curves = {}, {}
    for k, lab in enumerate(labels):
        rows = []
        for th in grid:
            prm = {lab: EventParams.for_label(lab, float(th))}
            items = items_for(val_recs, [p[:, [k]] for p in probs_list], durs, prm, [lab])
            res = evaluate(items, [lab], keep_details=False)["per_label"][lab]
            rows.append((float(th), res["recall"], res["false_alerts_per_hour"]))
        curves[lab] = rows
        n_pos = sum(len(r.target_events(lab)) for r in val_recs)
        if n_pos == 0:
            print(f"warning: no '{lab}' events in val; using default threshold")
            chosen[lab] = HEAD_DEFAULT_THRESHOLD
            continue
        ok = [r for r in rows if r[2] is not None and r[2] <= target_fa_per_hour]
        if ok:
            best = max(ok, key=lambda r: (r[1], -r[2], r[0]))
        else:
            best = min(rows, key=lambda r: (r[2], -r[1]))
            print(f"warning: '{lab}' cannot meet {target_fa_per_hour} FA/h on val; "
                  f"using lowest-FA threshold {best[0]}")
        chosen[lab] = best[0]
    head.params = {l: EventParams.for_label(l, chosen[l]) for l in labels}
    return chosen, curves, hours


def train(recs: list[Recording], backend, out_dir: str, kind="logreg", context=1, C=0.5,
          target_fa_per_hour=0.5, aug_copies=0, seed=0, cache_dir=".cache/embeddings"):
    train_recs = [r for r in recs if r.split == "train"]
    val_recs = [r for r in recs if r.split == "val"]
    if not train_recs or not val_recs:
        raise ValueError("need files in both 'train' and 'val' splits (run `split` first)")
    cache = EmbeddingCache(backend, cache_dir)
    X, Y, M, _ = _collect(train_recs, cache, context, aug_copies, seed)
    print(f"train frames: {len(X)}  positives per label: "
          + ", ".join(f"{l}={int(Y[:, k].sum())}" for k, l in enumerate(LABELS)))
    mean, scale, layers = fit_head(X, Y, M, kind, C, seed=seed)
    params = {l: EventParams.for_label(l, HEAD_DEFAULT_THRESHOLD) for l in LABELS}
    head = HeadScorer(LABELS, mean, scale, layers, context, params)
    chosen, curves, val_hours = tune_thresholds(head, val_recs, cache, target_fa_per_hour)
    if val_hours < 1.0:
        print(f"warning: only {val_hours:.2f} h of validation audio; FA/h tuning is noisy. "
              "Add long realistic background recordings (TV, cooking, conversation).")
    head.meta = {
        "head_kind": kind, "C": C, "augment_copies": aug_copies, "seed": seed,
        "yamnet_fingerprint": backend.fingerprint,
        "target_fa_per_hour": target_fa_per_hour,
        "val_hours": round(val_hours, 3),
        "thresholds": chosen,
        "val_curves": {l: [{"threshold": t, "recall": r, "fa_per_hour": f} for t, r, f in rows]
                       for l, rows in curves.items()},
        "train_files": manifest(train_recs),
        "val_files": manifest(val_recs),
    }
    head.save(out_dir)
    return head
