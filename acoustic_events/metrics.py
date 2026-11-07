"""Event-level metrics: event recall, false alerts per hour, detection delay.

Matching (per label): a detection matches a ground-truth event when the
detection's extent overlaps the ground truth widened by the collars.
  * recall      = ground-truth events matched by >= 1 detection / all events
  * false alert = a detection matching no ground-truth event of its label
                  (extra detections on an already-found event are counted as
                  'duplicates', not false alerts)
  * delay       = earliest matching detection's trigger_s - event onset, i.e.
                  how long after the sound began a live system would alert
An 'any' view treats all labels as one alert stream (any target event found by
any label's detection; alerts overlapping no target event at all).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm

from .config import COLLAR_AFTER_S, COLLAR_BEFORE_S


def wilson(k: int, n: int, conf=0.95):
    if n == 0:
        return (None, None)
    z = norm.ppf(0.5 + conf / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def poisson_rate_ci(k: int, hours: float, conf=0.95):
    if hours <= 0:
        return (None, None)
    a = 1 - conf
    lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2
    hi = chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return (round(lo / hours, 3), round(hi / hours, 3))


def _overlaps(d, on, off, cb, ca):
    return d.onset_s < off + ca and d.offset_s > on - cb


def match(gt: list[tuple[float, float]], dets: list, cb=COLLAR_BEFORE_S, ca=COLLAR_AFTER_S) -> dict:
    found = [False] * len(gt)
    delays: list[float | None] = [None] * len(gt)
    det_used = [False] * len(dets)
    for gi, (on, off) in enumerate(gt):
        for di, d in enumerate(dets):
            if _overlaps(d, on, off, cb, ca):
                dl = d.trigger_s - on
                if not found[gi] or dl < delays[gi]:
                    delays[gi] = dl
                found[gi] = True
                det_used[di] = True
    fa = [d for d, u in zip(dets, det_used) if not u]
    dup = sum(det_used) - sum(found)
    return {"found": found, "delays": delays, "false_alerts": fa, "duplicates": max(0, dup)}


def _delay_stats(delays: list[float]) -> dict:
    if not delays:
        return {"median_s": None, "p90_s": None, "max_s": None, "mean_s": None}
    a = np.asarray(delays)
    return {"median_s": round(float(np.median(a)), 3), "p90_s": round(float(np.percentile(a, 90)), 3),
            "max_s": round(float(a.max()), 3), "mean_s": round(float(a.mean()), 3)}


def evaluate(items: list[dict], labels: list[str], cb=COLLAR_BEFORE_S, ca=COLLAR_AFTER_S,
             keep_details: bool = True) -> dict:
    """items: [{file, duration_s, gt: {label: [(on, off)]}, det: [Event], room_id?}]."""
    hours = sum(it["duration_s"] for it in items) / 3600
    per_label, gt_rows, fa_rows = {}, [], []
    for lab in labels:
        n_gt = n_found = n_fa = n_dup = n_cross = 0
        delays = []
        for it in items:
            gt = it["gt"].get(lab, [])
            dets = [d for d in it["det"] if d.label == lab]
            m = match(gt, dets, cb, ca)
            n_gt += len(gt)
            n_found += sum(m["found"])
            n_fa += len(m["false_alerts"])
            n_dup += m["duplicates"]
            delays += [dl for dl, f in zip(m["delays"], m["found"]) if f]
            other = [ev for ol, evs in it["gt"].items() if ol != lab for ev in evs]
            for d in m["false_alerts"]:
                cross = any(_overlaps(d, on, off, cb, ca) for on, off in other)
                n_cross += cross
                if keep_details:
                    fa_rows.append({"file": it["file"], **d.to_dict(), "overlaps_other_label": cross})
            if keep_details:
                for (on, off), f, dl in zip(gt, m["found"], m["delays"]):
                    gt_rows.append({"file": it["file"], "label": lab, "onset_s": on, "offset_s": off,
                                    "detected": f, "delay_s": None if dl is None else round(dl, 3),
                                    "room_id": it.get("room_id", "")})
        per_label[lab] = {
            "gt_events": n_gt, "detected": n_found,
            "recall": round(n_found / n_gt, 4) if n_gt else None,
            "recall_ci95": wilson(n_found, n_gt),
            "false_alerts": n_fa,
            "false_alerts_per_hour": round(n_fa / hours, 3) if hours else None,
            "false_alerts_per_hour_ci95": poisson_rate_ci(n_fa, hours),
            "false_alerts_on_other_label_events": n_cross,
            "duplicate_detections": n_dup,
            "detection_delay": _delay_stats(delays),
        }
    # class-agnostic alert stream
    n_gt = n_found = n_fa = 0
    delays = []
    for it in items:
        gt = [ev for evs in it["gt"].items() for ev in evs[1] if evs[0] in labels]
        m = match(gt, it["det"], cb, ca)
        n_gt += len(gt)
        n_found += sum(m["found"])
        n_fa += len(m["false_alerts"])
        delays += [dl for dl, f in zip(m["delays"], m["found"]) if f]
    any_view = {"gt_events": n_gt, "detected": n_found,
                "recall": round(n_found / n_gt, 4) if n_gt else None,
                "recall_ci95": wilson(n_found, n_gt),
                "false_alerts": n_fa,
                "false_alerts_per_hour": round(n_fa / hours, 3) if hours else None,
                "false_alerts_per_hour_ci95": poisson_rate_ci(n_fa, hours),
                "detection_delay": _delay_stats(delays)}
    return {"hours": round(hours, 4), "files": len(items), "per_label": per_label,
            "any_label": any_view, "gt_rows": gt_rows, "fa_rows": fa_rows}
