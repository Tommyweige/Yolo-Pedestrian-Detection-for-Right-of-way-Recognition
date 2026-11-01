"""Event-level precision/recall/F1 of detected events against human event labels, over one or more videos.

Labels are .jsonl rows {"kind": "suspected_failure_to_yield" | "yielded", "time_s": <float>} (from label_events.py).
A prediction matches a label of the same kind within --tolerance seconds; matching is one-to-one (closest first).
`vehicle_on_crosswalk_with_pedestrian` (yield_image_rule.py) is scored as `suspected_failure_to_yield`.
A video with NO events must be declared with an empty labels file plus --allow-empty-truth, so its false alarms are
counted; without that flag an empty labels file is refused (it usually means labels are missing).
This is the EVENT level only; detection and tracking quality are reported separately.

Fixed evaluation rules (ADR 0003, used by scripts/eval_rule_events.py):
- The tolerance is derived from the labels-vs-predictions residual distribution (`derive_tolerance`), not chosen by hand.
- An event is one vehicle pass; a label inside a pair of labels closer than the tolerance belongs to the same incident
  (`incident_labels`), so recall is reported both per label and per incident.
- A second prediction for an incident already hit is a false alarm, but is counted separately as a duplicate.
"""
import argparse
import json
import sys
from pathlib import Path

ALIASES = {"vehicle_on_crosswalk_with_pedestrian": "suspected_failure_to_yield"}


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row["kind"] = ALIASES.get(row["kind"], row["kind"])
    return rows


def matched_pairs(predicted, truth, tolerance):
    """[(prediction index, label index)] matched one-to-one, closest first."""
    candidates = sorted((abs(p["time_s"] - t["time_s"]), i, j) for i, p in enumerate(predicted)
                        for j, t in enumerate(truth) if p["kind"] == t["kind"] and abs(p["time_s"] - t["time_s"]) <= tolerance)
    used_p, used_t, out = set(), set(), []
    for _, i, j in candidates:
        if i not in used_p and j not in used_t:
            used_p.add(i), used_t.add(j)
            out.append((i, j))
    return out


def match(predicted, truth, tolerance):
    return len(matched_pairs(predicted, truth, tolerance))


def residuals(predicted, truth, search_s=5.0):
    """Signed (prediction - label) seconds for each label's nearest same-kind prediction, within `search_s`.

    This is the distribution the tolerance is derived from: it is the only place the human label time and the rule's
    event time meet. A label with no prediction in range gets None (a miss at any tolerance up to `search_s`).
    """
    out = []
    for t in truth:
        same = [p for p in predicted if p["kind"] == t["kind"]]
        if not same:
            out.append({"label": t, "predicted": None, "delta_s": None})
            continue
        nearest = min(same, key=lambda p: abs(p["time_s"] - t["time_s"]))
        delta = nearest["time_s"] - t["time_s"]
        out.append({"label": t, "predicted": nearest["time_s"],
                    "delta_s": delta if abs(delta) <= search_s else None})
    return out


def derive_tolerance(magnitudes, min_gap_s=0.5, min_ratio=2.0, resolution_s=0.05, fallback_s=1.0):
    """Tolerance, in seconds, from the sorted |label - prediction| residuals: the edge of the dense cluster.

    Two timestamps refer to one incident when the distance comes from label placement and from the rule's event-time
    definition, i.e. when it sits in the same cluster as every other match. Beyond that cluster are pairs that only
    happen to be nearest to each other (a different vehicle's event), and a tolerance that reaches them inflates
    recall without any detection gain. So: walk the sorted magnitudes and stop at the first gap that is both
    `min_gap_s` wide and at least `min_ratio` times the previous value; the tolerance is that gap's midpoint.
    With no such gap (too few labels, or a smooth spread) the labeler's documented +-1 s is used and reported as such.
    """
    values = sorted(m for m in magnitudes if m is not None)
    for a, b in zip(values, values[1:]):
        if a > 0 and b - a >= min_gap_s and b / a >= min_ratio:
            midpoint = (a + b) / 2
            return round(-(-midpoint // resolution_s) * resolution_s, 3), {
                "method": "cluster_edge", "below_s": a, "above_s": b, "gap_s": round(b - a, 3),
                "ratio": round(b / a, 2), "sorted": values}
    return fallback_s, {"method": "fallback_labeler_resolution", "sorted": values}


def incident_labels(truth, gap_s):
    """Group label indices whose times are within `gap_s`: one incident needs one prediction, not one per label.

    Two labels closer than the tolerance are not two independent requirements (the labeller marked two moments of the
    same pass, or two candidates for it); counting them separately would make recall depend on how often someone pressed
    the key. The per-label numbers stay the headline; this is the second view.
    """
    groups = []
    for i in sorted(range(len(truth)), key=lambda i: truth[i]["time_s"]):
        if groups and truth[i]["time_s"] - truth[groups[-1][-1]]["time_s"] <= gap_s:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def counts(predicted, truth, tolerance):
    result = {}
    for kind in sorted({e["kind"] for e in predicted} | {e["kind"] for e in truth}):
        p = [e for e in predicted if e["kind"] == kind]
        t = [e for e in truth if e["kind"] == kind]
        tp = match(p, t, tolerance)
        result[kind] = {"predicted": len(p), "labelled": len(t), "true_positive": tp,
                        "false_alarm": len(p) - tp, "miss": len(t) - tp}
    return result


def with_rates(per_kind):
    out = {}
    for kind, c in per_kind.items():
        precision = c["true_positive"] / c["predicted"] if c["predicted"] else None
        recall = c["true_positive"] / c["labelled"] if c["labelled"] else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if c["predicted"] and c["labelled"] else None)
        out[kind] = {**c, "precision": precision, "recall": recall, "f1": f1}
    return out


def score(predicted, truth, tolerance):
    return with_rates(counts(predicted, truth, tolerance))


def aggregate(pairs, tolerance):
    """pairs: [(predicted, truth)] -> per-kind totals summed over videos (micro average)."""
    total = {}
    for predicted, truth in pairs:
        for kind, c in counts(predicted, truth, tolerance).items():
            t = total.setdefault(kind, {k: 0 for k in c})
            for k, v in c.items():
                t[k] += v
    return with_rates(total)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("EVENTS.jsonl", "LABELS.jsonl"),
                        help="one video's detected events and its human labels; repeat per video")
    parser.add_argument("--predicted", help="shorthand for a single --pair (with --truth)")
    parser.add_argument("--truth", help="shorthand for a single --pair (with --predicted)")
    parser.add_argument("--tolerance", type=float, default=1.0, help="seconds (default 1.0)")
    parser.add_argument("--allow-empty-truth", action="store_true",
                        help="a labels file with no rows means 'a human reviewed this video and it has no events'")
    args = parser.parse_args(argv)
    pairs = [list(p) for p in args.pair]
    if args.predicted or args.truth:
        if not (args.predicted and args.truth):
            parser.error("--predicted and --truth go together")
        pairs.append([args.predicted, args.truth])
    if not pairs:
        parser.error("give at least one --pair (or --predicted with --truth)")
    loaded, per_video = [], []
    for events_path, labels_path in pairs:
        predicted, truth = load(events_path), load(labels_path)
        if not truth and not args.allow_empty_truth:
            print(f"{labels_path} has no labelled events. If this video really has none, pass --allow-empty-truth; "
                  "otherwise the labels are missing and precision/recall are undefined.", file=sys.stderr)
            return 2
        loaded.append((predicted, truth))
        per_video.append({"events": events_path, "labels": labels_path, "result": score(predicted, truth, args.tolerance)})
    print(json.dumps({"tolerance_s": args.tolerance, "videos": len(loaded), "overall": aggregate(loaded, args.tolerance),
                      "per_video": per_video}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
