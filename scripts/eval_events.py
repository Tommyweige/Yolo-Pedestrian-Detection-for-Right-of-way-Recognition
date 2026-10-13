"""Event-level precision/recall/F1 of detected events against human event labels, over one or more videos.

Labels are .jsonl rows {"kind": "suspected_failure_to_yield" | "yielded", "time_s": <float>} (from label_events.py).
A prediction matches a label of the same kind within --tolerance seconds; matching is one-to-one (closest first).
`vehicle_on_crosswalk_with_pedestrian` (yield_image_rule.py) is scored as `suspected_failure_to_yield`.
A video with NO events must be declared with an empty labels file plus --allow-empty-truth, so its false alarms are
counted; without that flag an empty labels file is refused (it usually means labels are missing).
This is the EVENT level only; detection and tracking quality are reported separately.
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


def match(predicted, truth, tolerance):
    pairs = sorted(((abs(p["time_s"] - t["time_s"]), i, j) for i, p in enumerate(predicted)
                    for j, t in enumerate(truth) if p["kind"] == t["kind"] and abs(p["time_s"] - t["time_s"]) <= tolerance))
    used_p, used_t, hits = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i), used_t.add(j)
            hits += 1
    return hits


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
