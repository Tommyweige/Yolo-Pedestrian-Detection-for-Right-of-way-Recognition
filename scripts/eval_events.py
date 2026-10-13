"""Event-level precision/recall/F1 for yield_rules output against human event labels.

Ground truth is a .jsonl of {"kind": "suspected_failure_to_yield"|"yielded", "time_s": <float>}.
A prediction matches a label of the same kind within --tolerance seconds; matching is one-to-one
(closest first). Clips with no labelled event must still be listed via --clips so false alarms count.
This is the EVENT level only; detection and tracking quality are reported separately.
"""
import argparse
import json
import sys
from pathlib import Path


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def match(predicted, truth, tolerance):
    pairs = sorted(((abs(p["time_s"] - t["time_s"]), i, j) for i, p in enumerate(predicted)
                    for j, t in enumerate(truth) if p["kind"] == t["kind"] and abs(p["time_s"] - t["time_s"]) <= tolerance))
    used_p, used_t, hits = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i), used_t.add(j)
            hits += 1
    return hits


def score(predicted, truth, tolerance):
    result = {}
    for kind in sorted({e["kind"] for e in predicted} | {e["kind"] for e in truth}):
        p = [e for e in predicted if e["kind"] == kind]
        t = [e for e in truth if e["kind"] == kind]
        tp = match(p, t, tolerance)
        precision = tp / len(p) if p else None
        recall = tp / len(t) if t else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if p and t else None)
        result[kind] = {"predicted": len(p), "labelled": len(t), "true_positive": tp,
                        "false_alarm": len(p) - tp, "miss": len(t) - tp,
                        "precision": precision, "recall": recall, "f1": f1}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predicted", required=True, help="events.jsonl from yield_rules.py")
    parser.add_argument("--truth", required=True, help="human labels .jsonl")
    parser.add_argument("--tolerance", type=float, default=1.0, help="seconds (default 1.0)")
    args = parser.parse_args(argv)
    truth = load(args.truth)
    if not truth:
        print("No labelled events: precision/recall are undefined without human ground truth.", file=sys.stderr)
        return 2
    print(json.dumps(score(load(args.predicted), truth, args.tolerance), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
