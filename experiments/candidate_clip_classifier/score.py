"""Turn out-of-fold scores into events and score them with the rule route's scorer (eval_rule_events, 1.25 s).

Events of one clip: its candidates at or above that fold's threshold (chosen without the clip), one per vehicle (the
highest-scoring, as the rule reports a vehicle once). Also reported, for every model:
- candidate-level AUROC / AP over candidates with a known label (out-of-fold),
- human verdicts of the emitted events: an event is the judged event when it is the same vehicle within the tolerance;
  events no human has judged are counted as unjudged, not as correct,
- "@28": the 28 highest-scoring candidates over all clips, one per vehicle, i.e. the rule's event budget.
Usage: python score.py --data DIR --oof DIR --out DIR
"""
import argparse
import contextlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, RUN2, TOLERANCE_S, VIDEOS, load_verdicts  # noqa: E402
import eval_rule_events as er  # noqa: E402  (on sys.path through common)

BUDGET = 28


def one_per_vehicle(rows):
    best = {}
    for r in rows:
        if r["vehicle_id"] not in best or r["p"] > best[r["vehicle_id"]]["p"]:
            best[r["vehicle_id"]] = r
    return sorted(best.values(), key=lambda r: r["time_s"])


def verdict_of(event, verdicts):
    for v in verdicts:
        if v["stem"] == event["stem"] and v["vehicle"] == event["vehicle_id"] and abs(v["time_s"] - event["time_s"]) <= TOLERANCE_S:
            return v["verdict"]
    return "unjudged"


def benchmark(events, root):
    """eval_rule_events totals for events {stem: [event]} written as <stem>.jsonl under root."""
    root.mkdir(parents=True, exist_ok=True)
    fields = ("kind", "frame", "time_s", "vehicle_id", "vehicle_kind", "pedestrian_id", "pedestrian_state",
              "pedestrian_entered_crosswalk", "contact_frames", "p")
    for stem, rows in events.items():
        (root / f"{stem}.jsonl").write_text("".join(json.dumps({k: r[k] for k in fields}, ensure_ascii=False) + "\n"
                                                    for r in rows), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        er.main(["--videos", str(VIDEOS), "--masks-root", str(RUN2), "--tracks-root", str(RUN2), "--labels", str(LABELS),
                 "--tolerance", str(TOLERANCE_S), "--events-root", str(root), "--json", str(root / "eval.json")])
    return json.loads((root / "eval.json").read_text(encoding="utf-8"))["totals"]


def row(name, events, totals, verdicts, auc=None, ap=None):
    flat = [e for rows in events.values() for e in rows]
    judged = Counter(verdict_of(e, verdicts) for e in flat)
    m, c = totals["by_source"]["manual"], totals["by_source"]["confirmed_candidate"]
    return {"model": name, "events": totals["events"], "manual": f"{m['hit']}/{m['labelled']}",
            "confirmed": f"{c['hit']}/{c['labelled']}", "incidents": f"{totals['incident_hit']}/{totals['incidents']}",
            "label_precision": round(totals["precision"], 3) if totals["precision"] is not None else None,
            "A": judged["A"], "B": judged["B"], "C": judged["C"], "D": judged["D"], "unjudged": judged["unjudged"],
            "human_precision_A": round(judged["A"] / len(flat), 3) if flat else None,
            "auroc": auc, "ap": ap}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    samples = [s for s in json.loads((Path(args.data) / "samples.json").read_text(encoding="utf-8"))
               if s["role"] == "candidate"]
    verdicts = load_verdicts()
    out = Path(args.out)
    table = []
    everything = {}
    for s in samples:
        everything.setdefault(s["stem"], []).append({**s, "p": 1.0})
    table.append(row("generator (all candidates)", everything, benchmark(everything, out / "generator"), verdicts))
    for path in sorted(Path(args.oof).glob("oof_*.json")):
        name = path.stem[4:]
        oof = json.loads(path.read_text(encoding="utf-8"))
        scored = [{**s, "p": oof["scores"][s["id"]]} for s in samples]
        known = [r for r in scored if r["y"] is not None]
        y, p = np.array([r["y"] for r in known]), np.array([r["p"] for r in known])
        auc, ap = round(roc_auc_score(y, p), 3), round(average_precision_score(y, p), 3)
        events = {}
        for stem in sorted({r["stem"] for r in scored}):
            events[stem] = one_per_vehicle([r for r in scored if r["stem"] == stem and r["p"] >= oof["thresholds"][stem]])
        table.append(row(f"{name} (fold threshold)", events, benchmark(events, out / name), verdicts, auc, ap))
        top = sorted(one_per_vehicle_by_stem(scored), key=lambda r: -r["p"])[:BUDGET]
        budget = {stem: sorted([r for r in top if r["stem"] == stem], key=lambda r: r["time_s"])
                  for stem in sorted({r["stem"] for r in scored})}
        table.append(row(f"{name} (@{BUDGET})", budget, benchmark(budget, out / f"{name}@{BUDGET}"), verdicts, auc, ap))
        print(json.dumps(table[-2], ensure_ascii=False), flush=True)
    (out / "table.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    keys = list(table[0])
    print("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys))
    for r in table:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")
    return 0


def one_per_vehicle_by_stem(rows):
    out = []
    for stem in {r["stem"] for r in rows}:
        out += one_per_vehicle([r for r in rows if r["stem"] == stem])
    return out


if __name__ == "__main__":
    sys.exit(main())
