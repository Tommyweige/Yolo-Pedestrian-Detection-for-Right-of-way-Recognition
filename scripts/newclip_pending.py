"""One page holding exactly the events of a run that still have no verdict.

`scripts/newclip_review.py` renders every event of a run; this narrows that page to the leftovers, so a batch spread
over two older pages (18 events on one, 10 verdict rows that attach to nothing on the other) becomes one list. The
verdicts are read with `scripts/newclip_score.py`'s own join, so "pending" here means exactly "the scorer would count
it as unjudged", and judging this page closes the gap the scorer prints as `未判`.

Each card says which batch it belongs to:

- 全新事件 — no verdict row for this event exists anywhere.
- 舊判定對不上、需要重判 — a verdict row was made for this clip and cannot be attached to any event within the 1.25 s
  tolerance, but it names a vehicle that this event's track has (or lands within `--near-s` of it). The old row is
  printed on the card with its time and verdict; the judge decides whether it still holds.

Rows that match no unjudged event at all (their incident is already judged, or the rule no longer reports anything
there) are not cards: they are listed here and in the page's README with the nearest event and the time delta, because
they are an explanation of a partial precision, not something to judge.

Reads, read-only: the run directory and the verdict files. Writes the page into `--out` (default `<run>/pending`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import newclip_review as review  # noqa: E402
import newclip_score as score  # noqa: E402

ORIGIN_NAMES = {"new": "全新事件", "rejudge": "舊判定對不上、需要重判"}
VEHICLE_WINDOW_S = 8.0      # how far a vehicle's own id may drift before the row is about something else
NEAR_S = 3.0                # how far a row with no matching vehicle may land and still be aimed at this event
NEIGHBOUR_S = 2.0           # a judged event this close is worth naming on the card (same incident, re-reported?)


def solved(index, judged):
    return judged[index]["verdict"] in score.VERDICTS


def nearest_index(time_s, events, pool):
    best, best_delta = None, None
    for index in pool:
        delta = abs(events[index]["time_s"] - time_s)
        if best_delta is None or delta < best_delta:
            best, best_delta = index, delta
    return best, best_delta


def target_of(row, events, open_idx, near_s):
    """Which unjudged event an unattachable verdict row is probably about, or None."""
    same_vehicle = [i for i in open_idx
                    if row["vehicle"] is not None and events[i]["vehicle_id"] == row["vehicle"]]
    pool = same_vehicle or open_idx
    index, delta = nearest_index(row["time_s"], events, pool)
    if index is None:
        return None
    limit = VEHICLE_WINDOW_S if same_vehicle else near_s
    return index if delta <= limit else None


def prior_row(row, delta):
    return {"time_s": round(row["time_s"], 3), "vehicle": row["vehicle"], "verdict": row["verdict"],
            "note": row.get("note") or "", "delta_s": round(delta, 3)}


def neighbour_of(event, events, judged, limit_s):
    """The closest already judged event of the same clip, with who it was about."""
    done = [i for i, row in enumerate(judged) if row["verdict"] in score.VERDICTS]
    index, delta = nearest_index(event["time_s"], events, done)
    if index is None or delta > limit_s:
        return None
    other = events[index]
    return {"time_s": round(other["time_s"], 3), "vehicle_id": other["vehicle_id"],
            "pedestrian_id": other["pedestrian_id"], "verdict": judged[index]["verdict"],
            "delta_s": round(delta, 3)}


def clip_selection(clip, rows, near_s, neighbour_s):
    events, judged = clip["events"], clip["judged"]
    open_idx = [i for i in range(len(events)) if not solved(i, judged)]
    assigned, orphans = {}, []
    for row in rows:
        index = target_of(row, events, open_idx, near_s)
        if index is None:
            nearest, delta = nearest_index(row["time_s"], events, range(len(events)))
            judged_here = nearest is not None and solved(nearest, judged)
            # A row that lands inside the tolerance on an event that already has a verdict is the old file listing
            # another vehicle of a moment it already covered; one that lands nowhere is the rule no longer reporting
            # anything there, which is a recall question and not a card.
            duplicate = judged_here and delta <= score.TOLERANCE_S
            orphans.append({"row": row,
                            "kind": "duplicate" if duplicate else "gap",
                            "nearest_time_s": None if nearest is None else events[nearest]["time_s"],
                            "nearest_delta_s": None if delta is None else round(delta, 3),
                            "nearest_judged": None if nearest is None else
                            (judged[nearest]["verdict"] if judged_here else "未判")})
            continue
        assigned.setdefault(index, []).append(
            prior_row(row, abs(events[index]["time_s"] - row["time_s"])))
    items = []
    for index in open_idx:
        event = events[index]
        items.append({
            "stem": clip["stem"], "index": index, "time_s": event["time_s"],
            "vehicle_id": event["vehicle_id"], "pedestrian_id": event["pedestrian_id"],
            "origin": "rejudge" if index in assigned else "new",
            "priors": sorted(assigned.get(index, []), key=lambda row: row["time_s"]),
            "neighbour": neighbour_of(event, events, judged, neighbour_s),
        })
    return items, orphans


def select(run, strays, near_s=NEAR_S, neighbour_s=NEIGHBOUR_S):
    """The pending events of a loaded run, plus the counts and the rows that match nothing."""
    by_stem = {}
    for row in strays:
        by_stem.setdefault(row["stem"], []).append(row)
    items, orphans = [], []
    for clip in run["clips"]:
        clip_items, clip_orphans = clip_selection(clip, by_stem.get(clip["stem"], []), near_s, neighbour_s)
        items += clip_items
        orphans += clip_orphans
    events = sum(len(clip["events"]) for clip in run["clips"])
    judged = sum(score.clip_counts(clip)["judged"] for clip in run["clips"])
    counts = {"events": events, "judged": judged, "pending": len(items),
              "new": sum(1 for item in items if item["origin"] == "new"),
              "rejudge": sum(1 for item in items if item["origin"] == "rejudge"),
              "stray_rows": len(strays), "orphan_rows": len(orphans),
              "orphan_duplicates": sum(1 for orphan in orphans if orphan["kind"] == "duplicate"),
              "orphan_gaps": sum(1 for orphan in orphans if orphan["kind"] == "gap")}
    counts["covered"] = judged + len(items)
    return {"events": items, "counts": counts, "orphans": orphans}


def print_report(run, result, near_s, neighbour_s):
    counts = result["counts"]
    per_clip = {}
    for item in result["events"]:
        entry = per_clip.setdefault(item["stem"], {"pending": 0, "new": 0, "rejudge": 0})
        entry["pending"] += 1
        entry[item["origin"]] += 1
    print(f"\n待判清單：{run['dir']}")
    print(f"  規則預設參數 · 容差 {score.TOLERANCE_S} s · 舊判定對不上時，"
          f"同車輛 id 放寬到 {VEHICLE_WINDOW_S:.0f} s、無車輛 id 只放寬到 {near_s} s")
    print(f"\n{'待判':>4}{'全新':>6}{'重判':>6}  影片")
    for clip in run["clips"]:
        entry = per_clip.get(clip["stem"], {"pending": 0, "new": 0, "rejudge": 0})
        print(f"{entry['pending']:>4}{entry['new']:>6}{entry['rejudge']:>6}  {clip['stem']}")
        for item in [i for i in result["events"] if i["stem"] == clip["stem"]]:
            tag = ORIGIN_NAMES[item["origin"]]
            extra = ""
            if item["priors"]:
                first = item["priors"][0]
                more = f"（共 {len(item['priors'])} 列）" if len(item["priors"]) > 1 else ""
                extra = (f" · 舊判定 {first['time_s']:.3f}s/車 {first['vehicle']} → {first['verdict']}{more}")
            print(f"      {item['time_s']:>7.3f}s 車 {item['vehicle_id']:>5} 行人 {item['pedestrian_id']:>5}"
                  f"  {tag}{extra}")
    print(f"{counts['pending']:>4}{counts['new']:>6}{counts['rejudge']:>6}  小計"
          f"（全部 {counts['events']} 件、已判 {counts['judged']} 件）")

    if result["orphans"]:
        on_cards = counts["stray_rows"] - counts["orphan_rows"]
        print(f"\n{counts['stray_rows']} 列舊判定對不上事件：{on_cards} 列寫在卡片上（可能是同一件，要重判），"
              f"另外 {counts['orphan_rows']} 列沒有卡片：")
        for orphan in result["orphans"]:
            row = orphan["row"]
            kind = ("同一件事的其他車輛（那件已判 {0}）".format(orphan["nearest_judged"])
                    if orphan["kind"] == "duplicate" else "規則現在沒報到這一件")
            print(f"  {row['stem']}  {row['time_s']:.3f}s 車 {row['vehicle']} → {row['verdict']}"
                  f"　最近的事件 {orphan['nearest_time_s']}s（{orphan['nearest_judged']}，"
                  f"差 {orphan['nearest_delta_s']} s）· {kind}")

    print(f"\n判完這 {counts['pending']} 件，precision 覆蓋 {counts['covered']}/{counts['events']} 件。")
    if counts["covered"] < counts["events"]:
        print(f"  仍有 {counts['events'] - counts['covered']} 件既沒有判定、也不在這份清單裡。")
    if counts["orphan_duplicates"]:
        print(f"  {counts['orphan_duplicates']} 列舊判定是同一件事的其他車輛：不影響 precision 覆蓋。")
    if counts["orphan_gaps"]:
        print(f"  {counts['orphan_gaps']} 列舊判定指不到事件：那是 recall 的缺口（規則現在沒報那一件），"
              f"不是這份清單能補的。")


def build(run_dir, out_dir, verdicts, title=None, near_s=NEAR_S, neighbour_s=NEIGHBOUR_S):
    run = score.load_run(run_dir)
    for clip in run["clips"]:
        clip["run_role"] = run["manifest"]["role"]
    rows = score.load_verdicts(verdicts, sorted(clip["stem"] for clip in run["clips"]))
    strays = score.join_verdicts(run["clips"], rows)
    result = select(run, strays, near_s, neighbour_s)
    print_report(run, result, near_s, neighbour_s)
    if not result["events"]:
        print("\n沒有待判的事件：這批的每一件都有判定了。")
        return result, None
    payload = review.build_review(run_dir, out_dir=out_dir, title=title, selection=result)
    return result, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory written by scripts/newclip_loop.py")
    parser.add_argument("--verdicts", action="append", required=True,
                        help="every verdict file that already covers this run (repeatable; page export or the older "
                             "yolo-ped-analysis review/verdicts.json)")
    parser.add_argument("--out", help="output folder (default: <run>/pending)")
    parser.add_argument("--title", help="page title override")
    parser.add_argument("--near-s", type=float, default=NEAR_S,
                        help=f"a row without a matching vehicle id must land within this many seconds "
                             f"(default {NEAR_S})")
    parser.add_argument("--neighbour-s", type=float, default=NEIGHBOUR_S,
                        help=f"name the closest judged event on the card when it is this close (default {NEIGHBOUR_S})")
    args = parser.parse_args(argv)
    build(args.run, args.out, args.verdicts, args.title, args.near_s, args.neighbour_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
