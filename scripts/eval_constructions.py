"""One table for the two event constructions and one for the dedup variants, scored by the same fixed evaluation.

`yield_image_rule.Params.report_every_contact` selects how contacts become events: the rule's own construction (one
event per vehicle) or the candidate list (every (vehicle, pedestrian) contact). The candidate list then has to collapse
one vehicle's pass, and `contact_dedup` chooses which contact of the pass survives. The defaults run on the candidate
list with `pair` at depth 0.03; every variant and every depth stays selectable (`--variants`, `--depths`):

- `earliest`             the first contact of the pass (what the rule did before the parameter existed);
- `on_crosswalk_longest` the pedestrian already on the crosswalk for the longest catch, falling back to the first contact;
- `pair`                 only repeats of the same (vehicle, pedestrian) pair collapse, so a pass touching two pedestrians
                         is two reports (the default).

Everything is scored by eval_rule_events.summarise at one tolerance, so the rows are comparable, and one detect pass per
depth feeds every row (the rule construction is read off the same candidate list: its vehicle's earliest contact, which is
what `detect()` returns when `report_every_contact` is False). For the dedup variants the table also reports how many
contacts the collapse threw away that do match a judged event (`--verdicts`, the hand-judged A/B/C/D table), and how many
of those judged events end up with no report at all: that is the recall a dedup rule costs, which the label columns only
show when the judged event happens to carry a human label.

All numbers are in-sample: the labelled clips are the ones the depth was chosen on.

Usage:
  python scripts/eval_constructions.py --videos DIR --masks-root DIR --tracks-root DIR --labels DIR
                                       --verdicts verdicts.json [--depths 0.03] [--variants pair] [--tolerance 1.25]
                                       [--json OUT]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_events as ev  # noqa: E402
import eval_rule_events as rule_eval  # noqa: E402
import yield_image_rule as r  # noqa: E402

CONSTRUCTIONS = (("rule", {"report_every_contact": False}),
                 ("candidates", {"report_every_contact": True, "contact_dedup_s": 0}))
JUDGED_SEARCH_S = 5.0   # a dropped contact this close to a judged event of the same vehicle is that event


def score(collected, events_of, tolerance):
    """Totals for one event list per clip, scored exactly as eval_rule_events does."""
    clips = [{"stem": clip["stem"], "events": [event.to_dict() for event in events_of(clip)],
              "labels": clip["labels"], "groups": clip["groups"]} for clip in collected]
    return rule_eval.summarise(clips, tolerance)


def as_events(records):
    """`eval_rule_events.run_clip` hands back dicts; the construction code works on the rule's own `Event`."""
    return [r.Event(**record) for record in records]


def per_clip(clip, events, tolerance):
    """One clip's numbers and the manual labels it does not report, so a table row can be read clip by clip."""
    records = [event.to_dict() for event in events]
    hits = {j for _, j in ev.matched_pairs(rule_eval.predicted_of(records), clip["labels"], tolerance)}
    manual = [j for j, label in enumerate(clip["labels"]) if label.get("source", "manual") == "manual"]
    incidents = ev.incident_labels(clip["labels"], tolerance)
    return {"stem": clip["stem"], "events": len(records),
            "manual": f"{sum(j in hits for j in manual)}/{len(manual)}",
            "incidents": f"{sum(1 for group in incidents if hits & set(group))}/{len(incidents)}",
            "missed_manual": [clip["labels"][j]["time_s"] for j in manual if j not in hits]}


def identity(event):
    return (event.vehicle_id, event.pedestrian_id, event.time_s)


def first_per_vehicle(candidates):
    """The rule's own construction, taken from the same candidates: each vehicle's earliest qualifying contact.

    `detect()` does exactly this under `report_every_contact=False`, so one detect pass produces both the rule row and
    the candidate rows, which keeps them on the same detection instead of two GPU runs that could drift apart.
    """
    events, seen = [], set()
    for event in candidates:      # contact-start order
        if event.vehicle_id in seen:
            continue
        seen.add(event.vehicle_id)
        events.append(event)
    return events


def dropped_of(candidates, kept):
    """Candidates the dedup collapsed, matched back to the kept list by identity."""
    kept_ids = {identity(event) for event in kept}
    return [event for event in candidates if identity(event) not in kept_ids]


def matches(event, row, tolerance):
    """Does one contact report the judged event `row`? Same (vehicle, pedestrian) inside the wider search, or the same
    vehicle inside the scoring tolerance (a contact can sit a second or two off the time a labeller marks)."""
    if event.vehicle_id != row["vehicle"]:
        return False
    same_pair = row.get("ped") is not None and row["ped"] == event.pedestrian_id
    window = JUDGED_SEARCH_S if same_pair else tolerance
    return abs(event.time_s - row["time_s"]) <= window


def judged_cost(stem, candidates, kept, verdicts, tolerance):
    """What the collapse costs, measured against the hand-judged events (`--verdicts`).

    Two counts per judged event `row` that some *dropped* contact matches:
    - `contact`: the dropped contacts that match it (the column the table prints);
    - `lost_row`: 1 when no *kept* event matches it any more, i.e. the collapse left that judged event unreported.
    """
    dropped = dropped_of(candidates, kept)
    entries = []
    for row in verdicts.get(stem, []):
        hit_by = [event for event in dropped if matches(event, row, tolerance)]
        if not hit_by:
            continue
        still_reported = any(matches(event, row, tolerance) for event in kept)
        entries.append({"verdict": row["verdict"], "judged_time_s": row["time_s"], "vehicle": row["vehicle"],
                        "pedestrian": row.get("ped"), "lost_row": not still_reported,
                        "contacts": [{"time_s": event.time_s, "pedestrian": event.pedestrian_id,
                                      "state": event.pedestrian_state} for event in hit_by]})
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="folder of the source clips")
    parser.add_argument("--masks-root", required=True, help="per-clip <stem>/seg/masks.npz under here")
    parser.add_argument("--tracks-root", required=True, help="per-clip <stem>/tracks/raw_tracks.json under here")
    parser.add_argument("--labels", required=True, help="folder of <stem>.jsonl human labels")
    parser.add_argument("--verdicts", help="hand-judged events (clip/stem, time_s, vehicle, ped, verdict) as JSON")
    parser.add_argument("--depths", default="0.03",
                        help="min_inside_person_heights values to compare (default: the rule's own 0.03)")
    parser.add_argument("--variants", default=",".join(r.CONTACT_DEDUP_VARIANTS), help="contact_dedup values to compare")
    parser.add_argument("--tolerance", type=float, default=1.25, help="matching tolerance in seconds")
    parser.add_argument("--json", help="write the full table here")
    args = parser.parse_args(argv)
    print(f"rule defaults: depth {r.Params.min_inside_person_heights}, candidate-list dedup "
          f"{r.Params.contact_dedup!r}", flush=True)

    videos_dir, masks_root = Path(args.videos), Path(args.masks_root)
    tracks_root, labels_dir = Path(args.tracks_root), Path(args.labels)
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    verdicts = {}
    if args.verdicts:
        for row in json.loads(Path(args.verdicts).read_text(encoding="utf-8")):
            verdicts.setdefault(row["stem"], []).append(row)
    clip_inputs = []
    for path in sorted(labels_dir.glob("*.jsonl")):
        video = rule_eval.video_of(path.stem, videos_dir)
        if video is None:
            print(f"no video for {path.stem} in {videos_dir}", file=sys.stderr)
            return 2
        clip_inputs.append((path.stem, video, masks_root / path.stem / "seg" / "masks.npz",
                            tracks_root / path.stem / "tracks" / "raw_tracks.json", path))

    construction_rows, variant_rows = [], []
    for depth in (float(text) for text in args.depths.split(",")):
        print(f"depth {depth}: running one detect pass per clip, the rule and the candidate list come off it ...",
              flush=True)
        collected = []
        for stem, video, masks, tracks_path, labels_path in clip_inputs:
            params, tracks, records = rule_eval.run_clip(
                video, masks, tracks_path, {"min_inside_person_heights": depth, **CONSTRUCTIONS[1][1]})
            candidates = as_events(records)
            print(f"  {stem}: {len(candidates)} contacts", flush=True)
            collected.append({"stem": stem, "labels": ev.load(labels_path), "params": params,
                              "groups": r.vehicle_groups(tracks, params), "rule": first_per_vehicle(candidates),
                              "candidates": candidates})
        def deduped(clip, variant):
            return r.dedupe_contacts(clip["candidates"], _variant_params(clip, variant, args.tolerance))

        for name, events_of, emitted_of in (
                ("one event per vehicle", lambda clip: clip["rule"], lambda clip: clip["rule"]),
                (f"candidate list ({r.Params.contact_dedup})",
                 lambda clip: deduped(clip, clip["params"].contact_dedup), lambda clip: clip["candidates"])):
            totals = score(collected, events_of, args.tolerance)
            construction_rows.append({"construction": name, "depth": depth,
                                      "events": sum(len(emitted_of(clip)) for clip in collected),
                                      "deduped": totals["events"], "totals": totals,
                                      "per_clip": [per_clip(clip, events_of(clip), args.tolerance)
                                                   for clip in collected]})
        for variant in variants:
            kept_of = lambda clip: deduped(clip, variant)
            totals = score(collected, kept_of, args.tolerance)
            cost = [dict(clip=clip["stem"], **entry) for clip in collected
                    for entry in judged_cost(clip["stem"], clip["candidates"], kept_of(clip), verdicts,
                                             args.tolerance)]
            variant_rows.append({"variant": variant, "depth": depth,
                                 "raw": sum(len(clip["candidates"]) for clip in collected),
                                 "deduped": totals["events"], "totals": totals, "judged": cost,
                                 "per_clip": [per_clip(clip, kept_of(clip), args.tolerance) for clip in collected]})
            contacts = sum(len(entry["contacts"]) for entry in cost)
            print(f"  {variant} at {depth}: {variant_rows[-1]['raw']} raw -> {totals['events']} "
                  f"({contacts} dropped contacts on judged events, {sum(e['lost_row'] for e in cost)} of those rows "
                  f"now unreported)", flush=True)

    print(f"tolerance {args.tolerance}s, {len(clip_inputs)} clips, in-sample")
    print("constructions")
    print(f"{'construction':<23}{'depth':>6}{'events':>8}{'deduped':>9}{'manual':>9}{'confirmed':>11}"
          f"{'incidents':>11}{'redund':>8}{'precision':>11}")
    for row in construction_rows:
        print(_row(row["depth"], row["events"], row["deduped"], row["totals"], width=23, first=row["construction"]))
    print("dedup variants (candidate list)")
    print(f"{'variant':<23}{'depth':>6}{'raw':>6}{'deduped':>9}{'manual':>9}{'confirmed':>11}"
          f"{'incidents':>11}{'redund':>8}{'precision':>11}  dropped-but-judged")
    for row in variant_rows:
        print(_row(row["depth"], row["raw"], row["deduped"], row["totals"], width=23, first=row["variant"],
                   extra=_judged_summary(row["judged"])))
    print("manual labels no construction reports")
    for row in construction_rows + variant_rows:
        name = row.get("variant") or row["construction"]
        missed = [f"{clip['stem'][:6]}:{time_s}" for clip in row["per_clip"] for time_s in clip["missed_manual"]]
        print(f"  {name} @ {row['depth']}: {', '.join(missed) or 'none'}")
    if args.json:
        Path(args.json).write_text(json.dumps({"tolerance_s": args.tolerance, "in_sample": True,
                                               "constructions": construction_rows,
                                               "variants": variant_rows}, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


def _judged_summary(judged):
    """`N contacts (A2/C3) of which M judged events end up with no report at all` in one line."""
    contacts = sum(len(entry["contacts"]) for entry in judged)
    by_verdict = {}
    for entry in judged:
        by_verdict[entry["verdict"]] = by_verdict.get(entry["verdict"], 0) + len(entry["contacts"])
    composition = "/".join(f"{name}{by_verdict[name]}" for name in sorted(by_verdict))
    lost_rows = sum(entry["lost_row"] for entry in judged)
    if not contacts:
        return "0"
    return f"{contacts} ({composition}) of {len(judged)} judged events, {lost_rows} left unreported"


def _variant_params(clip, variant, tolerance):
    """The clip's params with one dedup choice; the pass window is the evaluation's own tolerance, so a collapse can never
    hide two contacts the scorer would have counted as two passes."""
    from dataclasses import replace
    return replace(clip["params"], contact_dedup=variant, contact_dedup_s=tolerance)


def _row(depth, emitted, deduped, totals, width, first, extra=""):
    manual, confirmed = totals["by_source"]["manual"], totals["by_source"]["confirmed_candidate"]
    return (f"{first:<{width}}{depth:>6.2f}{emitted:>8}{deduped:>9}"
            f"{manual['hit']:>7}/{manual['labelled']:<2}{confirmed['hit']:>9}/{confirmed['labelled']:<2}"
            f"{totals['incident_hit']:>9}/{totals['incidents']:<2}{totals['duplicate_reports']:>8}"
            f"{totals['precision']:>11.3f}" + (f"  {extra}" if extra else ""))


if __name__ == "__main__":
    sys.exit(main())
