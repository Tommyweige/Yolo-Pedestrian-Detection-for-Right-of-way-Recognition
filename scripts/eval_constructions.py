"""One table for the two event constructions, scored by the same fixed evaluation.

`yield_image_rule.Params.report_every_contact` selects how contacts become events: the rule's own construction (one
event per vehicle, its earliest qualifying pedestrian) or the candidate list (every (vehicle, pedestrian) contact,
collapsed per vehicle pass by `contact_dedup_s`). Both are scored here by eval_rule_events.score_clip/summarise at one
tolerance, so a number can be compared across constructions instead of two routes reporting different measurements.

Per depth the table gives: events (what the construction emits before the vehicle-pass collapse), deduped events (what
it reports), manual label hits, confirmed-candidate hits, incident recall, duplicate reports and precision (on the
reported events).

All numbers are in-sample: the labelled clips are the ones the depth was chosen on.

Usage:
  python scripts/eval_constructions.py --videos DIR --masks-root DIR --tracks-root DIR --labels DIR
                                       [--depths 0.1,0.04] [--tolerance 1.25] [--json OUT]
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

CONSTRUCTIONS = (("rule", {"report_every_contact": False}), ("candidates", {"report_every_contact": True}))


def score(clip_inputs, overrides, tolerance):
    """{totals, events, deduped} for one parameter set, scored exactly as eval_rule_events does."""
    clips, emitted, kept = [], 0, 0
    for stem, video, masks, tracks_path, labels_path in clip_inputs:
        params, tracks, events = rule_eval.run_clip(video, masks, tracks_path, overrides)
        labels = ev.load(labels_path)
        clips.append({"stem": stem, "events": events, "labels": labels,
                      "groups": r.vehicle_groups(tracks, params)})
        emitted += len(events)
    totals = rule_eval.summarise(clips, tolerance)
    return {"totals": totals, "events": emitted, "deduped": totals["events"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="folder of the source clips")
    parser.add_argument("--masks-root", required=True, help="per-clip <stem>/seg/masks.npz under here")
    parser.add_argument("--tracks-root", required=True, help="per-clip <stem>/tracks/raw_tracks.json under here")
    parser.add_argument("--labels", required=True, help="folder of <stem>.jsonl human labels")
    parser.add_argument("--depths", default="0.1,0.04", help="min_inside_person_heights values to compare")
    parser.add_argument("--tolerance", type=float, default=1.25, help="matching tolerance in seconds")
    parser.add_argument("--json", help="write the full table here")
    args = parser.parse_args(argv)

    videos_dir, masks_root = Path(args.videos), Path(args.masks_root)
    tracks_root, labels_dir = Path(args.tracks_root), Path(args.labels)
    clip_inputs = []
    for path in sorted(labels_dir.glob("*.jsonl")):
        video = rule_eval.video_of(path.stem, videos_dir)
        if video is None:
            print(f"no video for {path.stem} in {videos_dir}", file=sys.stderr)
            return 2
        clip_inputs.append((path.stem, video, masks_root / path.stem / "seg" / "masks.npz",
                            tracks_root / path.stem / "tracks" / "raw_tracks.json", path))

    rows = []
    for depth in (float(text) for text in args.depths.split(",")):
        for name, overrides in CONSTRUCTIONS:
            params = {"min_inside_person_heights": depth, **overrides}
            print(f"running {name} at depth {depth} ...", flush=True)
            if overrides["report_every_contact"]:
                # the candidate list before the vehicle-pass collapse, so the table shows what the collapse removes
                raw = score(clip_inputs, {**params, "contact_dedup_s": 0}, args.tolerance)
                reported = score(clip_inputs, params, args.tolerance)
                events = raw["events"]
            else:
                reported = score(clip_inputs, params, args.tolerance)
                events = reported["deduped"]
            totals = reported["totals"]
            rows.append({"construction": name, "depth": depth, "events": events, "deduped": reported["deduped"],
                         "manual": totals["by_source"]["manual"],
                         "confirmed": totals["by_source"]["confirmed_candidate"],
                         "incidents": [totals["incident_hit"], totals["incidents"]],
                         "duplicate_reports": totals["duplicate_reports"],
                         "duplicate_pairs": totals["duplicates"], "precision": totals["precision"]})
            print(f"  {name} at {depth}: {events} events, {reported['deduped']} after the pass collapse", flush=True)

    print(f"tolerance {args.tolerance}s, {len(clip_inputs)} clips, in-sample")
    print(f"{'construction':<13}{'depth':>6}{'events':>8}{'deduped':>9}{'manual':>9}{'confirmed':>11}"
          f"{'incidents':>11}{'redund':>8}{'precision':>11}")
    for row in rows:
        manual, confirmed = row["manual"], row["confirmed"]
        print(f"{row['construction']:<13}{row['depth']:>6.2f}{row['events']:>8}{row['deduped']:>9}"
              f"{manual['hit']:>7}/{manual['labelled']:<2}{confirmed['hit']:>9}/{confirmed['labelled']:<2}"
              f"{row['incidents'][0]:>9}/{row['incidents'][1]:<2}{row['duplicate_reports']:>8}"
              f"{row['precision']:>11.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps({"tolerance_s": args.tolerance, "in_sample": True, "rows": rows},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
