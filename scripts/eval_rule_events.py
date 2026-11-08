"""Fixed evaluation of yield_image_rule on labelled clips: run the rule, score it with ADR 0003 rules at a frozen tolerance.

Every change to the rule is judged by rerunning this script on the same clips, so a number can be compared across
versions. It differs from scripts/eval_events.py (which scores any events file) in that it runs the rule itself and
scores at the fixed evaluation's tolerance instead of a generic one.

What is fixed (docs/adr/0003-fixed-rule-evaluation.md):
- Tolerance: `event_scoring.TOLERANCE_S` (1.25 s), frozen by the ADR 0003 amendment of 2026-11-08. It was derived once
  by `event_scoring.derive_tolerance` (the midpoint of the first structural gap in the sorted |prediction - label|
  residuals). Each run still derives it from the manual labels' residuals (every label with a prediction within
  `--search` seconds) and reports the result under `tolerance.derived` without scoring with it, so a drift is visible
  without moving the yardstick. `--tolerance auto` scores at the derived value, `--tolerance SECONDS` at a given one;
  both are reported as overridden.
- One event per incident: a label matches at most one event and vice versa, closest first (`matched_pairs`). Labels
  closer together than the tolerance are one incident (`incident_labels`), and an incident counts as a hit when any of
  its labels is matched, so recall does not depend on how often the labeller pressed the key.
- One report per vehicle pass: an event is one (vehicle, pedestrian) contact and the rule reports a vehicle once. Two
  events for one physical vehicle are duplicates, not extra detections: that is measured on the tracks, with
  `yield_image_rule.vehicle_groups`, so it is visible whether the rule still emits them.
- Two vehicles at the same pedestrian within the tolerance are NOT duplicates. Two cars can fail to yield to one person
  at the same moment, and only a human can say whether that is one incident; they are reported as `coincident` pairs for
  review instead of being merged or counted as errors.

Usage:
  python scripts/eval_rule_events.py --videos DIR --masks-root DIR --tracks-root DIR --labels DIR [--json OUT]
                                     [--params '{"merge_duplicates": false}'] [--tolerance frozen|auto|SECONDS]

All numbers are in-sample: the clips are the ones the rule and its thresholds were developed on.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_artifacts  # noqa: E402
import event_scoring as scoring  # noqa: E402
import yield_image_rule as r  # noqa: E402

SENSITIVITY_S = (0.5, 1.0, 2.0, 3.0, 5.0)


def clip_params(video, overrides):
    info = clip_artifacts.video_info(video)
    return r.Params(fps=info.fps, src_size=info.size, **overrides)


def run_clip(video, masks, tracks_path, params_overrides):
    """The rule's events for one clip, with the fps and frame size of its video."""
    params = clip_params(video, params_overrides)
    tracks = clip_artifacts.load_tracks(tracks_path)
    return params, tracks, [e.to_dict() for e in r.run_on_clip(masks, tracks, params)]


def load_events(path):
    """A saved events .jsonl (e.g. from a rule version that no longer exists in the tree)."""
    return [{**row, "contact_frames": tuple(row.get("contact_frames") or ())} for row in clip_artifacts.read_events(path)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="folder of the source clips")
    parser.add_argument("--masks-root", required=True, help="per-clip <stem>/seg/masks.npz under here")
    parser.add_argument("--tracks-root", required=True, help="per-clip <stem>/tracks/raw_tracks.json under here")
    parser.add_argument("--labels", required=True, help="folder of <stem>.jsonl human labels")
    parser.add_argument("--events-root", help="score saved <stem>.jsonl events from here instead of running the rule")
    parser.add_argument("--params", default="{}", help="JSON of yield_image_rule.Params overrides (e.g. "
                                                      "'{\"merge_duplicates\": false}')")
    parser.add_argument("--tolerance", default="frozen",
                        help="'frozen' (event_scoring.TOLERANCE_S, default), 'auto' (derived on this run) or seconds")
    parser.add_argument("--search", type=float, default=5.0, help="a label farther than this from every event is a miss")
    parser.add_argument("--json", help="write the full report here")
    args = parser.parse_args(argv)

    videos_dir, masks_root = Path(args.videos), Path(args.masks_root)
    tracks_root, labels_dir = Path(args.tracks_root), Path(args.labels)
    overrides = json.loads(args.params)
    clip_inputs = []
    for path in sorted(labels_dir.glob("*.jsonl")):
        video = clip_artifacts.find_video(path.stem, videos_dir)
        if video is None:
            print(f"no video for {path.stem} in {videos_dir}", file=sys.stderr)
            return 2
        clip_inputs.append((path.stem, video, masks_root / path.stem / clip_artifacts.MASKS,
                            tracks_root / path.stem / clip_artifacts.TRACKS, path))

    clips = []
    for stem, video, masks, tracks_path, labels_path in clip_inputs:
        if args.events_root:
            params = clip_params(video, overrides)
            tracks = clip_artifacts.load_tracks(tracks_path)
            events = load_events(Path(args.events_root) / (stem + ".jsonl"))
        else:
            params, tracks, events = run_clip(video, masks, tracks_path, overrides)
        labels = scoring.load(labels_path)
        riders = r.rider_frames(r.merge_duplicate_tracks(tracks, params), params)
        groups = r.vehicle_groups(tracks, params)
        clips.append({"stem": stem, "fps": params.fps, "events": events, "labels": labels,
                      "groups": groups, "riders": riders})
        print(f"{stem[:16]} {params.fps:>6.2f}fps events {len(events):>2}: "
              f"{[e['time_s'] for e in events]}")

    residuals = []
    for clip in clips:
        residuals += scoring.residuals(scoring.predicted_of(clip["events"]), clip["labels"], args.search)
    manual = [abs(row["delta_s"]) for row in residuals
              if row["delta_s"] is not None and row["label"].get("source") == "manual"]
    every = [abs(row["delta_s"]) for row in residuals if row["delta_s"] is not None]
    derived_value, derived = scoring.derive_tolerance(manual)
    all_value, all_labels = scoring.derive_tolerance(every)
    derived = {"value_s": derived_value, "method": derived["method"], "below_s": derived.get("below_s"),
               "above_s": derived.get("above_s"), "gap_s": derived.get("gap_s"), "ratio": derived.get("ratio"),
               "from_labels": "manual", "sorted_abs_residuals": derived["sorted"],
               "all_labels": {"value_s": all_value, "method": all_labels["method"],
                              "sorted_abs_residuals": all_labels["sorted"]}}
    if args.tolerance == "frozen":
        tolerance, method = scoring.TOLERANCE_S, "frozen_adr_0003"
    elif args.tolerance == "auto":
        tolerance, method = derived_value, "derived_" + derived["method"]
    else:
        tolerance, method = float(args.tolerance), "given_on_command_line"
    basis = {"method": method, "overridden": args.tolerance != "frozen", "derived": derived}

    report = {"in_sample": True, "tolerance": {"value_s": tolerance, **basis},
              "params": {**r.Params().__dict__, **overrides}, "clips": [], "sensitivity": []}
    for clip in clips:
        result = scoring.score_clip(clip["events"], clip["labels"], tolerance, clip["groups"], clip["riders"])
        report["clips"].append({"stem": clip["stem"], "fps": clip["fps"], "events": clip["events"],
                                "labels": clip["labels"], "groups": clip["groups"], **result})
    report["totals"] = scoring.summarise(clips, tolerance)
    for value in sorted({*(round(v, 3) for v in SENSITIVITY_S), tolerance}):
        report["sensitivity"].append({"tolerance_s": value, **scoring.summarise(clips, value)})

    totals = report["totals"]
    manual, confirmed = totals["by_source"]["manual"], totals["by_source"]["confirmed_candidate"]
    print(f"tolerance {tolerance}s ({method}); derived from the manual residuals: {derived_value}s "
          f"({derived['method']}, abs residuals {derived['sorted_abs_residuals']})")
    print(f"{'clip':<18}{'events':>7}{'manual':>9}{'confirmed':>11}{'incid':>8}{'redun':>7}{'FA':>4}")
    for clip in report["clips"]:
        m, c = clip["by_source"]["manual"], clip["by_source"]["confirmed_candidate"]
        redundant = sum(len(cluster) - 1 for cluster in scoring.duplicate_clusters(clip["events"], clip["groups"], tolerance))
        print(f"{clip['stem'][:16]:<18}{clip['event_count']:>7}{m['hit']:>7}/{m['labelled']:<2}"
              f"{c['hit']:>9}/{c['labelled']:<2}{clip['incident_hit']:>6}/{clip['incidents']:<2}"
              f"{redundant:>7}{len(clip['false_alarm']):>4}")
    print(f"{'total':<18}{totals['events']:>7}{manual['hit']:>7}/{manual['labelled']:<2}"
          f"{confirmed['hit']:>9}/{confirmed['labelled']:<2}{totals['incident_hit']:>6}/{totals['incidents']:<2}"
          f"{totals['duplicate_reports']:>7}")
    print(f"precision {totals['precision']:.3f}, incident recall {totals['incident_recall']:.3f}, "
          f"redundant reports {totals['duplicate_reports']} ({totals['duplicates']} duplicate pairs), "
          f"in-sample on {len(clips)} clips")
    for row in report["sensitivity"]:
        print(f"  tol {row['tolerance_s']:>4}s  manual {row['by_source']['manual']['hit']}/{row['by_source']['manual']['labelled']}"
              f"  confirmed {row['by_source']['confirmed_candidate']['hit']}/{row['by_source']['confirmed_candidate']['labelled']}"
              f"  events {row['events']}  incidents {row['incident_hit']}/{row['incidents']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
