"""Fixed evaluation of yield_image_rule on labelled clips: run the rule, derive the tolerance, score with ADR 0003 rules.

Every change to the rule is judged by rerunning this script on the same clips, so a number can be compared across
versions. It differs from scripts/eval_events.py (which scores any events file) in that it runs the rule itself and
derives the matching tolerance from the residuals instead of taking one on the command line.

What is fixed (docs/adr/0003-fixed-rule-evaluation.md):
- Tolerance: `eval_events.derive_tolerance` on the sorted |prediction - label| residuals of every label that has a
  prediction within `--search` seconds. It is the midpoint of the first structural gap, so it is a property of how the
  labeller and the rule time an event, not of any one result. `--tolerance` overrides it (reported as overridden).
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
                                     [--params '{"merge_duplicates": false}'] [--tolerance auto|SECONDS]

All numbers are in-sample: the clips are the ones the rule and its thresholds were developed on.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_events as ev  # noqa: E402
import yield_image_rule as r  # noqa: E402

SOURCES = ("manual", "confirmed_candidate")
SENSITIVITY_S = (0.5, 1.0, 2.0, 3.0, 5.0)


def video_of(stem, videos_dir):
    for suffix in (".mp4", ".mov", ".mkv", ".avi"):
        path = videos_dir / (stem + suffix)
        if path.exists():
            return path
    matches = sorted(videos_dir.glob(stem + ".*"))
    return matches[0] if matches else None


def run_clip(video, masks, tracks_path, params_overrides):
    """The rule's events for one clip, with the fps and frame size of its video."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(3)), int(capture.get(4))
    capture.release()
    params = r.Params(fps=fps, src_size=(width, height), **params_overrides)
    mask_at, _ = r.load_mask_provider(str(masks))
    tracks = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
    last = max((int(b[0]) for t in tracks for b in t["boxes"]), default=0)
    return params, tracks, [e.to_dict() for e in r.detect(mask_at, tracks, params, range(1, last + 1))]


def predicted_of(events):
    """The events as eval_events sees them: rule event kinds aliased to the label vocabulary."""
    return [{"kind": ev.ALIASES.get(e["kind"], e["kind"]), "time_s": e["time_s"]} for e in events]


def duplicate_clusters(events, groups, tolerance):
    """[(indices)] groups of events within the tolerance that are one physical vehicle: each cluster of k events is k-1
    redundant reports. Measured on the tracks, so it shows what the rule would still emit without the merge."""
    clusters = []
    for i, event in enumerate(events):
        for cluster in clusters:
            if groups.get(event["vehicle_id"]) == groups.get(events[cluster[0]]["vehicle_id"]) \
                    and any(abs(event["time_s"] - events[j]["time_s"]) <= tolerance for j in cluster):
                cluster.append(i)
                break
        else:
            clusters.append([i])
    return [cluster for cluster in clusters if len(cluster) > 1]


def score_clip(events, labels, tolerance, groups, riders):
    """One clip's matches, per-source hits, duplicates and flags."""
    predicted = predicted_of(events)
    pairs = ev.matched_pairs(predicted, labels, tolerance)
    hit_events = {i for i, _ in pairs}
    hit_labels = {j for _, j in pairs}
    by_source = {source: {"labelled": 0, "hit": 0} for source in SOURCES}
    for j, label in enumerate(labels):
        source = label.get("source", "manual")
        by_source.setdefault(source, {"labelled": 0, "hit": 0})
        by_source[source]["labelled"] += 1
        if j in hit_labels:
            by_source[source]["hit"] += 1
    incidents = ev.incident_labels(labels, tolerance)
    incident_hits = sum(1 for group in incidents if hit_labels & set(group))
    unmatched = [e["time_s"] for i, e in enumerate(events) if i not in hit_events]
    duplicates = []
    for a in range(len(events)):
        for b in range(a + 1, len(events)):
            if abs(events[a]["time_s"] - events[b]["time_s"]) > tolerance:
                continue
            same = groups.get(events[a]["vehicle_id"]) == groups.get(events[b]["vehicle_id"])
            if same:
                duplicates.append({"time_s": [events[a]["time_s"], events[b]["time_s"]],
                                   "vehicles": [events[a]["vehicle_id"], events[b]["vehicle_id"]],
                                   "pedestrians": [events[a]["pedestrian_id"], events[b]["pedestrian_id"]]})
    coincident = [{"time_s": [events[a]["time_s"], events[b]["time_s"]],
                   "vehicles": [events[a]["vehicle_id"], events[b]["vehicle_id"]],
                   "pedestrian": events[a]["pedestrian_id"]}
                  for a in range(len(events)) for b in range(a + 1, len(events))
                  if abs(events[a]["time_s"] - events[b]["time_s"]) <= tolerance
                  and events[a]["pedestrian_id"] == events[b]["pedestrian_id"]
                  and groups.get(events[a]["vehicle_id"]) != groups.get(events[b]["vehicle_id"])]
    return {
        "event_count": len(events), "label_count": len(labels), "matched": len(pairs),
        "by_source": by_source,
        "false_alarm": unmatched,
        "missed_labels": [labels[j]["time_s"] for j in sorted(set(range(len(labels))) - hit_labels)],
        "incidents": len(incidents), "incident_hit": incident_hits,
        "duplicates": duplicates, "coincident": coincident,
        "rider_pedestrian_events": [e["time_s"] for e in events
                                    if e["frame"] in riders.get(e["pedestrian_id"], ())],
        "pairs": [[events[i]["time_s"], labels[j]["time_s"], round(abs(events[i]["time_s"] - labels[j]["time_s"]), 3)]
                  for i, j in pairs],
    }


def summarise(clips, tolerance):
    """Totals over clips for one tolerance: label hits per source, incident hits, events, false alarms, duplicates."""
    total = {"labels": 0, "hit": 0, "events": 0, "duplicates": 0, "duplicate_reports": 0,
             "incidents": 0, "incident_hit": 0}
    by_source = {source: {"labelled": 0, "hit": 0} for source in SOURCES}
    for clip in clips:
        events = clip["events"]
        labels = clip["labels"]
        pairs = ev.matched_pairs(predicted_of(events), labels, tolerance)
        hit_labels = {j for _, j in pairs}
        total["labels"] += len(labels)
        total["hit"] += len(hit_labels)
        total["events"] += len(events)
        for j, label in enumerate(labels):
            source = label.get("source", "manual")
            by_source.setdefault(source, {"labelled": 0, "hit": 0})
            by_source[source]["labelled"] += 1
            by_source[source]["hit"] += j in hit_labels
        groups = clip["groups"]
        total["duplicates"] += sum(1 for a in range(len(events)) for b in range(a + 1, len(events))
                                   if abs(events[a]["time_s"] - events[b]["time_s"]) <= tolerance
                                   and groups.get(events[a]["vehicle_id"]) == groups.get(events[b]["vehicle_id"]))
        total["duplicate_reports"] += sum(len(cluster) - 1 for cluster in duplicate_clusters(events, groups, tolerance))
        incidents = ev.incident_labels(labels, tolerance)
        total["incidents"] += len(incidents)
        total["incident_hit"] += sum(1 for group in incidents if hit_labels & set(group))
    per_source = {source: dict(**counts, recall=counts["hit"] / counts["labelled"] if counts["labelled"] else None)
                  for source, counts in by_source.items()}
    return {**total, "by_source": per_source,
            "precision": total["hit"] / total["events"] if total["events"] else None,
            "incident_recall": total["incident_hit"] / total["incidents"] if total["incidents"] else None}


def load_events(path):
    """A saved events .jsonl (e.g. from a rule version that no longer exists in the tree)."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{**row, "contact_frames": tuple(row.get("contact_frames") or ())} for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="folder of the source clips")
    parser.add_argument("--masks-root", required=True, help="per-clip <stem>/seg/masks.npz under here")
    parser.add_argument("--tracks-root", required=True, help="per-clip <stem>/tracks/raw_tracks.json under here")
    parser.add_argument("--labels", required=True, help="folder of <stem>.jsonl human labels")
    parser.add_argument("--events-root", help="score saved <stem>.jsonl events from here instead of running the rule")
    parser.add_argument("--params", default="{}", help="JSON of yield_image_rule.Params overrides (e.g. "
                                                      "'{\"merge_duplicates\": false}')")
    parser.add_argument("--tolerance", default="auto", help="'auto' (derived, default) or seconds")
    parser.add_argument("--search", type=float, default=5.0, help="a label farther than this from every event is a miss")
    parser.add_argument("--json", help="write the full report here")
    args = parser.parse_args(argv)

    videos_dir, masks_root = Path(args.videos), Path(args.masks_root)
    tracks_root, labels_dir = Path(args.tracks_root), Path(args.labels)
    overrides = json.loads(args.params)
    clip_inputs = []
    for path in sorted(labels_dir.glob("*.jsonl")):
        video = video_of(path.stem, videos_dir)
        if video is None:
            print(f"no video for {path.stem} in {videos_dir}", file=sys.stderr)
            return 2
        clip_inputs.append((path.stem, video, masks_root / path.stem / "seg" / "masks.npz",
                            tracks_root / path.stem / "tracks" / "raw_tracks.json", path))

    clips = []
    for stem, video, masks, tracks_path, labels_path in clip_inputs:
        if args.events_root:
            params = r.Params(fps=cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS), **overrides)
            tracks = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
            events = load_events(Path(args.events_root) / (stem + ".jsonl"))
        else:
            params, tracks, events = run_clip(video, masks, tracks_path, overrides)
        labels = ev.load(labels_path)
        riders = r.rider_frames(r.merge_duplicate_tracks(tracks, params), params)
        groups = r.vehicle_groups(tracks, params)
        clips.append({"stem": stem, "fps": params.fps, "events": events, "labels": labels,
                      "groups": groups, "riders": riders})
        print(f"{stem[:16]} {params.fps:>6.2f}fps events {len(events):>2}: "
              f"{[e['time_s'] for e in events]}")

    residuals = []
    for clip in clips:
        residuals += ev.residuals(predicted_of(clip["events"]), clip["labels"], args.search)
    if args.tolerance == "auto":
        manual = [abs(row["delta_s"]) for row in residuals
                  if row["delta_s"] is not None and row["label"].get("source") == "manual"]
        every = [abs(row["delta_s"]) for row in residuals if row["delta_s"] is not None]
        tolerance, basis = ev.derive_tolerance(manual)
        fallback, other = ev.derive_tolerance(every)
        basis = {"method": basis["method"], "below_s": basis.get("below_s"), "above_s": basis.get("above_s"),
                 "gap_s": basis.get("gap_s"), "ratio": basis.get("ratio"), "from_labels": "manual",
                 "sorted_abs_residuals": basis["sorted"],
                 "all_labels": {"value_s": fallback, "method": other["method"],
                                "sorted_abs_residuals": other["sorted"]},
                 "overridden": False}
    else:
        tolerance = float(args.tolerance)
        every = [abs(row["delta_s"]) for row in residuals if row["delta_s"] is not None]
        basis = {"method": "given_on_command_line", "from_labels": "manual", "overridden": True,
                 "sorted_abs_residuals": sorted(every)}

    labels_set = ev.label_set(labels_dir)
    report = {"in_sample": True, "label_set": {"name": labels_set, "dir": str(labels_dir), "clips": len(clips)},
              "tolerance": {"value_s": tolerance, **basis},
              "params": {**r.Params().__dict__, **overrides}, "clips": [], "sensitivity": []}
    for clip in clips:
        result = score_clip(clip["events"], clip["labels"], tolerance, clip["groups"], clip["riders"])
        report["clips"].append({"stem": clip["stem"], "fps": clip["fps"], "events": clip["events"],
                                "labels": clip["labels"], "groups": clip["groups"], **result})
    report["totals"] = summarise(clips, tolerance)
    for value in sorted({*(round(v, 3) for v in SENSITIVITY_S), tolerance}):
        report["sensitivity"].append({"tolerance_s": value, **summarise(clips, value)})

    totals = report["totals"]
    manual, confirmed = totals["by_source"]["manual"], totals["by_source"]["confirmed_candidate"]
    print(f"labels: {labels_set} = {ev.LABEL_SETS.get(labels_set, labels_set)}; {len(clips)} clips from {labels_dir}")
    print(f"tolerance {tolerance}s ({basis['method']}, abs residuals {basis['sorted_abs_residuals']})")
    print(f"{'clip':<18}{'events':>7}{'manual':>9}{'confirmed':>11}{'incid':>8}{'redun':>7}{'FA':>4}")
    for clip in report["clips"]:
        m, c = clip["by_source"]["manual"], clip["by_source"]["confirmed_candidate"]
        redundant = sum(len(cluster) - 1 for cluster in duplicate_clusters(clip["events"], clip["groups"], tolerance))
        print(f"{clip['stem'][:16]:<18}{clip['event_count']:>7}{m['hit']:>7}/{m['labelled']:<2}"
              f"{c['hit']:>9}/{c['labelled']:<2}{clip['incident_hit']:>6}/{clip['incidents']:<2}"
              f"{redundant:>7}{len(clip['false_alarm']):>4}")
    print(f"{'total':<18}{totals['events']:>7}{manual['hit']:>7}/{manual['labelled']:<2}"
          f"{confirmed['hit']:>9}/{confirmed['labelled']:<2}{totals['incident_hit']:>6}/{totals['incidents']:<2}"
          f"{totals['duplicate_reports']:>7}")
    print(f"precision {totals['precision']:.3f}, incident recall {totals['incident_recall']:.3f}, "
          f"redundant reports {totals['duplicate_reports']} ({totals['duplicates']} duplicate pairs), "
          f"in-sample on {len(clips)} clips, labels: {labels_set}")
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
