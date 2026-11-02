"""Event-level matching by EIoU (Crosswalk, IEEE T-ITS 2025) next to the fixed 1.25 s match, plus AP over scores.

The definitions are read from https://github.com/Nanasaki-Ai/Crosswalk (MIT-licensed reference implementation); this
module reproduces them and changes nothing about the rule:

- `tiou(det_start, det_end, label_start, label_end)`: temporal IoU of the two event intervals,
  intersection / union with the intervals measured as `end - start` (`tools.py: calculate_tiou`).
- `siou(det_box, label_box)`: ordinary box IoU of the vehicle boxes (`tools.py: calculate_siou`).
- `eiou(det, label)`: `tiou * (siou_start + siou_end) / 2`, the mean of the box IoU at the two events' start frames and
  at their end frames. `eiou_cal.py` computes `tiou * (siou_s + siou_e)` and stores `best_eiou / 2`; this is the same
  number, so it is the value compared against the threshold.
- A ground-truth event is retrieved once and a detection matches once, and a match needs `eiou > threshold` (strict;
  `ap_cal.py` uses `eiou_val > eiou_th` with 0.5). Crosswalk walks the labels one at a time and gives each its best
  still-unmatched detection; `match` instead sorts all (detection, label) pairs by EIoU, which yields the same pairs
  when they do not compete and does not depend on the order of the labels in the file.
- AP is `sklearn.metrics.average_precision_score` over per-event scores (`ap_cal.py`), reimplemented here so this
  module does not need scikit-learn, and with tied scores grouped like sklearn does.

What this module does NOT do: invent a ground-truth box, a ground-truth interval, or a confidence score. A label with
no single vehicle behind it gets no EIoU, and an events file with no `score` field gets no AP - both are reported as
unavailable with the reason, never filled in with a default (Crosswalk's own `eiou.get(sample_name, 1.0)` default is
exactly the shortcut this avoids).

Usage (7 dev clips, both metrics, no AP):
  python scripts/eval_eiou.py --run .amp/in/dev-run --labels LABELS_DIR [--json OUT]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_events as ev  # noqa: E402

AP_UNAVAILABLE = "no confidence score: yield_image_rule reports a yes/no event, so AP@EIoU cannot be computed from it"


def tiou(det_start, det_end, label_start, label_end):
    """Temporal IoU of two event intervals, in whatever unit both are given (frames or seconds)."""
    intersection = max(0, min(det_end, label_end) - max(det_start, label_start))
    union = (det_end - det_start) + (label_end - label_start) - intersection
    return intersection / union if union > 0 else 0.0


def siou(det_box, label_box):
    """Box IoU of two vehicle boxes, (x1, y1, x2, y2)."""
    iw = min(det_box[2], label_box[2]) - max(det_box[0], label_box[0])
    ih = min(det_box[3], label_box[3]) - max(det_box[1], label_box[1])
    intersection = max(0, iw) * max(0, ih)
    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
    label_area = (label_box[2] - label_box[0]) * (label_box[3] - label_box[1])
    union = det_area + label_area - intersection
    return intersection / union if union > 0 else 0.0


def eiou(det, label):
    """Event IoU: `det` and `label` each {'start', 'end', 'start_box', 'end_box'} -> tiou * mean(siou_start, siou_end).

    The temporal and spatial parts multiply, so an event that is right in time but on another vehicle scores 0 and so
    does one on the right vehicle at the wrong moment.
    """
    return tiou(det["start"], det["end"], label["start"], label["end"]) * (
        siou(det["start_box"], label["start_box"]) + siou(det["end_box"], label["end_box"])) / 2


def match(detections, labels, threshold=0.5):
    """[{'det', 'label', 'eiou'}] one-to-one matches with EIoU strictly above `threshold`, best EIoU first."""
    candidates = sorted(((eiou(det, label), i, j)
                         for i, det in enumerate(detections) for j, label in enumerate(labels)),
                        key=lambda row: -row[0])
    used_det, used_label, out = set(), set(), []
    for value, i, j in candidates:
        if value <= threshold or i in used_det or j in used_label:
            continue
        used_det.add(i), used_label.add(j)
        out.append({"det": i, "label": j, "eiou": round(value, 4)})
    return out


def average_precision(scores, hits):
    """Average precision of the detections ordered by `scores` (descending), `hits[i]` = detection i is a true positive.

    Same value as `sklearn.metrics.average_precision_score(hits, scores)` for binary labels: step-wise precision at
    each distinct score, weighted by the recall it adds. Ties are one threshold, so an uninformative constant score
    gives the precision of the whole set, not the precision of an arbitrary order. Raises when any detection has no
    score, because ranking an event that has none is the fabrication this evaluation must not do.
    """
    if len(scores) != len(hits):
        raise ValueError(f"{len(scores)} scores for {len(hits)} detections")
    if not scores:
        raise ValueError("AP needs at least one detection")
    if any(score is None for score in scores):
        raise ValueError(AP_UNAVAILABLE)
    positives = sum(1 for hit in hits if hit)
    if positives == 0:
        raise ValueError("AP needs at least one true positive; there is nothing to rank against")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    true_positive, precision_sum, i = 0, 0.0, 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1                                  # one threshold per distinct score
        block = sum(1 for k in order[i:j] if hits[k])
        true_positive += block
        precision_sum += (block / positives) * (true_positive / j)
        i = j
    return precision_sum


def residency_spans(frames, gap_frames=0):
    """{frame: (start, end)}: the crosswalk pass each on-crosswalk frame of one vehicle belongs to.

    `vehicle_on` is judged frame by frame and flickers at the mask's edge, and a tracker drop splits one pass, so runs
    separated by at most `gap_frames` are one pass (`gap_frames` comes from the tracker's own duplicate-reidentification
    gap: shorter dropouts are the same presence, not two crossings).
    """
    groups = []
    for frame in sorted(frames):
        if groups and frame - groups[-1][-1] <= gap_frames + 1:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return {frame: (group[0], group[-1]) for group in groups for frame in group}


# ---------------------------------------------------------------- clips
#
# Both sides of an EIoU need a vehicle box at the start and the end of an event. Detection side: the rule's event
# carries `vehicle_id` and `contact_frames`, and the tracks it was produced from (`raw_tracks.json`, merged with
# `yield_image_rule.merge_duplicate_tracks`, which is what the rule does before reporting) give that box per frame.
#
# Label side: a human label is a time point, with no vehicle and no box. This module recovers one only when exactly one
# vehicle track is judged on the crosswalk at the label's frame (or, if that frame has no vehicle at all, within
# `label_window_s` of it) and then uses that track's crosswalk pass as the label's interval (Crosswalk's definition of an
# event: it begins when the vehicle enters the crosswalk and ends when it leaves) and its boxes at the two ends. A label
# with two or three candidate vehicles, or with none, gets no ground-truth box: it is reported as unresolved and left out
# of the EIoU numbers rather than guessed.

def read_events(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def predicted_of(events):
    """The events as `eval_events` sees them: rule event kinds aliased to the label vocabulary."""
    return [{"kind": ev.ALIASES.get(event["kind"], event["kind"]), "time_s": event["time_s"]} for event in events]


def crosswalk_residency(tracks, params, mask_at):
    """{track id: [frames on the crosswalk]} for the vehicle tracks, judged frame by frame with `vehicle_on`."""
    import yield_image_rule as rule

    boxes = {}
    for track in tracks:
        if track["kind"] not in rule.VEHICLE_KINDS:
            continue
        for frame, *box in track["boxes"]:
            boxes.setdefault(int(frame), []).append((track["id"], tuple(box)))
    residency = {}
    for frame in sorted(boxes):
        mask = mask_at(frame)
        if mask is None:
            continue
        sx, sy = mask.shape[1] / params.src_size[0], mask.shape[0] / params.src_size[1]
        components = rule.crosswalk_components(mask, params)
        for vehicle_id, box in boxes[frame]:
            scaled = rule.scaled(box, sx, sy)
            if any(rule.vehicle_on(scaled, component, params) for component in components):
                residency.setdefault(vehicle_id, []).append(frame)
    return residency


def load_clip(entry, labels_path, pass_gap_s=None):
    """One run clip with the boxes and the crosswalk passes its events and labels are matched through."""
    import yield_image_rule as rule

    params = rule.Params(fps=entry["fps"], src_size=tuple(entry["size"]))
    mask_at, _ = rule.load_mask_provider(entry["masks"])
    tracks = rule.merge_duplicate_tracks(json.loads(Path(entry["tracks"]).read_text(encoding="utf-8")), params)
    boxes = {track["id"]: {int(b[0]): tuple(b[1:]) for b in track["boxes"]} for track in tracks}
    residency = crosswalk_residency(tracks, params, mask_at)
    gap_frames = round((params.same_vehicle_gap_s if pass_gap_s is None else pass_gap_s) * params.fps)
    on_frame = {}
    for vehicle_id, frames in residency.items():
        for frame in frames:
            on_frame.setdefault(frame, []).append(vehicle_id)
    return {
        "stem": entry["stem"], "fps": params.fps, "params": params, "boxes": boxes,
        "spans": {vehicle_id: residency_spans(frames, gap_frames) for vehicle_id, frames in residency.items()},
        "on_frame": {frame: sorted(ids) for frame, ids in on_frame.items()},
        "events": read_events(entry["events_jsonl"]), "labels": ev.load(labels_path),
    }


def detections_of(clip, span="contact"):
    """The rule's events as EIoU inputs, each carrying its own event index.

    `span="contact"` keeps the event's own contact window (what the rule reports). `span="residency"` replaces it with
    its vehicle's whole crosswalk pass, which is the interval Crosswalk uses on both sides; the two are reported side by
    side because their difference is how much of the gap comes from the rule timing an event by the vehicle-pedestrian
    contact instead of by the vehicle's entry and exit.
    """
    out, unusable = [], []
    for index, event in enumerate(clip["events"]):
        boxes = clip["boxes"].get(event["vehicle_id"], {})
        start, end = (int(f) for f in event["contact_frames"])
        if span == "residency":
            spans = clip["spans"].get(event["vehicle_id"], {})
            pass_span = spans.get(start) or spans.get(end)
            if pass_span is None:
                unusable.append({"event": index, "time_s": event["time_s"], "vehicle_id": event["vehicle_id"],
                                 "reason": "vehicle_not_on_crosswalk_in_tracks"})
                continue
            start, end = pass_span
        if start not in boxes or end not in boxes:
            unusable.append({"event": index, "time_s": event["time_s"], "vehicle_id": event["vehicle_id"],
                             "reason": "no_vehicle_box_at_event_frame"})
            continue
        out.append({"event": index, "start": start, "end": end, "start_box": boxes[start], "end_box": boxes[end],
                    "time_s": event["time_s"], "vehicle_id": event["vehicle_id"]})
    return out, unusable


def label_vehicle(clip, frame, window_frames):
    """(vehicle id, (start, end), basis) for one label, or (None, None, reason) when it has no single vehicle.

    Exact frame first, and that verdict stands: one vehicle on the crosswalk there and the label is about that vehicle;
    two or three and the labels do not say which was meant, so picking the nearest would be inventing the ground truth.
    Only a label with no vehicle at its frame at all looks `window_frames` frames either side - the labeller picks the
    moment by eye (their documented accuracy is +-1 s), so a frame that falls in a one-frame hole of its vehicle's
    residency is not evidence of anything - and it is resolved only if that leaves exactly one candidate, with `basis`
    saying it came from the window rather than the frame.
    """
    exact = clip["on_frame"].get(frame, [])
    if len(exact) == 1:
        span = clip["spans"][exact[0]].get(frame)
        if span is None or span[0] not in clip["boxes"][exact[0]] or span[1] not in clip["boxes"][exact[0]]:
            return None, None, "no_vehicle_box_at_residency_edge"
        return exact[0], span, "exact"
    if len(exact) > 1:
        return None, None, "several_vehicles_on_crosswalk"
    candidates = {}
    for offset in range(-window_frames, window_frames + 1):
        for vehicle_id in clip["on_frame"].get(frame + offset, []):
            distance = abs(offset)
            if vehicle_id not in candidates or distance < candidates[vehicle_id][0]:
                candidates[vehicle_id] = (distance, frame + offset)
    if not candidates:
        return None, None, "no_vehicle_on_crosswalk"
    if len(candidates) > 1:
        return None, None, "several_vehicles_on_crosswalk"
    vehicle_id, (_, at) = next(iter(candidates.items()))
    span = clip["spans"][vehicle_id].get(at)
    if span is None or span[0] not in clip["boxes"][vehicle_id] or span[1] not in clip["boxes"][vehicle_id]:
        return None, None, "no_vehicle_box_at_residency_edge"
    return vehicle_id, span, "window"


def ground_truths(clip, window_frames=0):
    """[{'label', 'vehicle', 'gt', 'basis', 'reason'}] per label; 'gt' is None when the label has no single vehicle."""
    out = []
    for index, label in enumerate(clip["labels"]):
        frame = int(round(label["time_s"] * clip["fps"]))
        if label["kind"] != "suspected_failure_to_yield":
            out.append({"label": index, "vehicle": None, "gt": None, "basis": None, "reason": "not_a_failure_label"})
            continue
        vehicle_id, span, basis = label_vehicle(clip, frame, window_frames)
        gt = None
        if span is not None:
            boxes = clip["boxes"][vehicle_id]
            gt = {"start": span[0], "end": span[1], "start_box": boxes[span[0]], "end_box": boxes[span[1]],
                  "time_s": label["time_s"], "vehicle_id": vehicle_id}
        out.append({"label": index, "vehicle": vehicle_id, "gt": gt,
                    "basis": basis if gt is not None else None,
                    "reason": None if gt is not None else basis})
    return out


def score_clip(clip, threshold=0.5, tolerance=1.25, label_window_s=0.5):
    """Both instruments on one clip: EIoU at `threshold`, and the fixed time tolerance on the same labels."""
    truth = ground_truths(clip, int(round(label_window_s * clip["fps"])))
    resolved = [row for row in truth if row["gt"] is not None]
    unresolved = [{"time_s": clip["labels"][row["label"]]["time_s"],
                   "source": clip["labels"][row["label"]].get("source"),
                   "reason": row["reason"],
                   "vehicles_on_crosswalk": clip["on_frame"].get(
                       int(round(clip["labels"][row["label"]]["time_s"] * clip["fps"])), [])}
                  for row in truth if row["gt"] is None and row["reason"] != "not_a_failure_label"]
    gt_events = [row["gt"] for row in resolved]
    detections, unusable = detections_of(clip)
    eiou_pairs = match(detections, gt_events, threshold)
    whole_pass, _ = detections_of(clip, span="residency")
    whole_pass_pairs = match(whole_pass, gt_events, threshold)

    predicted = predicted_of(clip["events"])
    all_pairs = ev.matched_pairs(predicted, clip["labels"], tolerance)
    resolved_pairs = ev.matched_pairs(predicted, [clip["labels"][row["label"]] for row in resolved], tolerance)

    def pair_detail(rows, pairs):
        return [{"event": rows[pair["det"]]["event"], "time_s": rows[pair["det"]]["time_s"],
                 "vehicle_id": rows[pair["det"]]["vehicle_id"],
                 "det_frames": [rows[pair["det"]]["start"], rows[pair["det"]]["end"]],
                 "label_time_s": gt_events[pair["label"]]["time_s"],
                 "gt_vehicle_id": gt_events[pair["label"]]["vehicle_id"],
                 "gt_frames": [gt_events[pair["label"]]["start"], gt_events[pair["label"]]["end"]],
                 "tiou": round(tiou(rows[pair["det"]]["start"], rows[pair["det"]]["end"],
                                    gt_events[pair["label"]]["start"], gt_events[pair["label"]]["end"]), 4),
                 "siou_start": round(siou(rows[pair["det"]]["start_box"],
                                          gt_events[pair["label"]]["start_box"]), 4),
                 "eiou": pair["eiou"]} for pair in pairs]

    return {
        "stem": clip["stem"], "fps": clip["fps"],
        "events": len(clip["events"]), "labels": len(clip["labels"]),
        "labels_with_gt_box": len(resolved), "labels_unresolved": unresolved,
        "events_unusable_for_eiou": unusable,
        "eiou": {"threshold": threshold, "matched": len(eiou_pairs), "pairs": pair_detail(detections, eiou_pairs)},
        "eiou_whole_pass": {"threshold": threshold, "matched": len(whole_pass_pairs),
                            "pairs": pair_detail(whole_pass, whole_pass_pairs)},
        "tolerance": {"value_s": tolerance, "matched_all_labels": len(all_pairs),
                      "matched_resolved_labels": len(resolved_pairs),
                      "pairs": [[predicted[i]["time_s"], clip["labels"][j]["time_s"]] for i, j in all_pairs]},
        "scores": [event.get("score") for event in clip["events"]],
        "eiou_hits": sorted({detections[pair["det"]]["event"] for pair in eiou_pairs}),
        "whole_pass_hits": sorted({whole_pass[pair["det"]]["event"] for pair in whole_pass_pairs}),
    }


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def summarise(clips, threshold=0.5, tolerance=1.25):
    """Totals over clips: the same events and the same labels scored by both instruments, plus AP when scores exist."""
    totals = {"events": 0, "labels": 0, "labels_with_gt_box": 0, "eiou_matched": 0, "whole_pass_matched": 0,
              "tolerance_matched_all_labels": 0, "tolerance_matched_resolved_labels": 0}
    for clip in clips:
        totals["events"] += clip["events"]
        totals["labels"] += clip["labels"]
        totals["labels_with_gt_box"] += clip["labels_with_gt_box"]
        totals["eiou_matched"] += clip["eiou"]["matched"]
        totals["whole_pass_matched"] += clip["eiou_whole_pass"]["matched"]
        totals["tolerance_matched_all_labels"] += clip["tolerance"]["matched_all_labels"]
        totals["tolerance_matched_resolved_labels"] += clip["tolerance"]["matched_resolved_labels"]

    scores = [score for clip in clips for score in clip["scores"]]
    # AP ranks detections by the literature-aligned match: the vehicle's whole crosswalk pass on both sides.
    hits = [index in clip["whole_pass_hits"] for clip in clips for index in range(clip["events"])]
    try:
        ap, ap_reason = average_precision(scores, hits), None
    except ValueError as error:
        ap, ap_reason = None, str(error)
    return {
        **totals,
        "eiou": {"threshold": threshold, "precision": rate(totals["eiou_matched"], totals["events"]),
                 "recall": rate(totals["eiou_matched"], totals["labels_with_gt_box"])},
        "eiou_whole_pass": {"threshold": threshold,
                            "precision": rate(totals["whole_pass_matched"], totals["events"]),
                            "recall": rate(totals["whole_pass_matched"], totals["labels_with_gt_box"])},
        "tolerance": {"value_s": tolerance,
                      "precision_all_labels": rate(totals["tolerance_matched_all_labels"], totals["events"]),
                      "recall_all_labels": rate(totals["tolerance_matched_all_labels"], totals["labels"]),
                      "precision_resolved_labels": rate(totals["tolerance_matched_resolved_labels"], totals["events"]),
                      "recall_resolved_labels": rate(totals["tolerance_matched_resolved_labels"],
                                                     totals["labels_with_gt_box"])},
        "ap": ap, "ap_reason": ap_reason,
    }


def format_rate(value):
    return "n/a" if value is None else f"{value:.3f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="a scripts/newclip_loop.py run directory (run.json + events)")
    parser.add_argument("--labels", required=True, help="folder of <stem>.jsonl human labels")
    parser.add_argument("--threshold", type=float, default=0.5, help="EIoU threshold, strict (default 0.5)")
    parser.add_argument("--tolerance", type=float, default=1.25, help="the fixed time tolerance (default 1.25 s)")
    parser.add_argument("--label-window", type=float, default=0.5,
                        help="seconds either side of a label when resolving its vehicle (default 0.5; 0 = its frame only)")
    parser.add_argument("--pass-gap", type=float, default=None,
                        help="seconds; on-crosswalk runs this close are one pass (default: the tracker's own "
                             "same_vehicle_gap_s, 0.5 s)")
    parser.add_argument("--json", help="write the full report here")
    args = parser.parse_args(argv)

    run_dir, labels_dir = Path(args.run), Path(args.labels)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    clips, bases = [], {"exact": 0, "window": 0}
    for entry in manifest["clips"]:
        labels_path = labels_dir / (entry["stem"] + ".jsonl")
        if not labels_path.exists():
            print(f"no labels for {entry['stem']}: {labels_path}", file=sys.stderr)
            return 2
        clip = load_clip(entry, labels_path, args.pass_gap)
        score = score_clip(clip, args.threshold, args.tolerance, args.label_window)
        clips.append(score)
        for row in ground_truths(clip, int(round(args.label_window * clip["fps"]))):
            if row["basis"]:
                bases[row["basis"]] += 1
        print(f"{entry['stem'][:20]:<22} events {score['events']:>3}  labels {score['labels']:>2}"
              f"  with GT box {score['labels_with_gt_box']:>2}"
              f"  EIoU {score['eiou']['matched']:>2}  whole-pass {score['eiou_whole_pass']['matched']:>2}"
              f"  {args.tolerance}s {score['tolerance']['matched_all_labels']:>2}"
              f"/{score['tolerance']['matched_resolved_labels']:>2}")
        for row in score["labels_unresolved"]:
            print(f"    unresolved label t={row['time_s']:>7.2f} {row['reason']}"
                  f" (on-crosswalk vehicles {row['vehicles_on_crosswalk']})")

    totals = summarise(clips, args.threshold, args.tolerance)
    print()
    print(f"events {totals['events']}  labels {totals['labels']}  labels with a ground-truth box "
          f"{totals['labels_with_gt_box']} (exact frame {bases['exact']}, within +-{args.label_window}s "
          f"{bases['window']})")
    print(f"EIoU @ {args.threshold} (strict, contact):   precision {format_rate(totals['eiou']['precision'])} "
          f"({totals['eiou_matched']}/{totals['events']})  recall {format_rate(totals['eiou']['recall'])} "
          f"({totals['eiou_matched']}/{totals['labels_with_gt_box']})")
    print(f"EIoU @ {args.threshold} (strict, whole pass): precision "
          f"{format_rate(totals['eiou_whole_pass']['precision'])} ({totals['whole_pass_matched']}/{totals['events']})"
          f"  recall {format_rate(totals['eiou_whole_pass']['recall'])} "
          f"({totals['whole_pass_matched']}/{totals['labels_with_gt_box']})")
    print(f"{args.tolerance}s (all labels):     precision "
          f"{format_rate(totals['tolerance']['precision_all_labels'])} "
          f"({totals['tolerance_matched_all_labels']}/{totals['events']})  recall "
          f"{format_rate(totals['tolerance']['recall_all_labels'])} "
          f"({totals['tolerance_matched_all_labels']}/{totals['labels']})")
    print(f"{args.tolerance}s (same {totals['labels_with_gt_box']} labels): precision "
          f"{format_rate(totals['tolerance']['precision_resolved_labels'])} "
          f"({totals['tolerance_matched_resolved_labels']}/{totals['events']})  recall "
          f"{format_rate(totals['tolerance']['recall_resolved_labels'])} "
          f"({totals['tolerance_matched_resolved_labels']}/{totals['labels_with_gt_box']})")
    print(f"AP @ EIoU {args.threshold}: " + (format_rate(totals["ap"]) if totals["ap"] is not None
                                            else f"not computable - {totals['ap_reason']}"))
    report = {"run": str(run_dir), "labels": str(labels_dir), "in_sample": manifest.get("role") == "dev",
              "threshold": args.threshold, "tolerance_s": args.tolerance, "label_window_s": args.label_window,
              "pass_gap_s": args.pass_gap,
              "totals": totals, "clips": clips}
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
