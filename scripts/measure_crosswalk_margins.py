"""Score the `cw_*` margins of `yield_image_rule.py --geometry crosswalk` against the crosswalk mask.

Without human event labels the only reference available is the mask itself: a box's ground point (bottom-centre) is
"on the crosswalk" when that pixel is inside the mask. For each margin the script prints how well the crosswalk
geometry reproduces that (precision / recall / F1, vehicles and pedestrians separately) and how many events the whole
rule produces, so the value can be picked from the curve instead of guessed:

    python scripts/measure_crosswalk_margins.py --video clip.mp4 --masks out/seg/masks.npz --tracks out/tracks/raw_tracks.json

- `cw_on_margin` and `cw_vehicle_margin` move precision and recall: a larger margin calls shallower feet and further
  ground points "on", so recall rises and precision falls.
- `cw_wait_margin` only labels pedestrians standing just outside the band and lets them pair; the crossing filter then
  drops them unless `--include-waiting` is used, so compare its event list too.

Read the numbers per camera: perspective, mask quality and box size differ per clip, and the defaults in `Params` are
one clip's numbers. masks.npz must come from `scripts/segment_crosswalk.py --stride` (packed 640x360 booleans).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import clip_artifacts  # noqa: E402
from crosswalk_space import fit_space, merge_components  # noqa: E402
from yield_image_rule import (  # noqa: E402
    Params, crosswalk_components, detect, person_state_in_crosswalk, rider_frames,
    scaled, vehicle_on_in_crosswalk,
)

GRIDS = {
    "cw_on_margin": (0.0, 0.02, 0.05, 0.08, 0.12, 0.2),
    "cw_vehicle_margin": (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3),
    "cw_wait_margin": (0.1, 0.25, 0.5, 0.75, 1.0),
}


def prepare(mask_at, mask_frames, tracks, params):
    """Per mask frame: the fitted crosswalk spaces, and each box with whether its ground point is on the mask.

    Riders are judged over time (`rider_frames`), the way the rule does, not frame by frame."""
    riding = rider_frames(tracks, params)
    by_frame = clip_artifacts.boxes_by_frame(tracks)
    prepared = []
    for frame in range(min(mask_frames), max(mask_frames) + 1):
        mask = mask_at(frame)
        if mask is None:
            continue
        sx, sy = mask.shape[1] / params.src_size[0], mask.shape[0] / params.src_size[1]
        here = by_frame.get(frame, [])
        boxes = []
        for track_id, kind, box in here:
            box = scaled(box, sx, sy)
            if kind in clip_artifacts.VEHICLE_KINDS:
                boxes.append(("vehicle", box, on_mask(mask, box)))
            elif kind == "person" and frame not in riding.get(track_id, ()):
                boxes.append(("pedestrian", box, on_mask(mask, box)))
        components = crosswalk_components(mask, params)
        spaces = [space for space in (fit_space(c) for c in merge_components(components)) if space is not None]
        prepared.append((len(components), spaces, boxes))
    return prepared


def on_mask(mask, box):
    x, y = int(round((box[0] + box[2]) / 2)), int(round(box[3]))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def score(prepared, params, kind):
    tp = fp = fn = 0
    for _, spaces, boxes in prepared:
        for box_kind, box, truth in boxes:
            if box_kind != kind:
                continue
            if kind == "vehicle":
                said = any(vehicle_on_in_crosswalk(space, box, params)[0] for space in spaces)
            else:
                said = any(person_state_in_crosswalk(space, box, params) == "on_crosswalk" for space in spaces)
            tp += said and truth
            fp += said and not truth
            fn += truth and not said
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 2), "recall": round(recall, 2),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 2)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--video", help="reads fps and size from it")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--json", help="also write the report here")
    args = parser.parse_args(argv)
    fps, width, height = args.fps, args.width, args.height
    if args.video:
        info = clip_artifacts.video_info(args.video)
        fps, width, height = fps or info.fps, width or info.width, height or info.height
    if not (fps and width and height):
        print("give --video, or all of --fps --width --height", file=sys.stderr)
        return 2
    base = Params(fps=fps, src_size=(width, height), geometry="crosswalk")
    mask_at, mask_frames = clip_artifacts.load_masks(args.masks)
    tracks = clip_artifacts.load_tracks(args.tracks)
    frames = range(min(mask_frames), max(mask_frames) + 1)
    prepared = prepare(mask_at, mask_frames, tracks, base)

    def events(params):
        found = detect(mask_at, tracks, params, frames)
        return [(e.frame, e.vehicle_id, e.pedestrian_id) for e in found]

    report = {"defaults": {"cw_on_margin": base.cw_on_margin, "cw_wait_margin": base.cw_wait_margin,
                           "cw_vehicle_margin": base.cw_vehicle_margin},
              "crosswalk_components_per_frame": dict(Counter(count for count, _, _ in prepared)),
              "on_crosswalk_ground_points_on_the_mask": {
                  kind: sum(truth for _, _, boxes in prepared for box_kind, _, truth in boxes if box_kind == kind)
                  for kind in ("vehicle", "pedestrian")},
              "default_events": events(base), "grids": {}}
    print(json.dumps({k: report[k] for k in ("defaults", "crosswalk_components_per_frame",
                                             "on_crosswalk_ground_points_on_the_mask", "default_events")}, indent=1))
    for name, values in GRIDS.items():
        kind = "vehicle" if name == "cw_vehicle_margin" else "pedestrian"
        print(f"\n-- {name} (scored against the mask for {kind}s, then the events of the whole rule)")
        report["grids"][name] = {}
        for value in values:
            params = replace(base, **{name: value})
            entry = {**score(prepared, params, kind), "events": events(params),
                     "events_include_waiting": events(replace(params, require_crossing=False))}
            report["grids"][name][str(value)] = entry
            print(f"  {value:<5} precision={entry['precision']:.2f} recall={entry['recall']:.2f} f1={entry['f1']:.2f} "
                  f"events={len(entry['events'])} with --include-waiting={len(entry['events_include_waiting'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
