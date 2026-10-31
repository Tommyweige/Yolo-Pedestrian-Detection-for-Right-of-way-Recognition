"""High-recall candidate generator: the rule's per-frame test, every (vehicle, pedestrian) contact, looser thresholds.

`yield_image_rule.detect` reports one event per vehicle (its earliest qualifying pedestrian) and drops pedestrians who
only waited at the edge. Both are judgments, and here the classifier makes them, so this keeps the rule's per-frame
geometry (`scan_frame`, duplicate-track merge, rider exclusion) and emits one candidate per contact. Each candidate also
carries per-frame measurements of the contact, used as the hand-made features of the lower-bound classifier.

Usage: python candidates.py --params '{"lateral_vehicle_widths": 4.0}' --out DIR   (writes DIR/<stem>.jsonl)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUN2, rule, stems, video_path  # noqa: E402

RELAXED = {"require_crossing": False}  # chosen by the recall sweep (README): the only knob that adds hits


def contact_measures(mask, vbox, pbox, params):
    """Geometry of one vehicle-pedestrian pair on one frame, in mask pixels: what the rule thresholds, unthresholded."""
    components = rule.crosswalk_components(mask, params)
    best = None
    for component in components:
        gap, extent = rule.along_gap(pbox, vbox, rule.crosswalk_axis(component))
        if best is None or gap / max(extent, 1e-6) < best[0]:
            best = (gap / max(extent, 1e-6), component)
    if best is None:
        return None
    rows, cols = mask.shape
    fx, fy = int(np.clip((pbox[0] + pbox[2]) / 2, 0, cols - 1)), int(np.clip(pbox[3], 0, rows - 1))
    depth = cv2.distanceTransform(best[1].astype(np.uint8), cv2.DIST_L2, 5)[fy, fx]
    return {"gap_widths": float(best[0]), "foot_depth_heights": float(depth / max(pbox[3] - pbox[1], 1e-6)),
            "vehicle_h": float(vbox[3] - vbox[1]), "person_h": float(pbox[3] - pbox[1]),
            "dy_bottom": float((pbox[3] - vbox[3]) / max(vbox[3] - vbox[1], 1e-6))}


def generate(stem, params_overrides):
    video = video_path(stem)
    capture = cv2.VideoCapture(str(video))
    fps, size = capture.get(cv2.CAP_PROP_FPS), (int(capture.get(3)), int(capture.get(4)))
    capture.release()
    params = rule.Params(fps=fps, src_size=size, **params_overrides)
    mask_at, mask_frames = rule.load_mask_provider(str(RUN2 / stem / "seg" / "masks.npz"))
    tracks = json.loads((RUN2 / stem / "tracks" / "raw_tracks.json").read_text(encoding="utf-8"))
    tracks = rule.merge_duplicate_tracks(tracks, params) if params.merge_duplicates else tracks
    riders = rule.rider_frames(tracks, params) if params.rider_exclusion else {}
    boxes = {}
    for t in tracks:
        for f, *b in t["boxes"]:
            boxes.setdefault(int(f), []).append((t["id"], t["kind"], tuple(b)))
    need = max(1, round(params.min_seconds * params.fps))
    window = round(params.entry_window_s * params.fps)
    last = max(boxes)
    streak, started, contacts, deep_frames, kinds, frames_of = {}, {}, {}, {}, {}, {}
    measured = set(mask_frames)  # masks exist every 3rd frame: measure geometry where the mask is fresh
    for f in range(1, last + 1):
        mask = mask_at(f)
        if mask is None:
            continue
        sx, sy = mask.shape[1] / size[0], mask.shape[0] / size[1]
        here = boxes.get(f, [])
        people = {i: rule.scaled(b, sx, sy) for i, k, b in here if k == "person"}
        vehicles = [(i, k, rule.scaled(b, sx, sy)) for i, k, b in here if k in rule.VEHICLE_KINDS]
        kinds.update({i: k for i, k, _ in vehicles})
        pairs, deep = rule.scan_frame(mask, list(people.items()), vehicles, params,
                                      [pid for pid, seats in riders.items() if f in seats])
        for pid in deep:
            deep_frames.setdefault(pid, []).append(f)
        current = {}
        for vid, pid, state in pairs:
            current.setdefault((vid, pid), state)
        for pair in list(streak):
            if pair not in current:
                del streak[pair], started[pair]
        vboxes = {i: b for i, _, b in vehicles}
        for pair, state in current.items():
            streak[pair] = streak.get(pair, 0) + 1
            started.setdefault(pair, (f, state))
            key = (pair, started[pair][0])
            if f in measured:
                frames_of.setdefault(key, []).append(
                    (f, state, contact_measures(mask, vboxes[pair[0]], people[pair[1]], params)))
            if streak[pair] == need:
                contacts[key] = [started[pair][0], f, state]
            elif streak[pair] > need:
                contacts[key][1] = f
    candidates = []
    for key, (start, end, state) in sorted(contacts.items(), key=lambda kv: kv[0][1]):
        (vid, pid), _ = key
        entered_at = [g for g in deep_frames.get(pid, []) if start - window <= g <= end + window]
        if params.require_crossing and not entered_at:
            continue
        frame = max(start, min(entered_at)) if entered_at else start
        rows = [m for _, _, m in frames_of.get(key, []) if m]
        states = [s for _, s, _ in frames_of.get(key, [])]
        candidates.append({
            "kind": "vehicle_on_crosswalk_with_pedestrian", "frame": frame, "time_s": round(frame / fps, 3),
            "vehicle_id": vid, "vehicle_kind": kinds[vid], "pedestrian_id": pid, "pedestrian_state": state,
            "pedestrian_entered_crosswalk": bool(entered_at), "contact_frames": [start, end],
            "features": {
                "contact_s": (end - start + 1) / fps,
                "on_share": states.count("on_crosswalk") / max(1, len(states)),
                "entered": float(bool(entered_at)),
                "entry_delay_s": (min(entered_at) - start) / fps if entered_at else 0.0,
                "min_gap_widths": min((m["gap_widths"] for m in rows), default=9.0),
                "mean_gap_widths": float(np.mean([m["gap_widths"] for m in rows])) if rows else 9.0,
                "max_foot_depth": max((m["foot_depth_heights"] for m in rows), default=0.0),
                "size_ratio": float(np.median([m["person_h"] / max(m["vehicle_h"], 1e-6) for m in rows])) if rows else 0.0,
                "dy_bottom": float(np.median([m["dy_bottom"] for m in rows])) if rows else 0.0,
                "is_motorcycle": float(kinds[vid] == "motorcycle"),
            },
        })
    return fps, candidates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", default=json.dumps(RELAXED))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stem in stems():
        fps, candidates = generate(stem, json.loads(args.params))
        (out / f"{stem}.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in candidates),
                                           encoding="utf-8")
        print(f"{stem[:16]} {fps:.2f}fps candidates {len(candidates)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
