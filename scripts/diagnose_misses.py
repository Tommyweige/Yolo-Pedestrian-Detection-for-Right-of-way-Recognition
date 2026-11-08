"""For each human-labelled event, report which stage of yield_image_rule blocks it (window = label time +- seconds).

Stages counted over the window, per frame: crosswalk found -> person present -> person on/at the crosswalk (non-rider)
-> a vehicle whose bottom edge is on the crosswalk -> vehicle and person close enough (a pair) -> pedestrian steps on within
entry_window_s. The first stage that is zero in every frame is the blocker.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import clip_artifacts  # noqa: E402
import yield_image_rule as r  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--window", type=float, default=1.0)
    args = parser.parse_args(argv)
    info = clip_artifacts.video_info(args.video)
    fps, (width, height) = info.fps, info.size
    params = r.Params(fps=fps, src_size=info.size)
    mask_at, _ = clip_artifacts.load_masks(args.masks)
    tracks = r.merge_duplicate_tracks(clip_artifacts.load_tracks(args.tracks), params)
    riders = r.rider_frames(tracks, params)
    boxes = clip_artifacts.boxes_by_frame(tracks)
    labels = [json.loads(l) for l in Path(args.labels).read_text(encoding="utf-8").splitlines() if l.strip()]
    for label in labels:
        centre = round(label["time_s"] * fps)
        counts = dict(frames=0, crosswalk=0, person=0, person_at_crosswalk=0, vehicle_on=0, pair=0, deep=0)
        for f in range(max(1, centre - round(args.window * fps)), centre + round(args.window * fps) + 1):
            mask = mask_at(f)
            if mask is None:
                continue
            counts["frames"] += 1
            sx, sy = mask.shape[1] / width, mask.shape[0] / height
            here = boxes.get(f, [])
            people = [(i, r.scaled(b, sx, sy)) for i, k, b in here if k == "person"]
            vehicles = [(i, k, r.scaled(b, sx, sy)) for i, k, b in here if k in clip_artifacts.VEHICLE_KINDS]
            components = r.crosswalk_components(mask, params)
            counts["crosswalk"] += bool(components)
            counts["person"] += bool(people)
            riding = [pid for pid, seats in riders.items() if f in seats]
            at, on = False, False
            for component in components:
                distance = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)
                depth = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
                standing = [s for s in ((i, b, r.person_state(b, component, distance, depth, params)) for i, b in people
                                        if i not in riding) if s[2]]
                at |= bool(standing)
                counts["deep"] += any(s[2] == "on_crosswalk" for s in standing)
                on |= any(r.vehicle_on(v[2], component, params) for v in vehicles)
            counts["person_at_crosswalk"] += at
            counts["vehicle_on"] += on
            pairs, _ = r.scan_frame(mask, people, vehicles, params, riding)
            counts["pair"] += bool(pairs)
        order = ["crosswalk", "person", "person_at_crosswalk", "vehicle_on", "pair"]
        blocker = next((s for s in order if counts[s] == 0), "none (all stages pass somewhere; check timing/entry filter)")
        print(json.dumps({"label_s": label["time_s"], **counts, "blocker": blocker}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
