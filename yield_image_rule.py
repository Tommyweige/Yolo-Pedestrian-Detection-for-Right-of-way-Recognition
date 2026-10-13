"""Image-space "vehicle on the crosswalk while a pedestrian is on/at it" candidate events (assistive analysis).

No camera calibration, speed or stabilisation: works with a moving camera. Inputs per frame are a crosswalk mask
(from scripts/segment_crosswalk.py) plus person/vehicle boxes with track ids (scripts/track_video.py raw tracks).
A vehicle "yielding" never enters the crosswalk while a pedestrian is there, so yielding simply produces no event.
Each event records the pedestrian state (on_crosswalk | waiting_at_edge) so a stricter definition can filter later.
Output is a candidate for review, not a legal finding.
"""
import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

RULE_VERSION = "yield-image-0.1"
DISCLAIMER = "輔助分析，非法律違規認定"
VEHICLE_KINDS = {"car", "motorcycle", "bus", "truck"}


@dataclass
class Params:
    fps: float = 60.0
    src_size: tuple = (1920, 1080)
    min_component_fraction: float = 0.005   # smaller crosswalk blobs are noise or far away
    close_px: int = 15                      # bridges gaps cut by vehicles/pedestrians (mask pixels)
    edge_margin_person_heights: float = 0.8 # a pedestrian this close to the crosswalk is "waiting at the edge"
    lateral_vehicle_widths: float = 1.5     # pedestrian-to-vehicle horizontal gap limit
    min_seconds: float = 0.2                # condition must hold this long
    rider_overlap: float = 0.5              # person box this much inside a vehicle box = rider, not a pedestrian
    band_fraction: float = 0.5              # share of the band below the vehicle that must be crosswalk
    min_track_frames: int = 15
    evidence_s: float = 3.0


def crosswalk_components(mask, params):
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((params.close_px, params.close_px), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    minimum = params.min_component_fraction * mask.size
    return [(labels == i) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum]


def scaled(box, sx, sy):
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def is_rider(person, vehicles, params):
    px1, py1, px2, py2 = person
    area = max(1e-6, (px2 - px1) * (py2 - py1))
    for vx1, vy1, vx2, vy2 in vehicles:
        iw, ih = min(px2, vx2) - max(px1, vx1), min(py2, vy2) - max(py1, vy1)
        if iw > 0 and ih > 0 and iw * ih / area >= params.rider_overlap:
            return True
    return False


def person_state(box, component, distance, params):
    """'on_crosswalk', 'waiting_at_edge' or None for one person against one crosswalk component."""
    x = int(np.clip((box[0] + box[2]) / 2, 0, component.shape[1] - 1))
    y = int(np.clip(box[3], 0, component.shape[0] - 1))
    if component[y, x]:
        return "on_crosswalk"
    if distance[y, x] <= params.edge_margin_person_heights * (box[3] - box[1]):
        return "waiting_at_edge"
    return None


def vehicle_on(box, component, params):
    """True when the road just below the vehicle's bottom edge is crosswalk."""
    h, w = component.shape
    x1, x2 = int(np.clip(box[0] + 0.2 * (box[2] - box[0]), 0, w - 1)), int(np.clip(box[2] - 0.2 * (box[2] - box[0]), 1, w))
    y1 = int(np.clip(box[3] + 1, 0, h - 1))
    y2 = int(np.clip(box[3] + max(2, 0.04 * (box[3] - box[1])), y1 + 1, h))
    band = component[y1:y2, x1:max(x1 + 1, x2)]
    return band.size > 0 and band.mean() >= params.band_fraction


def frame_pairs(mask, people, vehicles, params):
    """Yield (vehicle_id, person_id, state) for one frame. Boxes are in mask pixels."""
    components = crosswalk_components(mask, params)
    vehicle_boxes = [b for _, _, b in vehicles]  # a person inside any vehicle box is a rider/occupant
    for component in components:
        distance = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)
        standing = [(i, b, person_state(b, component, distance, params)) for i, b in people if not is_rider(b, vehicle_boxes, params)]
        standing = [s for s in standing if s[2]]
        for vid, kind, vbox in vehicles:
            if not standing or not vehicle_on(vbox, component, params):
                continue
            vwidth = vbox[2] - vbox[0]
            for pid, pbox, state in standing:
                if abs((pbox[0] + pbox[2]) / 2 - (vbox[0] + vbox[2]) / 2) <= params.lateral_vehicle_widths * vwidth:
                    yield vid, pid, state


@dataclass
class Event:
    kind: str
    frame: int
    time_s: float
    vehicle_id: int
    vehicle_kind: str
    pedestrian_id: int
    pedestrian_state: str          # state in the first qualifying frame
    evidence_frames: tuple
    needs_review: bool
    review_reasons: list
    rule_version: str = RULE_VERSION
    disclaimer: str = DISCLAIMER

    def to_dict(self):
        record = asdict(self)
        record["evidence_frames"] = list(self.evidence_frames)
        return record


def detect(mask_at, tracks, params, frames):
    """mask_at(frame) -> bool array; tracks: list of {"id","kind","boxes":[[frame,x1,y1,x2,y2],...]}."""
    boxes = {}
    for t in tracks:
        for f, *b in t["boxes"]:
            boxes.setdefault(int(f), []).append((t["id"], t["kind"], tuple(b)))
    lengths = {t["id"]: len(t["boxes"]) for t in tracks}
    need = max(1, round(params.min_seconds * params.fps))
    streak, started, events, done = {}, {}, [], set()
    for f in frames:
        mask = mask_at(f)
        if mask is None:
            continue
        sx, sy = mask.shape[1] / params.src_size[0], mask.shape[0] / params.src_size[1]
        here = boxes.get(f, [])
        people = [(i, scaled(b, sx, sy)) for i, k, b in here if k == "person"]
        vehicles = [(i, k, scaled(b, sx, sy)) for i, k, b in here if k in VEHICLE_KINDS]
        kinds = {i: k for i, k, _ in vehicles}
        current = {}
        for vid, pid, state in frame_pairs(mask, people, vehicles, params):
            current.setdefault((vid, pid), state)
        for pair in list(streak):
            if pair not in current:
                del streak[pair]
        for pair, state in current.items():
            streak[pair] = streak.get(pair, 0) + 1
            started.setdefault(pair, (f, state))
            states = started[pair]
            if streak[pair] >= need and pair[0] not in done:
                done.add(pair[0])
                window = round(params.evidence_s * params.fps)
                reasons = []
                if lengths.get(pair[0], 0) < params.min_track_frames:
                    reasons.append("short_vehicle_track")
                events.append(Event("vehicle_on_crosswalk_with_pedestrian", f, round(f / params.fps, 3), pair[0], kinds[pair[0]],
                                    pair[1], states[1], (max(1, f - window), f + window), bool(reasons), reasons))
        for pair in [p for p in started if p not in streak]:
            del started[pair]
    return events


def load_mask_provider(path, shape=(360, 640)):
    data = np.load(path)
    frames = sorted(int(k[1:]) for k in data.files)
    size = shape[0] * shape[1]

    def mask_at(frame):
        usable = [f for f in frames if f <= frame]
        if not usable:
            return None
        return np.unpackbits(data[f"f{usable[-1]}"])[:size].astype(bool).reshape(shape)
    return mask_at, frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masks", required=True, help="masks.npz from segment_crosswalk.py --stride")
    parser.add_argument("--tracks", required=True, help="raw_tracks.json from track_video.py (with boxes)")
    parser.add_argument("--output", required=True, help="events .jsonl")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args(argv)
    params = Params(fps=args.fps, src_size=(args.width, args.height))
    mask_at, _ = load_mask_provider(args.masks)
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    last = max((int(b[0]) for t in tracks for b in t["boxes"]), default=0)
    events = detect(mask_at, tracks, params, range(1, last + 1))
    Path(args.output).write_text("".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
    print(json.dumps({"events": len(events)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
