"""Image-space "vehicle on the crosswalk while a pedestrian is on/at it" candidate events (assistive analysis).

No camera calibration, speed or stabilisation: works with a moving camera. Inputs per frame are a crosswalk mask
(from scripts/segment_crosswalk.py) plus person/vehicle boxes with track ids (scripts/track_video.py raw tracks).
A vehicle "yielding" never enters the crosswalk while a pedestrian is there, so yielding simply produces no event.
A pedestrian merely waiting at the edge does not count (user decision). A pedestrian counts when they are on the crosswalk
(foot at least min_inside_person_heights inside the mask) at some frame within entry_window_s of the contact, so stepping on
as the vehicle passes is caught while standing still at the edge is not. This uses frames after the contact: offline only.
--include-waiting disables that filter; each event records the pedestrian state at contact and whether they entered.
Output is a candidate for review, not a legal finding.
"""
import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from crosswalk_space import fit_space, merge_components

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
    lateral_vehicle_widths: float = 1.5     # pedestrian-to-vehicle gap along the crosswalk axis, in vehicle extents
    min_seconds: float = 0.2                # condition must hold this long
    rider_overlap: float = 0.5              # person box this much inside a motorcycle box = rider, not a pedestrian
    rider_kinds: tuple = ("motorcycle",)    # cars/buses/trucks hide their occupants, so a person overlapping them is a pedestrian
    band_fraction: float = 0.5              # share of the band below the vehicle that must be crosswalk
    wheel_line_fraction: float = 0.1        # the wheel line is the bottom of the box up to this fraction of its height
    flank_vehicle_widths: float = 0.5       # crosswalk beside the vehicle, this many vehicle widths on each side
    flank_fraction: float = 0.3             # share of the flank rows that must be crosswalk
    foot_window_person_heights: float = 0.3  # half-width of the window around the foot that must be crosswalk
    foot_surround_fraction: float = 0.75     # ... at least this share of it, to count as standing on the crosswalk
    min_track_frames: int = 15
    min_inside_person_heights: float = 0.1   # foot must be this deep inside the mask: mask edges are noisy, and a person
                                             # standing right at the edge is waiting (chosen between the two clips' cases 0.03 and 0.25)
    require_crossing: bool = True           # user decision: a pedestrian merely waiting at the edge does not count
    entry_window_s: float = 1.0             # ... but one who steps onto the crosswalk within this window of the contact does
    evidence_s: float = 3.0
    geometry: str = "image"                  # "image": pixel strips and person heights; "crosswalk": coordinates of the fitted crosswalk
    cw_on_margin: float = 0.05               # a foot this deep (crosswalk widths) inside either long edge is on it: the image rule's
                                             # 0.1 person heights over a 3-4 m crossing, and above the foot jitter of 0.013
    cw_wait_margin: float = 0.5              # ... and this close outside it is waiting at the edge; 0.1-0.5 gave the same events on
                                             # the test clip and 0.75 gained a pair, so this is the widest value in the stable range
    cw_vehicle_margin: float = 0.02          # a vehicle's ground point this far (widths) outside the band still counts as touching it:
                                             # against the mask, precision falls from 0.71 at 0 to 0.63 at 0.05 with no recall gain


def crosswalk_components(mask, params):
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((params.close_px, params.close_px), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    minimum = params.min_component_fraction * mask.size
    return [(labels == i) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum]


def crosswalk_axis(component):
    """Unit vector (x, y) along the crosswalk (the direction pedestrians walk): the blob's principal axis.

    Works when the camera turns and the crosswalk is no longer horizontal in the image. A blob that is not elongated
    (axis ratio < 1.3) falls back to horizontal.
    """
    ys, xs = np.nonzero(component)
    if len(xs) < 10:
        return 1.0, 0.0
    values, vectors = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
    if np.sqrt(values[1] / max(values[0], 1e-9)) < 1.3:
        return 1.0, 0.0
    ux, uy = vectors[:, 1]
    return (float(ux), float(uy)) if ux >= 0 else (float(-ux), float(-uy))


def along_gap(pbox, vbox, axis):
    """Distance between pedestrian foot and vehicle centre along the axis, and the vehicle's extent along it."""
    ux, uy = axis
    foot = ((pbox[0] + pbox[2]) / 2, pbox[3])
    centre = ((vbox[0] + vbox[2]) / 2, (vbox[1] + vbox[3]) / 2)
    extent = abs(ux) * (vbox[2] - vbox[0]) + abs(uy) * (vbox[3] - vbox[1])
    return abs(ux * (centre[0] - foot[0]) + uy * (centre[1] - foot[1])), extent


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


def person_state(box, component, distance, depth, params):
    """'on_crosswalk', 'waiting_at_edge' or None. distance: to the crosswalk from outside; depth: from the edge inside.

    The segmentation labels the pedestrian's own pixels as person, so the foot pixel itself is often not crosswalk even
    when they stand in the middle of it: a foot with crosswalk all around it counts as on the crosswalk.
    """
    rows, cols = component.shape
    x = int(np.clip((box[0] + box[2]) / 2, 0, cols - 1))
    y = int(np.clip(box[3], 0, rows - 1))
    height = box[3] - box[1]
    rx, ry = max(2, int(params.foot_window_person_heights * height)), max(1, int(0.1 * height))
    around = component[max(0, y - ry):min(rows, y + ry + 1), max(0, x - rx):min(cols, x + rx + 1)]
    if around.size and around.mean() >= params.foot_surround_fraction:
        return "on_crosswalk"
    if component[y, x]:
        return "on_crosswalk" if depth[y, x] >= params.min_inside_person_heights * height else "waiting_at_edge"
    if distance[y, x] <= params.edge_margin_person_heights * height:
        return "waiting_at_edge"
    return None


def vehicle_on(box, component, params):
    """True when the vehicle is on the crosswalk: the road just below its bottom edge is crosswalk, or the crosswalk
    continues beside it at its wheel line (the segmentation never labels the crosswalk under the vehicle itself)."""
    h, w = component.shape
    width, height = box[2] - box[0], box[3] - box[1]
    x1, x2 = int(np.clip(box[0] + 0.2 * width, 0, w - 1)), int(np.clip(box[2] - 0.2 * width, 1, w))
    y1 = int(np.clip(box[3] + 1, 0, h - 1))
    y2 = int(np.clip(box[3] + max(2, 0.04 * height), y1 + 1, h))
    band = component[y1:y2, x1:max(x1 + 1, x2)]
    if band.size > 0 and band.mean() >= params.band_fraction:
        return True
    wy1 = int(np.clip(box[3] - params.wheel_line_fraction * height, 0, h - 1))
    wy2 = int(np.clip(box[3] + 0.04 * height + 1, wy1 + 1, h))
    reach = params.flank_vehicle_widths * width
    for fx1, fx2 in ((box[0] - reach, box[0]), (box[2], box[2] + reach)):
        flank = component[wy1:wy2, int(np.clip(fx1, 0, w - 1)):max(int(np.clip(fx1, 0, w - 1)) + 1, int(np.clip(fx2, 1, w)))]
        if flank.size > 0 and flank.mean() >= params.flank_fraction:
            return True
    return False


def person_state_in_crosswalk(space, box, params):
    """'on_crosswalk', 'waiting_at_edge' or None from the foot's position in crosswalk coordinates."""
    x, y = (box[0] + box[2]) / 2, box[3]
    length, width = space.to_crosswalk(x, y)
    if not space.along(x, y, length, 0.05):
        return None   # beside the crosswalk's end, not at it
    if params.cw_on_margin <= width <= 1 - params.cw_on_margin:
        return "on_crosswalk"
    if -params.cw_wait_margin <= width <= 1 + params.cw_wait_margin:
        return "waiting_at_edge"
    return None


def vehicle_on_in_crosswalk(space, box, params):
    """Ground contact of the vehicle (bottom centre of its box) inside the band; returns (on, L centre, L extent)."""
    (l1, _), (l2, _), (lc, wc) = (space.to_crosswalk(box[0], box[3]), space.to_crosswalk(box[2], box[3]),
                                  space.to_crosswalk((box[0] + box[2]) / 2, box[3]))
    inside = -params.cw_vehicle_margin <= wc <= 1 + params.cw_vehicle_margin
    along = space.along((box[0] + box[2]) / 2, box[3], lc, 0.1)
    return inside and along, lc, abs(l2 - l1)


def scan_component_in_crosswalk(component, people, vehicles, params):
    space = fit_space(component)
    if space is None:
        return None
    standing = [(i, b, person_state_in_crosswalk(space, b, params)) for i, b in people]
    standing = [s for s in standing if s[2]]
    pairs = []
    for vid, kind, vbox in vehicles:
        on, v_centre, extent = vehicle_on_in_crosswalk(space, vbox, params)
        if not (on and standing):
            continue
        for pid, pbox, state in standing:
            length, _ = space.to_crosswalk((pbox[0] + pbox[2]) / 2, pbox[3])
            if abs(length - v_centre) <= params.lateral_vehicle_widths * max(extent, 1e-6):
                pairs.append((vid, pid, state))
    return standing, pairs


def scan_frame(mask, people, vehicles, params):
    """Return ([(vehicle_id, person_id, state)], {person ids deep on a crosswalk}) for one frame. Boxes in mask pixels."""
    components = crosswalk_components(mask, params)
    if params.geometry == "crosswalk":
        components = merge_components(components)
    vehicle_boxes = [b for _, k, b in vehicles if k in params.rider_kinds]  # a person inside a motorcycle box is a rider
    pairs, deep = [], set()
    for component in components:
        if params.geometry == "crosswalk":
            found = scan_component_in_crosswalk(component, [(i, b) for i, b in people if not is_rider(b, vehicle_boxes, params)],
                                                vehicles, params)
            if found is not None:
                standing, component_pairs = found
                deep.update(i for i, _, state in standing if state == "on_crosswalk")
                pairs.extend(component_pairs)
                continue
        distance = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)
        depth = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        standing = [(i, b, person_state(b, component, distance, depth, params)) for i, b in people
                    if not is_rider(b, vehicle_boxes, params)]
        standing = [s for s in standing if s[2]]
        deep.update(i for i, _, state in standing if state == "on_crosswalk")
        axis = crosswalk_axis(component)
        for vid, kind, vbox in vehicles:
            if not standing or not vehicle_on(vbox, component, params):
                continue
            for pid, pbox, state in standing:
                gap, extent = along_gap(pbox, vbox, axis)
                if gap <= params.lateral_vehicle_widths * extent:
                    pairs.append((vid, pid, state))
    return pairs, deep


@dataclass
class Event:
    kind: str
    frame: int
    time_s: float
    vehicle_id: int
    vehicle_kind: str
    pedestrian_id: int
    pedestrian_state: str          # state in the first qualifying frame
    pedestrian_entered_crosswalk: bool  # on the crosswalk at some frame within entry_window_s of the contact
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
    window = round(params.entry_window_s * params.fps)
    streak, started, candidates, done, deep_frames = {}, {}, [], set(), {}
    for f in frames:
        mask = mask_at(f)
        if mask is None:
            continue
        sx, sy = mask.shape[1] / params.src_size[0], mask.shape[0] / params.src_size[1]
        here = boxes.get(f, [])
        people = [(i, scaled(b, sx, sy)) for i, k, b in here if k == "person"]
        vehicles = [(i, k, scaled(b, sx, sy)) for i, k, b in here if k in VEHICLE_KINDS]
        kinds = {i: k for i, k, _ in vehicles}
        pairs, deep = scan_frame(mask, people, vehicles, params)
        for pid in deep:
            deep_frames.setdefault(pid, []).append(f)
        current = {}
        for vid, pid, state in pairs:
            current.setdefault((vid, pid), state)
        for pair in list(streak):
            if pair not in current:
                del streak[pair]
        for pair, state in current.items():
            streak[pair] = streak.get(pair, 0) + 1
            started.setdefault(pair, (f, state))
            if streak[pair] >= need and pair not in done:  # one candidate per (vehicle, pedestrian); the crossing filter runs below
                done.add(pair)
                reasons = ["short_vehicle_track"] if lengths.get(pair[0], 0) < params.min_track_frames else []
                candidates.append((started[pair][0], pair, kinds[pair[0]], started[pair][1], reasons))  # time = first frame of contact, not when min_seconds was met
        for pair in [p for p in started if p not in streak]:
            del started[pair]
    events, reported = [], set()
    for f, (vid, pid), kind, state, reasons in sorted(candidates, key=lambda c: c[0]):
        if vid in reported:
            continue  # one event per vehicle: its earliest qualifying pedestrian
        entered = any(abs(g - f) <= window for g in deep_frames.get(pid, []))
        if params.require_crossing and not entered:
            continue  # stood at the edge the whole time: waiting, not crossing (does not use up the vehicle)
        reported.add(vid)
        span = round(params.evidence_s * params.fps)
        events.append(Event("vehicle_on_crosswalk_with_pedestrian", f, round(f / params.fps, 3), vid, kind, pid, state,
                            entered, (max(1, f - span), f + span), bool(reasons), reasons))
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
    parser.add_argument("--include-waiting", action="store_true", help="also report pedestrians who only waited at the edge (disables the crossing filter)")
    parser.add_argument("--geometry", choices=("image", "crosswalk"), default="image", help="judge in image pixels (default) or in crosswalk coordinates")
    parser.add_argument("--video", help="source video: its fps and size are read from it (time_s and all thresholds depend on them)")
    parser.add_argument("--fps", type=float, help="required unless --video is given")
    parser.add_argument("--width", type=int, help="required unless --video is given")
    parser.add_argument("--height", type=int, help="required unless --video is given")
    args = parser.parse_args(argv)
    fps, width, height = args.fps, args.width, args.height
    if args.video:
        capture = cv2.VideoCapture(args.video)
        fps, width, height = fps or capture.get(cv2.CAP_PROP_FPS), width or int(capture.get(3)), height or int(capture.get(4))
        capture.release()
    if not (fps and width and height):
        print("Give --video, or all of --fps --width --height (a wrong fps silently shifts every event time)", file=sys.stderr)
        return 2
    params = Params(fps=fps, src_size=(width, height),
                    require_crossing=not args.include_waiting, geometry=args.geometry)
    mask_at, _ = load_mask_provider(args.masks)
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    last = max((int(b[0]) for t in tracks for b in t["boxes"]), default=0)
    events = detect(mask_at, tracks, params, range(1, last + 1))
    Path(args.output).write_text("".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
    print(json.dumps({"events": len(events)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
