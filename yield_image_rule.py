"""Image-space "vehicle on the crosswalk while a pedestrian is on/at it" candidate events (assistive analysis).

No camera calibration, speed or stabilisation: works with a moving camera. Inputs per frame are a crosswalk mask
(from scripts/segment_crosswalk.py) plus person/vehicle boxes with track ids (scripts/track_video.py raw tracks).
A vehicle "yielding" never enters the crosswalk while a pedestrian is there, so yielding simply produces no event.
A pedestrian merely waiting at the edge does not count (user decision). A pedestrian counts when they are on the crosswalk
(foot at least min_inside_person_heights inside the mask) at some frame of the contact, from its start to entry_window_s after
it ends, so stepping on as the vehicle passes is caught while standing still at the edge is not. Judging the whole contact
(rather than a window around its start) matters because a slow or late-stepping pedestrian reaches the mask after the
vehicle's first frame on it. The reported time is then the frame the pedestrian is on the crosswalk, which is what a human
labeller marks; contact_frames keeps the whole span. This uses frames after the contact: offline only.
--include-waiting disables that filter; each event records the pedestrian state at contact and whether they entered.
How contacts become events is a parameter: by default one event per vehicle (its earliest qualifying pedestrian), or with
`report_every_contact` every (vehicle, pedestrian) contact, repeats of one pedestrian inside a pass merged (`contact_dedup`,
default "pair"). Both are scored by the same scripts/eval_rule_events.py, so the two constructions can be compared on one
table, but their event counts are not comparable to each other (docs/yield-image-rule.md).
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
    lateral_vehicle_widths: float = 3.0     # pedestrian-to-vehicle gap along the crosswalk axis, in vehicle extents
                                            # (the law is ~3 m / one lane; 1.5 vehicle widths missed motorcycles entirely)
    min_seconds: float = 0.2                # condition must hold this long
    rider_margin: float = 0.15              # a rider's box jitters; inflate the motorcycle box by this share of person height
    rider_overlap: float = 0.5              # person box this much inside the inflated motorcycle box = riding on that frame
    rider_gap_s: float = 0.25               # a tracker dropout between two riding frames this short is still the ride
    rider_min_frames: int = 5               # ... over at least this many frames, or the person was never on the bike
    rider_kinds: tuple = ("motorcycle",)    # cars/buses/trucks hide their occupants, so a person overlapping them is a pedestrian
    rider_exclusion: bool = True            # off reproduces the pre-fix rule, so the evaluation can measure what it changed
    same_vehicle_iou: float = 0.45          # two vehicle tracks this aligned are one vehicle: on the labelled clips
                                            # the duplicates sit at median IoU 0.32-0.70. Lowering the bar to 0.30 merged
                                            # every duplicate but also two different vehicles, costing two confirmed
                                            # labels and one incident, so the bar stays where it does not.
    same_vehicle_gap_s: float = 0.5         # ... when the second track starts within this long of the first ending
    merge_duplicates: bool = True           # off reproduces the pre-fix rule, so the evaluation can measure what it changed
    band_fraction: float = 0.5              # share of the band below the vehicle that must be crosswalk
    wheel_line_fraction: float = 0.1        # the wheel line is the bottom of the box up to this fraction of its height
    flank_vehicle_widths: float = 0.5       # crosswalk beside the vehicle, this many vehicle widths on each side
    flank_fraction: float = 0.3             # share of the flank rows that must be crosswalk
    foot_window_person_heights: float = 0.3  # half-width of the window around the foot that must be crosswalk
    foot_surround_fraction: float = 0.75     # ... at least this share of it, to count as standing on the crosswalk
    min_track_frames: int = 15
    min_inside_person_heights: float = 0.03  # foot must be this deep inside the mask: mask edges are noisy, and a person
                                             # standing right at the edge is waiting. 0.03 is chosen in-sample: on the 7
                                             # labelled clips incidents go 11/15 at 0.05 and up (0.06 is already back at
                                             # the 0.1 numbers) and 12/15 with one event per vehicle at 0.03-0.05, but the
                                             # candidate-list construction (the default) gains its last withheld label and
                                             # incident only at 0.03 (8/9, 14/15 against 7/9, 13/15 at 0.04). 0.03 is the
                                             # edge of that passing shelf, not its middle: one step further and recall falls
                                             # off the cliff. It must be re-validated on clips that were not tuned on.
    report_every_contact: bool = False      # False: one event per vehicle (its earliest qualifying pedestrian). True: report
                                            # every (vehicle, pedestrian) contact and let `contact_dedup` collapse one
                                            # vehicle's pass - the "candidate list" construction, for comparison only.
    contact_dedup_s: float = 1.25           # candidate list only, one pass = a vehicle's contacts chained by report time within
                                            # this window (the evaluation's tolerance, so both constructions are scored alike).
                                            # 0 keeps every contact, which is how the raw candidate count is measured.
    contact_dedup: str = "pair"             # candidate list only, which contact of a pass is reported: "pair" (the default:
                                            # only repeats of the same (vehicle, pedestrian) pair collapse, so two
                                            # pedestrians of one pass are two reports), "earliest" (the pass's first
                                            # contact) or "on_crosswalk_longest" (the pedestrian already standing on the
                                            # crosswalk for the longest catch, falling back to the first contact). All
                                            # three are measured in docs/yield-image-rule.md: "earliest" and
                                            # "on_crosswalk_longest" score the same and drop a labelled crossing, "pair"
                                            # recovers it at the price of one report per pedestrian of a pass.
    require_crossing: bool = True           # user decision: a pedestrian merely waiting at the edge does not count
    entry_window_s: float = 1.0             # ... but one who steps onto the crosswalk during the contact (or this long after it) does
    evidence_s: float = 3.0


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


def is_rider(person, vehicle, params):
    """True when the person box is inside this vehicle box once the box is inflated by the tracker's jitter."""
    margin = params.rider_margin * (person[3] - person[1])
    inflated = (vehicle[0] - margin, vehicle[1] - margin, vehicle[2] + margin, vehicle[3] + margin)
    return covered(person, inflated) >= params.rider_overlap


def covered(box, region):
    """Share of `box` that lies inside `region`."""
    iw = min(box[2], region[2]) - max(box[0], region[0])
    ih = min(box[3], region[3]) - max(box[1], region[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))


def rider_frames(tracks, params):
    """{person id: frames on which that person is riding a motorcycle}.

    Judged over time, not on each frame alone: a rider's box jitters in and out of the motorcycle's box, and judging
    the frame alone let the rider count as a pedestrian on the frames it was out - the largest single source of
    unreasonable events on the labelled clips. So a frame is a riding frame when the person's box sits `rider_overlap`
    inside a motorcycle box inflated by `rider_margin` of the person's height, and a dropout between two riding frames
    up to `rider_gap_s` apart is filled in (the tracker lost the box for a moment; the person did not get off). A
    person needs `rider_min_frames` riding frames before any of this applies (someone brushing past a parked scooter in
    fewer frames is not riding it).

    The mask is per person and per moment - not per person, and not per (person, vehicle) pair. Someone sitting on a
    scooter is not a pedestrian in front of any vehicle right then; the same person walking across the road three
    seconds later is one. Judging the whole track at once hides that, and pairing the decision to one vehicle lets a
    rider of a scooter be counted as a pedestrian in front of the car next to it, which is how this rule first read.
    Buses, cars and trucks hide their occupants instead of being ridden, so only `rider_kinds` counts.
    """
    bikes = [t for t in tracks if t["kind"] in params.rider_kinds]
    frames = {t["id"]: {int(b[0]): tuple(b[1:]) for b in t["boxes"]} for t in tracks}
    gap = round(params.rider_gap_s * params.fps)
    riding = {}
    for person in tracks:
        if person["kind"] != "person":
            continue
        mine = frames[person["id"]]
        seats = set()
        for bike in bikes:
            theirs = frames[bike["id"]]
            seats.update(f for f in mine if f in theirs and is_rider(mine[f], theirs[f], params))
        seats = sorted(seats)
        if len(seats) < params.rider_min_frames:
            continue
        filled = set(seats)                       # a dropout between two riding frames is still the same ride
        for a, b in zip(seats, seats[1:]):
            if b - a <= gap:
                filled.update(range(a, b + 1))
        riding[person["id"]] = filled
    return riding


def box_iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def nearest_iou(a, b, frame):
    """IoU of a's box at `frame` with b's box at the same frame or one frame either side (tracks flicker)."""
    for g in (frame, frame - 1, frame + 1):
        if g in b:
            return box_iou(a[frame], b[g])
    return 0.0


def same_vehicle(a, b, params):
    """True when two vehicle tracks are one physical vehicle, given their boxes ({frame: box}).

    Two ways one vehicle ends up as two tracks: the tracker keeps a second ID for an object it already tracks
    (the tracks coexist, and each frame's boxes sit on top of each other), or it drops the ID and picks up a new one
    (the second track starts where the first ends, boxes lined up). Both are one vehicle, so one report, not two.
    """
    if not a or not b:
        return False
    fa, fb = (min(a), max(a)), (min(b), max(b))
    shorter = min(len(a), len(b))
    overlap = min(fa[1], fb[1]) - max(fa[0], fb[0])
    if overlap >= max(3, 0.3 * shorter):
        window = [f for f in a if max(fa[0], fb[0]) <= f <= min(fa[1], fb[1])]
        if not window:
            return False
        matched = [f for f in window if nearest_iou(a, b, f) > 0]
        if len(matched) < 0.6 * len(window):
            return False
        ious = sorted(nearest_iou(a, b, f) for f in matched)
        return ious[len(ious) // 2] >= params.same_vehicle_iou
    if overlap > 0.25 * shorter:
        return False   # they see each other too long to be the same object, and too briefly to be a hand-off
    gap = max(0, fb[0] - fa[1] if fb[0] > fa[1] else fa[0] - fb[1])
    if gap > round(params.same_vehicle_gap_s * params.fps):
        return False
    shared = sorted(set(a) & set(b))
    edges = [(f, f) for f in shared] or [(fa[1], fb[0])]
    return any(f in a and g in b and box_iou(a[f], b[g]) >= params.same_vehicle_iou for f, g in edges)


def vehicle_groups(tracks, params):
    """{track id: group key} for the vehicle tracks that are one physical vehicle (everything else groups alone)."""
    boxes = {t["id"]: {int(b[0]): tuple(b[1:]) for b in t["boxes"]} for t in tracks}
    parent = {t["id"]: t["id"] for t in tracks}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    vehicles = [t for t in tracks if t["kind"] in VEHICLE_KINDS]
    for i, a in enumerate(vehicles):
        for b in vehicles[i + 1:]:
            if a["kind"] != b["kind"] or find(a["id"]) == find(b["id"]):
                continue
            if same_vehicle(boxes[a["id"]], boxes[b["id"]], params):
                parent[find(b["id"])] = find(a["id"])
    return {t["id"]: find(t["id"]) for t in tracks}


def merge_duplicate_tracks(tracks, params):
    """One track per physical vehicle: merge tracks of the same kind that are one object (`same_vehicle`).

    The merged track keeps the longest fragment's id and kind, and the boxes of every fragment (the longer fragment
    wins on a shared frame). Without this the same car or motorcycle is reported twice.
    """
    groups = vehicle_groups(tracks, params)
    members = {}
    for t in tracks:
        members.setdefault(groups[t["id"]], []).append(t)
    merged = []
    for group in members.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        longest = max(group, key=lambda t: len(t["boxes"]))
        frames = {}
        for member in sorted(group, key=lambda t: len(t["boxes"])):
            for f, *b in member["boxes"]:
                frames[int(f)] = b
        merged.append({"id": longest["id"], "kind": longest["kind"],
                       "boxes": [[f, *frames[f]] for f in sorted(frames)]})
    return merged



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


def scan_frame(mask, people, vehicles, params, riding=()):
    """Return ([(vehicle_id, person_id, state)], {person ids deep on a crosswalk}) for one frame. Boxes in mask pixels.

    `riding` is the set of person ids riding a motorcycle on this frame (`rider_frames`); someone riding is not a
    pedestrian in front of anything.
    """
    components = crosswalk_components(mask, params)
    pairs, deep = [], set()
    for component in components:
        distance = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)
        depth = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        standing = [(i, b, person_state(b, component, distance, depth, params)) for i, b in people
                    if i not in riding]
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
    pedestrian_entered_crosswalk: bool  # on the crosswalk at some frame of the contact
    contact_frames: tuple          # first/last frame of the vehicle-pedestrian contact that qualified
    evidence_frames: tuple
    needs_review: bool
    review_reasons: list
    rule_version: str = RULE_VERSION
    disclaimer: str = DISCLAIMER

    def to_dict(self):
        record = asdict(self)
        record["evidence_frames"] = list(self.evidence_frames)
        record["contact_frames"] = list(self.contact_frames)
        return record


CONTACT_DEDUP_VARIANTS = ("earliest", "on_crosswalk_longest", "pair")


def contact_passes(candidates, params):
    """[[candidate event]]: one vehicle's contacts grouped into passes, a pass continuing while the next contact is within
    `contact_dedup_s` of the previous one. Contacts of *different* vehicles never group, so a pass stays one vehicle's
    crossing, and a slow crossing whose contacts chain past the window still counts as one pass."""
    passes, current = [], {}
    for event in sorted(candidates, key=lambda e: e.time_s):
        group = current.get(event.vehicle_id)
        if group is not None and event.time_s - group[-1].time_s <= params.contact_dedup_s:
            group.append(event)
        else:
            group = [event]
            passes.append(group)
            current[event.vehicle_id] = group
    return passes


def contact_frames_duration(event):
    return event.contact_frames[1] - event.contact_frames[0]


def dedupe_contacts(candidates, params):
    """The candidate list with repeats of one pedestrian inside a vehicle pass merged, by `params.contact_dedup` (see Params).

    The default (`pair`) keeps one report per pedestrian of a pass: the earliest contact of a pass is not necessarily the one
    the law or the labeller cares about, so a car crossing while someone waits at the edge and someone else is already
    walking on the crosswalk reports both, and the two share the vehicle's pass. Duplicate reports of one physical vehicle
    are the price of that, and they are expected (docs/yield-image-rule.md) - the duplicate count is only meaningful inside
    one construction, because the other construction reports one event per vehicle by definition.
    """
    if params.contact_dedup not in CONTACT_DEDUP_VARIANTS:
        raise ValueError(f"contact_dedup must be one of {CONTACT_DEDUP_VARIANTS}, not {params.contact_dedup!r}")
    if params.contact_dedup_s <= 0:
        return list(candidates)   # the raw candidate list
    kept = []
    for group in contact_passes(candidates, params):
        if params.contact_dedup == "pair":
            seen = set()
            for event in group:            # one report per (vehicle, pedestrian) pair; pedestrians are not merged
                if event.pedestrian_id not in seen:
                    seen.add(event.pedestrian_id)
                    kept.append(event)
        elif params.contact_dedup == "on_crosswalk_longest":
            on = [event for event in group if event.pedestrian_state == "on_crosswalk"]
            kept.append(max(on, key=contact_frames_duration) if on else group[0])
        else:
            kept.append(group[0])
    return sorted(kept, key=lambda e: e.time_s)


def detect(mask_at, tracks, params, frames):
    """mask_at(frame) -> bool array; tracks: list of {"id","kind","boxes":[[frame,x1,y1,x2,y2],...]}."""
    if params.merge_duplicates:                        # one vehicle per track, so it can be reported once
        tracks = merge_duplicate_tracks(tracks, params)
    riders = rider_frames(tracks, params) if params.rider_exclusion else {}  # a rider is not a pedestrian
    boxes = {}
    for t in tracks:
        for f, *b in t["boxes"]:
            boxes.setdefault(int(f), []).append((t["id"], t["kind"], tuple(b)))
    lengths = {t["id"]: len(t["boxes"]) for t in tracks}
    need = max(1, round(params.min_seconds * params.fps))
    window = round(params.entry_window_s * params.fps)
    streak, started, contacts, deep_frames, kinds = {}, {}, {}, {}, {}
    for f in frames:
        mask = mask_at(f)
        if mask is None:
            continue
        sx, sy = mask.shape[1] / params.src_size[0], mask.shape[0] / params.src_size[1]
        here = boxes.get(f, [])
        people = [(i, scaled(b, sx, sy)) for i, k, b in here if k == "person"]
        vehicles = [(i, k, scaled(b, sx, sy)) for i, k, b in here if k in VEHICLE_KINDS]
        kinds.update({i: k for i, k, _ in vehicles})
        pairs, deep = scan_frame(mask, people, vehicles, params,
                                 [pid for pid, seats in riders.items() if f in seats])
        for pid in deep:
            deep_frames.setdefault(pid, []).append(f)
        current = {}
        for vid, pid, state in pairs:
            current.setdefault((vid, pid), state)
        for pair in list(streak):
            if pair not in current:
                del streak[pair], started[pair]
        for pair, state in current.items():
            streak[pair] = streak.get(pair, 0) + 1
            started.setdefault(pair, (f, state))
            if streak[pair] == need:  # first frame at which the contact qualifies: open its span
                contacts[(pair, started[pair][0])] = [started[pair][0], f, state]
            elif streak[pair] > need:
                contacts[(pair, started[pair][0])][1] = f
    candidates = []
    for (pair, _), (start, end, state) in sorted(contacts.items(), key=lambda kv: kv[0][1]):
        vid, pid = pair
        entered_at = [g for g in deep_frames.get(pid, []) if start - window <= g <= end + window]
        if params.require_crossing and not entered_at:
            continue  # stood at the edge the whole time: waiting, not crossing (does not use up the vehicle)
        frame = max(start, min(entered_at)) if entered_at else start
        span = round(params.evidence_s * params.fps)
        reasons = ["short_vehicle_track"] if lengths.get(vid, 0) < params.min_track_frames else []
        candidates.append(Event("vehicle_on_crosswalk_with_pedestrian", frame, round(frame / params.fps, 3), vid,
                                kinds[vid], pid, state, bool(entered_at), (start, end),
                                (max(1, frame - span), frame + span), bool(reasons), reasons))
    if params.report_every_contact:
        return dedupe_contacts(candidates, params)
    events, reported = [], set()
    for event in candidates:  # contact-start order, so this is the vehicle's earliest qualifying pedestrian
        if event.vehicle_id in reported:
            continue  # one event per vehicle
        reported.add(event.vehicle_id)
        events.append(event)
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
                    require_crossing=not args.include_waiting)
    mask_at, _ = load_mask_provider(args.masks)
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    last = max((int(b[0]) for t in tracks for b in t["boxes"]), default=0)
    events = detect(mask_at, tracks, params, range(1, last + 1))
    Path(args.output).write_text("".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
    print(json.dumps({"events": len(events)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
