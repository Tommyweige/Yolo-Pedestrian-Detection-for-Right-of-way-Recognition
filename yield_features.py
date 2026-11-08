"""Trajectory features for vehicle-yield-to-pedestrian candidate events.

Turns the tracking boxes around one candidate event into a fixed-size numeric
vector for a small downstream classifier. Only numpy and the standard library.

Boxes are (x1, y1, x2, y2) in source-video pixels; frames may be float in the
input and are cast with int(). A track's frames need not be consecutive.
"""
import math

import numpy as np

FEATURE_NAMES = [
    "is_car", "is_motorcycle", "is_bus", "is_truck",
    "ped_on", "entered",
    "v_h", "v_w", "v_bottom", "v_cx",
    "p_h", "p_bottom",
    "rel_dx", "rel_dy",
    "v_speed", "v_grow", "v_dy", "p_speed",
    "v_track_s", "p_track_s", "n_people",
]

EPS = 1e-6


def index_tracks(tracks):
    """{track_id: {"kind": str, "frames": int array sorted, "boxes": float array [N, 4]}}"""
    index = {}
    for track in tracks:
        rows = sorted((int(frame), box) for frame, *box in track["boxes"])
        frames = np.asarray([frame for frame, _ in rows], dtype=int)
        boxes = np.asarray([box for _, box in rows], dtype=float).reshape(-1, 4)
        index[int(track["id"])] = {"kind": track["kind"], "frames": frames, "boxes": boxes}
    return index


def _nearest(track, frame):
    """(frame, box) of the track sample closest in time; (None, None) when empty.

    Ties go to the earlier frame (frames are sorted, argmin returns the first).
    """
    frames = track["frames"]
    if len(frames) == 0:
        return None, None
    position = int(np.argmin(np.abs(frames - frame)))
    return int(frames[position]), track["boxes"][position]


def box_at(track, frame):
    """Box of this track at `frame`, else the temporally nearest one; None if empty."""
    _, box = _nearest(track, frame)
    return None if box is None else np.asarray(box, dtype=float)


def _geometry(box):
    x1, y1, x2, y2 = (float(value) for value in box)
    cx = (x1 + x2) / 2.0
    return {"w": x2 - x1, "h": y2 - y1, "cx": cx, "bottom_x": cx, "bottom_y": y2}


def _box_for(index, track_id, frame):
    box = box_at(index[track_id], frame)
    if box is None:
        raise KeyError(track_id)
    return box


def _seconds(frame_a, frame_b, fps):
    return abs(frame_b - frame_a) / max(fps, EPS)


def _people_in_frame(index, frame):
    return float(sum(1 for track in index.values()
                     if track["kind"] == "person" and np.any(track["frames"] == frame)))


def candidate_features(index, event, fps, src_size, people_by_frame=None):
    """Feature dict for one candidate event, keys in FEATURE_NAMES order."""
    frame = int(event["frame"])
    fps = float(fps)
    width, height = float(src_size[0]), float(src_size[1])
    window = int(round(0.5 * fps))

    vehicle_id, pedestrian_id = event["vehicle_id"], event["pedestrian_id"]
    vehicle_track, pedestrian_track = index[vehicle_id], index[pedestrian_id]

    v = _geometry(_box_for(index, vehicle_id, frame))
    p = _geometry(_box_for(index, pedestrian_id, frame))
    v_w, v_h, p_h = max(v["w"], EPS), max(v["h"], EPS), max(p["h"], EPS)

    # Vehicle motion between f - h and f + h (nearest available samples).
    v_lo_frame, v_lo_box = _nearest(vehicle_track, frame - window)
    v_hi_frame, v_hi_box = _nearest(vehicle_track, frame + window)
    v_seconds = _seconds(v_lo_frame, v_hi_frame, fps)
    if v_seconds <= 0:
        v_speed = v_grow = v_dy = 0.0
    else:
        lo, hi = _geometry(v_lo_box), _geometry(v_hi_box)
        centre_shift = math.hypot(hi["cx"] - lo["cx"], hi["bottom_y"] - lo["bottom_y"])
        v_speed = centre_shift / v_w / v_seconds
        v_grow = math.log(max(hi["h"], EPS) / max(lo["h"], EPS)) / v_seconds
        v_dy = (hi["bottom_y"] - lo["bottom_y"]) / v_h / v_seconds

    # Pedestrian motion over the same window.
    p_lo_frame, p_lo_box = _nearest(pedestrian_track, frame - window)
    p_hi_frame, p_hi_box = _nearest(pedestrian_track, frame + window)
    p_seconds = _seconds(p_lo_frame, p_hi_frame, fps)
    if p_seconds <= 0:
        p_speed = 0.0
    else:
        lo, hi = _geometry(p_lo_box), _geometry(p_hi_box)
        shift = math.hypot(hi["bottom_x"] - lo["bottom_x"], hi["bottom_y"] - lo["bottom_y"])
        p_speed = shift / p_h / p_seconds

    kind = event["vehicle_kind"]
    people = _people_in_frame(index, frame) if people_by_frame is None else float(people_by_frame.get(frame, 0))

    return {
        "is_car": float(kind == "car"),
        "is_motorcycle": float(kind == "motorcycle"),
        "is_bus": float(kind == "bus"),
        "is_truck": float(kind == "truck"),
        "ped_on": float(event["pedestrian_state"] == "on_crosswalk"),
        "entered": float(bool(event["pedestrian_entered_crosswalk"])),
        "v_h": v["h"] / max(height, EPS),
        "v_w": v["w"] / max(width, EPS),
        "v_bottom": v["bottom_y"] / max(height, EPS),
        "v_cx": v["cx"] / max(width, EPS),
        "p_h": p["h"] / max(height, EPS),
        "p_bottom": p["bottom_y"] / max(height, EPS),
        "rel_dx": abs(p["bottom_x"] - v["cx"]) / v_w,
        "rel_dy": (p["bottom_y"] - v["bottom_y"]) / p_h,
        "v_speed": v_speed,
        "v_grow": v_grow,
        "v_dy": v_dy,
        "p_speed": p_speed,
        "v_track_s": len(vehicle_track["frames"]) / max(fps, EPS),
        "p_track_s": len(pedestrian_track["frames"]) / max(fps, EPS),
        "n_people": people,
    }
