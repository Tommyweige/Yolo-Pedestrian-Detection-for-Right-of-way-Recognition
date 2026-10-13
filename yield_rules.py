"""Vehicle-yield-to-pedestrian event rules for a FIXED camera (assistive analysis, not a legal finding).

Input: a scene calibration and tracks of foot points in image pixels. Output: events.
No torch/ultralytics dependency; detectors and the GUI feed tracks in through an adapter.

World frame (metres), derived from the crosswalk corners:
  x runs ALONG the crosswalk (pedestrians walk along x), y runs ALONG THE ROAD (vehicles drive along y).
  Crosswalk = rectangle [0, length_m] x [0, depth_m]. `vehicle_direction` +1 means vehicles drive toward
  increasing y, so the stop line is y = 0; -1 means the stop line is y = depth_m.
"""
import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

RULE_VERSION = "yield-rules-0.1"
DISCLAIMER = "輔助分析，非法律違規認定"
VEHICLE_KINDS = {"car", "motorcycle", "bus", "truck"}


@dataclass
class Params:
    fps: float = 30.0
    min_ped_seconds: float = 0.3      # pedestrian must stay on the crosswalk this long
    speed_window_s: float = 0.5       # speed is displacement over about this window
    stop_speed_mps: float = 1.0       # below this a vehicle counts as stopped
    min_stop_seconds: float = 0.5     # stopped this long before the line = yielded
    approach_range_m: float = 15.0    # vehicles farther than this from the line are ignored
    lateral_margin_m: float = 2.0     # pedestrian within this lateral distance of the vehicle = conflict
    ttc_lane_s: float = 2.0           # or pedestrian reaches the vehicle's lane within this time
    ped_recent_s: float = 1.0         # pedestrian still counts this long after leaving the crosswalk
    max_gap_frames: int = 10          # larger track gaps near the event mark it for review
    min_points: int = 15              # shorter tracks mark the event for review
    evidence_s: float = 3.0           # evidence window before/after the event


def homography(source, destination):
    """Direct linear transform mapping 4+ source points to destination points."""
    rows = []
    for (x, y), (u, v) in zip(source, destination):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    matrix = vt[-1].reshape(3, 3)
    return matrix / matrix[2, 2]


def apply_homography(matrix, point):
    x, y, w = matrix @ np.array([point[0], point[1], 1.0])
    return (x / w, y / w)


@dataclass
class Scene:
    crosswalk_image: list      # 4 image corners: A->(0,0), B->(length,0), C->(length,depth), D->(0,depth)
    length_m: float
    depth_m: float
    vehicle_direction: int = 1
    matrix: np.ndarray = field(init=False, repr=False)
    reprojection_error_px: float = field(init=False)

    def __post_init__(self):
        if len(self.crosswalk_image) != 4:
            raise ValueError("crosswalk_image needs exactly 4 corners")
        if self.vehicle_direction not in (1, -1):
            raise ValueError("vehicle_direction must be 1 or -1")
        if self.length_m <= 0 or self.depth_m <= 0:
            raise ValueError("crosswalk size must be positive")
        world = [(0, 0), (self.length_m, 0), (self.length_m, self.depth_m), (0, self.depth_m)]
        self.matrix = homography(self.crosswalk_image, world)
        back = np.linalg.inv(self.matrix)
        self.reprojection_error_px = float(max(
            math.dist(apply_homography(back, w), p) for w, p in zip(world, self.crosswalk_image)))
        if self.reprojection_error_px > 1.0:
            raise ValueError(f"degenerate calibration (reprojection error {self.reprojection_error_px:.2f}px)")

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["crosswalk_image"], data["length_m"], data["depth_m"], data.get("vehicle_direction", 1))

    def to_world(self, point):
        return apply_homography(self.matrix, point)

    def on_crosswalk(self, world):
        return 0 <= world[0] <= self.length_m and 0 <= world[1] <= self.depth_m

    def distance_to_line(self, world):
        """Positive while the vehicle is still before the stop line, <= 0 once it has passed."""
        return -world[1] if self.vehicle_direction == 1 else world[1] - self.depth_m


@dataclass
class Track:
    id: int
    kind: str
    points: dict   # frame -> (x_px, y_px) foot point

    @classmethod
    def from_dict(cls, data):
        return cls(int(data["id"]), data["kind"], {int(f): (float(x), float(y)) for f, x, y in data["points"]})


@dataclass
class Event:
    kind: str                  # yielded | suspected_failure_to_yield
    frame: int
    time_s: float
    vehicle_id: int
    vehicle_kind: str
    pedestrian_id: int
    vehicle_speed_mps: float
    distance_to_line_m: float
    evidence_frames: tuple
    needs_review: bool
    review_reasons: list
    rule_version: str = RULE_VERSION
    disclaimer: str = DISCLAIMER

    def to_dict(self):
        record = asdict(self)
        record["evidence_frames"] = list(self.evidence_frames)
        return record


class _World:
    """Per-track world positions with speed lookup."""

    def __init__(self, scene, track, params):
        self.track, self.params = track, params
        self.pos = {f: scene.to_world(p) for f, p in track.points.items()}
        self.frames = sorted(self.pos)

    def speed(self, frame):
        window = max(1, round(self.params.speed_window_s * self.params.fps))
        if frame not in self.pos:
            return None
        earlier = [f for f in self.frames if window * 0.5 <= frame - f <= window * 2]
        if not earlier:
            return None
        base = min(earlier, key=lambda f: abs(frame - f - window))
        return math.dist(self.pos[frame], self.pos[base]) * self.params.fps / (frame - base)

    def lateral_velocity(self, frame):
        """x velocity (m/s) of the pedestrian over the speed window."""
        window = max(1, round(self.params.speed_window_s * self.params.fps))
        earlier = [f for f in self.frames if window * 0.5 <= frame - f <= window * 2]
        if frame not in self.pos or not earlier:
            return 0.0
        base = min(earlier, key=lambda f: abs(frame - f - window))
        return (self.pos[frame][0] - self.pos[base][0]) * self.params.fps / (frame - base)

    def max_gap(self, lo, hi):
        inside = [f for f in self.frames if lo <= f <= hi]
        return max((b - a for a, b in zip(inside, inside[1:])), default=hi - lo)


def _pedestrian_state(scene, ped, params):
    """Frames where the pedestrian has been on the crosswalk for at least min_ped_seconds."""
    need = max(1, round(params.min_ped_seconds * params.fps))
    active, run_start, previous = set(), None, None
    for f in ped.frames:
        if scene.on_crosswalk(ped.pos[f]):
            if run_start is None or f - previous > params.max_gap_frames:
                run_start = f
            if f - run_start + 1 >= need:
                active.add(f)
        else:
            run_start = None
        previous = f
    return active


def _in_conflict(vehicle_pos, ped_pos, ped_vx, params):
    lateral = abs(vehicle_pos[0] - ped_pos[0])
    if lateral <= params.lateral_margin_m:
        return True
    toward = (vehicle_pos[0] - ped_pos[0]) * ped_vx > 0
    return toward and lateral / max(abs(ped_vx), 1e-6) <= params.ttc_lane_s


def detect_events(scene, tracks, params=None):
    params = params or Params()
    people = [_World(scene, t, params) for t in tracks if t.kind == "person"]
    vehicles = [_World(scene, t, params) for t in tracks if t.kind in VEHICLE_KINDS]
    active = {p.track.id: _pedestrian_state(scene, p, params) for p in people}
    recent = max(0, round(params.ped_recent_s * params.fps))
    stop_need = max(1, round(params.min_stop_seconds * params.fps))
    events = []
    for vehicle in vehicles:
        for ped in people:
            emitted, stopped_run = set(), 0
            for index, f in enumerate(vehicle.frames):
                if f not in ped.pos and not any(g in ped.pos for g in range(f - recent, f + 1)):
                    continue
                near = [g for g in range(f - recent, f + 1) if g in active[ped.track.id]]
                if not near:
                    stopped_run = 0
                    continue
                pf = f if f in ped.pos else max(g for g in range(f - recent, f + 1) if g in ped.pos)
                v_pos, speed = vehicle.pos[f], vehicle.speed(f)
                distance = scene.distance_to_line(v_pos)
                if speed is None or distance > params.approach_range_m:
                    stopped_run = 0
                    continue
                conflict = _in_conflict(v_pos, ped.pos[pf], ped.lateral_velocity(pf), params)
                if not conflict:
                    stopped_run = 0
                    continue
                if distance > 0 and speed < params.stop_speed_mps:
                    stopped_run += 1
                    if stopped_run >= stop_need and "yielded" not in emitted:
                        emitted.add("yielded")
                        events.append(_event("yielded", f, vehicle, ped, speed, distance, params))
                else:
                    stopped_run = 0
                prev = vehicle.frames[index - 1] if index else None
                crossed = (prev is not None and f - prev <= params.max_gap_frames
                           and scene.distance_to_line(vehicle.pos[prev]) > 0 >= distance)
                if crossed and speed >= params.stop_speed_mps and "suspected_failure_to_yield" not in emitted:
                    emitted.add("suspected_failure_to_yield")
                    events.append(_event("suspected_failure_to_yield", f, vehicle, ped, speed, distance, params))
    return sorted(events, key=lambda e: (e.frame, e.vehicle_id, e.pedestrian_id))


def _event(kind, frame, vehicle, ped, speed, distance, params):
    window = round(params.evidence_s * params.fps)
    lo, hi = frame - window, frame + window
    reasons = []
    if len(vehicle.frames) < params.min_points or len(ped.frames) < params.min_points:
        reasons.append("short_track")
    if max(vehicle.max_gap(lo, hi), ped.max_gap(lo, hi)) > params.max_gap_frames:
        reasons.append("track_gap_near_event")
    return Event(kind, frame, round(frame / params.fps, 3), vehicle.track.id, vehicle.track.kind, ped.track.id,
                 round(speed, 3), round(distance, 3), (max(0, lo), hi), bool(reasons), reasons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="scene JSON: crosswalk_image, length_m, depth_m, vehicle_direction")
    parser.add_argument("--tracks", required=True, help='JSON list of {"id","kind","points":[[frame,x,y],...]}')
    parser.add_argument("--output", required=True, help="events .jsonl")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)
    scene = Scene.from_json(args.scene)
    tracks = [Track.from_dict(t) for t in json.loads(Path(args.tracks).read_text(encoding="utf-8"))]
    events = detect_events(scene, tracks, Params(fps=args.fps))
    Path(args.output).write_text("".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in events),
                                 encoding="utf-8")
    print(json.dumps({"events": len(events), "reprojection_error_px": scene.reprojection_error_px}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
