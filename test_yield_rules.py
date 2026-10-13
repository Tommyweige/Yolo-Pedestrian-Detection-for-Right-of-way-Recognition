"""yield_rules regressions on synthetic trajectories (no GPU, no torch)."""
import json
from pathlib import Path
import tempfile
import unittest

from yield_rules import Params, Scene, Track, apply_homography, detect_events, main

FPS = 30
PX = 50  # image pixels per metre in the simple scene


def simple_scene(direction=1):
    corners = [(100, 100), (600, 100), (600, 300), (100, 300)]  # 10 m x 4 m crosswalk
    return Scene(corners, 10.0, 4.0, direction)


def to_image(x, y):
    return (100 + PX * x, 100 + PX * y)


def track(identity, kind, world_by_frame):
    return Track(identity, kind, {f: to_image(*xy) for f, xy in world_by_frame.items()})


def walker(identity=1, frames=range(0, 181), x0=2.0, speed=1.0, y=2.0):
    return track(identity, "person", {f: (x0 + speed * f / FPS, y) for f in frames})


def straight_vehicle(identity=10, kind="car", x=5.0, frames=range(0, 121), direction=1):
    start = -20.0 if direction == 1 else 24.0
    return track(identity, kind, {f: (x, start + direction * 10 * f / FPS) for f in frames})


def stopping_vehicle(identity=10, x=5.0):
    points, y = {}, -20.0
    for f in range(0, 241):
        points[f] = (x, y)
        if f < 30:
            v = 10.0
        elif f < 60:
            v = 10.0 * (60 - f) / 30
        elif f < 200:
            v = 0.0
        else:
            v = 3.0 * (f - 200) / FPS
        y += v / FPS
    return track(identity, "car", points)


class YieldRules(unittest.TestCase):
    def test_calibration_maps_corners_and_handles_perspective(self):
        scene = Scene([(120, 80), (580, 90), (640, 310), (60, 300)], 10.0, 4.0)
        for image, world in zip(scene.crosswalk_image, [(0, 0), (10, 0), (10, 4), (0, 4)]):
            x, y = scene.to_world(image)
            self.assertAlmostEqual(x, world[0], places=6)
            self.assertAlmostEqual(y, world[1], places=6)
        self.assertTrue(scene.on_crosswalk(scene.to_world((350, 200))))
        with self.assertRaises(ValueError):
            Scene([(0, 0), (1, 1), (2, 2), (3, 3)], 10, 4)  # collinear corners

    def test_vehicle_driving_through_is_flagged(self):
        events = detect_events(simple_scene(), [walker(), straight_vehicle()])
        failures = [e for e in events if e.kind == "suspected_failure_to_yield"]
        self.assertEqual(len(failures), 1)
        self.assertLessEqual(abs(failures[0].frame - 60), 2)
        self.assertGreater(failures[0].vehicle_speed_mps, 9)
        self.assertIn("輔助", failures[0].disclaimer)
        self.assertFalse(failures[0].needs_review)

    def test_vehicle_that_stops_is_yielded_not_flagged(self):
        events = detect_events(simple_scene(), [walker(), stopping_vehicle()])
        self.assertEqual([e.kind for e in events], ["yielded"])

    def test_no_pedestrian_on_crosswalk_no_event(self):
        kerb = track(1, "person", {f: (5.0, -2.0) for f in range(0, 181)})
        self.assertEqual(detect_events(simple_scene(), [kerb, straight_vehicle()]), [])

    def test_pedestrian_far_from_vehicle_lane_no_event(self):
        parked_far = track(1, "person", {f: (9.5, 2.0) for f in range(0, 181)})  # stationary, 4.5 m away
        self.assertEqual(detect_events(simple_scene(), [parked_far, straight_vehicle()]), [])

    def test_reverse_direction_uses_far_edge_as_stop_line(self):
        events = detect_events(simple_scene(-1), [walker(), straight_vehicle(direction=-1)])
        self.assertEqual([e.kind for e in events], ["suspected_failure_to_yield"])

    def test_multiple_vehicles_each_get_one_event(self):
        events = detect_events(simple_scene(), [walker(), straight_vehicle(10), straight_vehicle(11, "motorcycle")])
        self.assertEqual(sorted(e.vehicle_id for e in events), [10, 11])

    def test_short_track_is_marked_for_review(self):
        short = straight_vehicle(frames=range(50, 63))
        events = detect_events(simple_scene(), [walker(), short])
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].needs_review)
        self.assertIn("short_track", events[0].review_reasons)

    def test_track_gap_prevents_false_crossing_claim(self):
        frames = [f for f in range(0, 121) if not 45 <= f <= 75]
        self.assertEqual(detect_events(simple_scene(), [walker(), straight_vehicle(frames=frames)]), [])

    def test_ignores_non_vehicle_kinds_and_is_deterministic(self):
        bicycle = straight_vehicle(kind="bicycle")
        self.assertEqual(detect_events(simple_scene(), [walker(), bicycle]), [])
        a = detect_events(simple_scene(), [walker(), straight_vehicle()])
        b = detect_events(simple_scene(), [straight_vehicle(), walker()])
        self.assertEqual([e.to_dict() for e in a], [e.to_dict() for e in b])

    def test_cli_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scene.json").write_text(json.dumps({
                "crosswalk_image": [(100, 100), (600, 100), (600, 300), (100, 300)],
                "length_m": 10, "depth_m": 4, "vehicle_direction": 1}))
            tracks = [{"id": t.id, "kind": t.kind, "points": [[f, *p] for f, p in t.points.items()]}
                      for t in (walker(), straight_vehicle())]
            (root / "tracks.json").write_text(json.dumps(tracks))
            self.assertEqual(main(["--scene", str(root / "scene.json"), "--tracks", str(root / "tracks.json"),
                                   "--output", str(root / "events.jsonl")]), 0)
            rows = [json.loads(l) for l in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["kind"] for r in rows], ["suspected_failure_to_yield"])


if __name__ == "__main__":
    unittest.main()
