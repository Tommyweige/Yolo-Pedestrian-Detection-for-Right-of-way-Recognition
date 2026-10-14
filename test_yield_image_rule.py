"""yield_image_rule regressions with synthetic masks and boxes (no GPU)."""
import unittest

import cv2
import numpy as np

from yield_image_rule import Params, detect

H, W = 360, 640
SRC = (1920, 1080)  # boxes below are in source pixels; masks are 640x360 (scale 1/3)
P = Params(fps=60.0, src_size=SRC, min_seconds=0.1)


def mask_with(*rects):
    m = np.zeros((H, W), bool)
    for y1, y2, x1, x2 in rects:
        m[y1:y2, x1:x2] = True
    return m


NEAR = (200, 300, 20, 620)    # near crosswalk (mask pixels)
FAR = (60, 80, 150, 450)      # small far crosswalk


def track(identity, kind, box, frames):
    return {"id": identity, "kind": kind, "boxes": [[f, *box] for f in frames]}


def run(tracks, mask=None, frames=range(1, 31), params=P):
    mask = mask if mask is not None else mask_with(NEAR)
    return detect(lambda f: mask, tracks, params, frames)


# source-pixel boxes. Crosswalk rows 200-300 in mask px == 600-900 in source px.
PED_ON = (1230, 700, 1290, 850)          # beside the car (not inside its box); foot (1260, 850) is on the crosswalk
CAR_ON = (700, 500, 1000, 870)           # bottom 870 -> band (870..) inside crosswalk (rows 600-900)
CAR_BEFORE = (700, 300, 1000, 540)       # bottom 540: the band below is road above the crosswalk, not on it


class ImageRule(unittest.TestCase):
    def test_vehicle_on_crosswalk_with_pedestrian_on_it(self):
        events = run([track(1, "person", PED_ON, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))])
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].vehicle_id, events[0].pedestrian_id, events[0].pedestrian_state), (10, 1, "on_crosswalk"))
        self.assertIn("輔助", events[0].disclaimer)

    def test_vehicle_stopped_before_the_crosswalk_is_not_flagged(self):
        self.assertEqual(run([track(1, "person", PED_ON, range(1, 31)), track(10, "car", CAR_BEFORE, range(1, 31))]), [])

    def test_pedestrian_waiting_above_the_edge_does_not_count_by_default(self):
        waiting = (1230, 480, 1290, 585)  # foot at mask row 195: 5 px above the crosswalk top edge (row 200)
        tracks = [track(1, "person", waiting, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))]
        self.assertEqual(run(tracks), [])
        events = run(tracks, params=Params(fps=60.0, src_size=SRC, min_seconds=0.1, require_crossing=False))
        self.assertEqual([(e.pedestrian_state, e.pedestrian_entered_crosswalk) for e in events], [("waiting_at_edge", False)])

    def test_pedestrian_standing_right_at_the_inside_edge_is_waiting_not_crossing(self):
        shallow = (1230, 303, 1290, 603)   # foot mask row 201: 2 px inside the top edge; person 100 px tall -> depth 0.02
        tracks = [track(1, "person", shallow, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))]
        self.assertEqual(run(tracks), [])

    def test_pedestrian_stepping_onto_the_crosswalk_as_the_vehicle_passes_counts(self):
        steps = {"id": 1, "kind": "person", "boxes":
                 [[f, 1230, 303, 1290, 603] for f in range(1, 16)] + [[f, 1230, 450, 1290, 750] for f in range(16, 31)]}  # foot row 201 -> 250
        events = run([steps, track(10, "car", CAR_ON, range(1, 31))])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].pedestrian_state, "waiting_at_edge")   # state at first contact
        self.assertTrue(events[0].pedestrian_entered_crosswalk)

    def test_pedestrian_far_from_the_crosswalk_is_ignored(self):
        far = (900, 20, 940, 110)        # foot far above the crosswalk, farther than 0.8 person heights
        self.assertEqual(run([track(1, "person", far, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))]), [])

    def test_rider_on_a_motorcycle_is_not_a_pedestrian(self):
        bike = (880, 600, 1040, 870)
        rider = (900, 620, 1000, 800)    # almost entirely inside the motorcycle box
        self.assertEqual(run([track(1, "person", rider, range(1, 31)), track(10, "motorcycle", bike, range(1, 31))]), [])

    def test_pedestrian_in_front_of_a_bus_is_not_a_rider(self):
        bus = (700, 400, 1100, 870)
        walker = (880, 650, 960, 850)    # entirely inside the bus box in the image, but buses have no riders
        events = run([track(1, "person", walker, range(1, 31)), track(10, "bus", bus, range(1, 31))])
        self.assertEqual(len(events), 1)

    def test_cli_refuses_to_guess_fps(self):
        from yield_image_rule import main
        self.assertEqual(main(["--masks", "m.npz", "--tracks", "t.json", "--output", "o.jsonl"]), 2)

    def test_pedestrian_on_the_far_crosswalk_does_not_pair_with_a_near_vehicle(self):
        far_ped = (1000, 190, 1040, 245)  # foot (1020, 245) -> mask (340, 81)... on the far blob
        mask = mask_with(NEAR, FAR)
        self.assertEqual(run([track(1, "person", far_ped, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))], mask=mask), [])

    def test_vehicle_too_far_sideways_from_the_pedestrian(self):
        far_left = (60, 700, 120, 850)
        car_right = (1500, 500, 1700, 870)  # width 200 -> limit 300 px, gap ~1400
        self.assertEqual(run([track(1, "person", far_left, range(1, 31)), track(10, "car", car_right, range(1, 31))]), [])

    def test_brief_contact_shorter_than_min_seconds_is_ignored(self):
        strict = Params(fps=60.0, src_size=SRC, min_seconds=0.5)   # needs 30 frames
        self.assertEqual(run([track(1, "person", PED_ON, range(1, 31)), track(10, "car", CAR_ON, range(1, 21))], params=strict), [])

    def test_event_time_is_the_first_frame_of_contact(self):
        strict = Params(fps=60.0, src_size=SRC, min_seconds=0.2)   # needs 12 frames before the event is confirmed
        events = run([track(1, "person", PED_ON, range(1, 31)), track(10, "car", CAR_ON, range(5, 31))], params=strict)
        self.assertEqual(events[0].frame, 5)   # the car's first frame on the crosswalk, not frame 16

    def test_one_event_per_vehicle_track(self):
        tracks = [track(1, "person", PED_ON, range(1, 31)), track(2, "person", (1350, 700, 1410, 850), range(1, 31)),
                  track(10, "car", CAR_ON, range(1, 31))]
        self.assertEqual(len(run(tracks)), 1)

    def test_short_vehicle_track_is_marked_for_review(self):
        events = run([track(1, "person", PED_ON, range(1, 31)), track(10, "car", CAR_ON, range(1, 13))])  # 12 < 15 frames
        self.assertEqual(events[0].review_reasons, ["short_vehicle_track"])
        self.assertTrue(events[0].needs_review)
        full = run([track(1, "person", PED_ON, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))])
        self.assertFalse(full[0].needs_review)


def src(box):
    return tuple(v * 3 for v in box)   # mask px -> source px (1920x1080 vs 640x360)


def tilted_mask(center, long_side, short_side, degrees):
    """Filled rectangle whose long axis points along `degrees` (0 = horizontal, 90 = vertical)."""
    a = np.radians(degrees)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.array(center, float)
    corners = [c + sx * u * long_side / 2 + sy * v * short_side / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(corners).astype(np.int32)], 1)
    return m.astype(bool)


class TurningCamera(unittest.TestCase):
    """The crosswalk is no longer horizontal in the image: gaps must be measured along its own axis."""

    def check(self, mask, ped, vehicle):
        tracks = [track(1, "person", src(ped), range(1, 31)), track(10, "car", src(vehicle), range(1, 31))]
        return run(tracks, mask=mask)

    def test_vertical_crosswalk_near_pair_is_flagged_and_far_pair_is_not(self):
        mask = tilted_mask((320, 180), 320, 80, 90)          # 80 wide, 320 tall: pedestrians walk vertically
        ped = (310, 100, 330, 150)                            # foot (320, 150), 40 px inside the long edges
        near = (290, 170, 350, 200)                           # 35 px below the pedestrian along the axis
        far = (290, 290, 350, 320)                            # 170 px away along the axis, same x: old horizontal rule says "close"
        self.assertEqual(len(self.check(mask, ped, near)), 1)
        self.assertEqual(self.check(mask, ped, far), [])

    def test_diagonal_crosswalk_pair_along_the_axis_is_flagged(self):
        mask = tilted_mask((320, 180), 400, 80, 45)
        ped = (310, 130, 330, 180)                            # foot (320, 180) on the axis centre
        vehicle = (342, 207, 382, 237)                        # about 60 px along the axis from the pedestrian
        self.assertEqual(len(self.check(mask, ped, vehicle)), 1)
        far = (420, 285, 460, 315)                            # about 170 px along the axis
        self.assertEqual(self.check(mask, ped, far), [])

    def test_axis_follows_the_blob_orientation(self):
        from yield_image_rule import crosswalk_axis
        ux, uy = crosswalk_axis(tilted_mask((320, 180), 400, 80, 0))
        self.assertAlmostEqual(abs(ux), 1.0, places=1)
        ux, uy = crosswalk_axis(tilted_mask((320, 180), 320, 80, 90))
        self.assertAlmostEqual(abs(uy), 1.0, places=1)
        ux, uy = crosswalk_axis(tilted_mask((320, 180), 400, 80, 45))
        self.assertAlmostEqual(abs(ux), abs(uy), places=1)
        self.assertEqual(crosswalk_axis(tilted_mask((320, 180), 90, 90, 0)), (1.0, 0.0))  # not elongated: horizontal fallback


if __name__ == "__main__":
    unittest.main()
