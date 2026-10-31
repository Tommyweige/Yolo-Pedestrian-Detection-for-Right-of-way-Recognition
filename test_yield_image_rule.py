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

    # The depth threshold (min_inside_person_heights) is 0.04, the middle of a narrow passing shelf: 0.05 and up lose an
    # incident and 0.06 is back to the old 0.1 numbers, while 0.03-0.04 both keep it. These three pin the shelf down, one
    # clear step either side and the boundary itself, so it cannot be moved to a "rounder" number without a test saying
    # what that costs. The foot sits 3 mask rows inside the crosswalk's top edge on every frame (source boxes are 3x the
    # mask, so a source height of 222-228 px is 74-76 mask px tall - the range where 3 rows straddles 0.04).
    def test_foot_just_past_the_depth_threshold_counts_as_crossing(self):
        # 74 mask px tall -> 3/74 = 0.0405 person heights, just over 0.04
        just_inside = (1230, 384, 1290, 606)
        events = run([track(1, "person", just_inside, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))])
        self.assertEqual([(e.vehicle_id, e.pedestrian_id, e.pedestrian_state) for e in events],
                         [(10, 1, "on_crosswalk")])

    def test_foot_just_short_of_the_depth_threshold_is_waiting(self):
        # 76 mask px tall -> 3/76 = 0.0395 person heights, just under 0.04
        just_short = (1230, 378, 1290, 606)
        tracks = [track(1, "person", just_short, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))]
        self.assertEqual(run(tracks), [])

    def test_foot_exactly_at_the_depth_threshold_counts_as_crossing(self):
        # 75 mask px tall -> 3/75 = 0.04 exactly, and the comparison is inclusive
        on_the_boundary = (1230, 381, 1290, 606)
        events = run([track(1, "person", on_the_boundary, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))])
        self.assertEqual([e.pedestrian_state for e in events], ["on_crosswalk"])

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

    def test_a_rider_whose_box_jitters_out_of_the_motorcycle_box_is_still_a_rider(self):
        # the tracker's box flickers a few pixels outside the motorcycle on some frames; per-frame containment lets the
        # rider through on those frames and reports the motorcycle it is riding as a failure to yield.
        bike = (880, 600, 1040, 870)
        jitter = [[f, *(900, 594, 1000, 800)] if f % 3 == 0 else [f, 900, 620, 1000, 800] for f in range(1, 31)]
        self.assertEqual(run([{"id": 1, "kind": "person", "boxes": jitter},
                              track(10, "motorcycle", bike, range(1, 31))]), [])

    def test_a_person_who_walked_past_a_motorcycle_is_a_pedestrian_in_front_of_another_vehicle(self):
        # v6 10.6 s: the same person passes a scooter here and rides one three seconds later. Pairing the rider
        # decision to the person alone hides the walking case, which is a real failure to yield.
        parked = (700, 600, 860, 870)         # the motorcycle they walk past
        ridden = (860, 560, 1060, 880)        # the one they ride at the end of the clip
        person = {"id": 1, "kind": "person", "boxes":
                  [[f, 880, 650, 960, 850] for f in range(1, 16)]        # walking on the crosswalk
                  + [[f, 900, 620, 1000, 800] for f in range(16, 31)]}   # sitting on `ridden`
        events = run([person, track(10, "motorcycle", parked, range(1, 31)),
                      track(11, "motorcycle", ridden, range(16, 31))])
        self.assertEqual([(e.vehicle_id, e.pedestrian_id) for e in events], [(10, 1)])

    def test_pedestrian_in_front_of_a_bus_is_not_a_rider(self):
        bus = (700, 400, 1100, 870)
        walker = (880, 650, 960, 850)    # entirely inside the bus box in the image, but buses have no riders
        events = run([track(1, "person", walker, range(1, 31)), track(10, "bus", bus, range(1, 31))])
        self.assertEqual(len(events), 1)

    def test_a_waiting_pedestrian_does_not_use_up_the_vehicle_for_a_later_crossing_pedestrian(self):
        # a bus stopped on the crosswalk first pairs with someone waiting at the edge (filtered out); a different
        # pedestrian then crosses in front of the same bus. That second pair must still be reported.
        waiting = (1230, 480, 1290, 585)
        tracks = [track(1, "person", waiting, range(1, 61)), track(2, "person", PED_ON, range(31, 61)),
                  track(10, "car", CAR_ON, range(1, 61))]
        events = run(tracks, frames=range(1, 61))
        self.assertEqual([(e.vehicle_id, e.pedestrian_id) for e in events], [(10, 2)])

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

    def test_event_time_is_the_contact_start_when_the_pedestrian_is_already_on_the_crosswalk(self):
        strict = Params(fps=60.0, src_size=SRC, min_seconds=0.2)   # needs 12 frames before the event is confirmed
        events = run([track(1, "person", PED_ON, range(1, 31)), track(10, "car", CAR_ON, range(5, 31))], params=strict)
        self.assertEqual(events[0].frame, 5)   # the car's first frame on the crosswalk, not frame 16
        self.assertEqual(events[0].contact_frames, (5, 30))

    def test_pedestrian_who_steps_on_late_in_a_long_contact_counts_and_is_timed_then(self):
        # the car sits on the crosswalk for 4 s; the pedestrian only steps on 3 s after the contact began, which the
        # old "1 s around the contact start" window threw away although the pair is real.
        frames = range(1, 241)
        steps = {"id": 1, "kind": "person", "boxes":
                 [[f, 1230, 303, 1290, 603] for f in range(1, 181)] + [[f, 1230, 450, 1290, 750] for f in range(181, 241)]}
        events = run([steps, track(10, "car", CAR_ON, frames)], frames=frames)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].frame, 181)
        self.assertEqual(events[0].pedestrian_state, "waiting_at_edge")   # state when the contact began
        self.assertEqual(events[0].contact_frames, (1, 240))

    def test_one_event_per_vehicle_track(self):
        tracks = [track(1, "person", PED_ON, range(1, 31)), track(2, "person", (1350, 700, 1410, 850), range(1, 31)),
                  track(10, "car", CAR_ON, range(1, 31))]
        self.assertEqual(len(run(tracks)), 1)

    # The candidate-list construction (report_every_contact) is what the paper-side comparison scores, so these two pin
    # the difference it is meant to make: a vehicle's later contact with a different pedestrian survives when it is a
    # separate pass, and does not when both contacts belong to the same pass.
    CANDIDATE = Params(fps=60.0, src_size=SRC, min_seconds=0.1, report_every_contact=True)

    def test_candidate_list_reports_a_later_contact_of_the_same_vehicle(self):
        first, second = (1230, 700, 1290, 850), (1350, 700, 1410, 850)
        tracks = [track(1, "person", first, range(1, 61)), track(2, "person", second, range(120, 181)),
                  track(10, "car", CAR_ON, range(1, 181))]
        frames = range(1, 181)
        self.assertEqual([(e.vehicle_id, e.pedestrian_id) for e in run(tracks, frames=frames)],
                         [(10, 1)])                                      # one event per vehicle keeps the first
        self.assertEqual([(e.vehicle_id, e.pedestrian_id) for e in run(tracks, frames=frames, params=self.CANDIDATE)],
                         [(10, 1), (10, 2)])                             # 2 s apart: two passes, two reports

    def test_candidate_list_collapses_contacts_of_one_pass(self):
        first, second = (1230, 700, 1290, 850), (1350, 700, 1410, 850)
        tracks = [track(1, "person", first, range(1, 61)), track(2, "person", second, range(70, 121)),
                  track(10, "car", CAR_ON, range(1, 121))]
        frames = range(1, 121)
        self.assertEqual([e.pedestrian_id for e in run(tracks, frames=frames, params=self.CANDIDATE)],
                         [1])                                            # 1.15 s apart: one pass, one report
        raw = Params(fps=60.0, src_size=SRC, min_seconds=0.1, report_every_contact=True, contact_dedup_s=0)
        self.assertEqual([e.pedestrian_id for e in run(tracks, frames=frames, params=raw)],
                         [1, 2])                                         # the undeduped candidate list

    # Which contact of a pass survives is a policy, and one pass can touch two pedestrians: someone who has been waiting at
    # the edge and steps on as the car arrives, and someone already walking across. The three variants must disagree here,
    # or the rule cannot tell them apart. The waiting one's event time is the frame they step on (16), which is earlier
    # than the crossing one's contact (20), so "earliest" picks the waiting contact.
    WAITING_THEN_STEPPING = {"id": 1, "kind": "person", "boxes":
                             [[f, 1230, 480, 1290, 585] for f in range(1, 16)]     # foot above the edge: waiting
                             + [[f, *PED_ON] for f in range(16, 41)]}              # steps on at frame 16
    ALREADY_CROSSING = (1350, 700, 1410, 850)

    def pass_tracks(self, second):
        return [self.WAITING_THEN_STEPPING, second, track(10, "car", CAR_ON, range(1, 41))]

    def variant(self, name):
        return Params(fps=60.0, src_size=SRC, min_seconds=0.1, report_every_contact=True, contact_dedup=name)

    def test_dedup_earliest_keeps_the_first_contact_of_the_pass(self):
        second = track(2, "person", self.ALREADY_CROSSING, range(20, 41))
        events = run(self.pass_tracks(second), frames=range(1, 41), params=self.variant("earliest"))
        self.assertEqual([(e.pedestrian_id, e.pedestrian_state, e.time_s) for e in events],
                         [(1, "waiting_at_edge", round(16 / 60, 3))])

    def test_dedup_prefers_the_pedestrian_on_the_crosswalk_over_the_earlier_contact(self):
        second = track(2, "person", self.ALREADY_CROSSING, range(20, 41))
        events = run(self.pass_tracks(second), frames=range(1, 41), params=self.variant("on_crosswalk_longest"))
        self.assertEqual([(e.pedestrian_id, e.pedestrian_state) for e in events], [(2, "on_crosswalk")])

    def test_dedup_preferring_the_crosswalk_falls_back_to_the_first_contact(self):
        # both pedestrians start out waiting, so nothing is on the crosswalk at contact start: the pass keeps its first
        waiting_too = {"id": 2, "kind": "person", "boxes":
                       [[f, 1230, 480, 1290, 585] for f in range(20, 26)] + [[f, *self.ALREADY_CROSSING] for f in range(26, 41)]}
        events = run(self.pass_tracks(waiting_too), frames=range(1, 41), params=self.variant("on_crosswalk_longest"))
        self.assertEqual([e.pedestrian_id for e in events], [1])

    def test_dedup_by_pair_keeps_both_pedestrians_of_one_pass(self):
        second = track(2, "person", self.ALREADY_CROSSING, range(20, 41))
        events = run(self.pass_tracks(second), frames=range(1, 41), params=self.variant("pair"))
        self.assertEqual([(e.vehicle_id, e.pedestrian_id) for e in events], [(10, 1), (10, 2)])

    def test_pass_chains_so_a_slow_crossing_stays_one_report(self):
        # Three pedestrians of one vehicle 1.17 s apart: no contact is more than the window from the next, but the first
        # and the last are 2.3 s apart. The pass is a chain, so this is one crossing and one report; measuring the window
        # from the first contact instead would call it two passes and report the 3.25 s pedestrian again.
        slow = [track(1, "person", PED_ON, range(55, 63)), track(2, "person", PED_ON, range(125, 133)),
                track(3, "person", PED_ON, range(195, 203)), track(10, "car", CAR_ON, range(1, 221))]
        events = run(slow, frames=range(1, 221), params=self.variant("earliest"))
        self.assertEqual([(e.vehicle_id, e.pedestrian_id) for e in events], [(10, 1)])
        raw = Params(fps=60.0, src_size=SRC, min_seconds=0.1, report_every_contact=True, contact_dedup_s=0)
        self.assertEqual([e.pedestrian_id for e in run(slow, frames=range(1, 221), params=raw)], [1, 2, 3])

    def test_short_vehicle_track_is_marked_for_review(self):
        events = run([track(1, "person", PED_ON, range(1, 31)), track(10, "car", CAR_ON, range(1, 13))])  # 12 < 15 frames
        self.assertEqual(events[0].review_reasons, ["short_vehicle_track"])
        self.assertTrue(events[0].needs_review)
        full = run([track(1, "person", PED_ON, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))])
        self.assertFalse(full[0].needs_review)

    def test_one_vehicle_tracked_twice_reports_once(self):
        # ByteTrack keeps a second id for an object it already tracks; both boxes sit on the same car.
        events = run([track(1, "person", PED_ON, range(1, 31)),
                      track(10, "car", CAR_ON, range(1, 31)),
                      track(11, "car", (706, 502, 1006, 872), range(1, 31))])
        self.assertEqual([(e.vehicle_id, e.pedestrian_id) for e in events], [(10, 1)])

    def test_a_vehicle_that_gets_a_new_track_id_reports_once(self):
        # the tracker drops the id and picks it up again while the car waits: boxes lined up across the gap
        events = run([track(1, "person", PED_ON, range(1, 31)),
                      track(10, "car", CAR_ON, range(1, 18)),
                      track(11, "car", CAR_ON, range(21, 31))])
        self.assertEqual([(e.vehicle_id, e.pedestrian_id) for e in events], [(10, 1)])

    def test_two_different_vehicles_are_not_merged(self):
        from yield_image_rule import vehicle_groups
        left, right = track(10, "car", CAR_ON, range(1, 31)), track(11, "car", (1400, 500, 1700, 870), range(1, 31))
        self.assertEqual(vehicle_groups([left, right], P), {10: 10, 11: 11})


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
