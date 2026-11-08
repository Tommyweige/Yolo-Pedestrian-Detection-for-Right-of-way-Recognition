"""yield_image_rule regressions with synthetic masks and boxes (no GPU)."""
import unittest
from dataclasses import replace

import cv2
import numpy as np

from yield_image_rule import Params, detect

H, W = 360, 640
SRC = (1920, 1080)  # boxes below are in source pixels; masks are 640x360 (scale 1/3)
P = Params(fps=60.0, src_size=SRC, min_seconds=0.1, report_every_contact=False)   # the low-noise construction, explicit
DEFAULT = Params(fps=60.0, src_size=SRC, min_seconds=0.1)                         # exactly what the shipped defaults give


def mask_with(*rects):
    m = np.zeros((H, W), bool)
    for y1, y2, x1, x2 in rects:
        m[y1:y2, x1:x2] = True
    return m


NEAR = (200, 300, 20, 620)    # near crosswalk (mask pixels)
FAR = (60, 80, 150, 450)      # small far crosswalk


def track(identity, kind, box, frames):
    return {"id": identity, "kind": kind, "boxes": [[f, *box] for f in frames]}


def run(tracks, mask=None, frames=range(1, 31), params=None):
    mask = mask if mask is not None else mask_with(NEAR)
    return detect(lambda f: mask, tracks, params or P, frames)


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

    # The depth threshold (min_inside_person_heights) is 0.03, the edge of a narrow passing shelf: 0.05 and up lose an
    # incident and 0.06 is back to the old 0.1 numbers, and the candidate-list construction keeps its last label and
    # incident only at 0.03. These three pin the shelf down, one clear step either side and the boundary itself, so it
    # cannot be moved without a test saying what that costs. The foot sits 3 mask rows inside the crosswalk's top edge on
    # every frame (source boxes are 3x the mask, so a source height of 288-309 px is 96-103 mask px tall - the range where
    # 3 rows straddles 0.03).
    def test_foot_just_past_the_depth_threshold_counts_as_crossing(self):
        # 96 mask px tall -> 3/96 = 0.031, just over 0.03
        just_inside = (1230, 318, 1290, 606)
        events = run([track(1, "person", just_inside, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))])
        self.assertEqual([(e.vehicle_id, e.pedestrian_id, e.pedestrian_state) for e in events],
                         [(10, 1, "on_crosswalk")])

    def test_foot_just_short_of_the_depth_threshold_is_waiting(self):
        # 103 mask px tall -> 3/103 = 0.029, just under 0.03
        just_short = (1230, 297, 1290, 606)
        tracks = [track(1, "person", just_short, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))]
        self.assertEqual(run(tracks), [])

    def test_foot_exactly_at_the_depth_threshold_counts_as_crossing(self):
        # 100 mask px tall -> 3/100 = 0.03 exactly, and the comparison is inclusive
        on_the_boundary = (1230, 306, 1290, 606)
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

    def test_review_mode_reports_the_crossing_pedestrian_rather_than_an_earlier_waiting_one(self):
        # one event per vehicle with the crossing filter off: the vehicle's one event must still be the pair a strict run
        # would report (the candidate list reports both pedestrians anyway)
        waiting = (1230, 480, 1290, 585)
        tracks = [track(1, "person", waiting, range(1, 61)), track(2, "person", PED_ON, range(31, 61)),
                  track(10, "car", CAR_ON, range(1, 61))]
        events = run(tracks, frames=range(1, 61), params=replace(P, require_crossing=False, report_every_contact=False))
        self.assertEqual([(e.vehicle_id, e.pedestrian_id, e.pedestrian_entered_crosswalk) for e in events], [(10, 2, True)])

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

    # The shipped construction is the candidate list with `pair` at depth 0.03 (user decision, 2026-10-31), so a vehicle
    # that obstructs two pedestrians in one pass reports both. Every other test here passes its own Params, which is why
    # this one test is the only thing that fails if the default is flipped back: it pins the default, which is a product
    # decision and not a free parameter, and the numbers behind it (4.4 -> 7.1 reports per clip) are in
    # docs/yield-image-rule.md.
    def test_default_params_report_every_contact_of_a_pass(self):
        self.assertTrue(DEFAULT.report_every_contact)
        self.assertEqual(DEFAULT.contact_dedup, "pair")
        self.assertEqual(DEFAULT.min_inside_person_heights, 0.03)
        first, second = (1230, 700, 1290, 850), (1350, 700, 1410, 850)
        tracks = [track(1, "person", first, range(1, 61)), track(2, "person", second, range(70, 121)),
                  track(10, "car", CAR_ON, range(1, 121))]
        self.assertEqual([(e.vehicle_id, e.pedestrian_id) for e in run(tracks, frames=range(1, 121), params=DEFAULT)],
                         [(10, 1), (10, 2)])                             # one pass, two pedestrians, two reports

    # The candidate-list construction is the shipped default; these tests spell it out so that a later change to the
    # default cannot silently rewrite what they check. They pin what its dedup (`pair`) is meant to do: repeats of one
    # pedestrian inside one vehicle pass are one report, two pedestrians of that pass are two reports.
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

    def test_candidate_list_merges_repeats_of_one_pedestrian_in_one_pass(self):
        # The vehicle leaves the crosswalk for 9 frames, so the same pedestrian is picked up again 1.15 s later: one
        # pedestrian, one pass, one report, and two contacts in the undeduped list.
        car = {"id": 10, "kind": "car",
               "boxes": [[f, *CAR_ON] for f in range(1, 61)] + [[f, *CAR_BEFORE] for f in range(61, 70)]
               + [[f, *CAR_ON] for f in range(70, 121)]}
        tracks = [track(1, "person", PED_ON, range(1, 121)), car]
        frames = range(1, 121)
        self.assertEqual([(e.vehicle_id, e.pedestrian_id) for e in run(tracks, frames=frames, params=self.CANDIDATE)],
                         [(10, 1)])
        raw = Params(fps=60.0, src_size=SRC, min_seconds=0.1, report_every_contact=True, contact_dedup_s=0)
        self.assertEqual(len(run(tracks, frames=frames, params=raw)), 2)     # the undeduped candidate list

    def test_candidate_list_reports_two_pedestrians_of_one_pass(self):
        # 1.15 s apart, so one pass: the default dedup is by (vehicle, pedestrian) pair, so both are reported. The earlier
        # pedestrian's contact is not a reason to withhold the crossing one (docs/yield-image-rule.md).
        first, second = (1230, 700, 1290, 850), (1350, 700, 1410, 850)
        tracks = [track(1, "person", first, range(1, 61)), track(2, "person", second, range(70, 121)),
                  track(10, "car", CAR_ON, range(1, 121))]
        frames = range(1, 121)
        self.assertEqual([(e.vehicle_id, e.pedestrian_id) for e in run(tracks, frames=frames, params=self.CANDIDATE)],
                         [(10, 1), (10, 2)])

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


class ImageRuleInCrosswalkSpace(ImageRule):
    """The same synthetic cases, judged in crosswalk coordinates instead of image pixels."""

    def setUp(self):
        global P
        self._saved, P = P, replace(P, geometry="crosswalk")

    def tearDown(self):
        global P
        P = self._saved

    # These pin the image geometry's depth threshold (min_inside_person_heights) to the pixel; crosswalk coordinates use
    # cw_on_margin instead, so the shelf they guard does not exist here.
    _image_threshold_only = unittest.skip("pins the image geometry's depth threshold, which crosswalk coordinates do not use")
    test_foot_just_past_the_depth_threshold_counts_as_crossing = _image_threshold_only(ImageRule.test_foot_just_past_the_depth_threshold_counts_as_crossing)
    test_foot_just_short_of_the_depth_threshold_is_waiting = _image_threshold_only(ImageRule.test_foot_just_short_of_the_depth_threshold_is_waiting)
    test_foot_exactly_at_the_depth_threshold_counts_as_crossing = _image_threshold_only(ImageRule.test_foot_exactly_at_the_depth_threshold_counts_as_crossing)


# perspective crosswalk (mask px): far edge row 200 from x=150..450, near edge row 300 from x=20..600
TRAPEZOID = [(150, 200), (450, 200), (600, 300), (20, 300)]


def skewed_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array([(100, 240), (560, 150), (620, 300), (40, 330)], np.int32)], 1)
    return m.astype(bool)


def trapezoid_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(TRAPEZOID, np.int32)], 1)
    return m.astype(bool)


class CrosswalkGeometry(unittest.TestCase):
    P = Params(fps=60.0, src_size=SRC, min_seconds=0.1, geometry="crosswalk")

    def test_a_car_beyond_a_sloping_far_edge_is_not_on_the_crosswalk_even_though_the_band_reaches_around_it(self):
        # the far edge climbs to the right; the car's bottom (row 190) is ~15 px beyond it at x=280, but the old
        # image rule sees crosswalk beside its wheel line and calls it "on"
        car = src((200, 110, 360, 190))
        ped = src((300, 240, 312, 285))
        tracks = [track(1, "person", ped, range(1, 31)), track(10, "car", car, range(1, 31))]
        self.assertEqual(run(tracks, mask=skewed_mask(), params=self.P), [])
        self.assertEqual(len(run(tracks, mask=skewed_mask(), params=replace(self.P, geometry="image"))), 1)

    def test_a_car_whose_wheel_line_is_inside_the_band_is_flagged(self):
        car = src((250, 150, 420, 245))
        ped = src((300, 235, 312, 268))
        events = run([track(1, "person", ped, range(1, 31)), track(10, "car", car, range(1, 31))],
                     mask=trapezoid_mask(), params=self.P)
        self.assertEqual([(e.vehicle_id, e.pedestrian_state) for e in events], [(10, "on_crosswalk")])

    def test_a_pedestrian_on_a_thin_far_strip_counts_as_on_it(self):
        strip = np.zeros((H, W), np.uint8)
        cv2.fillPoly(strip, [np.array([(150, 150), (450, 150), (456, 158), (144, 158)], np.int32)], 1)   # 8 px thick
        bus = src((260, 70, 420, 155))      # wheel line inside the strip
        walker = src((190, 78, 202, 154))   # 76 px tall: ten times the strip thickness; foot 4 px inside the strip
        tracks = [track(1, "person", walker, range(1, 31)), track(10, "bus", bus, range(1, 31))]
        events = run(tracks, mask=strip.astype(bool), params=self.P)
        self.assertEqual([e.pedestrian_state for e in events], ["on_crosswalk"])
        # (the image geometry missed this at the old 0.1 depth threshold; at 0.03 its 4/76 = 0.053 counts too)

    def test_a_small_piece_cut_by_the_image_edge_does_not_claim_people_and_cars_far_along_its_extension(self):
        # coordinates taken from the real 紅燈 clip (frame 52): a right-edge piece, a pedestrian and small cars ~200 px away
        piece = np.zeros((H, W), np.uint8)
        cv2.fillPoly(piece, [np.array([(562, 220), (596, 212), (639, 212), (639, 241)], np.int32)], 1)
        ped = src((240, 133, 252, 171))       # foot (246, 171): W 0.81 on the extended band, L 1.8
        car = src((280, 163, 290, 173))       # ground point (285, 173), the same L as the pedestrian on the extension
        tracks = [track(1, "person", ped, range(1, 31)), track(10, "car", car, range(1, 31))]
        self.assertEqual(run(tracks, mask=piece.astype(bool), params=self.P), [])


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


class GeometryInterface(unittest.TestCase):
    """component_geometry: one interface, two adapters (image pixels, fitted crosswalk band), same answers on a plain band."""

    def setUp(self):
        import yield_image_rule as rule
        self.rule = rule
        self.ped, self.car_on, self.car_before = (rule.scaled(b, 1 / 3, 1 / 3) for b in (PED_ON, CAR_ON, CAR_BEFORE))

    def test_both_adapters_judge_a_plain_crosswalk_the_same(self):
        for geometry in ("image", "crosswalk"):
            params = replace(DEFAULT, geometry=geometry)
            [component] = self.rule.frame_components(mask_with(NEAR), params)
            judge = self.rule.component_geometry(component, params)
            with self.subTest(geometry=geometry):
                self.assertEqual(judge.person_state(self.ped), "on_crosswalk")
                self.assertTrue(judge.vehicle_on(self.car_on))
                self.assertFalse(judge.vehicle_on(self.car_before))
                self.assertTrue(judge.close(self.ped, self.car_on))

    def test_a_component_no_band_fits_falls_back_to_the_image_adapter(self):
        tiny = mask_with((250, 253, 300, 303))
        params = replace(DEFAULT, geometry="crosswalk")
        self.assertIsInstance(self.rule.component_geometry(tiny, params), self.rule._ImageGeometry)
        self.assertIsInstance(self.rule.component_geometry(mask_with(NEAR), params), self.rule._CrosswalkGeometry)

    def test_detect_still_calls_scan_frame_by_module_name_so_experiments_can_replace_it(self):
        calls = []
        original = self.rule.scan_frame
        self.addCleanup(setattr, self.rule, "scan_frame", original)
        self.rule.scan_frame = lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs)
        run([track(1, "person", PED_ON, range(1, 31)), track(10, "car", CAR_ON, range(1, 31))])
        self.assertEqual(len(calls), 30)


if __name__ == "__main__":
    unittest.main()
