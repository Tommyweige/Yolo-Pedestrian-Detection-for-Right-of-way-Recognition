import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_eiou as e  # noqa: E402

BOX = (0, 0, 10, 10)
HALF = (5, 0, 15, 10)          # half of BOX, same area: IoU 1/3
HALF_AREA = (0, 0, 10, 5)      # half the area, inside BOX: IoU 0.5
FAR = (100, 100, 110, 110)


def event(start, end, start_box=BOX, end_box=BOX):
    return {"start": start, "end": end, "start_box": start_box, "end_box": end_box}


class TemporalIoU(unittest.TestCase):
    def test_identical_intervals_are_one(self):
        self.assertEqual(e.tiou(10, 20, 10, 20), 1.0)

    def test_intervals_that_do_not_overlap_are_zero(self):
        self.assertEqual(e.tiou(10, 20, 21, 30), 0.0)
        self.assertEqual(e.tiou(21, 30, 10, 20), 0.0)

    def test_touching_intervals_are_zero_not_an_error(self):
        self.assertEqual(e.tiou(10, 20, 20, 30), 0.0)

    def test_half_overlap_is_intersection_over_union(self):
        # 15..20 = 5 shared, union 10 + 10 - 5 = 15
        self.assertAlmostEqual(e.tiou(10, 20, 15, 25), 5 / 15)

    def test_one_interval_inside_the_other_is_never_above_one(self):
        self.assertAlmostEqual(e.tiou(10, 20, 12, 18), 6 / 10)

    def test_zero_length_intervals_score_zero(self):
        self.assertEqual(e.tiou(10, 10, 10, 10), 0.0)


class SpatialIoU(unittest.TestCase):
    def test_identical_boxes_are_one(self):
        self.assertEqual(e.siou(BOX, BOX), 1.0)

    def test_boxes_that_do_not_overlap_are_zero(self):
        self.assertEqual(e.siou(BOX, FAR), 0.0)

    def test_half_overlap(self):
        self.assertAlmostEqual(e.siou(BOX, HALF), 1 / 3)

    def test_exactly_half_the_area(self):
        self.assertAlmostEqual(e.siou(BOX, HALF_AREA), 0.5)

    def test_degenerate_box_is_zero_not_a_division_by_zero(self):
        self.assertEqual(e.siou((5, 5, 5, 5), BOX), 0.0)


class EventIoU(unittest.TestCase):
    def test_same_boxes_same_time_is_one(self):
        self.assertEqual(e.eiou(event(0, 10), event(0, 10)), 1.0)

    def test_time_does_not_overlap_scores_zero_however_good_the_boxes(self):
        self.assertEqual(e.eiou(event(0, 10), event(20, 30)), 0.0)

    def test_space_does_not_overlap_scores_zero_however_good_the_time(self):
        self.assertEqual(e.eiou(event(0, 10, FAR, FAR), event(0, 10)), 0.0)

    def test_half_overlap_on_both_axes_multiplies(self):
        # tiou 1/3, both box IoUs exactly 0.5 -> 1/3 * 0.5
        self.assertAlmostEqual(e.eiou(event(0, 10, HALF_AREA, HALF_AREA), event(5, 15)), 1 / 6)

    def test_only_the_start_box_overlapping_halves_the_spatial_term(self):
        # tiou 1, siou_start 0.5, siou_end 0 -> 0.25
        self.assertAlmostEqual(e.eiou(event(0, 10, HALF_AREA, FAR), event(0, 10)), 0.25)

    def test_it_is_the_mean_of_the_two_boxes_not_their_product(self):
        self.assertAlmostEqual(e.eiou(event(0, 10, HALF_AREA, FAR), event(0, 10)), 0.25)


class Matching(unittest.TestCase):
    def test_threshold_is_strict_like_crosswalk(self):
        half_area = event(0, 10, HALF_AREA, HALF_AREA)      # eiou exactly 0.5
        self.assertAlmostEqual(e.eiou(half_area, event(0, 10)), 0.5)
        self.assertEqual(e.match([half_area], [event(0, 10)], 0.5), [])
        self.assertEqual(len(e.match([half_area], [event(0, 10)], 0.49)), 1)

    def test_one_detection_retrieves_one_label(self):
        label = event(0, 10)
        self.assertEqual(len(e.match([event(0, 10)], [label], 0.5)), 1)
        self.assertEqual(e.match([event(0, 10)], [label, event(0, 10)], 0.5)[0]["label"], 0)

    def test_two_detections_of_one_label_take_the_better_one(self):
        good = event(0, 10)
        weak = event(0, 10, HALF_AREA, HALF_AREA)             # 0.5, below the threshold
        pairs = e.match([weak, good], [event(0, 10)], 0.5)
        self.assertEqual([(pair["det"], pair["eiou"]) for pair in pairs], [(1, 1.0)])

    def test_no_label_means_no_match(self):
        self.assertEqual(e.match([event(0, 10)], [], 0.5), [])


class AveragePrecision(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        self.assertEqual(e.average_precision([0.9, 0.8, 0.1, 0.05], [True, True, False, False]), 1.0)

    def test_the_sklearn_example(self):
        # sklearn.metrics.average_precision_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) = 0.83
        self.assertAlmostEqual(e.average_precision([0.1, 0.4, 0.35, 0.8], [False, False, True, True]), 5 / 6)

    def test_an_uninformative_constant_score_gives_the_precision_of_the_set(self):
        self.assertAlmostEqual(e.average_precision([1.0, 1.0, 1.0], [True, False, False]), 1 / 3)

    def test_worse_ranking_never_scores_higher(self):
        # the only difference is which of the two true positives sits above the false one
        better = e.average_precision([0.9, 0.8, 0.2], [True, False, True])
        worse = e.average_precision([0.9, 0.8, 0.2], [False, True, True])
        self.assertGreater(better, worse)

    def test_a_missing_score_is_refused_instead_of_defaulted(self):
        with self.assertRaises(ValueError) as caught:
            e.average_precision([1.0, None], [True, False])
        self.assertIn("no confidence score", str(caught.exception))

    def test_an_empty_detection_set_has_no_ap(self):
        with self.assertRaises(ValueError):
            e.average_precision([], [])

    def test_ap_needs_at_least_one_true_positive(self):
        with self.assertRaises(ValueError):
            e.average_precision([0.5, 0.4], [False, False])

    def test_lengths_must_agree(self):
        with self.assertRaises(ValueError):
            e.average_precision([0.5], [True, False])


class Passes(unittest.TestCase):
    def test_gap_zero_keeps_each_run(self):
        spans = e.residency_spans([1, 2, 4], 0)
        self.assertEqual(spans, {1: (1, 2), 2: (1, 2), 4: (4, 4)})

    def test_a_short_dropout_is_one_pass(self):
        self.assertEqual(e.residency_spans([1, 2, 4], 2), {1: (1, 4), 2: (1, 4), 4: (1, 4)})

    def test_a_long_gap_is_two_passes(self):
        spans = e.residency_spans([1, 2, 20], 5)
        self.assertEqual((spans[2], spans[20]), ((1, 2), (20, 20)))


def clip(**overrides):
    """A clip as `load_clip` returns it, with the mask-derived parts written out."""
    base = {
        "stem": "clip", "fps": 10.0,
        "boxes": {1: {0: BOX, 10: BOX, 20: BOX, 30: BOX}, 2: {0: FAR, 10: FAR, 20: FAR, 30: FAR}},
        "spans": {1: e.residency_spans([0, 10, 20, 30], 10), 2: e.residency_spans([0, 10, 20, 30], 10)},
        "on_frame": {frame: [1] for frame in (0, 10, 20, 30)},
        "events": [{"kind": "vehicle_on_crosswalk_with_pedestrian", "time_s": 1.0, "vehicle_id": 1,
                    "contact_frames": [10, 20]},
                   {"kind": "vehicle_on_crosswalk_with_pedestrian", "time_s": 1.0, "vehicle_id": 2,
                    "contact_frames": [10, 20]}],
        "labels": [{"kind": "suspected_failure_to_yield", "time_s": 1.0, "source": "manual"}],
    }
    return {**base, **overrides}


class LabelResolution(unittest.TestCase):
    def test_a_label_with_one_vehicle_on_its_frame_uses_that_vehicle(self):
        vehicle_id, span, basis = e.label_vehicle(clip(), 10, 5)
        self.assertEqual((vehicle_id, span, basis), (1, (0, 30), "exact"))

    def test_two_vehicles_on_the_frame_stay_unresolved(self):
        on_frame = {frame: [1, 2] for frame in (0, 10, 20, 30)}
        self.assertEqual(e.label_vehicle(clip(on_frame=on_frame), 10, 5),
                         (None, None, "several_vehicles_on_crosswalk"))

    def test_no_vehicle_anywhere_near_is_unresolved(self):
        self.assertEqual(e.label_vehicle(clip(on_frame={}), 10, 5), (None, None, "no_vehicle_on_crosswalk"))

    def test_a_frame_in_a_dropout_falls_back_to_the_window(self):
        # the label's own frame 10 has nothing; frame 13 does, and it is the only candidate within +-5
        dropped = clip(on_frame={13: [1]}, spans={1: {13: (13, 13)}}, boxes={1: {13: BOX}})
        self.assertEqual(e.label_vehicle(dropped, 10, 5), (1, (13, 13), "window"))

    def test_the_window_is_not_used_when_the_frame_itself_is_ambiguous(self):
        on_frame = {10: [1, 2]}
        self.assertEqual(e.label_vehicle(clip(on_frame=on_frame), 10, 5)[2], "several_vehicles_on_crosswalk")

    def test_window_zero_keeps_a_dropout_unresolved(self):
        dropped = clip(on_frame={13: [1]}, spans={1: {13: (13, 13)}}, boxes={1: {13: BOX}})
        self.assertEqual(e.label_vehicle(dropped, 10, 0)[2], "no_vehicle_on_crosswalk")

    def test_a_yielded_label_is_not_a_failure_ground_truth(self):
        labels = [{"kind": "yielded", "time_s": 1.0}]
        rows = e.ground_truths(clip(labels=labels), 5)
        self.assertEqual([(row["gt"], row["reason"]) for row in rows], [(None, "not_a_failure_label")])


class Scoring(unittest.TestCase):
    def test_both_instruments_score_the_same_clip(self):
        score = e.score_clip(clip())
        self.assertEqual((score["events"], score["labels"], score["labels_with_gt_box"]), (2, 1, 1))
        # the same vehicle, so the whole pass matches; the other vehicle's event is a false alarm
        self.assertEqual(score["eiou_whole_pass"]["matched"], 1)
        self.assertEqual(score["tolerance"]["matched_all_labels"], 1)
        self.assertEqual(score["tolerance"]["matched_resolved_labels"], 1)

    def test_a_tolerance_hit_on_another_vehicle_is_not_an_eiou_match(self):
        events = [{"kind": "vehicle_on_crosswalk_with_pedestrian", "time_s": 1.1, "vehicle_id": 2,
                   "contact_frames": [10, 20]}]
        score = e.score_clip(clip(events=events))
        self.assertEqual(score["tolerance"]["matched_all_labels"], 1)     # 0.1 s apart, inside the tolerance
        self.assertEqual(score["eiou_whole_pass"]["matched"], 0)          # but it is the other vehicle

    def test_the_rule_contact_window_is_a_different_event_than_the_pass(self):
        # contact 10..20 inside a pass of 0..30: tiou 1/3, so the contact-span row misses what the pass row matches
        score = e.score_clip(clip())
        self.assertEqual(score["eiou"]["matched"], 0)
        self.assertEqual(score["eiou_whole_pass"]["matched"], 1)

    def test_totals_report_both_denominators(self):
        scores = [e.score_clip(clip())]
        totals = e.summarise(scores)
        self.assertEqual((totals["events"], totals["labels"], totals["labels_with_gt_box"]), (2, 1, 1))
        self.assertEqual(totals["tolerance"]["recall_all_labels"], 1.0)
        self.assertEqual(totals["eiou_whole_pass"]["precision"], 0.5)
        self.assertIsNone(totals["ap"])
        self.assertIn("no confidence score", totals["ap_reason"])

    def test_a_scored_events_file_gets_an_ap(self):
        events = [{"kind": "vehicle_on_crosswalk_with_pedestrian", "time_s": 1.0, "vehicle_id": 1,
                   "contact_frames": [10, 20], "score": 0.9},
                  {"kind": "vehicle_on_crosswalk_with_pedestrian", "time_s": 1.0, "vehicle_id": 2,
                   "contact_frames": [10, 20], "score": 0.1}]
        totals = e.summarise([e.score_clip(clip(events=events))])
        self.assertIsNone(totals["ap_reason"])
        self.assertGreater(totals["ap"], 0.5)


if __name__ == "__main__":
    unittest.main()
