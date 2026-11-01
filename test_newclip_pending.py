"""The pending list: which events of a run still need a verdict, and which batch each one is."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import newclip_pending  # noqa: E402
import newclip_review  # noqa: E402
import newclip_score  # noqa: E402


def event(time_s, vehicle=7, ped=3):
    return {"kind": "suspected_failure_to_yield", "time_s": time_s, "frame": int(time_s * 30),
            "vehicle_id": vehicle, "pedestrian_id": ped, "vehicle_kind": "car",
            "pedestrian_state": "on_crosswalk", "contact_frames": [int(time_s * 30), int(time_s * 30) + 5]}


def row(time_s, vehicle, verdict, stem="clip-a", ped=None):
    return {"stem": stem, "time_s": time_s, "vehicle": vehicle, "ped": ped, "verdict": verdict,
            "note": "", "page": None}


class SelectTest(unittest.TestCase):
    def select(self, events, rows, **kwargs):
        clip = {"stem": "clip-a", "events": events}
        stray = newclip_score.join_verdicts([clip], rows)
        run = {"dir": Path("C:/runs/dev"), "manifest": {"role": "dev", "clips": []}, "clips": [clip]}
        return newclip_pending.select(run, stray, **kwargs)

    def test_a_judged_event_is_not_on_the_page_and_an_unjudged_one_is(self):
        result = self.select([event(4.0), event(9.0)], [row(4.0, 7, "A")])
        self.assertEqual([item["time_s"] for item in result["events"]], [9.0])
        self.assertEqual(result["counts"]["pending"], 1)
        self.assertEqual(result["counts"]["judged"], 1)

    def test_a_stray_row_on_the_same_vehicle_marks_its_event_as_needing_a_re_judge(self):
        # The old verdict was written for the same motorcycle at 5.033 s; the rule now reports it at 3.333 s.
        result = self.select([event(3.333, vehicle=9, ped=18)], [row(5.033, 9, "C")])
        item = result["events"][0]
        self.assertEqual(item["origin"], "rejudge")
        self.assertEqual(item["priors"][0]["verdict"], "C")
        self.assertEqual(item["priors"][0]["time_s"], 5.033)
        self.assertAlmostEqual(item["priors"][0]["delta_s"], 1.7)

    def test_a_stray_row_without_a_matching_vehicle_only_reaches_a_nearby_event(self):
        near = self.select([event(3.333, vehicle=9)], [row(5.033, 99, "C")])
        self.assertEqual(near["events"][0]["origin"], "rejudge")
        far = self.select([event(3.333, vehicle=9), event(20.0, vehicle=9)], [row(5.033, 99, "C")])
        self.assertEqual(far["events"][0]["origin"], "rejudge")
        self.assertEqual(far["events"][1]["origin"], "new")

    def test_a_row_whose_incident_is_already_judged_is_not_a_card(self):
        # Two A rows 0.15 s apart describe one moment; the event there already took one of them.
        result = self.select([event(15.082, vehicle=2879), event(21.838, vehicle=3372)],
                             [row(15.032, 2879, "A"), row(15.182, 3080, "A")])
        self.assertEqual([item["origin"] for item in result["events"]], ["new"])
        self.assertEqual(result["counts"]["orphan_rows"], 1)
        self.assertEqual(result["counts"]["stray_rows"], 1)
        self.assertEqual(result["orphans"][0]["nearest_delta_s"], 0.1)
        self.assertEqual(result["orphans"][0]["kind"], "duplicate")

    def test_an_orphan_row_is_reported_with_the_closest_event_and_its_verdict(self):
        # The 17.6 s row takes the event there; the 20.0 s row is 2.4 s away, past the tolerance, so the rule no
        # longer reports anything at that moment.
        result = self.select([event(17.6, vehicle=2515), event(60.0)],
                             [row(17.6, 2515, "C"), row(20.0, 2515, "C")])
        self.assertEqual(result["counts"]["judged"], 1)
        orphan = result["orphans"][0]
        self.assertEqual(orphan["nearest_time_s"], 17.6)
        self.assertEqual(orphan["nearest_judged"], "C")
        self.assertAlmostEqual(orphan["nearest_delta_s"], 2.4)
        self.assertEqual(orphan["kind"], "gap")

    def test_a_row_that_lands_nowhere_is_counted_as_a_recall_gap_not_a_duplicate(self):
        result = self.select([event(10.0, vehicle=7), event(40.0, vehicle=8)],
                             [row(10.0, 7, "A"), row(25.0, 8, "A")])
        self.assertEqual([orphan["kind"] for orphan in result["orphans"]], ["gap"])
        self.assertEqual(result["counts"]["orphan_gaps"], 1)
        self.assertEqual(result["counts"]["orphan_duplicates"], 0)

    def test_a_pending_event_names_the_judged_event_next_to_it(self):
        result = self.select([event(9.4, vehicle=516, ped=298), event(8.8, vehicle=516, ped=297)],
                             [row(9.4, 516, "D", ped=298)])
        item = [item for item in result["events"] if item["time_s"] == 8.8][0]
        self.assertEqual(item["neighbour"]["verdict"], "D")
        self.assertAlmostEqual(item["neighbour"]["delta_s"], 0.6)

    def test_no_neighbour_is_named_when_the_closest_verdict_is_far_away(self):
        result = self.select([event(9.4), event(30.0)], [row(9.4, 7, "A")])
        self.assertIsNone(result["events"][0]["neighbour"])

    def test_the_counts_say_how_much_of_the_batch_the_page_covers(self):
        result = self.select([event(4.0), event(9.0), event(12.0)], [row(4.0, 7, "A")])
        counts = result["counts"]
        self.assertEqual((counts["events"], counts["judged"], counts["pending"], counts["covered"]), (3, 1, 2, 3))
        self.assertEqual((counts["new"], counts["rejudge"]), (2, 0))

    def test_the_note_states_what_this_page_is_in_numbers(self):
        counts = {"events": 50, "judged": 25, "pending": 25, "new": 23, "rejudge": 2, "covered": 50}
        note = newclip_review.selection_note(counts)
        for part in ("25 件是 50 件", "23 件全新", "2 件是舊判定對不上", "50 件裡的 50 件"):
            self.assertIn(part, note)


class StorageKeyTest(unittest.TestCase):
    def test_a_selection_page_does_not_share_drafted_verdicts_with_the_full_page(self):
        run = Path("C:/runs/dev")
        self.assertNotEqual(newclip_review.storage_key_of(run, "pending"), newclip_review.storage_key_of(run))


if __name__ == "__main__":
    unittest.main()
