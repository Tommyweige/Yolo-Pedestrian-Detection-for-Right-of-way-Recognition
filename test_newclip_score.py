"""Run with a Python that has opencv-python, numpy and Pillow (the scorer imports the rule and eval_events)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import newclip_review  # noqa: E402
import newclip_score  # noqa: E402

STEMS = ["clip-a", "clip-b"]


def event(time_s, vehicle=7, ped=3, kind="suspected_failure_to_yield"):
    return {"kind": kind, "time_s": time_s, "frame": int(time_s * 30), "vehicle_id": vehicle,
            "pedestrian_id": ped, "vehicle_kind": "car", "pedestrian_state": "on_crosswalk",
            "contact_frames": [int(time_s * 30), int(time_s * 30) + 5]}


class VerdictFileTest(unittest.TestCase):
    def test_the_page_export_keeps_the_video_time_vehicle_and_pedestrian(self):
        rows = newclip_score.verdict_rows(
            {"schema": "yolo-ped/newclip-review@1",
             "verdicts": [{"id": "E03", "video": "clip-b", "time_s": 12.5, "vehicle_id": 41, "pedestrian_id": 2,
                           "verdict": "C", "note": "行人還在等"}]},
            STEMS)
        self.assertEqual(rows, [{"stem": "clip-b", "time_s": 12.5, "vehicle": 41, "ped": 2, "verdict": "C",
                                 "note": "行人還在等", "page": "E03"}])

    def test_the_old_file_maps_its_one_based_clip_number_onto_the_sorted_names(self):
        rows = newclip_score.verdict_rows([[2, 9.4, 413, "D"], [1, 4.733, 5, "A"]], STEMS)
        self.assertEqual([(row["stem"], row["time_s"], row["vehicle"], row["ped"], row["verdict"]) for row in rows],
                         [("clip-b", 9.4, 413, None, "D"), ("clip-a", 4.733, 5, None, "A")])

    def test_a_clip_number_that_the_run_does_not_have_says_so(self):
        with self.assertRaises(SystemExit) as caught:
            newclip_score.verdict_rows([[3, 1.0, 5, "A"]], STEMS)
        self.assertIn("3", str(caught.exception))

    def test_a_repeated_row_is_only_counted_once(self):
        payload = {"verdicts": [{"video": "clip-a", "time_s": 4.0, "vehicle_id": 5, "pedestrian_id": 1,
                                 "verdict": "A"},
                                {"video": "clip-a", "time_s": 4.0, "vehicle_id": 5, "pedestrian_id": 1,
                                 "verdict": "A"}]}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "verdicts.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(len(newclip_score.load_verdicts([path], STEMS)), 1)

    def test_two_vehicles_at_one_moment_are_two_verdicts(self):
        payload = {"verdicts": [{"video": "clip-a", "time_s": 4.0, "vehicle_id": 5, "pedestrian_id": 1,
                                 "verdict": "C"},
                                {"video": "clip-a", "time_s": 4.0, "vehicle_id": 6, "pedestrian_id": 1,
                                 "verdict": "A"}]}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "verdicts.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(len(newclip_score.load_verdicts([path], STEMS)), 2)


class JoinTest(unittest.TestCase):
    def clip(self, stem, events):
        return {"stem": stem, "events": events}

    def test_a_verdict_lands_on_its_own_vehicle_and_pedestrian_not_the_nearest_event(self):
        clip = self.clip("clip-a", [event(10.0, vehicle=7, ped=3), event(10.2, vehicle=9, ped=3)])
        stray = newclip_score.join_verdicts([clip], [{"stem": "clip-a", "time_s": 10.2, "vehicle": 7, "ped": 3,
                                                      "verdict": "A", "note": ""}])
        self.assertEqual(stray, [])
        self.assertEqual([judged["verdict"] for judged in clip["judged"]], ["A", None])

    def test_a_verdict_from_an_older_run_still_lands_by_time(self):
        clip = self.clip("clip-a", [event(10.0, vehicle=7)])
        newclip_score.join_verdicts([clip], [{"stem": "clip-a", "time_s": 10.9, "vehicle": 999, "ped": None,
                                              "verdict": "C", "note": ""}])
        self.assertEqual(clip["judged"][0]["verdict"], "C")

    def test_a_verdict_farther_than_the_tolerance_is_reported_not_attached(self):
        clip = self.clip("clip-a", [event(10.0)])
        stray = newclip_score.join_verdicts([clip], [{"stem": "clip-a", "time_s": 12.0, "vehicle": 7, "ped": 3,
                                                      "verdict": "C", "note": ""}])
        self.assertEqual(clip["judged"][0]["verdict"], None)
        self.assertEqual(len(stray), 1)

    def test_one_event_takes_one_verdict(self):
        clip = self.clip("clip-a", [event(10.0)])
        stray = newclip_score.join_verdicts([clip], [
            {"stem": "clip-a", "time_s": 10.1, "vehicle": 7, "ped": 3, "verdict": "A", "note": ""},
            {"stem": "clip-a", "time_s": 10.2, "vehicle": 7, "ped": 3, "verdict": "C", "note": ""}])
        self.assertEqual(clip["judged"][0]["verdict"], "A")
        self.assertEqual(len(stray), 1)


class PoolTest(unittest.TestCase):
    def judged(self, count, verdict):
        return [{"verdict": verdict} for _ in range(count)]

    def test_precision_is_a_over_judged_non_d(self):
        clips = [{"stem": "a", "events": [{}] * 4, "judged": self.judged(2, "A") + self.judged(1, "B")
                  + self.judged(1, "C")}]
        total = newclip_score.pool(clips)
        self.assertEqual((total["events"], total["judged"], total["unjudged"]), (4, 4, 0))
        self.assertAlmostEqual(total["precision"], 2 / 4)   # A / (A + B + C): a repeat is not a clean hit
        self.assertAlmostEqual(total["precision_all"], 2 / 4)

    def test_undecided_is_listed_but_not_a_false_alarm(self):
        clips = [{"stem": "a", "events": [{}] * 3, "judged": self.judged(1, "A") + self.judged(2, "D")}]
        total = newclip_score.pool(clips)
        self.assertEqual((total["D"], total["unjudged"], total["C"]), (2, 0, 0))
        self.assertAlmostEqual(total["precision"], 1.0)

    def test_totals_add_counts_instead_of_averaging_clip_rates(self):
        clips = [{"stem": "a", "events": [{}] * 2, "judged": self.judged(1, "A") + self.judged(1, "C")},
                 {"stem": "b", "events": [{}] * 10, "judged": self.judged(1, "A") + self.judged(9, "C")}]
        self.assertAlmostEqual(newclip_score.pool(clips)["precision"], 2 / 12)

    def test_an_unjudged_run_has_no_precision_instead_of_zero(self):
        total = newclip_score.pool([{"stem": "a", "events": [{}] * 3, "judged": self.judged(3, None)}])
        self.assertIsNone(total["precision"])
        self.assertEqual(total["unjudged"], 3)

    def test_combining_counts_a_clip_present_in_both_sets_once_and_prefers_the_development_copy(self):
        clips = [{"stem": "a", "run_role": "dev", "events": [{}] * 5, "judged": self.judged(5, "A")},
                 {"stem": "a", "run_role": "validation", "events": [{}] * 1, "judged": self.judged(1, "C")},
                 {"stem": "b", "run_role": "validation", "events": [{}] * 2, "judged": self.judged(1, "A")}]
        combined = newclip_score.dedupe(clips, preferred="dev")
        self.assertEqual([clip["stem"] for clip in combined], ["a", "b"])
        self.assertEqual(combined[0]["run_role"], "dev")
        self.assertEqual(newclip_score.pool(combined)["events"], 7)


class PageTest(unittest.TestCase):
    def payload(self, events):
        return {"schema": "yolo-ped/newclip-review@1", "generated_at": "2025-01-01T00:00:00+08:00",
                "storage_key": "yolo-ped-newclip-1234abcd", "export_prefix": "newclip-run",
                "events": events}

    def test_every_placeholder_is_filled_in(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        newclip_review.write_html(self.payload([{"id": "E01"}]), out, "標題", "說明", "頁尾")
        html = (out / "index.html").read_text(encoding="utf-8")
        for placeholder in ("__TITLE__", "__SUBTEXT__", "__TOTAL__", "__FOOTER__", "__STORAGE_KEY__",
                            "__EXPORT_PREFIX__", "__PAYLOAD__"):
            self.assertNotIn(placeholder, html)
        self.assertIn("標題", html)
        self.assertIn("newclip-run", html)

    def test_the_storage_key_follows_the_run_folder_so_two_batches_do_not_overwrite_each_other(self):
        first = newclip_review.storage_key_of(Path("C:/runs/one"))
        self.assertEqual(first, newclip_review.storage_key_of(Path("C:/runs/one")))
        self.assertNotEqual(first, newclip_review.storage_key_of(Path("C:/runs/two")))
        self.assertTrue(first.startswith("yolo-ped-newclip-"))


if __name__ == "__main__":
    unittest.main()
