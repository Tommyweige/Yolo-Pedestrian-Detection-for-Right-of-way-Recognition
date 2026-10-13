"""Labeling store and evaluation behaviour (no window, no GPU)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_events import aggregate, score  # noqa: E402
from label_events import FAILURE, YIELDED, LabelStore, time_of  # noqa: E402


def lines(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


class LabelStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "labels.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def test_time_uses_one_based_frame_over_fps_like_the_pipeline(self):
        self.assertEqual(time_of(660, 60.0), 11.0)   # the detected taxi event: f660 -> 11.0 s
        self.assertEqual(time_of(1, 30.0), 0.033)

    def test_marks_are_saved_immediately_sorted_and_undoable(self):
        store = LabelStore(self.path, "a.mp4", 60.0)
        store.add(FAILURE, 660)
        store.add(YIELDED, 300)
        self.assertEqual([r["frame"] for r in lines(self.path)], [300, 660])
        self.assertEqual(store.undo()["frame"], 300)  # undo removes the last mark made, not the last in time
        self.assertEqual([r["kind"] for r in lines(self.path)], [FAILURE])
        self.assertEqual(lines(self.path)[0]["time_s"], 11.0)

    def test_resumes_from_an_existing_file(self):
        LabelStore(self.path, "a.mp4", 60.0).add(FAILURE, 660)
        again = LabelStore(self.path, "a.mp4", 60.0)
        self.assertEqual(len(again.rows), 1)
        again.add(FAILURE, 900)
        self.assertEqual(len(lines(self.path)), 2)

    def test_one_file_per_video(self):
        LabelStore(self.path, "a.mp4", 60.0).add(FAILURE, 660)
        with self.assertRaises(ValueError):
            LabelStore(self.path, "b.mp4", 30.0)

    def test_nothing_is_written_until_something_is_decided(self):
        store = LabelStore(self.path, "a.mp4", 60.0)
        store.save()
        self.assertFalse(self.path.exists())
        self.assertTrue(store.declare_none())
        self.assertEqual(self.path.read_text(), "")  # empty file = reviewed, no events

    def test_declaring_no_events_cannot_wipe_existing_marks(self):
        store = LabelStore(self.path, "a.mp4", 60.0)
        store.add(FAILURE, 660)
        self.assertFalse(store.declare_none())
        self.assertEqual(len(lines(self.path)), 1)


class EvaluationTests(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_events.py"), *map(str, arguments)],
                              capture_output=True, text=True, encoding="utf-8")

    def test_pipeline_event_kind_is_scored_as_failure_to_yield(self):
        predicted = [{"kind": "vehicle_on_crosswalk_with_pedestrian", "time_s": 11.0}]
        from eval_events import ALIASES
        predicted = [{**e, "kind": ALIASES.get(e["kind"], e["kind"])} for e in predicted]
        truth = [{"kind": FAILURE, "time_s": 10.6}]
        self.assertEqual(score(predicted, truth, 1.0)[FAILURE]["true_positive"], 1)

    def test_negative_video_counts_false_alarms_in_the_overall_total(self):
        positive = ([{"kind": FAILURE, "time_s": 11.0}], [{"kind": FAILURE, "time_s": 11.2}])
        negative = ([{"kind": FAILURE, "time_s": 5.0}], [])      # a detection in a video with no events
        overall = aggregate([positive, negative], 1.0)[FAILURE]
        self.assertEqual((overall["true_positive"], overall["false_alarm"], overall["miss"]), (1, 1, 0))
        self.assertAlmostEqual(overall["precision"], 0.5)

    def test_cli_pairs_empty_labels_need_an_explicit_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "e.jsonl").write_text(json.dumps({"kind": "vehicle_on_crosswalk_with_pedestrian", "time_s": 5.0}) + "\n")
            (root / "none.jsonl").write_text("")
            (root / "yes.jsonl").write_text(json.dumps({"kind": FAILURE, "time_s": 5.3}) + "\n")
            refused = self.run_cli("--pair", root / "e.jsonl", root / "none.jsonl")
            self.assertEqual(refused.returncode, 2)
            self.assertIn("--allow-empty-truth", refused.stderr)
            allowed = self.run_cli("--pair", root / "e.jsonl", root / "none.jsonl", "--pair", root / "e.jsonl", root / "yes.jsonl",
                                   "--allow-empty-truth")
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            report = json.loads(allowed.stdout)
            self.assertEqual(report["videos"], 2)
            self.assertEqual(report["overall"][FAILURE]["false_alarm"], 1)
            self.assertEqual(report["overall"][FAILURE]["true_positive"], 1)


if __name__ == "__main__":
    unittest.main()
