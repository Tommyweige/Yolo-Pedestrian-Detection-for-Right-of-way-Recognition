"""Evaluation behaviour for human event labels (the labeling store itself is tested in rust-ui/src/labels.rs)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_events import aggregate, score  # noqa: E402

FAILURE = "suspected_failure_to_yield"


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
