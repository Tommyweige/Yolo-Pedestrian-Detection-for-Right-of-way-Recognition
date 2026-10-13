import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_events import score  # noqa: E402

F = "suspected_failure_to_yield"


class EvalEvents(unittest.TestCase):
    def test_counts_hits_false_alarms_and_misses(self):
        predicted = [{"kind": F, "time_s": 5.0}, {"kind": F, "time_s": 9.0}]
        truth = [{"kind": F, "time_s": 5.4}, {"kind": F, "time_s": 20.0}]
        r = score(predicted, truth, 1.0)[F]
        self.assertEqual((r["true_positive"], r["false_alarm"], r["miss"]), (1, 1, 1))
        self.assertAlmostEqual(r["precision"], 0.5)
        self.assertAlmostEqual(r["f1"], 0.5)

    def test_matching_is_one_to_one_and_kind_specific(self):
        predicted = [{"kind": F, "time_s": 5.0}, {"kind": F, "time_s": 5.2}]
        r = score(predicted, [{"kind": F, "time_s": 5.1}], 1.0)[F]
        self.assertEqual((r["true_positive"], r["false_alarm"]), (1, 1))
        other = score([{"kind": "yielded", "time_s": 5.0}], [{"kind": F, "time_s": 5.0}], 1.0)
        self.assertEqual(other[F]["true_positive"], 0)

    def test_empty_truth_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "t.jsonl").write_text("")
            (Path(folder) / "p.jsonl").write_text("")
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_events.py"), "--predicted",
                                     str(Path(folder) / "p.jsonl"), "--truth", str(Path(folder) / "t.jsonl")],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
