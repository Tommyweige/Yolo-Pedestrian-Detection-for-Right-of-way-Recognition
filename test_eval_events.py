import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from event_scoring import derive_tolerance, incident_labels, matched_pairs, score  # noqa: E402

F = "suspected_failure_to_yield"


class FixedEvaluation(unittest.TestCase):
    """The rules ADR 0003 fixes, so a number can be compared across rule versions."""

    def test_tolerance_is_the_edge_of_the_residual_cluster(self):
        # the labelled clips' manual residuals: five matches within 0.8 s, then a 0.87 s gap to 1.667 s
        tolerance, basis = derive_tolerance([0.433, 0.2, 0.8, 0.333, 0.4, 1.667, 1.72, 1.885])
        self.assertEqual(tolerance, 1.25)
        self.assertEqual(basis["method"], "cluster_edge")
        self.assertEqual(basis["gap_s"], 0.867)

    def test_a_smooth_spread_falls_back_to_the_documented_one_second(self):
        tolerance, basis = derive_tolerance([0.1, 0.3, 0.5, 0.7, 0.9])
        self.assertEqual((tolerance, basis["method"]), (1.0, "fallback_labeler_resolution"))
        self.assertEqual(derive_tolerance([]), (1.0, {"method": "fallback_labeler_resolution", "sorted": []}))

    def test_labels_closer_than_the_tolerance_are_one_incident(self):
        truth = [{"kind": F, "time_s": t} for t in (41.2, 41.867, 44.0)]
        self.assertEqual([[truth[i]["time_s"] for i in group] for group in incident_labels(truth, 1.25)],
                         [[41.2, 41.867], [44.0]])

    def test_matching_is_one_to_one_closest_first(self):
        predicted = [{"kind": F, "time_s": 5.2}, {"kind": F, "time_s": 5.0}]
        truth = [{"kind": F, "time_s": 5.1}]
        self.assertEqual(sorted(matched_pairs(predicted, truth, 1.0)), [(1, 0)])


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
