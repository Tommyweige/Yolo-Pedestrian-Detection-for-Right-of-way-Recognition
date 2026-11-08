import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import tune_rule_lovo as t  # noqa: E402


class TuneRuleLovo(unittest.TestCase):
    def test_configs_cartesian_in_key_and_list_order(self):
        self.assertEqual(t.configs({"a": [1, 2], "b": [3]}), [{"a": 1, "b": 3}, {"a": 2, "b": 3}])

    def test_config_key_is_sorted_json(self):
        self.assertEqual(t.config_key({"b": 1, "a": 2}), '{"a": 2, "b": 1}')

    def test_distance_relative_with_zero_default(self):
        cfg = {"a": 2.0, "b": 0.5, "c": 3.0}
        default = {"a": 1.0, "b": 0.0, "c": 1.0}
        self.assertAlmostEqual(t.distance(cfg, default), 4.0)
        self.assertEqual(t.distance({"a": 1.0, "b": 0.0, "c": 1.0}, default), 0.0)

    def test_micro_f1_sums_rows_and_empty_is_zero(self):
        self.assertAlmostEqual(t.micro_f1([(2, 1, 2), (0, 0, 1)]), 2 * 1 / (2 + 3))
        self.assertEqual(t.micro_f1([]), 0.0)

    def test_nested_lovo_hand_computed_folds_and_total(self):
        default = {"x": 1.0}
        a, b = {"x": 1.0}, {"x": 2.0}
        table = {
            t.config_key(a): {"v1": [1, 1, 1], "v2": [0, 0, 1], "v3": [2, 1, 2]},
            t.config_key(b): {"v1": [1, 0, 1], "v2": [2, 2, 2], "v3": [0, 0, 1]},
        }
        out = t.nested_lovo(table, [a, b], default)

        self.assertEqual([out["per_video"][v]["chosen"] for v in ("v1", "v2", "v3")], [b, a, a])
        self.assertEqual(
            {v: [out["per_video"][v][k] for k in ("predicted", "true_positive", "labelled")]
             for v in ("v1", "v2", "v3")},
            {"v1": [1, 0, 1], "v2": [0, 0, 1], "v3": [2, 1, 2]},
        )
        self.assertEqual((out["total"]["predicted"], out["total"]["true_positive"],
                          out["total"]["labelled"]), (3, 1, 4))
        self.assertAlmostEqual(out["total"]["precision"], 1 / 3)
        self.assertAlmostEqual(out["total"]["recall"], 1 / 4)
        self.assertAlmostEqual(out["total"]["f1"], 2 / 7)

        self.assertEqual(out["in_sample_best"]["config"], a)
        self.assertAlmostEqual(out["in_sample_best"]["f1"], 4 / 7)
        self.assertAlmostEqual(out["default_f1"], 4 / 7)

    def test_nested_lovo_tie_picks_closest_to_default(self):
        default = {"x": 1.0}
        a, b = {"x": 1.0}, {"x": 2.0}
        rows = {"v1": [1, 1, 1], "v2": [1, 0, 1]}
        table = {t.config_key(a): dict(rows), t.config_key(b): dict(rows)}
        out = t.nested_lovo(table, [a, b], default)

        self.assertTrue(all(out["per_video"][v]["chosen"] == a for v in ("v1", "v2")))
        self.assertEqual(out["in_sample_best"]["config"], a)

    def test_default_f1_is_none_when_default_is_not_in_table(self):
        a = {"x": 1.0}
        table = {t.config_key(a): {"v1": [1, 1, 1]}}
        out = t.nested_lovo(table, [a], {"x": 0.5})
        self.assertIsNone(out["default_f1"])

    def test_main_reads_cache_and_prints_json_summary(self):
        grid = {"entry_window_s": [1.0, 2.0]}
        a, b = {"entry_window_s": 1.0}, {"entry_window_s": 2.0}
        table = {
            t.config_key(a): {"v1": [1, 1, 1], "v2": [0, 0, 1]},
            t.config_key(b): {"v1": [1, 0, 1], "v2": [2, 2, 2]},
        }
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "cache.json"
            cache.write_text(json.dumps({"table": table, "grid": grid, "tolerance": t.scoring.TOLERANCE_S}),
                             encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = t.main(["--cache", str(cache)])
            self.assertEqual(code, 0)
            last = json.loads(buf.getvalue().strip().splitlines()[-1])
            self.assertIn("lovo", last)
            self.assertIn("chosen", last)
            self.assertEqual(set(last["chosen"]), {"v1", "v2"})

    def test_a_cache_counted_at_another_tolerance_is_refused(self):
        # a cache without the key predates the frozen tolerance and was counted at 1.0 s
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "cache.json"
            cache.write_text(json.dumps({"table": {}, "grid": {}}), encoding="utf-8")
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(t.main(["--cache", str(cache)]), 2)
            self.assertIn("built at an unrecorded tolerance, not 1.25s", err.getvalue())


if __name__ == "__main__":
    unittest.main()
