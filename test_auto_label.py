"""auto_label.py regressions with the offline stub teacher (CPU only, synthetic images)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "auto_label.py"
W, H = 200, 100


def run(*arguments):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)],
                          capture_output=True, text=True, encoding="utf-8")


class AutoLabel(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "數據 set"
        (self.dataset / "images").mkdir(parents=True)
        names = ["multi", "zero", "bad", "broken", "error"]
        rows = []
        for index, name in enumerate(names):
            file = self.dataset / "images" / f"{name}.png"
            if name == "broken":
                file.write_bytes(b"junk")
            else:
                ok, buffer = cv2.imencode(".png", np.full((H, W, 3), 40 * index, np.uint8))
                buffer.tofile(str(file))
            rows.append({"image_id": name, "file": f"images/{name}.png", "split": "train", "sha256": name})
        rows.append({"image_id": "val1", "file": "images/multi.png", "split": "val", "sha256": "v"})
        (self.dataset / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        nan = float("nan")
        stub = {
            "multi": [
                {"prompt": "pedestrian", "score": 0.9, "box": [10, 10, 50, 90]},
                {"prompt": "pedestrian", "score": 0.5, "box": [11, 11, 50, 90]},   # NMS duplicate
                {"prompt": "car", "score": 0.4, "box": [60, 40, 190, 95]},         # low confidence, kept
                {"prompt": "scooter", "score": 0.8, "box": [100, 5, 106, 20]},      # tiny, must be kept
                {"prompt": "zebra crossing", "score": 0.7, "box": [0, 60, 200, 100]},
                {"prompt": "bus", "score": 0.1, "box": [0, 0, 10, 10]},             # below threshold
            ],
            "zero": [],
            "bad": [
                {"prompt": "car", "score": 0.9, "box": [30, 30, 30, 60]},          # zero area
                {"prompt": "car", "score": 0.9, "box": [-50, 0, 20, 20]},          # outside
                {"prompt": "car", "score": 0.9, "box": [0, 0, nan, 20]},           # non finite
                {"prompt": "dog", "score": 0.9, "box": [0, 0, 20, 20]},            # unknown class
                {"prompt": "truck", "score": 0.9, "box": [0, 0, 201, 100]},        # within tolerance, clipped
            ],
            "error": "error",
        }
        # json cannot hold NaN strictly, but Python round-trips it; matches what a model could emit
        self.stub = self.root / "stub.json"
        self.stub.write_text(json.dumps(stub), encoding="utf-8")
        self.out = self.root / "out"

    def tearDown(self):
        self.temp.cleanup()

    def label(self, *extra):
        return run("--dataset", self.dataset, "--output", self.out, "--teacher", "stub",
                   "--stub-file", self.stub, *extra)

    def test_labels_sidecars_and_review(self):
        result = self.label()
        self.assertEqual(result.returncode, 0, result.stderr)
        multi = (self.out / "labels" / "multi.txt").read_text().splitlines()
        self.assertEqual(len(multi), 4)  # person, car, motorcycle, zebra; duplicate + bus removed
        classes = sorted(int(line.split()[0]) for line in multi)
        self.assertEqual(classes, [0, 1, 2, 6])
        for line in multi:
            fields = line.split()
            self.assertEqual(len(fields), 5)
            self.assertTrue(all(0 <= float(v) <= 1 for v in fields[1:]))
        self.assertEqual((self.out / "labels" / "zero.txt").read_text(), "")
        bad = (self.out / "labels" / "bad.txt").read_text().splitlines()
        self.assertEqual(len(bad), 1)  # only the clipped truck survives
        self.assertTrue(bad[0].startswith("4 "))
        sidecar = json.loads((self.out / "sidecars" / "multi.json").read_text())
        self.assertTrue(sidecar["pseudo_label"] and not sidecar["gold"])
        self.assertIn("prompt", sidecar["boxes"][0])
        self.assertFalse((self.out / "labels" / "val1.txt").exists())  # val not labelled by default
        reasons = {json.loads(l)["image_id"]: json.loads(l)["reasons"]
                   for l in (self.out / "review_queue.jsonl").read_text().splitlines()}
        self.assertIn("low_confidence", reasons["multi"])
        self.assertIn("zero_detection", reasons["zero"])
        self.assertIn("filtered_boxes", reasons["bad"])
        self.assertIn("corrupt_image", reasons["broken"])
        self.assertIn("teacher_error", reasons["error"])
        reasons_bad = {f["reason"] for f in json.loads((self.out / "sidecars" / "bad.json").read_text())["filtered"]}
        self.assertEqual(reasons_bad, {"non_positive_area", "outside_image", "non_finite", "unknown_class"})
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["small_objects"], 1)
        self.assertEqual(report["errors"], 2)
        self.assertAlmostEqual(report["zero_detection_rate"], 1 / 3, places=3)
        self.assertTrue((self.out / "gallery" / "index.html").is_file())

    def test_rerun_is_identical_and_resumes(self):
        self.label()
        first = {p.name: p.read_text() for p in (self.out / "labels").iterdir()}
        result = self.label()
        self.assertEqual({p.name: p.read_text() for p in (self.out / "labels").iterdir()}, first)
        self.assertEqual(json.loads((self.out / "report.json").read_text())["resumed_from_sidecar"], 5)

    def test_dry_run_sample_is_seeded_and_source_untouched(self):
        before = (self.dataset / "manifest.jsonl").read_bytes()
        self.label("--dry-run", "2", "--seed", "1")
        self.assertEqual(len(list((self.out / "sidecars").glob("*.json"))), 2)
        self.assertEqual((self.dataset / "manifest.jsonl").read_bytes(), before)

    def test_requires_manifest_and_separate_output(self):
        self.assertEqual(run("--dataset", self.root, "--output", self.out, "--stub-file", self.stub).returncode, 2)
        self.assertEqual(run("--dataset", self.dataset, "--output", self.dataset, "--stub-file", self.stub).returncode, 2)


if __name__ == "__main__":
    unittest.main()
