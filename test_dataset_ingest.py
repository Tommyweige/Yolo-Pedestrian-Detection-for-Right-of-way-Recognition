"""Dataset ingestion regressions using only synthetic media."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "dataset_ingest.py"


def write_video(path, seed, frames=30):
    rng = np.random.RandomState(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(frames):
        writer.write(rng.randint(0, 255, (48, 64, 3), dtype=np.uint8))
    writer.release()


def write_image(path, seed, gray=False):
    rng = np.random.RandomState(seed)
    shape = (32, 32) if gray else (32, 32, 3)
    ok, buffer = cv2.imencode(path.suffix, rng.randint(0, 255, shape, dtype=np.uint8))
    assert ok
    buffer.tofile(str(path))


def tree_hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def run(*arguments):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)],
                          capture_output=True, text=True, encoding="utf-8")


def read_manifest(output):
    return [json.loads(line) for line in (Path(output) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]


class DatasetIngest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.videos = self.root / "影片 source"  # Chinese + space path
        self.videos.mkdir()
        for index in range(6):
            write_video(self.videos / f"clip{index}.avi", seed=index)

    def tearDown(self):
        self.temp.cleanup()

    def test_help_lists_options(self):
        result = run("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--roboflow-export", result.stdout)

    def test_no_video_group_crosses_splits_and_sources_untouched(self):
        before = tree_hashes(self.videos)
        output = self.root / "out"
        result = run("--output", output, "--videos", self.videos, "--frame-interval", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(tree_hashes(self.videos), before)
        rows = read_manifest(output)
        self.assertEqual(len(rows), 6 * 6)
        splits_by_group = {}
        for row in rows:
            splits_by_group.setdefault(row["source_group"], set()).add(row["split"])
            self.assertFalse(row["gold"])
            self.assertEqual(row["annotations_origin"], "none")
        self.assertTrue(all(len(s) == 1 for s in splits_by_group.values()))
        self.assertEqual({"train", "val", "test"}, {s for v in splits_by_group.values() for s in v})
        self.assertEqual(rows[0]["source_fps"], 10.0)
        self.assertIn("video_sha256", rows[0])

    def test_seed_is_deterministic(self):
        first, second, other = (self.root / name for name in ("a", "b", "c"))
        run("--output", first, "--videos", self.videos, "--frame-interval", "10", "--seed", "7")
        run("--output", second, "--videos", self.videos, "--frame-interval", "10", "--seed", "7")
        run("--output", other, "--videos", self.videos, "--frame-interval", "10", "--seed", "8")
        self.assertEqual((first / "manifest.jsonl").read_text(), (second / "manifest.jsonl").read_text())
        self.assertNotEqual((first / "manifest.jsonl").read_text(), (other / "manifest.jsonl").read_text())

    def test_duplicate_content_never_crosses_splits(self):
        images = self.root / "imgs"
        for folder in ("a", "b", "c", "d"):
            (images / folder).mkdir(parents=True)
            write_image(images / folder / "same_name.jpg", seed=100)  # identical content everywhere
        write_image(images / "a" / "unique.png", seed=1)
        write_image(images / "b" / "unique.png", seed=2, gray=True)
        output = self.root / "out"
        result = run("--output", output, "--images", images)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = read_manifest(output)
        self.assertEqual(len([r for r in rows if r["sha256"] == rows[0]["sha256"]]), 1)
        hashes = [r["sha256"] for r in rows]
        self.assertEqual(len(hashes), len(set(hashes)))
        rejected = (output / "rejected.jsonl").read_text()
        self.assertEqual(rejected.count("duplicate_content"), 3)
        # groups holding the duplicate were merged
        self.assertEqual(len({r["source_group"] for r in rows if r["sha256"] == rows[0]["sha256"]}), 1)

    def test_corrupt_bad_extension_and_empty_inputs(self):
        images = self.root / "mixed"
        images.mkdir()
        (images / "broken.jpg").write_bytes(b"not an image")
        (images / "notes.txt").write_text("hi")
        write_image(images / "ok.png", seed=3)
        output = self.root / "out"
        result = run("--output", output, "--images", images)
        self.assertEqual(result.returncode, 0, result.stderr)
        reasons = (output / "rejected.jsonl").read_text()
        self.assertIn("corrupt_image", reasons)
        self.assertIn("unsupported_extension", reasons)
        self.assertEqual(len(read_manifest(output)), 1)
        empty = self.root / "empty"
        empty.mkdir()
        result = run("--output", self.root / "out2", "--images", empty)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Nothing to ingest", result.stderr)

    def test_train_only_and_no_overwrite(self):
        output = self.root / "out"
        run("--output", output, "--videos", self.videos, "--frame-interval", "15", "--train-only")
        self.assertEqual({r["split"] for r in read_manifest(output)}, {"train"})
        again = run("--output", output, "--videos", self.videos)
        self.assertEqual(again.returncode, 1)
        self.assertIn("Refusing to overwrite", again.stderr)

    def test_roboflow_export_keeps_labels_without_gold(self):
        export = self.root / "rf"
        (export / "train" / "images").mkdir(parents=True)
        (export / "train" / "labels").mkdir()
        write_image(export / "train" / "images" / "car_jpg.rf.abc.jpg", seed=9)
        (export / "train" / "labels" / "car_jpg.rf.abc.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        output = self.root / "out"
        result = run("--output", output, "--roboflow-export", export, "--train-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        row = read_manifest(output)[0]
        self.assertEqual(row["annotations_origin"], "roboflow_export")
        self.assertFalse(row["gold"])
        self.assertEqual((output / row["label_file"]).read_text(), "0 0.5 0.5 0.2 0.2\n")

    def test_output_inside_source_is_rejected(self):
        images = self.root / "imgs"
        images.mkdir()
        write_image(images / "a.jpg", seed=1)
        result = run("--output", images / "out", "--images", images)
        self.assertEqual(result.returncode, 2)
        self.assertIn("must not overlap", result.stderr)


if __name__ == "__main__":
    unittest.main()
