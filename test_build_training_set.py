from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_training_set import clip_box, merge_teacher, read_label_file, write_yolo  # noqa: E402

NAMES = ["bus", "car", "motorcycle", "traffic light", "zebra"]


class BuildTrainingSet(unittest.TestCase):
    def test_polygon_becomes_bounding_box_and_boxes_pass_through(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.txt"
            path.write_text("1 0.5 0.5 0.2 0.2\n4 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\nbroken\n")
            rows = read_label_file(path, NAMES)
        self.assertEqual(rows[0][0], "car")
        self.assertEqual(rows[1], ("zebra", (0.1, 0.2, 0.5, 0.6)))
        self.assertEqual(len(rows), 2)

    def test_teacher_never_adds_zebra_low_score_or_covered_objects(self):
        human = [("car", (0.1, 0.1, 0.3, 0.3))]
        teacher = [("car", (0.11, 0.11, 0.3, 0.3), 0.9),         # covered by human
                   ("person", (0.5, 0.5, 0.6, 0.8), 0.9),        # new, kept
                   ("person", (0.7, 0.5, 0.8, 0.8), 0.2),        # low score
                   ("zebra", (0.0, 0.8, 1.0, 1.0), 0.95)]        # teacher zebra is banned
        merged, added = merge_teacher(human, teacher, 0.45)
        self.assertEqual(added, 1)
        self.assertEqual([n for n, _ in merged], ["car", "person"])

    def test_boxes_are_clipped_and_degenerate_ones_dropped(self):
        self.assertEqual(clip_box((-0.1, 0.2, 0.5, 1.2)), (0.0, 0.2, 0.5, 1.0))
        self.assertIsNone(clip_box((0.5, 0.5, 0.5, 0.9)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.txt"
            count = write_yolo(path, [("person", (0.1, 0.1, 0.3, 0.5)), ("car", (0.4, 0.4, 0.4, 0.8))])
            self.assertEqual(count, 1)
            self.assertEqual(len(path.read_text().split()), 5)


if __name__ == "__main__":
    unittest.main()
