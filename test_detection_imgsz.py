"""Inference size and augmentation defaults for the legacy detection backend."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import detection_backend as backend

WIDTH, HEIGHT = 64, 48


class InferenceSizeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="inference imgsz ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.video = self.root / "clip.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (WIDTH, HEIGHT))
        self.assertTrue(writer.isOpened())
        writer.write(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        writer.release()

        self.detect_dir = self.root / "detect"
        self.detect_dir.mkdir()
        self.captured = self.root / "captured.txt"
        (self.detect_dir / "predict_tf.py").write_text(
            "import sys\n"
            "from pathlib import Path\n"
            f"Path({str(self.captured)!r}).write_text('\\n'.join(sys.argv[1:]), encoding='utf-8')\n"
            "print('(1/1)', flush=True)\n",
            encoding="utf-8",
        )
        weights = self.root / "weights"
        weights.mkdir()
        model = weights / "yolov8s_tf.pt"
        model.touch()
        checkpoint = weights / "ckpt.t7"
        checkpoint.touch()
        self.environment = {
            "YOLO_WEIGHTS_DIR": str(weights),
            "YOLO_TF_MODEL": str(model),
            "DEEPSORT_REID_CKPT": str(checkpoint),
        }
        patcher = patch.object(backend, "DETECT_DIR", self.detect_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        saved_imgsz = os.environ.pop("TRAFFIC_IMGSZ", None)
        if saved_imgsz is not None:
            self.addCleanup(os.environ.__setitem__, "TRAFFIC_IMGSZ", saved_imgsz)

    def run_backend(self, **environment):
        with patch.dict(os.environ, {**self.environment, **environment}):
            return list(backend.run_detection([str(self.video)], str(self.output), "yolov8s", "tf"))

    def captured_arguments(self):
        return self.captured.read_text(encoding="utf-8").splitlines()

    def test_environment_variable_overrides_the_video_width(self):
        events = self.run_backend(TRAFFIC_IMGSZ="1920")
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("imgsz=1920", self.captured_arguments())

    def test_video_width_is_used_when_the_environment_variable_is_unset(self):
        events = self.run_backend()
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn(f"imgsz={WIDTH}", self.captured_arguments())

    def test_blank_environment_variable_falls_back_to_the_video_width(self):
        events = self.run_backend(TRAFFIC_IMGSZ="  ")
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn(f"imgsz={WIDTH}", self.captured_arguments())

    def test_invalid_values_are_rejected_with_an_actionable_message(self):
        for value in ("fast", "0", "-1", "640.5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    self.run_backend(TRAFFIC_IMGSZ=value)
                message = str(caught.exception)
                self.assertIn("TRAFFIC_IMGSZ", message)
                self.assertIn(value, message)
                self.assertIn("正整數", message)

    def test_augment_is_disabled_by_default(self):
        events = self.run_backend()
        self.assertEqual(events[-1]["type"], "done")
        arguments = self.captured_arguments()
        self.assertIn("augment=False", arguments)
        self.assertNotIn("augment=True", arguments)


if __name__ == "__main__":
    unittest.main()
