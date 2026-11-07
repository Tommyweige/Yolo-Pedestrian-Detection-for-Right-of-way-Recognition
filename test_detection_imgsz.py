"""Inference size, augmentation and precision defaults for the legacy detection backend."""
import os
from pathlib import Path
import sys
import tempfile
import types
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
        saved_half = os.environ.pop("TRAFFIC_HALF", None)
        if saved_half is not None:
            self.addCleanup(os.environ.__setitem__, "TRAFFIC_HALF", saved_half)

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

    def test_environment_variable_controls_half_precision(self):
        for value, expected in (("1", "half=True"), ("true", "half=True"),
                                ("0", "half=False"), ("false", "half=False"),
                                ("TRUE", "half=True"), ("False", "half=False")):
            with self.subTest(value=value):
                events = self.run_backend(TRAFFIC_HALF=value)
                self.assertEqual(events[-1]["type"], "done")
                self.assertIn(expected, self.captured_arguments())

    def test_half_defaults_to_false_without_cuda(self):
        with patch.object(backend, "_CUDA_AVAILABLE", None), \
                patch.dict(sys.modules, {"torch": None}):
            events = self.run_backend()
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("half=False", self.captured_arguments())


class InferenceHalfTest(unittest.TestCase):
    """TRAFFIC_HALF parsing and the CUDA probe, without needing real torch or a GPU."""

    def setUp(self):
        saved = os.environ.pop("TRAFFIC_HALF", None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, "TRAFFIC_HALF", saved)

    def resolve(self, cuda):
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda))
        with patch.object(backend, "_CUDA_AVAILABLE", None), \
                patch.dict(sys.modules, {"torch": fake}):
            return backend._inference_half()

    def test_cuda_availability_selects_the_default(self):
        self.assertIs(self.resolve(True), True)
        self.assertIs(self.resolve(False), False)

    def test_invalid_values_are_rejected_with_an_actionable_message(self):
        for value in ("yes", "2", "tru", "on"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"TRAFFIC_HALF": value}):
                    with self.assertRaises(ValueError) as caught:
                        backend._inference_half()
                message = str(caught.exception)
                self.assertIn("TRAFFIC_HALF", message)
                self.assertIn(value, message)
                self.assertIn("true", message)

    def test_missing_torch_falls_back_to_fp32(self):
        with patch.object(backend, "_CUDA_AVAILABLE", None), \
                patch.dict(sys.modules, {"torch": None}):
            self.assertIs(backend._inference_half(), False)

    def test_probe_errors_fall_back_to_fp32(self):
        def is_available():
            raise RuntimeError("CUDA driver mismatch")

        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=is_available))
        with patch.object(backend, "_CUDA_AVAILABLE", None), \
                patch.dict(sys.modules, {"torch": fake}):
            self.assertIs(backend._inference_half(), False)

    def test_the_cuda_probe_runs_once_per_process(self):
        calls = []

        def is_available():
            calls.append(True)
            return True

        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=is_available))
        with patch.object(backend, "_CUDA_AVAILABLE", None), \
                patch.dict(sys.modules, {"torch": fake}):
            self.assertIs(backend._inference_half(), True)
            self.assertIs(backend._inference_half(), True)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
