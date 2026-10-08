"""Production per-frame regression checks; requires the trusted weights/ckpt.t7."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch

from detection_backend import DETECT_DIR, resolve_reid_checkpoint

sys.path.insert(0, str(DETECT_DIR))


def load_detector(task):
    path = DETECT_DIR / f"predict_{task}.py"
    spec = importlib.util.spec_from_file_location(f"traffic_{task}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrafficFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tf = load_detector("tf")
        cls.zebra = load_detector("zebra")
        cls.checkpoint = resolve_reid_checkpoint()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="traffic-rules-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.input_tensor = torch.zeros((3, 360, 640))

    def predictor(self, module, names, fps=60):
        # Known YOLO detections enter at the approved per-frame boundary; real
        # DeepSORT and traffic rules process them without collaborator mocks.
        module.deepsort = module.DeepSort(str(self.checkpoint), use_cuda=False, n_init=1, max_age=3)
        module.data_deque.clear()
        module.object_counter.clear()
        module.object_counter1.clear()
        module.log_flag = False
        if module is self.zebra:
            module.line.clear()
            module.direc.clear()
        predictor = module.DetectionPredictor(overrides={"save": False, "project": str(self.output), "name": "frames"})
        predictor.save_dir.mkdir(parents=True, exist_ok=True)
        predictor.model = SimpleNamespace(names=names)
        predictor.webcam = False
        predictor.seen = 0
        video = self.output / "source.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 360))
        assert writer.isOpened()
        writer.write(self.frame)
        writer.release()
        capture = cv2.VideoCapture(str(video))
        self.addCleanup(capture.release)
        predictor.dataset = SimpleNamespace(frame=1, cap=capture, mode="video")
        return predictor

    def process(self, predictor, boxes, frame=1):
        predictor.dataset.frame = frame
        detections = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 6)
        predictor.write_results(0, [detections], (Path("source.avi"), self.input_tensor, self.frame))
        return predictor.annotator.result()

    def test_bgr_traffic_lights_render_the_correct_decision(self):
        boxes = [[300, 70, 340, 140, .99, 0]]
        for color, expected in [((0, 0, 255), "Stop"), ((0, 255, 0), "Go"),
                                ((255, 0, 0), "Go"), ((0, 255, 255), "Stop")]:
            with self.subTest(color=color):
                self.frame[:] = 0
                self.frame[70:140, 300:340] = color
                predictor = self.predictor(self.tf, {0: "traffic light"})
                for frame in range(1, 4):
                    image = self.process(predictor, boxes, frame)
                expected_image = np.zeros_like(image)
                cv2.putText(expected_image, expected, (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0) if expected == "Stop" else (0, 255, 0), 1)
                np.testing.assert_array_equal(image[5:30, 15:110], expected_image[5:30, 15:110])

    def violation_sequence(self, module, fps=60, event_frame=61, zones=None, invalid_fps=False):
        names = {0: "car", 1: "traffic light" if module is self.tf else "person", 2: "zebra"}
        self.frame[:] = 0
        self.frame[70:140, 300:340] = (0, 0, 255)
        predictor = self.predictor(module, names, fps)
        other = [300, 70, 340, 140, .99, 1] if module is self.tf else [70, 140, 90, 180, .99, 1]
        zone = [50, 100, 500, 180 if module is self.tf else 220, .99, 2]
        zones = [zone] if zones is None else zones
        for frame in range(1, 4):
            self.process(predictor, [[130, 135, 150, 175, .99, 0], other, *zones], frame)
        self.process(predictor, [], event_frame - 1)
        if invalid_fps:
            predictor.dataset.cap.release()
        self.process(predictor, [[130, 148, 150, 188, .99, 0], other, *zones], event_frame)
        path = predictor.save_dir / "output.txt"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def test_event_time_uses_source_frames_in_both_modes(self):
        for module in (self.tf, self.zebra):
            for fps, frame in ((30, 31), (60, 61), (29.97, 31), (60, 1)):
                with self.subTest(mode=module.__name__, fps=fps):
                    # Each invocation has its own output; event logs are append-only.
                    log = self.violation_sequence(module, fps, frame).splitlines()[-1]
                    expected_time = "0.000 sec" if frame == 1 else "1.001 sec" if fps == 29.97 else "1.000 sec"
                    self.assertIn(expected_time, log)
                    self.assertIn(f"第{frame}幀", log)

    def test_event_with_invalid_source_fps_is_rejected(self):
        for module in (self.tf, self.zebra):
            with self.subTest(mode=module.__name__):
                with self.assertRaisesRegex(ValueError, "有效的來源幀率"):
                    self.violation_sequence(module, invalid_fps=True)

    def test_empty_frames_preserve_short_gaps_but_expire_old_identities(self):
        boxes = [[130, 135, 150, 175, .99, 0]]
        for module in (self.tf, self.zebra):
            with self.subTest(mode=module.__name__):
                predictor = self.predictor(module, {0: "car"})
                for frame in range(1, 4):
                    self.process(predictor, boxes, frame)
                def observed_id():
                    output = module.deepsort.update(torch.tensor([[140., 155., 20., 40.]]),
                                                   torch.tensor([[.99]]), [0], self.frame)
                    return int(output[0, -2])
                original = observed_id()
                for frame in (5, 6):
                    self.process(predictor, [], frame)
                self.assertEqual(observed_id(), original, "Two missing frames must not double-age the track")
                for frame in (8, 9, 10, 11):
                    self.process(predictor, [], frame)
                for frame in (12, 13):
                    self.process(predictor, boxes, frame)
                self.assertNotEqual(observed_id(), original, "Expired objects must get a new identity")

    def test_crosswalk_matching_is_order_independent(self):
        containing = [50, 100, 350, 220, .99, 2]
        unrelated = [400, 100, 600, 220, .99, 2]
        cases = [([containing, unrelated], True), ([unrelated, containing], True),
                 ([], False), ([unrelated], False), ([containing, containing], True)]
        for zones, expected_event in cases:
            with self.subTest(zones=zones):
                log = self.violation_sequence(self.zebra, zones=zones)
                self.assertEqual(bool(log), expected_event)
                if log:
                    self.assertEqual(len(log.splitlines()), 1)


if __name__ == "__main__":
    # Legacy trusted checkpoints use pickle; same process-scoped policy as production.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    unittest.main()
