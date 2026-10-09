"""Run with a Python environment containing opencv-python and numpy."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import importlib.util
from unittest.mock import patch

import cv2
import numpy as np

import detection_backend as backend


def check():
    root = Path(__file__).resolve().parent
    environment = backend._subprocess_environment(root / "weights" / "ckpt.t7")
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(backend.DETECT_DIR)
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    with tempfile.TemporaryDirectory(prefix="traffic 測試 ") as directory:
        temporary = Path(directory)
        video = temporary / "input 影片.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        assert writer.isOpened()
        for index in range(10):
            writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
        writer.release()
        weights = temporary / "weights"
        weights.mkdir()
        (weights / "yolov8s_tf.pt").touch()
        (weights / "ckpt.t7").touch()
        script = temporary / "predict_tf.py"
        script.write_text("import sys\nprint('(1/2)', flush=True)\nprint('(2/2)', flush=True)\n", encoding="utf-8")
        configured = {"YOLO_WEIGHTS_DIR": str(weights), "YOLO_TF_MODEL": str(weights / "yolov8s_tf.pt"),
                      "DEEPSORT_REID_CKPT": str(weights / "ckpt.t7")}
        with patch.object(backend, "DETECT_DIR", temporary), patch.dict(os.environ, configured):
            events = list(backend.run_detection([str(video)], str(temporary), "yolov8s", "tf"))
            assert events[-1]["type"] == "done"
            assert any(e.get("current") == 2 and e.get("total") == 2 for e in events)
            if importlib.util.find_spec("moviepy") is not None:
                events = list(backend.run_detection([str(video)], str(temporary), "yolov8s", "tf", 30))
                assert events[-1]["type"] == "done"
            script.write_text("import sys\nprint('failure detail', flush=True)\nsys.exit(7)\n", encoding="utf-8")
            try:
                list(backend.run_detection([str(video)], str(temporary), "yolov8s", "tf"))
                raise AssertionError("Expected subprocess failure")
            except RuntimeError as error:
                assert "failure detail" in str(error) and "7" in str(error)
            script.write_text("import time\nprint('(1/2)', flush=True)\ntime.sleep(60)\n", encoding="utf-8")
            cancelled = threading.Event()
            timer = threading.Timer(0.5, cancelled.set)
            timer.start()
            try:
                events = list(backend.run_detection([str(video)], str(temporary), "yolov8s", "tf", cancelled=cancelled.is_set))
                assert events[-1]["type"] == "cancelled"
            finally:
                timer.cancel()
        # No Qt dependency is loaded by either shared backend or bridge.
        assert not any(name.startswith("PyQt5") for name in sys.modules)
        process = subprocess.Popen([sys.executable, str(root / "desktop_bridge.py"), "detect"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", cwd=temporary,
                                   env={**os.environ, "YOLO_TF_MODEL": str(temporary / "missing.pt")})
        process.stdin.write(json.dumps({"videos": [str(video)], "output": str(temporary),
                                        "model": "yolov8s", "task": "tf"}) + "\n")
        process.stdin.flush()
        event = json.loads(process.stdout.readline())
        assert event["type"] == "error" and "權重" in event["message"]
        process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
        # A successful request must emit progress while its stdin remains open.
        # This catches the Windows NumPy initialization / pipe-listener deadlock.
        process = subprocess.Popen([sys.executable, str(root / "desktop_bridge.py"), "detect"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", cwd=temporary,
                                   env={**os.environ, **configured})
        process.stdin.write(json.dumps({"videos": [str(video)], "output": str(temporary),
                                        "model": "yolov8s", "task": "tf"}) + "\n")
        process.stdin.flush()
        response = []
        reader = threading.Thread(target=lambda: response.append(process.stdout.readline()), daemon=True)
        reader.start()
        reader.join(timeout=10)
        try:
            assert response and json.loads(response[0])["type"] == "progress", "Bridge stalled before first progress"
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
            process.stderr.close()
    print("PASS: detection rotation, unicode paths, progress, subprocess errors, cancellation, missing weights")


if __name__ == "__main__":
    check()
