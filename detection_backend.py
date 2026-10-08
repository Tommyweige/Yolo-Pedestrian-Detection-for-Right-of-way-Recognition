"""Shared detection paths and subprocess runner; no GUI imports."""
import os
import re
import subprocess
import sys
import tempfile
import math
import queue
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_PROJECT_ROOT = PROJECT_ROOT / "YOLOv8_DeepSORT_Object_Tracking"
DETECT_DIR = YOLO_PROJECT_ROOT / "ultralytics" / "yolo" / "v8" / "detect"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
PREVIEW_IMAGE_PATH = RUNTIME_DIR / "preview.jpg"

MODEL_NAMES = ("yolov8s", "yolov8l", "yolov8x6")
TASKS = {
    "tf": {"script": "predict_tf.py", "weight_suffix": "_tf.pt"},
    "zebra": {"script": "predict_zebra.py", "weight_suffix": "_zebra.pt"},
}

def _configured_path(value, base=PROJECT_ROOT):
    """Return an absolute path for a config value.

    Relative values in environment variables are interpreted relative to the
    repository root, not relative to whichever directory launched the GUI.
    """

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _weights_dir():
    return _configured_path(os.environ.get("YOLO_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR))


def resolve_model_path(model_name, task):
    """Resolve and validate the YOLO weights required by a detection task."""

    task_config = TASKS[task]
    env_name = f"YOLO_{task.upper()}_MODEL"
    configured_model = os.environ.get(env_name)
    if configured_model:
        model_path = _configured_path(configured_model)
    else:
        filename = f"{model_name}{task_config['weight_suffix']}"
        model_path = _weights_dir() / filename

    if not model_path.is_file():
        expected_name = f"{model_name}{task_config['weight_suffix']}"
        raise FileNotFoundError(
            "找不到 YOLO 模型權重："
            f"{model_path}\n"
            f"請將 {expected_name} 放到 {_weights_dir()}，"
            f"或設定環境變數 {env_name} 指向權重檔案。"
        )
    return model_path


def resolve_reid_checkpoint():
    """Resolve the DeepSORT appearance-model checkpoint."""

    candidates = []
    configured_checkpoint = os.environ.get("DEEPSORT_REID_CKPT")
    if configured_checkpoint:
        candidates.append(_configured_path(configured_checkpoint))

    candidates.extend(
        (
            _weights_dir() / "ckpt.t7",
            DETECT_DIR
            / "deep_sort_pytorch"
            / "deep_sort"
            / "deep"
            / "checkpoint"
            / "ckpt.t7",
        )
    )

    for checkpoint in candidates:
        if checkpoint.is_file():
            return checkpoint

    locations = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "找不到 DeepSORT 權重 ckpt.t7。請將檔案放到 weights/ckpt.t7、"
        "原本的 DeepSORT checkpoint 位置，或設定 DEEPSORT_REID_CKPT。"
        f"\n已檢查：\n{locations}"
    )


def _hydra_path(path):
    """Use a platform-independent path representation in Hydra overrides."""

    return path.resolve().as_posix()


def _hydra_value(value):
    """Quote a Hydra value so paths containing spaces remain one override."""

    if isinstance(value, Path):
        value = _hydra_path(value)
    value = str(value).replace('"', '\\"')
    return f'"{value}"'


def _subprocess_environment(reid_checkpoint):
    """Build the environment used by the local YOLO/DeepSORT scripts."""

    environment = os.environ.copy()
    environment.update(
        {
            "QT_DEBUG_PLUGINS": "1",
            "KMP_DUPLICATE_LIB_OK": "1",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "DEEPSORT_REID_CKPT": str(reid_checkpoint),
        }
    )

    # Use the pinned ultralytics==8.0.3 dependency for the inference core.
    # The vendored copy is incomplete (missing detect/predict.py and callbacks).
    # Keep our traffic scripts and DeepSORT imports local.
    python_path = [str(DETECT_DIR)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    return environment


def _video_width(video_path):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"無法開啟影片：{video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        capture.release()

    if width <= 0:
        raise ValueError(f"無法讀取影片寬度：{video_path}")
    return width


def run_detection(videos, output, model, task, angle=0, cancelled=lambda: False,
                  on_process=lambda process: None):
    """Yield progress events while running the existing prediction scripts."""
    if task not in TASKS or model not in MODEL_NAMES:
        raise ValueError("未知的偵測模式或模型。")
    if not math.isfinite(angle) or not -45 <= angle <= 45:
        raise ValueError("旋轉角度必須介於 -45 與 45 度。")
    if not videos or not output:
        raise ValueError("請先選擇影片與輸出資料夾。")
    output_dir = _configured_path(output)
    if not output_dir.is_dir():
        raise ValueError(f"輸出資料夾不存在：{output_dir}")
    model_path = resolve_model_path(model, task)
    checkpoint = resolve_reid_checkpoint()
    script = DETECT_DIR / TASKS[task]["script"]
    if not script.is_file():
        raise FileNotFoundError(f"找不到偵測腳本：{script}")
    environment = _subprocess_environment(checkpoint)
    for index, video in enumerate(videos):
        if cancelled():
            yield {"type": "cancelled"}
            return
        video = _configured_path(video)
        if not video.is_file():
            raise FileNotFoundError(f"找不到影片：{video}")
        width = _video_width(video)
        yield {"type": "progress", "index": index, "current": 0, "total": 1,
               "message": f"準備影片：{video.name}"}
        with tempfile.TemporaryDirectory(prefix="traffic_yolo_rotate_") as temporary:
            source = video
            if angle:
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(str(video))
                rotated = clip.rotate(angle)
                source = Path(temporary) / "rotated.mp4"
                try:
                    rotated.write_videofile(str(source), logger=None)
                finally:
                    rotated.close()
                    clip.close()
            if cancelled():
                yield {"type": "cancelled"}
                return
            command = [sys.executable, str(script), f"model={_hydra_value(model_path)}",
                       f"source={_hydra_value(source)}", f"project={_hydra_value(output_dir)}",
                       f"name={_hydra_value(video.name)}", f"imgsz={width}",
                       "conf=0.7", "iou=0.3", "augment=True", "half=True"]
            process = subprocess.Popen(command, cwd=DETECT_DIR, env=environment,
                                       stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            on_process(process)
            lines = queue.Queue(maxsize=128)

            def read_lines():
                for line in process.stdout:
                    lines.put(line)
                lines.put(None)

            reader = threading.Thread(target=read_lines, daemon=True)
            reader.start()
            tail = []
            try:
                while True:
                    if cancelled() and process.poll() is None:
                        process.terminate()
                    try:
                        line = lines.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if line is None:
                        break
                    tail = (tail + [line.strip()])[-12:]
                    match = re.search(r"\((\d+)/(\d+)\)", line)
                    if match:
                        current, total = map(int, match.groups())
                        yield {"type": "progress", "index": index, "current": current,
                               "total": total, "message": line.strip()}
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                # Drain the bounded queue so the reader can close its pipe.
                while reader.is_alive():
                    try:
                        lines.get(timeout=0.1)
                    except queue.Empty:
                        pass
                process.stdout.close()
                on_process(None)
            if cancelled():
                yield {"type": "cancelled"}
                return
            if code:
                raise RuntimeError(f"偵測失敗（{code}）：{video.name}\n" + "\n".join(tail))
            yield {"type": "progress", "index": index + 1, "current": 1, "total": 1,
                   "message": f"已完成：{video.name}"}
    yield {"type": "done"}
