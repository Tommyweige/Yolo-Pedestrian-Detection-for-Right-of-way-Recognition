"""The image-space yield pipeline for one video, as the GUI runs it.

The three commands are the documented ones (docs/yield-image-rule.md):
`scripts/segment_crosswalk.py` -> `scripts/track_video.py` -> `yield_image_rule.py`.
Segmentation needs the labeling environment (transformers) and tracking the modern one (Ultralytics 8.4), neither of
which the GUI interpreter has, so every step is a subprocess; the interpreter comes from TRAFFIC_LABELING_PYTHON /
TRAFFIC_MODERN_PYTHON, else the current one.

The rule step runs the rule's own command line with no construction flag, so the GUI reports whatever construction the
rule defaults to (the candidate list: every vehicle/pedestrian contact, repeats of one pair merged).

`scripts/run_pipeline.py` is deliberately not reused: it is a batch driver over a folder of videos, with its own
run directory and an overlay-video step, which does not fit a GUI that owns the file list, the output folder and the
progress protocol. See docs/adr/0004-gui-yield-image-rule-wiring.md.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK_DIRNAME = "yield"
MODEL_ENV = "TRAFFIC_CROSSWALK_MODEL"
DEFAULT_MODEL = Path("runtime") / "models" / "mask2former-mapillary"
DEVICE_ENV = "TRAFFIC_DEVICE"
WEIGHTS_ENV = "TRAFFIC_DETECTOR_WEIGHTS"
DEFAULT_WEIGHTS = Path("runtime") / "smoke-assets" / "yolo26x.pt"  # scripts/run_pipeline.py's default too
# Same values as scripts/run_pipeline.py, so a GUI run and a CLI run see the same input.
SEGMENT_EVERY_SECONDS = 0.1  # one crosswalk mask per 0.1 s: stride 3 at 25-30 fps, 6 at 60 fps
CONF = 0.15
IMGSZ = 1280


@dataclass(frozen=True)
class Step:
    number: int
    label: str
    command: tuple


def interpreter(environ, variable: str) -> str:
    configured = (environ or {}).get(variable)
    return str(configured) if configured else sys.executable


def crosswalk_model(environ) -> Path:
    """The Mask2Former folder; TRAFFIC_CROSSWALK_MODEL overrides the default runtime path."""
    configured = (environ or {}).get(MODEL_ENV)
    model = Path(configured).expanduser() if configured else ROOT / DEFAULT_MODEL
    if not model.is_absolute():
        model = ROOT / model
    if not model.is_dir():
        raise FileNotFoundError(
            "找不到斑馬線分割模型："
            f"{model}\n"
            "請將 mask2former-mapillary（config.json + model.safetensors）放到該資料夾，"
            f"或設定環境變數 {MODEL_ENV} 指向模型資料夾。"
        )
    return model


def detector_weights(environ, root=ROOT) -> Path:
    """The YOLO weights; TRAFFIC_DETECTOR_WEIGHTS overrides yolo26x (strict 14/17 on the labels, yolo26n 10/17)."""
    configured = (environ or {}).get(WEIGHTS_ENV)
    weights = Path(configured).expanduser() if configured else root / DEFAULT_WEIGHTS
    if not weights.is_absolute():
        weights = root / weights
    if not weights.is_file():
        raise FileNotFoundError(
            "找不到偵測模型權重："
            f"{weights}\n"
            f"請將 yolo26x.pt 放到 {root / DEFAULT_WEIGHTS.parent}，"
            f"或設定環境變數 {WEIGHTS_ENV} 指向權重檔。"
        )
    return weights


def workspace(output, video) -> Path:
    """Where one video's intermediates and events go: <output>/yield/<video stem>/."""
    return Path(output) / WORK_DIRNAME / Path(video).stem


def plan(video, work, environ=None) -> list:
    """The three steps for one video, in order. Pure: only the model and weights paths are validated."""
    environ = dict(environ or {})
    video, work = Path(video), Path(work)
    modern = interpreter(environ, "TRAFFIC_MODERN_PYTHON")
    segmentation = (
        interpreter(environ, "TRAFFIC_LABELING_PYTHON"),
        str(ROOT / "scripts" / "segment_crosswalk.py"),
        "--video", str(video),
        "--model", str(crosswalk_model(environ)),
        "--every-seconds", str(SEGMENT_EVERY_SECONDS),
        "--output", str(work / "seg"),
    )
    tracking = (
        modern,
        str(ROOT / "scripts" / "track_video.py"),
        "--video", str(video),
        "--weights", str(detector_weights(environ)),
        "--conf", str(CONF),
        "--imgsz", str(IMGSZ),
        "--device", environ.get(DEVICE_ENV, "0"),
        "--output", str(work / "tracks"),
    )
    rule = (
        modern,
        str(ROOT / "yield_image_rule.py"),
        "--video", str(video),
        "--masks", str(work / "seg" / "masks.npz"),
        "--tracks", str(work / "tracks" / "raw_tracks.json"),
        "--output", str(work / "events.jsonl"),
    )
    return [
        Step(1, "斑馬線分割", segmentation),
        Step(2, "偵測與追蹤", tracking),
        Step(3, "不禮讓規則", rule),
    ]


def read_events(path) -> list:
    """The rule's .jsonl output: one event per line."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def execute(command, label="", cancelled=lambda: False, on_process=lambda process: None) -> str:
    """Run one step from the repository root; returns "ok" or "cancelled", raises on failure.

    The child's output is kept in a bounded queue by a reader thread so a chatty step cannot block on a full pipe,
    and the last lines are reported when it fails.
    """
    command = [str(part) for part in command]
    process = subprocess.Popen(
        command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    on_process(process)
    lines = queue.Queue(maxsize=512)

    def read_lines():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_lines, daemon=True)
    reader.start()
    tail, stopped = [], False
    try:
        while True:
            if cancelled() and process.poll() is None:
                stopped = True
                process.terminate()
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    break
                continue
            if line is None:
                break
            tail = (tail + [line.strip()])[-12:]
        code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        while reader.is_alive():  # drain so the reader can close its pipe
            try:
                lines.get(timeout=0.1)
            except queue.Empty:
                pass
        process.stdout.close()
        on_process(None)
    if stopped:
        return "cancelled"
    if code:
        name = label or Path(command[1] if len(command) > 1 else command[0]).name
        raise RuntimeError(f"{name} 失敗（結束碼 {code}）：\n" + "\n".join(tail))
    return "ok"


def write_run_meta(work, video, events, timings, steps, video_info=None) -> dict:
    """run.json next to the events: what ran, how long each step took, and what it found."""
    info = dict(video_info or {})
    meta = {
        "video": Path(video).name,
        "video_path": str(Path(video).resolve()),
        "frames": info.get("frames"),
        "fps": info.get("fps"),
        "size": list(info.get("size") or []),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(float(sum(timings.values())), 1),
        "step_seconds": dict(timings),
        "events": len(events),
        "event_times_s": [event.get("time_s") for event in events],
        "commands": [list(step.command) for step in steps],
    }
    (Path(work) / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta
