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
progress protocol. See docs/adr/0004-gui-yield-image-rule-wiring.md. The driver does build its first three steps
with this module's `segment_command` / `track_command` / `rule_command`, so the flags and their defaults (model path,
sampling interval, detector, conf, imgsz) are written once and a GUI run and a CLI run see the same input.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import clip_artifacts
from child_process import Child

ROOT = Path(__file__).resolve().parent
WORK_DIRNAME = "yield"
MODEL_ENV = "TRAFFIC_CROSSWALK_MODEL"
DEFAULT_MODEL = Path("runtime") / "models" / "mask2former-mapillary"
DEVICE_ENV = "TRAFFIC_DEVICE"
WEIGHTS_ENV = "TRAFFIC_DETECTOR_WEIGHTS"
DEFAULT_WEIGHTS = Path("runtime") / "smoke-assets" / "yolo26x.pt"
# One crosswalk mask per 0.1 s: stride 3 at 25-30 fps (as before), 6 at 60 fps, half the segmentation; on the 7
# labelled clips strict stayed at 14/17 hits with 50 -> 51 candidates, review unchanged, crosswalk geometry 11 -> 12.
SEGMENT_EVERY_SECONDS = 0.1
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


def segment_command(python, video, model, output) -> tuple:
    """segment_crosswalk.py for one video: one mask per SEGMENT_EVERY_SECONDS into <output>/masks.npz."""
    return (str(python), str(ROOT / "scripts" / "segment_crosswalk.py"), "--video", str(video), "--model", str(model),
            "--every-seconds", str(SEGMENT_EVERY_SECONDS), "--output", str(output))


def track_command(python, video, weights, output, device="0", conf=CONF, imgsz=IMGSZ) -> tuple:
    """track_video.py for one video into <output>/raw_tracks.json; weights None leaves track_video's own default."""
    return (str(python), str(ROOT / "scripts" / "track_video.py"), "--video", str(video),
            *(("--weights", str(weights)) if weights else ()),
            "--conf", str(conf), "--imgsz", str(imgsz), "--device", str(device), "--output", str(output))


def rule_command(python, video, work, events, geometry=None, include_waiting=False) -> tuple:
    """yield_image_rule.py on <work>'s masks and tracks; without geometry/include_waiting it runs the rule's defaults."""
    return (str(python), str(ROOT / "yield_image_rule.py"), "--video", str(video),
            "--masks", str(Path(work) / clip_artifacts.MASKS), "--tracks", str(Path(work) / clip_artifacts.TRACKS),
            *(("--geometry", str(geometry)) if geometry else ()), "--output", str(events),
            *(("--include-waiting",) if include_waiting else ()))


def stale_masks(masks, video):
    """Why the masks.npz already at `masks` must not be reused for `video`, or None to reuse it.

    The interval is read back from the mask frame numbers (`clip_artifacts.mask_stride`), so masks segmented before a
    change of SEGMENT_EVERY_SECONDS (or with a hand-picked --stride) are redone instead of silently judged with.
    When either side cannot be read (one mask, or a video cv2 cannot open) the masks are reused."""
    have = clip_artifacts.mask_stride(masks)
    try:
        fps = clip_artifacts.video_info(video).fps
    except Exception:  # noqa: BLE001 - no cv2, or not a video: nothing to compare against
        fps = 0
    if not have or not fps:
        return None
    want = clip_artifacts.stride_for(fps, SEGMENT_EVERY_SECONDS)
    if have == want:
        return None
    return f"既有遮罩每 {have} 格一張，現在應為每 {want} 格（{SEGMENT_EVERY_SECONDS} 秒）"


def workspace(output, video) -> Path:
    """Where one video's intermediates and events go: <output>/yield/<video stem>/."""
    return Path(output) / WORK_DIRNAME / Path(video).stem


def plan(video, work, environ=None) -> list:
    """The three steps for one video, in order. Pure: only the model and weights paths are validated."""
    environ = dict(environ or {})
    video, work = Path(video), Path(work)
    modern = interpreter(environ, "TRAFFIC_MODERN_PYTHON")
    segmentation = segment_command(interpreter(environ, "TRAFFIC_LABELING_PYTHON"), video, crosswalk_model(environ),
                                   work / clip_artifacts.MASKS.parent)
    tracking = track_command(modern, video, detector_weights(environ), work / clip_artifacts.TRACKS.parent,
                             environ.get(DEVICE_ENV, "0"))
    rule = rule_command(modern, video, work, work / "events.jsonl")
    return [
        Step(1, "斑馬線分割", segmentation),
        Step(2, "偵測與追蹤", tracking),
        Step(3, "不禮讓規則", rule),
    ]


def execute(command, label="", cancelled=lambda: False, on_process=lambda process: None) -> str:
    """Run one step from the repository root; returns "ok" or "cancelled", raises on failure.

    The child is run by `child_process.Child` (bounded output queue, cancellation, cleanup); the last lines of its
    output are reported when it fails.
    """
    child = Child(command, cwd=ROOT, cancelled=cancelled, on_process=on_process)
    for _ in child.lines():
        pass
    if child.stopped:
        return "cancelled"
    if child.code:
        command = child.command
        name = label or Path(command[1] if len(command) > 1 else command[0]).name
        raise RuntimeError(f"{name} 失敗（結束碼 {child.code}）：\n" + "\n".join(child.tail))
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
