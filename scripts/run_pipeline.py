# 跨平台批次流程：斑馬線分割 -> 追蹤 -> 讓行人規則 -> 疊圖。每支影片一個資料夾 datasets/<run>/<stem>/
"""以 Python 重寫 run_pipeline.ps1 的流程，Windows 與 Linux 共用。"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def build_commands(video: Path, args, root: Path, environ) -> list[list[str]]:
    """回傳這支影片要執行的指令；純函式，只依檔案是否存在決定步驟。"""
    return [command for _, _, command in _plan(video, args, root, environ) if command is not None]


def _plan(video: Path, args, root: Path, environ):
    """回傳 [(步驟編號, 說明, 指令或 None)]，讓 main 印進度時不必重複判斷。"""
    video = Path(video)
    root = Path(root)
    run = getattr(args, "run", "eval-run2")
    stem = video.stem
    work = root / "datasets" / run / stem
    masks = work / "seg" / "masks.npz"
    tracks = work / "tracks" / "raw_tracks.json"
    events = root / "datasets" / run / "events" / f"{stem}.jsonl"
    overlay = root / "datasets" / run / "overlay" / f"{stem}.mp4"

    modern = _python(environ, "TRAFFIC_MODERN_PYTHON")
    labeling = _python(environ, "TRAFFIC_LABELING_PYTHON")

    if masks.exists():
        steps = [(1, "沿用既有斑馬線遮罩", None)]
    else:
        steps = [
            (
                1,
                "斑馬線分割 (GPU)",
                [
                    labeling,
                    str(root / "scripts" / "segment_crosswalk.py"),
                    "--video",
                    str(video),
                    "--model",
                    "runtime/models/mask2former-mapillary",
                    "--stride",
                    "3",
                    "--output",
                    str(work / "seg"),
                ],
            )
        ]

    if getattr(args, "reuse_tracks", False) and tracks.exists():
        steps.append(
            (
                2,
                "沿用既有追蹤結果(只重做後照鏡過濾)",
                [
                    modern,
                    str(root / "scripts" / "track_video.py"),
                    "--video",
                    str(video),
                    "--refilter",
                    str(tracks),
                ],
            )
        )
    else:
        track = [
            modern,
            str(root / "scripts" / "track_video.py"),
            "--video",
            str(video),
        ]
        if getattr(args, "weights", None):
            track += ["--weights", str(args.weights)]
        track += [
            "--conf",
            str(getattr(args, "conf", 0.15)),
            "--imgsz",
            str(getattr(args, "imgsz", 1280)),
            "--device",
            str(getattr(args, "device", "0")),
            "--output",
            str(work / "tracks"),
        ]
        steps.append((2, "偵測與追蹤", track))

    steps.append(
        (
            3,
            f"讓行人規則（{getattr(args, 'geometry', 'image')} 幾何）",
            [
                modern,
                str(root / "yield_image_rule.py"),
                "--video",
                str(video),
                "--masks",
                str(masks),
                "--tracks",
                str(tracks),
                "--geometry",
                str(getattr(args, "geometry", "image")),
                "--output",
                str(events),
            ],
        )
    )
    steps.append(
        (
            4,
            "疊圖影片",
            [
                modern,
                str(root / "scripts" / "render_overlay.py"),
                "--video",
                str(video),
                "--masks",
                str(masks),
                "--tracks",
                str(tracks),
                "--events",
                str(events),
                "--labels",
                str(root / "labels" / f"{stem}.jsonl"),
                "--output",
                str(overlay),
            ],
        )
    )
    return steps


def _python(environ, variable: str) -> str:
    return (environ or {}).get(variable) or sys.executable


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="批次跑斑馬線分割、追蹤、讓行人規則與疊圖。")
    parser.add_argument("--folder", default="datasets/來源影片")
    parser.add_argument("--run", default="eval-run2")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--reuse-tracks", action="store_true")
    parser.add_argument(
        "--geometry",
        choices=("image", "crosswalk"),
        default="image",
        help="讓行人規則的判斷方式：image（像素條帶，預設）或 crosswalk（斑馬線座標）",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _emit(message: str, log: Path | None) -> None:
    print(message)
    if log is None:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main(argv=None) -> int:
    args = _parse_args(argv)
    root = ROOT
    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = root / folder

    if not folder.is_dir():
        print(f"錯誤：資料夾不存在：{folder}", file=sys.stderr)
        return 2
    videos = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.name,
    )
    if not videos:
        print(f"錯誤：資料夾內沒有影片：{folder}", file=sys.stderr)
        return 2

    log = None if args.dry_run else root / "datasets" / f"{args.run}.log"
    total = len(videos)
    run_dir = root / "datasets" / args.run
    for number, video in enumerate(videos, start=1):
        stem = video.stem
        if not args.dry_run:
            for directory in (
                run_dir / stem / "seg",
                run_dir / stem / "tracks",
                run_dir / "events",
                run_dir / "overlay",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            _emit(f"=== [{number}/{total}] {video.name}  {_now()}", log)

        for step, description, command in _plan(video, args, root, os.environ):
            if args.dry_run:
                if command is not None:
                    print(shlex.join(command))
                continue
            _emit(f"  [{step}/4] {description}", log)
            if command is None:
                continue
            result = subprocess.run(command, check=False, cwd=root)
            if result.returncode != 0:
                _emit(f"失敗：{shlex.join(command)}（結束碼 {result.returncode}）", log)
                return 1

    if args.dry_run:
        return 0
    _emit(f"全部完成 {_now()}。事件:datasets/{args.run}/events  疊圖:datasets/{args.run}/overlay", log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
