# 跨平台批次流程：斑馬線分割 + 追蹤（同時跑）-> 讓行人規則 -> 疊圖 + 證據片段（同時跑），最後列出整批候選清單 datasets/<run>/candidates.csv
"""批次流程本體，Windows 與 Linux 共用；run_pipeline.ps1 只是幫它設好兩個 Python 環境的外殼。"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}
STEPS = 5
PAD_SECONDS = 2  # 證據片段在規則的證據窗前後各多留幾秒，方便人工判斷前因後果
# 互不依賴的步驟同時跑：分割與追蹤一起佔 GPU（各自的輸出與分開跑時逐位元相同，20 秒影片 109 -> 78 秒，
# 顯存峰值約 4 GB）；疊圖與證據片段都只讀事件檔。--serial 改回一次一步（顯存不夠時用）。
TOGETHER = ((1, 2), (3,), (4, 5))
# 跨影片：GPU 跑下一支的分割與追蹤時，CPU 同時跑上一支的規則、疊圖與證據片段（後段一次只跑一支）。
GPU_STEPS = (1, 2)
# 每 0.1 秒一張遮罩：25-30 fps 影片是每 3 格（和以前一樣），60 fps 影片從每 3 格改成每 6 格，分割量減半；
# 7 支標註影片 strict 14/17 命中不變、候選 50 -> 51，review 不變，crosswalk 幾何 11 -> 12 命中
DEFAULT_WEIGHTS = "runtime/smoke-assets/yolo26x.pt"  # same default as scripts/run_pipeline.ps1
SEGMENT_EVERY_SECONDS = "0.1"


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
    evidence = root / "datasets" / run / "evidence" / stem
    mode = getattr(args, "mode", "strict")

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
                    "--every-seconds",
                    SEGMENT_EVERY_SECONDS,
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

    rule_label = "含路緣等待者" if mode == "review" else "只報上了斑馬線的行人"
    steps.append(
        (
            3,
            f"讓行人規則（{getattr(args, 'geometry', 'image')} 幾何，{rule_label}）",
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
            ]
            + (["--include-waiting"] if mode == "review" else []),
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
    steps.append(
        (
            5,
            "證據片段與截圖",
            [
                modern,
                str(root / "scripts" / "export_evidence.py"),
                "--video",
                str(video),
                "--events",
                str(events),
                "--output",
                str(evidence),
                "--pad-seconds",
                str(PAD_SECONDS),
            ],
        )
    )
    return steps


def candidate_list_command(args, root: Path, environ) -> list[str]:
    """整批影片跑完後，把所有事件彙整成一張候選清單。"""
    run_dir = Path(root) / "datasets" / getattr(args, "run", "eval-run2")
    return [_python(environ, "TRAFFIC_MODERN_PYTHON"), str(Path(root) / "scripts" / "list_candidates.py"), "--run-dir", str(run_dir)]


def speed_text(video_seconds: float | None, elapsed: float) -> str:
    if not video_seconds:
        return f"處理 {elapsed:.0f} 秒"
    return f"影片 {video_seconds:.0f} 秒，處理 {elapsed:.0f} 秒（即時的 {elapsed / video_seconds:.1f} 倍）"


def _video_seconds(video: Path) -> float | None:
    try:
        import cv2
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(video))
    fps, frames = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    return frames / fps if fps and frames else None


def step_groups(steps, serial: bool):
    """把 _plan 的步驟依 TOGETHER 分組；serial 時每步自成一組。"""
    if serial:
        return [[step] for step in steps]
    groups = []
    for numbers in TOGETHER:
        group = [step for step in steps if step[0] in numbers]
        if group:
            groups.append(group)
    return groups


def run_together(commands, cwd, on_done=None, cancel=None):
    """同時啟動所有指令並等它們結束；有一個失敗就停掉其餘的。回傳 None 或 (失敗的指令, 結束碼)。

    cancel（threading.Event）被設起時也停掉全部，回傳 (None, None)：失敗發生在別處，已經回報過了。
    """
    started = time.perf_counter()
    pending = [(command, subprocess.Popen(command, cwd=cwd)) for command in commands]
    failed = None
    while pending:
        if failed is None and cancel is not None and cancel.is_set():
            failed = (None, None)
            for _, other in pending:
                other.terminate()
        for item in list(pending):
            command, process = item
            code = process.poll()
            if code is None:
                continue
            pending.remove(item)
            if failed is None and code != 0:
                failed = (command, code)
                for _, other in pending:
                    other.terminate()
            elif failed is None and on_done:
                on_done(command, time.perf_counter() - started)
        if pending:
            time.sleep(0.5)
    return failed


def _python(environ, variable: str) -> str:
    return (environ or {}).get(variable) or sys.executable


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="批次跑斑馬線分割、追蹤、讓行人規則與疊圖。")
    parser.add_argument("--folder", default="datasets/來源影片")
    parser.add_argument("--run", default="eval-run2")
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="偵測模型（預設 yolo26x：同一批遮罩下 strict 14/17，yolo26n 只有 10/17；追蹤較慢，但整批多半在等分割）",
    )
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
    parser.add_argument(
        "--mode",
        choices=("review", "strict"),
        default="strict",
        help="strict（預設：只報上了斑馬線的行人）或 review（路緣等待者也列為候選；多找到很少，候選多很多）",
    )
    parser.add_argument("--only", nargs="*", default=[], help="只處理檔名含這些字串的影片")
    parser.add_argument("--serial", action="store_true", help="一次只跑一步（顯存不夠同時放分割與追蹤時用）")
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
    if args.only:
        videos = [path for path in videos if any(text in path.name for text in args.only)]
    if not videos:
        print(f"錯誤：資料夾內沒有影片：{folder}", file=sys.stderr)
        return 2

    log = None if args.dry_run else root / "datasets" / f"{args.run}.log"
    total = len(videos)
    run_dir = root / "datasets" / args.run
    cancel = threading.Event()  # 任何一步失敗就設起，停掉另一條線上正在跑的步驟
    lengths = []  # 每支影片的長度（秒），算整批的速度
    batch_start = time.perf_counter()

    def run_groups(groups, tag, names):
        """依序跑幾組步驟；成功回傳 True。失敗時回報一次並設起 cancel。"""
        for group in groups:
            commands = [command for _, _, command in group if command is not None]
            together = "（同時跑）" if len(commands) > 1 else ""
            for step, description, _ in group:
                _emit(f"  {tag} [{step}/{STEPS}] {description}{together}", log)
            if not commands:
                continue
            failed = run_together(commands, root, lambda command, seconds: _emit(
                f"        {tag} {names[tuple(command)]} 完成，{seconds:.0f} 秒", log), cancel)
            if failed:
                if failed[0] is not None:
                    _emit(f"失敗：{shlex.join(failed[0])}（結束碼 {failed[1]}）", log)
                cancel.set()
                return False
        return True

    def finish(groups, video, tag, names, video_start):
        if run_groups(groups, tag, names):
            seconds = _video_seconds(video)
            lengths.append(seconds)
            _emit(f"  {tag} {video.name}：{speed_text(seconds, time.perf_counter() - video_start)}", log)

    tail = None  # 上一支影片的後段（在另一個執行緒跑）
    for number, video in enumerate(videos, start=1):
        stem = video.stem
        tag = f"({number}/{total})"
        if not args.dry_run:
            for directory in (
                run_dir / stem / "seg",
                run_dir / stem / "tracks",
                run_dir / "events",
                run_dir / "overlay",
                run_dir / "evidence" / stem,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            _emit(f"=== [{number}/{total}] {video.name}  {_now()}", log)
        video_start = time.perf_counter()

        steps = _plan(video, args, root, os.environ)
        if args.dry_run:
            for _, _, command in steps:
                if command is not None:
                    print(shlex.join(command))
            continue
        names = {tuple(command): f"[{step}/{STEPS}]" for step, _, command in steps if command is not None}
        groups = step_groups(steps, args.serial)
        gpu = [group for group in groups if any(step in GPU_STEPS for step, _, _ in group)]
        rest = [group for group in groups if group not in gpu]
        if not run_groups(gpu, tag, names):
            break
        if args.serial:  # 一次只跑一步：後段也不和下一支重疊
            finish(rest, video, tag, names, video_start)
            if cancel.is_set():
                break
            continue
        if tail is not None:
            tail.join()  # 後段一次只跑一支，CPU 不會被多支影片搶
        if cancel.is_set():
            break
        tail = threading.Thread(target=finish, args=(rest, video, tag, names, video_start))
        tail.start()
    if tail is not None:
        tail.join()
    if cancel.is_set():
        return 1
    if not args.dry_run:
        known = [seconds for seconds in lengths if seconds]
        _emit(f"整批：{speed_text(sum(known) if len(known) == len(lengths) else None, time.perf_counter() - batch_start)}", log)

    command = candidate_list_command(args, root, os.environ)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    failed = run_together([command], root)
    if failed:
        _emit(f"失敗：{shlex.join(command)}（結束碼 {failed[1]}）", log)
        return 1
    _emit(
        f"全部完成 {_now()}。候選清單:datasets/{args.run}/candidates.csv  證據:datasets/{args.run}/evidence"
        f"  事件:datasets/{args.run}/events  疊圖:datasets/{args.run}/overlay",
        log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
