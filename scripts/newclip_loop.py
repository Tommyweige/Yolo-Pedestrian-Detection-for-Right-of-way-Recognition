"""One command from a folder of new videos to a review page: masks -> tracks -> rule events -> page.

The point is the out-of-sample loop: every accuracy number so far comes from the seven clips the rule's thresholds
were tuned on. A clip that was never part of that set only has to be dropped into a folder to be scored, but doing it
by hand meant chaining `segment_crosswalk.py`, `track_video.py`, `yield_image_rule.py` and then a separate page
generator. This runs all of it and writes a manifest (`run.json`) that `scripts/newclip_score.py` reads.

The rule runs with its own defaults — the candidate-list construction, pair de-duplication, foot depth 0.03 — no
`--one-event-per-vehicle`, no parameter overrides. Changing any of that here would make the new clips incomparable.

Layout of a run folder:
  <out>/run.json                     manifest: role, rule defaults, per-clip paths, fps, event counts
  <out>/run.log                      one line per step, with timings
  <out>/<stem>/seg/masks.npz         crosswalk masks (labeling env, GPU)
  <out>/<stem>/tracks/raw_tracks.json  detections + ByteTrack (modern env, GPU)
  <out>/<stem>/events.jsonl          the rule's events
  <out>/review/index.html            the page to judge

Steps whose output already exists are skipped, so an interrupted run resumes. `--reuse-from` reads masks and tracks
from an earlier run (e.g. the seven development clips already processed in datasets/eval-run2) instead of redoing GPU
work; those clips then only run the rule.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yield_pipeline  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}
SCHEMA = "yolo-ped/newclip-run@1"
# The evaluation tolerance the development numbers were made with (docs/yield-image-rule.md, 1.25 s = the rule's own
# contact window). The scorer defaults to this value so both sets are scored alike.
TOLERANCE_S = 1.25


def log(message, logfile=None):
    print(message, flush=True)
    if logfile is not None:
        with open(logfile, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def videos_in(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"資料夾不存在：{folder}")
    found = sorted(path for path in folder.iterdir()
                   if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if not found:
        raise SystemExit(f"資料夾內沒有影片（{'、'.join(sorted(VIDEO_EXTENSIONS))}）：{folder}")
    return found


def video_info(video):
    """fps and pixel size of a clip, read once here so the scorer never has to open the video."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"打不開影片：{video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return {"fps": fps, "size": list(size), "frames": frames}


def clip_paths(out, stem):
    work = Path(out) / stem
    return {"work": work,
            "masks": work / "seg" / "masks.npz",
            "tracks": work / "tracks" / "raw_tracks.json",
            "events": work / "events.jsonl"}


def reuse_paths(reuse, stem):
    if reuse is None:
        return None, None
    root = Path(reuse) / stem
    return root / "seg" / "masks.npz", root / "tracks" / "raw_tracks.json"


def plan_clip(video, out, environ=None, reuse=None, weights=None, conf=None, imgsz=None, device=None,
              force=False):
    """[(number, description, command or None)] for one clip; None means 'nothing to run'.

    Masks and tracks are only segmented/tracked when no copy exists in the run folder (or in `--reuse-from`); the rule
    always runs, because its output is the thing being judged and it takes seconds.
    """
    environ = dict(environ or {})
    video = Path(video)
    paths = clip_paths(out, video.stem)
    reused_masks, reused_tracks = reuse_paths(reuse, video.stem)
    masks = paths["masks"] if paths["masks"].exists() or reused_masks is None else reused_masks
    tracks = paths["tracks"] if paths["tracks"].exists() or reused_tracks is None else reused_tracks
    masks_exist = masks.exists() and not force
    tracks_exist = tracks.exists() and not force
    work = paths["work"]
    work.mkdir(parents=True, exist_ok=True)
    (work / "seg").mkdir(parents=True, exist_ok=True)
    (work / "tracks").mkdir(parents=True, exist_ok=True)
    modern = yield_pipeline.interpreter(environ, "TRAFFIC_MODERN_PYTHON")
    steps = []

    if masks_exist:
        steps.append((1, f"沿用斑馬線遮罩 {masks}", None))
    else:
        steps.append((1, "斑馬線分割 (GPU)", (
            yield_pipeline.interpreter(environ, "TRAFFIC_LABELING_PYTHON"),
            str(ROOT / "scripts" / "segment_crosswalk.py"),
            "--video", str(video),
            "--model", str(yield_pipeline.crosswalk_model(environ)),
            "--stride", str(yield_pipeline.STRIDE),
            "--output", str(paths["work"] / "seg"),
        )))
        masks = paths["masks"]

    if tracks_exist:
        steps.append((2, f"沿用追蹤結果 {tracks}", None))
    else:
        tracking = [modern, str(ROOT / "scripts" / "track_video.py"),
                    "--video", str(video),
                    "--conf", str(conf if conf is not None else yield_pipeline.CONF),
                    "--imgsz", str(imgsz if imgsz is not None else yield_pipeline.IMGSZ),
                    "--device", str(device if device is not None else environ.get(yield_pipeline.DEVICE_ENV, "0")),
                    "--output", str(paths["work"] / "tracks")]
        if weights:
            tracking += ["--weights", str(weights)]
        steps.append((2, "偵測與追蹤 (GPU)", tuple(tracking)))
        tracks = paths["tracks"]

    steps.append((3, "不禮讓規則（預設參數）", (
        modern,
        str(ROOT / "yield_image_rule.py"),
        "--video", str(video),
        "--masks", str(masks),
        "--tracks", str(tracks),
        "--output", str(paths["events"]),
    )))
    return steps, masks, tracks


def git_state():
    def run(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        return result.stdout.strip() if result.returncode == 0 else ""
    return run("rev-parse", "--abbrev-ref", "HEAD") or "?", run("rev-parse", "--short", "HEAD") or "?"


def run_clip(video, out, logfile, environ=None, reuse=None, weights=None, conf=None, imgsz=None, device=None,
             force=False):
    """Run one clip's steps; returns its manifest entry."""
    environ = dict(os.environ) if environ is None else environ
    steps, masks, tracks = plan_clip(video, out, environ, reuse, weights, conf, imgsz, device, force)
    info = video_info(video)
    timings = {}
    for number, description, command in steps:
        if command is None:
            log(f"  [{number}/3] {description}", logfile)
            continue
        log(f"  [{number}/3] {description}", logfile)
        started = time.time()
        result = subprocess.run(list(command), cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        timings[str(number)] = round(time.time() - started, 1)
        if result.returncode != 0:
            tail = (result.stdout + result.stderr).strip().splitlines()[-12:]
            log("  失敗：" + " ".join(command), logfile)
            log("\n".join("    " + line for line in tail), logfile)
            raise SystemExit(f"{description} 失敗（結束碼 {result.returncode}）")
        log(f"  完成 {timings[str(number)]} s", logfile)
    paths = clip_paths(out, video.stem)
    events = [json.loads(line) for line in paths["events"].read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "stem": video.stem,
        "video_name": video.name,
        "video_path": str(video.resolve()),
        "role": None,                     # filled from the run's role
        "fps": round(info["fps"], 4),
        "size": info["size"],
        "frames": info["frames"],
        "masks": str(masks),
        "tracks": str(tracks),
        "events_jsonl": str(paths["events"]),
        "events": len(events),
        "event_times_s": [event["time_s"] for event in events],
        "seconds": round(sum(timings.values()), 1),
    }


def build_manifest(clips, role, videos, params, dev_labels=None):
    for clip in clips:
        clip["role"] = role
        clip["dev_overlap"] = bool(dev_labels) and (Path(dev_labels) / f"{clip['stem']}.jsonl").exists()
    branch, commit = git_state()
    return {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "role": role,
        "branch": branch,
        "commit": commit,
        "videos": str(Path(videos).resolve()),
        "tolerance_s": TOLERANCE_S,
        "params": params,
        "rule_cli": "yield_image_rule.py（無建構旗標、無參數覆寫）",
        "clips": clips,
    }


def rule_defaults():
    """The rule's thresholds as the run recorded them. fps and src_size are per clip, not defaults: they live on each
    clip entry, so the run cannot claim 60 fps for a 30 fps clip."""
    import yield_image_rule as rule
    params = dict(rule.Params().__dict__)
    params.pop("fps")
    params.pop("src_size")
    params["rider_kinds"] = list(params["rider_kinds"])
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="folder of the new clips")
    parser.add_argument("--out", required=True, help="run folder (manifest, per-clip outputs, review page)")
    parser.add_argument("--role", choices=("validation", "dev"), default="validation",
                        help="validation (default): clips that were never tuned on; dev: the seven labelled clips")
    parser.add_argument("--reuse-from", help="earlier run folder to read <stem>/seg and <stem>/tracks from")
    parser.add_argument("--dev-labels", help="folder of the development clips' <stem>.jsonl labels; a clip found "
                                             "there is flagged dev_overlap, i.e. not out-of-sample")
    parser.add_argument("--weights", help="detector .pt for track_video.py (default: the repo's own resolution)")
    parser.add_argument("--conf", type=float, help=f"detector confidence (default {yield_pipeline.CONF})")
    parser.add_argument("--imgsz", type=int, help=f"detector inference size (default {yield_pipeline.IMGSZ})")
    parser.add_argument("--device", help="CUDA device (default 0)")
    parser.add_argument("--force", action="store_true", help="redo masks and tracks even when they exist")
    parser.add_argument("--no-review", action="store_true", help="stop after the events, no page")
    args = parser.parse_args(argv)

    videos = videos_in(args.videos)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "run.log"
    log(f"=== new-clip run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"· {len(videos)} 支影片 · 角色 {args.role}", logfile)
    log(f"    影片 {Path(args.videos).resolve()}", logfile)

    clips = []
    for index, video in enumerate(videos, 1):
        log(f"[{index}/{len(videos)}] {video.name}", logfile)
        clips.append(run_clip(video, out, logfile, reuse=args.reuse_from, weights=args.weights,
                              conf=args.conf, imgsz=args.imgsz, device=args.device, force=args.force))

    manifest = build_manifest(clips, args.role, args.videos, rule_defaults(), args.dev_labels)
    (out / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(clip["events"] for clip in clips)
    log("事件：" + "、".join(f"{clip['stem']}={clip['events']}" for clip in clips) + f"，共 {total} 件", logfile)
    for clip in clips:
        if clip["dev_overlap"]:
            log(f"注意：{clip['stem']} 在開發集的標註裡（{args.dev_labels}），這支不是 out-of-sample 的新片。", logfile)
    log(f"manifest：{out / 'run.json'}", logfile)

    if args.no_review:
        return 0
    import newclip_review
    newclip_review.build_review(out)

    log("", logfile)
    log("下一步：在瀏覽器判完，按右上角匯出 JSON，然後", logfile)
    log(f"  python scripts/newclip_score.py --run {out} --verdicts <匯出的.json>", logfile)
    log("  要 recall，再加 --labels <整片獨立標註資料夾>（每支影片一個 <stem>.jsonl）", logfile)
    log("  要跟既有 7 支影片的判定合併，再加 --dev-run <開發集 run> --dev-verdicts <它的判定檔>", logfile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
