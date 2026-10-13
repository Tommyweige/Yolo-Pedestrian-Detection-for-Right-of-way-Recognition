"""Source-aware image/video ingestion with leakage-safe train/val/test splits.

Sources are only ever read. Output is a new folder holding copied images, extracted
video frames, a manifest.jsonl and dataset.json. Grouping happens before splitting:
every frame of one video (or every image sharing a group / content hash) lands in one split.
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv"}
SPLITS = ("train", "val", "test")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_image(path):
    """Return (height, width) or None when the file is not a decodable image."""
    try:
        data = np.fromfile(path, dtype=np.uint8)  # np.fromfile handles Unicode Windows paths
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)  # applies EXIF orientation, gray -> 3ch
    except Exception:
        return None
    return None if image is None else image.shape[:2]


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)  # smallest id wins: deterministic


def list_files(root, extensions):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def image_group(path, root, mode):
    if mode == "file":
        return f"file-{path.stem}"
    if mode == "single":
        return "single"
    relative = path.parent.relative_to(root)
    return "folder-" + (relative.as_posix() if relative.parts else root.name)


def roboflow_group(path):
    """Roboflow names files '<base>_jpg.rf.<hash>.jpg'; augmented copies share <base>."""
    return "roboflow-" + path.name.split(".rf.")[0]


def find_roboflow_label(image):
    label = image.parent.parent / "labels" / (image.stem + ".txt")
    return label if label.is_file() else None


def collect_candidates(args, rejected):
    """Yield dicts describing every raw image (not yet frames) with its group."""
    candidates = []
    for folder in args.images or []:
        root = Path(folder)
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                if path.suffix.lower() not in VIDEO_EXTENSIONS:
                    rejected.append({"source": str(path), "reason": "unsupported_extension"})
                continue
            candidates.append({"path": path, "group": image_group(path, root, args.group_by),
                               "origin": "local_images", "label": None})
    for folder in args.roboflow_export or []:
        for path in list_files(folder, IMAGE_EXTENSIONS):
            label = find_roboflow_label(path)
            candidates.append({"path": path, "group": roboflow_group(path),
                               "origin": "roboflow_export" if label else "local_images", "label": label})
    return candidates


def open_video(path):
    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        return capture, None
    capture.release()
    # OpenCV on Windows may fail on non-ASCII paths; retry through an ASCII temp copy.
    temp = tempfile.mkdtemp(prefix="ingest_")
    copy = Path(temp) / ("video" + path.suffix.lower())
    shutil.copyfile(path, copy)
    return cv2.VideoCapture(str(copy)), temp


def extract_frames(video, args, output_images, rejected):
    video_hash = sha256_file(video)
    video_id = f"video-{video_hash[:12]}"
    capture, temp = open_video(video)
    records = []
    try:
        if not capture.isOpened():
            rejected.append({"source": str(video), "reason": "video_unreadable"})
            return records
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        step = args.frame_interval or max(1, round(args.seconds_interval * (fps or 30.0)))
        frame_no = 0
        while True:
            pts_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            ok, frame = capture.read()
            if not ok:
                break
            if frame_no % step == 0:
                name = f"{video_id}_f{frame_no:07d}.png"
                destination = output_images / name
                ok_write, buffer = cv2.imencode(".png", frame)
                if ok_write:
                    buffer.tofile(str(destination))
                    height, width = frame.shape[:2]
                    records.append({
                        "path": destination, "group": video_id, "origin": "video_frame", "label": None,
                        "extra": {"video_id": video_id, "video_path": str(video), "video_sha256": video_hash,
                                  "frame_no": frame_no, "source_fps": round(fps, 4), "pts_ms": round(pts_ms, 3),
                                  "width": width, "height": height},
                    })
            frame_no += 1
    finally:
        capture.release()
        if temp:
            shutil.rmtree(temp, ignore_errors=True)
    return records


def assign_splits(group_sizes, proportions, seed, train_only):
    """Deterministically assign whole groups to splits, filling toward image-count targets."""
    if train_only:
        return {group: "train" for group in group_sizes}
    groups = sorted(group_sizes)
    random.Random(seed).shuffle(groups)
    total = sum(group_sizes.values())
    targets = {name: total * share for name, share in zip(SPLITS, proportions)}
    filled = {name: 0 for name in SPLITS}
    assignment = {}
    # Reserve one group per non-empty split first, so small datasets still populate val/test.
    wanted = [name for name, share in zip(SPLITS, proportions) if share > 0]
    for index, group in enumerate(groups):
        if index < len(wanted) and len(groups) >= len(wanted):
            choice = wanted[index]
        else:
            choice = max(wanted, key=lambda n: targets[n] - filled[n])
        assignment[group] = choice
        filled[choice] += group_sizes[group]
    return assignment


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="new dataset folder (must not already contain a manifest)")
    parser.add_argument("--images", nargs="*", help="folders of JPG/JPEG/PNG images")
    parser.add_argument("--videos", nargs="*", help="video files or folders (MP4/AVI/MKV)")
    parser.add_argument("--roboflow-export", nargs="*", help="unzipped Roboflow YOLO export folders (no login needed)")
    interval = parser.add_mutually_exclusive_group()
    interval.add_argument("--frame-interval", type=int, help="keep every Nth video frame")
    interval.add_argument("--seconds-interval", type=float, default=1.0, help="keep a frame every S seconds (default 1.0)")
    parser.add_argument("--group-by", choices=("folder", "file", "single"), default="folder",
                        help="source group for loose images: parent folder (default), each file, or one group")
    parser.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--train-only", action="store_true",
                        help="mark everything train; val/test must come from separate human gold labels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", default="v1")
    args = parser.parse_args(argv)
    if args.frame_interval is not None and args.frame_interval < 1:
        parser.error("--frame-interval must be >= 1")
    if args.seconds_interval <= 0:
        parser.error("--seconds-interval must be > 0")
    if any(share < 0 for share in args.split) or abs(sum(args.split) - 1.0) > 1e-6:
        parser.error("--split proportions must be non-negative and sum to 1")
    if not (args.images or args.videos or args.roboflow_export):
        parser.error("give at least one of --images, --videos, --roboflow-export")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if (output / "manifest.jsonl").exists():
        print(f"Refusing to overwrite existing dataset: {output / 'manifest.jsonl'}", file=sys.stderr)
        return 1
    source_roots = [Path(p).resolve() for p in (args.images or []) + (args.roboflow_export or [])]
    for root in source_roots:
        if not root.is_dir():
            print(f"Input folder not found: {root}", file=sys.stderr)
            return 2
    for root in source_roots:
        if root == output.resolve() or root in output.resolve().parents or output.resolve() in root.parents:
            print(f"Output must not overlap source folder: {root}", file=sys.stderr)
            return 2

    video_files = []
    for entry in args.videos or []:
        path = Path(entry)
        if path.is_dir():
            video_files += list_files(path, VIDEO_EXTENSIONS)
        elif path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(path)
        else:
            print(f"Video input not found or unsupported: {path}", file=sys.stderr)
            return 2

    rejected = []
    candidates = collect_candidates(args, rejected)
    if not candidates and not video_files:
        print("No supported images or videos found (JPG/JPEG/PNG, MP4/AVI/MKV). Nothing to ingest.", file=sys.stderr)
        return 2

    output_images = output / "images"
    output_labels = output / "labels"
    output_images.mkdir(parents=True, exist_ok=True)

    source_hashes = {}
    records = []
    for item in candidates:
        path = item["path"]
        size = decode_image(path)
        if size is None:
            rejected.append({"source": str(path), "reason": "corrupt_image"})
            continue
        digest = sha256_file(path)
        source_hashes[path] = digest
        item.update(sha256=digest, extra={"width": size[1], "height": size[0], "source_path": str(path)})
        records.append(item)
    for video in video_files:
        source_hashes[video] = sha256_file(video)
        for frame in extract_frames(video, args, output_images, rejected):
            frame["sha256"] = sha256_file(frame["path"])
            records.append(frame)

    # Merge groups that share content, then drop duplicate content so it cannot cross splits.
    union = UnionFind()
    first_by_hash = {}
    kept = []
    for item in records:
        union.find(item["group"])
        owner = first_by_hash.get(item["sha256"])
        if owner is None:
            first_by_hash[item["sha256"]] = item
            kept.append(item)
        else:
            union.union(owner["group"], item["group"])
            rejected.append({"source": str(item["extra"].get("source_path", item["extra"].get("video_path"))),
                             "reason": "duplicate_content", "duplicate_of": owner["sha256"]})
            if item["origin"] == "video_frame":
                item["path"].unlink(missing_ok=True)
    for item in kept:
        item["group"] = union.find(item["group"])

    if not kept:
        print("Every input was rejected; see rejected.jsonl. Nothing to ingest.", file=sys.stderr)
        (output / "rejected.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8")
        return 2

    sizes = {}
    for item in kept:
        sizes[item["group"]] = sizes.get(item["group"], 0) + 1
    assignment = assign_splits(sizes, args.split, args.seed, args.train_only)

    manifest = []
    for item in sorted(kept, key=lambda r: (r["group"], r["sha256"])):
        if item["origin"] == "video_frame":
            relative = Path("images") / item["path"].name
        else:
            relative = Path("images") / f"{item['sha256'][:16]}{item['path'].suffix.lower()}"
            shutil.copyfile(item["path"], output / relative)
        label_relative = None
        if item["label"] is not None:
            output_labels.mkdir(exist_ok=True)
            label_relative = Path("labels") / f"{item['sha256'][:16]}.txt"
            shutil.copyfile(item["label"], output / label_relative)
        manifest.append({
            "image_id": item["sha256"][:16], "sha256": item["sha256"], "file": relative.as_posix(),
            "source_group": item["group"], "split": assignment[item["group"]],
            "source_type": "video" if item["origin"] == "video_frame" else "image",
            "annotations_origin": item["origin"] if item["origin"] == "roboflow_export" else "none",
            "label_file": label_relative.as_posix() if label_relative else None,
            "gold": False,  # teacher or imported labels are never gold; humans promote gold separately
            "dataset_version": args.dataset_version, "seed": args.seed, **item["extra"],
        })

    # Source immutability: re-hash every input and fail loudly if anything changed.
    for path, digest in source_hashes.items():
        if sha256_file(path) != digest:
            print(f"Source changed during ingestion: {path}", file=sys.stderr)
            return 3

    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8")
    (output / "rejected.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8")
    counts = {name: sum(1 for row in manifest if row["split"] == name) for name in SPLITS}
    (output / "dataset.json").write_text(json.dumps({
        "dataset_version": args.dataset_version, "seed": args.seed, "split_proportions": args.split,
        "train_only": args.train_only, "images": len(manifest), "split_counts": counts,
        "groups": len(sizes), "rejected": len(rejected), "source_files_verified_unchanged": len(source_hashes),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(manifest), "split_counts": counts, "groups": len(sizes), "rejected": len(rejected)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
