"""Detect + track people/vehicles in a video and stabilise foot points against camera motion.

Runs in the modern env (YOLO26 + built-in ByteTrack). For a moving camera (dashcam, helmet cam) the
frame-to-frame homography is estimated from optical-flow-tracked road features outside detected boxes and chained to a
reference frame, so every track point is expressed in the reference image. The output feeds yield_rules.py.
Chained homographies drift over long clips and assume mostly planar/far content; check the overlay.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # repo root, where detector_weights.py lives
sys.path.insert(0, str(ROOT))
from detector_weights import MANIFEST, resolve_weights, verify_sha256  # noqa: E402

KIND_NAMES = ("person", "car", "motorcycle", "bus", "truck")  # class ids are looked up by NAME per model
SCALE = 0.5  # features are matched on a half-size image


def feature_mask(shape, boxes, crop_bottom, road_top):
    mask = np.full(shape[:2], 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        pad = 10
        mask[max(0, int(y1 * SCALE) - pad):int(y2 * SCALE) + pad, max(0, int(x1 * SCALE) - pad):int(x2 * SCALE) + pad] = 0
    mask[int(shape[0] * (1 - crop_bottom)):, :] = 0  # helmet / on-screen timestamp
    mask[:int(shape[0] * road_top), :] = 0  # buildings are off the road plane and break the homography
    return mask


def road_features(gray, mask):
    return cv2.goodFeaturesToTrack(gray, maxCorners=1500, qualityLevel=0.001, minDistance=8, mask=mask)


def estimate_step(prev_gray, prev_points, cur_gray):
    """Homography mapping cur-frame (half-size) points into prev-frame points, or (None, inliers)."""
    if prev_points is None or len(prev_points) < 20:
        return None, 0
    # Track prev -> cur with LK, then fit cur -> prev so the result matches the chaining convention.
    cur_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_points, None, winSize=(31, 31), maxLevel=4)
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, cur_points, None, winSize=(31, 31), maxLevel=4)
    ok = (status.ravel() == 1) & (status_back.ravel() == 1) & (np.linalg.norm(back - prev_points, axis=2).ravel() < 1.0)
    if ok.sum() < 20:
        return None, int(ok.sum())
    matrix, inliers = cv2.findHomography(cur_points[ok], prev_points[ok], cv2.RANSAC, 2.0)
    if matrix is None or int(inliers.sum()) < 15:
        return None, 0 if inliers is None else int(inliers.sum())
    return matrix, int(inliers.sum())


def to_full(matrix):
    """Conjugate a half-size homography so it works on full-size pixel coordinates."""
    s = np.diag([SCALE, SCALE, 1.0])
    return np.linalg.inv(s) @ matrix @ s


def drop_ego_parts(tracks, frame_size, bottom_fraction=0.88, max_height_fraction=0.25):
    """Remove vehicle boxes that are the camera's own mirror/handlebar; returns (kept tracks, boxes removed).

    A helmet-cam mirror is detected as a "car" in the bottom strip of the image, in many short ByteTrack fragments, so
    the test is per box: the centre lies in the bottom `1 - bottom_fraction` of the image and the box is low
    (under `max_height_fraction` of the image height). A real vehicle that close to the camera fills much more of the
    image. Persons are never removed.
    """
    _, height = frame_size
    removed = 0

    def ego(box):
        return (box[2] + box[4]) / 2 >= bottom_fraction * height and box[4] - box[2] < max_height_fraction * height

    kept = []
    for track in tracks:
        if track["kind"] == "person":
            kept.append(track)
            continue
        boxes = [b for b in track["boxes"] if not ego(b)]
        removed += len(track["boxes"]) - len(boxes)
        if boxes:
            frames = {b[0] for b in boxes}
            kept.append({**track, "boxes": boxes, "raw": [r for r in track["raw"] if r[0] in frames]})
    return kept, removed


def refilter(video, path):
    capture = cv2.VideoCapture(video)
    size, total = (int(capture.get(3)), int(capture.get(4))), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    tracks = json.loads(path.read_text(encoding="utf-8"))
    kept, removed = drop_ego_parts(tracks, size)
    path.write_text(json.dumps(kept), encoding="utf-8")
    print(f"removed {removed} camera-fixed vehicle box(es) at the bottom of the frame; {len(kept)} tracks kept", flush=True)
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", help="YOLO26 .pt; default: $TRAFFIC_DETECTOR_WEIGHTS, then weights/yolo26n-traffic.pt, then stock yolo26n.pt")
    parser.add_argument("--output", help="output folder")
    parser.add_argument("--refilter", metavar="RAW_TRACKS.json", help="only re-apply the camera-fixed-vehicle filter to an existing raw_tracks.json (no detection)")
    parser.add_argument("--reference-frame", type=int, help="stabilise into this frame's coordinates; omit to skip stabilisation (raw tracks only)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--crop-bottom", type=float, default=0.2, help="fraction of the image bottom ignored for features")
    parser.add_argument("--road-top", type=float, default=0.4, help="fraction of the image top ignored (not road)")
    parser.add_argument("--device", default="0", help="CUDA device for the detector (default 0); cpu must be asked for explicitly")
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.refilter:
        return refilter(args.video, Path(args.refilter))
    if not args.output:
        print("--output is required (unless --refilter)", file=sys.stderr)
        return 2
    weights, source = resolve_weights(args.weights, ROOT, os.environ)
    print(f"detector weights: {weights} ({source})", flush=True)
    from ultralytics import YOLO  # imported late so --help works without the modern env
    import torch
    if args.device != "cpu" and not torch.cuda.is_available():
        print("CUDA is unavailable; refusing to fall back to CPU silently (pass --device cpu to allow it).", file=sys.stderr)
        return 2
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(capture.get(3)), int(capture.get(4)))
    capture.release()
    model = YOLO(weights)
    resolved = Path(getattr(model, "ckpt_path", None) or weights)
    if resolved.exists():
        if verify_sha256(resolved, ROOT / MANIFEST) == "verified":
            print("weights sha256: verified", flush=True)
        else:
            print("weights sha256: unlisted (not in requirements/smoke-assets.json)", flush=True)
    kinds = {i: n for i, n in model.names.items() if n in KIND_NAMES}  # COCO-80 and our 7-class models differ in ids
    tracks, steps, failures, previous = {}, {}, [], None
    stream = model.track(source=args.video, stream=True, tracker="bytetrack.yaml", persist=True,
                         classes=list(kinds), conf=args.conf, imgsz=args.imgsz, device=args.device,
                         half=args.device != "cpu", verbose=False)
    for index, result in enumerate(stream):
        frame = index + 1  # one-based, same as the legacy event timestamps
        if args.max_frames and index >= args.max_frames:
            break
        boxes = result.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else []
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
        classes = boxes.cls.int().tolist() if len(boxes) else []
        if args.reference_frame:
            gray = cv2.cvtColor(cv2.resize(result.orig_img, None, fx=SCALE, fy=SCALE), cv2.COLOR_BGR2GRAY)
            current = (gray, road_features(gray, feature_mask(gray.shape, xyxy, args.crop_bottom, args.road_top)))
        if previous is not None and args.reference_frame:
            matrix, inliers = estimate_step(previous[0], previous[1], gray)
            if matrix is None:
                failures.append(frame)
                matrix, inliers = np.eye(3), 0
            steps[frame] = {"H_to_prev": to_full(matrix).tolist(), "inliers": inliers}
        if args.reference_frame:
            previous = current
        for identity, box, cls in zip(ids, xyxy, classes):
            foot = [float((box[0] + box[2]) / 2), float(box[3])]
            entry = tracks.setdefault(identity, {"id": identity, "kind": kinds[cls], "raw": [], "boxes": []})
            entry["raw"].append([frame, *foot])
            entry["boxes"].append([frame, *(float(v) for v in box)])  # x1, y1, x2, y2 in source pixels

    kept, removed = drop_ego_parts(list(tracks.values()), size)
    if removed:
        print(f"removed {removed} camera-fixed vehicle box(es) at the bottom of the frame (mirror/handlebar)", flush=True)
    tracks = {t["id"]: t for t in kept}
    if not args.reference_frame:
        (output / "raw_tracks.json").write_text(json.dumps(list(tracks.values())), encoding="utf-8")
        print(json.dumps({"tracks": len(tracks), "stabilised": False}))
        return 0
    # Chain frame->previous homographies into frame->reference, then map every foot point.
    if not 1 <= args.reference_frame <= max(steps, default=1):
        print("--reference-frame is outside the processed frames", file=sys.stderr)
        return 2
    to_ref = {args.reference_frame: np.eye(3)}
    for frame in range(args.reference_frame + 1, max(steps) + 1):  # later frames: ref <- ... <- frame
        to_ref[frame] = to_ref[frame - 1] @ np.array(steps[frame]["H_to_prev"])
    for frame in range(args.reference_frame - 1, 0, -1):  # earlier frames: invert the step forward in time
        to_ref[frame] = to_ref[frame + 1] @ np.linalg.inv(np.array(steps[frame + 1]["H_to_prev"]))
    stabilised = []
    for track in tracks.values():
        points = []
        for frame, x, y in track["raw"]:
            if frame in to_ref:
                px, py, pw = to_ref[frame] @ np.array([x, y, 1.0])
                points.append([frame, px / pw, py / pw])
        stabilised.append({"id": track["id"], "kind": track["kind"], "points": points})
    (output / "tracks.json").write_text(json.dumps(stabilised), encoding="utf-8")
    (output / "raw_tracks.json").write_text(json.dumps(list(tracks.values())), encoding="utf-8")
    (output / "stabilisation.json").write_text(json.dumps({
        "video": Path(args.video).name, "fps": fps, "frames_total": total, "reference_frame": args.reference_frame,
        "weights": Path(weights).name, "failed_frames": failures,
        "to_reference": {str(f): m.tolist() for f, m in to_ref.items()},
    }), encoding="utf-8")
    print(json.dumps({"tracks": len(stabilised), "frames": len(steps) + 1, "stabilisation_failures": len(failures)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
