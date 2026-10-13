"""Detect + track people/vehicles in a video and stabilise foot points against camera motion.

Runs in the modern env (YOLO26 + built-in ByteTrack). For a moving camera (dashcam, helmet cam) the
frame-to-frame homography is estimated from optical-flow-tracked road features outside detected boxes and chained to a
reference frame, so every track point is expressed in the reference image. The output feeds yield_rules.py.
Chained homographies drift over long clips and assume mostly planar/far content; check the overlay.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

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


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", required=True, help="verified YOLO26 .pt (see docs/environments.md)")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument("--reference-frame", type=int, required=True, help="frame whose image coordinates are used")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--crop-bottom", type=float, default=0.2, help="fraction of the image bottom ignored for features")
    parser.add_argument("--road-top", type=float, default=0.4, help="fraction of the image top ignored (not road)")
    parser.add_argument("--device", default="0", help="CUDA device for the detector (default 0); cpu must be asked for explicitly")
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
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
    capture.release()
    model = YOLO(args.weights)
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
        gray = cv2.cvtColor(cv2.resize(result.orig_img, None, fx=SCALE, fy=SCALE), cv2.COLOR_BGR2GRAY)
        current = (gray, road_features(gray, feature_mask(gray.shape, xyxy, args.crop_bottom, args.road_top)))
        if previous is not None:
            matrix, inliers = estimate_step(previous[0], previous[1], gray)
            if matrix is None:
                failures.append(frame)
                matrix, inliers = np.eye(3), 0
            steps[frame] = {"H_to_prev": to_full(matrix).tolist(), "inliers": inliers}
        previous = current
        for identity, box, cls in zip(ids, xyxy, classes):
            foot = [float((box[0] + box[2]) / 2), float(box[3])]
            tracks.setdefault(identity, {"id": identity, "kind": kinds[cls], "raw": []})["raw"].append([frame, *foot])

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
        "weights": Path(args.weights).name, "failed_frames": failures,
        "to_reference": {str(f): m.tolist() for f, m in to_ref.items()},
    }), encoding="utf-8")
    print(json.dumps({"tracks": len(stabilised), "frames": len(steps) + 1, "stabilisation_failures": len(failures)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
