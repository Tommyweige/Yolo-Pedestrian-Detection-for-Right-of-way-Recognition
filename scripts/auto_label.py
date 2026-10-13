"""Teacher-assisted pseudo-labelling for a dataset made by scripts/dataset_ingest.py.

Reads <dataset>/manifest.jsonl, never modifies it or the images, and writes into --output:
  labels/<image_id>.txt      YOLO rows: class xc yc w h (normalized, 5 columns only)
  sidecars/<image_id>.json   per-box score/prompt/teacher/checkpoint + filtered boxes
  review_queue.jsonl         images needing a human look (corrupt, low score, filtered, zero boxes)
  report.json, gallery/      class counts, zero-detection rate, small objects, overlay images
These are PSEUDO labels from a teacher model; they are never ground truth.
Teachers: `stub` (offline, JSON detections, for CPU tests) and `grounding_dino` (needs the labeling env).
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ONTOLOGY = Path(__file__).resolve().parent.parent / "ontology" / "traffic.json"


def load_ontology(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts, prompt_class = [], {}
    for class_id, name in enumerate(config["classes"]):
        for prompt in config["prompts"][name]:
            prompts.append(prompt)
            prompt_class[prompt] = class_id
    config["prompt_list"], config["prompt_class"] = prompts, prompt_class
    return config


class StubTeacher:
    """Offline teacher: --stub-file maps image_id (or file name) -> [{prompt, score, box:[x1,y1,x2,y2]}]."""
    name, checkpoint = "stub", "none"

    def __init__(self, stub_file):
        self.table = json.loads(Path(stub_file).read_text(encoding="utf-8"))

    def predict(self, image_path, row):
        for key in (row["image_id"], Path(row["file"]).name):
            if key in self.table:
                result = self.table[key]
                if result == "error":
                    raise RuntimeError("stub teacher failure")
                return result
        return []


class GroundingDinoTeacher:
    name = "grounding_dino"

    def __init__(self, ontology, box_threshold, text_threshold):
        try:
            from autodistill.detection import CaptionOntology
            from autodistill_grounding_dino import GroundingDINO
        except ImportError as error:
            raise SystemExit(f"Grounding DINO needs the labeling environment (docs/environments.md): {error}")
        self.prompts = ontology["prompt_list"]
        self.model = GroundingDINO(CaptionOntology({p: p for p in self.prompts}),
                                   box_threshold=box_threshold, text_threshold=text_threshold)
        self.checkpoint = "autodistill-grounding-dino"

    def predict(self, image_path, row):
        try:
            detections = self.model.predict(str(image_path))
        except Exception as error:  # includes CUDA OOM; logged by the caller as a teacher error
            raise RuntimeError(f"{type(error).__name__}: {error}")
        scores = detections.confidence if detections.confidence is not None else [1.0] * len(detections)
        return [{"prompt": self.prompts[int(c)], "score": float(s), "box": [float(v) for v in b]}
                for b, c, s in zip(detections.xyxy, detections.class_id, scores)]


def validate_box(box, width, height, tolerance):
    """Return (clipped_box, None) or (None, reason)."""
    if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        return None, "non_finite"
    x1, y1, x2, y2 = box
    if x1 < -tolerance or y1 < -tolerance or x2 > width + tolerance or y2 > height + tolerance:
        return None, "outside_image"
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None, "non_positive_area"
    return [x1, y1, x2, y2], None


def class_filter_reason(box, class_id, width, height, ontology):
    """Reject boxes whose shape is implausible for their class (e.g. a 'zebra' box covering the whole road)."""
    limits = ontology.get("class_filters", {}).get(ontology["classes"][class_id])
    if not limits:
        return None
    w, h = box[2] - box[0], box[3] - box[1]
    if w * h / (width * height) > limits.get("max_area_frac", 1.0):
        return "class_filter_area"
    if w / h < limits.get("min_aspect", 0.0):
        return "class_filter_aspect"
    return None


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def class_nms(boxes, threshold):
    kept, removed = [], []
    for box in sorted(boxes, key=lambda b: -b["score"]):
        if any(k["class_id"] == box["class_id"] and iou(k["box"], box["box"]) > threshold for k in kept):
            removed.append(box)
        else:
            kept.append(box)
    return kept, removed


def process_image(teacher, ontology, image_path, row):
    """Return (kept_boxes, filtered_boxes, size, error)."""
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        image = None
    if image is None:
        return [], [], None, "corrupt_image"
    height, width = image.shape[:2]
    try:
        raw = teacher.predict(image_path, row)
    except Exception as error:
        return [], [], (width, height), f"teacher_error: {error}"
    accepted, filtered = [], []
    for item in raw:
        class_id = ontology["prompt_class"].get(item.get("prompt"))
        record = {"prompt": item.get("prompt"), "score": item.get("score"), "class_id": class_id,
                  "box": item.get("box")}
        score = item.get("score")
        if class_id is None:
            filtered.append({**record, "reason": "unknown_class"})
        elif not isinstance(score, (int, float)) or not math.isfinite(score) or score < ontology["box_threshold"]:
            filtered.append({**record, "reason": "below_box_threshold"})
        else:
            box, reason = validate_box(item.get("box", []), width, height, ontology["edge_tolerance_px"])
            if box is not None:
                reason = class_filter_reason(box, class_id, width, height, ontology)
            if reason:
                filtered.append({**record, "reason": reason})
            else:
                accepted.append({**record, "box": box})
    kept, removed = class_nms(accepted, ontology["nms_iou"])
    filtered += [{**b, "reason": "nms_duplicate"} for b in removed]
    return kept, filtered, (width, height), None


def yolo_line(box, class_id, width, height):
    x1, y1, x2, y2 = box["box"]
    return (f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def draw_overlay(image_path, kept, ontology, destination):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    for box in kept:
        x1, y1, x2, y2 = (int(v) for v in box["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{ontology['classes'][box['class_id']]} {box['score']:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    ok, buffer = cv2.imencode(".jpg", image)
    if ok:
        buffer.tofile(str(destination))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="folder produced by dataset_ingest.py")
    parser.add_argument("--output", required=True, help="folder for labels/sidecars/report (not the dataset folder)")
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    parser.add_argument("--teacher", choices=("stub", "grounding_dino"), default="stub")
    parser.add_argument("--stub-file", help="detections JSON for --teacher stub")
    parser.add_argument("--splits", nargs="*", default=["train"], help="splits to label (default: train only)")
    parser.add_argument("--dry-run", type=int, metavar="N", help="label only a seeded random sample of N images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reverse", action="store_true", help="process in reverse order (a second worker can meet the first in the middle)")
    parser.add_argument("--retry-failed", action="store_true", help="re-run images whose earlier attempt errored")
    args = parser.parse_args(argv)
    if args.teacher == "stub" and not args.stub_file:
        parser.error("--teacher stub needs --stub-file")
    return args


def main(argv=None):
    args = parse_args(argv)
    dataset, output = Path(args.dataset), Path(args.output)
    manifest_path = dataset / "manifest.jsonl"
    if not manifest_path.is_file():
        print(f"No manifest.jsonl in {dataset}; run scripts/dataset_ingest.py first.", file=sys.stderr)
        return 2
    if output.resolve() == dataset.resolve():
        print("--output must differ from the dataset folder (source images stay untouched).", file=sys.stderr)
        return 2
    ontology = load_ontology(args.ontology)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if r["split"] in args.splits]
    if args.dry_run:
        rows = random.Random(args.seed).sample(rows, min(args.dry_run, len(rows)))
    if not rows:
        print("No images selected (check --splits / manifest).", file=sys.stderr)
        return 2
    teacher = (StubTeacher(args.stub_file) if args.teacher == "stub" else
               GroundingDinoTeacher(ontology, ontology["box_threshold"], ontology["text_threshold"]))

    for folder in ("labels", "sidecars", "gallery"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    review, counts = [], {name: 0 for name in ontology["classes"]}
    zero = small = filtered_total = processed = resumed = 0
    scores = []
    filtered_reasons = {}
    for row in sorted(rows, key=lambda r: r["image_id"], reverse=args.reverse):
        sidecar_path = output / "sidecars" / f"{row['image_id']}.json"
        label_path = output / "labels" / f"{row['image_id']}.txt"
        previous = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.is_file() else None
        if previous and (previous.get("error") is None or not args.retry_failed):
            resumed += 1  # resume: finished (or failed without --retry-failed) images are not redone
            kept, filtered, error = previous["boxes"], previous["filtered"], previous.get("error")
            size = previous.get("size")
        else:
            image_path = dataset / row["file"]
            kept, filtered, size, error = process_image(teacher, ontology, image_path, row)
            sidecar_path.write_text(json.dumps({
                "image_id": row["image_id"], "source_file": row["file"], "teacher": teacher.name,
                "checkpoint": teacher.checkpoint, "pseudo_label": True, "gold": False, "size": size,
                "error": error, "boxes": kept, "filtered": filtered,
            }, indent=1), encoding="utf-8")
            if error is None:
                label_path.write_text("".join(yolo_line(b, b["class_id"], *size) + "\n" for b in kept),
                                      encoding="utf-8")  # empty file = valid background image
                draw_overlay(image_path, kept, ontology, output / "gallery" / f"{row['image_id']}.jpg")
        processed += 1
        reasons = []
        if error:
            reasons.append(error.split(":")[0])
        else:
            if not kept:
                zero += 1
                reasons.append("zero_detection")
            if any(b["score"] < ontology["review_score_below"] for b in kept):
                reasons.append("low_confidence")
            if filtered:
                reasons.append("filtered_boxes")
            for b in kept:
                counts[ontology["classes"][b["class_id"]]] += 1
                scores.append(b["score"])
                if b["box"][3] - b["box"][1] < ontology["small_object_height_px"]:
                    small += 1
            for f in filtered:
                filtered_reasons[f["reason"]] = filtered_reasons.get(f["reason"], 0) + 1
            filtered_total += len(filtered)
        if reasons:
            review.append({"image_id": row["image_id"], "file": row["file"], "reasons": reasons})

    errors = sum(1 for r in review if any(x in ("corrupt_image", "teacher_error") for x in r["reasons"]))
    (output / "review_queue.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review), encoding="utf-8")
    histogram = np.histogram(scores, bins=10, range=(0, 1))[0].tolist() if scores else [0] * 10
    report = {"teacher": teacher.name, "checkpoint": teacher.checkpoint, "pseudo_label": True,
              "images": processed, "resumed_from_sidecar": resumed, "errors": errors,
              "zero_detection_images": zero,
              "zero_detection_rate": round(zero / max(1, processed - errors), 4),
              "class_counts": counts, "filtered_boxes": filtered_total, "filtered_by_reason": filtered_reasons,
              "small_objects_height_below_px": ontology["small_object_height_px"], "small_objects": small,
              "score_histogram_0_to_1_in_10_bins": histogram, "review_queue": len(review),
              "caveats": ontology["notes"]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows_html = "".join(f'<figure><img src="{r["image_id"]}.jpg" width="320"><figcaption>{r["image_id"]} '
                        f'{", ".join(r["reasons"])}</figcaption></figure>'
                        for r in review if not any(x in ("corrupt_image", "teacher_error") for x in r["reasons"]))
    (output / "gallery" / "index.html").write_text(
        "<meta charset='utf-8'><title>Review gallery</title><h1>Pseudo-labels to review</h1>" + rows_html,
        encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("images", "errors", "zero_detection_rate", "class_counts",
                                            "review_queue")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
