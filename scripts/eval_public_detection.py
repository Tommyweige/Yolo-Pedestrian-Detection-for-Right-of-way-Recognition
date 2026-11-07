"""Measure how many boxes the pipeline detector misses on public datasets (iVS and BDD100K).

This is recall-first, not mAP: predictions from the pipeline's own confidence threshold are greedily matched to the
ground truth at IoU 0.5, and the result is split by size, time of day and occlusion. Images are cached so an
interrupted run can resume without touching the GPU again.

  python scripts/eval_public_detection.py --ivs datasets/ivs --bdd-images datasets/bdd/images \
      --bdd-labels datasets/bdd/labels.json --cache runtime/public-detection-cache.json \
      --output runtime/public-detection.json
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

GROUPS = ("person", "motorcycle", "vehicle", "bicycle")
PRED_MAP = {"person": "person", "motorcycle": "motorcycle", "car": "vehicle", "bus": "vehicle",
            "truck": "vehicle", "bicycle": "bicycle"}
IVS_MAP = {"pedestrian": "person", "scooter": "motorcycle", "vehicle": "vehicle", "bike": "bicycle"}
BDD_MAP = {"person": "person", "motor": "motorcycle", "car": "vehicle", "bus": "vehicle",
           "truck": "vehicle", "bike": "bicycle"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
CONDITIONS = ("size", "timeofday", "occluded")


def parse_voc(xml_text) -> list[dict]:
    """Return the mapped iVS objects in a VOC annotation; unmapped names are dropped."""
    root = ET.fromstring(xml_text)
    objects = []
    for obj in root.findall(".//object"):
        name = (obj.findtext("name") or "").strip()
        group = IVS_MAP.get(name)
        if group is None:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            coordinates = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            continue
        objects.append({"group": group, "box": coordinates, "occluded": None})
    return objects


def parse_bdd_item(item) -> tuple[dict, list[dict], list[list[float]]]:
    """Return (image attributes, mapped ground truths, rider ignore boxes) for one BDD100K label item."""
    source = item.get("attributes") or {}
    attributes = {key: source.get(key) for key in ("timeofday", "weather", "scene")}
    gts, ignore_boxes = [], []
    for label in item.get("labels") or []:
        box2d = label.get("box2d")
        if not box2d:
            continue
        try:
            box = [float(box2d[key]) for key in ("x1", "y1", "x2", "y2")]
        except (KeyError, TypeError, ValueError):
            continue
        category = label.get("category")
        if category == "rider":
            ignore_boxes.append(box)
            continue
        group = BDD_MAP.get(category)
        if group is None:
            continue
        label_attributes = label.get("attributes") or {}
        gts.append({"group": group, "box": box, "occluded": bool(label_attributes.get("occluded", False))})
    return attributes, gts, ignore_boxes


def iou(a, b) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match(gts, preds, ignore_boxes=(), iou_thr=0.5) -> dict:
    """Greedily match predictions to same-group ground truths; return aligned hits and false positives per group."""
    gt_hit = [False] * len(gts)
    fp = {group: 0 for group in GROUPS}
    for pred in sorted(preds, key=lambda item: float(item.get("conf", 0.0)), reverse=True):
        group, box = pred.get("group"), pred.get("box")
        best_index, best_iou = -1, 0.0
        for index, gt in enumerate(gts):
            if gt_hit[index] or gt.get("group") != group:
                continue
            overlap = iou(box, gt["box"])
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index >= 0 and best_iou >= iou_thr:
            gt_hit[best_index] = True
            continue
        if group == "person" and any(iou(box, ignore) >= 0.5 for ignore in ignore_boxes):
            continue
        fp[group] = fp.get(group, 0) + 1
    return {"gt_hit": gt_hit, "fp": fp}


def size_bucket(box, image_height) -> str:
    """Classify a box by its height relative to the image height."""
    ratio = (float(box[3]) - float(box[1])) / float(image_height)
    if ratio < 0.05:
        return "small"
    if ratio < 0.15:
        return "medium"
    return "large"


def summarize(records) -> dict:
    """Aggregate ground-truth hits and per-image false positives into dataset/group and condition tables."""
    summary, false_positives = {}, {}
    condition_tables = {condition: {} for condition in CONDITIONS}
    for record in records:
        if record.get("kind") == "fp":
            key = (record.get("dataset"), record.get("group"))
            false_positives[key] = false_positives.get(key, 0) + int(record.get("fp", 0))
            continue
        dataset, group = record.get("dataset"), record.get("group")
        row = summary.setdefault(dataset, {}).setdefault(group, {"gt": 0, "hit": 0, "fp": 0})
        row["gt"] += 1
        row["hit"] += 1 if record.get("hit") else 0
        for condition in CONDITIONS:
            value = record.get(condition)
            if value is None:
                continue
            bucket = condition_tables[condition].setdefault(group, {}).setdefault(
                value, {"gt": 0, "hit": 0})
            bucket["gt"] += 1
            bucket["hit"] += 1 if record.get("hit") else 0

    for (dataset, group), count in false_positives.items():
        row = summary.setdefault(dataset, {}).setdefault(group, {"gt": 0, "hit": 0, "fp": 0})
        row["fp"] += count

    for dataset in summary.values():
        for row in dataset.values():
            row["recall"] = row["hit"] / row["gt"] if row["gt"] else None
            row["precision"] = row["hit"] / (row["hit"] + row["fp"]) if row["hit"] + row["fp"] else None

    for condition in CONDITIONS:
        for group in condition_tables[condition].values():
            for row in group.values():
                row["recall"] = row["hit"] / row["gt"] if row["gt"] else None
        summary[f"by_{condition}"] = condition_tables[condition]
    return summary


def _jpeg_height(path):
    """Read the height from a JPEG's SOF header without decoding the pixels; None if not a parsable JPEG."""
    with open(path, "rb") as handle:
        if handle.read(2) != bytes((0xFF, 0xD8)):
            return None
        while True:
            marker = handle.read(2)
            if len(marker) < 2 or marker[0] != 0xFF:
                return None
            if marker[1] in (0xD8, 0x01) or 0xD0 <= marker[1] <= 0xD7:
                continue
            length = handle.read(2)
            if len(length) < 2:
                return None
            size = int.from_bytes(length, "big")
            if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
                header = handle.read(3)
                return int.from_bytes(header[1:3], "big") if len(header) == 3 else None
            handle.seek(size - 2, 1)


def _image_height(path):
    if Path(path).suffix.lower() in (".jpg", ".jpeg"):
        height = _jpeg_height(path)
        if height:
            return height
    image = cv2.imread(str(path))
    return int(image.shape[0]) if image is not None else None


def _dataset_candidates(dataset, root):
    return [dataset, root.name, root.stem, str(root)]


def _prediction_keys(item):
    image = item["image"]
    return [item["key"], image.name, f"JPEGImages/{image.name}", str(image), image.as_posix()]


def _lookup(cache, item):
    keys = _prediction_keys(item)
    for dataset in item["dataset_candidates"]:
        predictions = cache.get(dataset)
        if isinstance(predictions, dict):
            for key in keys:
                if key in predictions:
                    return predictions[key]
    if len(cache) == 1:
        predictions = next(iter(cache.values()))
        if isinstance(predictions, dict):
            for key in keys:
                if key in predictions:
                    return predictions[key]
    return None


def _write_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _ivs_items(directories, limit):
    items = []
    for directory in directories:
        root = Path(directory)
        images = root / "JPEGImages"
        if not images.is_dir():
            continue
        found = sorted((path for path in images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
                       key=lambda path: path.name)
        for image in found:
            items.append({
                "dataset": "ivs",
                "dataset_candidates": _dataset_candidates("ivs", root),
                "root": root,
                "image": image,
                "key": image.relative_to(root).as_posix(),
                "annotation": root / "Annotations" / f"{image.stem}.xml",
            })
    if limit is not None:
        items = sorted(items, key=lambda item: item["image"].name)[:limit]
    return items


def _bdd_items(images_dir, labels_path, limit):
    if not images_dir:
        return []
    root = Path(images_dir)
    images = sorted((path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
                    key=lambda path: path.name)
    if limit is not None:
        images = images[:limit]
    labels = {}
    if labels_path:
        for item in json.loads(Path(labels_path).read_text(encoding="utf-8")):
            labels[item.get("name")] = item
    return [{
        "dataset": "bdd",
        "dataset_candidates": _dataset_candidates("bdd", root) + ["bdd100k"],
        "root": root,
        "image": image,
        "key": image.name,
        "item": labels.get(image.name),
    } for image in images]


def _predict(items, cache, cache_path, weights, conf, imgsz, batch, log):
    """Return {id(item): predictions}, reading the cache first and only loading YOLO for uncached images."""
    predictions = {}
    uncached = []
    for item in items:
        cached = _lookup(cache, item)
        if cached is None:
            uncached.append(item)
        else:
            predictions[id(item)] = cached
    if not uncached:
        return predictions

    import torch
    if not torch.cuda.is_available():
        log("CUDA is unavailable; refusing to evaluate on CPU.")
        return None
    from ultralytics import YOLO

    model = YOLO(weights)
    names = dict(model.names)
    classes = [index for index, name in names.items() if name in PRED_MAP]
    from concurrent.futures import ThreadPoolExecutor

    def load(chunk):  # decode the next batch on worker threads while the GPU runs the current one
        return list(pool.map(lambda item: cv2.imread(str(item["image"])), chunk))

    chunks = [uncached[start:start + batch] for start in range(0, len(uncached), batch)]
    written = 0
    pool = ThreadPoolExecutor(max_workers=8)
    prefetch = ThreadPoolExecutor(max_workers=1)
    pending = prefetch.submit(load, chunks[0])
    for index, chunk in enumerate(chunks):
        images = pending.result()
        if index + 1 < len(chunks):
            pending = prefetch.submit(load, chunks[index + 1])
        keep = [(item, image) for item, image in zip(chunk, images) if image is not None]
        chunk = [item for item, _ in keep]
        results = model.predict([image for _, image in keep], conf=conf, imgsz=imgsz,
                                device=0, half=True, verbose=False, classes=classes) if keep else []
        for item, result in zip(chunk, results):
            found = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                for box, class_id, score in zip(boxes.xyxy.cpu().numpy(), boxes.cls.int().tolist(),
                                                boxes.conf.cpu().numpy()):
                    group = PRED_MAP.get(names.get(int(class_id)))
                    if group is not None:
                        found.append({"group": group, "box": [float(value) for value in box],
                                      "conf": float(score)})
            predictions[id(item)] = found
            cache.setdefault(item["dataset"], {})[item["key"]] = found
        done = len(predictions)
        if cache_path and done - written >= 200:  # rewriting the whole JSON every batch starved the GPU
            _write_cache(cache_path, cache)
            written = done
        if done // 200 != (done - len(chunk)) // 200:
            log(f"[{done}/{len(items)}] predicted")
    pool.shutdown()
    prefetch.shutdown()
    if cache_path:
        _write_cache(cache_path, cache)
    return predictions


def _records(items, predictions):
    records = []
    for item in items:
        preds = predictions.get(id(item), [])
        if item["dataset"] == "ivs":
            annotation = item["annotation"]
            gts = parse_voc(annotation.read_text(encoding="utf-8")) if annotation.exists() else []
            ignore_boxes, attributes = [], {"timeofday": None}
        else:
            label = item.get("item")
            attributes, gts, ignore_boxes = (parse_bdd_item(label) if label
                                             else ({"timeofday": None}, [], []))
        result = match(gts, preds, ignore_boxes)
        height = _image_height(item["image"])
        for index, gt in enumerate(gts):
            records.append({
                "dataset": item["dataset"],
                "group": gt["group"],
                "hit": result["gt_hit"][index],
                "size": size_bucket(gt["box"], height) if height else None,
                "timeofday": attributes.get("timeofday"),
                "occluded": gt["occluded"],
            })
        for group, count in result["fp"].items():
            if count:
                records.append({"kind": "fp", "dataset": item["dataset"], "group": group,
                                "fp": count, "timeofday": attributes.get("timeofday")})
    return records


def _rate(value):
    return "None" if value is None else f"{value:.3f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="runtime/smoke-assets/yolo26x.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--ivs", nargs="*", default=[], metavar="DIR",
                        help="iVS folders with Annotations/*.xml and JPEGImages/*.jpg")
    parser.add_argument("--bdd-images", help="BDD100K image folder")
    parser.add_argument("--bdd-labels", help="BDD100K label JSON list")
    parser.add_argument("--limit", type=int, help="at most N images per dataset, by file name")
    parser.add_argument("--cache", help="prediction cache JSON, written every 200 images")
    parser.add_argument("--output", help="summary JSON")
    parser.add_argument("--log", help="also append progress lines to this UTF-8 file")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    log_handle = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        if log_handle:
            print(message, file=log_handle, flush=True)

    try:
        cache_path = Path(args.cache) if args.cache else None
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path and cache_path.exists() else {}
        items = _ivs_items(args.ivs, args.limit) + _bdd_items(args.bdd_images, args.bdd_labels, args.limit)
        predictions = _predict(items, cache, cache_path, args.weights, args.conf, args.imgsz, args.batch, log)
        if predictions is None:
            return 2
        summary = summarize(_records(items, predictions))

        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        for dataset in sorted(key for key in summary if key not in {f"by_{c}" for c in CONDITIONS}):
            for group in GROUPS:
                row = summary[dataset].get(group)
                if row is not None:
                    log(f"{dataset} {group}: recall={_rate(row['recall'])} "
                        f"precision={_rate(row['precision'])} gt={row['gt']} fp={row['fp']}")
        for condition in CONDITIONS:
            log(f"by_{condition}:")
            table = summary.get(f"by_{condition}", {})
            for group in GROUPS:
                for value, row in sorted(table.get(group, {}).items(), key=lambda pair: str(pair[0])):
                    log(f"  {group} {value}: recall={_rate(row['recall'])} gt={row['gt']} hit={row['hit']}")
        log(json.dumps(summary, ensure_ascii=False))
        return 0
    finally:
        if log_handle:
            log_handle.close()


if __name__ == "__main__":
    sys.exit(main())
