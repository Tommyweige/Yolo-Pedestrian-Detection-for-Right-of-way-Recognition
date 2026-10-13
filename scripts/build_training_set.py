"""Assemble a 7-class YOLO training set from three sources with explicit provenance.

1. Human labels from a Roboflow export (via the dataset_ingest manifest). Polygons become bounding boxes.
2. Teacher pseudo-labels from auto_label.py, TRAIN images only, added where no human box already covers
   the object. The teacher is never used for `zebra` (its zebra boxes are unreliable) and never for val/test.
3. COCO val2017 (fully human-labelled person/vehicle/traffic light) as replay data so `person` is learned and
   not suppressed. A fixed holdout of COCO images joins the validation set.
Validation = Roboflow val split (human labels only; its labels are sparse, so read only the classes it labels)
plus the COCO holdout. Pseudo-labelled images never appear in validation.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

CLASSES = ["person", "car", "motorcycle", "bus", "truck", "traffic light", "zebra"]
COCO_IDS = {1: "person", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck", 10: "traffic light"}
NO_TEACHER = {"zebra"}


def read_label_file(path, names):
    """Return [(class_name, (x1, y1, x2, y2))] normalized; polygons are reduced to their bounding box."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        name = names[int(v[0])]
        values = [float(x) for x in v[1:]]
        if len(values) == 4:
            x, y, w, h = values
            box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        else:
            xs, ys = values[0::2], values[1::2]
            box = (min(xs), min(ys), max(xs), max(ys))
        rows.append((name, box))
    return rows


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_teacher(human, teacher, min_score, cover_iou=0.5):
    """Add teacher boxes that no human box already covers (any class: human labels win)."""
    merged = list(human)
    added = 0
    for item in teacher:
        name, box, score = item
        if name in NO_TEACHER or score < min_score:
            continue
        if any(iou(box, hbox) >= cover_iou for _, hbox in human):
            continue
        merged.append((name, box))
        added += 1
    return merged, added


def clip_box(box):
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in box)
    return (x1, y1, x2, y2) if x2 - x1 > 1e-4 and y2 - y1 > 1e-4 else None


def write_yolo(path, rows):
    lines = []
    for name, box in rows:
        box = clip_box(box)
        if box is not None:
            x1, y1, x2, y2 = box
            lines.append(f"{CLASSES.index(name)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return len(lines)


def teacher_rows(sidecar_path):
    data = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    if data.get("error") or not data.get("size"):
        return []
    width, height = data["size"]
    return [(CLASSES[b["class_id"]], (b["box"][0] / width, b["box"][1] / height, b["box"][2] / width, b["box"][3] / height),
             b["score"]) for b in data["boxes"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset_ingest folder built from the Roboflow export")
    parser.add_argument("--roboflow-names", nargs="+", default=["bus", "car", "motorcycle", "traffic light", "zebra"],
                        help="class names in the Roboflow data.yaml order")
    parser.add_argument("--teacher-labels", required=True, help="auto_label.py output folder")
    parser.add_argument("--coco-images", help="folder with COCO val2017 images")
    parser.add_argument("--coco-annotations", help="instances_val2017.json")
    parser.add_argument("--coco-train", type=int, default=2500)
    parser.add_argument("--coco-holdout", type=int, default=500)
    parser.add_argument("--min-teacher-score", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    dataset, teacher, output = Path(args.dataset), Path(args.teacher_labels), Path(args.output)
    if output.exists() and any(output.iterdir()):
        print(f"Refusing to write into a non-empty folder: {output}", file=sys.stderr)
        return 1
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    stats = {"roboflow_train": 0, "roboflow_val": 0, "roboflow_test_skipped": 0, "teacher_boxes_added": 0,
             "coco_train": 0, "coco_holdout": 0}
    provenance = []
    for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] == "test":
            stats["roboflow_test_skipped"] += 1  # test stays untouched for the final comparison
            continue
        split = "train" if row["split"] == "train" else "val"
        human = read_label_file(dataset / row["label_file"], args.roboflow_names) if row["label_file"] else []
        added = 0
        if split == "train":
            sidecar = teacher / "sidecars" / f"{row['image_id']}.json"
            if sidecar.is_file():
                human, added = merge_teacher(human, teacher_rows(sidecar), args.min_teacher_score)
        stem = f"rf_{row['image_id']}"
        shutil.copyfile(dataset / row["file"], output / "images" / split / f"{stem}{Path(row['file']).suffix}")
        write_yolo(output / "labels" / split / f"{stem}.txt", human)
        stats[f"roboflow_{split}"] += 1
        stats["teacher_boxes_added"] += added
        provenance.append({"file": stem, "split": split, "source": "roboflow+teacher" if added else "roboflow", "teacher_added": added})
    if args.coco_images and args.coco_annotations:
        coco = json.loads(Path(args.coco_annotations).read_text(encoding="utf-8"))
        images = {i["id"]: i for i in coco["images"]}
        per_image = {}
        for a in coco["annotations"]:
            if a["category_id"] in COCO_IDS and not a.get("iscrowd"):
                i = images[a["image_id"]]
                x, y, w, h = a["bbox"]
                per_image.setdefault(a["image_id"], []).append(
                    (COCO_IDS[a["category_id"]], (x / i["width"], y / i["height"], (x + w) / i["width"], (y + h) / i["height"])))
        ids = sorted(per_image)
        random.Random(args.seed).shuffle(ids)
        holdout, train = ids[:args.coco_holdout], ids[args.coco_holdout:args.coco_holdout + args.coco_train]
        for split, chosen in (("val", holdout), ("train", train)):
            for image_id in chosen:
                name = images[image_id]["file_name"]
                stem = f"coco_{Path(name).stem}"
                shutil.copyfile(Path(args.coco_images) / name, output / "images" / split / f"{stem}.jpg")
                write_yolo(output / "labels" / split / f"{stem}.txt", per_image[image_id])
                provenance.append({"file": stem, "split": split, "source": "coco_val2017", "teacher_added": 0})
            stats["coco_holdout" if split == "val" else "coco_train"] = len(chosen)
    (output / "data.yaml").write_text(
        f"path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)), encoding="utf-8")
    (output / "provenance.jsonl").write_text("".join(json.dumps(p) + "\n" for p in provenance), encoding="utf-8")
    (output / "build_report.json").write_text(json.dumps({"seed": args.seed, "min_teacher_score": args.min_teacher_score,
                                                           **stats}, indent=2), encoding="utf-8")
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
