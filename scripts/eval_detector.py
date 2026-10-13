"""Per-class detector metrics on the validation images from build_training_set.py (GPU).

Works for the COCO-pretrained YOLO26n (80 classes) and the fine-tuned 7-class model on the SAME images and
labels: for an 80-class model our labels are remapped to COCO ids and `zebra` (not a COCO class) is dropped.
--subset rf_ evaluates only the Roboflow val images (sparse human labels: read only the classes they label),
--subset coco_ only the fully-labelled COCO holdout. Numbers are detection quality, not event accuracy.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

OURS = ["person", "car", "motorcycle", "bus", "truck", "traffic light", "zebra"]
COCO_ID = {"person": 0, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7, "traffic light": 9}


def stage(val_images, val_labels, work, subset, remap):
    images, labels = work / "images" / "val", work / "labels" / "val"
    images.mkdir(parents=True), labels.mkdir(parents=True)
    count = 0
    for image in sorted(val_images.iterdir()):
        if not image.name.startswith(subset):
            continue
        shutil.copyfile(image, images / image.name)
        lines = []
        for line in (val_labels / f"{image.stem}.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            name = OURS[int(parts[0])]
            if remap:
                if name not in COCO_ID:
                    continue
                parts[0] = str(COCO_ID[name])
            lines.append(" ".join(parts))
        (labels / f"{image.stem}.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--train-set", required=True, help="folder made by build_training_set.py")
    parser.add_argument("--subset", choices=("rf_", "coco_"), required=True)
    parser.add_argument("--work", required=True, help="empty scratch folder for the remapped copy")
    parser.add_argument("--output", required=True, help="metrics json")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        print("CUDA is unavailable; refusing to evaluate on CPU.", file=sys.stderr)
        return 2
    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = dict(model.names)
    remap = len(names) == 80
    work, train_set = Path(args.work), Path(args.train_set)
    count = stage(train_set / "images" / "val", train_set / "labels" / "val", work, args.subset, remap)
    (work / "data.yaml").write_text(f"path: {work.resolve().as_posix()}\ntrain: images/val\nval: images/val\nnames:\n"
                                    + "".join(f"  {i}: {n}\n" for i, n in names.items()), encoding="utf-8")
    metrics = model.val(data=str(work / "data.yaml"), imgsz=args.imgsz, device=0, half=True, batch=16,
                        plots=False, verbose=False)
    wanted = [n for n in OURS if (n in COCO_ID if remap else True)]
    report = {"weights": Path(args.weights).name, "subset": args.subset, "images": count, "model_classes": len(names),
              "all_classes_map50": float(metrics.box.map50), "all_classes_map50_95": float(metrics.box.map), "per_class": {}}
    present = {metrics.names[i]: j for j, i in enumerate(metrics.box.ap_class_index)}
    for name in wanted:
        if name in present:
            j = present[name]
            report["per_class"][name] = {"ap50": float(metrics.box.ap50[j]), "ap50_95": float(metrics.box.ap[j]),
                                         "precision": float(metrics.box.p[j]), "recall": float(metrics.box.r[j])}
        else:
            report["per_class"][name] = None  # no labelled instances in this subset
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["per_class"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
