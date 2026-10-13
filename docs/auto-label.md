# Teacher pseudo-labelling (`scripts/auto_label.py`)

Pseudo labels only. A teacher model's boxes are **not ground truth**; `gold` is always `false`.
`zebra` boxes may cover only the white stripes, and traffic-light colour is decided by separate logic.

```powershell
# 1. ingest (see docs/dataset-ingest.md)
python scripts/dataset_ingest.py --output datasets/v1 --videos D:\clips --seconds-interval 1
# 2. offline dry-run on 20 images with the stub teacher (CPU, no weights)
python scripts/auto_label.py --dataset datasets/v1 --output datasets/v1-labels --teacher stub --stub-file detections.json --dry-run 20
# 3. real teacher, labeling env interpreter (docs/environments.md)
<labeling-env>\Scripts\python.exe scripts/auto_label.py --dataset datasets/v1 --output datasets/v1-labels --teacher grounding_dino --dry-run 50
```
Open `datasets/v1-labels/gallery/index.html` to review, read `report.json` for class counts,
zero-detection rate, filtered boxes, small objects and the score histogram.

- Classes/prompts/thresholds: `ontology/traffic.json` (fixed ids: person, car, motorcycle, bus, truck, traffic light, zebra; two zebra prompts to compare).
- Labels are 5-column YOLO rows; empty file = background image. Sidecars keep score, prompt, teacher, checkpoint.
- Boxes dropped (and logged with a reason): non-finite, outside image (>2px), zero area, unknown prompt, below threshold, same-class NMS duplicates. No minimum-area filter, so small pedestrians are kept.
- `review_queue.jsonl`: corrupt image, teacher error, zero detection, low confidence, filtered boxes.
- Re-running resumes from sidecars; `--retry-failed` redoes errored images. Only `train` is labelled by default.

## Status
Tested: `python -m unittest test_auto_label` (offline stub, CPU).
GPU smoke (2026-10-13, RTX 4060, conda env `traffic-labeling`, torch 2.13.0+cu130): real Grounding DINO labelled
50 sampled images from a 2085-image Roboflow export with 0 errors. Checkpoint/config SHA-256 matched `requirements/smoke-assets.json`.
Observed with the default thresholds (box 0.30): most boxes score 0.3-0.4, there are spurious full-width low-score `zebra`
boxes, and distant cars are often missed. Thresholds and prompts are untuned; this is a smoke run, not a quality result.
Agreement with the (sparse, partly polygon) Roboflow labels is not accuracy: unlabeled objects count as "extra".
