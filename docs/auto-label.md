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
Tested: `python -m unittest test_auto_label` (offline stub, CPU). **Not tested:** the real Grounding DINO
teacher (no GPU smoke run for this ticket yet); its adapter is unverified until run in the labeling env.
