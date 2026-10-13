# YOLO26n fine-tune v1 (2026-10-13) — measured, with caveats

Pipeline: `dataset_ingest.py` -> `auto_label.py` (Grounding DINO, GPU) -> `build_training_set.py` -> `train_yolo26.py` (GPU, AMP)
-> `eval_detector.py`. Base weights: official yolo26n.pt, SHA-256 9b09cc8b...fad4fef. 60 epochs, 640 px, batch 16, RTX 4060.

## Training data (`datasets/train-v1`, not committed)
- 4151 train images: 1667 from the user's Roboflow project (human labels; zebra human-only) + 3842 teacher boxes (score >= 0.45,
  only where no human box exists) + 2484 COCO val2017 images as replay. 710 val images: 210 Roboflow val + 500 COCO holdout.
- The Roboflow images are browser screenshots of street-view imagery (browser chrome, logos, black borders), not dashcam frames.
  Their copyright is not covered by the CC BY 4.0 label licence; do not redistribute them.

## Detector metrics (AP50, same val images and labels for both models)
| class | Roboflow val: base -> tuned | COCO holdout: base -> tuned |
|---|---|---|
| person | - (not labelled) | 0.765 -> 0.699 |
| car | 0.581 -> 0.781 | 0.643 -> 0.550 |
| motorcycle | 0.273 -> 0.751 | 0.729 -> 0.483 |
| bus | 0.345 -> 0.472 | 0.805 -> 0.684 |
| truck | - | 0.392 -> 0.313 |
| traffic light | 0.166 -> 0.792 | 0.368 -> 0.231 |
| zebra | - -> 0.869 | - |
Caveats: Roboflow val labels are sparse (unlabelled objects count as false positives), and its val images come from the same
street-view source as train (possible near-duplicate leakage), so the gains there are likely inflated. The COCO regression is
real forgetting of general ability (only 2484 replay images). The user's 208-image test split has not been used.

## Real-video check (helmet-cam night clip, 4 key frames, visual only)
- Pedestrian and taxi are detected well by both models. The tuned model detected the black car near f540 only as a wheel-sized
  "motorcycle" (baseline boxed the whole car): a regression for near vehicles.
- The tuned zebra class did NOT find the near crosswalk in this clip (one oversized box at f420, nothing at f540/f660; only the far
  crosswalk was found). The image-space yield rule depends on exactly that near crosswalk, so it is not yet viable on this footage.
- Tracking: fewer fragmented person tracks (159 -> 78) but slightly fewer long vehicle tracks (27 -> 23).

## Next options
Crosswalk segmentation from a model trained on dashcam-like data (e.g. a Mapillary-Vistas-pretrained segmenter), more night
dashcam training images, more replay data against forgetting, then re-run this check on ~20 frames.
