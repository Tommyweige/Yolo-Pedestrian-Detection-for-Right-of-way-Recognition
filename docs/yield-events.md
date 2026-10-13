# Yield-to-pedestrian events (`yield_rules.py`) — assistive analysis, not a legal finding

Pipeline: `scripts/track_video.py` (YOLO26 + ByteTrack, camera-motion stabilisation) -> `yield_rules.py`
(road-plane geometry + event state machine) -> `scripts/export_evidence.py` / `scripts/eval_events.py`.
Design and open items: issue #17. Rule module is torch-free; run tests with any env having numpy:
`python -m unittest test_yield_rules test_eval_events`.

```powershell
$py = "$env:USERPROFILE\anaconda3\envs\traffic-modern\python.exe"
& $py scripts/track_video.py --video clip.mp4 --weights runtime/smoke-assets/yolo26n.pt --output out --reference-frame 450
# write out/scene.json: crosswalk_image (4 corners in the REFERENCE frame: near-left, near-right, far-right, far-left),
# length_m (along the crosswalk), depth_m (along the road), vehicle_direction 0 = infer per vehicle
& $py yield_rules.py --scene out/scene.json --tracks out/tracks.json --output out/events.jsonl --fps 60
& $py scripts/export_evidence.py --video clip.mp4 --events out/events.jsonl --output out/clips
& $py scripts/eval_events.py --predicted out/events.jsonl --truth labels.jsonl --tolerance 1.0   # needs human labels
```

## What is verified
Synthetic trajectories (13 tests, mutation-checked): drive-through, stop-and-yield, kerb pedestrian, far pedestrian,
two-way traffic, parked vehicle, short track, track gaps, wild stabilisation points; plus the evaluation maths.

## Real-video check (user's helmet-camera clip, 900 frames, 60 fps) — NOT a result
- Camera moves, so every point is mapped into one reference frame with optical-flow homographies on road features.
  Overlaying the reference crosswalk on other frames: good within ~150 frames of the reference, visibly drifting
  (tens of pixels) further away. Visual check only; no numeric error.
- Crosswalk size (8 m x 3.5 m) is a guess from stripe spacing; metric speeds/distances are therefore approximate.
- YOLO26n (COCO) at night fragments tracks (the one pedestrian became several IDs; only 37 of 262 vehicle tracks
  had >= 30 usable points) and travel direction is noisy while the rider moves.
- Result: 0 events. That is a limit of detection/tracking/stabilisation quality, not evidence that nothing happened.
  There is no human event label for this clip, so no precision/recall is claimed.

## Not done
Red-light module; a stronger detector (YOLO26s/m fine-tuned incl. `person`), track re-linking for fragmented IDs,
automatic crosswalk detection for moving cameras, and human event labels for evaluation.
