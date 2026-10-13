# Image-space yield candidates (`yield_image_rule.py`) — assistive analysis, not a legal finding

Why: the world-geometry pipeline (`yield_rules.py` + stabilisation) returned 0 events on a moving helmet-cam clip because it
solved a harder problem than needed. A yielding vehicle never enters the crosswalk while a pedestrian is there, so the rule is
just co-occurrence in the image: no calibration, no speed, no camera-motion compensation.

Pipeline (run each step in the env noted):
```powershell
# labeling env (transformers): per-frame crosswalk masks, GPU, ~0.25 s/frame at 720p
python scripts/segment_crosswalk.py --video clip.mp4 --model runtime/models/mask2former-mapillary --stride 3 --output out/seg
# modern env: detection + ByteTrack, GPU, no stabilisation (omit --reference-frame); saves boxes
python scripts/track_video.py --video clip.mp4 --weights runtime/smoke-assets/yolo26n.pt --output out/tracks
# any env with numpy/opencv
python yield_image_rule.py --masks out/seg/masks.npz --tracks out/tracks/raw_tracks.json --output out/events.jsonl
```
Segmentation model: facebook/mask2former-swin-large-mapillary-vistas-semantic @ 4772b6bf (safetensors SHA-256 72721afb...e2ea7).
Mapillary Vistas is CC BY-NC-SA: research/demo use only.

Event = a vehicle whose bottom edge is on the same crosswalk blob as a pedestrian (foot on it, or within 0.8 person heights of its
edge), horizontal gap <= 1.5 vehicle widths, for >= 0.2 s; motorcycle riders/occupants (person box inside a vehicle box) are
excluded; one event per vehicle track. `pedestrian_state` (on_crosswalk | waiting_at_edge) is stored so a stricter definition can
filter without re-running. The definition of "failure to yield" (does a pedestrian merely waiting count?) is still the user's call.

## Real-video result (helmet-cam night clip, 900 frames, 60 fps) — NOT a measured accuracy
- Near-crosswalk segmentation found the crosswalk in all 18 visually checked frames (even with vehicles/pedestrians occluding).
- 2 candidate events: f660 (11.0 s): a taxi drives across the crosswalk as the pedestrian steps onto its left end (matches the clip's
  topic). f352 (5.87 s): a white SUV (detected as "truck") crosses while the pedestrian stands at the left end of the crosswalk; whether
  that is a violation depends on the definition above and on the mask edge there.
- There are no human event labels, so misses (e.g. vehicles passing while the pedestrian is out of frame) cannot be counted and no
  precision/recall is claimed. One clip cannot support "better than" any other method.
