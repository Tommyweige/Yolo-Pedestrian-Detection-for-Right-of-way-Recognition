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
excluded; one event per vehicle track.
**Definition (user decision, 2026-10-13): a pedestrian merely waiting at the edge does not count.** A pedestrian counts only if
their foot is at least 0.1 person heights inside the crosswalk mask at some frame within 1 s of the contact, so stepping onto the
crosswalk as a vehicle passes counts and standing still at the edge does not. (Image motion cannot be used: the camera moves.)
This looks at frames after the contact, so it is an offline review tool. `--include-waiting` disables the filter. The 0.1 threshold
was chosen between the only two pedestrians seen (0.03 and 0.25 person heights deep): thin evidence, to be revisited with more clips.

## Real-video result (helmet-cam night clip, 900 frames, 60 fps) — NOT a measured accuracy
- Near-crosswalk segmentation found the crosswalk in all 18 visually checked frames (even with vehicles/pedestrians occluding).
- With the user's definition: 1 event, f660 (11.0 s): a taxi drives across the crosswalk as the pedestrian steps onto its left end.
- With `--include-waiting`: also f352 (5.87 s): a white SUV (detected as "truck") crosses while the pedestrian stands motionless
  at the left edge of the crosswalk (foot 0.02-0.04 person heights inside the mask). Excluded by default.
- There are no human event labels, so misses (e.g. vehicles passing while the pedestrian is out of frame) cannot be counted and no
  precision/recall is claimed. One clip cannot support "better than" any other method.

## Turning camera (case A: the camera car turns, judging OTHER vehicles)
The crosswalk is no longer horizontal in the image. Segmentation, "same crosswalk blob", "foot depth inside the mask" and the
rider exclusion are orientation-free. The pedestrian-to-vehicle gap used to be measured along the image x axis; it is now measured
along the crosswalk's own principal axis (`crosswalk_axis`, PCA of the mask blob; falls back to horizontal for non-elongated blobs),
and the vehicle's size is its box extent projected on that axis.
Verified with synthetic horizontal / vertical / 45-degree crosswalk masks (15 tests). The vertical case fails under the old
horizontal rule; the 45-degree case passes under both, so it does not separate them. Real footage of a sharp turn has NOT been tested
(the clip used so far only has head-turn rotation). Still assumed: the vehicle's ground contact line is roughly the box bottom, and
the picture is not rolled by much; the pedestrian-entry filter still looks 1 s past the contact (offline only).
Case B (judging whether the camera car itself yielded) needs the camera car's own motion and is not implemented.

## First measurement on human labels (7 compilation clips, 13 labelled events, 2026-10-14) — in-sample, not a result
The first run used fps 60 / 1920x1080 for every clip although the clips are 25, 30 and 59.94 fps (one is 720p): event times were wrong
and thresholds mis-scaled. `--video` now reads fps/size from the file and the CLI refuses to guess. Even so the first corrected run found
1 of 13. `scripts/diagnose_misses.py` showed crosswalk and person detection were never the blocker; the blockers were
(1) "vehicle on the crosswalk" (the segmentation never labels the crosswalk under the vehicle itself) and
(2) the pedestrian's own pixels breaking the mask under their feet, so a person mid-crosswalk was "waiting at the edge", and
(3) a person in front of a bus counted as a rider. All three are fixed (rider exclusion now motorcycles only).
After the fixes: 2 of 13 matched, 9 predicted, 7 not matching a label (they may be real events the labeller did not mark: review them
in the labeler before calling them false alarms). The fixes were chosen by looking at these same 13 events, so this is in-sample;
a fair number needs new clips. A bigger detector (yolo26x) did not change recall.
