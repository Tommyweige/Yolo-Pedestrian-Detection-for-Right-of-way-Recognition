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
edge), horizontal gap <= 3 vehicle widths (the law is ~3 m / one lane; 1.5 vehicle widths missed motorcycles, whose boxes are
under a metre wide), for >= 0.2 s; one event per vehicle track.
Motorcycle riders are not pedestrians (`rider_frames`): a frame counts as riding when the person box sits 50% inside a
motorcycle box inflated by 15% of the person's height, dropouts up to 0.25 s between riding frames are filled in, and the
track needs 5 such frames. It is decided per person and per moment, not per frame (a rider's box jitters out of the bike box
and was reported as a pedestrian) and not per person (someone who walks across the road and rides a scooter in the same clip
is a pedestrian while walking). One vehicle tracked twice is one vehicle (`merge_duplicate_tracks`: coexisting tracks of the
same kind with median box IoU >= 0.45, or a new id starting within 0.5 s where the old one ended with IoU >= 0.45).
**Evaluation is fixed before the rule is tuned** (`docs/adr/0003-fixed-rule-evaluation.md`,
`scripts/eval_rule_events.py`): the matching tolerance is derived from the labels-vs-predictions residual distribution
(1.25 s on the current 7 clips), labels closer than the tolerance are one incident, and duplicate reports are measured on the
tracks, not on vehicle ids. Numbers from the labelled clips are in-sample.
**Definition (user decision, 2026-10-13): a pedestrian merely waiting at the edge does not count.** A pedestrian counts only if
their foot is at least 0.04 person heights inside the crosswalk mask at some frame of the contact with the vehicle (from its
start to 1 s after it ends), so stepping onto the crosswalk as a vehicle passes counts and standing still at the edge does not.
Judging the whole contact matters: a slow or late-stepping pedestrian reaches the mask 1.5-2 s after the vehicle's first frame
on it, and a window around the contact start threw those pairs away. The reported `time_s` is then the frame the pedestrian is
on the crosswalk (what a human labeller marks); `contact_frames` keeps the whole span. (Image motion cannot be used: the camera
moves.) This looks at frames after the contact, so it is an offline review tool. `--include-waiting` disables the filter.
The 0.04 depth is **chosen in-sample and sits on a narrow shelf between two cliffs, not in a flat valley**: on the 7 labelled
clips, incident recall is 11/15 at 0.05 and above (0.06 is already back at the old 0.1 numbers) but 12/15 at 0.03-0.04, where
manual label hits go 5/9 -> 6/9, confirmed stays 7/8, duplicate reports stay 0 and precision goes 0.429 -> 0.419 with three
more events. 0.04 is the middle of that shelf and 0.03 is its edge, so 0.04 is what the rule uses. It must be re-validated on
clips that did not take part in that sweep before the number can be called anything but in-sample. Tested at 0.02, 0.03, 0.04,
0.05, 0.06, 0.08 and 0.1; the gain window is 0.03-0.04 only.

## Real-video result (helmet-cam night clip, 900 frames, 60 fps) — NOT a measured accuracy
- Near-crosswalk segmentation found the crosswalk in all 18 visually checked frames (even with vehicles/pedestrians occluding).
- With the user's definition: 1 event, f660 (11.0 s): a taxi drives across the crosswalk as the pedestrian steps onto its left end.
- With `--include-waiting`: also f352 (5.87 s): a white SUV (detected as "truck") crosses while the pedestrian stands motionless
  at the left edge of the crosswalk (foot 0.02-0.04 person heights inside the mask). Excluded by default *at the 0.1 depth the
  run used*; at 0.04 a foot 0.04 or deeper inside counts, so this observation (a range straddling the threshold) needs
  re-running before it is quoted again.
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
