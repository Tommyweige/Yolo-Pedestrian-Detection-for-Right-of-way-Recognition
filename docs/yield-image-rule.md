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

## Crosswalk coordinates: `--geometry crosswalk` (opt-in; the default stays `image`)

`crosswalk_space.py` fits a quadrilateral to each crosswalk mask blob and maps it to the unit square, so a point gets
`(L, W)`: L along the crosswalk (the way pedestrians walk), W across it, 0 and 1 on the two long edges. `--geometry
crosswalk` judges "pedestrian on the crosswalk", "pedestrian waiting at the edge" and "vehicle touching the crosswalk" in
those coordinates instead of in pixel strips and person heights, so a foreshortened crosswalk no longer needs a
person-height fudge. Pieces of one crosswalk cut apart by occlusion are merged; a blob too small to fit gives no space and
the component falls back to the image geometry. `scripts/run_pipeline.py --geometry crosswalk` runs the same pipeline this
way.

Decision in crosswalk coordinates: a foot with `cw_on_margin <= W <= 1 - cw_on_margin` is on the crosswalk, a foot up to
`cw_wait_margin` outside the band is waiting at the edge, and a vehicle whose ground point (bottom centre of its box) is
within `cw_vehicle_margin` of the band touches it; the lateral gap is compared with the vehicle's own box extent mapped
into the same space. L is bounded by the quad's ends. A crosswalk cut by the image border has no real end there, so it
claims only points inside the fitted quadrilateral: what continues past the picture cannot be checked.

### How the three margins were chosen (2026-10-30)

Against the crosswalk mask itself — a box's ground point is "on the crosswalk" when that pixel is inside the mask, the
only reference available without human labels — on the clip below:

`python scripts/measure_crosswalk_margins.py --video clip.mp4 --masks out/seg/masks.npz --tracks out/tracks/raw_tracks.json`

| margin | swept | what changed | chosen |
| --- | --- | --- | --- |
| `cw_on_margin` | 0.0 / 0.02 / 0.05 / 0.08 / 0.12 / 0.2 | pedestrian precision 0.62 / 0.62 / 0.61 / 0.60 / 0.62 / 0.82, recall 0.40 / 0.38 / 0.33 / 0.27 / 0.24 / 0.20; 2 events until 0.2 drops one | 0.05 |
| `cw_vehicle_margin` | 0.0 / 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 | vehicle precision 0.71 / 0.69 / 0.69 / 0.63 / 0.57 / 0.53 / 0.50, recall 0.45 throughout; events unchanged | 0.02 |
| `cw_wait_margin` | 0.1 / 0.25 / 0.5 / 0.75 / 1.0 | events unchanged to 0.5, `--include-waiting` gains a pair at 0.75 | 0.5 (widest value in the stable range) |

`cw_on_margin` also has to stay above the foot-point jitter (3 sigma 0.013 here) and is the image rule's 0.1 person heights
expressed in crosswalk widths: 0.17 m of a 3-4 m wide crossing. The margins are per camera — perspective, mask quality and
box size differ per clip — so rerun the script on a new clip instead of trusting these numbers there.

### One real clip, measured (no human labels, so no accuracy claim)

Clip: Wikimedia Commons `DiagonalCrosswalkYongeDundas.webm` (CC-licensed real footage of the Yonge/Dundas diagonal
crossing, 1920x1080, 23.98 fps), frames 960-1317 (14.9 s) of it, chosen because vehicles cross the visible foreground
crosswalk there. Not the project's own footage: on the machine used (aarch64, no GPU, no `runtime/models`) the project's
own clips were not present, so this is a public clip used to get a real-image comparison at all. Masks from the pinned
Mask2Former (120 frames at stride 3, CPU, 17.1 s/frame), tracks from `scripts/track_video.py --weights yolo26n.pt
--device cpu --imgsz 1280`. 358 frames, 219 vehicle and 428 pedestrian box-ground-points on the mask.

Judging "on the crosswalk" from the box's ground point against the mask pixel under it:

| decision | geometry | precision | recall |
| --- | --- | --- | --- |
| vehicle | `image` | 0.69 | 0.54 |
| vehicle | `crosswalk`, before the border fix below | 0.41 | 0.56 |
| vehicle | `crosswalk`, as merged (fix + margins above) | 0.69 | 0.45 |
| pedestrian | `image` | 0.87 | 0.25 |
| pedestrian | `crosswalk`, before the border fix | 0.23 | 0.32 |
| pedestrian | `crosswalk`, as merged | 0.61 | 0.33 |

Whole-rule events (one per vehicle, crossing filter on): `image` 4, `crosswalk` 9 before the fix and 2 after it; the two
geometries share one of the four (vehicle 4787 at f1038/f1040). Per frame, 31 person-vehicle pairs agree, 197 are `image`
only and 89 `crosswalk` only (before the fix: 48 / 180 / 429). So on this clip the crosswalk geometry does not beat the
image geometry: it calls deeper feet on more often (pedestrian recall 0.33 vs 0.25) at lower precision, and calls fewer
vehicles on.

Fixed on the way: a clipped crosswalk used to claim any point within half its longest edge, which put 42% of the vehicles
it called "on the crosswalk" on ground the mask does not cover (e.g. f1131: a car in the intersection 135 px past the
visible piece). It now claims only points inside the fitted quad, which is what took 9 events down to 2.

Measured weakness that is not fixed: the fitted quad is not stable while the mask flickers. In those 120 mask frames the
blob splits into 1, 2, 3 or 4 components in 99 / 184 / 72 / 3 of them, and a fixed image point on the crossing reads W
between 0.32 and 0.80 (sd ~0.1 widths) across frames. That drift, not the margins, sets the noise floor: a pedestrian
standing near a long edge will flicker between "on" and "waiting" whatever `cw_on_margin` is. Fitting one quad per shot
(or smoothing the mask over time) would be the next step.

Still not measured: no human event labels for this or any available clip, so nothing here is a precision/recall on
events; one clip, static camera, daytime, a fairly regular crosswalk — the cases the geometry was written for (strong
perspective, a thin far strip, a turning camera) are covered by the synthetic tests only. The project's own helmet-cam and
compilation clips still need to be run through `scripts/measure_crosswalk_margins.py` on a machine that has them.
