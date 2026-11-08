# Image-space yield candidates (`yield_image_rule.py`) — assistive analysis, not a legal finding

Why: the world-geometry pipeline (`yield_rules.py` + stabilisation) returned 0 events on a moving helmet-cam clip because it
solved a harder problem than needed. A yielding vehicle never enters the crosswalk while a pedestrian is there, so the rule is
just co-occurrence in the image: no calibration, no speed, no camera-motion compensation.

Pipeline (run each step in the env noted):
```powershell
# labeling env (transformers): per-frame crosswalk masks, GPU, ~0.25 s/frame at 720p
python scripts/segment_crosswalk.py --video clip.mp4 --model runtime/models/mask2former-mapillary --every-seconds 0.1 --output out/seg
# modern env: detection + ByteTrack, GPU, no stabilisation (omit --reference-frame); saves boxes
python scripts/track_video.py --video clip.mp4 --weights runtime/smoke-assets/yolo26x.pt --conf 0.15 --imgsz 1280 --output out/tracks
# any env with numpy/opencv; fps and frame size are read from --video
python yield_image_rule.py --video clip.mp4 --masks out/seg/masks.npz --tracks out/tracks/raw_tracks.json --output out/events.jsonl
# low-noise mode (the old one event per vehicle, 4.4 instead of 7.1 reports per clip on the labelled clips)
python yield_image_rule.py ... --one-event-per-vehicle
```
These are the commands `scripts/run_pipeline.py` and the GUI run; their flags and defaults (sampling interval, model
path, yolo26x, conf, imgsz) are built in one place, `yield_pipeline.segment_command` / `track_command` / `rule_command`.
Segmentation model: facebook/mask2former-swin-large-mapillary-vistas-semantic @ 4772b6bf (safetensors SHA-256 72721afb...e2ea7).
Mapillary Vistas is CC BY-NC-SA: research/demo use only.

Event = a vehicle whose bottom edge is on the same crosswalk blob as a pedestrian (foot on it, or within 0.8 person heights of its
edge), horizontal gap <= 3 vehicle widths (the law is ~3 m / one lane; 1.5 vehicle widths missed motorcycles, whose boxes are
under a metre wide), for >= 0.2 s. **The command line reports the candidate list by default** (user decision, 2026-10-31):
every (vehicle, pedestrian) contact, with repeats of one pedestrian inside that vehicle's pass merged by `contact_dedup`
(default `pair`). `--one-event-per-vehicle` switches back to the old construction, one event per vehicle (its earliest
qualifying pedestrian), which is kept as the low-noise mode and as the comparison row in `eval_constructions.py`.
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
their foot is at least 0.03 person heights inside the crosswalk mask at some frame of the contact with the vehicle (from its
start to 1 s after it ends), so stepping onto the crosswalk as a vehicle passes counts and standing still at the edge does not.
Judging the whole contact matters: a slow or late-stepping pedestrian reaches the mask 1.5-2 s after the vehicle's first frame
on it, and a window around the contact start threw those pairs away. The reported `time_s` is then the frame the pedestrian is
on the crosswalk (what a human labeller marks); `contact_frames` keeps the whole span. (Image motion cannot be used: the camera
moves.) This looks at frames after the contact, so it is an offline review tool. `--include-waiting` disables the filter.
The 0.03 depth is **chosen in-sample and sits on the edge of a narrow shelf, not in a flat valley**, and the choice depends on
the event construction, which is why the two constructions do not agree on it. On the 7 labelled clips, one event per vehicle
gives incident recall 11/15 at 0.05 and above (0.06 is already back at the old 0.1 numbers) and the same 12/15 with 6/9 manual
hits at both 0.03 and 0.04, so that construction cannot separate the two depths. The candidate-list construction (the default,
see below) does: it keeps its last withheld manual label and incident only at 0.03 (8/9, 14/15, 50 reports) and loses both at
0.04 (7/9, 13/15, 49 reports). 0.03 is therefore what the rule uses, with the risk stated: it is the edge of the shelf, not
its middle, and 0.05 and up falls off it. Tested at 0.02, 0.03, 0.04, 0.05, 0.06, 0.08 and 0.1; the gain window is
0.03-0.04 only. It must be re-validated on clips that did not take part in that sweep before the number can be called
anything but in-sample; the boundary tests pin 0.029 / 0.03 / 0.031 so a later move has to say what it costs.

## Two event constructions, one table (2026-10-31, in-sample)
How contacts become events is a construction choice, not a threshold, so both constructions are measured by the same script
(`scripts/eval_constructions.py`, 7 clips, tolerance 1.25 s, same scorer as `eval_rule_events.py`, one detect pass per depth
feeding every row below), and only these rows are quoted anywhere:

| construction | depth | events | after the pass collapse | manual | confirmed | incident recall | duplicate reports | precision |
|---|---|---|---|---|---|---|---|---|
| one event per vehicle | 0.10 | 28 | 28 | 5/9 | 7/8 | 11/15 | 0 | 0.429 |
| candidate list (`earliest`) | 0.10 | 53 | 31 | 5/9 | 7/8 | 11/15 | 0 | 0.387 |
| one event per vehicle | 0.04 | 31 | 31 | 6/9 | 7/8 | 12/15 | 0 | 0.419 |
| candidate list (`earliest`) | 0.04 | 72 | 35 | 6/9 | 7/8 | 12/15 | 0 | 0.371 |
| one event per vehicle | 0.03 | 31 | 31 | 6/9 | 7/8 | 12/15 | 0 | 0.419 |
| candidate list (`earliest`) | 0.03 | 74 | 36 | 7/9 | 7/8 | 13/15 | 0 | 0.389 |
| **candidate list (`pair`) — the default combination** | **0.03** | **74** | **50** | **8/9** | **8/8** | **14/15** | **14** | **0.320** |

- The candidate list (`report_every_contact`) reports every (vehicle, pedestrian) contact and then merges repeats of one
  pedestrian inside that vehicle's pass. A pass is the vehicle's contacts chained by report time within `contact_dedup_s`
  (1.25 s, the evaluation's own tolerance); "events" is what it emits before the merge and every other column is scored
  after it. Which contact of a pass survives is `contact_dedup`, measured in the next section.
- The default combination, and the one row to quote, is the last one: candidate list, `contact_dedup="pair"`, depth 0.03,
  **74 contacts -> 50 reports, manual 8/9, confirmed 8/8, incident recall 14/15, 14 duplicate reports, precision 0.320**.
  `scripts/eval_constructions.py` runs that combination by default; every other variant and depth stays selectable. This is
  also what the command line emits since 2026-10-31; the old construction is still there as `--one-event-per-vehicle`
  (`report_every_contact=False`), for the comparison row above and as a low-noise mode.
- The 0.04 row is kept only as the depth comparison: the same default at 0.04 is 72 -> 49, 7/9, 8/8, 13/15, 14, 0.306, so
  the two depths differ by exactly one manual label and one incident (clip 5's 21.667 s and its incident).
- Clip 5 is where the depth and the reported time are coupled: from 0.1 to 0.04 the same contacts report 2.7 s and 4.2 s
  earlier ("first frame on the crosswalk" arrives sooner once the shallow edge counts), which leaves the 8.9 s and 21.7 s
  labels 1.9 s and 1.6 s from the nearest report, outside the 1.25 s tolerance. At 0.03 the 21.667 s contact reports at
  22.8 s (1.13 s away) and that label is hit again; the 8.9 s label needs a different contact, see the next section.

## Product output: what the command line emits (2026-10-31, in-sample)
The default construction changed, so the CLI's own output changed. Counted by running `yield_image_rule.py` on all 7
labelled clips (same argv, output file read back), and scored by `scripts/eval_rule_events.py` at 1.25 s:

| CLI | events / 7 clips | per clip | manual | confirmed | incident recall | duplicate reports | precision |
|---|---|---|---|---|---|---|---|
| `--one-event-per-vehicle` (old default) | 31 | 4, 2, 4, 3, 4, 7, 7 | 6/9 | 7/8 | 12/15 | 0 | 0.419 |
| default (candidate list, `pair`, 0.03) | 50 | 4, 2, 16, 3, 7, 8, 10 | 8/9 | 8/8 | 14/15 | 14 | 0.320 |

- The cost is real and not uniform: 4.4 -> 7.1 events per clip on average, but one clip goes 4 -> 16. Those extra reports
  are mostly the same vehicle meeting several pedestrians, and they are the price of not dropping the pedestrian a labeller
  marked. The GUI shows them collapsed into one row per vehicle, so a reviewer does not see the growth, but anything that
  counts the events file sees it, and precision drops 0.419 -> 0.320 because the duplicates count as extra reports.
- To get the old output back, use `--one-event-per-vehicle` on the CLI, `report_every_contact=False` in `Params`, or
  `--params '{"report_every_contact": false}'` in `scripts/eval_rule_events.py`. `--contact-dedup earliest` gives a
  middle ground at this construction (36 reports, 7/9, 13/15, 0.389) but it drops a labelled crossing, so it is not the
  default.

## Which contact of a pass survives (2026-10-31, in-sample)
`Params.contact_dedup` decides which report stands for a vehicle's pass. The rule's own construction makes the same choice
implicitly (one event per vehicle = that vehicle's first qualifying contact), so this is not a candidate-list detail. Three
choices, same script, same scorer, both depths:

| variant | depth | raw | after collapse | manual | confirmed | incident recall | duplicate reports | precision | dropped by the collapse but a judged event |
|---|---|---|---|---|---|---|---|---|---|
| `earliest` | 0.10 | 53 | 31 | 5/9 | 7/8 | 11/15 | 0 | 0.387 | 13 contacts (A12/C1) on 6 judged events, 0 unreported |
| `on_crosswalk_longest` | 0.10 | 53 | 31 | 5/9 | 7/8 | 11/15 | 0 | 0.387 | 14 (A12/B1/C1) on 7, 0 unreported |
| `pair` | 0.10 | 53 | 38 | 5/9 | 8/8 | 11/15 | 7 | 0.342 | 8 (A7/C1) on 6, 0 unreported |
| `earliest` | 0.04 | 72 | 35 | 6/9 | 7/8 | 12/15 | 0 | 0.371 | 24 (A17/B1/C6) on 8, 1 unreported |
| `on_crosswalk_longest` | 0.04 | 72 | 35 | 6/9 | 7/8 | 12/15 | 0 | 0.371 | 25 (A17/B2/C6) on 9, 1 unreported |
| `pair` | 0.04 | 72 | 49 | 7/9 | 8/8 | 13/15 | 14 | 0.306 | 12 (A9/C3) on 7, 0 unreported |
| `earliest` | 0.03 | 74 | 36 | 7/9 | 7/8 | 13/15 | 0 | 0.389 | 24 (A17/B1/C6) on 8, 1 unreported |
| `on_crosswalk_longest` | 0.03 | 74 | 36 | 7/9 | 7/8 | 13/15 | 0 | 0.389 | 25 (A17/B2/C6) on 9, 1 unreported |
| **`pair` — default** | **0.03** | **74** | **50** | **8/9** | **8/8** | **14/15** | **14** | **0.320** | **12 (A9/C3) on 7, 0 unreported** |

The last column is measured against the hand-judged table (`verdicts_after.json`, 28 events, A15/B3/C6/D4): a dropped
contact counts when it names the same vehicle and pedestrian within 5 s of a judged event, or the same vehicle within the
scoring tolerance; "unreported" counts judged events that no surviving report matches any more.

- `on_crosswalk_longest` is the same rule as `earliest` in every scored column at every depth and discards one more judged
  contact (25 vs 24): the pedestrian already on the crosswalk is usually the earlier one of the pass, so preferring it adds
  a preference without changing a report. Not adopted.
- `pair` (only repeats of the same vehicle-pedestrian pair collapse) is the only variant that moves a label: it recovers
  clip 5's 8.9 s manual label at both depths (6/9 -> 7/9 at 0.04, 7/9 -> 8/9 at 0.03, one more incident each) and cuts the
  A contacts it discards from 17 to 9. It pays with one report per pedestrian of a pass: 14 duplicate reports at both
  depths, precision 0.371 -> 0.306 and 0.389 -> 0.320. One manual label and one incident for 14 extra reports. **Adopted as
  the default** (2026-10-31): a vehicle that meets two different pedestrians in one pass reports both, because the second
  pedestrian is the one the label can name (clip 5 at 8.2 s) and the collapse has no way to know which of them matters. The
  14 duplicate reports are the intended output of that choice, not a defect: they are the same vehicle reported against each
  pedestrian it actually obstructed, and a consumer that wants one report per vehicle can still collapse by vehicle and time.
- Where the collapse really loses a label: clip 5. Vehicle 27 touches p3 at 7.0 s (`on_crosswalk`) and p2 at 8.2 s
  (`waiting_at_edge`) 1.2 s apart, and the human label is 8.9 s. `earliest` keeps 7.0 s and drops 8.2 s, leaving the label
  1.9 s from the only report; it is the pass collapse, not detection (the contact exists at 8.2 s) and not the depth (0.03
  has the same two contacts there). `pair` keeps both, and the label is hit 0.7 s away.
- Where a dropped contact costs nothing: clip 3. v2299 has contacts at 41.333 s and 42.2 s; `earliest` keeps 41.333 s and
  drops the 42.2 s ones, but the labels at 41.2/41.867/42.2 s are one incident inside the tolerance and 41.333 s reports it.
  Most of the 24 dropped-but-judged contacts are like this: they sit on a judged event that another report of the pass still
  covers.
- The one judged event the collapse leaves unreported (`earliest` and `on_crosswalk_longest`, both depths) is clip 7's
  27.68 s, verdict B: vehicle 3586 touches p3496 at 26.08 s and p3570 at 26.8 s, `earliest` keeps 26.08 s and the judged B
  ends up 1.6 s from any report. `pair` reports both.
- Consequence, and how big it is: the defaults are now candidate list + `pair` + depth 0.03 (74 contacts -> 50 reports,
  8/9, 8/8, 14/15, 14 duplicate reports, 0.320) against 31 events, 6/9, 7/8, 12/15, 0, 0.419 for one event per vehicle at the
  same depth. The whole difference is clip 5's 8.9 s and 21.667 s: two manual labels and two incidents out of 15, i.e. one
  label per case, one-event scale. On 15 incidents that is inside the noise this table can resolve - a single re-labelled
  event moves it as much as this whole change does. Do not quote it as a large improvement, and re-check both choices on
  clips that did not take part in this table: `pair` and 0.03 were both picked by looking at these 7 clips.

## Comparing the constructions
- **The counts are not the same thing and must not be compared across constructions.** The candidate list's "raw" column
  counts (vehicle, pedestrian) contacts; one event per vehicle counts vehicles. The same pass is 1 event in one row and
  2-4 contacts in the other, so raw counts, "after collapse" counts and precision are all construction-specific: 0.429
  (28 events) and 0.320 (50 reports) do not say one construction is worse, they say the denominators are different units.
  Compare precision only inside one construction - e.g. `pair` 0.320 against `earliest` 0.389 on the same 74 contacts.
- Manual labels, confirmed events and incident recall are counted per judged event, so those three are comparable across
  constructions. Duplicate reports are not - the old construction reports one event per vehicle, so it cannot produce one.
  Within the candidate list they are the report the `pair` trade is about. On the same 28 judged events, one event per
  vehicle at 0.03 gets 6/9 manual and 12/15; the candidate list with `pair` gets 8/9 and 14/15.
- Any future table (including the clip-classifier route's) has to be scored by `scripts/eval_constructions.py` with its
  dedup stated. A number quoted without its construction, dedup and depth is not comparable with any row here.

**Pipeline modes (2026-11-07).** `scripts/run_pipeline.py --mode strict` (the default) runs the rule as is. `--mode review`
adds `--include-waiting`. With the candidate-list construction and depth 0.03, strict finds 14/17 with 51 candidates (one mask per 0.1 s; 50 with the earlier every-3rd-frame stride). Review
finds 15/17 with 121 candidates: one more hit for 71 more reports, about one every 2 s of video, more than a human can check.
So review is opt-in. (Before the candidate list, under one event per vehicle at depth 0.1, review was 12/17 with 39 candidates
and strict 8/17 with 17, which is why review was briefly the default on its branch.)
With the filter off and `--one-event-per-vehicle`, a vehicle that meets both a waiting and a crossing pedestrian reports the
crossing pair, not whichever came first. Each event line carries `"mode": "review"|"strict"`. The existing
`pedestrian_entered_crosswalk` field becomes the 優先 column of `candidates.csv` (高 = entered).

## Real-video result (helmet-cam night clip, 900 frames, 60 fps) — NOT a measured accuracy
- Near-crosswalk segmentation found the crosswalk in all 18 visually checked frames (even with vehicles/pedestrians occluding).
- With the user's definition: 1 event, f660 (11.0 s): a taxi drives across the crosswalk as the pedestrian steps onto its left end.
- With `--include-waiting`: also f352 (5.87 s): a white SUV (detected as "truck") crosses while the pedestrian stands motionless
  at the left edge of the crosswalk (foot 0.02-0.04 person heights inside the mask, so the 0.03 depth cuts through the middle of
  that range). Never re-run at the current depth, so this observation is unresolved: it may now count by default, or still fall
  under the threshold, and either way it is one unlabelled frame range.
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

## Rule-only evaluation and two failed experiments (2026-11-07) — in-sample, 7 clips, 17 labels
`python scripts/eval_rule.py --geometry image crosswalk [--param NAME=VALUE ...]` re-runs the rule on the stored masks/tracks and scores
it against `labels/` (no GPU, ~30 s per geometry). Baseline at HEAD: image geometry P 0.47 / R 0.47 / F1 0.47 (8 of 17; 9 false alarms,
8 of them rejected by the human reviewer); crosswalk geometry P 0.38 / R 0.53 / F1 0.44.
`diagnose_misses.py` on the 9 misses: the crosswalk mask is never the blocker; the misses stop at the vehicle-person pairing or at the
"pedestrian stepped onto the crosswalk" filter, whose 0.1-person-height depth test is impossible on a far or foreshortened band
(a band 0.2 person heights thick has no pixel 0.1 from both edges).
Tried, both behind default-off `Params` flags: `adaptive_depth` (scale that depth need to the local band) and `surround_excludes_body`
(foot-surround test ignores the person's own box). `surround_excludes_body`: F1 0.42, worse (more false alarms). `adaptive_depth`:
F1 0.51 at fraction 0.5 (one extra hit), but 0.44 at 0.3 and 0.47 at 0.7/1.0 — a knife edge on one event, so NOT adopted.
Remaining false alarms are mostly motorcycles at night near a pedestrian who is crossing; the reviewer rejected them although they look
plausible, so the event definition (not only the code) decides what counts. No claim of improved accuracy: needs unseen clips and clips with no event.

## Honest out-of-sample attempts (2026-11-07): none beat the default rule

Every number here is leave-one-video-out (LOVO): a choice is made without the clip it is scored on. 17 labelled events in 7 clips,
so one event more or less moves F1 by about 0.06. Baseline: the default rule, F1 0.471 (P 17, TP 8). That baseline is itself
optimistic, since the defaults were tuned on these clips earlier.

| attempt | tool | LOVO F1 |
|---|---|---|
| default rule (no re-selection) | `scripts/eval_rule.py` | 0.471 |
| loose candidates + logistic scorer, 21 trajectory features, l2 0.3–30 | `scripts/learn_yield.py` | 0.24–0.28 |
| stacking: `rule_fires` + entered, ped_on, v_speed (fixed a priori, one run) | ad hoc on `datasets/learn-cache-v1.json` | 0.235 |
| rule-parameter grid (27 configs: entry_window_s × lateral_vehicle_widths × min_inside_person_heights), nested selection | `scripts/tune_rule_lovo.py` | 0.465 |

The learned scorer ranks held-out candidates no better than noise (accepting every loose candidate gives 0.306), so that line was
dropped. The grid's best in-sample config (`lateral_vehicle_widths` 2.0, `min_inside_person_heights` 0.0) reaches 0.583 on all 7
clips, and most folds pick it, yet the held-out clips gain nothing: the extra hits it finds on some clips are paid for by extra false
alarms on others. Defaults are therefore unchanged. What would move this: more labelled clips (especially clips with no event),
not more tuning on these seven.

Note (2026-11-08): every number in this section and the one above was scored at a fixed 1.0 s tolerance (the `--tolerance`
default of `eval_rule.py`, `tune_rule_lovo.py` and `learn_yield.py`), not ADR 0003's 1.25 s.
The 27-config grid also lacked the current defaults (`lateral_vehicle_widths` 3.0, `min_inside_person_heights` 0.03);
`tune_rule_lovo.py`'s grid now holds them (48 configs), so its numbers are not comparable with the 0.465 row.
