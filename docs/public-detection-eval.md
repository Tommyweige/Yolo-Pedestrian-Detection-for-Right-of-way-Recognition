# YOLO26x on the iVS test set (2026-11-07)

Question: how much does the pipeline's detector miss on Taiwanese dashcam footage, independent of our 7 clips?

- Data: iVS-Dataset (NYCU PAIR Labs), `ivslab_test` + `ivslab_test2`, 5,400 frames, 1920x1080, Taiwanese dashcams
  (2015-2016, day/night/rain). Classes: vehicle, scooter, pedestrian, bike (no bike boxes in these splits). Stored outside
  the repo (`E:\Learning\public-datasets`), not under version control.
- Detector exactly as in the pipeline: `runtime/smoke-assets/yolo26x.pt`, conf 0.15, imgsz 1280. Greedy matching at IoU 0.5.
- Tool: `scripts/eval_public_detection.py` (the BDD100K path in it is unused: that archive is the per-image detection set,
  not the tracking set, so it cannot test tracking).
- Per-frame numbers. A track spans many frames, so per-frame misses mostly show up as fragmented tracks, not as a missing object.

| class | labelled | recall | by box height / frame height (small <5%, medium 5-15%, large) |
|---|---|---|---|
| vehicle (car/bus/truck) | 25,494 | 0.87 | small 0.82, medium 0.82, large 0.97 |
| pedestrian | 2,004 | 0.72 | medium 0.65, large 0.80 |
| scooter, raw | 7,054 | 0.19 | not comparable: see below |
| scooter, YOLO motorcycle box merged with its rider box | 7,054 | 0.45 (0.47 at IoU 0.3) | |

Reading it:

- **Scooter box definition differs.** iVS draws one box around scooter + rider; COCO/YOLO boxes the vehicle only and the rider
  as a separate `person`. Most of the raw 0.19 is that mismatch, and 5,481 of the 9,987 "unmatched" person detections are
  riders sitting inside an iVS scooter box (iVS does not label riders). Precision numbers from the raw run are therefore
  meaningless for person and scooter.
- **Genuine scooter misses remain.** 2,559 scooters (36%) have no motorcycle detection and no rider detection at all: mostly
  small, far, or at night. In another 1,290 (18%), the rider is detected as `person` but the scooter is not. That case matters
  for the yield rule: it decides "rider, not pedestrian" from overlap with a motorcycle box, so a lone rider box can be taken
  for a pedestrian. Unverified link: most of the rule's human-rejected false alarms involve motorcycles at night.
- **Missed pedestrians** (564) are mostly dark night scenes, motion blur, and people half-hidden behind parked scooters on
  the sidewalk (visual check of a sample). 35% of them have a detection at IoU 0.3-0.5 (a loose box rather than none).

## Follow-up: what the rule's false alarms actually are (2026-11-07)

The rule's 8 unmatched events (default params, image geometry, all in two clips, all human-rejected) were checked frame
by frame. The "rider without a motorcycle box" hypothesis explains only **1 of 8**:

| cause | count |
|---|---|
| zoomed slow-motion **replay** of an incident already shown (the compilation videos repeat the clip enlarged) | 3 |
| real pedestrian on/at the crosswalk, vehicle nearby, but the reviewer judged it not a failure to yield | 2 |
| person standing at the curb or frame edge beside waiting scooters, not crossing | 2 |
| rider + scooter boxed as one `person`, no motorcycle box (the iVS-predicted failure) | 1 |

Vehicle box motion around the event (shift and scale over 0.5 s) does not separate hits from false alarms; the camera is
moving too. Replays are an artifact of the YouTube compilations, not of real dashcam use: they inflate false alarms
here but would not occur in deployment. Fine-tuning the detector for scooters would remove at most one false alarm.

## Follow-up: track fragments and the rule's misses (2026-11-07)

Tracks on our 7 clips are very fragmented: the median person track is 7-24 frames (under 0.5 s on most clips), and a
simple linker (successor starts within 0.5 s, within one box height of the end, similar size) finds 297 person and 402
vehicle links. **Stitching those links and re-running the rule changes nothing** (P 17 / TP 8 / L 17, F1 0.471 both
ways), so fragmentation is not what limits it.

Where each of the 9 missed labels drops out (rule stages checked frame by frame over ±1.5 s, then visually):

| stage | misses |
|---|---|
| pedestrian is at the crosswalk but classed `waiting_at_edge` (foot at or just past the mask edge, or on a thin far band), so the crossing filter drops it | 5 |
| person and vehicle both on the crosswalk but never paired (lateral gap) | 1 |
| pair seen in only 1 frame (shorter than `min_seconds`) | 1 |
| event found but 1.5 s after the label (outside the ±1 s match) | 1 |
| person and vehicle rarely detected on the crosswalk at all | 1 |

The biggest lever is therefore the on-crosswalk test at the mask edge, which is also what decides the "curb stander"
false alarms above: loosening it trades misses for false alarms (the nested-LOVO grid over `min_inside_person_heights`
found no net gain). Two structural variants tried after seeing the misses, in-sample only:

| variant | P / TP | F1 |
|---|---|---|
| default | 17 / 8 | 0.471 |
| `adaptive_depth=True` (existing option: depth need scaled to a thin band) | 18 / 9 | 0.514 |
| crossing test over the whole contact, not just ±1 s of first contact | 21 / 9 | 0.474 |
| both | 22 / 9 | 0.462 |

`adaptive_depth` gains one event in one clip; under nested LOVO that clip's fold falls back to the default, so there is
no out-of-sample evidence for it. Not adopted.

## Follow-up: does the pedestrian's trajectory separate edge cases? (2026-11-07)

Question before building a trajectory simulator: can motion tell a labelled "edge" pedestrian (rule says waiting) from
waiting candidates nobody labelled? Candidates = rule events with `require_crossing=False`; per candidate, over ±1.5 s of
the pedestrian track: foot depth into the mask (negative = outside, in person heights), its trend, and foot speed.

| group | n | deepest foot, median (p25-p75) | depth trend /s | speed, person heights/s |
|---|---|---|---|---|
| hit (entered) | 5 | 0.11 (0.08-0.11) | 0.05 | 2.9 |
| false alarm (entered, unlabelled) | 8 | 0.08 (0.07-0.10) | 0.03 | 2.4 |
| labelled, rule says waiting | 11 | 0.03 (-0.09-0.04) | -0.03 | 4.5 |
| waiting, unlabelled | 15 | -0.20 (-0.25 to -0.06) | -0.01 | 4.0 |

- **Motion does not separate them**: depth trend and speed overlap completely. "Is the pedestrian moving toward the
  crosswalk" carries no signal at this data size, so a trajectory model trained on simulated motion has nothing to learn here.
- **Only the foot position does, partly**: labelled edge cases reach the mask edge (about 0), unlabelled waiters stay
  about 0.2 person heights outside, but 4 of 15 waiters sit in the same band.
- Counting a foot up to 0.05 person heights outside the mask as on the crosswalk: in-sample F1 0.512 (P 26 / TP 11), but
  nested LOVO over margins {0, 0.05, 0.1} gives **0.462**, below the default 0.471: two of the three gained events come
  from one clip and vanish when it is held out. Not adopted.

So the edge decision is ambiguous given the current mask: whether the gap is mask error or labelling judgement can only be
told by measuring the mask edge against hand-marked crosswalk edges.
Fine-tuning on iVS train (89k frames) could help night pedestrians and scooters, but its scooter boxes include the rider,
so the rule's rider test would have to change with it.
