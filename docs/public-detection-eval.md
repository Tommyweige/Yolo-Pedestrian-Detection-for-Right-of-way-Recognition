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

Next check: count ID switches / track fragments on our own clips (are the rule's pairing failures caused by fragmented
pedestrian tracks?).
Fine-tuning on iVS train (89k frames) could help night pedestrians and scooters, but its scooter boxes include the rider,
so the rule's rider test would have to change with it.
