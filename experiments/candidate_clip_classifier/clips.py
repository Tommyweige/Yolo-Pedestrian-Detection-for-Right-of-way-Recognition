"""Cut each sample into the thesis's input: 9 frames x 224 x 224 x (RGB + 1 binary spatial-prior channel).

Window: 9 frames evenly over the event time +-1.6 s (0.4 s apart). The rule and the labeller disagree on the moment by
up to the 1.25 s tolerance, so +-1.6 s keeps the labelled moment inside a window centred on a candidate, and a car
crossing a crosswalk takes ~2-3 s, so the window sees it approach, pass and leave.

Crop: one fixed square around the candidate pair (vehicle and pedestrian boxes over all 9 frames, x1.5, at least 224
source px), not the full frame: at 1920 -> 224 a pedestrian would be ~10 px tall. Fixed over the window so the motion
of the pair stays in the pixels.

Prior channel (binary, like the thesis): crosswalk mask | vehicle box | dilated pedestrian box, per frame.
- Crosswalk: kept sharp. The decision is whether the vehicle is on it, i.e. its boundary is the evidence (the thesis
  keeps the weapon sharp for the same reason).
- Vehicle box: sharp, for the same reason: its wheel line against the crosswalk edge is what matters.
- Pedestrian box: dilated by 0.5 x its own height on every side (the thesis dilates people by a fixed 50 px; a fixed
  pixel count means a different real distance at every camera distance, so it scales with the person). The dilated
  region is the space in front of and around the pedestrian that a yielding vehicle must stay out of.
- Only the candidate pair, not every object: the prior says which vehicle and which pedestrian the question is about.
V1 variant (thesis): RGB with the background (prior == 0) dimmed to 30%, for backbones that only take 3 channels.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUN2, rule, video_path  # noqa: E402

T, SIZE, HALF_S, PED_DILATE = 9, 224, 1.6, 0.5


def window_frames(center, fps, last):
    return [int(np.clip(round(center + k * HALF_S / 4 * fps), 1, last)) for k in range(-4, 5)]


def box_at(track_boxes, frame):
    """The track's box on `frame`, or its nearest box in time (tracks have gaps)."""
    if not track_boxes:
        return None
    frames = np.array([b[0] for b in track_boxes])
    return tuple(track_boxes[int(np.argmin(np.abs(frames - frame)))][1:5])


def crop_square(boxes, width, height):
    xs = [v for b in boxes for v in (b[0], b[2])]
    ys = [v for b in boxes for v in (b[1], b[3])]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = min(max(SIZE, 1.5 * max(max(xs) - min(xs), max(ys) - min(ys))), width, height)
    x1 = int(np.clip(cx - side / 2, 0, width - side))
    y1 = int(np.clip(cy - side / 2, 0, height - side))
    return x1, y1, int(side)


def prior(mask, vbox, pbox, crop, src_size):
    """Binary prior on the crop at SIZE x SIZE. mask: 640x360 crosswalk mask; boxes in source pixels."""
    x1, y1, side = crop
    full = cv2.resize(mask.astype(np.uint8), src_size, interpolation=cv2.INTER_NEAREST)
    out = cv2.resize(full[y1:y1 + side, x1:x1 + side], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    scale = SIZE / side

    def paint(box, grow):
        bx1, by1, bx2, by2 = box
        g = grow * (by2 - by1)
        a = [int(np.clip(((bx1 - g) - x1) * scale, 0, SIZE)), int(np.clip(((by1 - g) - y1) * scale, 0, SIZE)),
             int(np.clip(((bx2 + g) - x1) * scale, 0, SIZE)), int(np.clip(((by2 + g) - y1) * scale, 0, SIZE))]
        out[a[1]:a[3], a[0]:a[2]] = 1
    if vbox is not None:
        paint(vbox, 0.0)
    if pbox is not None:
        paint(pbox, PED_DILATE)
    return out


def extract(stem, samples, params):
    """{sample id: uint8 (T, SIZE, SIZE, 4)} for samples of one clip: dicts with id, frame, vehicle_id, pedestrian_id."""
    capture = cv2.VideoCapture(str(video_path(stem)))
    fps, width, height, count = (capture.get(cv2.CAP_PROP_FPS), int(capture.get(3)), int(capture.get(4)),
                                 int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    tracks = json.loads((RUN2 / stem / "tracks" / "raw_tracks.json").read_text(encoding="utf-8"))
    tracks = {t["id"]: t["boxes"] for t in rule.merge_duplicate_tracks(tracks, params)}
    mask_at, _ = rule.load_mask_provider(str(RUN2 / stem / "seg" / "masks.npz"))
    plans, wanted = {}, set()
    for s in samples:
        frames = window_frames(s["frame"], fps, count)
        vb = [box_at(tracks.get(s["vehicle_id"], []), f) for f in frames]
        pb = [box_at(tracks.get(s["pedestrian_id"], []), f) if s.get("pedestrian_id") is not None else None
              for f in frames]
        crop = crop_square([b for b in vb + pb if b is not None], width, height)
        plans[s["id"]] = (frames, vb, pb, crop)
        wanted.update(frames)
    images, f = {}, 0
    while wanted - images.keys():
        ok, image = capture.read()
        f += 1  # frame numbers in the tracks start at 1
        if not ok:
            break
        if f in wanted:
            images[f] = image
    capture.release()
    out = {}
    for sid, (frames, vb, pb, crop) in plans.items():
        x1, y1, side = crop
        stack = []
        for frame, vbox, pbox in zip(frames, vb, pb):
            image = images.get(frame, images[max(images)])
            rgb = cv2.cvtColor(cv2.resize(image[y1:y1 + side, x1:x1 + side], (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
            mask = mask_at(frame)
            p = prior(mask if mask is not None else np.zeros((360, 640), bool), vbox, pbox, crop, (width, height))
            stack.append(np.dstack([rgb, p * 255]))
        out[sid] = np.stack(stack).astype(np.uint8)
    return out
