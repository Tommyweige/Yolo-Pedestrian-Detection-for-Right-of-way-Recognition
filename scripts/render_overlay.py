"""Draw what the pipeline sees onto a video: crosswalk mask, tracked boxes, predicted events, human labels.

Read-only on the source video. Overlay: green tint = crosswalk mask (nearest segmented frame), cyan box = person,
yellow box = vehicle (id shown), red banner = predicted event (+-1 s), blue banner = human label (+-1 s).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/msjh.ttc"
VEHICLES = {"car", "motorcycle", "bus", "truck"}


def read_jsonl(path):
    if not path or not Path(path).is_file():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", required=True, help="masks.npz from segment_crosswalk.py")
    parser.add_argument("--tracks", required=True, help="raw_tracks.json from track_video.py")
    parser.add_argument("--events", help="predicted events.jsonl")
    parser.add_argument("--labels", help="human labels .jsonl")
    parser.add_argument("--output", required=True, help="output .mp4")
    parser.add_argument("--width", type=int, default=1280, help="output width (keeps aspect ratio)")
    parser.add_argument("--window", type=float, default=1.0, help="seconds around an event/label that the banner stays")
    args = parser.parse_args(argv)

    masks = np.load(args.masks)
    mask_frames = sorted(int(k[1:]) for k in masks.files)
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    by_frame = {}
    for t in tracks:
        for number, x1, y1, x2, y2 in t["boxes"]:
            by_frame.setdefault(int(number), []).append((t["id"], t["kind"], x1, y1, x2, y2))
    events, labels = read_jsonl(args.events), read_jsonl(args.labels)

    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    src_w, src_h = int(capture.get(3)), int(capture.get(4))
    out_w = args.width
    out_h = round(src_h * out_w / src_w / 2) * 2
    scale = out_w / src_w
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    font = ImageFont.truetype(FONT_PATH, max(18, out_w // 50))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        number += 1
        frame = cv2.resize(frame, (out_w, out_h))
        nearest = min(mask_frames, key=lambda f: abs(f - number))
        mask = np.unpackbits(masks[f"f{nearest}"])[:360 * 640].astype(bool).reshape(360, 640)
        mask = cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        tint = frame.copy()
        tint[mask] = (0.45 * tint[mask] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
        frame = tint
        for track_id, kind, x1, y1, x2, y2 in by_frame.get(number, []):
            if kind != "person" and kind not in VEHICLES:
                continue
            colour = (255, 255, 0) if kind == "person" else (0, 215, 255)
            p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(frame, p1, p2, colour, 2)
            cv2.putText(frame, f"{kind[:3]}{track_id}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        seconds = number / fps
        banners = [(f"偵測:沒讓行人 @{e['time_s']:.2f}s", (200, 0, 0)) for e in events if abs(seconds - e["time_s"]) <= args.window]
        banners += [(f"人工標記 @{l['time_s']:.2f}s", (0, 90, 220)) for l in labels if abs(seconds - l["time_s"]) <= args.window]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        draw.text((10, 6), f"{seconds:6.2f}s  frame {number}", font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        for i, (text, colour) in enumerate(banners):
            y = 40 + i * (font.size + 10)
            draw.rectangle((10, y, 10 + len(text) * font.size, y + font.size + 6), fill=colour)
            draw.text((14, y), text, font=font, fill=(255, 255, 255))
        writer.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        if number % 60 == 0:
            print(f"  {number}/{total}", flush=True)
    writer.release()
    print(f"wrote {args.output} ({number} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
