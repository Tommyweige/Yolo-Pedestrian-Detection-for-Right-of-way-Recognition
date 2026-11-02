"""Build the browser review page for every event a new-clip run produced.

Adapted from the review-11 builder (E:\\Learning\\yolo-ped-analysis\\review-11\\_build\\build_review.py): the card layout,
the template and the export format are the same, so a page built here is judged and exported the same way. What changed
is where the events come from — a run directory written by scripts/newclip_loop.py instead of one hand-made report — and
that every event of the run is a card, because every event still needs a verdict.

Reads, read-only: the run's videos, <run>/<stem>/seg/masks.npz and <run>/<stem>/tracks/raw_tracks.json (or wherever the
run manifest says they are). Writes index.html, events.json, media/<id>.* and README.md under <run>/review/.

The rule's own helpers (`person_state`, `vehicle_on`'s arithmetic, `crosswalk_axis`, `along_gap`) are imported from
yield_image_rule.py, so the numbers on a card are the rule's actual numbers, not a restatement of the docs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import yield_image_rule as rule  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "newclip_review_template.html"
SCHEMA = "yolo-ped/newclip-review@1"
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"
ROLE_NAMES = {"dev": "開發集", "validation": "驗證集"}
KIND_NAMES = {"car": "汽車", "motorcycle": "機車", "bus": "公車", "truck": "貨車", "bicycle": "自行車"}
VERDICT_LEGEND = [
    ("A", "合理：這確實是車輛未讓行人，該報", "precision 的分子"),
    ("B", "重報：情境沒問題，但同一件事已經報過／被拆成好幾件", "不算命中、不算誤報，算重報"),
    ("C", "不合理：不該報（誤判、行人其實在等、車沒進入路徑等）", "precision 的分母，算誤報"),
    ("D", "不確定：畫面或資訊不足", "預設不進 precision 分母，另外列出"),
]
_fonts = {}


def font(size):
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(FONT_PATH, size)
    return _fonts[size]


def log(*parts):
    print(*parts, flush=True)


def read_events(path):
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------- clip facts

class Clip:
    """One clip's video, masks, merged tracks, riders and the rule parameters the run used."""

    def __init__(self, clip_meta, params_overrides):
        self.stem = clip_meta["stem"]
        self.video = Path(clip_meta["video_path"])
        self.masks = Path(clip_meta["masks"])
        self.tracks = Path(clip_meta["tracks"])
        for path in (self.video, self.masks, self.tracks):
            if not path.exists():
                raise FileNotFoundError(path)
        fps = float(clip_meta["fps"])
        self.width, self.height = (int(value) for value in clip_meta["size"])
        overrides = {k: v for k, v in params_overrides.items() if k not in ("fps", "src_size")}
        if isinstance(overrides.get("rider_kinds"), list):
            overrides["rider_kinds"] = tuple(overrides["rider_kinds"])
        self.params = rule.Params(fps=fps, src_size=(self.width, self.height), **overrides)
        raw = json.loads(self.tracks.read_text(encoding="utf-8"))
        merged = rule.merge_duplicate_tracks(raw, self.params) if self.params.merge_duplicates else raw
        self.riders = rule.rider_frames(merged, self.params) if self.params.rider_exclusion else {}
        self.by_frame = {}
        for track in merged:
            for frame, *box in track["boxes"]:
                self.by_frame.setdefault(int(frame), []).append((track["id"], track["kind"], tuple(box)))
        self.track_length = {t["id"]: len(t["boxes"]) for t in merged}
        self.mask_at, self.mask_frames = rule.load_mask_provider(str(self.masks))
        self.sx, self.sy = 640 / self.width, 360 / self.height

    def box(self, frame, track_id):
        for i, kind, box in self.by_frame.get(frame, []):
            if i == track_id:
                return kind, box
        return None, None

    def scaled(self, box):
        return rule.scaled(box, self.sx, self.sy)


def vehicle_evidence(vbox, component, params):
    """The numbers behind rule.vehicle_on: share of the band under the box and of the flank rows."""
    h, w = component.shape
    width, height = vbox[2] - vbox[0], vbox[3] - vbox[1]
    x1, x2 = int(np.clip(vbox[0] + 0.2 * width, 0, w - 1)), int(np.clip(vbox[2] - 0.2 * width, 1, w))
    y1 = int(np.clip(vbox[3] + 1, 0, h - 1))
    y2 = int(np.clip(vbox[3] + max(2, 0.04 * height), y1 + 1, h))
    band = component[y1:y2, x1:max(x1 + 1, x2)]
    band_fraction = float(band.mean()) if band.size else 0.0
    wy1 = int(np.clip(vbox[3] - params.wheel_line_fraction * height, 0, h - 1))
    wy2 = int(np.clip(vbox[3] + 0.04 * height + 1, wy1 + 1, h))
    reach = params.flank_vehicle_widths * width
    flank_fraction = 0.0
    for fx1, fx2 in ((vbox[0] - reach, vbox[0]), (vbox[2], vbox[2] + reach)):
        a, b = int(np.clip(fx1, 0, w - 1)), max(int(np.clip(fx1, 0, w - 1)) + 1, int(np.clip(fx2, 1, w)))
        flank = component[wy1:wy2, a:b]
        if flank.size:
            flank_fraction = max(flank_fraction, float(flank.mean()))
    return {"band_fraction": band_fraction, "flank_fraction": flank_fraction,
            "band_on": band_fraction >= params.band_fraction,
            "flank_on": flank_fraction >= params.flank_fraction}


def frame_facts(clip, frame, vehicle_id, pedestrian_id):
    """The rule's decision inputs for one (vehicle, pedestrian) pair on one frame."""
    mask = clip.mask_at(frame)
    if mask is None:
        return None
    _, vbox_full = clip.box(frame, vehicle_id)
    _, pbox_full = clip.box(frame, pedestrian_id)
    if vbox_full is None or pbox_full is None:
        return {"frame": frame, "t": round(frame / clip.params.fps, 3), "present": False}
    vbox, pbox = clip.scaled(vbox_full), clip.scaled(pbox_full)
    best = None
    for component in rule.crosswalk_components(mask, clip.params):
        distance = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)
        depth = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        state = rule.person_state(pbox, component, distance, depth, clip.params)
        if state is None:
            continue
        height = pbox[3] - pbox[1]
        x = int(np.clip((pbox[0] + pbox[2]) / 2, 0, component.shape[1] - 1))
        y = int(np.clip(pbox[3], 0, component.shape[0] - 1))
        rx, ry = max(2, int(clip.params.foot_window_person_heights * height)), max(1, int(0.1 * height))
        around = component[max(0, y - ry):y + ry + 1, max(0, x - rx):x + rx + 1]
        veh = vehicle_evidence(vbox, component, clip.params)
        gap, extent = rule.along_gap(pbox, vbox, rule.crosswalk_axis(component))
        record = {"frame": frame, "t": round(frame / clip.params.fps, 3), "present": True,
                  "state": state, "inside": bool(component[y, x]),
                  "depth_ph": float(depth[y, x]) / max(1e-6, height),
                  "dist_ph": float(distance[y, x]) / max(1e-6, height),
                  "surround": float(around.mean()) if around.size else 0.0,
                  "gap_px": float(gap), "extent_px": float(extent),
                  "ratio": float(gap) / max(1e-6, float(extent)),
                  "rider": frame in clip.riders.get(pedestrian_id, ()),
                  "component": component, **veh}
        record["vehicle_on"] = record["band_on"] or record["flank_on"]
        record["in_gap"] = record["ratio"] <= clip.params.lateral_vehicle_widths
        record["qualifies"] = record["vehicle_on"] and record["in_gap"] and not record["rider"]
        rank = (record["qualifies"], record["vehicle_on"], record["state"] == "on_crosswalk")
        if best is None or rank > best[0]:
            best = (rank, record)
    if best is None:
        return {"frame": frame, "t": round(frame / clip.params.fps, 3), "present": True, "state": None,
                "component": None, "qualifies": False, "vehicle_on": False, "ratio": None,
                "rider": frame in clip.riders.get(pedestrian_id, ())}
    return best[1]


# ---------------------------------------------------------------- drawing

def put_text(image, lines, size, origin=(10, 8), spacing=1.18):
    """Chinese labels on a BGR array: PIL draws the text, cv2 draws the boxes."""
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, "RGBA")
    fnt = font(size)
    x, y = origin
    for line, colour in lines:
        left, top, right, bottom = draw.textbbox((x, y), line, font=fnt)
        draw.rectangle((left - 4, top - 3, right + 4, bottom + 3), fill=(0, 0, 0, 150))
        draw.text((x, y), line, font=fnt, fill=colour)
        y = bottom + int(size * spacing * 0.5)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def annotate(frame_bgr, crop, component, vbox, pbox, vehicle_id, vehicle_kind, pedestrian_id, hud, size):
    """One frame cropped to `crop`, with the crosswalk blob, both boxes and a HUD."""
    x0, y0, x1, y1 = crop
    out = frame_bgr[y0:y1, x0:x1].copy()
    clip_w = out.shape[1]
    if component is not None:
        mask = cv2.resize(component.astype(np.uint8), (frame_bgr.shape[1], frame_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1]
        tint = np.zeros_like(out)
        tint[:] = (200, 170, 30)
        where = mask > 0
        out[where] = (0.72 * out[where] + 0.28 * tint[where]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (60, 220, 255), 2)
    # Plain names, not track ids: the judge has to see at a glance which two of the people on screen to watch.
    for box, colour, label in ((vbox, (60, 210, 60), f"這台車（{KIND_NAMES.get(vehicle_kind, vehicle_kind)}）"),
                               (pbox, (60, 60, 240), "這位行人")):
        if box is None:
            continue
        ax0, ay0 = int(box[0]) - x0, int(box[1]) - y0
        ax1, ay1 = int(box[2]) - x0, int(box[3]) - y0
        cv2.rectangle(out, (ax0, ay0), (ax1, ay1), colour, 3)
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil, "RGBA")
        fnt = font(max(13, size - 3))
        left, top, right, bottom = draw.textbbox((0, 0), label, font=fnt)
        ty = max(0, ay0 - (bottom - top) - 8)
        lx = min(max(0, ax0), max(0, clip_w - (right - left) - 10))
        draw.rectangle((lx, ty, lx + (right - left) + 8, ty + (bottom - top) + 8), fill=colour[::-1] + (230,))  # BGR -> RGB
        draw.text((lx + 4, ty + 2), label, font=fnt, fill=(10, 10, 10, 255))
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return put_text(out, hud, size)


def crop_for(clip, frames, vehicle_id, pedestrian_id):
    """A fixed 16:9 crop around the pair for the whole window, so the frame does not jump.

    The pair can be far apart on screen (a car at one end of the crosswalk, a pedestrian near the camera), so the crop
    is grown vertically to 16:9 instead of being squashed into a letterbox: the tiles stay readable.
    """
    xs, ys = [], []
    for frame in frames:
        for track_id in (vehicle_id, pedestrian_id):
            _, box = clip.box(frame, track_id)
            if box:
                xs += [box[0], box[2]]
                ys += [box[1], box[3]]
    if not xs:
        return 0, 0, clip.width, clip.height
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    padx, pady = max(70, 0.12 * w), max(60, 0.20 * h)
    x0, x1 = min(xs) - padx, max(xs) + padx
    y0, y1 = min(ys) - pady, max(ys) + pady
    aspect = 16 / 9
    width, height = x1 - x0, y1 - y0
    if width / max(1, height) < aspect:                 # too tall: widen
        width = height * aspect
    else:                                               # too wide: grow the height
        height = width / aspect
    width = min(clip.width, max(640, width))
    height = min(clip.height, max(360, height))
    if width / height < aspect:                         # clamped by the frame: take what the frame allows
        width = min(clip.width, height * aspect)
    else:
        height = min(clip.height, width / aspect)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cx = min(max(cx, width / 2), clip.width - width / 2)
    cy = min(max(cy, height / 2), clip.height - height / 2)
    return (int(cx - width / 2), int(cy - height / 2), int(cx + width / 2), int(cy + height / 2))


# ---------------------------------------------------------------- per event

def read_window(clip, first, last):
    """Frames first..last as BGR arrays, seeking once and reading forward."""
    capture = cv2.VideoCapture(str(clip.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
    frames = {}
    frame = first
    while frame <= last:
        ok, image = capture.read()
        if not ok:
            break
        frames[frame] = image
        frame += 1
    capture.release()
    return frames


def build_event(number, event, clip, media_dir):
    fps = clip.params.fps
    start, end = event["contact_frames"]
    window = (max(1, int(round(start - 1.5 * fps))), int(round(end + 1.5 * fps)))
    if window[1] - window[0] < round(2.5 * fps):
        window = (window[0], window[0] + round(2.5 * fps))
    event_id = f"E{number:02d}"
    video_frames = read_window(clip, window[0], window[1])
    facts = []
    for frame in sorted(video_frames):
        record = frame_facts(clip, frame, event["vehicle_id"], event["pedestrian_id"])
        if record and record.get("present"):
            facts.append(record)
    if not facts:
        raise RuntimeError(f"{event_id}: no frames with both tracks in {window}")
    by_frame = {r["frame"]: r for r in facts}
    at = by_frame.get(event["frame"]) or min(facts, key=lambda r: abs(r["frame"] - event["frame"]))
    at_end = by_frame.get(end) or at
    entered = [r["frame"] for r in facts
               if r["state"] == "on_crosswalk" and start - round(clip.params.entry_window_s * fps) <= r["frame"]
               <= end + round(clip.params.entry_window_s * fps)]
    entered_at = min(entered) if entered else None

    crop = crop_for(clip, sorted(video_frames), event["vehicle_id"], event["pedestrian_id"])
    size = max(15, int((crop[2] - crop[0]) / 32))
    size_anim = max(14, int((crop[2] - crop[0]) / 40))

    def label_of(record):
        if record.get("present") is False:
            return "車輛或行人這一幀沒被偵測到"
        return {"on_crosswalk": "行人在斑馬線上", "waiting_at_edge": "行人在路緣",
                None: "行人不在斑馬線遮罩附近"}[record.get("state")]

    def when(frame):
        offset = (frame - event["frame"]) / fps
        near = "系統標記的時刻" if abs(offset) < 0.07 else f"標記{'前' if offset < 0 else '後'} {abs(offset):.1f} 秒"
        return f"{frame / fps:.2f} 秒（{near}）"

    # Words only on the frames; the rule's numbers stay on the page's collapsed technical section.
    def hud_for(record):
        flagged = bool(record.get("qualifies"))
        return [(when(record["frame"]), (255, 255, 255)),
                ("系統：車在斑馬線上、離行人很近" if flagged else label_of(record),
                 (255, 220, 90) if flagged else (205, 205, 205))]
    # ---- continuous-frames animation
    step = max(1, int(round(fps / 8)))
    sampled = [f for f in sorted(video_frames) if (f - window[0]) % step == 0]
    images = []
    for frame in sampled:
        record = by_frame.get(frame) or frame_facts(clip, frame, event["vehicle_id"], event["pedestrian_id"])
        _, vbox = clip.box(frame, event["vehicle_id"])
        _, pbox = clip.box(frame, event["pedestrian_id"])
        image = annotate(video_frames[frame], crop, (record or {}).get("component"), vbox, pbox,
                         event["vehicle_id"], event["vehicle_kind"], event["pedestrian_id"],
                         hud_for(record or {"t": frame / fps, "frame": frame, "state": None, "ratio": None}), size_anim)
        images.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    width = 760
    images = [im.resize((width, max(1, int(im.height * width / im.width))), Image.LANCZOS) for im in images]
    images[0].save(media_dir / f"{event_id}.webp", save_all=True, append_images=images[1:],
                   duration=int(1000 * step / fps), loop=0, quality=74, method=4)

    # ---- film strip of the same window, times spelled out
    strip_frames = [int(round(window[0] + i * (window[1] - window[0]) / 6)) for i in range(7)]
    tiles = []
    for frame in strip_frames:
        if frame not in video_frames:
            continue
        record = by_frame.get(frame) or frame_facts(clip, frame, event["vehicle_id"], event["pedestrian_id"]) or {}
        _, vbox = clip.box(frame, event["vehicle_id"])
        _, pbox = clip.box(frame, event["pedestrian_id"])
        hud = hud_for({"frame": frame, **record})
        image = annotate(video_frames[frame], crop, record.get("component"), vbox, pbox,
                         event["vehicle_id"], event["vehicle_kind"], event["pedestrian_id"], hud, size)
        tile = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tile = tile.resize((640, max(1, int(tile.height * 640 / tile.width))), Image.LANCZOS)
        tiles.append(np.array(tile)[:, :, ::-1].copy())
    height = max(t.shape[0] for t in tiles)
    strip = np.hstack([cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(20, 20, 20))
                       for t in tiles])
    cv2.imwrite(str(media_dir / f"{event_id}_strip.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # ---- the whole frame at the marked moment, for context
    _, vbox = clip.box(at["frame"], event["vehicle_id"])
    _, pbox = clip.box(at["frame"], event["pedestrian_id"])
    wide = annotate(video_frames[at["frame"]], (0, 0, clip.width, clip.height), at.get("component"),
                    vbox, pbox, event["vehicle_id"], event["vehicle_kind"], event["pedestrian_id"],
                    hud_for(at), max(18, int(clip.width / 44)))
    wide = cv2.resize(wide, (900, int(wide.shape[0] * 900 / wide.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(media_dir / f"{event_id}_wide.jpg"), wide, [cv2.IMWRITE_JPEG_QUALITY, 86])

    # ---- the reasons, with the rule's own numbers
    reasons = []
    state = at.get("state")
    if state == "on_crosswalk":
        if at["surround"] >= clip.params.foot_surround_fraction:
            reasons.append(f"行人 {event['pedestrian_id']} 的腳點在斑馬線上：腳點周圍（半徑 0.3 個人高）"
                           f"{at['surround']:.0%} 是斑馬線遮罩（門檻 {clip.params.foot_surround_fraction:.0%}）。")
        else:
            reasons.append(f"行人 {event['pedestrian_id']} 的腳點在遮罩內、離邊緣 {at['depth_ph']:.2f} 個人高"
                           f"（門檻 {clip.params.min_inside_person_heights:.2f}），判為「在斑馬線上」。")
    elif state == "waiting_at_edge":
        if at["inside"]:
            reasons.append(f"在標記的這一幀（{event['time_s']:.3f} s、幀 {event['frame']}），行人 {event['pedestrian_id']} "
                           f"的腳點在遮罩內，但只有 {at['depth_ph']:.2f} 個人高"
                           f"（門檻 {clip.params.min_inside_person_heights:.2f}）——這一刻的狀態是「在路緣等」。")
        else:
            reasons.append(f"在標記的這一幀（{event['time_s']:.3f} s、幀 {event['frame']}），行人 {event['pedestrian_id']} "
                           f"的腳點在遮罩外，離邊緣 {at['dist_ph']:.2f} 個人高"
                           f"（門檻 {clip.params.edge_margin_person_heights:.2f}）——這一刻的狀態是「在路緣等」。")
    else:
        reasons.append(f"行人 {event['pedestrian_id']} 這一刻不在任何斑馬線遮罩附近。")
    if at.get("band_on") and at.get("flank_on"):
        vehicle_line = (f"車 {event['vehicle_id']}（{event['vehicle_kind']}）在斑馬線上：車底帶 "
                        f"{at['band_fraction']:.0%}、車輪線左右斑馬線 {at['flank_fraction']:.0%}"
                        f"（門檻 {clip.params.band_fraction:.0%} / {clip.params.flank_fraction:.0%}）。")
    elif at.get("band_on"):
        vehicle_line = (f"車 {event['vehicle_id']}（{event['vehicle_kind']}）在斑馬線上：車底帶 "
                        f"{at['band_fraction']:.0%} 是斑馬線（門檻 {clip.params.band_fraction:.0%}）；"
                        f"車輪線旁的覆蓋 {at['flank_fraction']:.0%} 未達 {clip.params.flank_fraction:.0%}，靠車底帶成立。")
    elif at.get("flank_on"):
        tight = "，剛好壓在門檻上" if at["flank_fraction"] < clip.params.flank_fraction + 0.02 else ""
        vehicle_line = (f"車 {event['vehicle_id']}（{event['vehicle_kind']}）在斑馬線上：車底帶只有 "
                        f"{at['band_fraction']:.0%}（門檻 {clip.params.band_fraction:.0%}），但車輪線左右各 "
                        f"{clip.params.flank_vehicle_widths} 車寬的斑馬線佔 {at['flank_fraction']:.0%}"
                        f"（門檻 {clip.params.flank_fraction:.0%}{tight}）——遮罩常被車輛自己擋掉，這條是替代路徑。")
    else:
        vehicle_line = (f"車 {event['vehicle_id']} 在標記時間的這一幀兩條「在斑馬線上」的測試都不成立"
                        f"（車底帶 {at['band_fraction']:.0%}、車輪線旁 {at['flank_fraction']:.0%}）。")
    reasons.append(vehicle_line)
    gap_full = at["gap_px"] / clip.sx
    extent_full = at["extent_px"] / clip.sx
    # `at` is the marked frame, which the rule times at max(contact start, entry) and can therefore be after the
    # contact that qualified: say which of the two the printed ratio belongs to instead of calling every ratio "at the
    # threshold" (a 7x gap is not).
    if at["ratio"] <= clip.params.lateral_vehicle_widths + 1e-9:
        gap_tight = ("，剛好壓在門檻上"
                     if at["ratio"] > clip.params.lateral_vehicle_widths - 0.05 else "")
    else:
        gap_tight = "；標記的這一幀已經超過門檻，間距是在接觸期間才成立"
    reasons.append(f"兩者在斑馬線主軸方向相距 {gap_full:.0f} px，是這輛車在該方向寬度（{extent_full:.0f} px）的 "
                   f"{at['ratio']:.1f} 倍（門檻 ≤ {clip.params.lateral_vehicle_widths:.1f}{gap_tight}）；"
                   f"量的是行人腳點到車中心。")
    seconds = (end - start + 1) / fps
    edge = "，剛好等於門檻" if seconds <= clip.params.min_seconds + 1e-9 else ""
    reasons.append(f"這個配對連續成立 {end - start + 1} 幀（{seconds:.2f} s，門檻 {clip.params.min_seconds:.1f} s{edge}），"
                   f"接觸區間 {start / fps:.2f}–{end / fps:.2f} s。預設的候選清單建構把同一趟（相鄰接觸在 "
                   f"{clip.params.contact_dedup_s:.2f} s 內）同一個（車, 行人）配對的重複接觸合成一件；"
                   f"一趟裡的不同行人是不同件。")
    if entered_at is None:
        reasons.append("重算時找不到踏入斑馬線的幀——規則不該報這件，值得優先看。")
    elif entered_at < start:
        reasons.append(f"行人在接觸開始前 {(start - entered_at) / fps:.2f} s（幀 {entered_at}、{entered_at / fps:.2f} s）"
                       f"就已踏入斑馬線深處。規則只把「整個接觸期間（含前後 {clip.params.entry_window_s:.0f} s）"
                       f"都沒有踏入」的行人當成在路緣等而排除，所以他過了這一關；"
                       f"這次的標記時間是接觸開始的 {event['time_s']:.3f} s（max(接觸開始, 踏入幀)）。")
    elif entered_at == at["frame"]:
        reasons.append(f"行人在標記的這一幀（{event['time_s']:.3f} s、幀 {entered_at}）踏入斑馬線深處；"
                       f"規則只把「整個接觸期間（含前後 {clip.params.entry_window_s:.0f} s）都沒有一幀踏入」"
                       f"的行人排除，所以他過了這一關。")
    else:
        reasons.append(f"行人在接觸期間的 {entered_at / fps:.2f} s（幀 {entered_at}）才踏入斑馬線深處，"
                       f"比標記時間 {event['time_s']:.3f} s 晚。")
    if at.get("rider"):
        reasons.append(f"行人 {event['pedestrian_id']} 這一刻被判為騎士（在摩托車框內），照規則不該報——這是個 bug 訊號。")
    else:
        riding = sorted(clip.riders.get(event["pedestrian_id"], ()))
        during = [f for f in riding if start <= f <= end]
        detail = ("這個接觸期間沒有任何一幀被判為騎車" if riding and not during
                  else f"整段有 {len(riding)} 幀被判為騎車、其中 {len(during)} 幀落在接觸區間" if riding
                  else "整段都沒有被判為騎車")
        reasons.append(f"沒有觸發排除條件：行人 {event['pedestrian_id']} 不是騎士（{detail}），"
                       f"車輛軌跡 {clip.track_length.get(event['vehicle_id'], 0)} 幀 ≥ {clip.params.min_track_frames}。")

    states = {"on_crosswalk": "在斑馬線上", "waiting_at_edge": "在路緣等"}
    facts_table = [
        ["影片", f"{clip.stem}（{clip.width}×{clip.height}, {fps:.2f} fps）"],
        ["標記時間", f"{event['time_s']:.3f} s · 幀 {event['frame']}"],
        ["車輛", f"id {event['vehicle_id']} · {event['vehicle_kind']} · 軌跡 {clip.track_length.get(event['vehicle_id'], 0)} 幀"],
        ["行人", f"id {event['pedestrian_id']} · 標記幀狀態「{states.get(event['pedestrian_state'], event['pedestrian_state'])}」"],
        ["行人腳深", f"{at.get('depth_ph', 0):.2f} 個人高（門檻 {clip.params.min_inside_person_heights:.2f}）"],
        ["行人離邊緣", f"{at.get('dist_ph', 0):.2f} 個人高（門檻 {clip.params.edge_margin_person_heights:.2f}）"],
        ["腳點周圍", f"{at.get('surround', 0):.0%} 是斑馬線（門檻 {clip.params.foot_surround_fraction:.0%}）"],
        ["車底帶", f"{at.get('band_fraction', 0):.0%}（門檻 {clip.params.band_fraction:.0%}）"],
        ["車輪線旁", f"{at.get('flank_fraction', 0):.0%}（門檻 {clip.params.flank_fraction:.0%}）"],
        ["橫向間距", f"{at.get('gap_px', 0) / clip.sx:.0f} px / 車寬 {at.get('extent_px', 0) / clip.sx:.0f} px = "
                     f"{at.get('ratio', 0):.1f} 倍（門檻 ≤ {clip.params.lateral_vehicle_widths:.1f}）"],
        ["接觸", f"幀 {start}–{end} = {end - start + 1} 幀 = {seconds:.2f} s（門檻 {clip.params.min_seconds:.1f} s）"],
        ["踏入斑馬線", f"幀 {entered_at}（{entered_at / fps:.2f} s）" if entered_at else "無"],
        ["接觸結束時", f"行人「{states.get(at_end.get('state'), '不在遮罩附近')}」"],
        ["系統標記", "時間 = max(接觸開始, 第一次踏入斑馬線的幀)"],
    ]

    timeline = []
    for frame in range(window[0], window[1] + 1):
        record = by_frame.get(frame)
        if record is None:                      # one of the two tracks was not detected on this frame
            timeline.append({"f": frame, "t": round(frame / fps, 3), "c": 3, "q": False, "r": 0})
            continue
        timeline.append({"f": frame, "t": record["t"],
                         "c": {"on_crosswalk": 2, "waiting_at_edge": 1}.get(record.get("state"), 0),
                         "q": bool(record.get("qualifies")), "r": round(record.get("ratio") or 0, 1)})
    return {
        "id": event_id, "video": clip.stem,
        "time_s": event["time_s"], "frame": event["frame"],
        "vehicle_id": event["vehicle_id"], "vehicle_kind": event["vehicle_kind"],
        "pedestrian_id": event["pedestrian_id"], "pedestrian_state": event["pedestrian_state"],
        "contact": [start, end], "contact_s": [round(start / fps, 3), round(end / fps, 3)],
        "window": [window[0], window[1]], "entered_at": entered_at,
        "review_reasons": event.get("review_reasons") or [],
        "reasons": reasons, "facts": facts_table, "timeline": timeline,
        "media": {"clip": f"media/{event_id}.webp",
                  "strip": f"media/{event_id}_strip.jpg",
                  "wide": f"media/{event_id}_wide.jpg"},
        "video_path": str(clip.video),
    }


# ---------------------------------------------------------------- output

def storage_key_of(run_dir, salt=""):
    """A localStorage key that survives a rebuild (hash() is randomised per process) but differs per run.

    `salt` keeps a selection page built from the same run (only the unjudged events) from sharing its drafted
    verdicts with the run's full page.
    """
    text = str(Path(run_dir).resolve()) + "|" + salt
    return "yolo-ped-newclip-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def write_html(payload, out_dir, title, subtext, footer):
    template = TEMPLATE.read_text(encoding="utf-8")
    html = (template.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
                    .replace("__TITLE__", title)
                    .replace("__SUBTEXT__", subtext)
                    .replace("__TOTAL__", str(len(payload["events"])))
                    .replace("__FOOTER__", footer)
                    .replace("__STORAGE_KEY__", payload["storage_key"])
                    .replace("__EXPORT_PREFIX__", payload["export_prefix"]))
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    return len(html)


def legend_lines():
    lines = ["## 判斷代號", "", "| 代號 | 意思 | 計分時的作用 |", "|---|---|---|"]
    lines += [f"| **{code}** | {what} | {effect} |" for code, what, effect in VERDICT_LEGEND]
    return lines


def operation_lines(storage_key):
    return [
        "## 操作",
        "",
        "| 按鍵 | 作用 |",
        "|---|---|",
        "| `1` 沒有違規（記成 C）、`2` 有違規（A） | 判目前這件，並自動跳到下一件還沒判的 |",
        "| `3` 重複報告（B）、`4` 看不出來（D） | 同上，次要情況 |",
        "| `←` / `→` | 上一件／下一件；最後一件再按 `→` 到匯出頁 |",
        "",
        "頁面一次只顯示一件。系統為什麼報這件（腳深、車底帶、距離、接觸秒數）收在每件下方的「技術細節」裡。",
        "",
        f"判定即時存在瀏覽器（localStorage，key `{storage_key}`），關掉再開還在。"
        f"同一批換了 run 資料夾就會是新的 key，不會蓋掉舊的。",
        "",
    ]


def selection_note(counts):
    """The one paragraph that says what this partial page is, in numbers."""
    return (f"這 {counts['pending']} 件是 {counts['events']} 件裡還沒判的："
            f"{counts['new']} 件全新、{counts['rejudge']} 件是舊判定對不上、需要重判。"
            f"判完匯出 JSON，把檔案加進算分的 <code>--verdicts</code>，"
            f"precision 就涵蓋 {counts['events']} 件裡的 {counts['covered']} 件。")


def write_readme(manifest, payload, out_dir, run_dir):
    """How to judge this batch, and what to do with the export."""
    if payload.get("selection"):
        return write_selection_readme(manifest, payload, out_dir, run_dir)
    events = payload["events"]
    lines = [
        f"# 未讓事件判斷頁 · {len(manifest['clips'])} 支影片、{len(events)} 件",
        "",
        "## 打開這個檔案",
        "",
        "```text",
        str(out_dir / "index.html"),
        "```",
        "",
        "雙擊即可，不需要伺服器、不需要網路。圖片在同一個資料夾的 `media\\` 裡。",
        "",
        "## 這是哪一批",
        "",
        f"- run 資料夾：`{run_dir}`（角色：**{ROLE_NAMES.get(manifest['role'], manifest['role'])}**）",
        f"- 產生於 {manifest['created_at']}，規則 `yield_image_rule.py` 的預設參數"
        f"（候選清單建構、`contact_dedup=pair`、腳深 {manifest['params']['min_inside_person_heights']:.2f} 個人高）",
        f"- 影片：`{manifest['videos']}`（只讀）",
        "- 事件來自每支影片的 `events.jsonl`（規則原始輸出）；卡片上的數字是同一個 `yield_image_rule.py` 重算的",
        "",
        "| 影片 | 事件 | 影片檔 |",
        "|---|---|---|",
    ]
    for clip in manifest["clips"]:
        lines.append(f"| {clip['stem']} | {clip['events']} | `{clip['video_name']}` |")
    lines += [""] + legend_lines() + [""] + operation_lines(payload["storage_key"]) + [
        "## 匯出與算分",
        "",
        f"判完最後一件會到匯出頁（隨時也能點頁面上方的「匯出」），按 **匯出 JSON** 會下載 `{payload['export_prefix']}-verdicts-<時間>.json`。"
        "每一列都帶著影片檔名、時間、車輛 id、行人 id、接觸區間與備註，所以判定結果對得回影片、對得回那一件。"
        "把檔案路徑丟給算分指令：",
        "",
        "```powershell",
        f"python scripts/newclip_score.py --run {run_dir} --verdicts <匯出的.json>",
        "```",
        "",
        "要跟既有 7 支影片的判定合併，再加上開發集的 run 與它的判定：",
        "",
        "```powershell",
        "python scripts/newclip_score.py --run <這批> --dev-run <開發集 run> \\",
        "    --verdicts <這批匯出的.json> --dev-verdicts <開發集判定.json>",
        "```",
        "",
        "## 檔案",
        "",
        "```text",
        f"{out_dir.name}\\",
        "  index.html          ← 打開這個",
        "  README.md",
        "  events.json         卡片資料（規則理由、數字、時間軸）",
        "  media\\E01…         連續幀動畫、逐幀膠捲、全景一幀",
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selection_readme(manifest, payload, out_dir, run_dir):
    """The pending list: which events are left, which batch each one is, and what judging them buys."""
    selection = payload["selection"]
    counts = selection["counts"]
    lines = [
        f"# 待判清單 · {len(manifest['clips'])} 支影片、{counts['pending']} 件"
        f"（全部 {counts['events']} 件、已判 {counts['judged']} 件）",
        "",
        "## 打開這個檔案",
        "",
        "```text",
        str(out_dir / "index.html"),
        "```",
        "",
        "雙擊即可，不需要伺服器、不需要網路。圖片在同一個資料夾的 `media\\` 裡。",
        "",
        "## 這是哪一批",
        "",
        f"- run 資料夾：`{run_dir}`（角色：**{ROLE_NAMES.get(manifest['role'], manifest['role'])}**）",
        f"- 產生於 {manifest['created_at']}，規則 `yield_image_rule.py` 的預設參數"
        f"（候選清單建構、`contact_dedup=pair`、腳深 {manifest['params']['min_inside_person_heights']:.2f} 個人高）",
        f"- 影片：`{manifest['videos']}`（只讀）",
        f"- 事件總數 {counts['events']}，已經有判定的 {counts['judged']} 件不在這頁；"
        f"這頁只有還沒判的 {counts['pending']} 件",
        f"- 舊判定檔 {counts['stray_rows']} 列對不上任何事件：{counts['stray_rows'] - counts['orphan_rows']} 列"
        f"寫在卡片上，{counts['orphan_duplicates']} 列是同一件事的其他車輛，"
        f"{counts['orphan_gaps']} 列是規則現在沒報到的位置（見下面「沒有卡片的舊判定」）",
        "",
        "| 影片 | 待判 | 全新 | 重判 |",
        "|---|---|---|---|",
    ]
    per_clip = {}
    for item in payload["events"]:
        entry = per_clip.setdefault(item["video"], {"pending": 0, "new": 0, "rejudge": 0})
        entry["pending"] += 1
        entry[item["origin"]] += 1
    for clip in manifest["clips"]:
        entry = per_clip.get(clip["stem"], {"pending": 0, "new": 0, "rejudge": 0})
        lines.append(f"| {clip['stem']} | {entry['pending']} | {entry['new']} | {entry['rejudge']} |")
    lines += [
        f"| **小計** | **{counts['pending']}** | **{counts['new']}** | **{counts['rejudge']}** |",
        "",
        "## 卡片上的兩批",
        "",
        "- **全新事件**：這一件以前沒有任何判定，判它就是在補一個新的判定。",
        "- **舊判定對不上、需要重判**（頁面上會寫出舊紀錄）：有一列舊判定是針對這支影片做的，但時間差超過 1.25 s 的容差、接不到任何事件；"
        "它指的車輛 id 跟這一件相同（或落在這一件附近），所以這一列印在卡片上，連同舊的時間與判定，"
        "由判的人決定它還成不成立。",
        "- 頁面上另有「附近已經判過一件」：同一支影片在附近已經有一件判過了（含時間差、車輛、行人）。"
        "如果這一件其實是同一件事被拆成兩件，按「重複報告」（**B**）。",
        "",
    ] + legend_lines() + [""] + operation_lines(payload["storage_key"]) + [
        "## 判完能算到什麼",
        "",
        f"這 {counts['pending']} 件判完，precision 覆蓋 {counts['covered']}/{counts['events']} 件。",
    ]
    if counts["orphan_rows"]:
        lines += [
            "",
            "## 沒有卡片的舊判定",
            "",
            "這些列沒有卡片可以判，所以只列出來：",
            "",
            "| 影片 | 舊判定時間 | 車輛 | 判定 | 最近的事件 | 差 | 那件是 | 性質 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for orphan in selection["orphans"]:
            row = orphan["row"]
            kind = "同一件事的其他車輛" if orphan["kind"] == "duplicate" else "規則現在沒報到這一件"
            lines.append(f"| {row['stem']} | {row['time_s']:.3f}s | {row['vehicle']} | {row['verdict']} | "
                         f"{orphan['nearest_time_s']}s | {orphan['nearest_delta_s']} s | {orphan['nearest_judged']} | "
                         f"{kind} |")
        if counts["orphan_gaps"]:
            lines += ["", f"其中 {counts['orphan_gaps']} 列是 recall 的缺口（規則沒報到那件事），"
                          "不是這份清單能補的；同一件事的其他車輛只是舊檔多列的行，不影響 precision。"]
        else:
            lines += ["", "它們都是同一件事的其他車輛（舊檔案一個時刻列了好幾台車），不影響 precision 覆蓋。"]
    lines += [
        "",
        "## 匯出與算分",
        "",
        f"判完最後一件會到匯出頁（隨時也能點頁面上方的「匯出」），按 **匯出 JSON** 會下載 `{payload['export_prefix']}-verdicts-<時間>.json`。"
        "每一列都帶著影片檔名、時間、車輛 id、行人 id、接觸區間、屬於哪一批與備註。",
        "",
        "算分時把這份新判定跟**既有的判定檔一起**丟進去（舊檔還是要給，它涵蓋早先判過的 "
        f"{counts['judged']} 件）：",
        "",
        "```powershell",
        "python scripts/newclip_score.py --run <這批> --dev-run <開發集 run> \\",
        f"    --dev-verdicts <既有判定.json> --verdicts <這份匯出的.json>",
        "```",
        "",
        "## 檔案",
        "",
        "```text",
        f"{out_dir.name}\\",
        "  index.html          ← 打開這個",
        "  README.md",
        "  events.json         卡片資料（規則理由、數字、時間軸、屬於哪一批）",
        "  media\\E01…         連續幀動畫、逐幀膠捲、全景一幀",
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_review(run_dir, out_dir=None, title=None, selection=None):
    """The run's events as one page. Returns the payload; writes into <run>/review by default.

    `selection` (from scripts/newclip_pending.py) narrows the page to the events it lists, and tags each card with
    which batch it belongs to; without it every event of the run is a card.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    out_dir = Path(out_dir) if out_dir else run_dir / ("pending" if selection else "review")
    wanted = None
    if selection:
        wanted = {(item["stem"], item["index"]): item for item in selection["events"]}
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    for stale in media_dir.iterdir():           # a rebuild must not leave cards of events this page no longer has
        if stale.is_file():
            stale.unlink()
    role = ROLE_NAMES.get(manifest["role"], manifest["role"])
    total = sum(clip["events"] for clip in manifest["clips"])
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    salt = out_dir.name if selection else ""
    payload = {
        "schema": SCHEMA,
        "generated_at": generated,
        "source": {
            "run": str(run_dir),
            "role": manifest["role"],
            "videos": manifest["videos"],
            "branch": manifest["branch"],
            "commit": manifest["commit"],
            "tolerance_s": manifest["tolerance_s"],
            "clips": [{"stem": clip["stem"], "video": clip["video_name"], "events": clip["events"]}
                      for clip in manifest["clips"]],
        },
        "params": manifest["params"],
        "storage_key": storage_key_of(run_dir, salt),
        "export_prefix": f"newclip-{run_dir.name}" if not selection else f"newclip-{out_dir.name}",
        "events": [],
    }
    if selection:
        counts = selection["counts"]
        payload["selection"] = {"counts": counts, "note": selection_note(counts),
                                "orphans": selection["orphans"]}

    clips = {clip["stem"]: Clip(clip, manifest["params"]) for clip in manifest["clips"]}
    built = set()
    for clip_meta in manifest["clips"]:
        clip = clips[clip_meta["stem"]]
        for index, event in enumerate(read_events(clip_meta["events_jsonl"])):
            meta = None
            if wanted is not None:
                meta = wanted.get((clip_meta["stem"], index))
                if meta is None:
                    continue
                if abs(meta["time_s"] - event["time_s"]) > 1e-6:
                    raise RuntimeError(f"{clip_meta['stem']} 第 {index} 件對不上：清單寫 {meta['time_s']}，"
                                       f"run 寫 {event['time_s']}——events.jsonl 換過了，重跑 newclip_pending")
            number = len(payload["events"]) + 1
            card = build_event(number, event, clip, media_dir)
            if meta is not None:
                card["origin"] = meta["origin"]
                card["priors"] = meta["priors"]
                card["neighbour"] = meta["neighbour"]
                built.add((clip_meta["stem"], index))
            payload["events"].append(card)
            log(f"  E{number:02d} {clip_meta['stem'][:20]} {event['time_s']:>7.3f}s "
                f"veh {event['vehicle_id']:>5} ped {event['pedestrian_id']:>5}")
    if wanted is not None and built != set(wanted):
        missing = sorted(set(wanted) - built)
        raise RuntimeError(f"清單有 {len(missing)} 件在 run 裡找不到：{missing[:5]}")

    (out_dir / "events.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if selection:
        counts = selection["counts"]
        page_title = title or (f"行人未讓事件 · {role} 待判清單（{counts['pending']} / {counts['events']} 件）")
        subtext = (f"<code>{manifest['branch']}</code> @ <code>{manifest['commit']}</code> · "
                   f"規則預設參數（候選清單建構）· 這 {counts['pending']} 件是 {total} 件裡還沒判的"
                   f"（全新 {counts['new']}、重判 {counts['rejudge']}）")
    else:
        page_title = title or f"行人未讓事件 · {role} {len(manifest['clips'])} 支影片、{len(payload['events'])} 件"
        subtext = (f"<code>{manifest['branch']}</code> @ <code>{manifest['commit']}</code> · "
                   f"規則預設參數（候選清單建構）· 這 {total} 件全部要判")
    footer = (f"產出 <code>{generated}</code> · 影片 <code>{manifest['videos']}</code>（只讀）· "
              f"事件 <code>{run_dir}</code> · 這是輔助分析，不是法律違規認定。")
    size = write_html(payload, out_dir, page_title, subtext, footer)
    write_readme(manifest, payload, out_dir, run_dir)
    log(f"wrote {out_dir / 'index.html'} ({size / 1024:.0f} KB), {len(payload['events'])} event(s)")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory written by scripts/newclip_loop.py")
    parser.add_argument("--out", help="output folder (default: <run>/review)")
    parser.add_argument("--title", help="page title override")
    args = parser.parse_args(argv)
    build_review(args.run, args.out, args.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
