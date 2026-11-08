"""單支影片的中間產物 (clip artefacts): the files one clip's pipeline steps hand each other, and how to read and write them.

Every step and every evaluation script goes through here, so each format is defined once:

- masks.npz (segment_crosswalk.py): one packed MASK_SHAPE boolean array per segmented frame, key f<frame>. A frame
  between two segmented ones uses the latest mask at or before it (`load_masks`); that is the mask the rule judges with,
  so anything shown or measured about "the crosswalk on frame N" must use the same lookup.
- raw_tracks.json (track_video.py): [{"id", "kind", "boxes": [[frame, x1, y1, x2, y2], ...], ...}], frames one-based,
  boxes in source-video pixels.
- events .jsonl (yield_image_rule.py): one event dict per line.
- the source video (`video_info`): fps, size and frame count. Event times and every threshold given in seconds depend on
  the fps, and box coordinates on the size, so they are read from the video rather than passed by hand.

A clip folder holds its masks at `MASKS` and its tracks at `TRACKS` (relative paths). Only numpy is imported at module
level: cv2 is imported when a video is opened, so the GUI and the tests can import this cheaply.
"""
import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VEHICLE_KINDS = frozenset({"car", "motorcycle", "bus", "truck"})
MASK_SHAPE = (360, 640)  # rows, columns of every stored mask, whatever the video size
MASKS = Path("seg") / "masks.npz"
TRACKS = Path("tracks") / "raw_tracks.json"
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi")


def pack_mask(mask):
    """One MASK_SHAPE boolean mask as stored in masks.npz (an eighth of the memory, so a long clip fits)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != MASK_SHAPE:
        raise ValueError(f"masks are stored at {MASK_SHAPE}, got {mask.shape}")
    return np.packbits(mask.ravel())


def save_masks(path, packed):
    """Write {frame: pack_mask(...)} as masks.npz."""
    np.savez_compressed(path, **{f"f{frame}": value for frame, value in packed.items()})


def load_masks(path):
    """(mask_at, frames): mask_at(frame) is the latest mask at or before the frame, None before the first; frames are
    the segmented frame numbers, sorted. The arrays are read-only and shared between calls; ask in frame order, since
    only the current mask is kept decoded."""
    data = np.load(path)
    frames = sorted(int(key[1:]) for key in data.files)
    size = MASK_SHAPE[0] * MASK_SHAPE[1]
    cache = {}

    def mask_at(frame):
        index = bisect.bisect_right(frames, frame)  # a linear scan here made an hour of video quadratic
        if index == 0:
            return None
        key = frames[index - 1]
        if key not in cache:
            cache.clear()
            mask = np.unpackbits(data[f"f{key}"])[:size].astype(bool).reshape(MASK_SHAPE)
            mask.setflags(write=False)
            cache[key] = mask
        return cache[key]
    return mask_at, frames


def load_tracks(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_tracks(path, tracks):
    Path(path).write_text(json.dumps(list(tracks)), encoding="utf-8")


def last_frame(tracks):
    """The last frame any track has a box on (0 for no tracks): the rule has nothing to judge after it."""
    return max((int(box[0]) for track in tracks for box in track["boxes"]), default=0)


def boxes_by_frame(tracks):
    """{frame: [(track id, kind, (x1, y1, x2, y2)), ...]} in track order."""
    boxes = {}
    for track in tracks:
        for frame, *box in track["boxes"]:
            boxes.setdefault(int(frame), []).append((track["id"], track["kind"], tuple(box)))
    return boxes


def read_events(path):
    """The events in an events .jsonl; an empty file is no events."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_events(path, events):
    Path(path).write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8")


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frames: int  # as the container reports it; 0 when unknown

    @property
    def size(self):
        return self.width, self.height


def video_info(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        return VideoInfo(capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()


def find_video(stem, folder):
    """The source video named <stem> in a folder (common suffixes first), or None."""
    folder = Path(folder)
    for suffix in VIDEO_SUFFIXES:
        path = folder / (stem + suffix)
        if path.exists():
            return path
    matches = sorted(folder.glob(stem + ".*"))
    return matches[0] if matches else None
