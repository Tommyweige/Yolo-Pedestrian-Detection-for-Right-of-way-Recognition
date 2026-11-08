"""Cut each event's evidence window (plus context before/after) from the source video into an mp4, and save the event
frame as an untouched full-resolution png (for reading the plate). The source video is only read.

The clip is re-encoded with OpenCV (mp4v, no audio), so it is somewhat softer than the source; the png is lossless.
File names match scripts/list_candidates.py: event<NNN>_t<time>s.mp4 / .png, NNN = line number in the events file.
"""
import argparse
import json
import queue
import sys
import threading
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import clip_artifacts  # noqa: E402
from list_candidates import clip_stem  # noqa: E402


SEEK_GAP = 300  # frames; a longer gap between clips is crossed by seeking (~0.1 s) instead of grabbing frame by frame


def clip_windows(events, pad, total):
    """[(lo, hi)] one-based, per event: its evidence window plus `pad` frames each side, inside the video."""
    windows = []
    for event in events:
        lo, hi = event["evidence_frames"]
        lo, hi = max(1, lo - pad), hi + pad
        windows.append((lo, min(hi, total) if total else hi))
    return windows


def merged(windows):
    """Sorted, non-overlapping (lo, hi) ranges covering every window."""
    spans = []
    for lo, hi in sorted(windows):
        if spans and lo <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    return spans


def frames_in(capture, spans):
    """Yield (number, frame) for every frame of the spans, decoding each frame once, front to back.

    Clips overlap a lot (many candidates within a few seconds), so seeking to each clip's start decoded the same frames
    again and again; seeking returns the same frames as reading in order, so long gaps are still crossed by seeking.
    """
    position = 0  # frames consumed so far
    for lo, hi in spans:
        if lo - 1 - position > SEEK_GAP:
            capture.set(cv2.CAP_PROP_POS_FRAMES, lo - 1)
            position = lo - 1
        while position < lo - 1:
            if not capture.grab():
                return
            position += 1
        for number in range(lo, hi + 1):
            ok, frame = capture.read()
            position += 1
            if not ok:
                return
            yield number, frame


class ThreadedWriter:
    """cv2.VideoWriter that encodes on its own thread (OpenCV releases the GIL), so overlapping clips encode in parallel.

    Encoding, not decoding, is most of the cost: every frame of every clip is encoded once per clip it belongs to.
    Frames handed to write() must not be changed afterwards.
    """

    def __init__(self, *args):
        self._writer = cv2.VideoWriter(*args)
        self._frames = queue.Queue(maxsize=8)
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while (frame := self._frames.get()) is not None:
            if self._error is None:
                try:
                    self._writer.write(frame)
                except BaseException as error:  # noqa: BLE001 - re-raised by release()
                    self._error = error
        self._writer.release()

    def write(self, frame):
        self._frames.put(frame)

    def release(self):
        self._frames.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error


def number_frame(frame, number):
    cv2.putText(frame, f"{number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--events", required=True, help="events .jsonl from yield_image_rule.py")
    parser.add_argument("--output", required=True, help="folder for event clips")
    parser.add_argument("--pad-seconds", type=float, default=0.0, help="extra context before and after the evidence window")
    args = parser.parse_args(argv)
    events = clip_artifacts.read_events(args.events)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    info = clip_artifacts.video_info(args.video)
    fps, size, total = info.fps, info.size, info.frames or None
    capture = cv2.VideoCapture(args.video)
    windows = clip_windows(events, round(args.pad_seconds * fps), total)
    names = [clip_stem(index, event) for index, event in enumerate(events)]
    writers, opened = {}, set()

    def open_writer(index):
        opened.add(index)
        writers[index] = ThreadedWriter(str(output / f"{names[index]}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    starts = {}
    for index, (lo, _) in enumerate(windows):
        starts.setdefault(lo, []).append(index)
    for number, frame in frames_in(capture, merged(windows)):
        numbered = None
        for index in starts.get(number, []):
            open_writer(index)
        for index in sorted(writers):  # only the clips this frame belongs to
            if number == events[index]["frame"]:
                cv2.imencode(".png", frame)[1].tofile(str(output / f"{names[index]}.png"))  # tofile: non-ASCII paths on Windows
                marked = frame.copy()
                cv2.rectangle(marked, (0, 0), (size[0] - 1, size[1] - 1), (0, 0, 255), 8)  # marks the event frame in the clip
                writers[index].write(number_frame(marked, number))
            else:
                if numbered is None:  # the frame stays untouched for the stills of other clips
                    numbered = number_frame(frame.copy(), number)
                writers[index].write(numbered)
            if number == windows[index][1]:
                writers.pop(index).release()
    for index in range(len(events)):  # clips the video ended before (or inside) still get their file, as before
        if index not in opened:
            open_writer(index)
        if index in writers:
            writers.pop(index).release()
    capture.release()
    print(json.dumps({"clips": len(events)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
