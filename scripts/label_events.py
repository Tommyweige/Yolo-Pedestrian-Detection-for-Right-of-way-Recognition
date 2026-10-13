"""Mark yield-to-pedestrian events in a video by hand; writes labels.jsonl for scripts/eval_events.py.

Keys (click the video window first):
  SPACE play/pause      A / D   back / forward 1 frame      Z / C   back / forward 10 frames
  J / L  back / forward 1 second                            1 / 2 / 3   play at 0.25x / 0.5x / 1x
  F  mark "vehicle did NOT yield" at this frame             Y  mark "vehicle clearly yielded"
  U  undo the last mark                                     N  declare "this video has no events"
  Q  save and quit (marks are also saved immediately)

What to mark: pause on the first frame where a vehicle's front wheels are on the crosswalk (past the stripe edge) while a
pedestrian is on it or stepping onto it. One mark per vehicle. Waiting at the kerb does not count. +-1 s is good enough.
Time is (one-based frame number) / fps, the same convention as the detection pipeline.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

FAILURE, YIELDED = "suspected_failure_to_yield", "yielded"


def time_of(frame, fps):
    """frame is one-based, as in the detection pipeline."""
    return round(frame / fps, 3)


class LabelStore:
    """Labels for ONE video, saved after every change; resumes from an existing file."""

    def __init__(self, path, video_name, fps):
        self.path, self.video, self.fps = Path(path), video_name, fps
        self.rows, self.declared_none = [], False
        if self.path.is_file():
            rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.rows = [r for r in rows if r.get("video") == video_name]
            others = [r for r in rows if r.get("video") != video_name]
            if others:
                raise ValueError(f"{self.path} already holds labels for another video; use one labels file per video")

    def add(self, kind, frame):
        self.declared_none = False
        self.rows.append({"video": self.video, "kind": kind, "frame": frame, "time_s": time_of(frame, self.fps)})
        self.save()

    def undo(self):
        removed = self.rows.pop() if self.rows else None
        self.save()
        return removed

    def declare_none(self):
        """The human watched the whole video and saw no event (a negative clip). Refused while marks exist."""
        if self.rows:
            return False
        self.declared_none = True
        self.save()
        return True

    def save(self):
        """Atomic write. Nothing is written when there are no marks and 'no events' was not declared."""
        if not self.rows and not self.declared_none and not self.path.is_file():
            return
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(self.rows, key=lambda r: r["frame"]))
        handle, temporary = tempfile.mkstemp(dir=self.path.parent or ".", suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, self.path)


def run_window(video, store, max_width=1280):
    import cv2
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = store.fps
    title = f"label events - {Path(video).name}"
    frame_no, playing, speed, flash = 1, False, 1.0, ("", 0)
    cache = {}

    def read(number):
        if number not in cache:
            capture.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
            ok, image = capture.read()
            cache.clear()
            cache[number] = image if ok else None
        return cache[number]

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    while True:
        image = read(frame_no)
        if image is None:
            frame_no = max(1, min(frame_no, total))
            playing = False
            image = read(frame_no)
            if image is None:
                break
        shown = image.copy()
        scale = min(1.0, max_width / shown.shape[1])
        if scale < 1.0:
            shown = cv2.resize(shown, None, fx=scale, fy=scale)
        banner = f"frame {frame_no}/{total}  t={time_of(frame_no, fps):.3f}s  {'PLAYING' if playing else 'PAUSED'} {speed}x   marks: {len(store.rows)}"
        cv2.putText(shown, banner, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(shown, banner, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        for i, row in enumerate(store.rows[-3:]):
            line = f"{row['kind']} @ {row['time_s']}s"
            cv2.putText(shown, line, (12, 66 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(shown, line, (12, 66 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if flash[1] > 0:
            cv2.putText(shown, flash[0], (12, shown.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            flash = (flash[0], flash[1] - 1)
        cv2.imshow(title, shown)
        delay = max(1, int(1000 / fps / speed)) if playing else 0
        key = cv2.waitKeyEx(delay)
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == -1:
            if playing:
                frame_no = min(total, frame_no + 1)
                playing = frame_no < total
            continue
        key = key & 0xFF
        char = chr(key).lower() if key < 128 else ""
        if char == " ":
            playing = not playing
        elif char == "a":
            frame_no, playing = max(1, frame_no - 1), False
        elif char == "d":
            frame_no, playing = min(total, frame_no + 1), False
        elif char == "z":
            frame_no, playing = max(1, frame_no - 10), False
        elif char == "c":
            frame_no, playing = min(total, frame_no + 10), False
        elif char == "j":
            frame_no, playing = max(1, frame_no - round(fps)), False
        elif char == "l":
            frame_no, playing = min(total, frame_no + round(fps)), False
        elif char in "123":
            speed = {"1": 0.25, "2": 0.5, "3": 1.0}[char]
        elif char == "f":
            store.add(FAILURE, frame_no)
            flash = (f"MARKED: did NOT yield @ {time_of(frame_no, fps)}s", 40)
        elif char == "y":
            store.add(YIELDED, frame_no)
            flash = (f"MARKED: yielded @ {time_of(frame_no, fps)}s", 40)
        elif char == "u":
            removed = store.undo()
            flash = (f"UNDONE {removed['kind']} @ {removed['time_s']}s" if removed else "nothing to undo", 40)
        elif char == "n":
            flash = (("DECLARED: no events in this video" if store.declare_none()
                      else "marks exist: undo them (U) before declaring no events"), 60)
        elif char == "q":
            break
    cv2.destroyAllWindows()
    capture.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True, help="labels .jsonl for THIS video (resumes if it exists)")
    args = parser.parse_args(argv)
    import cv2
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if not fps or fps <= 0:
        print(f"Cannot read the video or its frame rate: {args.video}", file=sys.stderr)
        return 2
    store = LabelStore(args.output, Path(args.video).name, fps)
    run_window(args.video, store)
    store.save()
    print(json.dumps({"marks": len(store.rows), "declared_no_events": store.declared_none, "file": str(store.path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
