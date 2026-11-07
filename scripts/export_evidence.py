"""Cut each event's evidence window (plus context before/after) from the source video into an mp4, and save the event
frame as an untouched full-resolution png (for reading the plate). The source video is only read.

The clip is re-encoded with OpenCV (mp4v, no audio), so it is somewhat softer than the source; the png is lossless.
File names match scripts/list_candidates.py: event<NNN>_t<time>s.mp4 / .png, NNN = line number in the events file.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from list_candidates import clip_stem  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--events", required=True, help="events .jsonl from yield_image_rule.py")
    parser.add_argument("--output", required=True, help="folder for event clips")
    parser.add_argument("--pad-seconds", type=float, default=0.0, help="extra context before and after the evidence window")
    args = parser.parse_args(argv)
    events = [json.loads(l) for l in Path(args.events).read_text(encoding="utf-8").splitlines() if l.strip()]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.video)
    fps, size = capture.get(cv2.CAP_PROP_FPS), (int(capture.get(3)), int(capture.get(4)))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    pad = round(args.pad_seconds * fps)
    for index, event in enumerate(events):
        lo, hi = event["evidence_frames"]  # one-based frame numbers
        lo, hi = max(1, lo - pad), hi + pad
        if total:
            hi = min(hi, total)
        name = clip_stem(index, event)
        writer = cv2.VideoWriter(str(output / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        capture.set(cv2.CAP_PROP_POS_FRAMES, lo - 1)
        for number in range(lo, hi + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if number == event["frame"]:
                cv2.imencode(".png", frame)[1].tofile(str(output / f"{name}.png"))  # tofile: non-ASCII paths on Windows
                frame = frame.copy()
                cv2.rectangle(frame, (0, 0), (size[0] - 1, size[1] - 1), (0, 0, 255), 8)  # marks the event frame in the clip
            cv2.putText(frame, f"{number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            writer.write(frame)
        writer.release()
    capture.release()
    print(json.dumps({"clips": len(events)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
