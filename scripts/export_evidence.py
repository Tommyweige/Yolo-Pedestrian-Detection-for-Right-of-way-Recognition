"""Cut the evidence window of each event from the source video into an mp4 (source is only read)."""
import argparse
import json
import sys
from pathlib import Path

import cv2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--events", required=True, help="events.jsonl from yield_rules.py")
    parser.add_argument("--output", required=True, help="folder for event clips")
    args = parser.parse_args(argv)
    events = [json.loads(l) for l in Path(args.events).read_text(encoding="utf-8").splitlines() if l.strip()]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.video)
    fps, size = capture.get(cv2.CAP_PROP_FPS), (int(capture.get(3)), int(capture.get(4)))
    for index, event in enumerate(events):
        lo, hi = event["evidence_frames"]  # one-based frame numbers
        clip = output / f"event{index:03d}_{event['kind']}_f{event['frame']}.mp4"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, lo - 1))
        for number in range(max(1, lo), hi + 1):
            ok, frame = capture.read()
            if not ok:
                break
            label = f"{event['kind']} t={event['time_s']}s {event['disclaimer']}"
            cv2.putText(frame, f"{number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            if number == event["frame"]:
                cv2.rectangle(frame, (0, 0), (size[0] - 1, size[1] - 1), (0, 0, 255), 8)
            writer.write(frame)
        writer.release()
    print(json.dumps({"clips": len(events)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
