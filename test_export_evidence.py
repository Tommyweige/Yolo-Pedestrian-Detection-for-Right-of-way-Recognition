"""export_evidence: clips with context before/after, and an untouched full-resolution still of the event frame."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import export_evidence  # noqa: E402
import list_candidates  # noqa: E402


class ExportEvidenceTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.video = self.dir / "clip.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for number in range(1, 101):  # 10 s; frame n is filled with grey level 2n so stills can be identified
            writer.write(np.full((48, 64, 3), 2 * number, np.uint8))
        writer.release()
        self.events = self.dir / "events.jsonl"
        self.event = {"kind": "vehicle_on_crosswalk_with_pedestrian", "frame": 50, "time_s": 5.0,
                      "evidence_frames": [40, 60], "disclaimer": "x"}
        self.events.write_text(json.dumps(self.event) + "\n", encoding="utf-8")

    def export(self, *extra):
        out = self.dir / "out"
        code = export_evidence.main(["--video", str(self.video), "--events", str(self.events), "--output", str(out), *extra])
        return code, out

    def test_clip_is_padded_by_the_given_seconds_and_named_like_the_candidate_list(self):
        code, out = self.export("--pad-seconds", "2")
        self.assertEqual(code, 0)
        name = list_candidates.clip_stem(0, self.event)
        clip = cv2.VideoCapture(str(out / f"{name}.mp4"))
        count = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
        clip.release()
        self.assertEqual(count, 61)  # evidence 40..60, 2 s = 20 frames each side -> frames 20..80

    def test_padding_stops_at_the_video_ends(self):
        self.event["evidence_frames"] = [1, 95]
        self.events.write_text(json.dumps(self.event) + "\n", encoding="utf-8")
        _, out = self.export("--pad-seconds", "2")
        clip = cv2.VideoCapture(str(out / f"{list_candidates.clip_stem(0, self.event)}.mp4"))
        count = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
        clip.release()
        self.assertEqual(count, 100)

    def test_still_is_the_event_frame_at_full_size_without_drawing(self):
        _, out = self.export()
        still = cv2.imread(str(out / f"{list_candidates.clip_stem(0, self.event)}.png"))
        self.assertEqual(still.shape, (48, 64, 3))
        self.assertLess(abs(int(still.mean()) - 100), 6)  # frame 50 -> grey 100 (codec noise allowed), no text or border


def frames_of(path):
    capture, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            return frames
        frames.append(frame)


def seek_per_clip(video, events, output, pad):
    """The earlier implementation: seek to every clip's start (overlapping clips decode the same frames again)."""
    capture = cv2.VideoCapture(str(video))
    fps, size = capture.get(cv2.CAP_PROP_FPS), (int(capture.get(3)), int(capture.get(4)))
    total, pad = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), round(pad * fps)
    for index, event in enumerate(events):
        lo, hi = event["evidence_frames"]
        lo, hi = max(1, lo - pad), min(hi + pad, total)
        name = list_candidates.clip_stem(index, event)
        writer = cv2.VideoWriter(str(output / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        capture.set(cv2.CAP_PROP_POS_FRAMES, lo - 1)
        for number in range(lo, hi + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if number == event["frame"]:
                cv2.imencode(".png", frame)[1].tofile(str(output / f"{name}.png"))
                frame = frame.copy()
                cv2.rectangle(frame, (0, 0), (size[0] - 1, size[1] - 1), (0, 0, 255), 8)
            cv2.putText(frame, f"{number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            writer.write(frame)
        writer.release()
    capture.release()


class ReadOnceTest(unittest.TestCase):
    """Overlapping clips come from one front-to-back read and match the per-clip seeking they replace."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.video = self.dir / "clip.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
        rng = np.random.default_rng(0)
        for _ in range(400):  # 40 s of noise, so a wrong frame cannot look like the right one
            writer.write(rng.integers(0, 255, (64, 96, 3), np.uint8))
        writer.release()
        kind = "vehicle_on_crosswalk_with_pedestrian"
        self.events = [  # two overlapping clips, one sharing its event frame with the other's window, one far away
            {"kind": kind, "frame": 50, "time_s": 5.0, "evidence_frames": [40, 60]},
            {"kind": kind, "frame": 55, "time_s": 5.5, "evidence_frames": [45, 70]},
            {"kind": kind, "frame": 350, "time_s": 35.0, "evidence_frames": [345, 390]},
            {"kind": kind, "frame": 30, "time_s": 3.0, "evidence_frames": [25, 35]},  # out of order in the file
        ]
        (self.dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in self.events), encoding="utf-8")

    def test_clips_and_stills_match_seeking_to_each_clip(self):
        new, old = self.dir / "new", self.dir / "old"
        old.mkdir()
        seek_per_clip(self.video, self.events, old, 1.0)
        code = export_evidence.main(["--video", str(self.video), "--events", str(self.dir / "events.jsonl"),
                                     "--output", str(new), "--pad-seconds", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in new.iterdir()), sorted(p.name for p in old.iterdir()))
        for path in old.iterdir():
            if path.suffix == ".png":
                self.assertEqual((new / path.name).read_bytes(), path.read_bytes(), path.name)
            else:
                a, b = frames_of(new / path.name), frames_of(path)
                self.assertEqual(len(a), len(b), path.name)
                self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)), path.name)

    def test_every_needed_frame_is_decoded_once(self):
        reads = []
        real = cv2.VideoCapture

        class Counting:
            def __init__(self, *args):
                self.inner = real(*args)

            def read(self):
                reads.append(1)
                return self.inner.read()

            def __getattr__(self, name):
                return getattr(self.inner, name)

        with mock.patch.object(export_evidence.cv2, "VideoCapture", Counting):
            export_evidence.main(["--video", str(self.video), "--events", str(self.dir / "events.jsonl"),
                                  "--output", str(self.dir / "out"), "--pad-seconds", "1"])
        # windows 15..45, 30..70, 35..80 and 335..400 overlap into 15..80 and 335..400: 66 + 66 frames
        self.assertEqual(len(reads), 132)


if __name__ == "__main__":
    unittest.main()
