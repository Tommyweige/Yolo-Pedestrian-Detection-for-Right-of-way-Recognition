"""export_evidence: clips with context before/after, and an untouched full-resolution still of the event frame."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
