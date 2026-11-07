"""list_candidates: one review list per run, built from the events files (no GPU, no video)."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import list_candidates  # noqa: E402


def event(time_s, entered, **extra):
    record = {"kind": "vehicle_on_crosswalk_with_pedestrian", "frame": round(time_s * 30), "time_s": time_s,
              "vehicle_kind": "car", "pedestrian_state": "on_crosswalk" if entered else "waiting_at_edge",
              "pedestrian_entered_crosswalk": entered, "review_reasons": [], "mode": "review"}
    record.update(extra)
    return record


class CandidateListTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = Path(tmp.name) / "my-run"
        (self.run / "events").mkdir(parents=True)

    def write_events(self, stem, events):
        path = self.run / "events" / f"{stem}.jsonl"
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")

    def test_rows_follow_video_then_time_and_mark_priority(self):
        self.write_events("b影片", [event(75.25, False), event(12.0, True)])
        self.write_events("a", [event(3.0, True, review_reasons=["short_vehicle_track"])])
        rows = list_candidates.candidate_rows(self.run)
        self.assertEqual([(r["影片"], r["時間"]) for r in rows], [("a", "0:03.0"), ("b影片", "0:12.0"), ("b影片", "1:15.3")])
        self.assertEqual([r["優先"] for r in rows], ["高", "高", "低"])
        self.assertEqual(rows[0]["提醒"], "short_vehicle_track")

    def test_clip_and_still_are_listed_only_when_they_exist(self):
        events = [event(12.0, True), event(20.0, False)]
        self.write_events("clip", events)
        folder = self.run / "evidence" / "clip"
        folder.mkdir(parents=True)
        name = list_candidates.clip_stem(0, events[0])
        (folder / f"{name}.mp4").write_bytes(b"")
        (folder / f"{name}.png").write_bytes(b"")
        rows = list_candidates.candidate_rows(self.run)
        self.assertEqual(rows[0]["片段"], f"evidence/clip/{name}.mp4")
        self.assertEqual(rows[0]["截圖"], f"evidence/clip/{name}.png")
        self.assertEqual((rows[1]["片段"], rows[1]["截圖"]), ("", ""))

    def test_main_writes_a_csv_excel_opens_as_utf8_and_counts_rows(self):
        self.write_events("clip", [event(12.0, True), event(20.0, False)])
        self.write_events("empty", [])
        self.assertEqual(list_candidates.main(["--run-dir", str(self.run)]), 0)
        raw = (self.run / "candidates.csv").read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))  # BOM: Excel otherwise garbles the Chinese headers
        rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["秒"], "12.0")

    def test_missing_events_folder_returns_two(self):
        self.assertEqual(list_candidates.main(["--run-dir", str(self.run / "nope")]), 2)


if __name__ == "__main__":
    unittest.main()
