import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import clip_artifacts as clip


class MasksTest(unittest.TestCase):
    """masks.npz: the latest segmented frame at or before the asked frame is the one every reader sees."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "masks.npz"
        packed = {}
        for number in (1, 4, 7):
            mask = np.zeros(clip.MASK_SHAPE, bool)
            mask[0, :number] = True  # row 0 holds the frame number as a count of set pixels
            packed[number] = clip.pack_mask(mask)
        clip.save_masks(self.path, packed)

    def test_returns_the_latest_mask_at_or_before_the_frame(self):
        mask_at, frames = clip.load_masks(self.path)
        self.assertEqual(frames, [1, 4, 7])
        self.assertIsNone(mask_at(0))
        got = {frame: int(mask_at(frame)[0].sum()) for frame in range(1, 10)}
        self.assertEqual(got, {1: 1, 2: 1, 3: 1, 4: 4, 5: 4, 6: 4, 7: 7, 8: 7, 9: 7})

    def test_the_shared_mask_cannot_be_changed_by_a_caller(self):
        mask_at, _ = clip.load_masks(self.path)
        with self.assertRaises(ValueError):
            mask_at(5)[0, 0] = False
        self.assertEqual(int(mask_at(4)[0].sum()), 4)

    def test_saved_with_the_npz_layout_older_files_use(self):
        """Key f<frame>, packed row-major booleans: files written before this module still load."""
        data = np.load(self.path)
        self.assertEqual(sorted(data.files), ["f1", "f4", "f7"])
        self.assertEqual(int(np.unpackbits(data["f4"]).sum()), 4)

    def test_a_mask_of_another_size_is_refused(self):
        with self.assertRaises(ValueError):
            clip.pack_mask(np.zeros((720, 1280), bool))


class TracksTest(unittest.TestCase):
    tracks = [{"id": 1, "kind": "person", "boxes": [[2, 0, 0, 1, 1], [3, 0, 0, 1, 1]]},
              {"id": 5, "kind": "car", "boxes": [[3, 1, 1, 2, 2]]}]

    def test_boxes_are_grouped_by_frame_in_track_order(self):
        self.assertEqual(clip.boxes_by_frame(self.tracks),
                         {2: [(1, "person", (0, 0, 1, 1))], 3: [(1, "person", (0, 0, 1, 1)), (5, "car", (1, 1, 2, 2))]})

    def test_last_frame_is_the_last_box(self):
        self.assertEqual(clip.last_frame(self.tracks), 3)
        self.assertEqual(clip.last_frame([]), 0)

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_tracks.json"
            clip.save_tracks(path, self.tracks)
            self.assertEqual(clip.load_tracks(path), self.tracks)


class EventsTest(unittest.TestCase):
    def test_events_round_trip_one_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            clip.write_events(path, [{"frame": 4, "time_s": 0.5, "note": "行人"}])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8").splitlines()[0])["note"], "行人")
            self.assertEqual(clip.read_events(path), [{"frame": 4, "time_s": 0.5, "note": "行人"}])

    def test_no_events_is_an_empty_list_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            clip.write_events(path, [])
            self.assertEqual(clip.read_events(path), [])


class VideoTest(unittest.TestCase):
    def test_video_info_reads_fps_size_and_frames(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
            for _ in range(5):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            info = clip.video_info(path)
            self.assertEqual((round(info.fps), info.size, info.frames), (12, (64, 48), 5))
            self.assertEqual(clip.find_video("clip", tmp), path)
            self.assertIsNone(clip.find_video("other", tmp))


if __name__ == "__main__":
    unittest.main()
