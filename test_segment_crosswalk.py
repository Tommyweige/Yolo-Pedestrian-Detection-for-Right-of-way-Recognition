import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import segment_crosswalk as seg  # noqa: E402


class FakeCapture:
    """Frames are 1x1 images whose value is their one-based frame number."""

    def __init__(self, count):
        self.count, self.position, self.reads, self.grabs = count, 0, 0, 0

    def grab(self):
        self.grabs += 1
        self.position += 1
        return self.position <= self.count

    def read(self):
        self.reads += 1
        self.position += 1
        if self.position > self.count:
            return False, None
        return True, np.full((1, 1, 3), self.position, np.uint8)


class SampledFramesTest(unittest.TestCase):
    def test_reads_the_wanted_frames_in_order_without_seeking(self):
        capture = FakeCapture(10)
        got = [(number, int(frame[0, 0, 0])) for number, frame in seg.sampled_frames(capture, [1, 4, 7, 10])]
        self.assertEqual(got, [(1, 1), (4, 4), (7, 7), (10, 10)])
        self.assertEqual(capture.reads, 4)  # skipped frames are grabbed, not decoded
        self.assertEqual(capture.grabs, 6)

    def test_unsorted_or_repeated_numbers_are_read_once_in_order(self):
        got = [number for number, _ in seg.sampled_frames(FakeCapture(10), [7, 2, 7])]
        self.assertEqual(got, [2, 7])

    def test_frames_past_the_end_are_skipped(self):
        got = [number for number, _ in seg.sampled_frames(FakeCapture(5), [3, 9])]
        self.assertEqual(got, [3])


class PrefetchTest(unittest.TestCase):
    def test_yields_every_item_in_order(self):
        self.assertEqual(list(seg.prefetch(iter(range(50)), depth=3)), list(range(50)))

    def test_an_error_in_the_source_reaches_the_consumer(self):
        def broken():
            yield 1
            raise ValueError("decode failed")

        with self.assertRaises(ValueError):
            list(seg.prefetch(broken()))


class StrideTest(unittest.TestCase):
    def test_one_mask_per_tenth_of_a_second_rounded(self):
        self.assertEqual([seg.stride_for(fps, 0.1) for fps in (25, 29.97, 30, 50, 59.94, 60, 24)], [3, 3, 3, 5, 6, 6, 2])

    def test_never_below_one_frame(self):
        self.assertEqual(seg.stride_for(5, 0.1), 1)
        self.assertEqual(seg.stride_for(0, 0.1), 1)  # unknown fps: every frame


class NormalizeTest(unittest.TestCase):
    def test_matches_rescale_then_mean_std_from_the_config(self):
        config = {"rescale_factor": 1 / 255, "image_mean": [0.5, 0.25, 0.0], "image_std": [0.5, 0.25, 1.0]}
        scale, mean, std = seg.normalize_constants(config)
        rgb = np.array([[[255, 0, 51]]], np.float32)
        out = (rgb * scale - mean) / std
        np.testing.assert_allclose(out[0, 0], [1.0, -1.0, 0.2], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
