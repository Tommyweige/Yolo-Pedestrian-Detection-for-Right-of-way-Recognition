import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import render_overlay  # noqa: E402


class NearestMaskFrameTest(unittest.TestCase):
    def test_matches_the_closest_segmented_frame_and_prefers_the_earlier_on_a_tie(self):
        frames = [1, 4, 7, 10]
        for number in range(0, 14):
            expected = min(frames, key=lambda f: abs(f - number))  # the old linear scan
            self.assertEqual(render_overlay.nearest_frame(frames, number), expected, number)


class TextBandTest(unittest.TestCase):
    def test_band_reaches_two_font_sizes_below_the_last_line_drawn(self):
        self.assertEqual(render_overlay.text_band_height(font_size=25, banners=0, frame_height=720), 6 + 50)
        self.assertEqual(render_overlay.text_band_height(font_size=25, banners=2, frame_height=720), 40 + 35 + 50)
        self.assertEqual(render_overlay.text_band_height(font_size=25, banners=40, frame_height=720), 720)


if __name__ == "__main__":
    unittest.main()
