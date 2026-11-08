import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import render_overlay  # noqa: E402


def _write_video(path, frames, size=(640, 360), fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()


def _write_masks(path, masks):
    np.savez_compressed(path, **{f"f{frame}": np.packbits(mask.ravel()) for frame, mask in masks.items()})


class OverlayMaskTest(unittest.TestCase):
    @unittest.skipUnless(Path(render_overlay.FONT_PATH).is_file(), "overlay font is Windows-only")
    def test_overlay_tints_with_the_mask_the_rule_uses_not_the_nearest_one(self):
        """Frame 4 sits between masks 1 (empty) and 5 (full). The rule uses mask 1 there, so the overlay must too:
        the nearest mask (5) would show a crosswalk the rule never saw."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_video(tmp / "clip.mp4", 6)
            empty, full = np.zeros((360, 640), bool), np.ones((360, 640), bool)
            _write_masks(tmp / "masks.npz", {1: empty, 5: full})
            (tmp / "tracks.json").write_text("[]", encoding="utf-8")
            self.assertEqual(render_overlay.main(["--video", str(tmp / "clip.mp4"), "--masks", str(tmp / "masks.npz"),
                                                  "--tracks", str(tmp / "tracks.json"), "--width", "640",
                                                  "--output", str(tmp / "out.mp4")]), 0)
            capture = cv2.VideoCapture(str(tmp / "out.mp4"))
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            capture.release()
            green = [float(f[200:, :, 1].mean()) for f in frames]  # below the text band
            self.assertLess(green[3], 20, "frame 4 must not be tinted: the rule's mask there is frame 1's (empty)")
            self.assertGreater(green[4], 100, "frame 5 has the full mask")


class TextBandTest(unittest.TestCase):
    def test_band_reaches_two_font_sizes_below_the_last_line_drawn(self):
        self.assertEqual(render_overlay.text_band_height(font_size=25, banners=0, frame_height=720), 6 + 50)
        self.assertEqual(render_overlay.text_band_height(font_size=25, banners=2, frame_height=720), 40 + 35 + 50)
        self.assertEqual(render_overlay.text_band_height(font_size=25, banners=40, frame_height=720), 720)


if __name__ == "__main__":
    unittest.main()
