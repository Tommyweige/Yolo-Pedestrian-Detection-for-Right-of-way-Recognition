import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from track_video import drop_ego_parts  # noqa: E402

SIZE = (1280, 720)


def track(kind, y1, y2, frames=range(10), x1=600, x2=760, ident=1):
    return {"id": ident, "kind": kind, "boxes": [[f, x1, y1, x2, y2] for f in frames], "raw": [[f, (x1 + x2) / 2, y2] for f in frames]}


class EgoPartTest(unittest.TestCase):
    def test_low_box_in_the_bottom_strip_is_the_cameras_mirror(self):
        self.assertEqual(drop_ego_parts([track("car", 650, 715)], SIZE), ([], 10))

    def test_a_close_vehicle_fills_the_image_and_is_kept(self):
        self.assertEqual(drop_ego_parts([track("car", 450, 720)], SIZE)[1], 0)

    def test_vehicle_higher_up_is_kept(self):
        self.assertEqual(drop_ego_parts([track("car", 300, 360)], SIZE)[1], 0)

    def test_persons_are_never_removed(self):
        walker = track("person", 650, 715)
        self.assertEqual(drop_ego_parts([walker], SIZE), ([walker], 0))

    def test_only_the_mirror_frames_of_a_track_are_removed(self):
        mixed = track("car", 650, 715, frames=range(5))
        mixed["boxes"] += [[f, 600, 300, 760, 360] for f in range(5, 10)]
        mixed["raw"] += [[f, 680, 360] for f in range(5, 10)]
        kept, removed = drop_ego_parts([mixed], SIZE)
        self.assertEqual((removed, [b[0] for b in kept[0]["boxes"]], [r[0] for r in kept[0]["raw"]]), (5, list(range(5, 10)), list(range(5, 10))))


if __name__ == "__main__":
    unittest.main()
