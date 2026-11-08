import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import measure_crosswalk_margins  # noqa: E402


class MeasureCrosswalkMarginsTest(unittest.TestCase):
    def test_runs_on_a_clip_with_a_pedestrian_and_a_rider(self):
        """A pedestrian on the crosswalk and a person riding a motorcycle: both used to crash the per-frame rider test."""
        mask = np.zeros((360, 640), bool)
        mask[150:250, 40:600] = True
        frames = range(1, 31)
        tracks = [
            {"id": 1, "kind": "person", "boxes": [[f, 600, 300, 660, 600] for f in frames]},     # foot inside the band
            {"id": 2, "kind": "car", "boxes": [[f, 700, 250, 1000, 470] for f in frames]},
            {"id": 3, "kind": "motorcycle", "boxes": [[f, 1300, 300, 1420, 640] for f in frames]},
            {"id": 4, "kind": "person", "boxes": [[f, 1310, 280, 1410, 600] for f in frames]},   # riding 3
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.savez_compressed(tmp / "masks.npz", **{f"f{f}": np.packbits(mask.ravel()) for f in (1, 10, 20)})
            (tmp / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
            code = measure_crosswalk_margins.main(["--masks", str(tmp / "masks.npz"), "--tracks", str(tmp / "tracks.json"),
                                                   "--fps", "30", "--width", "1920", "--height", "1080",
                                                   "--json", str(tmp / "report.json")])
            self.assertEqual(code, 0)
            report = json.loads((tmp / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["on_crosswalk_ground_points_on_the_mask"]["pedestrian"], 20,
                             "only the walking pedestrian counts, on the 20 frames from the first mask's 1 to 20; "
                             "the rider is excluded")


if __name__ == "__main__":
    unittest.main()
