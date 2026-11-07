"""crosswalk_space: fit a crosswalk quadrilateral and express points in crosswalk coordinates (no GPU, synthetic masks)."""
import unittest

import cv2
import numpy as np

from crosswalk_space import fit_space, merge_components

H, W = 360, 640


def poly_mask(*polys):
    m = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p, np.int32)], 1)
    return m.astype(bool)


# a perspective crosswalk: far (narrow) edge at y=200, near (wide) edge at y=300; it is crossed along x
TRAPEZOID = [(150, 200), (450, 200), (600, 300), (20, 300)]


class FitAndProject(unittest.TestCase):
    def test_the_band_edges_are_zero_and_one_across_the_crosswalk(self):
        space = fit_space(poly_mask(TRAPEZOID))
        _, far = space.to_crosswalk(300, 202)    # just inside the far edge
        _, near = space.to_crosswalk(300, 298)   # just inside the near edge
        _, mid = space.to_crosswalk(300, 250)
        self.assertAlmostEqual(min(far, near), 0.0, delta=0.06)
        self.assertAlmostEqual(max(far, near), 1.0, delta=0.06)
        self.assertGreater(mid, 0.3), self.assertLess(mid, 0.7)

    def test_a_point_beyond_either_long_edge_is_outside_the_band(self):
        space = fit_space(poly_mask(TRAPEZOID))
        for y in (150, 340):
            _, w = space.to_crosswalk(300, y)
            self.assertTrue(w < -0.1 or w > 1.1, (y, w))   # perspective: 40 px below the near edge is only ~0.17 widths

    def test_a_thin_far_strip_still_has_a_width_of_one(self):
        # grazing view: only 10 px thick, so "person heights" would be hopeless but crosswalk widths are not
        strip = [(180, 150), (330, 150), (340, 160), (170, 160)]
        space = fit_space(poly_mask(strip))
        _, inside = space.to_crosswalk(250, 157)
        _, above = space.to_crosswalk(250, 140)   # a person-height (~50 px) above the strip is far outside
        self.assertTrue(0.0 <= inside <= 1.0)
        self.assertGreater(abs(above - 0.5), 1.0)

    def test_a_crosswalk_cut_by_the_image_edge_keeps_a_usable_width(self):
        wide = [(-200, 200), (840, 200), (900, 300), (-260, 300)]   # runs off both sides of the 640 px image
        space = fit_space(poly_mask(wide))
        self.assertTrue(space.clipped)
        _, w = space.to_crosswalk(300, 250)
        self.assertGreater(w, 0.3), self.assertLess(w, 0.7)

    def test_a_piece_cut_by_the_image_edge_only_claims_what_it_shows(self):
        # a crosswalk strip running off the right edge: it has one real end, at x=300. On a real clip, letting a
        # clipped piece claim "within half its longest edge" put vehicles 135 px outside it on the crosswalk.
        space = fit_space(poly_mask([(300, 200), (639, 200), (639, 260), (300, 260)]))
        self.assertTrue(space.clipped)
        for x, y in ((400, 230), (620, 230)):        # on the visible piece
            self.assertTrue(space.along(x, y, *space.to_crosswalk(x, y)))
        for x, y in ((200, 230), (120, 230)):        # in the same band, 100 and 180 px past its real end
            self.assertFalse(space.along(x, y, *space.to_crosswalk(x, y)))

    def test_a_pedestrian_standing_in_the_crosswalk_does_not_split_it(self):
        left = [(150, 200), (290, 200), (270, 300), (20, 300)]
        right = [(310, 200), (450, 200), (600, 300), (290, 300)]
        components = [poly_mask(left), poly_mask(right)]
        merged = merge_components(components)
        self.assertEqual(len(merged), 1)
        space = fit_space(merged[0])
        _, w = space.to_crosswalk(300, 250)
        self.assertTrue(0.3 < w < 0.7)

    def test_two_different_crosswalks_are_not_merged(self):
        near = poly_mask([(20, 300), (600, 300), (620, 345), (10, 345)])
        far = poly_mask([(250, 60), (400, 60), (405, 70), (245, 70)])
        self.assertEqual(len(merge_components([near, far])), 2)

    def test_too_small_or_empty_masks_give_no_space(self):
        self.assertIsNone(fit_space(np.zeros((H, W), bool)))


if __name__ == "__main__":
    unittest.main()
