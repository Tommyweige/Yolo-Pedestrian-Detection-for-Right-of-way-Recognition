"""Crosswalk coordinates: judge pedestrians and vehicles against the crosswalk itself, not in image pixels.

A crosswalk mask is fitted with a quadrilateral. Its two long edges bound the band that vehicles drive across; a
homography maps the quadrilateral to the unit square, so a point gets (L, W): L runs along the crosswalk (the way
pedestrians walk), W across it, 0 and 1 on the two long edges. Being inside the band is then 0 <= W <= 1 however
foreshortened the crosswalk is in the image, so "a person-height from the edge" and "a strip below the vehicle" are not needed.
Pure geometry, no calibration: lengths along L are compared with a vehicle's own width mapped into the same space.
"""
from dataclasses import dataclass

import cv2
import numpy as np

MIN_AREA_PX = 30  # smaller masks cannot give a usable quadrilateral


def _quad_of(component):
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    perimeter = cv2.arcLength(hull, True)
    for epsilon in np.linspace(0.005, 0.2, 80):
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True).reshape(-1, 2)
        if len(approx) == 4:
            return approx.astype(np.float32)
        if len(approx) < 4:
            break
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)  # a triangle or a ragged hull: fall back to a rectangle


def _ordered(points):
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    return points[np.argsort(angles)]


@dataclass
class CrosswalkSpace:
    matrix: np.ndarray       # image (x, y) -> (L, W)
    quad: np.ndarray         # the fitted corners, image pixels
    clipped: bool            # the crosswalk runs off the image: its ends (L) are not real ends

    def to_crosswalk(self, x, y):
        point = self.matrix @ np.array([x, y, 1.0])
        return float(point[0] / point[2]), float(point[1] / point[2])

    def to_crosswalk_many(self, points):
        pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)


def fit_space(component):
    """CrosswalkSpace for one crosswalk component (bool mask), or None when it is too small to fit."""
    if component.sum() < MIN_AREA_PX:
        return None
    quad = _ordered(_quad_of(component))
    edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    if edges[0] + edges[2] >= edges[1] + edges[3]:   # long edges are 0-1 and 2-3: W runs from edge 0-1 to edge 2-3
        ring = [quad[0], quad[1], quad[2], quad[3]]
    else:
        ring = [quad[1], quad[2], quad[3], quad[0]]
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(np.array(ring, np.float32), unit)
    rows, cols = component.shape
    on_border = int(quad[:, 0].min() <= 1) + int(quad[:, 0].max() >= cols - 2) + int(quad[:, 1].min() <= 1) + int(quad[:, 1].max() >= rows - 2)
    return CrosswalkSpace(matrix=matrix, quad=quad, clipped=on_border > 0)


def merge_components(components, max_hull_ratio=1.6):
    """Join pieces of one crosswalk that a standing pedestrian or vehicle cut apart.

    Two pieces are one crosswalk when the convex hull of both is hardly larger than the pieces themselves; a near and a
    far crosswalk give a hull several times larger and stay separate.
    """
    pieces = [c.copy() for c in components]
    merged = True
    while merged and len(pieces) > 1:
        merged = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                union = pieces[i] | pieces[j]
                ys, xs = np.nonzero(union)
                hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))
                if cv2.contourArea(hull) <= max_hull_ratio * union.sum():
                    pieces[i] = union
                    del pieces[j]
                    merged = True
                    break
            if merged:
                break
    return pieces
