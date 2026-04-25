"""
algorithms.py — Pure-Python geometry and scoring helpers.

Zero dependencies beyond numpy.  Imported by both production modules
and the smoke-test suite.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# IoU
# ──────────────────────────────────────────────────────────────────────────

def bbox_iou(a: List[float], b: List[float]) -> float:
    """Axis-aligned IoU of two boxes [x1,y1,x2,y2]."""
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ──────────────────────────────────────────────────────────────────────────
# ATM  (Area of Trajectory Movement)  — paper Eq. 8–9
# ──────────────────────────────────────────────────────────────────────────

def atm(trajectory: List[Tuple[float, float]]) -> float:
    """
    Shoelace area of the polygon formed by trajectory centre-points.

    Self-intersecting paths (oscillation, ID-switch back-and-forth) yield
    near-zero because the loop areas cancel.  Genuine walkers accumulate
    positive non-overlapping area.

    Works best with ≥ 10 points.  Returns 0 for < 3 points.
    """
    n = len(trajectory)
    if n < 3:
        return 0.0
    pts = np.asarray(trajectory, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    # Signed area via cross product (Gauss / shoelace)
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) -
                            np.dot(y, np.roll(x, -1))))
    return area


# ──────────────────────────────────────────────────────────────────────────
# Bbox ↔ polygon overlap  (intrusion / loitering)
# ──────────────────────────────────────────────────────────────────────────

def bbox_polygon_overlap(
    bbox: List[float],
    poly: List[Tuple[float, float]],
) -> float:
    """
    Fraction of `bbox` area that lies inside polygon `poly`.

    Uses OpenCV rasterisation — works for convex *and* concave polygons.
    Returns value in [0, 1].
    """
    import cv2
    x1, y1 = int(bbox[0]), int(bbox[1])
    x2, y2 = int(bbox[2]), int(bbox[3])
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    # Translate polygon vertices to local (bbox-relative) coordinates
    pts = np.array(
        [(int(px - x1), int(py - y1)) for px, py in poly],
        dtype=np.int32,
    )
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return float(np.clip(mask.sum() / (bw * bh), 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────
# LuggageState  (pure-data, no torch)
# ──────────────────────────────────────────────────────────────────────────

from typing import Optional


class LuggageState:
    """
    Tracks per-item ownership and abandonment timer.

    Lives entirely in Python — no ML dependencies.
    """

    def __init__(
        self,
        bbox:     List[float],
        cls_name: str,
        owner_id: Optional[int],
    ):
        self.bbox         = list(bbox)
        self.cls_name     = cls_name
        self.owner_id     = owner_id
        self.last_seen:   float           = 0.0
        self.abandon_start: Optional[float] = None
        self.abandon_duration: float      = 0.0
        self.is_abandoned: bool           = False

    def update(
        self,
        bbox:           List[float],
        owner_id:       Optional[int],
        now:            float,
        abandon_thresh: float = 10.0,
    ) -> None:
        """Call every frame with the current detection and resolved owner."""
        self.bbox      = list(bbox)
        self.last_seen = now

        if owner_id is None:
            # No owner in sight — start/advance abandonment timer
            if self.abandon_start is None:
                self.abandon_start = now
            self.abandon_duration = now - self.abandon_start
            self.is_abandoned     = self.abandon_duration >= abandon_thresh
        else:
            # Owner present — reset everything
            self.owner_id      = owner_id
            self.abandon_start = None
            self.abandon_duration = 0.0
            self.is_abandoned  = False
