"""
Tracklet — per-person state machine.

Holds bounding-box history, feature gallery, trajectory points, loitering
timer, ATM-based false-positive filter and luggage-ownership bookkeeping.
"""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _iou(a: List[float], b: List[float]) -> float:
    """Intersection-over-Union of two axis-aligned boxes [x1,y1,x2,y2]."""
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _atm(traj: List[Tuple[float, float]]) -> float:
    """
    Area of Trajectory Movement — bounding-box span of the trajectory.

    Concept from paper Eq. 8-9 (trajectory polygon area), implemented as
    bounding-box area for robustness:

    * Stationary oscillator: moves a few pixels  ->  tiny bbox -> ATM near 0.
    * Real walker crossing the frame: large bbox  ->  ATM >> threshold.

    The raw shoelace formula fails for straight-line walkers (polygon area = 0
    for any collinear sequence). Bounding-box span handles all motion types.
    """
    if len(traj) < 3:
        return 0.0
    pts = np.asarray(traj, dtype=np.float64)
    return float((pts[:, 0].max() - pts[:, 0].min()) *
                 (pts[:, 1].max() - pts[:, 1].min()))


# ────────────────────────────────────────────────────────────────────────────
# Tracklet
# ────────────────────────────────────────────────────────────────────────────

class Tracklet:
    """Single-person track with full anomaly state."""

    _id_counter: int = 0

    # ATM thresholds (pixels²) — a real person walking even 20 px covers
    # ~200 px² of area; stationary oscillation barely reaches 50.
    ATM_MIN_AREA: float = 4.0
    ATM_EVAL_FRAMES: int = 40   # evaluate after this many frames

    def __init__(self, bbox: List[float], feature: np.ndarray, frame_id: int):
        self.track_id: int = Tracklet._id_counter
        Tracklet._id_counter += 1

        # Detection history (capped at 90 frames ≈ 3 s @ 30 fps)
        self._bboxes: deque = deque(maxlen=90)
        self._features: deque = deque(maxlen=10)   # rolling feature gallery
        self._frame_ids: deque = deque(maxlen=90)

        self._bboxes.append(list(bbox))
        self._features.append(feature)
        self._frame_ids.append(frame_id)

        # Tracking lifecycle
        self.hits: int = 1
        self.age: int = 1
        self.time_since_update: int = 0
        self.confirmed: bool = False          # set by tracker once min_hits reached

        # ── Anomaly state ──────────────────────────────────────────────────
        self.loiter_start: Optional[float] = None   # wall-clock seconds
        self.loiter_duration: float = 0.0
        self.is_loitering: bool = False

        self.intruding: bool = False

        # Luggage ownership assigned by LuggageTracker
        self.owned_object_ids: set = set()

        # Stationary / ATM
        self.is_stationary: bool = False
        self._stationary_count: int = 0

        # False-positive filter result
        self.is_false_positive: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def update(self, bbox: List[float], feature: np.ndarray, frame_id: int) -> None:
        self._bboxes.append(list(bbox))
        self._features.append(feature)
        self._frame_ids.append(frame_id)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self._update_stationary()
        self._update_false_positive_filter()

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.age += 1
        # Only reset loitering if truly gone a long time (not brief occlusion)
        if self.time_since_update > 20:
            self.loiter_start = None
            self.loiter_duration = 0.0
            self.is_loitering = False

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def bbox(self) -> Optional[List[float]]:
        return list(self._bboxes[-1]) if self._bboxes else None

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        b = self.bbox
        if b is None:
            return None
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    @property
    def feature(self) -> Optional[np.ndarray]:
        if not self._features:
            return None
        # Exponential moving average — recent appearance weighted 2× older
        # This makes the gallery adapt quickly after occlusion/merge events,
        # reducing ID swaps when people separate after being close together.
        feats = list(self._features)
        n = len(feats)
        alpha = 0.35   # EMA decay — 0.35 means ~3 frames half-life
        ema = np.array(feats[0], dtype=np.float64)
        for i in range(1, n):
            ema = alpha * np.array(feats[i], dtype=np.float64) + (1 - alpha) * ema
        norm = np.linalg.norm(ema)
        return (ema / (norm + 1e-6)).astype(np.float32)

    # ── Trajectory ────────────────────────────────────────────────────────

    def trajectory_centers(self, max_pts: int = 60) -> List[Tuple[float, float]]:
        """Centre points of recent bboxes — used for drawing the trail."""
        pts = []
        for b in list(self._bboxes)[-max_pts:]:
            pts.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        return pts

    def trajectory_topleft(self, max_pts: int = 60) -> List[Tuple[float, float]]:
        """Top-left coords — used for ATM calculation per paper."""
        return [(b[0], b[1]) for b in list(self._bboxes)[-max_pts:]]

    # ── ATM false-positive filter ─────────────────────────────────────────

    def _update_false_positive_filter(self) -> None:
        """Flag this tracklet as a false positive if ATM area is too small."""
        if self.hits < self.ATM_EVAL_FRAMES:
            return  # not enough history yet
        tl = self.trajectory_topleft(self.ATM_EVAL_FRAMES)
        area = _atm(tl)
        self.is_false_positive = (area < self.ATM_MIN_AREA)

    # ── Stationary detection ──────────────────────────────────────────────

    def _update_stationary(self, iou_thresh: float = 0.88, window: int = 8) -> None:
        if len(self._bboxes) < window:
            self.is_stationary = False
            return
        recent = list(self._bboxes)[-window:]
        ious = [_iou(recent[i], recent[i + 1]) for i in range(len(recent) - 1)]
        if np.mean(ious) >= iou_thresh:
            self._stationary_count += 1
            self.is_stationary = self._stationary_count >= window
        else:
            self._stationary_count = 0
            self.is_stationary = False

    # ── Loitering ─────────────────────────────────────────────────────────

    def update_loiter_in_zone(self, in_zone: bool, now: float,
                               loiter_threshold: float = 10.0) -> None:
        if in_zone:
            if self.loiter_start is None:
                self.loiter_start = now
            self.loiter_duration = now - self.loiter_start
            self.is_loitering = (self.loiter_duration >= loiter_threshold)
        else:
            self.loiter_start = None
            self.loiter_duration = 0.0
            self.is_loitering = False

    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"Tracklet(id={self.track_id}, hits={self.hits}, "
                f"loitering={self.is_loitering}, fp={self.is_false_positive})")

    @classmethod
    def reset_counter(cls) -> None:
        cls._id_counter = 0