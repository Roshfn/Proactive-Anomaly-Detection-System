"""
matching.py — Appearance + IoU gated cascade matching.

Two-stage matching mirrors Deep SORT / PASS-CCTV:
  1) Appearance (cosine distance on coupled 512-d features), gated by IoU.
  2) IoU-only fall-back for any remaining unmatched pairs (handles occlusion).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ─── distance helpers ──────────────────────────────────────────────────────

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,D) × (M,D) → (N,M) cosine distance matrix."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
    return 1.0 - a @ b.T   # distance ∈ [0, 2]


def _iou(b1, b2) -> float:
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / u


def _iou_distance(bboxes1, bboxes2) -> np.ndarray:
    D = np.zeros((len(bboxes1), len(bboxes2)))
    for i, b1 in enumerate(bboxes1):
        for j, b2 in enumerate(bboxes2):
            D[i, j] = 1.0 - _iou(b1, b2)
    return D


def _hungarian(cost: np.ndarray, thresh: float):
    """Return (row, col) pairs where cost ≤ thresh."""
    rows, cols = linear_sum_assignment(cost)
    valid = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    return valid


# ─── main entry ────────────────────────────────────────────────────────────

def cascade_match(
    tracklets,
    det_bboxes: List[List[float]],
    det_features: np.ndarray,
    max_feat_dist: float = 0.70,
    max_iou_dist: float = 0.85,
    max_age: int = 60,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Two-stage cascade matching.

    Returns
    -------
    matches          : list of (tracklet_idx, det_idx)
    unmatched_trks   : list of tracklet indices with no match
    unmatched_dets   : list of detection indices with no match
    """
    if not tracklets or not det_bboxes:
        return [], list(range(len(tracklets))), list(range(len(det_bboxes)))

    matched: List[Tuple[int, int]] = []
    unmatched_dets: List[int] = list(range(len(det_bboxes)))

    # ── Stage 1: appearance cascade ──────────────────────────────────────
    for age in range(max_age):
        if not unmatched_dets:
            break
        trk_idx = [i for i, t in enumerate(tracklets)
                   if t.time_since_update == age]
        if not trk_idx:
            continue

        trk_feats = np.stack([tracklets[i].feature for i in trk_idx])
        det_feats_sub = det_features[unmatched_dets]
        trk_bboxes = [tracklets[i].bbox for i in trk_idx]
        det_bboxes_sub = [det_bboxes[j] for j in unmatched_dets]

        cost = _cosine_distance(trk_feats, det_feats_sub)

        iou_dist = _iou_distance(trk_bboxes, det_bboxes_sub)

        # Gate 1: boxes too far apart → reject appearance match
        cost[iou_dist > 0.92] = max_feat_dist + 1.0

        # Gate 2: boxes heavily overlapping (two people merged by detector)
        # → rely on position only, appearance is unreliable here
        cost[iou_dist < 0.05] = max_feat_dist + 1.0

        pairs = _hungarian(cost, max_feat_dist)
        for r, c in pairs:
            matched.append((trk_idx[r], unmatched_dets[c]))
        matched_det_local = {c for _, c in pairs}
        unmatched_dets = [d for k, d in enumerate(unmatched_dets)
                          if k not in matched_det_local]

    # ── Stage 2: IoU-only fall-back ──────────────────────────────────────
    if unmatched_dets:
        matched_trk_ids = {t for t, _ in matched}
        remaining_trks = [i for i in range(len(tracklets))
                          if i not in matched_trk_ids
                          and tracklets[i].time_since_update <= 3]
        if remaining_trks:
            trk_bboxes = [tracklets[i].bbox for i in remaining_trks]
            det_bboxes_sub = [det_bboxes[j] for j in unmatched_dets]
            cost = _iou_distance(trk_bboxes, det_bboxes_sub)
            pairs = _hungarian(cost, max_iou_dist)
            for r, c in pairs:
                matched.append((remaining_trks[r], unmatched_dets[c]))
            matched_det_local = {c for _, c in pairs}
            unmatched_dets = [d for k, d in enumerate(unmatched_dets)
                              if k not in matched_det_local]

    matched_trk_ids = {t for t, _ in matched}
    unmatched_trks = [i for i in range(len(tracklets))
                      if i not in matched_trk_ids]

    return matched, unmatched_trks, unmatched_dets