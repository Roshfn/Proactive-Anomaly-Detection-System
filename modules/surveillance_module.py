"""
surveillance_module.py
======================
Module 1 — handles all three non-fire anomalies:

  • Person tracking  (YOLOv8 + OSNet feature coupling + cascade matching)
  • Loitering        (10 s in user-defined polygon → bbox green→red)
  • Intrusion        (person bbox overlaps restricted zone)
  • Abandonment      (luggage owner walks away + 10 s timer → yellow→red bbox)

Design goals
------------
- Stable IDs: appearance gallery (mean of last 10 features) + IoU gating.
- ATM false-positive filter: oscillating non-persons are suppressed.
- Tracklet trail drawn on frame so trajectory is visible.
- Visual colour coding:
    Person bbox   green  → normal
                  red    → loitering / intrusion alert
    Luggage bbox  yellow → owned
                  red    → abandoned alert
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from modules.tracklet    import Tracklet
from modules.matching    import cascade_match
from modules.algorithms  import bbox_iou as _bbox_iou, bbox_polygon_overlap, LuggageState


# ═══════════════════════════════════════════════════════════════════════════
#  OSNet re-ID feature extractor (lightweight, no torchreid dependency)
# ═══════════════════════════════════════════════════════════════════════════


# ── decorator guard ────────────────────────────────────────────────────────
def _no_grad(fn):
    """@torch.no_grad() wrapper that degrades gracefully when torch is stubbed."""
    try:
        import torch as _t
        return _t.no_grad()(fn)
    except Exception:
        return fn


class _OSNetExtractor:
    """
    Thin wrapper that:
      1. Crops person patches from the frame.
      2. Resizes to 256×128 (standard re-ID input).
      3. Runs OSNet (loaded via torch.hub from torchreid).
      4. L2-normalises output embeddings.

    Falls back to colour-histogram features if OSNet is unavailable so the
    system still works without torchreid installed.
    """

    INPUT_SIZE = (256, 128)   # (H, W)
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._model: Optional[torch.nn.Module] = None
        self._use_histogram = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torchreid
            self._model = torchreid.models.build_model(
                name="osnet_x0_25", num_classes=1000, pretrained=True
            )
            self._model.eval().to(self.device)
            print("[SurveillanceModule] OSNet loaded via torchreid.")
        except Exception:
            try:
                # Try loading via torch.hub as fallback
                self._model = torch.hub.load(
                    "KaiyangZhou/deep-person-reid", "osnet_x0_25",
                    pretrained=True, verbose=False
                )
                self._model.eval().to(self.device)
                print("[SurveillanceModule] OSNet loaded via torch.hub.")
            except Exception as e:
                print(f"[SurveillanceModule] OSNet unavailable ({e}); "
                      f"using colour-histogram features (128-d).")
                self._use_histogram = True

    @_no_grad
    def extract(self, frame_bgr: np.ndarray,
                bboxes: List[List[float]]) -> np.ndarray:
        """
        Returns (N, D) float32 feature matrix, L2-normalised.
        D = 512 for OSNet, 128 for histogram fallback.
        """
        if not bboxes:
            return np.empty((0, 128 if self._use_histogram else 512),
                            dtype=np.float32)

        if self._use_histogram:
            return self._histogram_features(frame_bgr, bboxes)

        patches = self._crop_patches(frame_bgr, bboxes)
        tensor = self._preprocess(patches).to(self.device)
        feats = self._model(tensor)         # (N, 512)
        feats = feats.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6
        return feats / norms

    # ── helpers ────────────────────────────────────────────────────────────

    def _crop_patches(self, frame: np.ndarray,
                      bboxes: List[List[float]]) -> List[np.ndarray]:
        H, W = frame.shape[:2]
        patches = []
        for b in bboxes:
            x1, y1, x2, y2 = (int(max(0, b[0])), int(max(0, b[1])),
                               int(min(W, b[2])), int(min(H, b[3])))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((self.INPUT_SIZE[0], self.INPUT_SIZE[1], 3),
                                dtype=np.uint8)
            crop = cv2.resize(crop, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            patches.append(crop)
        return patches

    def _preprocess(self, patches: List[np.ndarray]) -> torch.Tensor:
        import torchvision.transforms.functional as F_tv
        from PIL import Image
        tensors = []
        for p in patches:
            t = F_tv.to_tensor(Image.fromarray(p))
            t = F_tv.normalize(t, self.MEAN, self.STD)
            tensors.append(t)
        return torch.stack(tensors)

    def _histogram_features(self, frame: np.ndarray,
                             bboxes: List[List[float]]) -> np.ndarray:
        H, W = frame.shape[:2]
        feats = []
        for b in bboxes:
            x1, y1, x2, y2 = (int(max(0, b[0])), int(max(0, b[1])),
                               int(min(W, b[2])), int(min(H, b[3])))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                feats.append(np.zeros(128, dtype=np.float32))
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
            feat = np.concatenate([h_hist, s_hist]).astype(np.float32)
            norm = np.linalg.norm(feat) + 1e-6
            feats.append(feat / norm)
        return np.stack(feats)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Module
# ═══════════════════════════════════════════════════════════════════════════

LUGGAGE_CLASSES = {"handbag", "backpack", "suitcase", "bag", "umbrella"}
PERSON_CLASS    = "person"

# re-export for tests
_bbox_polygon_overlap = bbox_polygon_overlap
_LuggageState         = LuggageState

# colour palette (BGR)
COL_GREEN  = (0,   220,  50)
COL_RED    = (0,    30, 230)
COL_YELLOW = (0,   200, 255)
COL_WHITE  = (255, 255, 255)
COL_ORANGE = (0,   140, 255)
COL_CYAN   = (255, 200,   0)


class SurveillanceModule:
    """
    Tracks people + luggage and detects loitering, intrusion, abandonment.

    Usage
    -----
    mod = SurveillanceModule(device="cuda")
    mod.set_intrusion_zone([(x1,y1), (x2,y2), ...])
    mod.set_loitering_zone([(x1,y1), (x2,y2), ...])

    result_frame, alerts = mod.process_frame(frame, timestamp)
    """

    # ── init ───────────────────────────────────────────────────────────────

    def __init__(
        self,
        yolo_model:      str   = "yolov8n.pt",
        device:          str   = "cpu",
        conf_person:     float = 0.30,
        conf_luggage:    float = 0.35,
        max_track_age:   int   = 60,
        min_hits:        int   = 1,
        loiter_seconds:  float = 6.0,
        abandon_seconds: float = 5.0,
        intrusion_ratio: float = 0.45,
    ):
        self._device = device

        print("[SurveillanceModule] Loading YOLOv8 …")
        self._yolo = YOLO(yolo_model)
        self._conf_person  = conf_person
        self._conf_luggage = conf_luggage

        print("[SurveillanceModule] Loading feature extractor …")
        self._extractor = _OSNetExtractor(device)

        # Tracker state
        self._tracklets: List[Tracklet] = []
        self._frame_id:  int = 0

        # Hyper-params
        self._max_age        = max_track_age
        self._min_hits       = min_hits
        self._loiter_thresh  = loiter_seconds
        self._abandon_thresh = abandon_seconds
        self._intrusion_ratio = intrusion_ratio

        # Zones (list of (x,y) tuples or None)
        self._intrusion_zone: Optional[List[Tuple[float, float]]] = None
        self._loitering_zone: Optional[List[Tuple[float, float]]] = None

        # Luggage tracking
        self._luggage: Dict[int, _LuggageState] = {}
        self._lug_id_counter = 0

        print("[SurveillanceModule] Ready.\n")

    # ── zone configuration ────────────────────────────────────────────────

    def set_intrusion_zone(self, vertices: List[Tuple[float, float]]) -> None:
        self._intrusion_zone = list(vertices)
        print(f"[SurveillanceModule] Intrusion zone set ({len(vertices)} pts).")

    def set_loitering_zone(self, vertices: List[Tuple[float, float]]) -> None:
        self._loitering_zone = list(vertices)
        print(f"[SurveillanceModule] Loitering zone set ({len(vertices)} pts).")

    def clear_zones(self) -> None:
        self._intrusion_zone = None
        self._loitering_zone = None

    # ── main processing entry point ────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Process one frame.

        Returns
        -------
        annotated_frame : frame with bboxes / trails drawn
        alerts          : list of alert dicts
        """
        if timestamp is None:
            timestamp = time.time()

        self._frame_id += 1
        vis = frame.copy()
        alerts: List[dict] = []

        # 1. Detect persons + luggage
        person_bboxes, luggage_detections = self._detect(frame)

        # 2. Extract appearance features for persons
        if person_bboxes:
            feats = self._extractor.extract(frame, person_bboxes)
        else:
            feats = np.empty((0, 512), dtype=np.float32)

        # 3. Update tracklets
        self._update_tracklets(person_bboxes, feats)

        # 4. Confirmed tracks (past min_hits, not false positive)
        active = [t for t in self._tracklets
                  if t.confirmed and not t.is_false_positive
                  and t.time_since_update == 0]

        # 5. Loitering + intrusion per tracklet
        for t in active:
            self._check_intrusion(t, timestamp, alerts)
            self._check_loitering(t, timestamp, alerts)

        # 6. Luggage ownership + abandonment
        # Use ALL recently seen tracklets (not just time_since_update==0).
        # A person 1-2 frames stale is still a valid owner — if we only
        # pass 'active' the owner lookup returns None immediately and the
        # abandon timer starts while the person is still standing there.
        recent_persons = [
            t for t in self._tracklets
            if t.time_since_update <= 5 and not t.is_false_positive
        ]
        self._update_luggage(luggage_detections, recent_persons, timestamp, alerts)

        # 7. Draw everything
        self._draw(vis, active, alerts)

        return vis, alerts

    # ── detection ─────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray):
        """Run YOLOv8, apply NMS, split into persons vs luggage."""
        results = self._yolo(frame, verbose=False)[0]
        person_raw: List[List[float]] = []
        person_confs: List[float] = []
        luggage_dets: List[dict] = []

        for box in results.boxes:
            cls_name = self._yolo.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf)

            if cls_name == PERSON_CLASS and conf >= self._conf_person:
                person_raw.append([x1, y1, x2, y2])
                person_confs.append(conf)
            elif cls_name in LUGGAGE_CLASSES and conf >= self._conf_luggage:
                luggage_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "cls":  cls_name,
                    "conf": conf,
                })

        # ── NMS on person detections to remove duplicates ─────────────
        # YOLOv8 already runs NMS internally, but with aggressive settings
        # some overlapping boxes still slip through — we re-apply here.
        person_bboxes = self._nms(person_raw, person_confs, iou_thresh=0.45)
        return person_bboxes, luggage_dets

    @staticmethod
    def _nms(bboxes: List[List[float]], confs: List[float],
             iou_thresh: float = 0.45) -> List[List[float]]:
        """Simple greedy NMS — keeps highest-confidence box in each cluster."""
        if not bboxes:
            return []
        order = sorted(range(len(bboxes)), key=lambda i: confs[i], reverse=True)
        kept: List[int] = []
        suppressed = set()
        for i in order:
            if i in suppressed:
                continue
            kept.append(i)
            for j in order:
                if j == i or j in suppressed:
                    continue
                if _bbox_iou(bboxes[i], bboxes[j]) > iou_thresh:
                    suppressed.add(j)
        return [bboxes[i] for i in kept]

    # ── tracklet lifecycle ─────────────────────────────────────────────────

    def _update_tracklets(
        self,
        bboxes: List[List[float]],
        feats:  np.ndarray,
    ) -> None:
        # Match detections to existing tracks
        matches, unmatched_trks, unmatched_dets = cascade_match(
            self._tracklets, bboxes, feats
        )

        # Update matched
        for ti, di in matches:
            self._tracklets[ti].update(bboxes[di], feats[di], self._frame_id)

        # Mark missed
        for ti in unmatched_trks:
            self._tracklets[ti].mark_missed()

        # Birth new tracklets
        for di in unmatched_dets:
            t = Tracklet(bboxes[di], feats[di], self._frame_id)
            self._tracklets.append(t)

        # Confirm + prune
        for t in self._tracklets:
            if t.hits >= self._min_hits:
                t.confirmed = True

        self._tracklets = [
            t for t in self._tracklets
            if t.time_since_update <= self._max_age
        ]

    # ── anomaly checks ────────────────────────────────────────────────────

    def _check_intrusion(self, t: Tracklet, now: float,
                          alerts: List[dict]) -> None:
        if self._intrusion_zone is None or t.bbox is None:
            t.intruding = False
            return
        ratio = _bbox_polygon_overlap(t.bbox, self._intrusion_zone)
        t.intruding = ratio >= self._intrusion_ratio
        if t.intruding:
            alerts.append({
                "type":       "INTRUSION",
                "track_id":   t.track_id,
                "ratio":      round(ratio, 2),
                "bbox":       t.bbox,
                "timestamp":  now,
            })

    def _check_loitering(self, t: Tracklet, now: float,
                          alerts: List[dict]) -> None:
        # ONLY check loitering when a zone has been explicitly defined
        if self._loitering_zone is None:
            t.update_loiter_in_zone(False, now, self._loiter_thresh)
            return

        if t.bbox is None:
            return

        ratio = _bbox_polygon_overlap(t.bbox, self._loitering_zone)
        in_zone = ratio >= 0.30
        t.update_loiter_in_zone(in_zone, now, self._loiter_thresh)

        if t.is_loitering:
            alerts.append({
                "type":      "LOITERING",
                "track_id":  t.track_id,
                "duration":  round(t.loiter_duration, 1),
                "bbox":      t.bbox,
                "timestamp": now,
            })

    # ── luggage ───────────────────────────────────────────────────────────

    def _update_luggage(
        self,
        dets: List[dict],
        persons: List[Tracklet],
        now: float,
        alerts: List[dict],
    ) -> None:
        """Top-down luggage ownership + abandonment detection."""
        if not dets:
            # Age out stale luggage
            self._luggage = {
                k: v for k, v in self._luggage.items()
                if now - v.last_seen < 15.0
            }
            return

        # Match detected luggage to existing luggage states by IoU
        used_det = set()
        for lug_id, state in list(self._luggage.items()):
            best_iou, best_det_idx = 0.0, -1
            for i, det in enumerate(dets):
                if i in used_det:
                    continue
                from modules.matching import _iou as iou_fn
                score = iou_fn(state.bbox, det["bbox"])
                if score > best_iou:
                    best_iou, best_det_idx = score, i
            if best_iou > 0.05 and best_det_idx >= 0:
                used_det.add(best_det_idx)
                owner = self._find_owner(dets[best_det_idx]["bbox"], persons)
                state.update(dets[best_det_idx]["bbox"], owner, now,
                             self._abandon_thresh)
                if state.is_abandoned:
                    alerts.append({
                        "type":       "ABANDONMENT",
                        "lug_id":     lug_id,
                        "owner_id":   state.owner_id,
                        "cls":        state.cls_name,
                        "duration":   round(state.abandon_duration, 1),
                        "bbox":       state.bbox,
                        "timestamp":  now,
                    })
            elif now - state.last_seen > 15.0:
                del self._luggage[lug_id]

        # New luggage detections
        for i, det in enumerate(dets):
            if i in used_det:
                continue
            owner = self._find_owner(det["bbox"], persons)
            state = _LuggageState(det["bbox"], det["cls"], owner)
            state.last_seen = now
            self._luggage[self._lug_id_counter] = state
            self._lug_id_counter += 1
            # Register with owner tracklet
            if owner is not None:
                for t in persons:
                    if t.track_id == owner:
                        t.owned_object_ids.add(self._lug_id_counter - 1)

    def _find_owner(self, lug_bbox: List[float],
                    persons: List[Tracklet]) -> Optional[int]:
        """
        Assign luggage to nearest person.
        Threshold = min(1.5 x person_height, 120px hard cap).
        Once person walks more than ~2 steps away the bag is unowned
        and the abandon timer starts.
        """
        lc_x = (lug_bbox[0] + lug_bbox[2]) / 2
        lc_y = (lug_bbox[1] + lug_bbox[3]) / 2
        best_dist, best_id = float("inf"), None
        for t in persons:
            if t.bbox is None:
                continue
            ph = t.bbox[3] - t.bbox[1]          # person height
            pc_x = (t.bbox[0] + t.bbox[2]) / 2
            pc_y = (t.bbox[1] + t.bbox[3]) / 2
            dist = np.hypot(lc_x - pc_x, lc_y - pc_y)
            threshold = min(1.5 * ph, 120.0)
            if dist < threshold and dist < best_dist:
                best_dist, best_id = dist, t.track_id
        return best_id

    # ── drawing ───────────────────────────────────────────────────────────

    def _draw(self, vis: np.ndarray, active: List[Tracklet],
              alerts: List[dict]) -> None:
        # Collect alert track IDs for fast lookup
        intrusion_ids  = {a["track_id"] for a in alerts if a["type"] == "INTRUSION"}
        loitering_ids  = {a["track_id"] for a in alerts if a["type"] == "LOITERING"}
        abandon_lug_ids = {a["lug_id"]  for a in alerts if a["type"] == "ABANDONMENT"}

        # Draw intrusion / loitering zone outlines
        self._draw_zone(vis, self._intrusion_zone, (0, 0, 200), "INTRUSION ZONE")
        self._draw_zone(vis, self._loitering_zone, (0, 200, 0), "LOITER ZONE")

        # Draw persons
        for t in active:
            if t.bbox is None:
                continue
            is_alert = (t.track_id in intrusion_ids or
                        t.track_id in loitering_ids)
            colour = COL_RED if is_alert else COL_GREEN
            self._draw_person(vis, t, colour, intrusion_ids, loitering_ids)

        # Draw luggage
        for lug_id, state in self._luggage.items():
            is_abandoned = lug_id in abandon_lug_ids
            self._draw_luggage(vis, lug_id, state, is_abandoned)

    def _draw_person(self, vis, t, colour, intrusion_ids, loitering_ids):
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        thick = 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thick)

        # Label
        label_parts = [f"ID:{t.track_id}"]
        if t.track_id in loitering_ids:
            label_parts.append(f"LOITER {t.loiter_duration:.0f}s")
        if t.track_id in intrusion_ids:
            label_parts.append("INTRUSION!")
        label = "  ".join(label_parts)

        lh = 16
        cv2.rectangle(vis, (x1, y1 - lh - 2), (x1 + len(label) * 7 + 4, y1),
                      colour, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_WHITE, 1, cv2.LINE_AA)

        # Trajectory trail
        trail = t.trajectory_centers(max_pts=40)
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            tc = tuple(int(c * alpha + (1 - alpha) * 80) for c in colour)
            cv2.line(vis,
                     (int(trail[i-1][0]), int(trail[i-1][1])),
                     (int(trail[i][0]),   int(trail[i][1])),
                     tc, 1, cv2.LINE_AA)

    def _draw_luggage(self, vis, lug_id, state: _LuggageState,
                      is_abandoned: bool) -> None:
        x1, y1, x2, y2 = [int(v) for v in state.bbox]
        colour = COL_RED if is_abandoned else COL_YELLOW
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        parts = [state.cls_name]
        if state.owner_id is not None:
            parts.append(f"owner:{state.owner_id}")
        if is_abandoned:
            parts.append(f"ABANDONED {state.abandon_duration:.0f}s")
        label = " ".join(parts)
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_zone(vis, zone, colour, label):
        if zone is None:
            return
        pts = np.array(zone, dtype=np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
        cv2.polylines(vis, [pts], isClosed=True, color=colour, thickness=2)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(vis, label, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._tracklets = []
        self._luggage   = {}
        self._frame_id  = 0
        Tracklet.reset_counter()