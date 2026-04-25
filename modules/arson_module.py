"""
arson_module.py
===============
Module 2 — Fire and arson detection.

PRIMARY: OpenCV dual-signal detection
  1. HSV colour mask  — fire has orange/yellow/red hue, high saturation
  2. Temporal flicker — fire pixels have high frame-to-frame variance
     (unique signature — no other common CCTV object flickers this way)

Combined mask: colour AND flicker must overlap.
This eliminates false positives from:
  - Orange/yellow high-vis vests (colour present, NO flicker)
  - Traffic lights and signs   (colour present, NO flicker)
  - Sunlight reflections       (brief, < 3 consecutive frames)

SECONDARY: CLIP used as additional boost when available.
"""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    import clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

from ultralytics import YOLO
from PIL import Image


def _no_grad(fn):
    try:
        import torch as _t
        return _t.no_grad()(fn)
    except Exception:
        return fn


COL_RED    = (0,  30, 230)
COL_WHITE  = (255, 255, 255)
COL_ORANGE = (0, 140, 255)


class ArsonModule:
    """Fire and arson detector — HSV colour + temporal flicker analysis."""

    # HSV fire colour ranges
    # Tuned to fire, NOT to high-vis vests:
    # High-vis vests are typically H=25-35 (yellow-green), V can be lower.
    # We require V >= 150 to demand bright/glowing pixels.
    _FIRE_HSV = [
        ((0,   100, 150), (20,  255, 255)),   # orange-red (flame core)
        ((20,  120, 180), (32,  255, 255)),   # yellow-orange (flame tip)
        ((160, 100, 150), (180, 255, 255)),   # wraparound red
    ]

    # Minimum connected fire region size (pixels²) — ignore tiny specks
    _MIN_FIRE_AREA = 400   # raised from 200 to reduce sensitivity to small bright spots

    # Temporal flicker: min pixel std-dev across frames to count as flickering
    # Raised to 25 (was 15) — moving people/vests cause ~15-20px variance,
    # real fire causes 30-60px variance due to rapid luminance oscillation.
    _FLICKER_THRESH = 25

    def __init__(
        self,
        yolo_model:      str   = "yolov8n.pt",
        clip_model_name: str   = "ViT-B/32",
        device:          str   = "cpu",
        fire_threshold:  float = 0.45,
        person_conf:     float = 0.40,
    ):
        self._device      = torch.device(device)
        self._fire_thresh = fire_threshold
        self._conf_person = person_conf

        print("[ArsonModule] Loading YOLOv8 ...")
        self._yolo = YOLO(yolo_model)

        # Frame history for flicker analysis
        self._prev_frames: deque = deque(maxlen=6)

        # Rolling detection history — fraction of recent frames with fire
        self._detection_history: deque = deque(maxlen=10)

        # Consecutive frames above threshold (require sustained detection)
        self._consecutive_fire: int = 0

        # CLIP secondary signal
        self._clip_ready = False
        if _CLIP_AVAILABLE:
            try:
                print(f"[ArsonModule] Loading CLIP {clip_model_name} ...")
                self._clip_model, self._preprocess = clip.load(
                    clip_model_name, device=self._device
                )
                self._clip_model.eval()
                self._clip_prompts = self._encode_clip_prompts()
                self._clip_ready = True
                print("[ArsonModule] CLIP ready (secondary signal).")
            except Exception as e:
                print(f"[ArsonModule] CLIP unavailable ({e}) — OpenCV only.")
        else:
            print("[ArsonModule] CLIP not installed — using OpenCV fire detection.")

    # ── main entry ────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[dict]]:
        if timestamp is None:
            timestamp = time.time()

        vis = frame.copy()
        alerts: List[dict] = []

        # 1. HSV colour detection
        fire_mask, colour_score = self._detect_fire_colour(frame)

        # 2. Temporal flicker
        flicker_score, flicker_mask = self._detect_flicker(frame)

        # 3. Combined mask — must have BOTH colour AND flicker
        if flicker_mask is not None:
            combined_mask = cv2.bitwise_and(fire_mask, flicker_mask)
        else:
            combined_mask = fire_mask   # not enough frames yet — colour only

        combined_area = int(combined_mask.sum() / 255)

        # 4. Frame-level fire decision
        frame_has_fire = combined_area >= self._MIN_FIRE_AREA

        # 5. CLIP boost (only when colour signal is present — avoids wasting time)
        clip_boost = 0.0
        if self._clip_ready and colour_score > 0.005:
            clip_boost = self._clip_score(frame)

        # 6. Accumulate history
        frame_score = min((1.0 if frame_has_fire else 0.0) + clip_boost, 1.0)
        self._detection_history.append(frame_score)
        self._prev_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # 7. Alert only when MAJORITY of recent frames show fire
        #    AND at least 4 consecutive frames triggered
        if len(self._detection_history) >= 5:
            recent_avg = float(np.mean(list(self._detection_history)[-8:]))
            if recent_avg >= self._fire_thresh:
                self._consecutive_fire += 1
            else:
                self._consecutive_fire = 0
        else:
            self._consecutive_fire = 0

        is_fire = self._consecutive_fire >= 4

        if is_fire:
            alerts.append({
                "type":          "ARSON/FIRE",
                "fire_area_px":  combined_area,
                "colour_score":  round(colour_score, 4),
                "flicker_score": round(flicker_score, 4),
                "timestamp":     timestamp,
            })

        # 8. Draw
        self._draw(vis, combined_mask, colour_score, flicker_score, is_fire)

        return vis, alerts

    # ── HSV colour detection ──────────────────────────────────────────────

    def _detect_fire_colour(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for (lo, hi) in self._FIRE_HSV:
            m = cv2.inRange(hsv, lo, hi)
            mask = cv2.bitwise_or(mask, m)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        total = frame.shape[0] * frame.shape[1]
        score = float(mask.sum() / 255) / total
        return mask, score

    # ── Temporal flicker detection ────────────────────────────────────────

    def _detect_flicker(self, frame: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        if len(self._prev_frames) < 4:
            return 0.0, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_array = np.stack(
            list(self._prev_frames) + [gray], axis=0
        ).astype(np.float32)
        pixel_std = frames_array.std(axis=0)
        flicker_mask = (pixel_std > self._FLICKER_THRESH).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        flicker_mask = cv2.morphologyEx(flicker_mask, cv2.MORPH_OPEN,  kernel)
        flicker_mask = cv2.morphologyEx(flicker_mask, cv2.MORPH_CLOSE, kernel)
        flicker_area = int(flicker_mask.sum() / 255)
        total = frame.shape[0] * frame.shape[1]
        return flicker_area / total, flicker_mask

    # ── CLIP secondary signal ─────────────────────────────────────────────

    @_no_grad
    def _encode_clip_prompts(self) -> dict:
        fire_texts = [
            "fire", "flames", "burning", "smoke and fire",
            "fire on the ground", "something burning",
        ]
        nofire_texts = [
            "normal scene", "people walking", "no fire",
        ]
        ft = clip.tokenize(fire_texts).to(self._device)
        nt = clip.tokenize(nofire_texts).to(self._device)
        ff = self._clip_model.encode_text(ft).float()
        nf = self._clip_model.encode_text(nt).float()
        ff = ff / ff.norm(dim=-1, keepdim=True)
        nf = nf / nf.norm(dim=-1, keepdim=True)
        return {"fire": ff, "nofire": nf}

    @_no_grad
    def _clip_score(self, frame: np.ndarray) -> float:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = self._preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self._device)
        img_feat = self._clip_model.encode_image(img).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        fire_sim   = float((img_feat @ self._clip_prompts["fire"].T).max())
        nofire_sim = float((img_feat @ self._clip_prompts["nofire"].T).max())
        return float(np.clip((fire_sim - nofire_sim) * 3.0, 0.0, 0.3))

    # ── Drawing ───────────────────────────────────────────────────────────

    def _draw(self, vis: np.ndarray, combined_mask: np.ndarray,
              colour_score: float, flicker_score: float, is_fire: bool) -> None:
        H, W = vis.shape[:2]

        # Highlight fire pixels in orange
        if combined_mask is not None and combined_mask.any():
            fire_overlay = vis.copy()
            fire_overlay[combined_mask > 0] = (0, 100, 255)
            cv2.addWeighted(fire_overlay, 0.4, vis, 0.6, 0, vis)
            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > self._MIN_FIRE_AREA:
                    cv2.drawContours(vis, [cnt], -1, COL_ORANGE, 2)

        # Score display
        cv2.rectangle(vis, (8, 8), (240, 60), (20, 20, 20), -1)
        cv2.putText(vis, f"Colour: {colour_score:.4f}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COL_WHITE, 1)
        cv2.putText(vis, f"Flicker: {flicker_score:.4f}",
                    (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COL_WHITE, 1)

        if is_fire:
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
            text = "!!! FIRE / ARSON DETECTED !!!"
            (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            tx = (W - tw) // 2
            cv2.putText(vis, text, (tx, 55),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, COL_WHITE, 3, cv2.LINE_AA)
            cv2.putText(vis, text, (tx, 55),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, COL_RED,   2, cv2.LINE_AA)