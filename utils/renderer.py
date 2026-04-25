"""
renderer.py — On-screen HUD, alert log overlay, alert sound/print dispatch.
"""

from __future__ import annotations

import time
from collections import deque
from typing import List

import cv2
import numpy as np


# ── colour palette ─────────────────────────────────────────────────────────
_C = {
    "red":    (0,  30, 230),
    "green":  (0, 200,  50),
    "yellow": (0, 200, 255),
    "white":  (255, 255, 255),
    "black":  (0,   0,   0),
    "orange": (0, 140, 255),
    "bg":     (20,  20,  20),
}

_ALERT_COLOURS = {
    "INTRUSION":   _C["red"],
    "LOITERING":   _C["orange"],
    "ABANDONMENT": _C["yellow"],
    "ARSON/FIRE":  _C["red"],
}


class AlertRenderer:
    """
    Maintains a rolling log of recent alerts and renders them as an
    on-screen panel in the bottom-left corner of the frame.

    Also prints to stdout with rate-limiting (one print per alert per 3 s).
    """

    MAX_LOG = 12          # lines shown on screen
    LOG_HOLD = 8.0        # seconds an alert stays on screen

    def __init__(self):
        # deque of (timestamp, alert_dict)
        self._log: deque = deque(maxlen=self.MAX_LOG * 2)
        self._last_print: dict = {}   # alert_key -> last print time

    # ── public ────────────────────────────────────────────────────────────

    def ingest(self, alerts: List[dict]) -> None:
        """Feed new alerts into the log."""
        now = time.time()
        for a in alerts:
            self._log.append((now, a))
            self._maybe_print(a, now)

    def render(self, frame: np.ndarray) -> None:
        """Draw the alert log panel onto `frame` in-place."""
        now = time.time()
        # Filter to recent alerts
        visible = [(ts, a) for ts, a in self._log
                   if now - ts <= self.LOG_HOLD]
        if not visible:
            return

        H, W = frame.shape[:2]
        line_h = 22
        panel_h = line_h * len(visible) + 10
        panel_w = 340
        px, py = 8, H - panel_h - 8

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                      _C["bg"], -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        for i, (ts, a) in enumerate(visible[-self.MAX_LOG:]):
            atype  = a.get("type", "?")
            colour = _ALERT_COLOURS.get(atype, _C["white"])
            age    = now - ts
            alpha  = max(0.3, 1.0 - age / self.LOG_HOLD)
            c = tuple(int(v * alpha) for v in colour)

            label = self._fmt(a)
            y = py + 12 + i * line_h
            cv2.putText(frame, f"[{atype}] {label}", (px + 6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)

        # Header
        cv2.putText(frame, "ALERTS", (px + 6, py + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _C["white"], 1)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(a: dict) -> str:
        t = a.get("type", "")
        if t == "INTRUSION":
            return f"ID {a.get('track_id')} ratio={a.get('ratio',0):.2f}"
        if t == "LOITERING":
            return f"ID {a.get('track_id')} {a.get('duration',0):.0f}s"
        if t == "ABANDONMENT":
            owner = a.get("owner_id")
            return (f"{a.get('cls','')} owner:{owner} "
                    f"{a.get('duration',0):.0f}s")
        if t == "ARSON/FIRE":
            return f"score={a.get('fire_score',0):.3f}"
        return str(a)

    def _maybe_print(self, a: dict, now: float) -> None:
        key = (a.get("type"), a.get("track_id"), a.get("lug_id"))
        if now - self._last_print.get(key, 0) >= 3.0:
            self._last_print[key] = now
            print(f"  ⚠  ALERT [{a.get('type')}] {self._fmt(a)}")


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def draw_header(frame: np.ndarray, label: str = "Proactive Anomaly Detection") -> None:
    W = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (W, 28), (20, 20, 20), -1)
    cv2.putText(frame, label, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
