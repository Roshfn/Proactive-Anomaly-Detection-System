"""
zone_editor.py — Interactive zone definition tool.

Opens the first frame of a video (or any image) and lets you click to
define polygon vertices for intrusion and loitering zones.  Saves the
result as zones.json ready for use with main.py.

Usage
-----
  python zone_editor.py --input footage.mp4
  python zone_editor.py --input frame.jpg

Controls
--------
  Left-click        Add vertex to current polygon
  Right-click       Undo last vertex
  ENTER             Finish current polygon and move to next zone type
  ESC               Cancel current polygon / quit
  R                 Reset all zones
  S                 Save zones.json and exit

Order of definition
-------------------
  1. Intrusion zone  (drawn in red)
  2. Loitering zone  (drawn in green)

Both zones are optional — press ENTER with no points to skip a zone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── colours ───────────────────────────────────────────────────────────────
_COLOURS = {
    "intrusion": (0,  60, 230),   # red-ish
    "loitering": (0, 200,  60),   # green
    "vertex":    (255, 255, 255),
    "hud":       (230, 230, 230),
    "bg":        (20,  20,  20),
}


# ─── state ─────────────────────────────────────────────────────────────────

class ZoneEditor:

    ZONE_ORDER = ["intrusion", "loitering"]
    INSTRUCTIONS = {
        "intrusion": "Draw INTRUSION zone  (left-click = add pt, ENTER = done, ESC = skip)",
        "loitering": "Draw LOITERING zone  (left-click = add pt, ENTER = done, ESC = skip)",
    }

    def __init__(self, base_frame: np.ndarray, output_path: str):
        self._base   = base_frame.copy()
        self._output = output_path
        self._zones: dict = {}
        self._current_pts: List[Tuple[int, int]] = []
        self._zone_idx = 0
        self._mouse_pos: Tuple[int, int] = (0, 0)
        self._done = False

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self) -> Optional[dict]:
        cv2.namedWindow("Zone Editor", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Zone Editor", self._on_mouse)

        print("\n" + "=" * 58)
        print("  PASS-CCTV Zone Editor")
        print("=" * 58)

        while self._zone_idx < len(self.ZONE_ORDER) and not self._done:
            zone_name = self.ZONE_ORDER[self._zone_idx]
            print(f"\n[ZoneEditor] {self.INSTRUCTIONS[zone_name]}")

            while True:
                frame = self._render()
                cv2.imshow("Zone Editor", frame)
                key = cv2.waitKey(20) & 0xFF

                if key == 13 or key == 10:   # ENTER — finish zone
                    if len(self._current_pts) >= 3:
                        self._zones[zone_name] = [list(p) for p in self._current_pts]
                        print(f"  ✓ {zone_name} zone saved "
                              f"({len(self._current_pts)} vertices).")
                    else:
                        print(f"  — {zone_name} zone skipped.")
                    self._current_pts = []
                    self._zone_idx += 1
                    break

                elif key == 27:              # ESC — skip zone
                    print(f"  — {zone_name} zone skipped.")
                    self._current_pts = []
                    self._zone_idx += 1
                    break

                elif key == ord("r"):        # R — reset
                    self._current_pts = []
                    self._zones = {}
                    self._zone_idx = 0
                    print("  Zones reset.")
                    break

                elif key == ord("s"):        # S — save and exit
                    self._zone_idx = len(self.ZONE_ORDER)
                    break

                elif key == ord("q"):
                    self._done = True
                    break

        cv2.destroyAllWindows()

        if not self._zones:
            print("\n[ZoneEditor] No zones defined.")
            return None

        self._save()
        return self._zones

    # ── rendering ─────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        vis = self._base.copy()

        # Already-saved zones
        for zname, pts in self._zones.items():
            poly  = np.array(pts, dtype=np.int32)
            col   = _COLOURS[zname]
            overlay = vis.copy()
            cv2.fillPoly(overlay, [poly], col)
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
            cv2.polylines(vis, [poly], True, col, 2)
            cx, cy = int(np.mean(poly[:, 0])), int(np.mean(poly[:, 1]))
            cv2.putText(vis, zname.upper(), (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

        # Current zone being drawn
        if self._zone_idx < len(self.ZONE_ORDER):
            zone_name = self.ZONE_ORDER[self._zone_idx]
            col = _COLOURS[zone_name]

            for pt in self._current_pts:
                cv2.circle(vis, pt, 5, _COLOURS["vertex"], -1)

            if len(self._current_pts) >= 2:
                pts_arr = np.array(self._current_pts, dtype=np.int32)
                cv2.polylines(vis, [pts_arr], False, col, 2)

            # Rubber-band line to mouse
            if self._current_pts:
                cv2.line(vis, self._current_pts[-1], self._mouse_pos, col, 1,
                         cv2.LINE_AA)

            # HUD instruction
            instr = self.INSTRUCTIONS[zone_name]
            self._draw_hud(vis, instr, len(self._current_pts))

        return vis

    @staticmethod
    def _draw_hud(vis: np.ndarray, instr: str, n_pts: int) -> None:
        H, W = vis.shape[:2]
        cv2.rectangle(vis, (0, H - 50), (W, H), (20, 20, 20), -1)
        cv2.putText(vis, instr, (10, H - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, _COLOURS["hud"], 1)
        cv2.putText(vis, f"Vertices: {n_pts}  |  R=reset  S=save  ESC=skip",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (160, 160, 160), 1)

    # ── mouse callback ────────────────────────────────────────────────────

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        self._mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._current_pts:
                self._current_pts.pop()

    # ── save ──────────────────────────────────────────────────────────────

    def _save(self) -> None:
        Path(self._output).parent.mkdir(parents=True, exist_ok=True)
        with open(self._output, "w") as f:
            json.dump(self._zones, f, indent=2)
        print(f"\n[ZoneEditor] Saved → {self._output}")
        print("  Use with:  python main.py --input <video> "
              f"--zones {self._output}")


# ─── entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively define intrusion/loitering zones for PASS-CCTV."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Video file or image to use as background.")
    parser.add_argument("--output", "-o", default="zones.json",
                        help="Output JSON file path (default: zones.json).")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to use from video (default: 0).")
    args = parser.parse_args()

    src = args.input
    if src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        frame = cv2.imread(src)
        if frame is None:
            sys.exit(f"Cannot read image: {src}")
    else:
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            sys.exit(f"Cannot read frame {args.frame} from: {src}")

    editor = ZoneEditor(frame, args.output)
    editor.run()


if __name__ == "__main__":
    main()
