"""
draw_zones.py — Interactive zone editor
========================================
Run this BEFORE running main.py to visually define your intrusion and
loitering zones by clicking on a reference frame from your video.

Usage
-----
  python draw_zones.py --input /path/to/video.mp4 --output zones.json

Controls
--------
  Left-click      Add vertex to current polygon
  Right-click     Finish current polygon (closes it)
  I               Switch to drawing Intrusion zone
  L               Switch to drawing Loitering zone
  C               Clear current incomplete polygon
  Z               Undo last vertex
  S               Save zones.json and exit
  Q / Esc         Quit without saving

The tool grabs frame 0 of the video (or uses an image) as the canvas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── colours ───────────────────────────────────────────────────────────────
_COLOURS: Dict[str, tuple] = {
    "intrusion": (0,  0, 220),    # red
    "loitering": (0, 200,  50),   # green
}
_DOT_R = 5


class ZoneDrawer:
    def __init__(self, canvas: np.ndarray, output_path: str):
        self._base      = canvas.copy()
        self._output    = output_path
        self._mode      = "intrusion"          # current zone type
        self._zones: Dict[str, Optional[List[Tuple[int, int]]]] = {
            "intrusion": None,
            "loitering": None,
        }
        self._current: List[Tuple[int, int]] = []  # in-progress polygon
        self._saved     = False

    # ── OpenCV callback ───────────────────────────────────────────────────

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._finish_polygon()

    def _finish_polygon(self):
        if len(self._current) >= 3:
            self._zones[self._mode] = list(self._current)
            print(f"  ✓ {self._mode.capitalize()} zone saved "
                  f"({len(self._current)} vertices).")
        else:
            print("  ✗ Need at least 3 points — discarded.")
        self._current = []

    # ── drawing ───────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        vis = self._base.copy()
        H, W = vis.shape[:2]

        # Draw finished zones
        for ztype, pts in self._zones.items():
            if pts is None:
                continue
            col = _COLOURS[ztype]
            arr = np.array(pts, dtype=np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [arr], col)
            cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
            cv2.polylines(vis, [arr], isClosed=True, color=col, thickness=2)
            cx, cy = int(np.mean(arr[:, 0])), int(np.mean(arr[:, 1]))
            cv2.putText(vis, ztype.upper(), (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

        # Draw in-progress polygon
        col = _COLOURS[self._mode]
        for i, pt in enumerate(self._current):
            cv2.circle(vis, pt, _DOT_R, col, -1)
            if i > 0:
                cv2.line(vis, self._current[i - 1], pt, col, 2, cv2.LINE_AA)
        if len(self._current) > 2:
            cv2.line(vis, self._current[-1], self._current[0],
                     col, 1, cv2.LINE_AA)   # closing preview

        # HUD
        lines = [
            f"Mode: [{self._mode.upper()}]  (I=intrusion  L=loitering)",
            "L-click=add pt  R-click=finish  C=clear  Z=undo  S=save  Q=quit",
            f"Intrusion: {'SET' if self._zones['intrusion'] else 'not set'}   "
            f"Loitering: {'SET' if self._zones['loitering'] else 'not set'}",
        ]
        for i, ln in enumerate(lines):
            y = H - 70 + i * 22
            cv2.putText(vis, ln, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240),
                        1, cv2.LINE_AA)

        return vis

    # ── save ─────────────────────────────────────────────────────────────

    def _save(self):
        data = {}
        if self._zones["intrusion"]:
            data["intrusion"] = [[x, y] for x, y in self._zones["intrusion"]]
        if self._zones["loitering"]:
            data["loitering"] = [[x, y] for x, y in self._zones["loitering"]]
        if not data:
            print("  Nothing to save.")
            return
        with open(self._output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  ✓ Zones saved → {self._output}")
        self._saved = True

    # ── main loop ────────────────────────────────────────────────────────

    def run(self) -> bool:
        win = "PASS-CCTV Zone Editor"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._mouse)

        print("\n=== Zone Editor ===")
        print("  I = intrusion zone  |  L = loitering zone")
        print("  Left-click = add vertex  |  Right-click = finish polygon")
        print("  S = save  |  Q / Esc = quit\n")

        while True:
            vis = self._render()
            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("i"):
                self._mode = "intrusion"
                print("  Mode → INTRUSION")
            elif key == ord("l"):
                self._mode = "loitering"
                print("  Mode → LOITERING")
            elif key == ord("c"):
                self._current = []
                print("  Cleared current polygon.")
            elif key == ord("z") and self._current:
                self._current.pop()
            elif key == ord("s"):
                self._save()
                break
            elif key in (ord("q"), 27):
                print("  Quit without saving.")
                break

        cv2.destroyAllWindows()
        return self._saved


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Interactive zone editor for PASS-CCTV")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",  "-i", help="Video file path")
    src.add_argument("--image",        help="Image file path")
    p.add_argument("--output", "-o", default="zones.json",
                   help="Output JSON path (default: zones.json)")
    p.add_argument("--frame", type=int, default=0,
                   help="Frame index to use as canvas (default: 0)")
    args = p.parse_args()

    if args.input:
        cap = cv2.VideoCapture(args.input)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx   = min(args.frame, total - 1) if total > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, canvas = cap.read()
        cap.release()
        if not ret:
            sys.exit(f"Cannot read frame {idx} from {args.input}")
    else:
        canvas = cv2.imread(args.image)
        if canvas is None:
            sys.exit(f"Cannot read image: {args.image}")

    drawer = ZoneDrawer(canvas, args.output)
    drawer.run()


if __name__ == "__main__":
    main()
