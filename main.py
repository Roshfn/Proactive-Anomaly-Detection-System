"""
main.py — PASS-CCTV Revamped  ·  Entry point
============================================

Accepts a video file path (or webcam index) and runs both detection modules
in sequence per frame, then writes the annotated output video.

Usage
-----
  # On a video file (processes entire file, writes output):
  python main.py --input /path/to/video.mp4

  # With custom zones defined in a JSON file:
  python main.py --input video.mp4 --zones zones.json

  # Webcam real-time (default camera 0):
  python main.py --webcam

  # Skip arson module (faster):
  python main.py --input video.mp4 --no-arson

Zone JSON format (optional)
---------------------------
{
  "intrusion": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
  "loitering":  [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
}
If no zone file is given the system runs without intrusion/loitering zones
(still detects abandonment, tracking, and fire).

Output
------
Processed frames are written to  output/<input_stem>_annotated.mp4
A JSON alert log is written to   output/<input_stem>_alerts.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# ── local imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from modules.surveillance_module import SurveillanceModule
from modules.arson_module        import ArsonModule
from utils.renderer              import AlertRenderer, draw_fps, draw_header


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class PASSCCTVPipeline:
    """
    Orchestrates SurveillanceModule + ArsonModule frame-by-frame.

    Both modules annotate their own copy of the frame; we composite them
    together:  Surveillance annotations take the base, then the arson
    fire overlay is blended on top (only its red tint layer, if any).
    """

    def __init__(
        self,
        device:          str   = "cpu",
        yolo_model:      str   = "yolov8n.pt",
        enable_arson:    bool  = True,
        loiter_seconds:  float = 10.0,
        abandon_seconds: float = 10.0,
        intrusion_ratio: float = 0.45,
        fire_threshold:  float = 0.26,
    ):
        print("\n" + "═" * 60)
        print("  PASS-CCTV Revamped — Initialising")
        print("═" * 60)

        self._surv = SurveillanceModule(
            yolo_model      = yolo_model,
            device          = device,
            loiter_seconds  = loiter_seconds,
            abandon_seconds = abandon_seconds,
            intrusion_ratio = intrusion_ratio,
        )

        self._arson_enabled = enable_arson
        if enable_arson:
            self._arson = ArsonModule(
                yolo_model     = yolo_model,
                device         = device,
                fire_threshold = fire_threshold,
            )

        self._renderer = AlertRenderer()
        self._all_alerts: List[dict] = []

        print("═" * 60 + "\n")

    # ── zone setup ────────────────────────────────────────────────────────

    def set_zones(self, zone_data: dict) -> None:
        if "intrusion" in zone_data:
            pts = [tuple(p) for p in zone_data["intrusion"]]
            self._surv.set_intrusion_zone(pts)
        if "loitering" in zone_data:
            pts = [tuple(p) for p in zone_data["loitering"]]
            self._surv.set_loitering_zone(pts)

    # ── frame processing ──────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> tuple[np.ndarray, List[dict]]:
        if timestamp is None:
            timestamp = time.time()

        alerts: List[dict] = []

        # ── Module 1: surveillance (persons / loitering / intrusion / abandon)
        vis, surv_alerts = self._surv.process_frame(frame, timestamp)
        alerts.extend(surv_alerts)

        # ── Module 2: arson (CLIP)
        if self._arson_enabled:
            _, arson_alerts = self._arson.process_frame(frame, timestamp)
            alerts.extend(arson_alerts)

            # Blend arson fire overlay on top of surv annotations
            if arson_alerts:
                # Re-run draw-only path to get the red tint from arson module
                arson_vis, _ = self._arson.process_frame(frame, timestamp)
                # Blend only arson overlay into vis (50% weight)
                cv2.addWeighted(arson_vis, 0.40, vis, 0.60, 0, vis)

        # ── HUD
        self._renderer.ingest(alerts)
        self._renderer.render(vis)
        draw_header(vis)

        self._all_alerts.extend(alerts)
        return vis, alerts

    def get_all_alerts(self) -> List[dict]:
        return self._all_alerts

    def reset(self) -> None:
        self._surv.reset()
        self._all_alerts = []


# ═══════════════════════════════════════════════════════════════════════════
#  Video processing helper
# ═══════════════════════════════════════════════════════════════════════════

def _open_source(source: str | int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    return cap


def process_video(
    pipeline:    PASSCCTVPipeline,
    source:      str | int,
    output_path: str,
    show_live:   bool = True,
    skip_frames: int  = 0,
) -> List[dict]:
    """
    Run the pipeline on a video source.

    Parameters
    ----------
    source       : path to video file or webcam index (0, 1, …)
    output_path  : path for annotated output video
    show_live    : whether to display frames in a window while processing
    skip_frames  : process every (skip_frames+1)-th frame (0 = all frames)

    Returns list of all alerts generated.
    """
    cap = _open_source(source)

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    os.makedirs(Path(output_path).parent, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps / max(1, skip_frames + 1),
                             (W, H))

    print(f"[Pipeline] Source : {source}")
    print(f"[Pipeline] Output : {output_path}")
    print(f"[Pipeline] Size   : {W}×{H}  FPS: {fps:.1f}")
    print("[Pipeline] Processing — press Q to quit early.\n")

    frame_idx = 0
    t_last = time.perf_counter()
    display_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if skip_frames > 0 and (frame_idx % (skip_frames + 1)) != 0:
            continue

        timestamp = frame_idx / fps

        vis, alerts = pipeline.process_frame(frame, timestamp)

        # FPS counter
        now = time.perf_counter()
        display_fps = 0.8 * display_fps + 0.2 / max(now - t_last, 1e-6)
        t_last = now
        draw_fps(vis, display_fps)

        writer.write(vis)

        if show_live:
            cv2.imshow("PASS-CCTV", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[Pipeline] Interrupted by user.")
                break

    cap.release()
    writer.release()
    if show_live:
        cv2.destroyAllWindows()

    print(f"\n[Pipeline] Done.  Processed {frame_idx} frames.")
    print(f"[Pipeline] Total alerts: {len(pipeline.get_all_alerts())}")
    return pipeline.get_all_alerts()


def process_image(
    pipeline:    PASSCCTVPipeline,
    image_path:  str,
    output_path: str,
) -> List[dict]:
    """Run pipeline on a single image (useful for testing)."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Run multiple times to warm up tracker (simulate frames)
    vis, alerts = None, []
    for _ in range(5):
        vis, alerts = pipeline.process_frame(frame.copy())

    os.makedirs(Path(output_path).parent, exist_ok=True)
    cv2.imwrite(output_path, vis)
    print(f"[Pipeline] Output image saved: {output_path}")
    return alerts


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pass_cctv",
        description="PASS-CCTV Revamped — Proactive Anomaly Surveillance System",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input",  "-i", type=str,
                     help="Path to input video file or image.")
    src.add_argument("--webcam", "-w", type=int, default=None, nargs="?",
                     const=0, metavar="CAM_IDX",
                     help="Use webcam (default index 0).")

    p.add_argument("--output",  "-o",  type=str, default=None,
                   help="Output path (auto-generated if omitted).")
    p.add_argument("--zones",          type=str, default=None,
                   help="JSON file defining intrusion/loitering zones.")
    p.add_argument("--device",         type=str, default="cpu",
                   help="Torch device: cpu | cuda | mps  (default: cpu)")
    p.add_argument("--yolo",           type=str, default="yolov8n.pt",
                   help="YOLOv8 weights file (auto-downloaded if missing).")
    p.add_argument("--no-arson",       action="store_true",
                   help="Disable arson/fire module (faster).")
    p.add_argument("--no-display",     action="store_true",
                   help="Do not show live preview window.")
    p.add_argument("--skip",           type=int, default=0,
                   help="Process every N+1 frames (0 = all, 1 = every 2nd).")
    p.add_argument("--loiter-sec",     type=float, default=10.0,
                   help="Seconds in zone before loitering alert (default 10).")
    p.add_argument("--abandon-sec",    type=float, default=10.0,
                   help="Seconds before abandoned-luggage alert (default 10).")
    p.add_argument("--fire-thresh",    type=float, default=0.26,
                   help="CLIP fire score threshold (default 0.26).")
    p.add_argument("--intrusion-ratio",type=float, default=0.45,
                   help="Bbox-zone overlap ratio for intrusion (default 0.45).")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    # Must supply at least one source
    if args.input is None and args.webcam is None:
        parser.error("Provide --input <file> or --webcam [index].")

    # ── build pipeline ────────────────────────────────────────────────────
    pipeline = PASSCCTVPipeline(
        device          = args.device,
        yolo_model      = args.yolo,
        enable_arson    = not args.no_arson,
        loiter_seconds  = args.loiter_sec,
        abandon_seconds = args.abandon_sec,
        fire_threshold  = args.fire_thresh,
        intrusion_ratio = args.intrusion_ratio,
    )

    # ── load zones ────────────────────────────────────────────────────────
    if args.zones:
        with open(args.zones) as f:
            pipeline.set_zones(json.load(f))

    # ── determine output path ─────────────────────────────────────────────
    def _default_output(src_path: str, suffix: str) -> str:
        stem = Path(src_path).stem
        return str(Path("output") / f"{stem}_annotated{suffix}")

    # ── run ───────────────────────────────────────────────────────────────
    source = args.input if args.input else args.webcam

    is_image = (args.input and args.input.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")))

    if is_image:
        out = args.output or _default_output(args.input, ".jpg")
        alerts = process_image(pipeline, args.input, out)
    else:
        src_label = str(source)
        out = args.output or _default_output(
            src_label if isinstance(source, str) else "webcam", ".mp4")
        alerts = process_video(
            pipeline,
            source,
            out,
            show_live   = not args.no_display,
            skip_frames = args.skip,
        )

    # ── save alert log ────────────────────────────────────────────────────
    if alerts:
        alert_log_path = str(Path(out).with_suffix("")) + "_alerts.json"
        with open(alert_log_path, "w") as f:
            # timestamp not JSON-serialisable directly → round-trip as str
            serialisable = [
                {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in a.items()}
                for a in alerts
            ]
            json.dump(serialisable, f, indent=2)
        print(f"[Pipeline] Alert log saved: {alert_log_path}")

    print("\n[Pipeline] All done.")


if __name__ == "__main__":
    main()
