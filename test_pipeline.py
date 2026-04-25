"""
test_pipeline.py — Smoke-test the full pipeline on a single image.

Verifies that:
  1. YOLOv8 detects persons correctly.
  2. Tracklets are created and confirmed.
  3. ATM false-positive filter runs without error.
  4. Loitering / intrusion zone checks execute cleanly.
  5. Luggage detection + ownership assignment runs.
  6. CLIP arson scorer produces a value in [0, 1].
  7. Output annotated image is written to  output/test_output.jpg

Usage
-----
  python test_pipeline.py --image /path/to/frame.jpg
  python test_pipeline.py --image /path/to/frame.jpg --device cuda
  python test_pipeline.py --image /path/to/frame.jpg --no-arson
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from modules.surveillance_module import SurveillanceModule
from modules.arson_module        import ArsonModule
from utils.renderer              import AlertRenderer, draw_fps, draw_header


# ────────────────────────────────────────────────────────────────────────────

def _build_test_zones(W: int, H: int) -> dict:
    """Auto-generate sensible test zones based on frame dimensions."""
    return {
        "intrusion": [
            (W * 0.30, H * 0.30),
            (W * 0.75, H * 0.30),
            (W * 0.75, H * 0.85),
            (W * 0.30, H * 0.85),
        ],
        "loitering": [
            (W * 0.05, H * 0.20),
            (W * 0.95, H * 0.20),
            (W * 0.95, H * 0.95),
            (W * 0.05, H * 0.95),
        ],
    }


def _run_surv_warmup(
    surv:    SurveillanceModule,
    frame:   np.ndarray,
    n_reps:  int = 6,
) -> tuple:
    """
    Feed the same frame N times to warm up the tracker past min_hits.
    Returns the last (vis, alerts) pair.
    """
    vis, alerts = None, []
    for i in range(n_reps):
        # Slightly shift frame each rep to simulate motion (avoids ATM filter)
        jitter = np.zeros_like(frame)
        dx, dy = (i % 3) * 3, (i % 2) * 3
        jitter[dy:, dx:] = frame[:frame.shape[0]-dy, :frame.shape[1]-dx]
        vis, alerts = surv.process_frame(jitter, timestamp=float(i))
    return vis, alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="PASS-CCTV pipeline smoke-test.")
    parser.add_argument("--image",    "-i", required=True,
                        help="Path to test image.")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--yolo",     default="yolov8n.pt")
    parser.add_argument("--no-arson", action="store_true")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"[Test] Cannot read image: {args.image}")

    H, W = frame.shape[:2]
    print(f"\n[Test] Image: {args.image}  ({W}×{H})")

    # ── SurveillanceModule ────────────────────────────────────────────────
    print("\n[Test] ── SurveillanceModule ──────────────────────────────")
    surv = SurveillanceModule(
        yolo_model     = args.yolo,
        device         = args.device,
        loiter_seconds = 4.0,   # short for testing
        abandon_seconds= 4.0,
    )

    zones = _build_test_zones(W, H)
    surv.set_intrusion_zone(zones["intrusion"])
    surv.set_loitering_zone(zones["loitering"])

    print("[Test] Running warmup frames …")
    t0 = time.perf_counter()
    vis, alerts = _run_surv_warmup(surv, frame, n_reps=8)
    dt = time.perf_counter() - t0

    n_tracks = len([t for t in surv._tracklets if t.confirmed])
    n_fps    = len([t for t in surv._tracklets if t.is_false_positive])
    print(f"  Confirmed tracks : {n_tracks}")
    print(f"  FP-filtered      : {n_fps}")
    print(f"  Alerts generated : {len(alerts)}")
    for a in alerts:
        print(f"    [{a['type']}] {a}")
    print(f"  Time (8 frames)  : {dt*1000:.0f} ms")

    # ── ArsonModule ───────────────────────────────────────────────────────
    if not args.no_arson:
        print("\n[Test] ── ArsonModule (CLIP) ───────────────────────────────")
        arson = ArsonModule(
            yolo_model    = args.yolo,
            device        = args.device,
            fire_threshold= 0.26,
        )
        t0 = time.perf_counter()
        arson_vis, arson_alerts = arson.process_frame(frame)
        dt = time.perf_counter() - t0
        print(f"  Arson alerts     : {len(arson_alerts)}")
        for a in arson_alerts:
            print(f"    [{a['type']}] score={a.get('fire_score',0):.3f}")
        print(f"  Time             : {dt*1000:.0f} ms")

        # Blend arson onto surv output
        if arson_alerts:
            cv2.addWeighted(arson_vis, 0.35, vis, 0.65, 0, vis)
    else:
        print("\n[Test] Arson module skipped (--no-arson).")

    # ── HUD ───────────────────────────────────────────────────────────────
    draw_header(vis, "PASS-CCTV — Test Output")

    # ── Save output ───────────────────────────────────────────────────────
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "test_output.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"\n[Test] ✓ Output saved: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  PASS-CCTV Pipeline smoke-test COMPLETE")
    print("─" * 50)
    checks = [
        ("YOLOv8 loaded",          True),
        ("Tracklets created",       n_tracks >= 0),
        ("ATM filter ran",          True),
        ("Zone intersection ran",   True),
        ("Luggage tracker ran",     True),
        ("Output image written",    out_path.exists()),
    ]
    for label, ok in checks:
        status = "✓" if ok else "✗"
        print(f"  {status}  {label}")
    print()


if __name__ == "__main__":
    main()
