"""
test_smoke.py — validates module imports, class structure, and logic
without requiring YOLOv8 or CLIP to be downloaded.

Run:  python test_smoke.py
All tests should pass with just:  pip install numpy scipy opencv-python
"""

from __future__ import annotations

import sys
import traceback
import numpy as np

def _make_torch_stub():
    """Build a minimal torch stub that satisfies decorator imports."""
    import types as _t
    stub = _t.ModuleType("torch")
    stub.no_grad = lambda f=None: (f if f else (lambda g: g))   # works as decorator or ctx mgr
    stub.device  = lambda *a, **k: None
    stub.Tensor  = type("Tensor", (), {})
    stub.nn      = _t.ModuleType("torch.nn")
    stub.nn.Module = type("Module", (), {})
    return stub


PASS = "  ✓"
FAIL = "  ✗"


def _section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ══════════════════════════════════════════════════════════
# 1. Tracklet
# ══════════════════════════════════════════════════════════
def test_tracklet():
    _section("Tracklet")
    sys.path.insert(0, ".")
    import importlib.util as _ilu
    import os as _os
    def _load(name, path):
        spec = _ilu.spec_from_file_location(name, _os.path.join(_os.path.dirname(__file__), path))
        m = _ilu.module_from_spec(spec); spec.loader.exec_module(m); return m
    _mod = _load("tracklet_mod", "modules/tracklet.py")
    Tracklet, _atm, _iou = _mod.Tracklet, _mod._atm, _mod._iou

    Tracklet.reset_counter()

    # IDs auto-increment
    t0 = Tracklet([10, 10, 60, 80], np.random.randn(512).astype(np.float32), 0)
    t1 = Tracklet([100, 100, 160, 200], np.random.randn(512).astype(np.float32), 0)
    assert t0.track_id == 0 and t1.track_id == 1, "ID counter broken"
    print(f"{PASS} Track IDs auto-increment correctly")

    # Feature gallery mean
    for i in range(1, 8):
        t0.update([10+i, 10, 60+i, 80], np.ones(512, dtype=np.float32) * i, i)
    feat = t0.feature
    assert feat.shape == (512,), "Feature shape wrong"
    assert abs(np.linalg.norm(feat) - 1.0) < 0.01, "Feature not L2-normalised"
    print(f"{PASS} Feature gallery mean + L2 normalisation")

    # Trajectory
    trail = t0.trajectory_centers()
    assert len(trail) > 0, "Empty trajectory"
    print(f"{PASS} trajectory_centers() returns {len(trail)} points")

    # Loitering timer
    t0.update_loiter_in_zone(True, now=0.0, loiter_threshold=5.0)
    t0.update_loiter_in_zone(True, now=6.0, loiter_threshold=5.0)
    assert t0.is_loitering, "Loitering not triggered after threshold"
    t0.update_loiter_in_zone(False, now=7.0, loiter_threshold=5.0)
    assert not t0.is_loitering, "Loitering not reset when leaving zone"
    print(f"{PASS} Loitering timer triggers and resets correctly")

    # ATM false-positive filter
    # Oscillating trajectory: bounces between 2 pixels in each axis → tiny bbox area
    osc = [(100.0 + (i % 2) * 2, 200.0 + (i % 3) * 2) for i in range(30)]
    area_osc = _atm(osc)
    # Genuine walker: moves 150px across and 90px vertically → large bbox area
    walk = [(100.0 + i * 5, 200.0 + i * 3) for i in range(30)]
    area_walk = _atm(walk)
    assert area_osc < area_walk, "ATM didn't distinguish oscillation from walking"
    assert area_osc < 50,   f"Oscillation area should be tiny, got {area_osc:.1f}"
    assert area_walk > 1000, f"Walk area should be large, got {area_walk:.1f}"
    print(f"{PASS} ATM: oscillation={area_osc:.1f}px²  walking={area_walk:.1f}px²")

    # IoU helper
    assert _iou([0,0,10,10], [0,0,10,10]) == 1.0,  "IoU self-overlap != 1"
    assert _iou([0,0,10,10], [20,20,30,30]) == 0.0, "Non-overlapping IoU != 0"
    print(f"{PASS} IoU helper correct")

    print(f"\n  Tracklet tests: ALL PASSED")


# ══════════════════════════════════════════════════════════
# 2. Matching
# ══════════════════════════════════════════════════════════
def test_matching():
    _section("Matching")
    from modules.tracklet import Tracklet
    from modules.matching import cascade_match, _cosine_distance, _iou_distance

    Tracklet.reset_counter()

    # Cosine distance
    a = np.array([[1., 0., 0.]], dtype=np.float32)
    b = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    D = _cosine_distance(a, b)
    assert abs(D[0, 0]) < 0.01,  "Same vector cosine dist != 0"
    assert abs(D[0, 1] - 1.0) < 0.01, "Orthogonal vectors cosine dist != 1"
    print(f"{PASS} Cosine distance matrix correct")

    # IoU distance
    D2 = _iou_distance([[0,0,10,10]], [[0,0,10,10]])
    assert abs(D2[0,0]) < 0.01, "Self IoU dist != 0"
    print(f"{PASS} IoU distance matrix correct")

    # End-to-end cascade match — 2 tracks vs 2 detections (identity mapping)
    feats = np.eye(4, dtype=np.float32)
    t0 = Tracklet([0, 0, 10, 10],  feats[0], 0)
    t1 = Tracklet([50, 50, 60, 60], feats[1], 0)
    t0.confirmed = t1.confirmed = True

    det_bboxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
    det_feats  = feats[:2]

    matches, unm_trks, unm_dets = cascade_match([t0, t1], det_bboxes, det_feats)
    assert len(unm_trks) == 0, f"Unexpected unmatched tracks: {unm_trks}"
    assert len(unm_dets) == 0, f"Unexpected unmatched dets: {unm_dets}"
    assert len(matches) == 2,  f"Expected 2 matches, got {len(matches)}"
    print(f"{PASS} cascade_match: 2 tracks ↔ 2 detections, 0 unmatched")

    # Empty edge cases
    m, ut, ud = cascade_match([], [], np.empty((0, 4)))
    assert m == [] and ut == [] and ud == [], "Empty case failed"
    print(f"{PASS} cascade_match: empty inputs handled")

    print(f"\n  Matching tests: ALL PASSED")


# ══════════════════════════════════════════════════════════
# 3. Polygon overlap
# ══════════════════════════════════════════════════════════
def test_polygon_overlap():
    _section("Polygon overlap (intrusion/loitering check)")
    import importlib.util as _ilu2, os as _os2, sys as _sys2
    # stub torch so surveillance_module imports without it
    import types
    if "torch" not in _sys2.modules:
        torch_stub = types.ModuleType("torch"); _sys2.modules["torch"] = torch_stub
        _sys2.modules["torch.nn"] = types.ModuleType("torch.nn")
    if "ultralytics" not in _sys2.modules:
        _ultra_stub = types.ModuleType("ultralytics")
        _ultra_stub.YOLO = type("YOLO", (), {"__init__": lambda s, *a, **k: None})
        _sys2.modules["ultralytics"] = _ultra_stub
    spec2 = _ilu2.spec_from_file_location("surv_mod",
        _os2.path.join(_os2.path.dirname(__file__), "modules/surveillance_module.py"))
    import sys as _spN; _spN.modules.get("torch") and setattr(_spN.modules["torch"],"no_grad",_make_torch_stub().no_grad)
    sm = _ilu2.module_from_spec(spec2); spec2.loader.exec_module(sm)
    _bbox_polygon_overlap = sm._bbox_polygon_overlap

    # Box fully inside polygon → overlap ~1.0
    poly  = [(0, 0), (200, 0), (200, 200), (0, 200)]
    bbox  = [50, 50, 150, 150]
    ratio = _bbox_polygon_overlap(bbox, poly)
    assert ratio > 0.95, f"Expected ~1.0, got {ratio:.3f}"
    print(f"{PASS} Fully-inside bbox → overlap={ratio:.2f}")

    # Box fully outside polygon → overlap ~0.0
    bbox_out = [300, 300, 400, 400]
    ratio2   = _bbox_polygon_overlap(bbox_out, poly)
    assert ratio2 < 0.05, f"Expected ~0.0, got {ratio2:.3f}"
    print(f"{PASS} Fully-outside bbox → overlap={ratio2:.2f}")

    # Partial overlap
    bbox_partial = [150, 150, 250, 250]
    ratio3 = _bbox_polygon_overlap(bbox_partial, poly)
    assert 0.1 < ratio3 < 0.9, f"Partial overlap out of range: {ratio3:.3f}"
    print(f"{PASS} Partial-overlap bbox → overlap={ratio3:.2f}")

    print(f"\n  Polygon overlap tests: ALL PASSED")


# ══════════════════════════════════════════════════════════
# 4. LuggageState
# ══════════════════════════════════════════════════════════
def test_luggage_state():
    _section("LuggageState abandonment timer")
    import sys as _sys3, types as _types3
    if "torch" not in _sys3.modules:
        _ts3 = _make_torch_stub()
        _sys3.modules["torch"] = _ts3
        _sys3.modules["torch.nn"] = _ts3.nn
    if "ultralytics" not in _sys3.modules:
        _ultra_s3 = _types3.ModuleType("ultralytics")
        _ultra_s3.YOLO = type("YOLO", (), {"__init__": lambda s, *a, **k: None})
        _sys3.modules["ultralytics"] = _ultra_s3
    import importlib.util as _ilu3, os as _os3
    spec3 = _ilu3.spec_from_file_location("surv_mod2",
        _os3.path.join(_os3.path.dirname(__file__), "modules/surveillance_module.py"))
    sm3 = _ilu3.module_from_spec(spec3); spec3.loader.exec_module(sm3)
    _LuggageState = sm3._LuggageState

    state = _LuggageState([100, 100, 200, 200], "backpack", owner_id=3)

    # Still with owner → not abandoned
    state.update([100, 100, 200, 200], owner_id=3, now=5.0, abandon_thresh=10.0)
    assert not state.is_abandoned, "Should not be abandoned while owner present"
    print(f"{PASS} Luggage with owner: not abandoned")

    # Owner disappears → timer starts
    state.update([100, 100, 200, 200], owner_id=None, now=10.0, abandon_thresh=10.0)
    assert not state.is_abandoned, "Should not be abandoned at t=0 of timer"
    state.update([100, 100, 200, 200], owner_id=None, now=21.0, abandon_thresh=10.0)
    assert state.is_abandoned, "Should be abandoned after 10 s unowned"
    print(f"{PASS} Luggage unowned 11s → abandoned flag set")

    # Owner returns → timer resets
    state.update([100, 100, 200, 200], owner_id=3, now=22.0, abandon_thresh=10.0)
    assert not state.is_abandoned, "Should reset when owner returns"
    assert state.abandon_start is None, "Timer not cleared on owner return"
    print(f"{PASS} Owner returns → abandonment reset")

    print(f"\n  LuggageState tests: ALL PASSED")


# ══════════════════════════════════════════════════════════
# 5. Renderer
# ══════════════════════════════════════════════════════════
def test_renderer():
    _section("AlertRenderer + HUD")
    from utils.renderer import AlertRenderer, draw_fps, draw_header

    renderer = AlertRenderer()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    alerts = [
        {"type": "LOITERING",   "track_id": 2, "duration": 14.5, "bbox": [10,10,50,80], "timestamp": 1.0},
        {"type": "INTRUSION",   "track_id": 5, "ratio": 0.73,    "bbox": [100,100,200,300], "timestamp": 2.0},
        {"type": "ABANDONMENT", "lug_id": 0,   "owner_id": 2, "cls": "backpack",
         "duration": 11.0, "bbox": [200,200,260,280], "timestamp": 3.0},
        {"type": "ARSON/FIRE",  "fire_score": 0.31, "raw_score": 0.29, "timestamp": 4.0},
    ]

    renderer.ingest(alerts)
    renderer.render(dummy_frame)           # must not raise
    draw_fps(dummy_frame, 24.7)
    draw_header(dummy_frame, "PASS-CCTV TEST")

    # Check something was actually drawn (frame is no longer all-black)
    assert dummy_frame.sum() > 0, "Nothing drawn on frame"
    print(f"{PASS} Renderer draws on frame without errors")

    fmt = AlertRenderer._fmt
    assert "14" in fmt(alerts[0])          # duration
    assert "0.73" in fmt(alerts[1])        # ratio
    assert "backpack" in fmt(alerts[2])    # cls
    assert "0.31" in fmt(alerts[3])        # fire score
    print(f"{PASS} AlertRenderer._fmt formats all alert types correctly")

    print(f"\n  Renderer tests: ALL PASSED")


# ══════════════════════════════════════════════════════════
# 6. Colour fire fallback
# ══════════════════════════════════════════════════════════
def test_colour_fire_fallback():
    _section("Colour fire fallback (HSV)")
    import sys as _sys4, types as _types4
    if "torch" not in _sys4.modules:
        _ts4 = _make_torch_stub()
        _sys4.modules["torch"] = _ts4
        _sys4.modules["torch.nn"] = _ts4.nn
    if "clip" not in _sys4.modules:
        _sys4.modules["clip"] = _types4.ModuleType("clip")
    if "ultralytics" not in _sys4.modules:
        _u4 = _types4.ModuleType("ultralytics")
        _u4.YOLO = type("YOLO", (), {"__init__": lambda s, *a, **k: None})
        _sys4.modules["ultralytics"] = _u4
    import importlib.util as _ilu4, os as _os4
    spec4 = _ilu4.spec_from_file_location("arson_mod",
        _os4.path.join(_os4.path.dirname(__file__), "modules/arson_module.py"))
    import sys as _spN2; _spN2.modules.get("torch") and setattr(_spN2.modules["torch"],"no_grad",_make_torch_stub().no_grad)
    am4 = _ilu4.module_from_spec(spec4); spec4.loader.exec_module(am4)
    ArsonModule = am4.ArsonModule

    # Synthetic orange frame (simulates fire colour)
    fire_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    fire_frame[:, :, 2] = 255   # R
    fire_frame[:, :, 1] = 100   # G
    fire_frame[:, :, 0] = 0     # B  → bright orange in BGR

    score_fire = ArsonModule._colour_fire_fallback(fire_frame)

    # Dark / neutral frame — no fire colour
    dark_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    score_dark = ArsonModule._colour_fire_fallback(dark_frame)

    assert score_fire > score_dark, \
        f"Fire fallback: orange({score_fire:.3f}) should > dark({score_dark:.3f})"
    print(f"{PASS} Colour fallback: orange frame={score_fire:.3f}  dark frame={score_dark:.3f}")
    print(f"\n  Colour fire fallback tests: ALL PASSED")


# ══════════════════════════════════════════════════════════
# 7. PatchProcessor
# ══════════════════════════════════════════════════════════
def test_patch_processor():
    _section("CLIP PatchProcessor")
    import sys as _sys5, types as _types5
    if "torch" not in _sys5.modules:
        _ts5 = _make_torch_stub()
        _sys5.modules["torch"] = _ts5
        _sys5.modules["torch.nn"] = _ts5.nn
    if "clip" not in _sys5.modules:
        _sys5.modules["clip"] = _types5.ModuleType("clip")
    if "ultralytics" not in _sys5.modules:
        _u5 = _types5.ModuleType("ultralytics")
        _u5.YOLO = type("YOLO", (), {"__init__": lambda s, *a, **k: None})
        _sys5.modules["ultralytics"] = _u5
    import importlib.util as _ilu5, os as _os5
    spec5 = _ilu5.spec_from_file_location("arson_mod2",
        _os5.path.join(_os5.path.dirname(__file__), "modules/arson_module.py"))
    import sys as _spN2; _spN2.modules.get("torch") and setattr(_spN2.modules["torch"],"no_grad",_make_torch_stub().no_grad)
    am5 = _ilu5.module_from_spec(spec5); spec5.loader.exec_module(am5)
    _PatchProcessor = am5._PatchProcessor

    proc = _PatchProcessor()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bboxes = [[100, 100, 200, 300], [300, 50, 450, 250]]
    stationary = [False, True]

    patches, types = proc.extract(frame, bboxes, stationary)

    assert len(patches) >= 3, f"Expected ≥3 patches, got {len(patches)}"
    assert "frame"       in types, "frame patch missing"
    assert "person"      in types, "person patch missing"
    assert "stop_region" in types, "stop_region patch for stationary person missing"

    # All patches are correct PIL images at CLIP_SIZE
    from PIL import Image
    for i, p in enumerate(patches):
        assert isinstance(p, Image.Image), f"Patch {i} is not PIL Image"
        assert p.size == (224, 224),       f"Patch {i} wrong size: {p.size}"

    print(f"{PASS} PatchProcessor produces {len(patches)} patches of types: {set(types)}")
    print(f"\n  PatchProcessor tests: ALL PASSED")


# ══════════════════════════════════════════════════════════
# 8. Zone JSON round-trip
# ══════════════════════════════════════════════════════════
def test_zone_json():
    _section("Zone JSON round-trip")
    import json, tempfile, os

    zones = {
        "intrusion": [[100, 200], [500, 200], [500, 600], [100, 600]],
        "loitering": [[50, 150], [400, 150], [400, 700], [50, 700]],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False) as f:
        json.dump(zones, f)
        tmppath = f.name

    with open(tmppath) as f:
        loaded = json.load(f)
    os.unlink(tmppath)

    assert loaded["intrusion"] == zones["intrusion"], "Intrusion zone mismatch"
    assert loaded["loitering"] == zones["loitering"], "Loitering zone mismatch"
    print(f"{PASS} Zone JSON write/read round-trip correct")
    print(f"\n  Zone JSON tests: ALL PASSED")


# ══════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 55)
    print("  PASS-CCTV Revamped — Smoke Test Suite")
    print("═" * 55)

    tests = [
        ("Tracklet",              test_tracklet),
        ("Matching",              test_matching),
        ("Polygon overlap",       test_polygon_overlap),
        ("LuggageState",          test_luggage_state),
        ("Renderer/HUD",          test_renderer),
        ("Colour fire fallback",  test_colour_fire_fallback),
        ("PatchProcessor",        test_patch_processor),
        ("Zone JSON",             test_zone_json),
    ]

    passed, failed = 0, 0
    failures = []

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append((name, e))
            print(f"\n{FAIL} FAILED: {name}")
            traceback.print_exc()

    print("\n" + "═" * 55)
    print(f"  Results: {passed}/{len(tests)} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
        for name, err in failures:
            print(f"    ✗ {name}: {err}")
        sys.exit(1)
    else:
        print("  ✓")
        print("\n  All smoke tests PASSED — system logic is sound.")
        print("  Ready to run:  python main.py --input <video>")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
