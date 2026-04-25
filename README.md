# Proactive Anomaly Detection System 

**Proactive Anomaly Detection System — complete revamp**

Detects four anomaly types from CCTV video, based on
*Jeon et al., Expert Systems With Applications 2024*.

\---

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
  input frame  ───▶ │           Module 1 — SurveillanceModule          │
                    │  YOLOv8 detect → OSNet features → Cascade match  │
                    │  ┌──────────┐  ┌───────────┐  ┌─────────────┐   │
                    │  │ Tracking │  │ Loitering │  │ Intrusion   │   │
                    │  │ (ATM FP  │  │ (10 s in  │  │ (bbox-poly  │   │
                    │  │ filter)  │  │  zone)    │  │  overlap)   │   │
                    │  └──────────┘  └───────────┘  └─────────────┘   │
                    │              ┌──────────────┐                    │
                    │              │ Abandonment  │                    │
                    │              │ (top-down +  │                    │
                    │              │  owner track)│                    │
                    │              └──────────────┘                    │
                    └───────────────────┬─────────────────────────────┘
                                        │ annotated frame + alerts
                    ┌───────────────────▼─────────────────────────────┐
  same frame   ───▶ │           Module 2 — ArsonModule                 │
                    │  YOLOv8 detect → CLIP patches → fire score       │
                    │  (zero-shot, no fine-tuning needed)               │
                    └─────────────────────────────────────────────────┘
                                        │
                               composite output + HUD
```

\---

## What's fixed vs original

|Issue|Fix|
|-|-|
|IDs swapping / wrong|Two-stage cascade match: appearance (cosine) + IoU fallback; feature gallery (mean of last 10) instead of last-only|
|Non-person objects tracked|ATM (Area of Trajectory Movement) filter: oscillating blobs score near-zero and are suppressed after 25 frames|
|Tracklet trail not drawn|`trajectory\_centers()` rendered as fading polyline on every confirmed track|
|Loitering no colour change|Bbox colour is **green** normally, **red** when loitering duration ≥ threshold|
|Intrusion no alert|Bbox-polygon overlap ratio checked every frame; any track inside restricted zone gets red bbox + alert|
|Abandonment no alert|`\_LuggageState` tracks per-item ownership; 10 s unowned timer → box turns red + alert fires|
|Owner ID on luggage|`owner:{track\_id}` label drawn on luggage bbox at all times|
|Arson threshold too high|CLIP score = avg-top3-fire − 0.3×max-nonfire (contrast scoring); smoothed over 5 frames; default threshold 0.26|
|CLIP not detecting small fire|Stop-region patches + person patches added; 18 fire prompts covering sparks, small flames, arsonist behaviour|

\---

## Installation

```bash
# Clone / copy this folder, then:
pip install ultralytics torch torchvision scipy pillow opencv-python
pip install git+https://github.com/openai/CLIP.git

# Optional but recommended for better re-ID:
pip install torchreid   # or: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

YOLOv8 weights (`yolov8n.pt`) are downloaded automatically by Ultralytics on
first run.

\---

## Quick start

```bash
# Process a video file
python main.py --input /path/to/cctv\_footage.mp4

# With zones
python main.py --input footage.mp4 --zones zones.json

# Webcam real-time
python main.py --webcam 0

# GPU acceleration
python main.py --input footage.mp4 --device cuda

# Faster (every other frame, no arson)
python main.py --input footage.mp4 --skip 1 --no-arson
```

\---

## Zone configuration  (`zones.json`)

Coordinates are **pixel positions** in the video frame.

```json
{
  "intrusion": \[
    \[100, 200],
    \[500, 200],
    \[500, 600],
    \[100, 600]
  ],
  "loitering": \[
    \[50,  150],
    \[400, 150],
    \[400, 700],
    \[50,  700]
  ]
}
```

Polygons can be **any convex or concave shape** — just list vertices in order.

To find the right coordinates, open your video in any image viewer and note
the pixel positions of your zone corners.

\---

## CLI reference

```
--input    / -i   Path to video or image file
--webcam   / -w   Webcam index (default 0)
--output   / -o   Output path (auto-generated if omitted)
--zones           JSON file with intrusion/loitering zone polygons
--device          cpu | cuda | mps  (default cpu)
--yolo            YOLOv8 weights (default yolov8n.pt; yolov8s.pt is more accurate)
--no-arson        Disable CLIP fire module
--no-display      Don't show live preview window
--skip    N       Process every N+1 frames (speeds up long videos)
--loiter-sec      Seconds in zone → loitering (default 10)
--abandon-sec     Seconds unowned → abandonment alert (default 10)
--fire-thresh     CLIP fire score threshold (default 0.26)
--intrusion-ratio Overlap fraction for intrusion trigger (default 0.45)
```

\---

## Output

|File|Description|
|-|-|
|`output/<name>\_annotated.mp4`|Annotated video with all overlays|
|`output/<name>\_alerts.json`|Machine-readable alert log|

### Alert JSON schema

```json
\[
  {"type": "LOITERING",    "track\_id": 3, "duration": 12.4, "bbox": \[x1,y1,x2,y2], "timestamp": 1.23},
  {"type": "INTRUSION",    "track\_id": 7, "ratio": 0.82,    "bbox": \[...], "timestamp": 3.45},
  {"type": "ABANDONMENT",  "lug\_id": 0,  "owner\_id": 2,    "cls": "backpack", "duration": 11.0, ...},
  {"type": "ARSON/FIRE",   "fire\_score": 0.312, "raw\_score": 0.298, "timestamp": 8.90}
]
```

\---

## Visual legend

|Colour|Meaning|
|-|-|
|🟩 Green bbox|Person detected, no anomaly|
|🟥 Red bbox|Loitering or intrusion alert|
|🟨 Yellow bbox|Luggage — owner present|
|🟥 Red bbox|Luggage — **abandoned**|
|Red tint + banner|Fire / arson detected|
|Fading trail|Person trajectory (colour matches bbox)|
|Zone overlay (blue/green)|Intrusion / loitering zones|

\---

## File structure

```
proactive anomaly detection system/
├── main.py                     ← entry point
├── modules/
│   ├── tracklet.py             ← per-person state (ATM, loitering, ownership)
│   ├── matching.py             ← cascade matching (appearance + IoU)
│   ├── surveillance\_module.py  ← tracking + loitering + intrusion + abandonment
│   └── arson\_module.py         ← CLIP fire/arson detection
├── utils/
│   └── renderer.py             ← HUD, alert log overlay, FPS counter
└── README.md
```

\---

## Tips

* **Better accuracy**: use `--yolo yolov8s.pt` or `yolov8m.pt` (larger but more accurate).
* **GPU**: `--device cuda` is highly recommended for real-time on 1080p+.
* **Loitering time**: for testing set `--loiter-sec 5`; for deployment use 10–30.
* **Fire sensitivity**: lower `--fire-thresh` (e.g. 0.22) to catch earlier sparks;
raise it (0.30+) to reduce false positives from bright lights.
* **No zones needed** for abandonment and arson — they work across the whole frame.

