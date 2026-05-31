"""
Door-aware people counter for an oblique 45-degree entrance camera.

Why this is different from a simple vertical ROI line:
    A person walking from the bathroom to the water dispenser can cross the same
    line as a real door transition. This script counts only tracks that show
    door intent:

    OUT: the track is first seen/stabilized in the doorway zone, then reaches
         the outside approach floor.
    IN : the track is first seen/stabilized on the outside approach floor, then
         reaches the doorway zone.

Install:
    pip install ultralytics opencv-python

Run:
    python count_people.py

Controls:
    q - quit
    r - reset counts
    p - pause/play
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# Config
# =========================
VIDEO_SOURCE = "entrance.mov"
MODEL_PATH = "yolov8m.pt"
ROI_CONFIG_PATH = Path("roi_config.json")

PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.65
IOU_THRESHOLD = 0.55
IMG_SIZE = 960
TRACKER = "bytetrack.yaml"

# Visual reference for the doorway edge/gate. The actual count is decided from
# the zone transition because a short line is fragile with this oblique camera.
# Current value comes from roi_line_config.txt.
DOOR_GATE_START = (846, 98)
DOOR_GATE_END = (818, 358)

# Main area inside the doorway/open door. Tune these points if your camera
# moves. Use the person's foot point, so include the floor just inside the door.
DOOR_ZONE = np.array(
    [
        (420, 120),
        (930, 135),
        (910, 735),
        (425, 800),
    ],
    dtype=np.int32,
)

# Outside floor immediately in front of the door. A real entrance normally moves
# between this area and DOOR_ZONE. It deliberately avoids the water dispenser.
APPROACH_ZONE = np.array(
    [
        (210, 800),
        (930, 725),
        (1040, 980),
        (120, 1030),
    ],
    dtype=np.int32,
)

# Visual/no-count zone for the dispenser side. Tracks here are not counted
# unless they also had doorway evidence first.
WATER_DISPENSER_ZONE = np.array(
    [
        (950, 270),
        (1515, 285),
        (1585, 850),
        (890, 880),
    ],
    dtype=np.int32,
)

ZONE_KEY_MAP = {
    "door_zone": "DOOR_ZONE",
    "approach_zone": "APPROACH_ZONE",
    "water_zone": "WATER_DISPENSER_ZONE",
}

MIN_DOOR_HITS = 3
MIN_APPROACH_HITS = 3
MIN_TRACK_POINTS = 5
MIN_MOVEMENT_PX = 35
COUNT_COOLDOWN_SEC = 1.2
TRAIL_LENGTH = 48
MAX_STALE_TRACKS = 120
DISPLAY_WIDTH = 1300


CLR_DOOR = (0, 210, 255)
CLR_APPROACH = (70, 220, 70)
CLR_WATER = (255, 170, 0)
CLR_IN = (40, 210, 80)
CLR_OUT = (40, 90, 235)
CLR_TEXT = (245, 245, 245)
CLR_BG = (25, 25, 25)


@dataclass
class TrackState:
    points: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    door_hits: int = 0
    approach_hits: int = 0
    water_hits: int = 0
    counted_in: bool = False
    counted_out: bool = False
    last_seen_frame: int = 0

    @property
    def first_point(self) -> tuple[int, int] | None:
        return self.points[0] if self.points else None

    @property
    def last_point(self) -> tuple[int, int] | None:
        return self.points[-1] if self.points else None


def point_in_poly(point: tuple[int, int], poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, point, False) >= 0


def scale_normalized_zone(points, width: int, height: int) -> np.ndarray:
    return np.array(
        [(int(x * width), int(y * height)) for x, y in points],
        dtype=np.int32,
    )


def load_roi_zones(width: int, height: int):
    zones = {
        "door_zone": DOOR_ZONE.copy(),
        "approach_zone": APPROACH_ZONE.copy(),
        "water_zone": WATER_DISPENSER_ZONE.copy(),
    }

    if not ROI_CONFIG_PATH.exists() or ROI_CONFIG_PATH.stat().st_size == 0:
        print("ROI config not found. Using built-in default zones.")
        return zones

    try:
        with ROI_CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read {ROI_CONFIG_PATH}: {exc}. Using built-in default zones.")
        return zones

    config_zones = data.get("zones", {})
    for key in zones:
        points = config_zones.get(key, [])
        if len(points) >= 3:
            zones[key] = scale_normalized_zone(points, width, height)
        else:
            print(f"{ZONE_KEY_MAP[key]} missing or incomplete in config. Using default.")

    print(f"Loaded normalized ROI zones from {ROI_CONFIG_PATH}.")
    return zones


def moved_enough(state: TrackState) -> bool:
    if len(state.points) < MIN_TRACK_POINTS:
        return False
    first = state.first_point
    last = state.last_point
    if first is None or last is None:
        return False
    return math.hypot(last[0] - first[0], last[1] - first[1]) >= MIN_MOVEMENT_PX


def id_color(track_id: int) -> tuple[int, int, int]:
    palette = [
        (255, 110, 80),
        (80, 200, 255),
        (255, 220, 50),
        (150, 255, 100),
        (210, 90, 255),
        (255, 150, 50),
        (80, 255, 205),
        (255, 90, 180),
        (115, 180, 255),
        (245, 245, 105),
    ]
    return palette[track_id % len(palette)]


def draw_poly(frame, poly: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.13, frame, 0.87, 0, frame)
    cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)
    x, y = poly[0]
    cv2.putText(frame, label, (int(x) + 8, int(y) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_trail(frame, points: deque[tuple[int, int]], color: tuple[int, int, int]) -> None:
    pts = list(points)
    for i in range(1, len(pts)):
        alpha = i / max(1, len(pts) - 1)
        faded = tuple(int(c * alpha) for c in color)
        cv2.line(frame, pts[i - 1], pts[i], faded, max(1, int(3 * alpha)), cv2.LINE_AA)


def draw_hud(frame, count_in: int, count_out: int, active_tracks: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (380, 148), CLR_BG, -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.putText(frame, f"IN     : {count_in}", (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_IN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"OUT    : {count_out}", (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_OUT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"INSIDE : {count_in - count_out}", (16, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"TRACKS : {active_tracks}", (230, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, CLR_TEXT, 1, cv2.LINE_AA)


def resize_for_display(frame):
    if DISPLAY_WIDTH <= 0 or frame.shape[1] <= DISPLAY_WIDTH:
        return frame
    scale = DISPLAY_WIDTH / frame.shape[1]
    return cv2.resize(frame, (DISPLAY_WIDTH, int(frame.shape[0] * scale)))


def prune_tracks(states: dict[int, TrackState], frame_index: int) -> None:
    stale_ids = [tid for tid, state in states.items() if frame_index - state.last_seen_frame > MAX_STALE_TRACKS]
    for tid in stale_ids:
        del states[tid]


def run() -> None:
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_zones = load_roi_zones(width, height)
    door_zone = roi_zones["door_zone"]
    approach_zone = roi_zones["approach_zone"]
    water_zone = roi_zones["water_zone"]

    states: dict[int, TrackState] = defaultdict(TrackState)
    count_in = 0
    count_out = 0
    last_count_time = 0.0
    paused = False
    frame_index = 0
    frame = None

    print("Running door-aware counter. Press q to quit, r to reset, p to pause.")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            results = model.track(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                tracker=TRACKER,
                persist=True,
                verbose=False,
            )[0]

            if results.boxes.id is not None:
                for box, tid in zip(results.boxes, results.boxes.id.int().tolist()):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    foot = ((x1 + x2) // 2, y2)

                    state = states[tid]
                    state.last_seen_frame = frame_index
                    state.points.append(foot)

                    if point_in_poly(foot, door_zone):
                        state.door_hits += 1
                    if point_in_poly(foot, approach_zone):
                        state.approach_hits += 1
                    if point_in_poly(foot, water_zone):
                        state.water_hits += 1

                    now = time.monotonic()
                    enough_time = now - last_count_time >= COUNT_COOLDOWN_SEC
                    stable = moved_enough(state)
                    has_door = state.door_hits >= MIN_DOOR_HITS
                    has_approach = state.approach_hits >= MIN_APPROACH_HITS
                    in_door_now = point_in_poly(foot, door_zone)
                    in_approach_now = point_in_poly(foot, approach_zone)

                    # Count zone transitions, not random line crossings. This is
                    # what prevents bathroom-to-dispenser traffic from counting.
                    if enough_time and stable and has_approach and in_door_now and not in_approach_now and not state.counted_in:
                        count_in += 1
                        state.counted_in = True
                        last_count_time = now

                    if enough_time and stable and has_door and in_approach_now and not in_door_now and not state.counted_out:
                        count_out += 1
                        state.counted_out = True
                        last_count_time = now

                    color = id_color(tid)
                    draw_trail(frame, state.points, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, foot, 5, (255, 255, 255), -1)
                    cv2.circle(frame, foot, 6, color, 2)

                    status = []
                    if has_door:
                        status.append("door")
                    if has_approach:
                        status.append("approach")
                    if state.water_hits >= MIN_APPROACH_HITS and not has_door:
                        status.append("water-ignore")
                    label = f"ID {tid} {conf:.2f} {'/'.join(status)}"
                    cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            prune_tracks(states, frame_index)

        if frame is None:
            continue

        display = frame.copy()
        draw_poly(display, door_zone, CLR_DOOR, "DOOR ZONE")
        draw_poly(display, approach_zone, CLR_APPROACH, "APPROACH")
        draw_poly(display, water_zone, CLR_WATER, "WATER/IGNORE")
        cv2.line(display, DOOR_GATE_START, DOOR_GATE_END, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(display, DOOR_GATE_START, 7, (255, 255, 255), -1)
        cv2.circle(display, DOOR_GATE_END, 7, (255, 255, 255), -1)
        cv2.putText(display, "DOOR GATE", (DOOR_GATE_START[0] + 10, DOOR_GATE_START[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        draw_hud(display, count_in, count_out, len(states))

        cv2.imshow("YOLOv8 Door-Aware People Counter", resize_for_display(display))
        key = cv2.waitKey(0 if paused else 1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("r"):
            count_in = 0
            count_out = 0
            states.clear()
            last_count_time = 0.0
            print("Counts reset.")

    cap.release()
    cv2.destroyAllWindows()

    print("\nFinal results")
    print(f"IN     : {count_in}")
    print(f"OUT    : {count_out}")
    print(f"INSIDE : {count_in - count_out}")


if __name__ == "__main__":
    run()
