"""
Count people across an entire video using YOLOv8 + DeepSORT.

This does not count people per frame and does not count IN/OUT direction.
Each stable DeepSORT tracking ID is counted once. Short occlusions/overlaps keep
the same ID, but a person who leaves the frame and comes back later can be
counted again as a new track.

Install:
    pip install ultralytics opencv-python deep-sort-realtime

Run:
    python count_people.py

Controls:
    q - quit
    r - reset total count
    p - pause/play
""" 

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


# =========================
# Config
# =========================
VIDEO_SOURCE = "entrance_stable.mp4"
MODEL_PATH = "yolov8m.pt"

PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.8
IOU_THRESHOLD = 0.55
IMG_SIZE = 960

# DeepSORT settings:
# max_age keeps the same ID through short occlusion. If a person disappears for
# longer than this many frames, DeepSORT deletes the track; if they return later,
# they can be counted again as a new person pass.
DEEPSORT_MAX_AGE = 80
DEEPSORT_N_INIT = 3
DEEPSORT_MAX_COSINE_DISTANCE = 0.25
DEEPSORT_NN_BUDGET = 200

# If a counted ID disappears for a few frames and then reappears from a frame
# edge, treat it as a new person entering the frame.
REENTRY_MIN_ABSENT_FRAMES = 2

# Reused IDs that jump a long distance are probably a different person.
ID_SWITCH_DISTANCE_PX = 160

# A person appearing inside this edge band is treated as entering the frame.
ENTRY_EDGE_MARGIN_RATIO = 0.08

# A track must be detected this many frames before it becomes a counted person.
# This filters one-frame false detections.
MIN_TRACK_FRAMES = 5

# If a person is visible for enough frames but barely moves, still count them
# after this many detections. Useful when people stand in line.
STATIONARY_COUNT_FRAMES = 18

# If a person moves at least this far, count them as soon as MIN_TRACK_FRAMES is met.
MIN_MOVEMENT_PX = 25

TRAIL_LENGTH = 48
MAX_STALE_TRACKS = 120
DISPLAY_WIDTH = 1300


CLR_COUNTED = (0, 220, 0)
CLR_NOT_COUNTED = (0, 0, 255)
CLR_TEXT = (245, 245, 245)
CLR_BG = (25, 25, 25)


@dataclass
class TrackState:
    points: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    frames_seen: int = 0
    max_conf: float = 0.0
    counted: bool = False
    pass_count: int = 0
    last_seen_frame: int = 0
    last_box: tuple[int, int, int, int] | None = None

    @property
    def first_point(self) -> tuple[int, int] | None:
        return self.points[0] if self.points else None

    @property
    def last_point(self) -> tuple[int, int] | None:
        return self.points[-1] if self.points else None


def movement_distance(state: TrackState) -> float:
    first = state.first_point
    last = state.last_point
    if first is None or last is None:
        return 0.0
    return math.hypot(last[0] - first[0], last[1] - first[1])


def should_count_track(state: TrackState) -> bool:
    if state.counted:
        return False
    if state.frames_seen < MIN_TRACK_FRAMES:
        return False
    if movement_distance(state) >= MIN_MOVEMENT_PX:
        return True
    return state.frames_seen >= STATIONARY_COUNT_FRAMES


def box_touches_frame_edge(box: tuple[int, int, int, int], frame_width: int, frame_height: int) -> bool:
    x1, y1, x2, y2 = box
    margin_x = int(frame_width * ENTRY_EDGE_MARGIN_RATIO)
    margin_y = int(frame_height * ENTRY_EDGE_MARGIN_RATIO)
    return x1 <= margin_x or x2 >= frame_width - margin_x or y1 <= margin_y or y2 >= frame_height - margin_y


def should_start_new_episode(
    state: TrackState,
    foot: tuple[int, int],
    box: tuple[int, int, int, int],
    absent_frames: int,
    frame_width: int,
    frame_height: int,
) -> bool:
    if not state.counted:
        return False

    last_point = state.last_point
    if last_point is None:
        return False

    jump_distance = math.hypot(foot[0] - last_point[0], foot[1] - last_point[1])
    current_at_edge = box_touches_frame_edge(box, frame_width, frame_height)

    if absent_frames >= REENTRY_MIN_ABSENT_FRAMES and current_at_edge:
        return True

    if jump_distance >= ID_SWITCH_DISTANCE_PX:
        return True

    return False


def reset_for_new_episode(state: TrackState) -> None:
    state.points.clear()
    state.frames_seen = 0
    state.max_conf = 0.0
    state.counted = False


def draw_trail(frame, points: deque[tuple[int, int]], color: tuple[int, int, int]) -> None:
    pts = list(points)
    for i in range(1, len(pts)):
        alpha = i / max(1, len(pts) - 1)
        faded = tuple(int(c * alpha) for c in color)
        cv2.line(frame, pts[i - 1], pts[i], faded, max(1, int(3 * alpha)), cv2.LINE_AA)


def draw_person_overlay(frame, box, foot, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.circle(frame, foot, 5, (255, 255, 255), -1)
    cv2.circle(frame, foot, 7, color, 2)

    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    label_y1 = max(0, y1 - label_h - 10)
    cv2.rectangle(frame, (x1, label_y1), (x1 + label_w + 8, label_y1 + label_h + 8), color, -1)
    cv2.putText(frame, label, (x1 + 4, label_y1 + label_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, total_people: int, active_tracks: int, frame_index: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (420, 128), CLR_BG, -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    cv2.putText(frame, f"TOTAL PEOPLE : {total_people}", (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, CLR_COUNTED, 2, cv2.LINE_AA)
    cv2.putText(frame, f"ACTIVE TRACKS: {active_tracks}", (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.72, CLR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FRAME        : {frame_index}", (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, CLR_TEXT, 1, cv2.LINE_AA)


def resize_for_display(frame):
    if DISPLAY_WIDTH <= 0 or frame.shape[1] <= DISPLAY_WIDTH:
        return frame
    scale = DISPLAY_WIDTH / frame.shape[1]
    return cv2.resize(frame, (DISPLAY_WIDTH, int(frame.shape[0] * scale)))


def prune_tracks(states: dict[str, TrackState], frame_index: int) -> None:
    stale_ids = [tid for tid, state in states.items() if frame_index - state.last_seen_frame > MAX_STALE_TRACKS]
    for tid in stale_ids:
        del states[tid]


def run() -> None:
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    tracker = DeepSort(
        max_age=DEEPSORT_MAX_AGE,
        n_init=DEEPSORT_N_INIT,
        max_cosine_distance=DEEPSORT_MAX_COSINE_DISTANCE,
        nn_budget=DEEPSORT_NN_BUDGET,
        embedder="mobilenet",
        half=True,
        bgr=True,
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    states: dict[str, TrackState] = defaultdict(TrackState)
    total_people_count = 0
    paused = False
    frame_index = 0
    frame = None
    active_tracks = 0

    print("Running YOLOv8 + DeepSORT people counter. Press q to quit, r to reset, p to pause.")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            results = model(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False,
            )[0]

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                width = x2 - x1
                height = y2 - y1
                detections.append(([int(x1), int(y1), int(width), int(height)], conf, "person"))

            tracks = tracker.update_tracks(detections, frame=frame)
            active_tracks = 0
            for track in tracks:
                if not track.is_confirmed():
                    continue
                if track.time_since_update > 0:
                    continue

                active_tracks += 1
                track_id = str(track.track_id)
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                box = (x1, y1, x2, y2)
                foot = ((x1 + x2) // 2, y2)

                state = states[track_id]
                absent_frames = frame_index - state.last_seen_frame if state.last_seen_frame else 0
                if should_start_new_episode(state, foot, box, absent_frames, frame_width, frame_height):
                    reset_for_new_episode(state)

                state.frames_seen += 1
                state.last_seen_frame = frame_index
                state.last_box = box
                state.points.append(foot)

                if should_count_track(state):
                    state.counted = True
                    state.pass_count += 1
                    total_people_count += 1

                color = CLR_COUNTED if state.counted else CLR_NOT_COUNTED
                status = "COUNTED" if state.counted else "NOT-COUNTED"
                label = f"ID {track_id} {status} pass:{state.pass_count} seen:{state.frames_seen}"

                draw_trail(frame, state.points, color)
                draw_person_overlay(frame, box, foot, color, label)

            prune_tracks(states, frame_index)

        if frame is None:
            continue

        display = frame.copy()
        draw_hud(display, total_people_count, active_tracks, frame_index)
        cv2.imshow("YOLOv8 + DeepSORT People Counter", resize_for_display(display))

        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("r"):
            states.clear()
            total_people_count = 0
            print("Count reset.")

    cap.release()
    cv2.destroyAllWindows()

    print("\nFinal results")
    print(f"Total people counted: {total_people_count}")


if __name__ == "__main__":
    run()
