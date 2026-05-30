"""
People Detector using YOLOv8

Install:
    pip install ultralytics opencv-python

Run:
    python people_detector.py
"""

import cv2
from ultralytics import YOLO

# ─── Settings ────────────────────────────────────────────────
SOURCE = r"D:\Coding by Hugo\count_people\entrance.mov"
MODEL  = "yolov8m.pt"
CONF   = 0.7   # ← change confidence threshold here (0.0 – 1.0)
# ─────────────────────────────────────────────────────────────

PERSON_CLASS_ID = 0  # COCO class 0 = person


def run() -> None:
    print(f"Loading model: {MODEL}")
    model = YOLO(MODEL)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {SOURCE}")

    print("Running — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[PERSON_CLASS_ID], conf=CONF, verbose=False)[0]

        people_count = 0
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            people_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 80), 2)
            label = f"person {confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), (0, 200, 80), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # HUD overlay
        hud = f"People detected: {people_count}"
        cv2.rectangle(frame, (0, 0), (280, 40), (20, 20, 20), -1)
        cv2.putText(frame, hud, (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 90), 2, cv2.LINE_AA)
        frame = cv2.resize(frame , (1500,700))
        cv2.imshow("People Detector — press Q to quit", frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    run()