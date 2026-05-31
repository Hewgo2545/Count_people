# People Counting With YOLOv8, DeepOCSORT, and OSNet

This project counts people across a video using YOLOv8 detections and DeepOCSORT tracking from BoxMOT. It is designed for crowded/overlapping scenes where a person may be partially hidden and then reappear.

The current main script is `count_people.py`.

## Features

- Detects people with YOLOv8.
- Tracks people with DeepOCSORT.
- Optionally uses OSNet re-identification to reduce wrong ID reuse.
- Counts stable tracked people instead of counting every frame.
- Shows red overlay for not-counted tracks and green overlay for counted tracks.
- Saves an annotated output video as AVI.
- Supports output downscaling to reduce file size.

## Setup

Create and activate a virtual environment if you want an isolated install.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If `torchreid` asks for extra dependencies, install:

```powershell
pip install gdown
```

## Run

Make sure the video path and model path in `count_people.py` are correct:

```python
VIDEO_SOURCE = "entrance_stable.mp4"
MODEL_PATH = "yolov8m.pt"
```

Run:

```powershell
python count_people.py
```

Controls:

- `q`: quit
- `p`: pause/play
- `r`: reset count

## Output Video

The annotated result is saved to:

```python
OUTPUT_VIDEO_PATH = r"E:\Video\count_people_output.avi"
```

The output is downscaled to reduce file size:

```python
OUTPUT_SCALE = 0.5
```

For example, a `1920x1080` video is saved as `960x540`.

## Important Settings

Detection confidence:

```python
CONF_THRESHOLD = 0.8
```

DeepOCSORT memory:

```python
DEEPOCSORT_MAX_AGE = 30
DEEPOCSORT_MIN_HITS = 3
DEEPOCSORT_IOU_THRESHOLD = 0.3
```

Counting stability:

```python
MIN_TRACK_FRAMES = 5
STATIONARY_COUNT_FRAMES = 18
MIN_MOVEMENT_PX = 25
```

Re-entry / wrong-ID handling:

```python
REENTRY_MIN_ABSENT_FRAMES = 10
ID_SWITCH_DISTANCE_PX = 220
ENTRY_EDGE_MARGIN_RATIO = 0.08
```

OSNet re-identification:

```python
USE_OSNET_REID = True
OSNET_DISTANCE_THRESHOLD = 0.45
```

If people turn front-to-back and get duplicate IDs, try:

```python
USE_OSNET_REID = False
DEEPOCSORT_MAX_AGE = 60
```

## Video Stabilization

If the camera shakes, stabilize the video first:

```powershell
python stabilize_video_ecc.py --input entrance.mov --output entrance_stable.mp4
```

Then run `count_people.py` on the stabilized video.

## Disk Space Notes

If OpenCV prints:

```text
FFmpeg: Failed to write frame
```

the output drive is probably full or unavailable. Change:

```python
OUTPUT_VIDEO_PATH
```

to a drive with enough free space.
