# Real-Time Eye Detection System Using OpenCV

A university Computer Vision mini project. The program reads live video from a
webcam, finds faces, then finds eyes inside each face, and draws boxes on the
live video together with the measured FPS.

**Important:** detection uses the **pre-trained Haar Cascade XML classifiers that
ship with OpenCV** (`haarcascade_frontalface_default.xml` and
`haarcascade_eye.xml`). **No model was trained in this project** and no dataset
was collected or used for training.

## Objectives
- Capture real-time video from a webcam with OpenCV.
- Detect faces with a pre-trained Haar Cascade.
- Detect eyes inside each detected face with a second pre-trained Haar Cascade.
- Reduce duplicate eye boxes with a simple, explainable rule.
- Measure and display the real frames-per-second on a CPU-only laptop.

## Technologies
- Python 3.10, OpenCV (`opencv-python` 4.x), NumPy
- Windows 11, VS Code, Claude Code (development assistant)
- Development laptop: Intel i5-6300U, 8 GB RAM, no dedicated GPU

OpenCV is pinned to 4.x (`>=4.10,<5`) because OpenCV 5.x removed
`cv2.CascadeClassifier`, which Haar cascades need.

## Methodology: Haar Cascade classifiers
1. A *Haar feature* is a small rectangle pattern: (sum of pixel brightness in
   the white part) minus (sum in the black part). For example, the eye region
   is usually darker than the cheeks below it.
2. An *integral image* lets OpenCV compute any rectangle sum very quickly.
3. A window slides over the image at many sizes. A *cascade* is a chain of
   stages; each stage is a quick test, and a window that fails any stage is
   rejected immediately. Most windows are rejected early, so it is fast on a CPU.
4. The learned features and thresholds are stored in the XML files, which were
   trained by others (OpenCV), not by us.

## Pipeline
For every webcam frame (`main.py`):
1. `Camera.read()` grabs a BGR frame.
2. `FaceDetector.detect()` converts the frame to grayscale and runs
   `detectMultiScale` (`scaleFactor=1.1`, `minNeighbors=5`, `minSize=(60, 60)`).
3. For each face, `EyeDetector.detect()`:
   - cuts out the **face ROI** (Region Of Interest) and keeps only its **top 60%**
     (eyes are in the upper face; this skips the nose and mouth),
   - converts the ROI to grayscale (no histogram equalization),
   - runs `detectMultiScale` with `haarcascade_eye.xml`
     (`scaleFactor=1.1`, `minNeighbors=5`, `minSize=(20, 20)`),
   - applies duplicate suppression (below),
   - converts the boxes to full-frame coordinates.
4. Green boxes (faces), blue boxes (eyes) and the text `FPS / Faces / Eyes`
   are drawn, and the frame is shown. Press **Q** (or close the window) to quit.

### Face ROI and coordinates
The eye detector only sees the cropped ROI, so it reports positions relative to
the ROI's top-left corner. Because the ROI starts at the face's top-left corner,
we get frame coordinates by adding the face position:
`frame_x = face_x + eye_x`, `frame_y = face_y + eye_y`.

### Duplicate suppression (`suppress_duplicate_eyes`)
The eye cascade sometimes returns two or three overlapping boxes on one eye.
The rule keeps **at most one box per half of the face**:
1. Sort boxes into the left or right half of the face.
2. Reject implausible boxes: centre outside the upper-face zone (12%-50% down the
   face, 10%-45% or 55%-90% across), or wider than 40% of the face width.
3. Group boxes in the same half that overlap by more than 50% of the smaller
   box's area (they are the same eye).
4. Keep the **smallest** box of each group (usually the tight box on the eye).
5. If separate groups remain in a half, keep the one closest to a typical eye
   position (about 30% / 70% across, 33% down).

There is deliberately no minimum-width rule: it was tested and wrongly removed a
genuine small eye detection.

## Installation
```powershell
cd D:\Projects\eye-detection-system
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts\check_environment.py
```
`check_environment.py` prints the versions and confirms both cascade files load.

## Run
```powershell
python main.py
```
Press **Q** to quit. If the webcam cannot be opened, or a cascade file is
missing/invalid, the program prints a clear error message and exits.

## Testing
```powershell
python -m unittest discover -s tests -v
```
The 11 unit tests cover the duplicate-suppression rule (empty input, overlapping
duplicates, smaller-box choice, genuine small eyes, left/right separation,
implausible boxes, at most two outputs) and cascade-loading errors.

## Project structure
```
eye-detection-system/
├── main.py                     # Entry point: webcam loop, drawing, FPS, quit on Q
├── requirements.txt            # opencv-python>=4.10,<5 and numpy
├── README.md                   # This file
├── .gitignore                  # Ignores .venv, caches, output folder
├── src/
│   ├── __init__.py             # Makes src a Python package
│   ├── camera.py               # Camera class: open, read, properties, release
│   └── detector.py             # FaceDetector, EyeDetector, suppress_duplicate_eyes, drawing helpers
├── scripts/
│   └── check_environment.py    # Verifies Python/OpenCV/NumPy and cascade files
└── tests/
    ├── __init__.py
    └── test_detector.py        # Unit tests
```

## Results (limited, measured)
- **Average FPS with face + eye detection and display: 17.5**, measured manually
  by running `python main.py` on the development laptop (Intel i5-6300U, no GPU)
  at the webcam's 640x480 resolution.
- Earlier headless timing on the same laptop: about 32 ms per frame for face
  detection and about 12 ms per face for eye detection.
- In a controlled comparison on 90 saved webcam frames of one person, duplicate
  suppression reduced frames with three or more eye boxes per face from 5 to 0
  and the total number of eye boxes from 128 to 122. This is a count of boxes,
  **not an accuracy measurement**.

No accuracy, precision or recall has been measured, because no labelled
evaluation data was created.

## Limitations
- All testing was done with **one person and one webcam under limited lighting
  conditions**. Results may differ for other people, cameras or environments.
- Haar cascades work best on frontal, well-lit faces; turned or strongly tilted
  heads, occlusion (hands, hair) and dim light can cause missed detections.
  In an early test with a tilted face and a hand in front of it, eyes were
  detected in only 1 of 60 frames.
- A detected box is not guaranteed to be a correct eye; false detections and
  duplicates can still occur. The suppression rule only removes overlapping
  boxes in the same half of the face and cannot recover a missed eye.
- The rule's position thresholds assume a roughly frontal face.
- Only faces and eyes are boxed: no iris, pupil or blink detection.
- Performance is CPU-only and depends on the machine.

## Possible future work
Not implemented: blink detection, other face/eye detectors (for example deep
learning models), labelled evaluation, and camera index selection.
