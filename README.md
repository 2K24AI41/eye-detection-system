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
Run these commands from the project folder (the folder that contains this README):
```powershell
cd path\to\eye-detection-system
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

## Web-Based Eye Detection Interface

A small Flask page shows the same detection pipeline in a browser: live webcam preview
with green face boxes and blue eye boxes, Start/Stop buttons, a status indicator and the
current face count, eye count and FPS. It reuses `Camera`, `FaceDetector`, `EyeDetector`
and the drawing helpers from `src/` unchanged; `main.py` is not used or modified.

### Features
- Dark, single-page interface (heading, large live preview, Start Camera / Stop Camera,
  status dot, faces / eyes / FPS tiles).
- Clear messages for camera errors (for example "Could not open webcam 0"), server
  errors, a lost video stream, and an unreachable server.
- The preview reconnects up to 3 times if the video stream drops, then asks you to press
  Stop and Start.
- Designed for desktop and laptop browsers. The page is one centred column (at most
  820 px wide) that narrows on short browser windows so the preview, buttons and
  statistics fit without scrolling. There is no phone or tablet layout, and it has not
  been tested on those devices.

### Installation
Flask is listed in `requirements.txt`, so the normal installation is enough (from the
project folder):
```powershell
cd path\to\eye-detection-system
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Run and stop
```powershell
python app.py
```
Open **http://127.0.0.1:5000** in a browser and press **Start Camera**. Stop the server
with **Ctrl+C** in the terminal; the camera is released on exit. The server listens on
`127.0.0.1` only (not reachable from other computers), with Flask debug mode and the
reloader switched off on purpose (the reloader would start a second process and open the
webcam twice).

### Project structure (web files)
```
app.py                          # Flask app: routes and security checks, run with python app.py
src/web_camera_service.py       # DetectionService: the single camera worker thread
web/templates/index.html        # The page
web/static/css/style.css        # Dark theme and desktop layout
web/static/js/app.js            # Buttons, status polling, video stream handling
tests/test_web_camera_service.py        # Service and route tests (fake camera)
tests/test_web_phase11_regressions.py   # Regression tests for the Phase 1.1 fixes
```

### Flask routes
| Route | Purpose |
|---|---|
| `GET /` | The page |
| `POST /api/start` | Start the camera worker (does nothing if one is already alive); returns the status as JSON |
| `POST /api/stop` | Stop the worker and release the camera; returns the status |
| `GET /api/status` | JSON: `state` (stopped, starting, running, stopping, error), `fps`, `faces`, `eyes`, `error`, `note` |
| `GET /video_feed` | The MJPEG video stream |

Requests with an unexpected `Host` header, a foreign `Origin` on POST, or a cross-site
`Sec-Fetch-Site` on POST and `/video_feed` are refused with HTTP 403.

### MJPEG streaming architecture
- **One worker thread** owns the webcam. It reads a frame, runs face then eye detection,
  draws the boxes, encodes the frame as a JPEG (quality 80) and stores it as the "latest
  frame". It is capped at about 15 frames per second so detection does not starve the web
  server. The worker is the only code that opens or releases the camera (in a `finally`
  block, so also after errors).
- `/video_feed` never touches the camera. It sends the latest JPEG to the browser as a
  `multipart/x-mixed-replace` (MJPEG) stream, so extra browser tabs cost no extra
  detection and cannot open a second camera.
- Each start creates a new worker "generation"; a stream from an older worker ends
  instead of attaching to the new one.
- If `Stop` does not finish within 5 seconds the state stays `stopping`, a second worker
  is refused until the first one has really ended, and the status shows the problem.

### Camera sharing and automatic shutdown
- Only one program can use the webcam at a time. Do not run `python app.py` (with the
  camera started), `python main.py`, `scripts/capture_dataset.py` or a video call together;
  the second one reports a camera error if it cannot access the webcam.
- If no browser has fetched the video stream for **30 seconds** (for example you closed
  the tab), the worker stops itself and releases the camera; the page shows
  "stopped automatically: no viewer for 30 s".

### Testing the web interface
```powershell
python -m unittest tests.test_web_camera_service tests.test_web_phase11_regressions -v
python -m unittest discover -s tests          # whole project
```
The web tests use fake cameras and fake detectors only: they never open the webcam or
start the server. For the latest full test result, see "Testing" below.

### Current limitations
- The browser page itself (layout, stream reconnection, button states) has no automated
  browser test; it was checked by hand in one browser on one Windows computer.
- The 15 FPS cap and JPEG quality 80 were chosen, not tuned, and the web FPS has not been
  benchmarked separately from `main.py`.
- It uses Flask's built-in development server, with no login: it is meant for local use
  only and must not be exposed to a network.
- Detection is the unchanged Haar-cascade pipeline, so the accuracy limits listed under
  "Limitations" below apply equally to the web interface.
- The page streams one webcam (index 0) at the driver's default resolution.

## Testing
```powershell
python -m unittest discover -s tests -v
```
All tests use synthetic images, fake cameras and temporary folders: none opens the webcam,
shows a window or reads the real `dataset/`.

| Test file | Tests | Covers |
|---|---|---|
| `test_detector.py` | 11 | duplicate-suppression rule and cascade-loading errors |
| `test_capture_dataset.py` | 106 | dataset capture script (fake camera) |
| `test_dedup_manifest.py` | 21 | de-duplication manifest builder |
| `test_annotate_dataset.py` | 74 | annotation tool: controls, saving, backups, sessions (no window) |
| `test_annotate_reopen.py` | 76 | correcting saved annotations, data-consistency checks, pass-2 gate |
| `test_annotate_metadata_errors.py` | 19 | annotation tool with damaged `metadata.csv` (blind error messages) |
| `test_validate_labels.py` | 41 | annotation validator rules and the restore marker |
| `test_validate_metadata.py` | 41 | metadata validation shared by the tools |
| `test_restore_safety.py` | 56 | backup sets, restore and rollback, restore marker, dataset locks, consistent snapshots, message redaction |
| `test_web_camera_service.py` | 25 | web service and routes (fake camera) |
| `test_web_phase11_regressions.py` | 17 | regression tests for the web interface fixes |

Latest full result on the development machine (Windows 11): all 11 test files ran as
**487 tests, OK: 483 passed, 0 failed, 4 skipped**. The 4 skipped tests need symbolic
links, which this Windows account is not allowed to create, so symbolic-link behaviour was
not exercised there.

## Dataset capture (Milestone 5 evaluation)
`scripts/capture_dataset.py` records the evaluation images defined in
`docs/EVALUATION_PROTOCOL.md`. It only saves 640x480 PNG images and a metadata
CSV inside the Git-ignored `dataset/` folder; it does not run the detector.
```powershell
python scripts\capture_dataset.py --status     # progress table, no camera
python scripts\capture_dataset.py              # asks for condition / participant / sub-condition
python scripts\capture_dataset.py --condition turn --participant P01 --sub 30deg
```
In the video window press **SPACE** to save (at most one image every 2 seconds)
and **Q** or **ESC** to quit. Conditions: `normal, dim, strong, turn, tilt, dist,
glasses, bg`.

**The collection is complete and frozen.** The 125 images in `dataset/images/` and their
rows in `dataset/metadata.csv` are the final dataset (`docs/EVALUATION_PROTOCOL.md`,
section 6.6). The commands above describe how it was collected; they are not meant to be
run again to add, replace or retake images of this dataset. `--status` only reads.

Safety behaviour:
- **Consent.** Participant `P02` (the volunteer) needs a typed `yes` (or
  `--consent-confirmed`) before the camera opens. This is only a **self-declaration**:
  it does **not** replace the signed consent form, which must exist before any image
  of the volunteer is taken. Permission to show images in the report or slides is a
  **separate** yes/no on that form and is not recorded by the script.
- **Git.** The script asks `git check-ignore` whether the dataset paths are ignored
  and refuses to run if they are not (for example after a `!dataset/` rule, or if the
  files are already tracked). Git must be installed.
- **No overwriting.** Images are created exclusively. A failed or interrupted save
  removes its own incomplete image. Symbolic links and junctions in `dataset/`,
  `dataset/images/` or `dataset/metadata.csv` are refused.
- **Metadata** (`dataset/metadata.csv`) is validated on every start and save:
  duplicate filenames, mismatched condition/participant/filename and incomplete rows
  stop the script with a line number. It never appends onto an incomplete last row,
  and a failed write restores the file. A UTF-8 byte-order mark is accepted.
  Timestamps include the UTC offset, for example `2026-01-02T03:04:05+05:00`.
- **Do not delete, rename, replace or edit** any image or any row of
  `metadata.csv`, and do not edit the CSV by hand. The validator
  (`scripts/validate_labels.py`), the annotation tool and the de-duplication script
  (`scripts/build_dedup_manifest.py`) all need `dataset/images/` and `metadata.csv` to
  match exactly, one row per image and one image per row. A missing, extra, renamed or
  replaced file or row makes them stop or fail, and changes the recorded checksums of the
  frozen de-duplicated subset. If something looks wrong, do not fix it by hand: leave the
  files as they are and note it for review.
- **Unexpected files** in `dataset/images/` (for example `Thumbs.db`) are reported
  as warnings and ignored. An image-named file without a metadata row stops the script
  (this follows a crash).

## De-duplicated evaluation subset (Milestone 6)
Many consecutive dataset images are near-identical, so results are reported on the full
dataset **and** on a de-duplicated subset (details in `docs/EVALUATION_PROTOCOL.md`,
section 11). The approved rule is pixel-similarity only: 64x48 grayscale, grouped by
recorded condition and sub-condition, new cluster when the mean absolute difference to the
cluster's first image exceeds 2.5, earliest image kept. It gives 73 representatives (68
person images, 5 background images) out of 125 images; nothing is deleted.
```powershell
python scripts\build_dedup_manifest.py --verify    # checks dataset\deduplication\* byte for byte
```
Background-only images are not part of the person-image metrics.

## Ground-truth annotation (Milestone 7)
`scripts/annotate_dataset.py` is a blind manual annotation tool (OpenCV window): images come
in a fixed-seed random order, filenames and conditions are hidden, nothing is pre-filled and
no detector is ever run. Labels go to `dataset/labels.csv` and `dataset/annotation_status.csv`
(Git-ignored); timestamped backup sets are kept in `dataset/backups/`. Each set has a
manifest (`backupset_*.txt`) with the size and SHA-256 of every member; a file that did not
exist at that moment is recorded in the manifest as absent instead of being copied.
```powershell
python scripts\annotate_dataset.py --practice   # learn the controls on synthetic images
python scripts\annotate_dataset.py --status     # progress only
python scripts\annotate_dataset.py              # annotate (pass 1); starts at the first unannotated image
python scripts\validate_labels.py --partial     # check the files so far (omit --partial when finished)
```
| Control | Action |
|---|---|
| drag (left button) | draw the face box |
| click | place an eye centre: first the eye on the **left of the image**, then the right one |
| `h` | the next eye is hidden: no point is stored |
| `u` | toggle *uncertain* on the last placed face/eye |
| `n` | no face visible (`no_face`) |
| `f` / `l` / `r` | redo the face box / the left eye / the right eye: that one object is cleared and must be placed again (works on saved annotations too) |
| `z` | undo the last action (also puts a redone object back exactly as it was) |
| Enter | validate and save, then next image |
| `b` | previous image (press twice if this image has unsaved changes) |
| `q` / ESC | quit (press twice if unsaved changes); saved images are kept |

To correct an image that is already saved, go back to it (`b`, or start the tool again: it opens
at the first unannotated image, or at the first image once all are done) and press `f`, `l` or
`r` (upper case works too). Only that object is cleared, together with its *uncertain* flag;
everything else stays as saved. `z` puts the saved object back exactly, flag included, and
cannot go further back than what is on disk. Enter refuses to save until the object has been
placed again. The usual rules still apply: a new face box must still contain the existing eyes
(otherwise redo the eye first with `l` or `r`), and the left eye stays left of the right eye. To
change a visible eye to hidden, press `l` or `r` and then `h`. To turn a face into `no_face`,
redo the face and both eyes, then press `n` (`n` also switches `no_face` off again). `f`, `l`
and `r` are refused, with a message, on a `no_face` image (press `n` first) and for an eye that
is not placed yet. When a changed image is saved, the previous version of both files is kept in
`dataset/backups/` (an `_edit` backup set) and the status line says so; saving identical data
changes nothing. Leaving an image with unsaved edits needs `b` or `q` twice.

A magnifier at the top right of the window shows the area around the cursor (4x). Every image
is saved the moment you press Enter, so an interrupted session loses at most the image you
were on: run the same command again to resume. If a save was interrupted between the two
files, the tool notices, tells you, and the image is simply annotated again. If a file is ever
damaged, nothing is changed and the message names the latest backups;
`python scripts\annotate_dataset.py --restore-latest-backup` puts back the newest backup set
whose files all match its manifest, exactly as it was (a member recorded as absent is
removed again); incomplete or damaged sets are never used. The replaced files are kept in
`dataset/backups/` too, as a `prerestore` set. The repeat check is
`--pass 2` (14 person images, only 24 hours or more after pass 1 is finished); it writes
separate `labels_pass2.csv` / `annotation_status_pass2.csv` files and never changes pass 1.
Only one annotation session or restore can use the dataset at a time: a second instance is
refused with a message and changes nothing (`--status` and `validate_labels.py` still work).

### Interrupted restore

While `--restore-latest-backup` runs, it keeps a small marker file, `dataset/.restore_in_progress`,
that names the pass and the step it is on. It is removed when the restore finishes. If it is
still there, a restore was interrupted (for example by a crash or power cut), and the label and
status files of that pass may mix old and restored data. Annotation is blocked until this is
resolved. If the marker can be read, the message names the pass: run
`python scripts\annotate_dataset.py --pass N --restore-latest-backup` for that pass to finish it.

If the marker cannot be read (empty, damaged, or not a plain file), the tool cannot know which
pass was being restored, so it will not restore either pass automatically and never changes the
marker. Recover by hand:

1. Inspect both passes: `python scripts\validate_labels.py --partial` and
   `python scripts\validate_labels.py --pass 2 --partial`, and
   `python scripts\annotate_dataset.py --status` (add `--pass 2` for the repeat check). Each
   report shows the marker error; the remaining errors show whether that pass's labels and
   status files disagree.
2. Before changing anything, copy the current `labels*.csv` and `annotation_status*.csv` files
   and the `.restore_in_progress` marker to a safe place outside the dataset folder.
3. Decide which pass was interrupted only if the reports (or your own record of the restore)
   establish it clearly. If they do not, stop here and keep the copies; do not guess.
4. Only after this review, remove `.restore_in_progress` by hand. This deliberate step is the
   only way the marker is cleared; the tool never does it for an unreadable marker.
5. Run `python scripts\annotate_dataset.py --pass N --restore-latest-backup` for the pass you
   identified. The files it replaces are kept in `dataset/backups/` as a `prerestore` copy.

Never delete or edit images or `metadata.csv` as part of this. Backups made before the current
backup format (no `backupset_*.txt` manifest) can still be restored when they form an exact
pair, but their contents cannot be verified, because they have no checksums.

## Project structure
```
eye-detection-system/
├── main.py                     # Entry point: webcam loop, drawing, FPS, quit on Q
├── app.py                      # Flask web interface entry point (python app.py)
├── requirements.txt            # opencv-python>=4.10,<5, numpy and flask>=3,<4
├── README.md                   # This file
├── .gitignore                  # Ignores .venv, __pycache__ / *.pyc, .vscode, output, dataset,
│                               # local env/secret files, editor/tool caches and OS files
├── src/
│   ├── __init__.py             # Makes src a Python package
│   ├── camera.py               # Camera class: open, read, properties, release
│   ├── detector.py             # FaceDetector, EyeDetector, suppress_duplicate_eyes, drawing helpers
│   └── web_camera_service.py   # DetectionService: single camera worker thread for the web interface
├── scripts/
│   ├── check_environment.py    # Verifies Python/OpenCV/NumPy and cascade files
│   ├── capture_dataset.py      # Controlled webcam capture for the evaluation dataset
│   ├── build_dedup_manifest.py # Builds/verifies the de-duplicated evaluation subset
│   ├── annotate_dataset.py     # Blind manual ground-truth annotation tool (OpenCV)
│   └── validate_labels.py      # Read-only validation of the annotation files
├── docs/
│   └── EVALUATION_PROTOCOL.md  # Evaluation plan for Milestone 5
├── web/
│   ├── templates/
│   │   └── index.html          # The web page
│   └── static/
│       ├── css/style.css       # Dark theme and desktop layout
│       └── js/app.js           # Buttons, status polling, video stream handling
└── tests/
    ├── __init__.py
    ├── test_detector.py        # Unit tests for the detector
    ├── test_capture_dataset.py # Unit tests for the capture script (no webcam needed)
    ├── test_dedup_manifest.py  # Unit tests for the de-duplication manifest builder
    ├── test_annotate_dataset.py # Tests for the annotation tool (synthetic images, no window)
    ├── test_annotate_reopen.py # Tests for correcting saved annotations and the pass-2 gate
    ├── test_annotate_metadata_errors.py # Annotation tool with a damaged metadata.csv
    ├── test_validate_labels.py # Tests for the annotation validator
    ├── test_validate_metadata.py # Tests for the shared metadata validation
    ├── test_restore_safety.py  # Backups, restore, locks, snapshots and message redaction
    ├── test_web_camera_service.py       # Web service and route tests (fake camera, no webcam)
    ├── test_web_phase11_regressions.py  # Regression tests for the web interface fixes
    └── annotation_helpers.py   # Shared helpers for the annotation tests
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
