# Evaluation Protocol (Milestone 5 / Milestone 6 update)

Real-Time Eye Detection System Using OpenCV (pre-trained Haar Cascades).

Status: **protocol version 3 (Milestone 6: de-duplication rule frozen)**. Data collection is
complete (125 images). Ground-truth labels have not been created yet, no detector
evaluation has been run, and no results exist. The de-duplication rule (Method B, T = 2.5)
is approved and final (section 11).
Every number in this project report must come from running this protocol, and no number
may be quoted from anywhere else as "accuracy".

## 1. Purpose and scope

**Purpose.** Measure, honestly and reproducibly, how the existing detector behaves under
different conditions: how often it finds faces and eyes, how often it reports things that
are not there (false positives), how often it misses things that are there (false
negatives), and how fast it runs on the development laptop.

**In scope**
- Face detection and eye detection by the frozen baseline (section 2).
- Seven collected capture conditions (section 3) of one participant (and background-only
  images).
- Manual ground-truth annotation with the workflow and validation of section 6.
- Detection quality against hand-labelled ground truth (sections 6-8), reported on the
  full dataset and on a de-duplicated subset (section 11).
- Processing time measured offline (section 9).

**Out of scope**
- Changing, tuning or "fixing" the detector. Problems found are recorded and handled in a
  later, separate milestone.
- Training any model. The detector uses OpenCV's pre-trained Haar Cascade XML files only.
- Iris, pupil or blink detection; other detectors; any claim of broad generalisation.

**Functional testing versus formal evaluation.** Functional tests (the existing unit tests,
`scripts/check_environment.py`) show that the code runs and behaves as designed. They say
nothing about how often detections are correct. Only comparison against labelled ground
truth (this protocol) can support precision and recall figures. Without labels, the only
honest statement is a detection rate ("a face was reported in N of M images"), which must
not be called accuracy.

## 2. Frozen baseline

Nothing below may change from the start of data collection until the final results are
produced.

| Item | Value |
|---|---|
| Git branch / commit | `main` / `ada8c19` |
| Face cascade | `haarcascade_frontalface_default.xml` (OpenCV, pre-trained) |
| Face `detectMultiScale` | `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(60, 60)` |
| Eye cascade | `haarcascade_eye.xml` (OpenCV, pre-trained) |
| Eye `detectMultiScale` | `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(20, 20)` |
| Eye search region | Upper 60% of each detected face box |
| Histogram equalization | Not used |
| Duplicate suppression | v2 (`suppress_duplicate_eyes`, `src/detector.py`) |
| Input resolution | 640 x 480 |
| Software | Python 3.10.11, OpenCV 4.14.0, NumPy 2.2.6 |
| Hardware | Intel i5-6300U, 8 GB RAM, no dedicated GPU, Windows 11 |

**Duplicate suppression v2, as implemented** (all positions are fractions of the face box):
boxes are split into the left or right half of the face; a box is rejected if its centre is
not 12%-50% of the way down the face, if its centre is not 10%-45% (left) or 55%-90%
(right) across, or if it is wider than 40% of the face width; boxes in the same half that
overlap by more than 50% of the smaller box's area form one group and the smallest box of
each group is kept; if several groups remain in a half, the one closest to the typical eye
position (30% or 70% across, 33% down) is kept. At most one eye per half, so at most two
eyes per face.

**Freeze check (run before the final evaluation and benchmark):**
- `git rev-parse --short HEAD` must print `ada8c19`, or `git diff ada8c19 -- src/ main.py`
  must print nothing. If it does not, results must state the actual commit used.
- The versions in the table must still match (`python scripts/check_environment.py`).

New scripts (capture, annotation, validation, evaluation) are added in later steps. They
must not alter `src/detector.py`, `src/camera.py` or `main.py`.

**No detector in the annotation tools.** The annotation and validation tools (section 6)
must never import, call or display the detector or any detector output (no `src.detector`,
no `cv2.CascadeClassifier`). Unit tests will check this. The detector is first run on the
dataset only after pass-1 labels are complete and validated.

**Dataset snapshot.** Before labelling starts, a SHA-256 manifest of all 125 images and of
`dataset/metadata.csv` is recorded (`dataset/manifest_sha256.txt`, Git-ignored). If any of
these files change afterwards, previously computed results are no longer valid. Existing
dataset images and metadata are never edited silently: any correction needs an entry in
the change log (section 6.9).

## 3. Dataset (as collected)

New images only. The 90 development frames collected earlier were used to choose the
suppression thresholds, so they are **excluded** from every evaluation number and are not
included in the dataset.

The original plan (version 1) was 140 images with 8 conditions and two participants.
The collected dataset is 125 images, seven conditions, one participant:

| # | Condition (code) | Planned | Collected | Description and notes |
|---|---|---|---|---|
| 1 | Normal indoor lighting (`normal`) | 20 | 20 | Frontal, normal indoor light |
| 2 | Dim indoor lighting (`dim`) | 15 | 15 | Described as dim indoor lighting. Mean pixel brightness is only modestly lower than `normal` (automatic webcam exposure may have compensated), so no light level is claimed |
| 3 | Uneven / harsh lighting (`strong`) | 15 | 15 | All recorded as `side` in the metadata, no `back` images. Described cautiously as uneven/harsh lighting: the images are not one consistent lighting setup (see section 5) and the side of the light source was not verified |
| 4 | Head turned (`turn`) | 20 | 20 | 10 recorded `30deg`, 10 `60deg`. Angles are approximate (estimated by the participant, not measured) |
| 5 | Head tilted / looking down (`tilt`) | 15 | 15 | 7 recorded `roll`, 8 recorded `down`. See the note below |
| 6 | Nearer / farther from the webcam (`dist`) | 20 | 20 | 10 recorded `near`, 10 `far`. Described as nearer/farther: physical distances were **not** measured |
| 7 | Glasses (`glasses`) | 15 | **0** | Intentionally excluded (no glasses available). Glasses are not evaluated |
| 8 | Background-only (`bg`) | 20 | 20 | Limited diversity: 3 near-black noise frames and 17 near-identical frames of one plain wall view (see below) |
| | **Total** | **140** | **125** | 20+15+15+20+15+20+20 = 125 |

- **Participants.** All 105 person images are of **one participant (`P01`, the project
  author)**. The 20 background-only images have participant `NONE`. Participant `P02` has
  no images; the two-participant plan was not carried out.
- **Tilt sub-conditions and `tilt_P01_007`.** By decision, the metadata of the tilt images
  is **not changed**. `tilt_P01_007` is recorded as `roll`; in the author's visual review
  of the contact sheets it appears upright with a downward gaze, like the `down` images.
  Results that split `tilt` into `roll` and `down` must state the sub-condition counts as
  recorded (7 and 8) and mention this visual uncertainty.
- **Turn.** `30deg` and `60deg` are the intended approximate angles, not measurements.
- **Background.** In the author's visual review no person is visible in any of the 20
  images. `bg_NONE_001`-`003` are near-black noise frames (likely a covered lens or no
  light). `bg_NONE_004`-`020` show one plain wall/corner view with only slight tone and
  exposure drift. Background results are therefore a limited test of false positives on
  plain surfaces and must not be generalised to cluttered or varied scenes.
- Angles, distances and lighting are recorded as stated conditions, not as precise
  measurements.
- All images have exactly one person (or nobody). No images with two people.

Folder layout (all inside `dataset/`, which is excluded from Git):

```
dataset/
  images/                    # the 125 PNG files (never modified)
  metadata.csv               # one row per image (section 5), never edited silently
  metadata_backup.csv        # stale earlier snapshot (85 rows); kept, NOT part of the dataset
  manifest_sha256.txt        # SHA-256 of images + metadata.csv (created before labelling)
  labels.csv                 # pass-1 ground truth (section 6)
  annotation_status.csv      # one row per image: labelled / no_face (section 6)
  labels_pass2.csv           # repeat-check labels for 14 images (section 6.9)
  annotation_status_pass2.csv
  backups/                   # timestamped copies of the label files, one per session
  deduplication/             # manifest.csv, summary.csv, methodology.json (section 11)
  results/                   # outputs of later evaluation runs
```

`metadata_backup.csv` is kept unchanged, is Git-ignored like everything in `dataset/`, and
must not be used by any tool or analysis.

## 4. Participants, consent and privacy

- **One adult participant:** the project author (`P01`). No other person appears in the
  dataset, so no third-party consent form applies to the collected data. If a volunteer's
  images were ever added later, written consent would have to be obtained **before** any
  image is taken (what is captured, why, local storage only, not committed or uploaded,
  right to withdraw and have images deleted, and a separate yes/no for appearing in the
  report or slides), stored outside the project.
- Participants are identified only by `P01` / `NONE` in files. No names, emails or other
  personal details appear in filenames, metadata or labels.
- The images show the author's face. Any copy made for review (contact sheets, cluster
  previews, overlays) is stored **outside** the project folder (`%TEMP%`) and deleted after
  review. Images in the report or slides are the author's own choice.
- Retention: images are kept locally until the project assessment is complete, then deleted
  unless the participant decides otherwise.
- `dataset/` is Git-ignored (`.gitignore`) so images, metadata and labels cannot be
  committed by accident. The capture script verifies this with `git check-ignore`.

## 5. Image capture requirements and deviations

Requirements (version 1, unchanged):

- **Resolution:** 640 x 480, the webcam's native size in this project. Each image is
  checked to have shape (480, 640, 3); anything else is rejected.
- **Camera:** the same laptop webcam (index 0) for every image, with the same driver
  settings. No other camera, phone photo or downloaded image is mixed in.
- **Format:** lossless PNG, so compression does not affect detection.
- **Unique filenames:** `<cond>_<pid>_<nnn>.png`, for example `dim_P01_007.png`, where
  `<cond>` is one of `normal, dim, strong, turn, tilt, dist, glasses, bg` and `<nnn>` is a
  running number per condition. A filename is never reused or overwritten.
- **Metadata:** one row per image in `dataset/metadata.csv` with these columns:
  `filename, participant_id, condition, sub_condition, capture_time, notes`.
  `participant_id` is `P01`, `P02` or `NONE` (background-only). `sub_condition` records,
  for example, `30deg`/`60deg`, `near`/`far`, `side`/`back`, `roll`/`down`.
- **No near-duplicates:** images are taken by a deliberate key press, not by recording a
  stream. Wait at least 2 seconds between captures and change the pose slightly each time.
- **No post-processing:** no cropping, brightness change or retouching.
- **Selection rule:** capture first, then keep the images without looking at the detector's
  output. Do not delete an image because the detector failed on it; only technical faults
  (blur from camera shake, wrong condition, wrong resolution) justify rejecting an image.

**Documented deviations and observations** (from the dataset audit and visual review; the
dataset is kept as collected, with all 125 images):

- **Near-duplicate bursts.** The 2-second minimum interval was respected (minimum interval
  between captures 2.0 s, median about 3 s), but many consecutive images of the same
  condition are visually near-identical (for example `tilt_P01_008`-`015`,
  `dist_P01_001`-`010`, `strong_P01_009`-`015`, and most of `bg_NONE_005`-`020`). This does
  not meet the "change the pose slightly each time" requirement. The effective number of
  independent images is lower than 125. This is why results are also reported on a
  de-duplicated subset (section 11).
- **Four capture sessions, conditions confounded with session.** Session 1 (12:24-12:31):
  `normal` and `dim`. Session 2 (16:36-16:38): `strong`. Session 3 (16:43-16:51): `turn`
  and `tilt`. Session 4 (16:59-17:06): `dist` and `bg`. Time of day, scene set-up and
  webcam automatic exposure therefore differ together with the condition, and a difference
  between conditions cannot be separated from a session effect.
- **Exposure and scene changes inside conditions.** For example `strong` contains at least
  three different lighting set-ups (`strong_P01_001`-`006`, `007`-`008`, `009`-`015`), and
  `turn` shows an exposure/scene change between `turn_P01_011` and `turn_P01_012`.
- **Metadata edited by hand after capture.** `metadata.csv` starts with a UTF-8 byte-order
  mark that the capture script never writes, so another program re-saved it, and the
  earlier snapshot `metadata_backup.csv` records at least `tilt_P01_008`-`012` as `roll`
  where `metadata.csv` has `down`. The current `metadata.csv` is the authoritative
  version and is not changed.
- **Backgrounds.** The requirement of varied backgrounds was not met (section 3).
- **Distance.** Visually, faces in the `near` images are about 1.5 times as wide as in the
  `far` images (estimated by eye on a ruler overlay for a few images, roughly 235 px
  against 150 px; not a measurement). Even the `far` faces are large, so this set tests
  moderate scale change and not very small faces.
- Capture times all carry the +05:00 offset.

## 6. Ground-truth annotation

Ground truth is created by hand by the author (single annotator) with a purpose-built
annotation tool, **without displaying or using the detector's output**.

### 6.1 Tools

- `scripts/annotate_dataset.py` (OpenCV window, runs on Windows 11 without a GPU, no extra
  packages) and `scripts/validate_labels.py`. Both are built in a later task.
- Neither tool may import or run the detector (section 2). A unit test checks that their
  source does not reference `src.detector` or `CascadeClassifier`.
- A `--practice` mode writes only to a throw-away file outside `dataset/`, to learn the
  controls without creating real labels.

### 6.2 Blind, randomised workflow

- The images are shown **one at a time in a random order** produced with a fixed integer
  seed. The order is the list of filenames sorted by the SHA-256 of `<seed>:<filename>`
  (so it does not depend on the Python version). **Order seed: 20261120** (fixed in
  `scripts/validate_labels.py`; it must not change once labelling has started).
- **Filenames and condition names are not shown**; the screen shows only the position
  ("17/125"). The image content can of course still reveal its condition.
- Labels are **not pre-filled** from a previous image. Every image is labelled from
  scratch and confirmed.
- Controls (planned): drag to draw the face box; click the image-left eye, then the
  image-right eye; `h` = eye hidden (no point saved); `u` = toggle `uncertain` on the last
  object; `n` = no face visible; `z` = undo; `Enter` = save and next; `b` = back; `q` = quit
  (progress is kept). The image is shown at about 1.25x (fits a 1366x768 screen) with a
  small magnifier; stored coordinates are in the original 640x480 pixels.
- Labelling is done in several short sessions. The tool saves after every image.

### 6.3 Files and storage

- `dataset/labels.csv` (unchanged format):
  `filename, object_type (face|eye), x, y, w, h, side, uncertain`
  (`w, h` are empty for eyes; `side` is empty for faces). Integer pixel coordinates in the
  original image; `(x, y)` is the top-left corner of a face box or the centre of an eye.
  One row per object.
- `dataset/annotation_status.csv`: one row per image with
  `filename, status (labelled|no_face), n_face, n_eye, annotated_at, pass, tool_version`.
  This is what tells "reviewed and nothing to label" (`no_face`) apart from "not yet
  labelled" (no row). `annotated_at` is an ISO date-time with the UTC offset.
- `dataset/labels_pass2.csv` and `dataset/annotation_status_pass2.csv` hold the repeat
  check (section 6.9).
- **Safe saving.** Each save rewrites the file through a temporary file that is flushed,
  fsync'd and then moved over the real file. A timestamped copy of the label files is
  stored in `dataset/backups/` at the start of every session. Re-editing an image replaces
  only that image's rows.

### 6.4 What to label

- **Face:** one axis-aligned bounding box `(x, y, w, h)` per visible face, from the top of
  the forehead (hairline excluded as far as practical; the tall hair mass above the
  forehead is not included) to the chin, and from the left to the right edge of the face,
  including the ears only if clearly visible. For a rolled head draw the smallest upright
  box that contains the face. If the face reaches the image edge, the box is clipped at
  the edge. A face is labelled if a person could identify it as a face, that is, roughly
  half or more of the facial features (eyes, nose, mouth) are visible. This includes turned
  faces up to the 60 degree condition if that test holds.
- **Eyes:** one centre point `(x, y)` for each eye that is **visibly identifiable** in the
  image, placed at the centre of the visible eye (the middle of the pupil/iris area, or the
  centre of the eye opening if the eyelid covers it). Each point is recorded as `left` or
  `right` **relative to the image** (`left` = the eye on the left side of the image).
- **Occlusion and visibility:** an eye that is hidden (behind a hand, hair, a strong glare,
  a turned head, very dark shadow) is **not labelled as visible**. It gets no point.
  Closed but visible eyelids are labelled. Eyes behind ordinary glasses are labelled if the
  eye is visible through the lenses; if glare hides the eye, it is not labelled.
- **Background-only images:** no face label, no eye label. These images get status
  `no_face` and zero objects. In the dataset no person is expected in them; if a person is
  unexpectedly visible, this is recorded and reported, not silently relabelled.
- **Uncertain cases:** mark `uncertain=1` when a decision is genuinely ambiguous: when the
  point could not be placed confidently within roughly one third of the eye width, or when
  it is unclear whether a face or an eye is present. It is **not** a way to skip labelling,
  and it is not to be used out of habit. Lowered or nearly closed eyelids are expected in
  the `down` tilt images and in `tilt_P01_007`: label the visible slit, and mark
  `uncertain` only where the position is truly unclear. Uncertain objects are reported
  separately: results are given with and without them so the effect is visible.

### 6.5 Validation (`scripts/validate_labels.py`)

**Errors** (labelling cannot be accepted until fixed):
- every image of the dataset has a status row, and every label row belongs to a dataset
  image;
- the headers are exactly as in section 6.3 and the values are well-formed
  (`object_type` is `face` or `eye`, integers where required, `uncertain` is 0 or 1);
- at most one face per image; a face box has positive size and lies inside the 640x480
  frame;
- at most two eyes per image, at most one `left` and one `right`, and `left` has the smaller
  x; an eye point lies inside the frame and inside its face box (with a small margin);
- eyes exist only together with a face;
- `no_face` images have zero objects, and background-only images (`NONE`) have no face
  labels;
- `n_face` and `n_eye` in the status file agree with `labels.csv`;
- no duplicate rows.

**Warnings** (reviewed by the author): face-box sizes that are unusual for the condition;
person images with a face but zero or one eye; person images with no face; the list of
uncertain objects. An overlay sheet of the labels is also reviewed by eye (outside the
project folder). Neither check uses detector output.

### 6.6 Dataset freeze before labelling

The SHA-256 manifest described in section 2 is written before the first label is saved.
Images, `metadata.csv` and this protocol's dataset description stay fixed during labelling.

### 6.7 Label freeze before evaluation

Pass-1 labels are completed and validated, the repeat check (6.9) is done, and only then
is the detector evaluated. After the first evaluation run starts, labels are frozen.

### 6.8 Annotator

The author labels all images (single annotator). This is a limitation (section 10).

### 6.9 Repeat check (intra-annotator) and change log

- **Sample.** 14 images chosen at random, with a fixed seed (**repeat-sample seed:
  20261121**; the 14 person images with the smallest SHA-256 of `<seed>:<filename>`), from
  the 105 person images. 14 images is approximately 10% of the full 140-image plan; it is
  about 13% of the 105 person-containing images and 11% of the 125-image dataset.
  Background-only images have no labels to compare, so they are not sampled.
- **Timing.** Pass 2 starts **at least 24 hours** after pass 1 is finished, in a fresh random
  order, blind and without access to the pass-1 labels. The author performs both passes.
- **Comparison.** Per image: face IoU between passes; eye-centre distance as a percentage
  of the face width; whether the presence of a face, of each eye and of the `uncertain`
  flag agree. Reported as median and worst values and the raw list of disagreements. This
  estimates the **consistency of one person's labels**, not their correctness, and it says
  nothing about agreement between different annotators.
- **Use.** Pass 1 remains the ground truth. Pass 2 is not merged into it.
- **Corrections.** Pass-1 labels may be corrected only before any detector evaluation has
  been run, and only for clear mistakes (for example a swapped left/right eye). Every
  correction is written in a change log (filename, what changed, why, date, who) kept with
  the results. After the first evaluation run, labels are frozen.

## 7. Evaluation matching

Evaluation runs the frozen detector on each saved image and compares the result with the
labels. The detector is run exactly as `main.py` runs it: face detection on the frame, then
eye detection inside each detected face with duplicate suppression.

**Evaluation sets.** Every result is computed on (a) the **full dataset** (all 125 images)
and (b) the **de-duplicated subset** defined in section 11, with identical matching rules.
Both are always reported (section 8).

**Faces**
- A detected face box matches a labelled face box if their **IoU >= 0.50**, where
  IoU = (area of overlap) / (area of union).
- Matching is **one-to-one**. Candidate pairs are sorted by IoU from highest to lowest and
  accepted greedily; once a detection or a label is matched, it cannot be used again.
- Matched pairs are true positives (TP). Unmatched detections are false positives (FP).
  Unmatched labelled faces are false negatives (FN).

**Eyes**

Eye matching always happens **after** face matching and **only inside matched faces**. The
steps below must be followed in this order.

*Step 1 - face matching.* Done first, exactly as described above. The result is a set of
**matched face pairs** (detected face D, labelled face L), plus unmatched detected faces
(false-positive faces) and unmatched labelled faces (missed faces).

*Step 2 - keep each detected eye tied to its face.* The detector finds eyes one face at a
time (`EyeDetector.detect(frame, face)`). Every detected eye is therefore **tagged with the
detected face that produced it**, and this tag is used for matching. An eye is never
re-assigned to a different face because of where it happens to lie. Each labelled eye
belongs to the labelled face of its own image.

*Step 3 - eye matching inside each matched face pair.* For each matched pair (D, L):
- The candidate detected eyes are only those tagged with D. The candidate labelled eyes are
  only those belonging to L.
- A detected eye is represented by the **centre point** of its box. It can match a labelled
  eye point if the Euclidean distance between them is **<= 15% of the labelled face width
  (L's width)** in pixels. For example, for a 200 px wide labelled face the limit is 30 px.
- Matching is **one-to-one** within the pair: candidate pairs within the threshold are
  sorted by distance from smallest to largest and accepted greedily; once a detected eye or
  a labelled eye is matched, it cannot be used again.
- A matched eye pair is a TP. A detected eye of D with no match is a FP. A labelled eye of L
  with no match is a FN. (A face with no labelled eye, for example both eyes hidden,
  therefore contributes only FPs for any eyes detected.)

*Step 4 - eyes that do not belong to a matched face pair.* No distance test is applied to
these eyes. The result is fixed:
- Every detected eye tagged with an **unmatched (false-positive) detected face** is a **FP**,
  even if it lies close to a labelled eye. In background-only images, which have no labels,
  every detected face and eye is a FP by definition; these are counted only in the separate
  background false-positive figures (section 8), not in the person-image metrics.
- Every **visible labelled eye** that belongs to a **missed labelled face** (a labelled face
  with no matched detection) is a **FN**, even if some detection happens to lie near it.

*Primary (end-to-end) eye result.* Counts from step 3 **plus** step 4, over all images.
This reflects what a user of the system actually sees: a missed or wrong face also costs
eye detections.

*Secondary (conditional) eye result.* Counts from **step 3 only**, that is, only eyes inside
matched face pairs. Step 4 eyes are left out. This shows eye-stage errors separately from
face-stage errors.

*Both-eyes rate.* An image counts as "both eyes detected" only if its face is matched
(step 1) and both of its labelled eyes are matched in step 3. In the primary result the
denominator is all images with two labelled visible eyes (an image whose face was missed
counts as not detected). In the secondary result the denominator is only images with two
labelled visible eyes **and** a matched face.

## 8. Metrics and reporting

For faces, and separately for eyes (primary and secondary), counted over a set of
**person images** (images that contain a person; background-only images are not part of the
person-image metrics: 105 images in the full dataset, 68 in the de-duplicated subset):

- **TP, FP, FN** as defined above, reported as raw counts.
- **Precision** = TP / (TP + FP). Undefined when TP + FP = 0; reported as "n/a".
- **Recall** = TP / (TP + FN). Undefined when TP + FN = 0 (for example background-only);
  reported as "n/a".
- **Both-eyes rate** = (images in which both labelled eyes were matched) / (images with two
  labelled visible eyes). Images with only one visible eye are excluded from this rate and
  counted separately ("one eye visible: n images, eye found in k of them").
- **False positives per image** for background-only images:
  (detected faces + detected eyes) / number of background-only images (20 in the full
  dataset, 5 in the de-duplicated subset), reported for faces and eyes separately.

**Breakdown.** Every metric is reported **overall and per condition** (and per sub-condition
where the count allows) in tables that always show the counts (TP, FP, FN and number of
images) next to the percentages. There is **no per-participant breakdown**, because there is
only one participant. Each condition has only 5-20 images, so percentages are descriptive
for this dataset only. Uncertainty is shown by listing the raw counts; optionally a 95%
Wilson score interval is added for the overall results, with the explicit note that the
images are not independent samples of the population. Results are given with and without
`uncertain` objects (section 6.4). Sub-condition results for `tilt` state the recorded split
(7 `roll`, 8 `down`) and the visual uncertainty about `tilt_P01_007`; `turn` results state
that the angles are approximate; `dist` results are described as nearer/farther.

**Full and de-duplicated reporting.** The two evaluation sets are reported **side by side in
the same tables**: the full 125-image results are always shown, never only the subset. Each
table states the evaluation set, the de-duplication threshold T, the number of images per
condition in each set, and the date and commit of the run. The subset is defined (section 11)
before any detector result exists and is never changed afterwards to improve a number. Because
the subset has only a handful of images per condition, its results are particularly uncertain
and are described as such.

**Error review.** All FP and FN cases are listed with filename, condition and a short note
written after looking at the image (for example "box on eyebrow", "face turned, no
detection"). No metric is used to claim a cause; observations are labelled as observations.

## 9. Performance evaluation

**Offline benchmark** on the 125 saved evaluation images (full dataset; timings are not
repeated for the subset), with `time.perf_counter()`.

- All images are loaded into memory first, so disk reading is not timed.
- **Stages timed separately** for each image: (a) face detection (grayscale conversion plus
  `detectMultiScale`), (b) eye detection for all detected faces (ROI crop, grayscale, cascade
  call, duplicate suppression), (c) drawing the boxes. Total = a + b + c.
  Camera capture and `cv2.imshow` need a live camera and window, so they are not part of the
  offline benchmark.
- **Warm-up:** one full pass over all images (and at least 20 detection calls) is run and
  discarded before timing starts.
- **Repeats:** 5 timed passes over the full image set, in the same order each time.
- **Reported:** mean and standard deviation of the per-image times (ms), plus median and
  95th percentile, overall and per condition. Mean and standard deviation across the 5
  pass averages are also given as an estimate of run-to-run variation. Offline FPS is
  `1000 / mean total ms` and is labelled as an offline figure.
- **Conditions of measurement:** laptop plugged in, power plan unchanged for all runs, no
  other heavy applications running, same Python and library versions as section 2. The
  settings are written down next to the results.
- **Live FPS** (separate, optional): run `python main.py` for a fixed duration (at least
  60 seconds), 3 times, in the same conditions, and note the "Average measured FPS" it
  prints. This includes camera reading and display. The previously recorded manual figure
  of 17.5 FPS is a single development-stage run and is not part of this evaluation unless
  repeated under this protocol.
- Timings depend on image content (for example no face means no eye stage), which is why
  they are reported per condition.

## 10. Limitations

- **Small dataset:** 125 images, 15-20 per condition (5-15 per condition in the
  de-duplicated subset). Differences between conditions can be due to chance; results are
  descriptive only.
- **Only one participant** (the project author). Variation in skin tone, face shape, age,
  hair and glasses is essentially absent, and glasses were not evaluated. No claim is made
  about other people.
- **One webcam, one room, one laptop.** Results do not transfer to other cameras, sensor
  noise levels, rooms or lighting.
- **Conditions confounded with capture session and exposure.** The conditions were
  recorded in four sessions at different times of day with automatic exposure, so a
  difference between conditions cannot be attributed to the condition alone (section 5).
- **Lighting is not controlled or measured.** `dim` is described as dim indoor lighting and
  `strong` as uneven/harsh lighting; no light levels were measured, and `strong` mixes more
  than one set-up.
- **Near-duplicates.** Many consecutive images are nearly identical, so the effective
  sample size is smaller than the image count. The de-duplicated subset (section 11)
  reduces but does not remove this, and it is itself an approximate, pixel-based rule.
- **Limited background diversity:** the 20 background-only images are 3 near-black frames
  and 17 near-identical frames of one plain wall view, which is a weak test of false
  positives.
- **Approximate conditions:** head angles and distances are estimates, not measurements,
  and `far` faces are still large; the sets do not test very small faces.
- **Metadata caveats:** the tilt sub-condition labels were edited by hand after capture,
  and `tilt_P01_007` (recorded `roll`) appears visually like the `down` set.
- **Hand-labelled ground truth by one person.** Labels can be wrong or inconsistent. The
  repeat check measures only the consistency of the same annotator after at least 24
  hours, not correctness and not agreement between annotators. Face box conventions and
  eye-centre positions are partly subjective, and nearly closed eyes make eye positions
  harder to place.
- **Posed images**, captured by key press, are not a recording of natural use. Real-time
  video behaviour (flicker between frames) is not measured by still images.
- **Haar Cascade limitations:** these classifiers are older, hand-engineered detectors
  trained mostly on frontal faces; they are sensitive to pose, lighting and occlusion. The
  evaluation measures this specific frozen configuration, not Haar cascades in general.
- **Matching thresholds** (IoU 0.50, 15% of face width) and the de-duplication rule
  (Method B, T = 2.5, section 11) are choices that affect the numbers; stricter or looser
  values would give different results.
- **Duplicate suppression v2 was tuned on the earlier 90 development frames** of one
  person; those frames are not used here, which keeps the evaluation separate from the
  tuning, but the rule still assumes roughly frontal faces.
- **No claims of accuracy beyond this dataset.** Any sentence in the report that quotes a
  precision or recall figure must state the dataset size (and which evaluation set), the
  number of participants and the conditions.

## 11. De-duplicated subset (approved final rule)

The de-duplicated subset reduces the influence of near-identical consecutive images. It is
chosen from **pixel similarity only**: no detector output and no labels are used.

**Status: approved and frozen (Milestone 6).** The rule is **Method B** (sub-condition-aware
raw pixel comparison) with **T = 2.5 gray levels**. It was fixed before any ground-truth
label or detector result exists and is never changed afterwards. Any change would need a new
protocol revision and would invalidate every result computed with the subset.

### 11.1 Algorithm (deterministic)

1. For every image: convert to grayscale (`cv2.cvtColor`, BGR to GRAY), resize to 64 x 48
   with `cv2.INTER_AREA` (area averaging), and keep it as `float32` raw gray levels
   (no brightness or contrast normalisation).
2. Group the images by their **recorded condition and recorded sub-condition** (for example
   `turn/30deg`, `tilt/roll`, `tilt/down`, `dist/near`). Groups are never mixed.
3. Within a group take the images in **capture order** (capture time, which equals the
   filename-number order; the tool stops if the two ever disagree).
4. Start the first cluster with the first image. For each following image compute the mean
   absolute difference **MAD** between its small gray image and the small gray image of the
   **first image of the current cluster**. If MAD <= T (2.5) the image joins the current
   cluster; if MAD > T it starts a new cluster.
5. The **earliest** image of each cluster is its **representative**. The subset is the set of
   representatives. Excluded images are never deleted or altered; they remain part of the
   full dataset. Cluster IDs are `<condition>-<sub_condition>-Cnn` (for example
   `tilt-down-C02`), numbered from 01 within each group.

### 11.2 Outputs and reproducibility

Generated by `scripts/build_dedup_manifest.py` (reads images and `metadata.csv`, never writes
to `dataset/images/`), into `dataset/deduplication/` (Git-ignored with the rest of `dataset/`):

- `manifest.csv`: one row per source image: `filename, condition, sub_condition, image_type`
  (`person`/`background`), `participant_id, capture_time, cluster_id, cluster_size,
  representative_filename, status` (`representative`/`excluded`), `mad_to_representative`
  (4 decimals), `in_dedup_subset`, four evaluation-set flags (`person_metrics_full`,
  `person_metrics_dedup`, `background_fp_full`, `background_fp_dedup`), and `source_sha256`.
- `summary.csv`: counts overall, for person images, for background images, per condition
  and per condition/sub-condition.
- `methodology.json`: the rule, parameters, expected counts, the visual-audit totals, the
  SHA-256 of `metadata.csv` and of all images (aggregate), the script's SHA-256 and the
  software versions. It contains no timestamps, so a re-run gives identical bytes.

Verification (`python scripts/build_dedup_manifest.py --verify`) re-derives everything and
checks that the files match byte for byte, and that: all 125 source images appear exactly
once; there are exactly 73 representatives (68 person images and 5 background images); no
cluster mixes conditions or recorded sub-conditions; each representative is the earliest
image of its cluster; every excluded image has a valid representative and a MAD <= T; and
none of the seven audit-flagged merges (section 11.4) exists. Recorded provenance at the
time of the decision: `metadata.csv` SHA-256 `a58b4144b2afbccd32bff58f9d28f2bb73f378458373588094026897692af544`,
aggregate image SHA-256 `dd51a1c9f221e5b0cff9b6a0c3a530e4778c58efd960f884bd52b71ba39c36c4`.

### 11.3 Result

| Group | Images | Representatives | Excluded |
|---|---|---|---|
| normal / frontal | 20 | 15 | 5 |
| dim / dim | 15 | 9 | 6 |
| strong / side | 15 | 11 | 4 |
| turn / 30deg | 10 | 5 | 5 |
| turn / 60deg | 10 | 6 | 4 |
| tilt / roll | 7 | 5 | 2 |
| tilt / down | 8 | 2 | 6 |
| dist / near | 10 | 6 | 4 |
| dist / far | 10 | 9 | 1 |
| bg / background | 20 | 5 | 15 |
| **Total** | **125** | **73** | **52** |

Person images: 105 images, **68** representatives (37 excluded). Background images: 20
images, **5** representatives (15 excluded). Largest cluster: 8 images (background).
Maximum MAD of an excluded image: 2.4925. Several groups have very few representatives
(`tilt/down` 2, `turn/30deg` 5), which makes sub-condition results of the subset very
uncertain. The five background representatives show only two different scenes (the dark
frames and one wall view), so they add little diversity.

### 11.4 Evidence for the choice

- **Visual audit of Method B.** At T = 2.0, 2.5, 3.0 and 4.0 there are 63 distinct
  multi-image clusters. All 63 were visually inspected from contact sheets (native-resolution
  face crops for `normal`), covering 104 distinct images: **56 safe, 6 questionable, 1
  unsafe, 0 unreviewed**. The judgements were made by the assistant by looking at the
  images, were reported to the author, and are not an independent expert annotation.
- **All seven questionable or unsafe clusters form only at T >= 3.0:** `turn_P01_005` merged
  with 002-004 (unsafe, T = 4.0); `turn_P01_010` with 006-009 (T = 3.0); `turn_P01_012`-`015`
  (T = 3.0); `turn_P01_016`-`019` (T = 3.0); `strong_P01_013` with 011-015 (T = 4.0);
  `dim_P01_008` with 005-007 (T = 4.0); `dim_P01_013`-`015` (T = 4.0). None exists at
  T = 2.5, and the script checks this.
- **Why 3.0 and 4.0 were not selected.** T = 3.0 would merge three clusters judged
  questionable (head position differences within `turn`). T = 4.0 would add the one
  unsafe merge and the remaining questionable ones (`turn`, `strong`, `dim`). T = 2.5 is the
  largest tested threshold at which the audit found no questionable or unsafe merge. T = 2.0
  also had none but retains more near-duplicates (82 representatives instead of 73).
- **Why not the other methods.** The original method (A: raw difference within each
  condition only) merges `tilt_P01_007` (recorded `roll`) with the `down` images. The
  brightness-normalised method (C) makes the threshold depend on image contrast, amplifies
  the noise of the near-black frames, and merges all wall frames into one cluster at the
  thresholds tried.

### 11.5 Known limitations

- **Pixel similarity is not pose similarity.** Whole-image differences are dominated by
  the background and by small shifts and exposure drift. Visually similar images can be
  split (for example several `normal` and `dist` images), and a few different poses could be
  merged.
- **Anchored to the first image.** Slow drift within a cluster can split a run of similar
  images, and a chain of images each close to the anchor need not be close to each other.
- **Only an approximate de-duplication.** The subset is still not a set of independent
  samples; images of one participant, one webcam and one room remain correlated.
- **Visual review limits.** It was done at thumbnail scale; eyelid openness could not be
  assessed in the dark `dim` images or reliably in the `tilt/down` images, and gaze
  differences were not assessable. Judgements are subjective and come from one reviewer
  (the assistant); borderline images were judged as seen, not measured. Thresholds other
  than 2.0, 2.5, 3.0 and 4.0 were not visually audited.
- **Threshold-specific.** T = 2.5 applies to this camera, scene and 64 x 48 representation.
  A different T gives a different subset.
- The `tilt/roll` representative `tilt_P01_007` looks visually like the `down` images; its
  metadata is unchanged by decision.

### 11.6 Full dataset versus de-duplicated evaluation subset

| | Full dataset | De-duplicated subset |
|---|---|---|
| Images | all 125 | the 73 representatives |
| Person images (person-image metrics) | 105 | 68 |
| Background-only images (false-positive figures only) | 20 | 5 |

Every result is reported for both sets with identical rules (sections 7 and 8). Person-image
metrics (precision, recall, both-eyes rate) are computed on person images only.
Background-only images are never part of the person-image metrics; they are used only for
the separate false-positive-per-image figures. The manifest flags give exactly these sets.
The full dataset is always reported; the subset is a robustness view of it.

## 12. Revision history

- **Version 1** (before data collection): evaluation protocol for a 140-image dataset with
  eight conditions and two participants.
- **Version 2** (Milestone 6 update, 2026-11-19): dataset as collected (125 images, one
  participant, glasses excluded) with the final dataset decisions and the descriptions of
  `strong`, `turn`, `tilt`, `dist` and `bg`; single-participant, session-confound, lighting,
  near-duplicate and background limitations (sections 3, 5 and 10); the blind randomised
  annotation workflow, `annotation_status.csv`, `dataset/backups/`, validation rules and
  the rule that the annotation and validation tools never run the detector (sections 2
  and 6); the intra-annotator repeat check with a delay of at least 24 hours (section 6.9);
  full-dataset and de-duplicated-subset reporting with a provisional threshold (sections 7,
  8 and 11); removal of the per-participant breakdown. The detector baseline, the matching
  rules (sections 7 and 2) and the performance method (section 9) are unchanged.
  No dataset image or metadata file was modified by this revision. The text of version 1
  is kept in a pre-edit copy outside the project.
- **Version 3** (Milestone 6, 2026-11-20): the de-duplication decision is frozen. Method B
  (sub-condition-aware raw pixel comparison) with T = 2.5 is recorded as the approved final
  rule, with the algorithm, representative selection, outputs
  (`dataset/deduplication/manifest.csv`, `summary.csv`, `methodology.json`), the result
  (73 representatives: 68 person images and 5 background images), the visual-audit totals
  (63 clusters reviewed: 56 safe, 6 questionable, 1 unsafe), the reasons for not selecting
  T = 3.0 or 4.0, the known limitations and the full-versus-subset reporting rule
  (section 11). Background-only images are excluded from the person-image metrics and used
  only for the separate false-positive figures (sections 7 and 8). The "provisional"
  wording of version 2 is removed; all earlier decisions and the revision history are kept.
  No dataset image or metadata file was modified.
