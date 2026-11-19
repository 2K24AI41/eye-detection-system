"""Face detection with a pre-trained Haar cascade (Milestone 3).

The XML model file comes with OpenCV. We did NOT train it.
"""
import os

import cv2

FACE_CASCADE_FILE = "haarcascade_frontalface_default.xml"
EYE_CASCADE_FILE = "haarcascade_eye.xml"


class DetectorError(Exception):
    """Raised when a cascade file is missing or cannot be loaded."""


def load_cascade(path: str) -> cv2.CascadeClassifier:
    """Load a Haar cascade XML file, raising DetectorError if it is bad."""
    # CascadeClassifier(path) does NOT raise on a bad path; it silently
    # creates an empty classifier. So we check the file and empty() ourselves.
    if not os.path.isfile(path):
        raise DetectorError(f"Cascade file not found: {path}")
    try:
        cascade = cv2.CascadeClassifier(path)
    except (cv2.error, SystemError) as error:  # malformed XML (type varies by build)
        raise DetectorError(f"Cascade file could not be parsed: {path}") from error
    if cascade.empty():
        raise DetectorError(f"Cascade file could not be loaded (invalid XML?): {path}")
    return cascade


class FaceDetector:
    def __init__(self, cascade_path: str = None,
                 scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: tuple = (60, 60)):
        if cascade_path is None:
            cascade_path = os.path.join(cv2.data.haarcascades, FACE_CASCADE_FILE)
        self.cascade = load_cascade(cascade_path)
        # Tuning parameters, explained in detect():
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame):
        """Find faces in a BGR frame. Returns a list of (x, y, w, h) boxes."""
        # Haar features measure brightness differences, so colour is not
        # needed. Grayscale has 1 channel instead of 3 -> faster.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detectMultiScale slides a window over the image, at several sizes,
        # and asks the cascade "is this window a face?".
        faces = self.cascade.detectMultiScale(
            gray,
            # scaleFactor: the image is shrunk by this much (10%) each pass so
            # faces of different sizes are found. Smaller = more thorough but slower.
            scaleFactor=self.scale_factor,
            # minNeighbors: a real face triggers many overlapping boxes. We keep
            # a box only if at least this many agree. Higher = fewer false
            # alarms, but may miss real faces.
            minNeighbors=self.min_neighbors,
            # minSize: ignore windows smaller than this (in pixels). Saves time
            # and removes tiny false detections.
            minSize=self.min_size,
        )
        # Returns an empty tuple when nothing is found, a NumPy array otherwise.
        return [tuple(int(v) for v in box) for box in faces]


# Rule values for suppress_duplicate_eyes (fractions of the face size).
EYE_MIN_CY, EYE_MAX_CY = 0.12, 0.50      # eye centre must be 12%-50% down the face
EYE_MAX_WIDTH = 0.40                     # an eye box wider than 40% of the face is rejected
EYE_X_ZONES = ((0.10, 0.45), (0.55, 0.90))  # allowed centre x ranges (left half, right half)
EYE_TYPICAL_X = (0.30, 0.70)             # typical eye centre x (left, right)
EYE_TYPICAL_Y = 0.33                     # typical eye centre y
EYE_OVERLAP_THRESHOLD = 0.50             # same-eye boxes overlap > 50% of the smaller box


def _overlap_ratio(a, b) -> float:
    """Intersection area divided by the area of the smaller box (0..1)."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return (ix * iy) / min(a[2] * a[3], b[2] * b[3])


def suppress_duplicate_eyes(boxes, face_w: int, face_h: int):
    """Keep at most one eye box per face half, preferring the tighter box.

    `boxes` are (x, y, w, h) relative to the face's top-left corner.
    `face_w`, `face_h` are the FACE size (not the search-region size).
    """
    kept = []
    for half, zone in enumerate(EYE_X_ZONES):
        # 1. Keep only plausible boxes that belong to this half of the face.
        candidates = []
        for (x, y, w, h) in boxes:
            cx = (x + w / 2) / face_w
            cy = (y + h / 2) / face_h
            if (EYE_MIN_CY < cy < EYE_MAX_CY and w < EYE_MAX_WIDTH * face_w
                    and zone[0] < cx < zone[1]):
                candidates.append((x, y, w, h))

        # 2. Group boxes that overlap a lot: they are the same eye.
        groups = []
        for box in candidates:
            for group in groups:
                if any(_overlap_ratio(box, other) > EYE_OVERLAP_THRESHOLD for other in group):
                    group.append(box)
                    break
            else:
                groups.append([box])

        # 3. From each group keep the smallest (tightest) box.
        representatives = [min(g, key=lambda b: b[2] * b[3]) for g in groups]

        # 4. If separate groups remain, keep the one closest to a typical eye spot.
        if representatives:
            def distance(b, tx=EYE_TYPICAL_X[half]):
                return (abs((b[0] + b[2] / 2) / face_w - tx)
                        + abs((b[1] + b[3] / 2) / face_h - EYE_TYPICAL_Y))
            kept.append(min(representatives, key=distance))
    return kept


class EyeDetector:
    """Finds eyes inside an already-detected face (pre-trained, not trained by us)."""

    def __init__(self, cascade_path: str = None,
                 scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: tuple = (20, 20), search_fraction: float = 0.6):
        if cascade_path is None:
            cascade_path = os.path.join(cv2.data.haarcascades, EYE_CASCADE_FILE)
        self.cascade = load_cascade(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        # Eyes are in the upper part of a face. Searching only the top 60%
        # skips nostrils and mouth, which the eye cascade often mistakes for eyes.
        self.search_fraction = search_fraction

    def detect(self, frame, face):
        """Find eyes in one face. `face` is (x, y, w, h) in FRAME coordinates.

        Returns eye boxes (x, y, w, h) also in FRAME coordinates.
        """
        x, y, w, h = face
        # ROI = Region Of Interest: a rectangular slice of the frame.
        # NumPy slicing is [rows, columns] = [y range, x range].
        roi_h = int(h * self.search_fraction)
        roi = frame[y:y + roi_h, x:x + w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        eyes = self.cascade.detectMultiScale(
            gray_roi,
            scaleFactor=self.scale_factor,    # same meaning as for faces
            minNeighbors=self.min_neighbors,  # same meaning as for faces
            minSize=self.min_size,            # ignore boxes smaller than 20x20 px
        )
        # The cascade reports positions relative to the ROI's top-left corner
        # (the ROI is its own little image starting at (0, 0)). The ROI starts
        # at the face's top-left corner, so these are also face-relative.
        eyes = [tuple(int(v) for v in box) for box in eyes]
        # Remove duplicate boxes on the same eye (at most one per face half).
        eyes = suppress_duplicate_eyes(eyes, w, h)
        # To draw on the full frame we add the ROI's offset:
        # frame_x = x + ex, frame_y = y + ey.
        return [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]


def draw_eyes(frame, eyes) -> None:
    """Draw a blue rectangle around each eye (modifies frame in place)."""
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


def draw_faces(frame, faces) -> None:
    """Draw a green rectangle around each face (modifies frame in place)."""
    for (x, y, w, h) in faces:
        # (x, y) = top-left corner, (x+w, y+h) = bottom-right. Colour is BGR.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
