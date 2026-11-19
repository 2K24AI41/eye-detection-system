"""Milestone 1 check: confirms Python, OpenCV, NumPy and the Haar cascade files are usable.

Run:  python scripts/check_environment.py
"""
import sys

import cv2
import numpy as np

# Cascade files we plan to use later (they ship inside the opencv-python package).
CASCADES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml",
]


def main() -> int:
    print(f"Python : {sys.version.split()[0]}")
    print(f"OpenCV : {cv2.__version__}")
    print(f"NumPy  : {np.__version__}")

    # CascadeClassifier is the OpenCV class that loads and runs a Haar cascade.
    if not hasattr(cv2, "CascadeClassifier"):
        print("FAIL: this OpenCV build has no CascadeClassifier (use opencv-python 4.x).")
        return 1

    ok = True
    for name in CASCADES:
        # cv2.data.haarcascades is the folder containing the bundled XML files.
        classifier = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        loaded = not classifier.empty()  # empty() is True if loading failed
        print(f"{'OK  ' if loaded else 'FAIL'}: {name}")
        ok = ok and loaded
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
