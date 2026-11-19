"""Webcam access (Milestone 2).

This module hides the OpenCV webcam details behind a small Camera class,
so the rest of the project only needs: open, read a frame, release.
"""
import sys

import cv2


class CameraError(Exception):
    """Raised when the webcam cannot be opened or read."""


class Camera:
    def __init__(self, index: int = 0):
        # index 0 = the default (built-in) webcam, 1 = a second camera, ...
        self.index = index
        self.capture = None

    def open(self) -> None:
        """Open the webcam, or raise CameraError with a helpful message."""
        # cv2.VideoCapture(index) asks the operating system for the camera.
        # On Windows the default backend (MSMF) can be slow to start, so we
        # try DirectShow (CAP_DSHOW) first and fall back to the default.
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for backend in backends:
            capture = cv2.VideoCapture(self.index, backend)
            # isOpened() is the ONLY reliable way to know it worked:
            # VideoCapture() itself never raises an error on failure.
            if capture.isOpened():
                self.capture = capture
                return
            capture.release()
        raise CameraError(
            f"Could not open webcam {self.index}. Check that it is connected, "
            "not used by another app (Zoom/Teams), and that Windows camera "
            "privacy settings allow desktop apps."
        )

    def read(self):
        """Grab one frame. Returns a BGR NumPy array of shape (height, width, 3)."""
        # capture.read() returns two values:
        #   success -> True/False, did we get a picture?
        #   frame   -> the image (a NumPy array) if success, otherwise None
        success, frame = self.capture.read()
        if not success:
            raise CameraError("Failed to read a frame (camera disconnected or busy).")
        return frame

    def properties(self) -> dict:
        """Return width, height (pixels) and the FPS the driver reports."""
        return {
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            # Some webcams report 0 here; it is the driver's claim, not a measurement.
            "fps": self.capture.get(cv2.CAP_PROP_FPS),
        }

    def release(self) -> None:
        """Give the camera back to the OS. Safe to call more than once."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
