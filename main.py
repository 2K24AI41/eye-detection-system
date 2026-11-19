"""Entry point: show the live webcam feed. Press Q to quit.

Run:  python main.py
"""
import sys
import time

import cv2

from src.camera import Camera, CameraError
from src.detector import (DetectorError, EyeDetector, FaceDetector,
                          draw_eyes, draw_faces)

WINDOW_NAME = "Eye Detection System - press Q to quit"


def run() -> int:
    camera = Camera(index=0)
    try:
        detector = FaceDetector()  # fails early if a cascade file is bad
        eye_detector = EyeDetector()
        camera.open()
        props = camera.properties()
        print(f"Camera opened: {props['width']}x{props['height']} pixels, "
              f"driver-reported FPS: {props['fps']}")

        frames = 0
        start = time.time()
        while True:
            frame = camera.read()
            frames += 1

            faces = detector.detect(frame)
            draw_faces(frame, faces)

            # Eyes are searched only inside each detected face.
            eyes = []
            for face in faces:
                eyes.extend(eye_detector.detect(frame, face))
            draw_eyes(frame, eyes)

            # Measured FPS = frames shown / seconds elapsed (the real speed).
            # It now includes the time spent on face detection.
            fps = frames / (time.time() - start)
            cv2.putText(frame, f"FPS: {fps:.1f}  Faces: {len(faces)}  Eyes: {len(eyes)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, frame)

            # waitKey(1) lets the window refresh and waits up to 1 ms for a key.
            # It returns the key code, or -1 if none. "& 0xFF" keeps the last
            # 8 bits so the comparison works on all systems.
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
            # Also quit if the user closes the window with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        print(f"Stopped. Average measured FPS: {fps:.1f}")
        return 0
    except (CameraError, DetectorError) as error:
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 0
    finally:
        # 'finally' runs on success, error, or Ctrl+C, so the camera is
        # ALWAYS released and the window closed.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(run())
