"""Background camera worker for the web frontend (Phase 1).

One DetectionService owns ONE worker thread. That thread is the only code that ever
opens, reads or releases the webcam. Flask routes only start/stop the worker and read
the latest finished JPEG, so any number of browser requests can never open the camera
twice. The existing Camera / FaceDetector / EyeDetector / draw helpers are reused
unchanged; this module only composes them.
"""
import threading
import time

import cv2

from src.camera import Camera, CameraError
from src.detector import (DetectorError, EyeDetector, FaceDetector,
                          draw_eyes, draw_faces)

STOPPED, STARTING, RUNNING, STOPPING, ERROR = "stopped", "starting", "running", "stopping", "error"

VIEWER_TIMEOUT_SECONDS = 30.0   # worker stops itself if no browser fetches frames this long
STOP_JOIN_TIMEOUT_SECONDS = 5.0  # how long stop() waits for the worker to finish
MAX_STREAM_FPS = 15              # cap, so detection does not starve the web server
JPEG_QUALITY = 80
STALLED_MESSAGE = "the camera worker has not stopped yet"   # shown while a stop is overdue


class DetectionService:
    def __init__(self, camera_factory=Camera, face_factory=FaceDetector,
                 eye_factory=EyeDetector, viewer_timeout=VIEWER_TIMEOUT_SECONDS,
                 stop_timeout=STOP_JOIN_TIMEOUT_SECONDS, max_fps=MAX_STREAM_FPS,
                 clock=time.monotonic):
        self._camera_factory = camera_factory
        self._face_factory = face_factory
        self._eye_factory = eye_factory
        self._viewer_timeout = viewer_timeout
        self._stop_timeout = stop_timeout
        self._min_interval = 1.0 / max_fps if max_fps else 0.0
        self._clock = clock

        self._lock = threading.RLock()          # guards state, status and the thread handle
        self._frame_cond = threading.Condition()  # guards the latest JPEG
        self._stop_event = threading.Event()
        self._thread = None
        self._state = STOPPED
        self._error = ""
        self._stalled = False          # a stop() timed out and the worker is still alive
        self._generation = 0           # increases with every worker; ties streams to one worker
        self._note = ""
        self._fps = 0.0
        self._faces = 0
        self._eyes = 0
        self._jpeg = None
        self._seq = 0
        self._last_viewer = 0.0

    # ------------------------------------------------------------------ public API
    def start(self):
        """Start the worker unless one is alive. Returns the status dictionary."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.status()          # never a second worker while one is alive
            self._stop_event.clear()
            self._generation += 1
            self._stalled = False
            self._state, self._error, self._note = STARTING, "", ""
            self._fps, self._faces, self._eyes = 0.0, 0, 0
            self._last_viewer = self._clock()
            with self._frame_cond:
                self._jpeg = None
            self._thread = threading.Thread(target=self._run, name="camera-worker", daemon=True)
            self._thread.start()
            return self.status()

    def stop(self):
        """Ask the worker to stop and wait for it. Returns the status dictionary.

        If the worker does not finish in time it is still alive, so the state stays
        'stopping' (never 'stopped') and start() keeps refusing to create another worker.
        """
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._state, self._error, self._note = STOPPED, "", ""
                self._stalled = False
                return self.status()
            self._stop_event.set()
            self._state = STOPPING
        thread.join(self._stop_timeout)       # outside the lock so the worker can finish
        with self._lock:
            if thread.is_alive():
                self._stalled = True               # cleared when the worker really finishes
            return self.status()

    def status(self):
        with self._lock:
            error = self._error or (STALLED_MESSAGE if self._stalled else "")
            return {"state": self._state, "fps": round(self._fps, 1), "faces": self._faces,
                    "eyes": self._eyes, "error": error, "note": self._note}

    def is_active(self):
        """True while a worker exists that is starting, running or stopping."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def generation(self):
        """Identifies the current worker. A stream remembers it and ends when it changes."""
        with self._lock:
            return self._generation

    def mark_viewer(self):
        """Called by the video stream: a browser is still watching."""
        with self._lock:
            self._last_viewer = self._clock()

    def wait_for_frame(self, last_seq, timeout=1.0):
        """Block until a frame newer than last_seq exists. Returns (seq, jpeg) or None."""
        with self._frame_cond:
            self._frame_cond.wait_for(lambda: self._seq != last_seq and self._jpeg is not None,
                                      timeout)
            if self._seq != last_seq and self._jpeg is not None:
                return self._seq, self._jpeg
            return None

    # ------------------------------------------------------------------ worker thread
    def _fail(self, message):
        with self._lock:
            self._error = message

    def _run(self):
        camera = None
        try:
            face_detector = self._face_factory()     # fails early on a bad cascade file
            eye_detector = self._eye_factory()
            camera = self._camera_factory()
            camera.open()
            with self._lock:
                if self._stop_event.is_set():        # checked under the same lock as stop()
                    return                           # so STOPPING is never overwritten
                self._state = RUNNING
            self._loop(camera, face_detector, eye_detector)
        except (CameraError, DetectorError) as error:
            self._fail(str(error))
        except Exception as error:                   # never let the thread die silently
            self._fail(f"unexpected error: {type(error).__name__}: {error}")
        finally:
            if camera is not None:
                try:
                    camera.release()                 # the ONLY release, done once, here
                except Exception as error:
                    self._fail(self._error or f"camera release failed: {error}")
            with self._lock:
                self._stalled = False               # the stop did complete, just late
                self._state = ERROR if self._error else STOPPED
                self._fps, self._faces, self._eyes = 0.0, 0, 0

    def _loop(self, camera, face_detector, eye_detector):
        frames = 0
        start = self._clock()
        next_time = start
        while not self._stop_event.is_set():
            with self._lock:
                idle = self._clock() - self._last_viewer
            if idle > self._viewer_timeout:
                with self._lock:
                    self._note = f"stopped automatically: no viewer for {self._viewer_timeout:g} s"
                return
            frame = camera.read()
            faces = face_detector.detect(frame)
            draw_faces(frame, faces)
            eyes = []
            for face in faces:                       # eyes only inside each detected face
                eyes.extend(eye_detector.detect(frame, face))
            draw_eyes(frame, eyes)
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            frames += 1
            elapsed = self._clock() - start
            with self._lock:
                self._faces, self._eyes = len(faces), len(eyes)
                self._fps = frames / elapsed if elapsed > 0 else 0.0
            with self._frame_cond:
                self._jpeg = buffer.tobytes()
                self._seq += 1
                self._frame_cond.notify_all()
            next_time += self._min_interval          # frame-rate cap; wakes early on stop
            delay = next_time - self._clock()
            if delay > 0:
                self._stop_event.wait(delay)
            else:
                next_time = self._clock()
