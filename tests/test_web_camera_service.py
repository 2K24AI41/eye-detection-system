"""Tests for the web camera service and Flask app. Only FAKE cameras/detectors are used:
no test opens a real webcam or starts the real server."""
import threading
import time
import unittest

import numpy as np

import app as web_app
from src.camera import CameraError
from src.detector import DetectorError
from src.web_camera_service import (ERROR, RUNNING, STOPPED, STOPPING,
                                    DetectionService)


class FakeCamera:
    instances = []

    def __init__(self, fail_open=False, fail_read_after=None, block_read=None):
        self.fail_open = fail_open
        self.fail_read_after = fail_read_after
        self.block_read = block_read          # a threading.Event the read waits for
        self.opened = 0
        self.released = 0
        self.reads = 0
        FakeCamera.instances.append(self)

    def open(self):
        self.opened += 1
        if self.fail_open:
            raise CameraError("Could not open webcam 0 (fake)")

    def read(self):
        self.reads += 1
        if self.block_read is not None:
            self.block_read.wait(10)
        if self.fail_read_after is not None and self.reads > self.fail_read_after:
            raise CameraError("Failed to read a frame (fake)")
        time.sleep(0.002)
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released += 1


class FakeFace:
    def detect(self, frame):
        return [(5, 5, 30, 30)]


class FakeEye:
    def detect(self, frame, face):
        return [(8, 8, 8, 8), (20, 8, 8, 8)]


def make_service(camera=None, **kwargs):
    FakeCamera.instances = []
    factory = (lambda: camera) if camera is not None else FakeCamera
    options = dict(camera_factory=factory, face_factory=FakeFace, eye_factory=FakeEye,
                   viewer_timeout=30, stop_timeout=2, max_fps=0)
    options.update(kwargs)
    return DetectionService(**options)


def wait_until(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class ServiceTests(unittest.TestCase):
    def tearDown(self):
        service = getattr(self, "service", None)
        if service is not None:
            service.stop()

    def test_start_runs_and_publishes_frames_and_counts(self):
        self.service = make_service()
        self.service.start()
        self.assertTrue(wait_until(lambda: self.service.status()["state"] == RUNNING))
        item = self.service.wait_for_frame(None, timeout=2)
        self.assertIsNotNone(item)
        self.assertTrue(item[1].startswith(b"\xff\xd8"))          # a JPEG
        status = self.service.status()
        self.assertEqual((status["faces"], status["eyes"]), (1, 2))
        self.assertEqual(status["error"], "")

    def test_stop_releases_camera_exactly_once(self):
        self.service = make_service()
        self.service.start()
        wait_until(lambda: self.service.status()["state"] == RUNNING)
        self.assertEqual(self.service.stop()["state"], STOPPED)
        camera = FakeCamera.instances[0]
        self.assertEqual((camera.opened, camera.released), (1, 1))
        self.service.stop()                                         # a second stop is harmless
        self.assertEqual(camera.released, 1)

    def test_double_start_creates_one_worker_and_one_camera(self):
        self.service = make_service()
        for _ in range(5):
            self.service.start()
        wait_until(lambda: self.service.status()["state"] == RUNNING)
        self.assertEqual(len(FakeCamera.instances), 1)
        self.assertEqual(sum(t.name == "camera-worker" for t in threading.enumerate()), 1)

    def test_concurrent_starts_open_one_camera(self):
        self.service = make_service()
        threads = [threading.Thread(target=self.service.start) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wait_until(lambda: self.service.status()["state"] == RUNNING)
        self.assertEqual(len(FakeCamera.instances), 1)

    def test_restart_after_stop_opens_a_fresh_camera(self):
        self.service = make_service()
        self.service.start()
        wait_until(lambda: self.service.status()["state"] == RUNNING)
        self.service.stop()
        self.service.start()
        wait_until(lambda: self.service.status()["state"] == RUNNING)
        self.assertEqual(len(FakeCamera.instances), 2)
        self.assertEqual(FakeCamera.instances[0].released, 1)

    def test_camera_open_failure_is_reported_and_released(self):
        camera = FakeCamera(fail_open=True)
        self.service = make_service(camera)
        self.service.start()
        self.assertTrue(wait_until(lambda: self.service.status()["state"] == ERROR))
        self.assertIn("Could not open webcam", self.service.status()["error"])
        self.assertEqual(camera.released, 1)

    def test_cascade_failure_is_reported_and_no_camera_is_opened(self):
        def bad_face():
            raise DetectorError("Cascade file not found: x")
        self.service = make_service(face_factory=bad_face)
        self.service.start()
        self.assertTrue(wait_until(lambda: self.service.status()["state"] == ERROR))
        self.assertIn("Cascade file not found", self.service.status()["error"])
        self.assertEqual(FakeCamera.instances, [])

    def test_read_failure_while_running_is_reported_and_released(self):
        camera = FakeCamera(fail_read_after=3)
        self.service = make_service(camera)
        self.service.start()
        self.assertTrue(wait_until(lambda: self.service.status()["state"] == ERROR))
        self.assertIn("Failed to read", self.service.status()["error"])
        self.assertEqual(camera.released, 1)

    def test_unexpected_exception_is_reported(self):
        class Exploding(FakeFace):
            def detect(self, frame):
                raise ValueError("boom")
        camera = FakeCamera()
        self.service = make_service(camera, face_factory=Exploding)
        self.service.start()
        self.assertTrue(wait_until(lambda: self.service.status()["state"] == ERROR))
        self.assertIn("ValueError", self.service.status()["error"])
        self.assertEqual(camera.released, 1)

    def test_start_after_error_is_allowed_and_clears_the_error(self):
        bad = FakeCamera(fail_open=True)
        self.service = make_service(bad)
        self.service.start()
        wait_until(lambda: self.service.status()["state"] == ERROR)
        wait_until(lambda: not self.service.is_active())
        good = FakeCamera()
        self.service._camera_factory = lambda: good
        self.service.start()
        self.assertTrue(wait_until(lambda: self.service.status()["state"] == RUNNING))
        self.assertEqual(self.service.status()["error"], "")

    def test_stop_during_startup_still_releases_once(self):
        gate = threading.Event()

        class SlowOpen(FakeCamera):
            def open(self):
                gate.wait(5)
                super().open()
        camera = SlowOpen()
        self.service = make_service(camera, stop_timeout=0.05)
        self.service.start()
        self.service.stop()                    # worker still blocked in open()
        gate.set()
        self.assertTrue(wait_until(lambda: not self.service.is_active()))
        self.assertEqual(camera.released, 1)
        self.assertEqual(self.service.status()["state"], STOPPED)

    def test_stop_timeout_keeps_stopping_state_and_blocks_a_second_worker(self):
        gate = threading.Event()
        camera = FakeCamera(block_read=gate)
        self.service = make_service(camera, stop_timeout=0.05)
        self.service.start()
        wait_until(lambda: camera.reads >= 1)
        status = self.service.stop()           # worker is stuck inside read()
        self.assertEqual(status["state"], STOPPING)
        self.assertNotEqual(status["state"], STOPPED)
        self.assertIn("not stopped yet", status["error"])
        self.assertTrue(self.service.is_active())
        self.service.start()                   # must not create another worker
        self.assertEqual(sum(t.name == "camera-worker" for t in threading.enumerate()), 1)
        self.assertEqual(camera.opened, 1)
        self.assertEqual(self.service.status()["state"], STOPPING)
        gate.set()                             # unblock; the worker finishes and releases
        self.assertTrue(wait_until(lambda: not self.service.is_active()))
        self.assertEqual(camera.released, 1)
        self.assertEqual(self.service.status()["state"], STOPPED)

    def test_viewer_inactivity_stops_the_worker_and_releases_camera(self):
        camera = FakeCamera()
        self.service = make_service(camera, viewer_timeout=0.15)
        self.service.start()
        self.assertTrue(wait_until(lambda: not self.service.is_active(), timeout=3))
        self.assertEqual(camera.released, 1)
        status = self.service.status()
        self.assertEqual(status["state"], STOPPED)
        self.assertIn("no viewer", status["note"])

    def test_viewer_activity_keeps_the_worker_alive(self):
        self.service = make_service(viewer_timeout=0.3)
        self.service.start()
        end = time.time() + 0.9
        while time.time() < end:
            self.service.mark_viewer()
            time.sleep(0.05)
        self.assertEqual(self.service.status()["state"], RUNNING)

    def test_frame_rate_cap_limits_the_frames(self):
        camera = FakeCamera()
        self.service = make_service(camera, max_fps=20)
        self.service.start()
        time.sleep(0.5)
        self.service.mark_viewer()
        self.service.stop()
        self.assertLessEqual(camera.reads, 13)                    # ~10 expected at 20 fps

    def test_stop_without_start_is_harmless(self):
        self.service = make_service()
        self.assertEqual(self.service.stop()["state"], STOPPED)
        self.assertEqual(FakeCamera.instances, [])


class AppTests(unittest.TestCase):
    def setUp(self):
        self.service = make_service()
        self.app = web_app.create_app(self.service)
        self.app.config["ENFORCE_HOST"] = False
        self.client = self.app.test_client()

    def tearDown(self):
        self.service.stop()

    def test_index_page_has_heading_and_controls(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("Eye Detection System", html)
        self.assertIn("Start Camera", html)
        self.assertIn("Stop Camera", html)

    def test_static_files_are_served(self):
        for path in ("/static/css/style.css", "/static/js/app.js"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                response.close()

    def test_status_start_stop_routes(self):
        self.assertEqual(self.client.get("/api/status").get_json()["state"], STOPPED)
        self.assertIn(self.client.post("/api/start").get_json()["state"], ("starting", RUNNING))
        wait_until(lambda: self.service.status()["state"] == RUNNING)
        self.assertEqual(self.client.get("/api/status").get_json()["state"], RUNNING)
        self.assertEqual(self.client.post("/api/stop").get_json()["state"], STOPPED)
        self.assertEqual(len(FakeCamera.instances), 1)
        self.assertEqual(FakeCamera.instances[0].released, 1)

    def test_error_is_visible_through_the_status_endpoint(self):
        self.service._camera_factory = lambda: FakeCamera(fail_open=True)
        self.client.post("/api/start")
        wait_until(lambda: self.service.status()["state"] == ERROR)
        body = self.client.get("/api/status").get_json()
        self.assertEqual(body["state"], ERROR)
        self.assertIn("Could not open webcam", body["error"])

    def test_wrong_methods_are_rejected(self):
        self.assertEqual(self.client.get("/api/start").status_code, 405)
        self.assertEqual(self.client.post("/api/status").status_code, 405)

    def test_video_feed_yields_mjpeg_and_never_opens_a_camera_itself(self):
        response = self.client.get("/video_feed")                  # service is stopped
        self.assertEqual(response.mimetype, "multipart/x-mixed-replace")
        self.assertEqual(response.get_data(), b"")                  # ends at once
        self.assertEqual(FakeCamera.instances, [])

    def test_video_feed_streams_frames_while_running(self):
        self.client.post("/api/start")
        wait_until(lambda: self.service.status()["state"] == RUNNING)
        response = self.client.get("/video_feed", buffered=False)
        chunk = next(iter(response.response))
        response.close()
        self.assertTrue(chunk.startswith(b"--frame\r\nContent-Type: image/jpeg"))
        self.assertIn(b"\xff\xd8", chunk)
        self.assertEqual(len(FakeCamera.instances), 1)

    def test_host_header_and_cross_origin_post_are_refused(self):
        self.app.config["ENFORCE_HOST"] = True
        client = self.app.test_client()
        self.assertEqual(client.get("/api/status", base_url="http://evil.example:5000").status_code, 403)
        ok = client.get("/api/status", base_url="http://127.0.0.1:5000")
        self.assertEqual(ok.status_code, 200)
        bad = client.post("/api/start", base_url="http://127.0.0.1:5000",
                          headers={"Origin": "http://evil.example"})
        self.assertEqual(bad.status_code, 403)
        self.assertEqual(FakeCamera.instances, [])
        good = client.post("/api/stop", base_url="http://127.0.0.1:5000",
                           headers={"Origin": "http://127.0.0.1:5000"})
        self.assertEqual(good.status_code, 200)

    def test_server_is_configured_for_localhost_and_no_debug(self):
        self.assertEqual((web_app.HOST, web_app.PORT), ("127.0.0.1", 5000))
        import inspect
        source = inspect.getsource(web_app.main)
        self.assertIn("debug=False", source)
        self.assertIn("use_reloader=False", source)


if __name__ == "__main__":
    unittest.main()
