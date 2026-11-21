"""Regression tests for the Phase 1.1 fixes. Fake cameras/detectors only; no webcam, no server."""
import re
import threading
import time
import unittest

import app as web_app
from src.web_camera_service import (RUNNING, STALLED_MESSAGE, STOPPED, STOPPING,
                                    DetectionService)
from tests.test_web_camera_service import (FakeCamera, FakeEye, FakeFace, make_service,
                                           wait_until)

HOST_URL = "http://127.0.0.1:5000"


class StoppingIsNeverOverwrittenTests(unittest.TestCase):
    def test_stop_arriving_during_startup_never_shows_running(self):
        gate = threading.Event()

        class SlowOpen(FakeCamera):
            def open(self):
                gate.wait(5)
                super().open()
        camera = SlowOpen()
        service = make_service(camera, stop_timeout=0.05)
        seen = set()
        loops = []
        original_loop = service._loop
        service._loop = lambda *args: (loops.append(1), original_loop(*args))
        service.start()
        service.stop()                                   # times out: worker is inside open()
        seen.add(service.status()["state"])
        gate.set()                                       # open() returns; worker must not go RUNNING
        while service.is_active():
            seen.add(service.status()["state"])
            time.sleep(0.001)
        seen.add(service.status()["state"])
        self.assertNotIn(RUNNING, seen)
        self.assertEqual(loops, [])                      # the RUNNING loop was never entered
        self.assertEqual(service.status()["state"], STOPPED)
        self.assertEqual(camera.released, 1)
        self.assertEqual(camera.reads, 0)                # it never even read a frame


class StalledStopFlagTests(unittest.TestCase):
    def test_stalled_stop_uses_a_flag_not_the_error_text(self):
        gate = threading.Event()
        camera = FakeCamera(block_read=gate)
        service = make_service(camera, stop_timeout=0.05)
        service.start()
        wait_until(lambda: camera.reads >= 1)
        status = service.stop()
        self.assertEqual((status["state"], status["error"]), (STOPPING, STALLED_MESSAGE))
        self.assertTrue(service._stalled)
        self.assertEqual(service._error, "")             # the real error field stays empty
        gate.set()
        self.assertTrue(wait_until(lambda: not service.is_active()))
        status = service.status()
        self.assertEqual((status["state"], status["error"]), (STOPPED, ""))
        self.assertFalse(service._stalled)

    def test_a_real_error_text_equal_to_the_old_sentinel_is_still_an_error(self):
        # With the old string-compare design this error would have been silently erased.
        class Fails(FakeCamera):
            def open(self):
                from src.camera import CameraError
                raise CameraError(STALLED_MESSAGE)
        service = make_service(Fails())
        service.start()
        self.assertTrue(wait_until(lambda: service.status()["state"] == "error"))
        self.assertEqual(service.status()["error"], STALLED_MESSAGE)


class StreamGenerationTests(unittest.TestCase):
    def setUp(self):
        self.service = make_service()
        self.app = web_app.create_app(self.service)
        self.client = self.app.test_client()

    def tearDown(self):
        self.service.stop()

    def test_each_start_gets_a_new_generation(self):
        first = self.service.generation()
        self.service.start()
        second = self.service.generation()
        wait_until(lambda: self.service.status()["state"] == RUNNING)
        self.service.stop()
        self.service.start()
        self.assertEqual(second, first + 1)
        self.assertEqual(self.service.generation(), second + 1)

    def test_old_stream_ends_after_stop_and_restart_instead_of_attaching(self):
        self.service.start()
        wait_until(lambda: self.service.status()["state"] == RUNNING)
        old = self.client.get(f"{HOST_URL}/video_feed", buffered=False)
        stream = iter(old.response)
        self.assertTrue(next(stream).startswith(b"--frame"))
        self.service.stop()
        self.service.start()                             # a NEW worker is now active
        wait_until(lambda: self.service.status()["state"] == RUNNING)
        remaining = []
        deadline = time.time() + 5
        for chunk in stream:                             # the old stream must finish by itself
            remaining.append(chunk)
            if time.time() > deadline:
                break
        old.close()
        self.assertLess(len(remaining), 3)               # at most a frame that was already queued
        self.assertLess(time.time(), deadline)
        self.assertEqual(self.service.status()["state"], RUNNING)

    def test_stream_requested_for_a_stopped_service_stays_empty_after_a_later_start(self):
        response = self.client.get(f"{HOST_URL}/video_feed", buffered=False)  # generation captured now
        self.service.start()
        wait_until(lambda: self.service.status()["state"] == RUNNING)
        self.assertEqual(b"".join(response.response), b"")
        response.close()


class FetchSiteTests(unittest.TestCase):
    def setUp(self):
        self.service = make_service()
        self.client = web_app.create_app(self.service).test_client()

    def tearDown(self):
        self.service.stop()

    def test_cross_site_stream_and_post_are_refused(self):
        for site in ("cross-site", "same-site"):
            with self.subTest(site=site):
                feed = self.client.get(f"{HOST_URL}/video_feed", headers={"Sec-Fetch-Site": site})
                self.assertEqual(feed.status_code, 403)
                post = self.client.post(f"{HOST_URL}/api/start", headers={"Sec-Fetch-Site": site})
                self.assertEqual(post.status_code, 403)
        self.assertEqual(FakeCamera.instances, [])

    def test_own_page_and_typed_address_are_allowed(self):
        for site in ("same-origin", "none"):
            with self.subTest(site=site):
                response = self.client.get(f"{HOST_URL}/video_feed", headers={"Sec-Fetch-Site": site})
                self.assertEqual(response.status_code, 200)
                response.close()

    def test_page_and_status_do_not_require_the_header(self):
        self.assertEqual(self.client.get(f"{HOST_URL}/").status_code, 200)
        self.assertEqual(self.client.get(f"{HOST_URL}/api/status",
                                         headers={"Sec-Fetch-Site": "cross-site"}).status_code, 200)

    def test_host_and_origin_protection_still_works(self):
        self.assertEqual(self.client.get("http://evil.example:5000/api/status").status_code, 403)
        self.assertEqual(self.client.post(f"{HOST_URL}/api/start",
                                          headers={"Origin": "http://evil.example"}).status_code, 403)
        self.assertEqual(self.client.get("http://localhost:5000/api/status").status_code, 200)
        self.assertEqual(FakeCamera.instances, [])


class FrontendSourceTests(unittest.TestCase):
    """The page logic cannot run here, so these check that each fix is present in the source."""

    @classmethod
    def setUpClass(cls):
        base = web_app.BASE_DIR
        with open(f"{base}/web/static/js/app.js", encoding="utf-8") as handle:
            cls.js = handle.read()
        with open(f"{base}/web/static/css/style.css", encoding="utf-8") as handle:
            cls.css = handle.read()

    def test_dark_color_scheme(self):
        self.assertRegex(self.css, r"color-scheme:\s*dark")

    def test_responses_are_checked_and_non_json_is_handled(self):
        self.assertIn("response.ok", self.js)
        self.assertRegex(self.js, r"response\.json\(\)\s*;?\s*\}\s*catch")
        self.assertIn("unexpected reply", self.js)

    def test_stream_error_handling_with_bounded_reconnection(self):
        self.assertRegex(self.js, r'video\.addEventListener\("error"')
        self.assertRegex(self.js, r"MAX_STREAM_RETRIES\s*=\s*\d+")

    def test_placeholder_reflects_each_state(self):
        for state in ("stopped", "starting", "error", "stopping", "unknown"):
            self.assertRegex(self.js, rf"PLACEHOLDERS\s*=\s*\{{[^}}]*\b{state}:")

    def test_buttons_are_disabled_while_a_request_is_pending(self):
        self.assertRegex(self.js, r"pending\s*=\s*true")
        self.assertRegex(self.js, r"startButton\.disabled\s*=\s*true")

    def test_older_responses_cannot_overwrite_newer_ones(self):
        self.assertRegex(self.js, r"id\s*<\s*applied")

    def test_old_state_is_not_kept_when_the_server_is_unreachable(self):
        self.assertIn("renderUnreachable", self.js)


if __name__ == "__main__":
    unittest.main()
