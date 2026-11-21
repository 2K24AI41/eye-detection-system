"""Tests for scripts/capture_dataset.py.

No real webcam is used and the real dataset/ folder is never touched: everything runs
in temporary folders that are deleted afterwards. The webcam is replaced by a fake
camera object, and the OpenCV window functions are replaced by recorders.

Run:  python -m unittest discover -s tests -v
"""
import contextlib
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import capture_dataset as cd  # noqa: E402

GOOD_TIME = "2026-01-02T03:04:05+00:00"


def frame():
    return np.zeros((480, 640, 3), np.uint8)


def row(cond="dim", pid="P01", sub="dim", number=1, when=GOOD_TIME, notes=""):
    return {"filename": f"{cond}_{pid}_{number:03d}.png", "participant_id": pid,
            "condition": cond, "sub_condition": sub, "capture_time": when, "notes": notes}


def write_csv(path, rows, header=True, final_newline=True, bom=False):
    """Write a metadata file by hand (the way a crash or an editor might leave it)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=cd.METADATA_FIELDS)
    if header:
        writer.writeheader()
    for item in rows:
        writer.writerow(item)
    text = buffer.getvalue()
    if not final_newline:
        text = text.rstrip("\r\n")
    data = text.encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes((b"\xef\xbb\xbf" if bom else b"") + data)


class TempDatasetCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dir = self.root / "dataset"

    def tearDown(self):
        self._tmp.cleanup()

    def save(self, cond="dim", pid="P01", sub="dim", notes="", image=None):
        return cd.save_capture(self.dir, frame() if image is None else image,
                               cond, pid, sub, notes)

    def images(self):
        return sorted(cd.scan_images(self.dir)[0])

    def meta_bytes(self):
        return cd.metadata_path(self.dir).read_bytes()


# ================================================================== basics
class FilenameTests(unittest.TestCase):
    def test_format(self):
        self.assertEqual(cd.build_filename("dim", "P02", 7), "dim_P02_007.png")
        self.assertEqual(cd.build_filename("bg", "NONE", 120), "bg_NONE_120.png")

    def test_number_out_of_range(self):
        for bad in (0, 1000, -1):
            with self.assertRaises(cd.CaptureError):
                cd.build_filename("dim", "P01", bad)

    def test_pattern_matches_generated_names(self):
        for cond in cd.CONDITIONS:
            self.assertIsNotNone(cd.FILENAME_PATTERN.match(cd.build_filename(cond, "P01", 1)))


class ConfigTests(unittest.TestCase):
    def test_protocol_counts(self):
        targets = {c: i["target"] for c, i in cd.CONDITIONS.items()}
        self.assertEqual(targets, {"normal": 20, "dim": 15, "strong": 15, "turn": 20,
                                   "tilt": 15, "dist": 20, "glasses": 15, "bg": 20})
        self.assertEqual(sum(targets.values()), 140)

    def test_sub_splits(self):
        self.assertEqual(cd.CONDITIONS["turn"]["subs"], {"30deg": 10, "60deg": 10})
        self.assertEqual(cd.CONDITIONS["dist"]["subs"], {"near": 10, "far": 10})


class ValidationTests(unittest.TestCase):
    def test_valid_combinations(self):
        cd.validate_selection("turn", "P01", "30deg")
        cd.validate_selection("bg", "NONE", "background")

    def test_invalid_inputs(self):
        bad = [("nope", "P01", "dim"), ("dim", "P03", "dim"), ("dim", "NONE", "dim"),
               ("bg", "P01", "background"), ("turn", "P01", "45deg"), ("dim", "P01", "side")]
        for combo in bad:
            with self.subTest(combo=combo), self.assertRaises(cd.CaptureError):
                cd.validate_selection(*combo)

    def test_note_cleaning(self):
        self.assertEqual(cd.clean_note("  a\n b  "), "a b")
        self.assertEqual(cd.clean_note(None), "")
        with self.assertRaises(cd.CaptureError):
            cd.clean_note("x" * 201)


class GateTests(unittest.TestCase):
    def test_two_second_interval(self):
        now = [100.0]
        gate = cd.CaptureGate(clock=lambda: now[0])
        self.assertTrue(gate.ready())               # first capture is allowed
        gate.mark()
        self.assertFalse(gate.ready())
        now[0] += 1.9
        self.assertFalse(gate.ready())
        self.assertAlmostEqual(gate.seconds_remaining(), 0.1, places=6)
        now[0] += 0.1
        self.assertTrue(gate.ready())
        gate.mark()
        self.assertFalse(gate.ready())

    def test_default_interval_and_clock(self):
        self.assertEqual(cd.MIN_INTERVAL_SECONDS, 2.0)
        import time
        self.assertIs(cd.CaptureGate().clock, time.monotonic)


# ===================================================== saving and metadata
class SaveAndMetadataTests(TempDatasetCase):
    def test_saves_png_and_metadata_with_timezone(self):
        when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        name = cd.save_capture(self.dir, frame(), "turn", "P02", "60deg", "test note", now=when)
        self.assertEqual(name, "turn_P02_001.png")
        self.assertTrue((self.dir / "images" / name).is_file())
        with open(self.dir / "metadata.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows, [{"filename": name, "participant_id": "P02",
                                 "condition": "turn", "sub_condition": "60deg",
                                 "capture_time": "2026-01-02T03:04:05+00:00",
                                 "notes": "test note"}])

    def test_naive_time_gets_local_offset(self):
        cd.save_capture(self.dir, frame(), "dim", "P01", "dim",
                        now=datetime(2026, 1, 2, 3, 4, 5))
        stamp = cd.read_metadata(self.dir)[0]["capture_time"]
        self.assertRegex(stamp, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$")

    def test_default_time_has_offset(self):
        self.save()
        self.assertRegex(cd.read_metadata(self.dir)[0]["capture_time"], r"[+-]\d\d:\d\d$")

    def test_saved_png_is_640x480_and_unmodified(self):
        original = np.random.RandomState(0).randint(0, 255, (480, 640, 3), np.uint8)
        name = self.save(image=original)
        loaded = cv2.imdecode(np.fromfile(str(self.dir / "images" / name), np.uint8),
                              cv2.IMREAD_COLOR)
        self.assertEqual(loaded.shape, (480, 640, 3))
        self.assertTrue(np.array_equal(loaded, original))       # lossless

    def test_numbers_increase_per_condition(self):
        self.assertEqual(self.save("dim"), "dim_P01_001.png")
        self.assertEqual(self.save("dim"), "dim_P01_002.png")
        self.assertEqual(self.save("normal", sub="frontal"), "normal_P01_001.png")

    def test_number_continues_across_participants(self):
        self.save("dim", "P01")
        self.assertEqual(self.save("dim", "P02"), "dim_P02_002.png")

    def test_wrong_resolution_or_type_rejected_and_nothing_written(self):
        for bad in (np.zeros((240, 320, 3), np.uint8), np.zeros((480, 640, 3), np.uint16), None):
            with self.subTest(shape=getattr(bad, "shape", None)):
                with self.assertRaises(cd.CaptureError):
                    cd.save_capture(self.dir, bad, "dim", "P01", "dim")
        self.assertFalse(self.dir.exists())

    def test_invalid_selection_writes_nothing(self):
        with self.assertRaises(cd.CaptureError):
            self.save("dim", "NONE")
        self.assertFalse(self.dir.exists())

    def test_header_written_once(self):
        self.save()
        self.save()
        lines = (self.dir / "metadata.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], ",".join(cd.METADATA_FIELDS))
        self.assertEqual(len(lines), 3)

    def test_progress_counts(self):
        self.save("turn", sub="30deg")
        self.save("turn", sub="60deg")
        self.save("turn", sub="60deg")
        by_condition, by_sub, by_participant = cd.progress(self.dir)
        self.assertEqual(by_condition["turn"], 3)
        self.assertEqual(by_sub[("turn", "60deg")], 2)
        self.assertEqual(by_participant["P01"], 3)

    def test_progress_on_missing_dataset_is_empty_and_creates_nothing(self):
        self.assertEqual(cd.progress(self.dir), ({}, {}, {}))
        self.assertFalse(self.dir.exists())

    def test_fsync_used_for_image_and_metadata(self):
        calls = []
        real = os.fsync
        with mock.patch.object(cd.os, "fsync", side_effect=lambda fd: (calls.append(fd),
                                                                      real(fd))[1]):
            self.save()
        self.assertGreaterEqual(len(calls), 2)                  # PNG + CSV


# ================================================== overwrite protection
class OverwriteProtectionTests(TempDatasetCase):
    def test_existing_orphan_image_stops_capture(self):
        (self.dir / "images").mkdir(parents=True)
        (self.dir / "images" / "dim_P01_001.png").write_bytes(b"precious")
        with self.assertRaises(cd.CaptureError):
            self.save()
        self.assertEqual((self.dir / "images" / "dim_P01_001.png").read_bytes(), b"precious")

    def test_exclusive_write_never_overwrites_or_deletes_existing(self):
        name = self.save()
        path = self.dir / "images" / name
        before = path.read_bytes()
        state = cd.load_state(self.dir)
        state["used"] = {}                       # pretend no number is taken: force a clash
        with mock.patch.object(cd, "load_state", return_value=state):
            with self.assertRaises(cd.CaptureError):
                self.save()
        self.assertEqual(path.read_bytes(), before)             # still there, unchanged
        self.assertEqual(len(cd.read_metadata(self.dir)), 1)

    def test_target_reached_blocks_more_images(self):
        for _ in range(10):
            self.save("turn", sub="30deg")
        with self.assertRaises(cd.CaptureError):                # sub-target 10 reached
            self.save("turn", sub="30deg")
        self.save("turn", sub="60deg")                          # other split still open
        self.assertEqual(cd.progress(self.dir)[0]["turn"], 11)

    def test_total_target_blocks(self):
        for _ in range(15):
            self.save("dim")
        with self.assertRaises(cd.CaptureError):
            self.save("dim")

    def test_writes_stay_inside_dataset_dir(self):
        with self.assertRaises(cd.CaptureError):
            cd.ensure_inside(self.dir, self.dir.parent / "elsewhere.png")
        with self.assertRaises(cd.CaptureError):
            cd.ensure_inside(self.dir, self.dir / ".." / "elsewhere.png")
        cd.ensure_inside(self.dir, self.dir / "images" / "x.png")


# ============================================ I1: partial image rollback
class PartialWriteRollbackTests(TempDatasetCase):
    def failing_open(self, error):
        """open() replacement: creates the real file, writes 10 bytes, then fails."""
        real_open = open

        class Half:
            def __init__(self, path, mode):
                self.handle = real_open(path, mode)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.handle.close()

            def write(self, data):
                self.handle.write(data[:10])
                self.handle.flush()
                raise error

            def flush(self):
                self.handle.flush()

            def fileno(self):
                return self.handle.fileno()

        return lambda path, mode="r", *a, **k: Half(path, mode) if mode == "xb" else \
            real_open(path, mode, *a, **k)

    def test_partial_png_is_removed_and_original_error_kept(self):
        original = OSError("disk full")
        with mock.patch.object(cd, "open", self.failing_open(original), create=True):
            with self.assertRaises(cd.CaptureError) as caught:
                self.save()
        self.assertIs(caught.exception.__cause__, original)
        self.assertEqual(self.images(), [])
        self.assertFalse((self.dir / "metadata.csv").exists())
        self.assertEqual(self.save(), "dim_P01_001.png")        # number is free again

    def test_keyboard_interrupt_during_png_write(self):
        with mock.patch.object(cd, "open", self.failing_open(KeyboardInterrupt()), create=True):
            with self.assertRaises(KeyboardInterrupt):
                self.save()
        self.assertEqual(self.images(), [])
        self.assertFalse((self.dir / "metadata.csv").exists())

    def test_keyboard_interrupt_between_png_and_metadata(self):
        with mock.patch.object(cd, "append_metadata", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.save()
        self.assertEqual(self.images(), [])
        self.assertFalse((self.dir / "metadata.csv").exists())

    def test_metadata_failure_removes_new_image(self):
        with mock.patch.object(cd, "append_metadata", side_effect=cd.CaptureError("disk full")):
            with self.assertRaises(cd.CaptureError):
                self.save()
        self.assertEqual(self.images(), [])

    def test_cleanup_failure_is_reported_explicitly(self):
        original = OSError("disk full")
        stderr = io.StringIO()
        with mock.patch.object(cd, "open", self.failing_open(original), create=True), \
                mock.patch.object(cd, "_remove_file", side_effect=OSError("locked")), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaises(cd.CaptureError) as caught:
                self.save()
        message = str(caught.exception)
        self.assertIn("could not remove the incomplete image", message)
        self.assertIn("locked", message)
        self.assertIn("dim_P01_001.png", message)
        self.assertIs(caught.exception.__cause__, original)
        self.assertIn("WARNING", stderr.getvalue())
        with self.assertRaises(cd.CaptureError):                # leftover blocks further saves
            self.save()

    def test_keyboard_interrupt_is_reraised_even_if_cleanup_fails(self):
        stderr = io.StringIO()
        with mock.patch.object(cd, "append_metadata", side_effect=KeyboardInterrupt), \
                mock.patch.object(cd, "_remove_file", side_effect=OSError("locked")), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaises(KeyboardInterrupt):
                self.save()
        self.assertIn("could not remove", stderr.getvalue())


# ================================================ I2: metadata integrity
class MetadataValidationTests(TempDatasetCase):
    @property
    def meta(self):
        return cd.metadata_path(self.dir)

    def assert_rejected(self, rows, fragment, **kwargs):
        write_csv(self.meta, rows, **kwargs)
        with self.assertRaises(cd.CaptureError) as caught:
            cd.read_metadata(self.dir)
        self.assertIn(fragment, str(caught.exception))
        return str(caught.exception)

    def test_valid_file_is_read(self):
        write_csv(self.meta, [row(number=1), row(number=2, pid="P02")])
        self.assertEqual(len(cd.read_metadata(self.dir)), 2)

    def test_duplicate_filename_rejected_with_line_numbers(self):
        message = self.assert_rejected([row(number=1), row(number=1)], "duplicate row")
        self.assertIn("line 3", message)
        self.assertIn("line 2", message)

    def test_duplicate_row_cannot_inflate_progress(self):
        self.save("turn", sub="30deg")
        cd.append_metadata(self.dir, cd.read_metadata(self.dir)[0])  # forged duplicate
        with self.assertRaises(cd.CaptureError):
            cd.progress(self.dir)
        with self.assertRaises(cd.CaptureError):
            self.save("turn", sub="30deg")
        self.assertEqual(len(self.images()), 1)                     # nothing new written

    def test_same_number_for_two_participants_rejected(self):
        self.assert_rejected([row(pid="P01", number=3), row(pid="P02", number=3)],
                             "used twice")

    def test_condition_disagreeing_with_filename(self):
        bad = row(number=1)
        bad["condition"] = "normal"
        self.assert_rejected([bad], "disagrees")

    def test_participant_disagreeing_with_filename(self):
        bad = row(number=1)
        bad["participant_id"] = "P02"
        self.assert_rejected([bad], "disagrees")

    def test_unknown_sub_condition(self):
        self.assert_rejected([row(sub="45deg")], "sub-condition")

    def test_malformed_filename(self):
        bad = row()
        bad["filename"] = "photo.png"
        self.assert_rejected([bad], "does not match")

    def test_path_in_filename_rejected(self):
        bad = row()
        bad["filename"] = "../dim_P01_001.png"
        self.assert_rejected([bad], "does not match")

    def test_bad_timestamp(self):
        self.assert_rejected([row(when="yesterday")], "ISO")

    def test_incomplete_and_overlong_rows(self):
        write_csv(self.meta, [row()])
        with open(self.meta, "ab") as handle:
            handle.write(b"dim_P02_002.png,P02,dim\r\n")
        with self.assertRaises(cd.CaptureError) as caught:
            cd.read_metadata(self.dir)
        self.assertIn("fewer values", str(caught.exception))
        write_csv(self.meta, [row()])
        with open(self.meta, "ab") as handle:
            handle.write(b"dim_P02_002.png,P02,dim,dim,2026-01-02T03:04:05+00:00,,extra\r\n")
        with self.assertRaises(cd.CaptureError) as caught:
            cd.read_metadata(self.dir)
        self.assertIn("more values", str(caught.exception))

    def test_unexpected_header(self):
        self.meta.parent.mkdir(parents=True)
        self.meta.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        with self.assertRaises(cd.CaptureError):
            self.save()
        self.assertEqual(self.meta.read_text(encoding="utf-8"), "a,b,c\n1,2,3\n")

    def test_empty_file_rejected(self):
        self.meta.parent.mkdir(parents=True)
        self.meta.write_bytes(b"")
        with self.assertRaises(cd.CaptureError) as caught:
            cd.read_metadata(self.dir)
        self.assertIn("empty", str(caught.exception))

    def test_bom_is_accepted_and_preserved(self):
        self.save()
        raw = self.meta_bytes()
        self.meta.write_bytes(b"\xef\xbb\xbf" + raw)
        self.assertEqual(len(cd.read_metadata(self.dir)), 1)
        self.assertEqual(self.save(), "dim_P01_002.png")
        self.assertTrue(self.meta_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertEqual(len(cd.read_metadata(self.dir)), 2)

    def test_old_timestamp_without_offset_still_valid(self):
        write_csv(self.meta, [row(when="2026-01-02T03:04:05")])
        self.assertEqual(len(cd.read_metadata(self.dir)), 1)


# =================================================== I3: CSV append safety
class CsvAppendSafetyTests(TempDatasetCase):
    def setUp(self):
        super().setUp()
        self.save()
        self.before = self.meta_bytes()

    @property
    def meta(self):
        return cd.metadata_path(self.dir)

    def test_truncated_final_row_blocks_capture_before_any_image(self):
        with open(self.meta, "ab") as handle:
            handle.write(b"dim_P01_002.png,P01,di")
        broken = self.meta_bytes()
        with self.assertRaises(cd.CaptureError):
            self.save()
        self.assertEqual(self.meta_bytes(), broken)             # not appended onto
        self.assertEqual(self.images(), ["dim_P01_001.png"])    # no image created

    def test_complete_row_without_final_newline_is_not_appended_onto(self):
        self.meta.write_bytes(self.before.rstrip(b"\r\n"))
        without_newline = self.meta_bytes()
        with self.assertRaises(cd.CaptureError) as caught:
            cd.append_metadata(self.dir, row(number=2))
        self.assertIn("line break", str(caught.exception))
        with self.assertRaises(cd.CaptureError):
            cd.read_metadata(self.dir)
        with self.assertRaises(cd.CaptureError):
            self.save()
        self.assertEqual(self.meta_bytes(), without_newline)
        self.assertEqual(self.images(), ["dim_P01_001.png"])

    def test_append_failure_restores_previous_length(self):
        real = os.fsync
        calls = []

        def fail_once(fd):                      # the append fails; the rollback's fsync works
            calls.append(1)
            if len(calls) == 1:
                raise OSError("disk full")
            real(fd)
        with mock.patch.object(cd.os, "fsync", side_effect=fail_once):
            with self.assertRaises(cd.CaptureError) as caught:
                cd.append_metadata(self.dir, row(number=2))
        self.assertIn("restored", str(caught.exception))
        self.assertEqual(self.meta_bytes(), self.before)

    def test_partial_write_is_truncated_away(self):
        real_open = open

        class Half:
            def __init__(self, path, mode):
                self.handle = real_open(path, mode)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.handle.close()

            def write(self, data):
                self.handle.write(data[:15])               # torn write
                self.handle.flush()
                raise OSError("disk full")

        opener = lambda p, m="r", *a, **k: (Half(p, m) if m == "ab"          # noqa: E731
                                            else real_open(p, m, *a, **k))
        with mock.patch.object(cd, "open", opener, create=True):
            with self.assertRaises(cd.CaptureError):
                cd.append_metadata(self.dir, row(number=2))
        self.assertEqual(self.meta_bytes(), self.before)

    def test_failed_first_append_removes_the_new_file(self):
        shutil.rmtree(self.dir)
        with mock.patch.object(cd.os, "fsync", side_effect=OSError("disk full")):
            with self.assertRaises(cd.CaptureError):
                cd.append_metadata(self.dir, row(number=1))
        self.assertFalse(self.meta.exists())

    def test_keyboard_interrupt_during_append_restores_and_reraises(self):
        with mock.patch.object(cd.os, "fsync", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                cd.append_metadata(self.dir, row(number=2))
        self.assertEqual(self.meta_bytes(), self.before)

    def test_recovery_failure_is_reported_clearly(self):
        with mock.patch.object(cd.os, "fsync", side_effect=OSError("disk full")), \
                mock.patch.object(cd, "_truncate_file", side_effect=OSError("locked")):
            with self.assertRaises(cd.CaptureError) as caught:
                cd.append_metadata(self.dir, row(number=2))
        message = str(caught.exception)
        self.assertIn("disk full", message)
        self.assertIn("restoring it failed", message)
        self.assertIn("locked", message)

    def test_full_save_with_failing_csv_leaves_no_new_files(self):
        with mock.patch.object(cd, "_truncate_file", side_effect=OSError("locked")):
            real = os.fsync
            calls = []

            def fsync(fd):
                calls.append(1)
                if len(calls) == 2:                             # 1st = PNG, 2nd = CSV
                    raise OSError("disk full")
                real(fd)
            with mock.patch.object(cd.os, "fsync", side_effect=fsync):
                with self.assertRaises(cd.CaptureError):
                    self.save()
        self.assertEqual(self.images(), ["dim_P01_001.png"])    # new PNG was rolled back


# ================================================ I4: Git ignore validation
HAVE_GIT = shutil.which("git") is not None


@unittest.skipUnless(HAVE_GIT, "git is not installed")
class GitIgnoreTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": os.devnull,
                                               "GIT_CONFIG_NOSYSTEM": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)

    def repo_with(self, gitignore):
        (self.repo / ".gitignore").write_text(gitignore, encoding="utf-8")

    def test_valid_pattern_variants_pass(self):
        for content in ("dataset/\n", "/dataset/\n", "dataset\n", "dataset/*\n",
                        ".venv/\ndataset/\n", "dataset/  \n"):
            with self.subTest(content=content):
                self.repo_with(content)
                cd.check_gitignore(self.repo)

    def test_missing_or_commented_rule_stops(self):
        for content in (".venv/\n", "# dataset/\n", ""):
            with self.subTest(content=content):
                self.repo_with(content)
                with self.assertRaises(cd.CaptureError) as caught:
                    cd.check_gitignore(self.repo)
                self.assertIn("does NOT ignore", str(caught.exception))

    def test_negation_rule_is_detected(self):
        self.repo_with("dataset/\n!dataset/\n")
        with self.assertRaises(cd.CaptureError):
            cd.check_gitignore(self.repo)

    def test_negation_of_contents_is_detected(self):
        self.repo_with("dataset/*\n!dataset/images/\n!dataset/metadata.csv\n")
        with self.assertRaises(cd.CaptureError):
            cd.check_gitignore(self.repo)

    def test_tracked_dataset_file_stops(self):
        self.repo_with("dataset/\n")
        (self.repo / "dataset").mkdir()
        (self.repo / "dataset" / "metadata.csv").write_text("x", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.repo), "add", "-f", "dataset/metadata.csv"],
                       check=True, capture_output=True)
        with self.assertRaises(cd.CaptureError):
            cd.check_gitignore(self.repo)

    def test_not_a_repository_stops_clearly(self):
        with tempfile.TemporaryDirectory() as plain:
            with self.assertRaises(cd.CaptureError) as caught:
                cd.check_gitignore(plain)
            self.assertIn("Git repository", str(caught.exception))

    def test_real_project_ignores_dataset(self):
        cd.check_gitignore(cd.PROJECT_ROOT)


class GitMissingTests(unittest.TestCase):
    def test_git_not_found(self):
        with mock.patch.object(cd.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaises(cd.CaptureError) as caught:
                cd.check_gitignore(".")
        self.assertIn("not found", str(caught.exception))


# ============================================== I6: path and link safety
class LinkSafetyTests(TempDatasetCase):
    def make_link(self, link, target, directory=True):
        """Create a symlink; on Windows fall back to a junction (needs no privilege)."""
        try:
            os.symlink(target, link, target_is_directory=directory)
            return "symlink"
        except (OSError, NotImplementedError):
            pass
        if sys.platform == "win32" and directory:
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return "junction"
        self.skipTest("this system does not allow creating symlinks/junctions "
                      "(path-validation logic is covered by the mocked tests below)")

    def assert_blocked(self, outside):
        before = sorted(p.name for p in outside.iterdir())
        for action in (lambda: cd.load_state(self.dir),
                       lambda: cd.read_metadata(self.dir),
                       lambda: cd.scan_images(self.dir),
                       lambda: cd.append_metadata(self.dir, row()),
                       lambda: self.save()):
            with self.assertRaises(cd.CaptureError) as caught:
                action()
            self.assertRegex(str(caught.exception), "link|junction|different location")
        self.assertEqual(sorted(p.name for p in outside.iterdir()), before)

    def test_dataset_folder_link_is_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        kind = self.make_link(self.dir, outside)
        self.assertTrue(cd.is_link(self.dir), kind)
        self.assert_blocked(outside)
        self.assertEqual(list(outside.iterdir()), [])

    def test_images_folder_link_is_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.dir.mkdir()
        self.make_link(self.dir / "images", outside)
        self.assert_blocked(outside)
        self.assertEqual(list(outside.iterdir()), [])

    def test_metadata_symlink_is_not_followed(self):
        outside = self.root / "outside"
        outside.mkdir()
        target = outside / "real.csv"
        write_csv(target, [row()])
        self.dir.mkdir()
        self.make_link(self.dir / "metadata.csv", target, directory=False)
        before = target.read_bytes()
        for action in (lambda: cd.read_metadata(self.dir),
                       lambda: cd.append_metadata(self.dir, row(number=2)),
                       lambda: self.save()):
            with self.assertRaises(cd.CaptureError):
                action()
        self.assertEqual(target.read_bytes(), before)

    def test_mocked_link_detection_covers_each_path(self):
        self.save()
        for target in (self.dir, self.dir / "images", self.dir / "metadata.csv"):
            with self.subTest(path=target.name):
                real = cd.is_link
                with mock.patch.object(cd, "is_link",
                                       side_effect=lambda p, t=target: Path(p) == t or real(p)):
                    with self.assertRaises(cd.CaptureError) as caught:
                        cd.check_layout(self.dir)
                    self.assertIn("link or junction", str(caught.exception))
                    with self.assertRaises(cd.CaptureError):
                        self.save()
        self.assertEqual(self.images(), ["dim_P01_001.png"])

    def test_is_link_false_for_plain_paths_and_missing(self):
        self.save()
        self.assertFalse(cd.is_link(self.dir))
        self.assertFalse(cd.is_link(self.dir / "images" / "dim_P01_001.png"))
        self.assertFalse(cd.is_link(self.dir / "nothing"))

    def test_is_link_detects_reparse_point_attribute(self):
        fake = os.stat_result((0o100644, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        fake_info = mock.Mock(st_mode=fake.st_mode, st_file_attributes=0x400)
        with mock.patch.object(cd.os, "lstat", return_value=fake_info):
            self.assertTrue(cd.is_link("anything"))

    def test_image_entry_that_is_a_folder_is_refused(self):
        (self.dir / "images" / "dim_P01_001.png").mkdir(parents=True)
        with self.assertRaises(cd.CaptureError) as caught:
            self.save()
        self.assertIn("link or folder", str(caught.exception))

    def test_metadata_that_is_a_folder_is_refused(self):
        (self.dir / "metadata.csv").mkdir(parents=True)
        with self.assertRaises(cd.CaptureError):
            self.save()

    def test_images_path_that_is_a_file_is_refused(self):
        self.dir.mkdir()
        (self.dir / "images").write_bytes(b"x")
        with self.assertRaises(cd.CaptureError):
            self.save()


# ====================================== I7: reservation, unexpected files
class ReservationAndFolderTests(TempDatasetCase):
    def test_deleted_image_number_is_not_reused(self):
        name = self.save()
        (self.dir / "images" / name).unlink()                   # user discarded the image
        self.assertEqual(self.save(), "dim_P01_002.png")

    def test_missing_image_is_warned_and_not_counted(self):
        name = self.save()
        (self.dir / "images" / name).unlink()
        warnings = cd.check_consistency(self.dir)
        self.assertTrue(any("reserved" in w for w in warnings))
        self.assertEqual(cd.progress(self.dir)[0].get("dim", 0), 0)

    def test_metadata_only_row_reserves_its_number(self):
        write_csv(cd.metadata_path(self.dir), [row(number=3)])
        self.assertEqual(cd.next_index(self.dir, "dim"), 4)
        self.assertEqual(self.save(), "dim_P01_004.png")

    def test_missing_image_does_not_count_toward_target(self):
        for _ in range(10):
            self.save("turn", sub="30deg")
        (self.dir / "images" / "turn_P01_001.png").unlink()
        self.assertEqual(cd.progress(self.dir)[1][("turn", "30deg")], 9)
        self.save("turn", sub="30deg")                          # one retake is allowed
        with self.assertRaises(cd.CaptureError):
            self.save("turn", sub="30deg")

    def test_gap_without_image_and_row_is_warned(self):
        for _ in range(3):
            self.save()
        (self.dir / "images" / "dim_P01_002.png").unlink()
        rows = [r for r in cd.read_metadata(self.dir) if r["filename"] != "dim_P01_002.png"]
        write_csv(cd.metadata_path(self.dir), rows)
        warnings = cd.check_consistency(self.dir)
        self.assertTrue(any("never delete" in w and "[2]" in w for w in warnings))

    def test_unexpected_files_warn_but_do_not_block(self):
        self.save()
        (self.dir / "images" / "Thumbs.db").write_bytes(b"x")
        (self.dir / "images" / "notes").mkdir()
        warnings = cd.check_consistency(self.dir)
        self.assertTrue(any("unexpected" in w and "Thumbs.db" in w for w in warnings))
        self.assertEqual(self.save(), "dim_P01_002.png")
        self.assertEqual((self.dir / "images" / "Thumbs.db").read_bytes(), b"x")

    def test_unrecorded_image_still_blocks(self):
        self.save()
        (self.dir / "images" / "dim_P01_002.png").write_bytes(b"x")
        with self.assertRaises(cd.CaptureError) as caught:
            self.save()
        self.assertIn("no metadata row", str(caught.exception))


# ================================================ consent and main()
class FakeCamera:
    """Stands in for src.camera.Camera: no hardware, scripted frames."""

    def __init__(self, fail_open=False, fail_read_after=None, size=(640, 480)):
        self.fail_open, self.fail_read_after, self.size = fail_open, fail_read_after, size
        self.opened = self.released = False
        self.reads = 0
        self.master = np.random.RandomState(1).randint(0, 255, (480, 640, 3), np.uint8)

    def open(self):
        if self.fail_open:
            raise cd.CameraError("no camera")
        self.opened = True

    def properties(self):
        return {"width": self.size[0], "height": self.size[1], "fps": 0.0}

    def read(self):
        self.reads += 1
        if self.fail_read_after is not None and self.reads > self.fail_read_after:
            raise cd.CameraError("camera disconnected")
        return self.master.copy()

    def release(self):
        self.released = True


class WindowFakes:
    """Replaces the OpenCV window functions with scripted recorders."""

    def __init__(self, keys, visible=1):
        self.keys = iter(keys)
        self.visible = visible
        self.destroyed = 0
        self.shown = []

    def __enter__(self):
        self.patches = [
            mock.patch.object(cd.cv2, "imshow", side_effect=lambda w, f: self.shown.append(f)),
            mock.patch.object(cd.cv2, "waitKey", side_effect=lambda ms: next(self.keys, 113)),
            mock.patch.object(cd.cv2, "getWindowProperty", side_effect=lambda *a: self.visible),
            mock.patch.object(cd.cv2, "destroyAllWindows",
                              side_effect=lambda: setattr(self, "destroyed", self.destroyed + 1)),
        ]
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in self.patches:
            patch.stop()


SPACE, Q, ESC, NOKEY = 32, 113, 27, -1


class ConsentTests(unittest.TestCase):
    def run_consent(self, participant, answer=None, flag=False):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            if answer is None:
                with mock.patch("builtins.input", side_effect=EOFError):
                    cd.confirm_consent(participant, flag)
            else:
                with mock.patch("builtins.input", return_value=answer):
                    cd.confirm_consent(participant, flag)
        return out.getvalue()

    def test_p01_needs_no_prompt(self):
        self.assertEqual(self.run_consent("P01"), "")

    def test_p02_yes_passes_and_notice_explains_limits(self):
        text = self.run_consent("P02", "Yes")
        self.assertIn("SELF-DECLARATION", text)
        self.assertIn("SEPARATE", text)
        self.assertIn("does not replace the signed consent", text)

    def test_p02_anything_else_is_refused(self):
        for answer in ("no", "", "y", "yes please"):
            with self.subTest(answer=answer):
                with self.assertRaises(cd.CaptureError):
                    self.run_consent("P02", answer)

    def test_p02_without_input_is_refused(self):
        with self.assertRaises(cd.CaptureError):
            self.run_consent("P02", None)

    def test_flag_skips_prompt_but_says_so(self):
        text = self.run_consent("P02", None, flag=True)
        self.assertIn("not checked by this script", text)


class MainFlowTests(TempDatasetCase):
    def run_main(self, argv, camera, keys=(Q,), input_answer="yes"):
        out = io.StringIO()
        with mock.patch.object(cd, "DATASET_DIR", self.dir), \
                mock.patch.object(cd, "check_gitignore"), \
                mock.patch.object(cd, "Camera", lambda index=0: camera), \
                mock.patch("builtins.input", return_value=input_answer), \
                WindowFakes(keys) as fakes, contextlib.redirect_stdout(out):
            code = cd.main(argv)
        return code, out.getvalue(), fakes

    def test_p02_without_consent_never_opens_camera_or_writes(self):
        camera = FakeCamera()
        code, output, _ = self.run_main(
            ["--condition", "dim", "--participant", "P02", "--sub", "dim"],
            camera, input_answer="no")
        self.assertEqual(code, 1)
        self.assertIn("Consent not confirmed", output)
        self.assertFalse(camera.opened)
        self.assertEqual(camera.reads, 0)
        self.assertFalse(self.dir.exists())

    def test_p02_with_consent_opens_camera(self):
        camera = FakeCamera()
        code, _, _ = self.run_main(
            ["--condition", "dim", "--participant", "P02", "--sub", "dim"], camera)
        self.assertEqual(code, 0)
        self.assertTrue(camera.opened)

    def test_p01_does_not_ask_for_consent(self):
        camera = FakeCamera()
        with mock.patch("builtins.input", side_effect=AssertionError("must not ask")):
            out = io.StringIO()
            with mock.patch.object(cd, "DATASET_DIR", self.dir), \
                    mock.patch.object(cd, "check_gitignore"), \
                    mock.patch.object(cd, "Camera", lambda index=0: camera), \
                    WindowFakes((Q,)), contextlib.redirect_stdout(out):
                code = cd.main(["--condition", "dim", "--participant", "P01", "--sub", "dim"])
        self.assertEqual(code, 0)

    def test_status_does_not_open_camera_or_create_files(self):
        camera = FakeCamera()
        code, output, _ = self.run_main(["--status"], camera)
        self.assertEqual(code, 0)
        self.assertIn("Dataset progress", output)
        self.assertFalse(camera.opened)
        self.assertFalse(self.dir.exists())

    def test_gitignore_failure_stops_before_camera(self):
        camera = FakeCamera()
        out = io.StringIO()
        with mock.patch.object(cd, "DATASET_DIR", self.dir), \
                mock.patch.object(cd, "check_gitignore", side_effect=cd.CaptureError("nope")), \
                mock.patch.object(cd, "Camera", lambda index=0: camera), \
                contextlib.redirect_stdout(out):
            code = cd.main(["--condition", "dim", "--participant", "P01", "--sub", "dim"])
        self.assertEqual(code, 1)
        self.assertFalse(camera.opened)


# ============================================ I5 and session behaviour
class SessionTests(TempDatasetCase):
    def run_session(self, keys, camera=None, cond="dim", pid="P01", sub="dim", state=None):
        camera = camera or FakeCamera()
        state = state or cd.load_state(self.dir)
        out = io.StringIO()
        with mock.patch.object(cd, "Camera", lambda index=0: camera), \
                WindowFakes(keys) as fakes, contextlib.redirect_stdout(out):
            with mock.patch.object(cd, "load_state", wraps=cd.load_state) as spy:
                try:
                    code = cd.run_capture(self.dir, cond, pid, sub, "", state)
                    error = None
                except BaseException as exc:                    # noqa: BLE001
                    code, error = None, exc
        return camera, fakes, spy, out.getvalue(), code, error

    def test_no_disk_scan_per_frame(self):
        camera, fakes, spy, _, code, _ = self.run_session([NOKEY] * 40 + [Q])
        self.assertEqual(code, 0)
        self.assertEqual(camera.reads, 41)
        self.assertEqual(spy.call_count, 0)                     # nothing re-read per frame
        self.assertEqual(len(fakes.shown), 41)

    def test_only_one_scan_per_save(self):
        _, _, spy, _, _, _ = self.run_session([NOKEY] * 10 + [SPACE] + [NOKEY] * 10 + [Q])
        self.assertEqual(spy.call_count, 1)

    def test_space_saves_untouched_frame_and_overlay_is_a_copy(self):
        camera, fakes, _, _, code, _ = self.run_session([NOKEY, SPACE, NOKEY, Q])
        self.assertEqual(code, 0)
        name = self.images()[0]
        saved = cv2.imdecode(np.fromfile(str(self.dir / "images" / name), np.uint8),
                             cv2.IMREAD_COLOR)
        self.assertTrue(np.array_equal(saved, camera.master))   # no text drawn into the file
        self.assertFalse(np.array_equal(fakes.shown[0], camera.master))  # screen has overlay

    def test_second_space_within_two_seconds_is_refused(self):
        _, _, _, output, _, _ = self.run_session([SPACE, SPACE, SPACE, Q])
        self.assertEqual(len(self.images()), 1)
        self.assertEqual(output.count("Saved dim_P01_"), 1)

    def test_q_and_esc_quit_and_cleanup(self):
        for key in (Q, ESC, ord("Q")):
            with self.subTest(key=key):
                camera, fakes, _, _, code, _ = self.run_session([NOKEY, key])
                self.assertEqual(code, 0)
                self.assertTrue(camera.released)
                self.assertEqual(fakes.destroyed, 1)

    def test_closing_window_quits(self):
        camera = FakeCamera()
        out = io.StringIO()
        with mock.patch.object(cd, "Camera", lambda index=0: camera), \
                WindowFakes([NOKEY] * 5, visible=0) as fakes, contextlib.redirect_stdout(out):
            code = cd.run_capture(self.dir, "dim", "P01", "dim", "", cd.load_state(self.dir))
        self.assertEqual(code, 0)
        self.assertEqual(camera.reads, 1)
        self.assertTrue(camera.released)
        self.assertEqual(fakes.destroyed, 1)

    def test_camera_open_failure_cleans_up(self):
        camera, fakes, _, _, code, error = self.run_session([Q], FakeCamera(fail_open=True))
        self.assertIsInstance(error, cd.CameraError)
        self.assertTrue(camera.released)
        self.assertEqual(fakes.destroyed, 1)

    def test_frame_read_failure_cleans_up_and_keeps_saved_images(self):
        camera = FakeCamera(fail_read_after=3)
        camera_, fakes, _, _, code, error = self.run_session([SPACE, NOKEY, NOKEY, NOKEY], camera)
        self.assertIsInstance(error, cd.CameraError)
        self.assertTrue(camera.released)
        self.assertEqual(fakes.destroyed, 1)
        self.assertEqual(len(self.images()), 1)

    def test_wrong_camera_resolution_is_refused(self):
        camera, fakes, _, _, code, error = self.run_session(
            [SPACE, Q], FakeCamera(size=(320, 240)))
        self.assertIsInstance(error, cd.CaptureError)
        self.assertTrue(camera.released)
        self.assertEqual(self.images(), [])

    def test_keyboard_interrupt_in_loop_still_cleans_up(self):
        camera = FakeCamera()
        out = io.StringIO()
        with mock.patch.object(cd, "Camera", lambda index=0: camera), \
                WindowFakes([NOKEY]) as fakes, contextlib.redirect_stdout(out):
            with mock.patch.object(cd.cv2, "waitKey", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    cd.run_capture(self.dir, "dim", "P01", "dim", "", cd.load_state(self.dir))
        self.assertTrue(camera.released)

    def test_target_enforced_from_disk_even_with_stale_session_counts(self):
        for _ in range(9):
            self.save("turn", sub="30deg")
        stale = cd.load_state(self.dir)                         # the session thinks 9 are saved
        self.save("turn", sub="30deg")                          # ...but disk now has 10
        _, _, _, output, _, _ = self.run_session([SPACE, Q], cond="turn", sub="30deg",
                                                 state=stale)
        self.assertIn("Target reached", output)
        self.assertEqual(len(self.images()), 10)

    def test_on_screen_counter_updates_after_a_save(self):
        texts = []
        real = cd.overlay
        with mock.patch.object(cd, "overlay",
                               side_effect=lambda f, lines, *a: (texts.append(lines[0]),
                                                                  real(f, lines, *a))[1]):
            self.run_session([NOKEY, SPACE, NOKEY, Q])
        self.assertIn("0/15 saved", texts[0])
        self.assertIn("1/15 saved", texts[-1])


# =============================================================== arguments
class ArgumentTests(unittest.TestCase):
    def test_selection_from_arguments(self):
        args = cd.parse_args(["--condition", "turn", "--participant", "P01",
                              "--sub", "30deg", "--notes", "  hello "])
        self.assertEqual(cd.resolve_selection(args), ("turn", "P01", "30deg", "hello"))

    def test_background_forces_none(self):
        args = cd.parse_args(["--condition", "bg"])
        self.assertEqual(cd.resolve_selection(args), ("bg", "NONE", "background", ""))

    def test_background_with_person_rejected(self):
        args = cd.parse_args(["--condition", "bg", "--participant", "P01"])
        with self.assertRaises(cd.CaptureError):
            cd.resolve_selection(args)

    def test_status_text_lists_every_condition(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(out):
            cd.print_status(Path(tmp) / "dataset")
        for code in cd.CONDITIONS:
            self.assertIn(code, out.getvalue())
        self.assertTrue(re.search(r"total\s+0/140", out.getvalue()))


if __name__ == "__main__":
    unittest.main()
