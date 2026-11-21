"""Tests for scripts/validate_labels.py (synthetic data in temporary folders only)."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotation_helpers as h  # noqa: E402

vl = h.vl
P1, P2, P3 = "normal_P01_001.png", "normal_P01_002.png", "normal_P01_003.png"
BG = "bg_NONE_001.png"


class ValidatorCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dataset = h.make_dataset(self._tmp.name, n_person=3, n_bg=1)

    def good(self, extra_labels=(), extra_status=()):
        """Labels/status for a fully valid, complete dataset (3 persons + 1 background)."""
        labels, status = [], []
        for n in (P1, P2, P3):
            labels += [h.face_row(n), h.eye_row(n, "left", 270, 190), h.eye_row(n, "right", 370, 190)]
            status.append(h.status_row(n))
        status.append(h.status_row(BG, "no_face", 0, 0))
        h.write_labels(self.dataset, labels + list(extra_labels))
        h.write_status(self.dataset, status + list(extra_status))

    def check(self, **kw):
        return vl.validate_dataset(self.dataset, **kw)

    def assert_error(self, fragment, **kw):
        report = self.check(**kw)
        self.assertTrue(any(fragment in e for e in report["errors"]),
                        f"{fragment!r} not in {report['errors']}")
        return report


class ValidDatasetTests(ValidatorCase):
    def test_valid_dataset_has_no_errors(self):
        self.good()
        report = self.check()
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["stats"]["annotated_images"], 4)
        self.assertEqual((report["stats"]["faces"], report["stats"]["eyes"]), (3, 6))

    def test_empty_files_are_ok_when_partial_but_incomplete_otherwise(self):
        self.assertEqual(self.check(require_complete=False)["errors"], [])
        self.assert_error("have no status row")

    def test_hidden_eye_is_valid(self):
        self.good()
        rows = [r for r in (h.face_row(P1), h.eye_row(P1, "left", 270, 190))]
        h.write_labels(self.dataset, rows + [h.face_row(n) for n in (P2, P3)] +
                       [h.eye_row(n, s, x, 190) for n in (P2, P3)
                        for s, x in (("left", 270), ("right", 370))])
        h.write_status(self.dataset, [h.status_row(P1, n_eye=1), h.status_row(P2),
                                      h.status_row(P3), h.status_row(BG, "no_face", 0, 0)])
        self.assertEqual(self.check()["errors"], [])

    def test_no_face_person_image_valid_with_warning(self):
        h.write_labels(self.dataset, [])
        h.write_status(self.dataset, [h.status_row(P1, "no_face", 0, 0)])
        report = self.check(require_complete=False)
        self.assertEqual(report["errors"], [])
        self.assertTrue(any("marked no_face" in w for w in report["warnings"]))


class RuleTests(ValidatorCase):
    def test_every_image_exactly_one_status(self):
        self.good(extra_status=[h.status_row(P1)])
        self.assert_error("more than one status row")

    def test_missing_status_when_complete(self):
        self.good()
        h.write_status(self.dataset, [h.status_row(P1)])
        h.write_labels(self.dataset, [h.face_row(P1), h.eye_row(P1, "left", 270, 190),
                                      h.eye_row(P1, "right", 370, 190)])
        self.assert_error("3 of 4 images have no status row")
        self.assertEqual(self.check(require_complete=False)["errors"], [])

    def test_face_outside_image(self):
        self.good()
        rows = [h.face_row(P1, x=500, w=200)] + [h.eye_row(P1, "left", 510, 190)]
        h.write_labels(self.dataset, rows)
        h.write_status(self.dataset, [h.status_row(P1, n_eye=1)])
        self.assert_error("outside the 640x480 image", require_complete=False)

    def test_non_positive_face(self):
        h.write_labels(self.dataset, [h.face_row(P1, w=0)])
        h.write_status(self.dataset, [h.status_row(P1, n_eye=0)])
        self.assert_error("non-positive size", require_complete=False)

    def test_more_than_one_face(self):
        h.write_labels(self.dataset, [h.face_row(P1), h.face_row(P1, x=10, w=50, h=60)])
        h.write_status(self.dataset, [h.status_row(P1, n_face=2, n_eye=0)])
        self.assert_error("2 face boxes", require_complete=False)

    def test_more_than_two_eyes(self):
        h.write_labels(self.dataset, [h.face_row(P1), h.eye_row(P1, "left", 270, 190),
                                      h.eye_row(P1, "right", 370, 190),
                                      h.eye_row(P1, "left", 300, 190)])
        h.write_status(self.dataset, [h.status_row(P1, n_eye=3)])
        report = self.check(require_complete=False)
        self.assertTrue(any("3 eye points" in e for e in report["errors"]))
        self.assertTrue(any("same eye side" in e for e in report["errors"]))

    def test_eye_outside_face(self):
        h.write_labels(self.dataset, [h.face_row(P1), h.eye_row(P1, "left", 20, 20)])
        h.write_status(self.dataset, [h.status_row(P1, n_eye=1)])
        self.assert_error("not inside the face box", require_complete=False)

    def test_eye_slightly_outside_is_within_margin(self):
        # face x=200..440; margin = 5% of 240 = 12 px
        h.write_labels(self.dataset, [h.face_row(P1), h.eye_row(P1, "left", 190, 190)])
        h.write_status(self.dataset, [h.status_row(P1, n_eye=1)])
        self.assertEqual(self.check(require_complete=False)["errors"], [])

    def test_invalid_side_and_order(self):
        h.write_labels(self.dataset, [h.face_row(P1), h.eye_row(P1, "up", 270, 190)])
        h.write_status(self.dataset, [h.status_row(P1, n_eye=1)])
        self.assert_error("must be left or right", require_complete=False)
        h.write_labels(self.dataset, [h.face_row(P1), h.eye_row(P1, "left", 380, 190),
                                      h.eye_row(P1, "right", 270, 190)])
        h.write_status(self.dataset, [h.status_row(P1, n_eye=2)])
        self.assert_error("left eye must be left of the right eye", require_complete=False)

    def test_eyes_without_face(self):
        h.write_labels(self.dataset, [h.eye_row(P1, "left", 270, 190)])
        h.write_status(self.dataset, [h.status_row(P1, "labelled", 0, 1)])
        self.assert_error("eye points but no face", require_complete=False)

    def test_background_must_have_nothing(self):
        h.write_labels(self.dataset, [h.face_row(BG)])
        h.write_status(self.dataset, [h.status_row(BG, "no_face", 1, 0)])
        self.assert_error("background-only image but has face or eye", require_complete=False)
        h.write_labels(self.dataset, [])
        h.write_status(self.dataset, [h.status_row(BG, "labelled", 0, 0)])
        self.assert_error("must have status no_face", require_complete=False)

    def test_uncertain_values(self):
        h.write_labels(self.dataset, [h.face_row(P1, unc=2)])
        h.write_status(self.dataset, [h.status_row(P1, n_eye=0)])
        self.assert_error("must be 0 or 1", require_complete=False)

    def test_labels_reference_real_images(self):
        h.write_labels(self.dataset, [h.face_row("ghost.png")])
        h.write_status(self.dataset, [h.status_row("ghost.png", n_eye=0)])
        report = self.check(require_complete=False)
        self.assertTrue(any("not in the dataset" in e for e in report["errors"]))

    def test_status_counts_agree(self):
        h.write_labels(self.dataset, [h.face_row(P1), h.eye_row(P1, "left", 270, 190)])
        h.write_status(self.dataset, [h.status_row(P1, n_eye=2)])
        self.assert_error("disagree with the labels", require_complete=False)

    def test_no_face_status_with_rows_and_labelled_without_face(self):
        h.write_labels(self.dataset, [h.face_row(P1)])
        h.write_status(self.dataset, [h.status_row(P1, "no_face", 1, 0)])
        self.assert_error("status no_face but also has annotations", require_complete=False)
        h.write_labels(self.dataset, [])
        h.write_status(self.dataset, [h.status_row(P1, "labelled", 0, 0)])
        self.assert_error("not exactly one face", require_complete=False)

    def test_label_rows_without_status_mean_interrupted_save(self):
        h.write_labels(self.dataset, [h.face_row(P1)])
        h.write_status(self.dataset, [])
        self.assert_error("interrupted save", require_complete=False)


class FileFormatTests(ValidatorCase):
    def test_wrong_header(self):
        path, _ = vl.label_paths(self.dataset, 1)
        path.write_text("filename,type\nx,y\n", encoding="utf-8")
        self.assert_error("differ from the required", require_complete=False)

    def test_short_row_names_the_line(self):
        self.good()
        path, _ = vl.label_paths(self.dataset, 1)
        with open(path, "a", encoding="utf-8", newline="") as handle:
            handle.write("normal_P01_001.png,face,1\n")
        report = self.check()
        self.assertTrue(any("line" in e and "wrong number of values" in e for e in report["errors"]))

    def test_bad_numbers_and_timestamps(self):
        h.write_labels(self.dataset, [h.face_row(P1, x="abc")])
        h.write_status(self.dataset, [])
        self.assert_error("not an integer", require_complete=False)
        h.write_labels(self.dataset, [])
        h.write_status(self.dataset, [h.status_row(P1, at="yesterday")])
        self.assert_error("ISO date-time", require_complete=False)
        h.write_status(self.dataset, [h.status_row(P1, at="2026-11-20T10:00:00")])
        self.assert_error("UTC offset", require_complete=False)

    def test_bom_is_accepted(self):
        self.good()
        for p in vl.label_paths(self.dataset, 1):
            p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())
        self.assertEqual(self.check()["errors"], [])


class WarningTests(ValidatorCase):
    def test_uncertain_and_few_eyes_warnings(self):
        h.write_labels(self.dataset, [h.face_row(P1, unc=1), h.eye_row(P1, "left", 270, 190, 1)])
        h.write_status(self.dataset, [h.status_row(P1, n_eye=1)])
        report = self.check(require_complete=False)
        self.assertEqual(report["errors"], [])
        text = " ".join(report["warnings"])
        self.assertIn("uncertain face", text)
        self.assertIn("uncertain eye (left)", text)
        self.assertIn("1 eye point", text)

    def test_unusual_face_size_warning(self):
        rows, status = [], []
        for n, w in ((P1, 240), (P2, 245), (P3, 238)):
            rows += [h.face_row(n, w=w)]
            status.append(h.status_row(n, n_eye=0))
        rows[2] = h.face_row(P3, x=200, w=60, h=60)
        h.write_labels(self.dataset, rows)
        h.write_status(self.dataset, status)
        report = self.check(require_complete=False)
        self.assertTrue(any("unusual for condition" in w for w in report["warnings"]))


class RepeatPassValidationTests(ValidatorCase):
    def setUp(self):
        super().setUp()
        self._tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp2.cleanup)
        self.big = h.make_dataset(self._tmp2.name, n_person=20, n_bg=2)

    def test_sample_is_14_person_images_and_deterministic(self):
        meta = vl.load_metadata(self.big)
        person = [n for n, m in meta.items() if not vl.is_background(m)]
        a = vl.repeat_sample(person)
        self.assertEqual(len(a), 14)
        self.assertTrue(set(a) <= set(person))
        self.assertEqual(a, vl.repeat_sample(list(reversed(person))))
        self.assertNotEqual(a, vl.repeat_sample(person, seed=vl.REPEAT_SEED + 1))

    def test_pass2_rejects_images_outside_sample_and_background(self):
        meta = vl.load_metadata(self.big)
        person = [n for n, m in meta.items() if not vl.is_background(m)]
        sample = set(vl.repeat_sample(person))
        outside = next(n for n in person if n not in sample)
        h.write_labels(self.big, [h.face_row(outside)], pass_no=2)
        h.write_status(self.big, [h.status_row(outside, n_eye=0, pass_no=2)], pass_no=2)
        report = vl.validate_dataset(self.big, 2, require_complete=False)
        self.assertTrue(any("repeat-check images" in e for e in report["errors"]))

    def test_pass2_requires_pass_number_in_status(self):
        meta = vl.load_metadata(self.big)
        name = vl.repeat_sample([n for n, m in meta.items() if not vl.is_background(m)])[0]
        h.write_labels(self.big, [h.face_row(name)], pass_no=2)
        h.write_status(self.big, [h.status_row(name, n_eye=0, pass_no=1)], pass_no=2)
        report = vl.validate_dataset(self.big, 2, require_complete=False)
        self.assertTrue(any("should be 2" in e for e in report["errors"]))


class OrderTests(unittest.TestCase):
    def test_order_is_deterministic_and_independent_of_input_order(self):
        names = [f"img_{i:03d}.png" for i in range(40)]
        a = vl.annotation_order(names)
        self.assertEqual(a, vl.annotation_order(list(reversed(names))))
        self.assertEqual(sorted(a), sorted(names))
        self.assertNotEqual(a, sorted(names))
        self.assertNotEqual(a, vl.annotation_order(names, seed=1))

    def test_seeds_are_recorded_constants(self):
        self.assertEqual((vl.ORDER_SEED, vl.REPEAT_SEED, vl.REPEAT_SAMPLE_SIZE),
                         (20261120, 20261121, 14))


class ReadOnlyTests(ValidatorCase):
    def test_validator_changes_nothing(self):
        self.good()
        before = {p.name: p.read_bytes() for p in self.dataset.iterdir() if p.is_file()}
        vl.validate_dataset(self.dataset)
        after = {p.name: p.read_bytes() for p in self.dataset.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_validator_never_uses_a_detector(self):
        source = Path(vl.__file__).read_text(encoding="utf-8")
        for word in ("src.detector", "CascadeClassifier", "detectMultiScale", "EyeDetector"):
            self.assertNotIn(word, source)

    def test_validator_never_creates_the_lock_file(self):
        self.good()
        before = tree(self.dataset)
        vl.validate_dataset(self.dataset)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vl.main(["--dataset", str(self.dataset)]), 0)
        self.assertFalse(os.path.lexists(vl.lock_path(self.dataset)))
        self.assertEqual(tree(self.dataset), before)


def tree(folder):
    """Every path under folder -> bytes (files) or 'dir' (directories): a full snapshot."""
    out = {}
    for p in sorted(Path(folder).rglob("*")):
        out[str(p.relative_to(folder))] = p.read_bytes() if p.is_file() else "dir"
    return out


MALFORMED_MARKERS = {
    "empty": b"",
    "pass line only": b"pass=1\n",
    "phase line only": b"phase=labels\n",
    "lines swapped": b"phase=labels\npass=1\n",
    "unknown phase": b"pass=1\nphase=finish\n",
    "pass 3": b"pass=3\nphase=labels\n",
    "CRLF": b"pass=1\r\nphase=labels\r\n",
    "truncated": b"pass=1\nph",
    "too long": b"pass=1\nphase=labels\n" + b" " * 100,
    "invalid UTF-8": b"\xff\xfe\x00pass",
    "NUL": b"pass=1\x00phase=labels\n",
    "old single field": b"pass=1",
}


class RestoreMarkerTests(ValidatorCase):
    """.restore_in_progress: detected by existence alone, reported, never changed."""

    def setUp(self):
        super().setUp()
        self.good()
        self.marker = vl.restore_marker_path(self.dataset)
        self.clean = self.check()

    def test_format_is_exact_and_round_trips_for_every_pass_and_phase(self):
        self.assertEqual(vl.format_restore_marker(1, "labels"), b"pass=1\nphase=labels\n")
        self.assertEqual(vl.MARKER_PHASES, ("backup", "labels", "status"))
        for pass_no in (1, 2):
            for phase in vl.MARKER_PHASES:
                self.marker.write_bytes(vl.format_restore_marker(pass_no, phase))
                self.assertEqual(vl.read_restore_marker(self.dataset), (True, pass_no, phase))
        for bad in ((3, "labels"), (1, "done"), ("1", "labels")):
            with self.assertRaises(ValueError):
                vl.format_restore_marker(*bad)

    def test_absent_marker_leaves_validation_unchanged(self):
        self.assertEqual(vl.read_restore_marker(self.dataset), (False, None, None))
        self.assertEqual(self.clean["errors"], [])
        self.marker.write_bytes(vl.format_restore_marker(1, "labels"))
        self.marker.unlink()
        self.assertEqual(self.check(), self.clean)

    def test_readable_marker_is_an_error_naming_pass_and_phase_and_keeps_the_rest(self):
        self.marker.write_bytes(vl.format_restore_marker(2, "status"))
        before = tree(self.dataset)
        report = self.check()
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn(".restore_in_progress", report["errors"][0])
        self.assertIn("restore of pass 2 was interrupted", report["errors"][0])
        self.assertIn("restoring the status file", report["errors"][0])
        self.assertEqual(report["stats"], self.clean["stats"])            # statistics kept
        self.assertEqual(tree(self.dataset), before)                       # marker untouched

    def test_malformed_markers_are_errors_with_the_generic_message(self):
        for label, data in MALFORMED_MARKERS.items():
            with self.subTest(marker=label):
                self.marker.write_bytes(data)
                before = tree(self.dataset)
                self.assertEqual(vl.read_restore_marker(self.dataset), (True, None, None))
                report = self.check()
                self.assertEqual(len(report["errors"]), 1, report["errors"])
                error = report["errors"][0]
                self.assertIn("a restore was interrupted", error)
                self.assertIn("affected pass cannot be determined", error)
                self.assertIn("manual recovery is required", error)
                self.assertIn('README section "Interrupted restore"', error)
                self.assertNotIn("pass 1", error)
                self.assertNotIn("pass 2", error)
                self.assertEqual(report["stats"], self.clean["stats"])
                self.assertEqual(tree(self.dataset), before)               # never changed

    def test_directory_and_unreadable_markers_still_count(self):
        self.marker.mkdir()
        report = self.check()
        self.assertIn("affected pass cannot be determined", report["errors"][0])
        self.assertTrue(self.marker.is_dir())
        self.marker.rmdir()
        self.marker.write_bytes(vl.format_restore_marker(1, "labels"))
        real_open = open

        def refuse(path, *args, **kw):
            if Path(path) == self.marker:
                raise PermissionError(13, "denied")
            return real_open(path, *args, **kw)

        with mock.patch("builtins.open", side_effect=refuse):
            self.assertEqual(vl.read_restore_marker(self.dataset), (True, None, None))
        self.assertEqual(self.marker.read_bytes(), vl.format_restore_marker(1, "labels"))

    def test_broken_link_marker_counts(self):
        try:
            os.symlink(self.dataset / "nowhere", self.marker)
        except (OSError, NotImplementedError):
            self.skipTest("creating symbolic links is not permitted here")
        self.assertEqual(vl.read_restore_marker(self.dataset), (True, None, None))
        self.assertIn("affected pass cannot be determined", self.check()["errors"][0])
        self.assertTrue(os.path.lexists(self.marker))

    def test_command_line_fails_while_a_marker_exists(self):
        self.marker.write_bytes(b"")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = vl.main(["--dataset", str(self.dataset)])
        self.assertEqual(code, 1)
        self.assertIn(".restore_in_progress", out.getvalue())
        self.assertIn("RESULT: FAILED", out.getvalue())
        self.assertTrue(self.marker.exists())
        self.assertFalse(os.path.lexists(vl.lock_path(self.dataset)))


if __name__ == "__main__":
    unittest.main()
