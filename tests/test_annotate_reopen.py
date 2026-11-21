"""Tests for correcting saved annotations (keys f / l / r, Annotation.reopen) and for the
consistency check of saved data.

Only synthetic images in temporary folders are used. No real dataset file, annotation or
backup is read or written, and no window is opened (OpenCV's window functions are replaced by
the scripted fake from test_annotate_dataset).
"""
import contextlib
import io
import re
import subprocess
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotation_helpers as h  # noqa: E402
from test_annotate_dataset import (ENTER, QUIT, SCALE, STOP, DatasetCase, Harness,  # noqa: E402
                                   click, drag, key)

ad, vl = h.ad, h.vl
AnnotationError = ad.AnnotationError
NEW_FACE = (190, 90, 450, 370)            # contains both synthetic eyes
SMALL_FACE = (300, 100, 440, 360)         # excludes the left eye (270, 190)


def saved_annotation(**flags):
    """A complete annotation as it is after loading from disk (empty undo history)."""
    a = h.full_annotation()
    rows, status = a.records("x.png")
    faces = [{"x": int(r["x"]), "y": int(r["y"]), "w": int(r["w"]), "h": int(r["h"]),
              "uncertain": flags.get("face", 0)} for r in rows if r["object_type"] == "face"]
    eyes = [{"x": int(r["x"]), "y": int(r["y"]), "side": r["side"],
             "uncertain": flags.get(r["side"], 0)}
            for r in rows if r["object_type"] == "eye"]
    return ad.Annotation.from_saved(faces, eyes, {"status": "labelled"})


# ================================================================ the data class
class ReopenAnnotationTests(unittest.TestCase):
    def test_loaded_annotation_has_no_history_and_blocks_direct_edits(self):
        a = saved_annotation()
        for action in (lambda: a.set_face(*NEW_FACE), a.undo, a.hide_eye, a.mark_no_face):
            with self.assertRaises(AnnotationError):
                action()

    def test_reopen_face_clears_only_the_face(self):
        a = saved_annotation(left=1)
        a.reopen("face")
        self.assertIsNone(a.face)
        self.assertEqual(a.target(), "face")
        self.assertEqual((a.eyes["left"], a.eyes["right"]), (h.LEFT_EYE, h.RIGHT_EYE))
        self.assertEqual(a.uncertain["left"], 1)                       # untouched

    def test_redrawn_face_replaces_the_old_one_and_keeps_the_eyes(self):
        a = saved_annotation()
        a.reopen("face")
        a.set_face(*NEW_FACE)
        self.assertEqual(a.face, (190, 90, 260, 280))
        self.assertTrue(a.is_complete())
        self.assertEqual((a.eyes["left"], a.eyes["right"]), (h.LEFT_EYE, h.RIGHT_EYE))

    def test_face_that_would_exclude_an_eye_is_rejected_and_changes_nothing(self):
        a = saved_annotation()
        a.reopen("face")
        before = a.fingerprint()
        with self.assertRaises(AnnotationError) as caught:
            a.set_face(*SMALL_FACE)
        self.assertIn("left eye would be outside the new face box", str(caught.exception))
        self.assertIn("press l", str(caught.exception))
        self.assertEqual(a.fingerprint(), before)
        self.assertIsNone(a.face)

    def test_eye_can_be_redone_while_the_face_is_cleared_to_fit_a_smaller_box(self):
        a = saved_annotation()
        a.reopen("face")
        a.reopen("left")                                               # allowed with no face
        a.set_face(*SMALL_FACE)
        a.place_eye(320, 190)
        self.assertEqual(a.eyes["left"], (320, 190))
        self.assertTrue(a.is_complete())

    def test_reopen_left_eye_resets_only_its_own_flag(self):
        a = saved_annotation(left=1, right=1, face=1)
        a.reopen("left")
        self.assertEqual((a.eyes["left"], a.uncertain["left"]), (None, 0))
        self.assertEqual((a.eyes["right"], a.uncertain["right"], a.uncertain["face"]),
                         (h.RIGHT_EYE, 1, 1))
        self.assertEqual(a.target(), "left")

    def test_replacement_eye_still_obeys_the_existing_rules(self):
        a = saved_annotation()
        a.reopen("left")
        with self.assertRaises(AnnotationError):
            a.place_eye(5, 5)                                          # outside the face
        with self.assertRaises(AnnotationError):
            a.place_eye(380, 190)                                      # right of the right eye
        a.place_eye(280, 200)
        self.assertEqual(a.eyes["left"], (280, 200))
        b = saved_annotation()
        b.reopen("right")
        with self.assertRaises(AnnotationError):
            b.place_eye(260, 190)                                      # left of the left eye
        b.place_eye(380, 195)
        self.assertEqual(b.eyes["right"], (380, 195))

    def test_hidden_eye_can_be_reopened_and_placed(self):
        a = ad.Annotation()
        a.set_face(*h.FACE_DRAG)
        a.hide_eye()
        a.place_eye(*h.RIGHT_EYE)
        self.assertTrue(a.hidden["left"])
        a.reopen("left")
        self.assertFalse(a.hidden["left"])
        self.assertEqual(a.target(), "left")
        a.place_eye(*h.LEFT_EYE)
        self.assertEqual((a.eyes["left"], a.hidden["left"]), (h.LEFT_EYE, False))

    def test_visible_eye_can_be_changed_to_hidden(self):
        a = saved_annotation()
        a.reopen("left")
        a.hide_eye()
        rows, status = a.records("x.png")
        self.assertEqual(status["n_eye"], 1)
        self.assertEqual([r["side"] for r in rows if r["object_type"] == "eye"], ["right"])

    def test_reopen_errors(self):
        with self.assertRaises(AnnotationError):
            ad.Annotation().reopen("face")                             # nothing drawn
        a = ad.Annotation()
        a.set_face(*h.FACE_DRAG)
        with self.assertRaises(AnnotationError) as caught:
            a.reopen("left")                                           # not placed yet
        self.assertIn("not placed yet", str(caught.exception))
        with self.assertRaises(AnnotationError):
            a.reopen("nose")
        n = h.no_face_annotation()
        with self.assertRaises(AnnotationError) as caught:
            n.reopen("face")
        self.assertIn("no_face", str(caught.exception))
        self.assertEqual(len(a._history), 1)                           # failures push nothing

    def test_undo_restores_the_exact_previous_annotation_including_flags(self):
        a = saved_annotation(face=1, left=1, right=0)
        original = a.fingerprint()
        a.reopen("left")
        a.place_eye(280, 200)
        a.reopen("face")
        self.assertNotEqual(a.fingerprint(), original)
        a.undo()                                                       # the face comes back
        self.assertEqual((a.face, a.uncertain["face"]), ((200, 100, 240, 260), 1))
        a.undo()                                                       # the placed eye is removed
        self.assertIsNone(a.eyes["left"])
        a.undo()                                                       # the original left eye
        self.assertEqual((a.eyes["left"], a.uncertain["left"]), (h.LEFT_EYE, 1))
        self.assertEqual(a.fingerprint(), original)                    # clean again
        with self.assertRaises(AnnotationError):
            a.undo()                                                   # never past what was saved

    def test_half_corrected_annotation_cannot_be_recorded(self):
        a = saved_annotation()
        a.reopen("right")
        with self.assertRaises(AnnotationError) as caught:
            a.records("x.png")
        self.assertIn("right eye", str(caught.exception))

    def test_reopening_everything_allows_no_face(self):
        a = saved_annotation()
        for obj in ("face", "left", "right"):
            a.reopen(obj)
        self.assertTrue(a.is_empty())
        a.mark_no_face()
        rows, status = a.records("x.png")
        self.assertEqual((rows, status["status"], status["n_face"], status["n_eye"]),
                         ([], "no_face", 0, 0))

    def test_inconsistent_saved_rows_give_a_clear_error_not_an_index_error(self):
        eye = {"x": 270, "y": 190, "side": "left", "uncertain": 0}
        face = {"x": 200, "y": 100, "w": 240, "h": 260, "uncertain": 0}
        for faces, eyes in (([], []), ([face, face], []), ([face], [eye, eye]),
                            ([face], [eye, dict(eye, side="right"), dict(eye, side="left")])):
            with self.subTest(faces=len(faces), eyes=len(eyes)):
                with self.assertRaises(AnnotationError) as caught:
                    ad.Annotation.from_saved(faces, eyes, {"status": "labelled"})
                self.assertIn("inconsistent", str(caught.exception))


# ==================================================================== file storage
class ReopenStorageTests(DatasetCase):
    n_person, n_bg = 4, 1

    def setUp(self):
        super().setUp()
        h.annotate_everything(self.dataset)
        self.name = self.persons[0]
        self.later = h.T0 + timedelta(hours=1)

    def files(self, pass_no=1):
        return {p.name: p.read_bytes() for p in self.paths(pass_no)}

    def loaded(self, name=None):
        data = ad.load_pass(self.dataset, 1)
        name = name or self.name
        return ad.Annotation.from_saved(data["faces"].get(name, []), data["eyes"].get(name, []),
                                        data["status"][name])

    def others(self, blob):
        return [line for line in blob.decode().splitlines()
                if not line.startswith(self.name + ",")]

    def assert_only_this_image_changed(self, before):
        after = self.files()
        for fname in before:
            self.assertEqual(self.others(before[fname]), self.others(after[fname]))

    def test_corrected_face_is_saved_and_persists_after_reopening(self):
        before = self.files()
        a = self.loaded()
        a.reopen("face")
        a.set_face(*NEW_FACE)
        self.assertEqual(ad.save_image(self.dataset, 1, self.name, a, False, self.later), "saved")
        face = ad.load_pass(self.dataset, 1)["faces"][self.name][0]       # a fresh read
        self.assertEqual((face["x"], face["y"], face["w"], face["h"]), (190, 90, 260, 280))
        reopened = self.loaded()
        self.assertEqual(reopened.face, (190, 90, 260, 280))
        self.assertEqual((reopened.eyes["left"], reopened.eyes["right"]), (h.LEFT_EYE, h.RIGHT_EYE))
        self.assert_only_this_image_changed(before)
        self.assertEqual(vl.validate_dataset(self.dataset, 1)["errors"], [])

    def test_corrected_eye_is_saved_and_persists(self):
        before = self.files()
        a = self.loaded()
        a.reopen("left")
        a.place_eye(280, 200)
        ad.save_image(self.dataset, 1, self.name, a, False, self.later)
        reopened = self.loaded()
        self.assertEqual((reopened.eyes["left"], reopened.eyes["right"]), ((280, 200), h.RIGHT_EYE))
        self.assertEqual(reopened.face, (200, 100, 240, 260))
        self.assert_only_this_image_changed(before)
        self.assertEqual(vl.validate_dataset(self.dataset, 1)["errors"], [])

    def test_visible_eye_changed_to_hidden_removes_its_row(self):
        a = self.loaded()
        a.reopen("right")
        a.hide_eye()
        ad.save_image(self.dataset, 1, self.name, a, False, self.later)
        data = ad.load_pass(self.dataset, 1)
        self.assertEqual([e["side"] for e in data["eyes"][self.name]], ["left"])
        st = data["status"][self.name]
        self.assertEqual((st["n_face"], st["n_eye"]), (1, 1))
        self.assertEqual(vl.validate_dataset(self.dataset, 1)["errors"], [])

    def test_edit_backup_preserves_the_original_files_exactly(self):
        original = self.files()
        ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(minutes=30), check_git=False)
        session_backups = {b.name: b.read_bytes() for b in self.backups()}
        a = self.loaded()
        a.reopen("face")
        a.set_face(*NEW_FACE)
        ad.save_image(self.dataset, 1, self.name, a, False, self.later)
        edit = [b for b in self.backups() if "_edit" in b.name]
        self.assertEqual(len(edit), 2)                                    # labels + status
        self.assertEqual(sorted(b.read_bytes() for b in edit), sorted(original.values()))
        for name, content in session_backups.items():                    # older backups untouched
            self.assertEqual((ad.backups_dir(self.dataset) / name).read_bytes(), content)
        self.assertNotEqual(self.files(), original)

    def test_each_actual_edit_keeps_its_own_original(self):
        first = self.files()
        a = self.loaded()
        a.reopen("left")
        a.place_eye(280, 200)
        ad.save_image(self.dataset, 1, self.name, a, False, self.later)
        second = self.files()
        b = self.loaded()
        b.reopen("right")
        b.place_eye(380, 195)
        ad.save_image(self.dataset, 1, self.name, b, False, self.later + timedelta(minutes=1))
        kept = sorted(p.read_bytes() for p in self.backups() if "_edit" in p.name)
        self.assertEqual(kept, sorted(list(first.values()) + list(second.values())))

    def test_identical_replacement_makes_no_backup_and_no_change(self):
        before = self.files()
        a = self.loaded()
        a.reopen("left")
        a.place_eye(*h.LEFT_EYE)                                           # the same point again
        self.assertEqual(ad.save_image(self.dataset, 1, self.name, a, False, self.later),
                         "unchanged")
        self.assertEqual(self.files(), before)
        self.assertEqual(self.backups(), [])

    def test_changing_only_a_flag_after_replacement_is_a_real_edit(self):
        a = self.loaded()
        a.reopen("face")
        a.set_face(*h.FACE_DRAG)                                           # same box
        a.toggle_uncertain()
        self.assertEqual(ad.save_image(self.dataset, 1, self.name, a, False, self.later), "saved")
        self.assertEqual(self.loaded().uncertain["face"], 1)

    def test_conversion_to_no_face_removes_the_rows(self):
        before = self.files()
        a = self.loaded()
        for obj in ("face", "left", "right"):
            a.reopen(obj)
        a.mark_no_face()
        ad.save_image(self.dataset, 1, self.name, a, False, self.later)
        data = ad.load_pass(self.dataset, 1)
        self.assertEqual(data["status"][self.name]["status"], "no_face")
        self.assertNotIn(self.name, data["faces"])
        self.assertNotIn(self.name, data["eyes"])
        self.assert_only_this_image_changed(before)
        self.assertTrue(any("_edit" in b.name for b in self.backups()))

    def test_pass2_correction_never_touches_pass1(self):
        ad.save_image(self.dataset, 2, self.name, h.full_annotation(), False, self.later)
        pass1 = self.files(1)
        pass2_original = self.files(2)
        data = ad.load_pass(self.dataset, 2)
        a = ad.Annotation.from_saved(data["faces"][self.name], data["eyes"][self.name],
                                     data["status"][self.name])
        a.reopen("left")
        a.place_eye(280, 200)
        ad.save_image(self.dataset, 2, self.name, a, False, self.later + timedelta(minutes=1))
        self.assertEqual(self.files(1), pass1)
        self.assertNotEqual(self.files(2), pass2_original)
        edits = [b for b in self.backups(2) if "_edit" in b.name]
        self.assertEqual(sorted(b.read_bytes() for b in edits), sorted(pass2_original.values()))
        self.assertTrue(all("_pass2_" in b.name for b in edits))
        self.assertEqual([b for b in self.backups(1) if "_edit" in b.name], [])

    def test_half_corrected_annotation_is_never_written(self):
        before, backups = self.files(), self.backups()
        a = self.loaded()
        a.reopen("left")
        with self.assertRaises(AnnotationError):
            ad.save_image(self.dataset, 1, self.name, a, False, self.later)
        self.assertEqual((self.files(), self.backups()), (before, backups))


# =========================================== inconsistent saved data (load-time check)
class SavedDataConsistencyTests(DatasetCase):
    n_person, n_bg = 3, 1

    def setUp(self):
        super().setUp()
        h.annotate_everything(self.dataset)
        self.name = self.persons[0]
        self.status_path = self.paths()[1]
        self.label_path = self.paths()[0]

    def snapshot(self):
        return [p.read_bytes() for p in self.paths()]

    def edit(self, path, old, new):
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")

    def assert_refused(self, *fragments):
        before = self.snapshot()
        with self.assertRaises(AnnotationError) as caught:
            ad.load_pass(self.dataset, 1)
        message = str(caught.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)
        self.assertIn("Nothing was changed", message)
        self.assertNotIn(self.name, message)                               # the tool is blind
        self.assertEqual(self.snapshot(), before)
        return message

    def test_clean_data_loads(self):
        self.assertEqual(ad.load_pass(self.dataset, 1)["orphans"], [])

    def test_labelled_status_without_face_row(self):
        lines = [ln for ln in self.label_path.read_text().splitlines(True)
                 if not ln.startswith(self.name + ",face,")]
        self.label_path.write_text("".join(lines), newline="")
        self.assert_refused("labelled but 0 face rows")

    def test_no_face_status_with_label_rows(self):
        self.edit(self.status_path, f"{self.name},labelled,1,2", f"{self.name},no_face,0,0")
        self.assert_refused("status no_face but 1 face and 2 eye label rows")

    def test_count_mismatch_between_status_and_labels(self):
        self.edit(self.status_path, f"{self.name},labelled,1,2", f"{self.name},labelled,1,1")
        message = self.assert_refused("status says 1 face and 1 eye rows")
        self.assertRegex(message, r"line \d+")

    def test_duplicate_eye_side(self):
        lines = self.label_path.read_text().splitlines(True)
        text = "".join(ln.replace(",right,", ",left,") if ln.startswith(self.name + ",eye,")
                       else ln for ln in lines)
        self.label_path.write_text(text, newline="")
        self.assert_refused("eye rows with sides")

    def test_save_image_and_session_start_refuse_inconsistent_data(self):
        self.edit(self.status_path, f"{self.name},labelled,1,2", f"{self.name},labelled,1,1")
        before = self.snapshot()
        with self.assertRaises(AnnotationError):
            ad.save_image(self.dataset, 1, self.persons[1], h.full_annotation(), False,
                          h.T0 + timedelta(hours=1))
        with self.assertRaises(AnnotationError):
            ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(hours=2), check_git=False)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.backups(), [])                               # nothing was backed up

    def test_orphans_from_an_interrupted_new_save_are_still_only_orphans(self):
        extra = self.persons[2]
        data = ad.load_pass(self.dataset, 1)
        lines = [ln for ln in self.status_path.read_text().splitlines(True)
                 if not ln.startswith(extra + ",")]
        self.status_path.write_text("".join(lines), newline="")
        self.assertEqual(ad.load_pass(self.dataset, 1)["orphans"], [extra])
        self.assertIn(extra, data["status"])

    def test_interrupted_edit_is_detected_at_the_next_start(self):
        """An edit that changes the number of eye rows, interrupted before the status write."""
        a = ad.Annotation.from_saved(*self._rows(), {"status": "labelled"})
        a.reopen("left")
        a.hide_eye()
        real = ad.write_atomic
        calls = []

        def crash_on_status(path, data):
            calls.append(path.name)
            if len(calls) == 2:
                raise KeyboardInterrupt
            real(path, data)
        with mock.patch.object(ad, "write_atomic", side_effect=crash_on_status):
            with self.assertRaises(KeyboardInterrupt):
                ad.save_image(self.dataset, 1, self.name, a, False, h.T0 + timedelta(hours=1))
        self.assert_refused("status says 1 face and 2 eye rows")
        edit = [b for b in self.backups() if "_edit" in b.name]
        self.assertEqual(len(edit), 2)                                     # the originals exist

    def _rows(self):
        data = ad.load_pass(self.dataset, 1)
        return data["faces"][self.name], data["eyes"][self.name]

    # -- no_face rows: the stored counts must be 0 and 0 as well
    def test_valid_no_face_rows_pass(self):
        bg = self.bgs[0]
        data = ad.load_pass(self.dataset, 1)
        st = data["status"][bg]
        self.assertEqual((st["status"], st["n_face"], st["n_eye"]), ("no_face", 0, 0))
        self.assertNotIn(bg, data["faces"])
        self.assertNotIn(bg, data["eyes"])
        self.assertEqual(ad.saved_data_problems(
            [{"filename": "a.png", "status": "no_face", "n_face": 0, "n_eye": 0}], {}, {}), [])

    def test_no_face_status_with_a_nonzero_face_count_is_refused(self):
        bg = self.bgs[0]
        self.edit(self.status_path, f"{bg},no_face,0,0", f"{bg},no_face,1,0")
        message = self.assert_refused("status no_face but it says 1 face and 0 eye rows")
        self.assertRegex(message, r"line \d+")

    def test_no_face_status_with_a_nonzero_eye_count_is_refused(self):
        bg = self.bgs[0]
        self.edit(self.status_path, f"{bg},no_face,0,0", f"{bg},no_face,0,2")
        self.assert_refused("status no_face but it says 0 face and 2 eye rows (must be 0 and 0)")

    def test_no_face_status_with_both_counts_nonzero_is_refused(self):
        bg = self.bgs[0]
        self.edit(self.status_path, f"{bg},no_face,0,0", f"{bg},no_face,1,2")
        self.assert_refused("it says 1 face and 2 eye rows")

    def test_no_face_status_with_label_rows_and_counts_reports_both_problems(self):
        self.edit(self.status_path, f"{self.name},labelled,1,2", f"{self.name},no_face,1,2")
        message = self.assert_refused("status no_face but 1 face and 2 eye label rows",
                                      "status no_face but it says 1 face and 2 eye rows")
        self.assertEqual(message.count("line "), 2)

    def test_the_counts_check_is_pure_and_names_each_bad_line(self):
        rows = [{"filename": "a.png", "status": "no_face", "n_face": 0, "n_eye": 1},
                {"filename": "b.png", "status": "no_face", "n_face": 0, "n_eye": 0},
                {"filename": "c.png", "status": "no_face", "n_face": 1, "n_eye": 0}]
        problems = ad.saved_data_problems(rows, {}, {})
        self.assertEqual(len(problems), 2)
        self.assertTrue(problems[0].startswith("line 2:"))
        self.assertTrue(problems[1].startswith("line 4:"))


# ================================================ recovery and the pass-2 gate
class RestoreAndPass2GateTests(DatasetCase):
    """Synthetic temporary datasets only: nothing under the real dataset/ is touched."""
    n_person, n_bg = 4, 1

    def setUp(self):
        super().setUp()
        h.annotate_everything(self.dataset)
        self.status_path = self.paths()[1]
        self.name = self.persons[0]

    def snapshot(self, pass_no=1):
        return [p.read_bytes() for p in self.paths(pass_no)]

    def make_inconsistent(self):
        text = self.status_path.read_text(encoding="utf-8")
        old = f"{self.name},labelled,1,2"
        self.assertIn(old, text)
        self.status_path.write_text(text.replace(old, f"{self.name},labelled,1,1", 1),
                                    encoding="utf-8", newline="")

    def run_cli(self, *args, answer="yes"):
        out = io.StringIO()
        with mock.patch("builtins.input", return_value=answer), \
                contextlib.redirect_stdout(out):
            code = ad.main(["--dataset", str(self.dataset), *args])
        return code, out.getvalue()

    # -- --restore-latest-backup despite the inconsistency
    def test_restore_recovers_an_inconsistent_dataset(self):
        ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(hours=1), check_git=False)
        good = self.snapshot()
        session_backups = {b.name: b.read_bytes() for b in self.backups()}
        self.make_inconsistent()
        damaged = self.snapshot()
        self.assertNotEqual(damaged, good)
        with self.assertRaises(AnnotationError) as caught:                 # normal loading refuses
            ad.load_pass(self.dataset, 1)
        self.assertIn("does not match", str(caught.exception))
        self.assertIn("--restore-latest-backup", str(caught.exception))
        with self.assertRaises(AnnotationError):
            ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(hours=2), check_git=False)
        self.assertEqual(self.snapshot(), damaged)                         # refusal changed nothing
        restored = ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=3))
        self.assertEqual(sorted(restored), ["annotation_status.csv", "labels.csv"])
        self.assertEqual(self.snapshot(), good)                            # the original bytes
        data = ad.load_pass(self.dataset, 1)                               # loads again
        self.assertEqual((len(data["status"]), data["orphans"]), (5, []))
        self.assertEqual(vl.validate_dataset(self.dataset, 1)["errors"], [])
        for name, content in session_backups.items():                      # the backup is intact
            self.assertEqual((ad.backups_dir(self.dataset) / name).read_bytes(), content)
        kept = [b for b in ad.backups_dir(self.dataset).iterdir() if "prerestore" in b.name]
        self.assertIn(damaged[1], [b.read_bytes() for b in kept])          # damaged copy is kept

    def test_restore_command_line_recovers_and_a_no_answer_changes_nothing(self):
        ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(hours=1), check_git=False)
        good = self.snapshot()
        self.make_inconsistent()
        damaged = self.snapshot()
        code, output = self.run_cli("--restore-latest-backup", answer="no")
        self.assertEqual(code, 1)
        self.assertIn("not restored", output)
        self.assertEqual(self.snapshot(), damaged)
        code, output = self.run_cli()                                      # normal start refuses
        self.assertEqual(code, 1)
        self.assertIn("does not match", output)
        self.assertEqual(self.snapshot(), damaged)
        code, output = self.run_cli("--restore-latest-backup")
        self.assertEqual(code, 0)
        self.assertIn("restored:", output)
        self.assertEqual(self.snapshot(), good)
        self.assertEqual(ad.load_pass(self.dataset, 1)["orphans"], [])

    def test_restore_gives_back_the_last_valid_state_not_the_damaged_one(self):
        """A newer backup made BEFORE the damage is what gets restored."""
        ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(hours=1), check_git=False)
        a = ad.Annotation.from_saved(*self._rows(), {"status": "labelled"})
        a.reopen("left")
        a.place_eye(280, 200)
        ad.save_image(self.dataset, 1, self.name, a, False, h.T0 + timedelta(hours=2))
        before_edit = [b.read_bytes() for b in self.backups() if "_edit" in b.name]
        self.assertEqual(len(before_edit), 2)
        edited = self.snapshot()
        self.make_inconsistent()
        ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=3))
        self.assertEqual(sorted(self.snapshot()), sorted(before_edit))      # newest valid backup
        self.assertNotEqual(self.snapshot(), edited)
        self.assertEqual(ad.load_pass(self.dataset, 1)["orphans"], [])

    def _rows(self):
        data = ad.load_pass(self.dataset, 1)
        return data["faces"][self.name], data["eyes"][self.name]

    # -- the pass-2 gate
    def test_valid_pass1_permits_pass2(self):
        later = h.T0 + timedelta(days=3)
        ad.check_pass2_allowed(self.dataset, later)                        # no exception
        order = ad.build_order(self.dataset, 2, later)
        self.assertTrue(order)
        self.assertTrue(all(not vl.is_background(self.meta[n]) for n in order))

    def test_inconsistent_pass1_blocks_pass2_clearly_and_changes_nothing(self):
        self.make_inconsistent()
        pass1 = self.snapshot()
        later = h.T0 + timedelta(days=3)
        for call in (lambda: ad.check_pass2_allowed(self.dataset, later),
                     lambda: ad.build_order(self.dataset, 2, later),
                     lambda: ad.start_session_checks(self.dataset, 2, later, check_git=False)):
            with self.assertRaises(AnnotationError) as caught:
                call()
            message = str(caught.exception)
            self.assertTrue("pass 2 needs a complete, valid pass 1" in message
                            or "does not match" in message, message)
            self.assertIn("validate_labels.py", message)                   # says what to run
        self.assertEqual(self.snapshot(), pass1)                           # pass 1 untouched
        self.assertFalse(any(p.exists() for p in self.paths(2)))           # no pass-2 files
        self.assertEqual(self.backups(1), [])
        self.assertEqual(self.backups(2), [])

    def test_pass1_with_a_no_face_count_error_also_blocks_pass2(self):
        bg = self.bgs[0]
        text = self.status_path.read_text(encoding="utf-8")
        self.status_path.write_text(text.replace(f"{bg},no_face,0,0", f"{bg},no_face,1,0", 1),
                                    encoding="utf-8", newline="")
        pass1 = self.snapshot()
        with self.assertRaises(AnnotationError):
            ad.check_pass2_allowed(self.dataset, h.T0 + timedelta(days=3))
        self.assertEqual(self.snapshot(), pass1)
        self.assertFalse(any(p.exists() for p in self.paths(2)))

    def test_incomplete_pass1_still_blocks_pass2_and_nothing_is_written(self):
        lines = self.status_path.read_text(encoding="utf-8").splitlines(True)
        kept = [ln for ln in lines if not ln.startswith(self.persons[1] + ",")]
        label_path = self.paths()[0]
        keep_labels = [ln for ln in label_path.read_text(encoding="utf-8").splitlines(True)
                       if not ln.startswith(self.persons[1] + ",")]
        self.status_path.write_text("".join(kept), encoding="utf-8", newline="")
        label_path.write_text("".join(keep_labels), encoding="utf-8", newline="")
        pass1 = self.snapshot()
        with self.assertRaises(AnnotationError):
            ad.check_pass2_allowed(self.dataset, h.T0 + timedelta(days=3))
        self.assertEqual(self.snapshot(), pass1)
        self.assertFalse(any(p.exists() for p in self.paths(2)))


# ===================================== the pass-2 gate message is blind (no filenames)
class Pass2GateBlindMessageTests(DatasetCase):
    """The gate reports WHAT is wrong with pass 1 but never WHICH image or which folder.
    Synthetic temporary data only."""
    n_person, n_bg = 4, 1

    def setUp(self):
        super().setUp()
        h.annotate_everything(self.dataset)
        self.label_path, self.status_path = self.paths()
        self.name = self.persons[0]
        self.bg = self.bgs[0]
        self.later = h.T0 + timedelta(days=3)

    def edit(self, path, old, new):
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")

    def gate_message(self):
        with self.assertRaises(AnnotationError) as caught:
            ad.check_pass2_allowed(self.dataset, self.later)
        return str(caught.exception)

    def assert_blind(self, message):
        for name in self.meta:
            self.assertNotIn(name, message)
            self.assertNotIn(Path(name).stem, message)
        for hidden in ("normal", "P01", "NONE", "bg_", str(self.root), self.root.name):
            self.assertNotIn(hidden, message)
        self.assertIn("pass 2 needs a complete, valid pass 1", message)

    def break_count(self):                                  # status counts disagree with labels
        self.edit(self.status_path, f"{self.name},labelled,1,2", f"{self.name},labelled,1,1")
        return "status counts (face 1, eye 1) disagree with the labels (face 1, eye 2)"

    def break_missing_face(self):                           # labelled, but the face row is gone
        lines = [ln for ln in self.label_path.read_text(encoding="utf-8").splitlines(True)
                 if not ln.startswith(self.name + ",face,")]
        self.label_path.write_text("".join(lines), encoding="utf-8", newline="")
        return "has eye points but no face"

    def break_eye_outside_face(self):
        self.edit(self.label_path, f"{self.name},eye,270,190,,,left,0",
                  f"{self.name},eye,5,190,,,left,0")
        return "left eye point is not inside the face box"

    def break_background_labelled(self):
        self.edit(self.status_path, f"{self.bg},no_face,0,0", f"{self.bg},labelled,1,0")
        return "background-only images must have status no_face"

    def break_unknown_file(self):                           # a name that is not in the dataset
        with open(self.status_path, "a", encoding="utf-8", newline="") as handle:
            handle.write("ghost_P99_001.png,no_face,0,0,2026-11-20T10:00:00+05:00,1,1.0\n")
        return "status row for a file that is not in the dataset"

    def test_no_case_reveals_a_filename_a_condition_or_a_path(self):
        for breaker in (self.break_count, self.break_missing_face, self.break_eye_outside_face,
                        self.break_background_labelled, self.break_unknown_file):
            with self.subTest(case=breaker.__name__):
                self.tearDown_dataset()
                self.setUp_dataset()
                message = self.gate_message_after(breaker)
                self.assert_blind(message)
                self.assertNotIn("ghost", message)
                self.assertNotRegex(message, r"\.png")

    def test_the_message_still_says_what_is_wrong(self):
        for breaker in (self.break_count, self.break_missing_face, self.break_eye_outside_face,
                        self.break_background_labelled, self.break_unknown_file):
            with self.subTest(case=breaker.__name__):
                self.tearDown_dataset()
                self.setUp_dataset()
                expected = breaker()
                message = self.gate_message()
                self.assertIn(expected, message)
                self.assertIn("First problem:", message)
                self.assertIn("validate_labels.py", message)

    def test_the_counts_survive_the_redaction(self):
        self.break_count()
        message = self.gate_message()
        self.assertIn("face 1, eye 1", message)
        self.assertIn("face 1, eye 2", message)
        self.assertIn("an image:", message)                 # the name is replaced, not dropped

    def test_the_validator_itself_is_unchanged_and_still_names_the_image(self):
        self.break_count()
        errors = vl.validate_dataset(self.dataset, 1, require_complete=True)["errors"]
        self.assertTrue(any(self.name in e for e in errors))          # for scripts/validate_labels.py

    def test_unreadable_metadata_does_not_leak_the_folder_path(self):
        (self.dataset / "metadata.csv").write_text("not,a,metadata,file\n", encoding="utf-8")
        message = self.gate_message()
        self.assertNotIn(str(self.root), message)
        self.assertNotIn(str(self.dataset), message)
        self.assertNotIn(self.dataset.as_posix(), message)

    def test_blind_text_helper_directly(self):
        text = (f"{self.name}: has status labelled but not exactly one face; see "
                f"{self.dataset}\\images\\{self.persons[1]} and {Path(self.persons[2]).stem} "
                f"and {self.dataset.as_posix()}/metadata.csv; counts (face 1, eye 2) other.PNG")
        out = ad.blind_text(text, self.dataset)
        for name in self.persons + self.bgs:
            self.assertNotIn(name, out)
            self.assertNotIn(Path(name).stem, out)
        self.assertNotIn(str(self.dataset), out)
        self.assertNotIn(self.dataset.as_posix(), out)
        self.assertNotIn("other.PNG", out)
        self.assertIn("counts (face 1, eye 2)", out)
        self.assertIn("has status labelled but not exactly one face", out)           # text kept
        self.assertEqual(ad.blind_text(out, self.dataset), out)         # idempotent
        self.assertEqual(ad.blind_text("no names here: 3 of 5", self.dataset),
                         "no names here: 3 of 5")

    # -- gating behaviour is unchanged
    def test_invalid_pass1_still_blocks_pass2_and_pass2_files_are_never_created(self):
        pass1_before = [p.read_bytes() for p in self.paths(1)]
        self.break_count()
        pass1_broken = [p.read_bytes() for p in self.paths(1)]
        self.assertNotEqual(pass1_before, pass1_broken)
        for call in (lambda: ad.check_pass2_allowed(self.dataset, self.later),
                     lambda: ad.build_order(self.dataset, 2, self.later),
                     lambda: ad.start_session_checks(self.dataset, 2, self.later,
                                                     check_git=False)):
            with self.assertRaises(AnnotationError):
                call()
        self.assertEqual([p.read_bytes() for p in self.paths(1)], pass1_broken)  # not modified
        self.assertFalse(any(p.exists() for p in self.paths(2)))                 # nothing created
        self.assertEqual((self.backups(1), self.backups(2)), ([], []))

    def test_command_line_refusal_is_blind_and_creates_nothing(self):
        self.break_missing_face()
        before = [p.read_bytes() for p in self.paths(1)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(ad, "check_gitignore"):
            code = ad.main(["--pass", "2", "--dataset", str(self.dataset)])
        self.assertEqual(code, 1)
        text = out.getvalue()
        self.assertTrue(text.startswith("Error: pass 2 needs a complete, valid pass 1"))
        self.assert_blind(text)
        self.assertIn("has eye points but no face", text)
        self.assertEqual([p.read_bytes() for p in self.paths(1)], before)
        self.assertFalse(any(p.exists() for p in self.paths(2)))
        self.assertEqual((self.backups(1), self.backups(2)), ([], []))

    def test_valid_pass1_still_permits_pass2_and_the_check_writes_nothing(self):
        before = [p.read_bytes() for p in self.paths(1)]
        ad.check_pass2_allowed(self.dataset, self.later)                # no exception
        order = ad.build_order(self.dataset, 2, self.later)
        self.assertTrue(order)
        self.assertTrue(all(not vl.is_background(self.meta[n]) for n in order))
        self.assertEqual([p.read_bytes() for p in self.paths(1)], before)
        self.assertFalse(any(p.exists() for p in self.paths(2)))
        self.assertEqual((self.backups(1), self.backups(2)), ([], []))

    def test_the_24_hour_rule_still_applies_to_valid_pass1(self):
        too_early = h.T0 + timedelta(seconds=len(self.meta), hours=1)
        with self.assertRaises(AnnotationError) as caught:
            ad.check_pass2_allowed(self.dataset, too_early)
        self.assertIn("24 h gap required", str(caught.exception))

    # -- helpers that rebuild a fresh dataset for each sub-case
    def tearDown_dataset(self):
        self._tmp.cleanup()

    def setUp_dataset(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dataset = h.make_dataset(self.root, self.n_person, self.n_bg)
        self.meta = vl.load_metadata(self.dataset)
        self.persons = sorted(n for n, m in self.meta.items() if not vl.is_background(m))
        self.bgs = sorted(n for n, m in self.meta.items() if vl.is_background(m))
        h.annotate_everything(self.dataset)
        self.label_path, self.status_path = self.paths()
        self.name, self.bg = self.persons[0], self.bgs[0]

    def gate_message_after(self, breaker):
        breaker()
        return self.gate_message()


# ================================================ --status output is blind as well
class StatusOutputBlindTests(DatasetCase):
    """`annotate_dataset.py --status` shows progress and problems without naming images or
    folders. Synthetic temporary data only; the validator script is not changed."""
    n_person, n_bg = 4, 1

    def setUp(self):
        super().setUp()
        h.annotate_everything(self.dataset)
        self.label_path, self.status_path = self.paths()
        self.name, self.bg = self.persons[0], self.bgs[0]

    def edit(self, path, old, new):
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")

    def status(self, pass_no=1):
        out = io.StringIO()
        args = ["--status", "--dataset", str(self.dataset)] + (["--pass", "2"] if pass_no == 2 else [])
        with contextlib.redirect_stdout(out):
            code = ad.main(args)
        return code, out.getvalue()

    def assert_blind(self, text):
        for name in self.meta:
            self.assertNotIn(name, text)
            self.assertNotIn(Path(name).stem, text)
        for hidden in ("P01", "NONE", "normal_", "bg_", "ghost", ".png", str(self.root),
                       str(self.dataset), self.dataset.as_posix(), self.root.name):
            self.assertNotIn(hidden, text)

    def break_count(self):
        self.edit(self.status_path, f"{self.name},labelled,1,2", f"{self.name},labelled,1,1")
        return "status counts (face 1, eye 1) disagree with the labels (face 1, eye 2)"

    def break_eye_outside_face(self):
        self.edit(self.label_path, f"{self.name},eye,270,190,,,left,0",
                  f"{self.name},eye,5,190,,,left,0")
        return "left eye point is not inside the face box"

    def break_background(self):
        self.edit(self.status_path, f"{self.bg},no_face,0,0", f"{self.bg},labelled,1,0")
        return "background-only images must have status no_face"

    def break_unknown_file(self):
        with open(self.status_path, "a", encoding="utf-8", newline="") as handle:
            handle.write("ghost_P99_001.png,no_face,0,0,2026-11-20T10:00:00+05:00,1,1.0\n")
        return "status row for a file that is not in the dataset"

    def break_interrupted_save(self):                       # label rows without a status row
        lines = [ln for ln in self.status_path.read_text(encoding="utf-8").splitlines(True)
                 if not ln.startswith(self.persons[1] + ",")]
        self.status_path.write_text("".join(lines), encoding="utf-8", newline="")
        return "has label rows but no status row (interrupted save)"

    def fresh(self):
        import tempfile
        self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dataset = h.make_dataset(self.root, self.n_person, self.n_bg)
        self.meta = vl.load_metadata(self.dataset)
        self.persons = sorted(n for n, m in self.meta.items() if not vl.is_background(m))
        self.bgs = sorted(n for n, m in self.meta.items() if vl.is_background(m))
        h.annotate_everything(self.dataset)
        self.label_path, self.status_path = self.paths()
        self.name, self.bg = self.persons[0], self.bgs[0]

    def test_valid_status_output_is_unchanged(self):
        code, text = self.status()
        self.assertEqual(code, 0)
        self.assertEqual(text, "Pass 1: 5/5 images annotated (4 with a face, 1 no_face, "
                               "0 uncertain objects)\n")
        self.assertNotIn("ERROR", text)

    def test_partial_progress_output_is_unchanged(self):
        self.fresh()
        for path in self.paths():
            path.unlink()
        h.annotate_everything(self.dataset, names=sorted(self.meta)[:2])
        code, text = self.status()
        self.assertEqual(code, 0)
        self.assertTrue(text.startswith("Pass 1: 2/5 images annotated ("))
        self.assert_blind(text)

    def test_no_problem_text_reveals_a_filename_a_participant_or_a_path(self):
        for breaker in (self.break_count, self.break_eye_outside_face, self.break_background,
                        self.break_unknown_file, self.break_interrupted_save):
            with self.subTest(case=breaker.__name__):
                self.fresh()
                expected = breaker()
                code, text = self.status()
                self.assertEqual(code, 0)                              # status itself still works
                self.assertIn("ERROR", text)
                self.assert_blind(text)
                self.assertIn(expected, text)                          # the real problem is shown

    def test_counts_and_the_progress_line_remain_visible(self):
        self.break_count()
        code, text = self.status()
        first = text.splitlines()[0]
        self.assertEqual(first, "Pass 1: 5/5 images annotated (4 with a face, 1 no_face, "
                                "0 uncertain objects)")
        self.assertIn("ERROR   an image: status counts (face 1, eye 1) disagree with the labels "
                      "(face 1, eye 2)", text)

    def test_every_error_line_is_redacted_not_just_the_first(self):
        self.break_count()
        self.break_eye_outside_face()
        self.break_background()
        code, text = self.status()
        errors = [ln for ln in text.splitlines() if ln.startswith("ERROR")]
        self.assertGreaterEqual(len(errors), 3)
        self.assertTrue(all("an image:" in ln for ln in errors))
        self.assert_blind(text)

    def test_unusable_dataset_error_is_blind_too(self):
        (self.dataset / "metadata.csv").write_text("not,a,metadata,file\n", encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ad.main(["--status", "--dataset", str(self.dataset)])
        self.assertEqual(code, 1)
        text = out.getvalue()
        self.assertTrue(text.startswith("Error: "))
        self.assertIn("lacks the columns", text)                       # the problem remains
        self.assertNotIn(str(self.dataset), text)
        self.assertNotIn(self.dataset.as_posix(), text)
        self.assertNotIn(str(self.root), text)

    def test_the_validator_still_gives_filename_specific_diagnostics(self):
        self.break_count()
        self.break_eye_outside_face()
        errors = vl.validate_dataset(self.dataset, 1, require_complete=False)["errors"]
        self.assertTrue(any(e.startswith(self.name + ": status counts") for e in errors))
        self.assertTrue(any(self.name in e and "not inside the face box" in e for e in errors))
        result = subprocess.run([sys.executable, "-I", str(Path(vl.__file__)), "--dataset",
                                 str(self.dataset)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(self.name, result.stdout + result.stderr)        # script output names it

    def test_status_never_writes_anything(self):
        self.break_count()
        before = {p.name: p.read_bytes() for p in self.paths()}
        self.status()
        self.assertEqual({p.name: p.read_bytes() for p in self.paths()}, before)
        self.assertFalse(any(p.exists() for p in self.paths(2)))
        self.assertEqual((self.backups(1), self.backups(2)), ([], []))


# ================================================================= keyboard session
class ReopenSessionTests(DatasetCase):
    n_person, n_bg = 3, 1

    def setUp(self):
        super().setUp()
        h.annotate_everything(self.dataset)
        order, _ = ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(days=1),
                                           check_git=False)
        self.session = ad.Session(self.dataset, 1, order, SCALE,
                                  clock=lambda: h.T0 + timedelta(days=1, minutes=5))
        self.name = next(n for n in order if not vl.is_background(self.meta[n]))
        self.session.index = order.index(self.name)
        self.session._load_current()

    def play(self, script):
        with Harness(script) as harness:
            self.session.run()
        return harness

    def saved(self):
        data = ad.load_pass(self.dataset, 1)
        return data["faces"][self.name][0], data["eyes"][self.name], data["status"][self.name]

    def test_f_redraws_the_face_and_enter_saves_it(self):
        s = self.session
        self.assertTrue(s.annotation.is_complete())
        self.play(key("f") + drag(*NEW_FACE) + key(ENTER) + STOP)
        face, eyes, status = self.saved()
        self.assertEqual((face["x"], face["y"], face["w"], face["h"]), (190, 90, 260, 280))
        self.assertEqual([(e["side"], e["x"], e["y"]) for e in eyes],
                         [("left", 270, 190), ("right", 370, 190)])
        self.assertEqual((status["status"], status["n_face"], status["n_eye"]), ("labelled", 1, 2))
        self.assertIn("previous version kept in backups", s.message)
        self.assertEqual(vl.validate_dataset(self.dataset, 1)["errors"], [])

    def test_uppercase_keys_work_too(self):
        self.play(key("L") + click(280, 200) + key(ENTER) + STOP)
        self.assertEqual((self.saved()[1][0]["x"], self.saved()[1][0]["y"]), (280, 200))

    def test_l_and_r_replace_single_eyes(self):
        self.play(key("l") + click(280, 200) + key("r") + click(380, 195) + key(ENTER) + STOP)
        _, eyes, _ = self.saved()
        self.assertEqual([(e["side"], e["x"], e["y"]) for e in eyes],
                         [("left", 280, 200), ("right", 380, 195)])

    def test_l_then_h_hides_the_eye(self):
        self.play(key("l") + key("h") + key(ENTER) + STOP)
        _, eyes, status = self.saved()
        self.assertEqual([e["side"] for e in eyes], ["right"])
        self.assertEqual(status["n_eye"], 1)

    def test_enter_before_the_object_is_placed_again_saves_nothing(self):
        before = {p.name: p.read_bytes() for p in self.paths()}
        s = self.session
        self.play(key("f") + key(ENTER) + STOP)
        self.assertIn("not finished", s.message)
        self.assertEqual({p.name: p.read_bytes() for p in self.paths()}, before)

    def test_rejected_replacement_is_reported_and_state_is_kept(self):
        s = self.session
        self.play(key("f") + drag(*SMALL_FACE) + STOP)
        self.assertIn("eye would be outside the new face box", s.message)
        self.assertIsNone(s.annotation.face)

    def test_z_restores_the_saved_object_and_leaves_a_clean_state(self):
        s = self.session
        self.play(key("f") + key("l") + key("z") + key("z") + STOP)
        self.assertFalse(s.dirty())
        self.assertEqual(s.annotation.face, (200, 100, 240, 260))
        self.assertEqual(s.annotation.eyes["left"], h.LEFT_EYE)
        self.play(key("z") + STOP)
        self.assertIn("nothing to undo", s.message)

    def test_quit_and_back_warn_about_unsaved_edits(self):
        s = self.session
        self.play(key("l") + STOP)
        self.assertTrue(s.dirty())
        self.assertIsNone(s.handle_key(ord("q")))
        self.assertIn("unsaved changes", s.message)
        self.assertIsNone(s.handle_key(ord("b")))                          # still on this image
        self.assertEqual(s.filename(), self.name)
        self.assertIn("unsaved changes", s.message)
        s.handle_key(ord("z"))                                             # back to clean
        self.assertEqual(s.handle_key(ord("q")), "quit")

    def test_quit_after_an_edit_needs_a_second_press_and_discards_it(self):
        before = {p.name: p.read_bytes() for p in self.paths()}
        s = self.session
        self.play(key("l") + QUIT)
        self.assertEqual({p.name: p.read_bytes() for p in self.paths()}, before)
        self.assertEqual(s.exit_reason, "quit key pressed (key code 27)")

    def test_edit_survives_a_new_session(self):
        self.play(key("l") + click(280, 200) + key(ENTER) + STOP)
        order = ad.build_order(self.dataset, 1)
        fresh = ad.Session(self.dataset, 1, order, SCALE)
        fresh.index = order.index(self.name)
        fresh._load_current()
        self.assertEqual(fresh.annotation.eyes["left"], (280, 200))
        self.assertFalse(fresh.dirty())

    def test_unchanged_replacement_reports_no_change_and_makes_no_backup(self):
        count = len(self.backups())
        self.play(key("l") + click(*h.LEFT_EYE) + key(ENTER) + STOP)
        self.assertEqual(self.session.message, "no change")
        self.assertEqual(len(self.backups()), count)

    def test_blind_display_is_kept_while_editing(self):
        harness = self.play(key("f") + drag(*NEW_FACE) + key("l") + click(280, 200)
                            + key(ENTER) + STOP)
        shown = " ".join(harness.texts)
        for name in self.meta:
            self.assertNotIn(name, shown)
            self.assertNotIn(Path(name).stem, shown)
        for word in ("normal", "frontal", "background", "P01", "NONE", "sub_condition", "bg_"):
            self.assertNotIn(word, shown)
        self.assertTrue(any(re.search(r"f/l/r=redo", t) for t in harness.texts))

    def test_status_strip_is_not_wider_than_the_window(self):
        harness = self.play(STOP)
        help_line = next(t for t in harness.texts if t.startswith("drag=face"))
        for needed in ("f/l/r=redo", "z=undo", "q=quit"):
            self.assertIn(needed, help_line)
        width = cv2.getTextSize(help_line, cv2.FONT_HERSHEY_SIMPLEX, ad.HELP_SCALE, 1)[0][0]
        self.assertLessEqual(8 + width, int(vl.IMAGE_W * SCALE))           # whole line visible


if __name__ == "__main__":
    unittest.main()
