"""Tests for scripts/annotate_dataset.py.

Only synthetic images in temporary folders are used: no real dataset image is annotated,
no webcam is needed, and no window is opened (OpenCV's window functions are replaced by a
scripted fake).
"""
import contextlib
import io
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotation_helpers as h  # noqa: E402

ad, vl = h.ad, h.vl
SCALE = 1.25


class DatasetCase(unittest.TestCase):
    n_person, n_bg = 20, 4

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dataset = h.make_dataset(self.root, self.n_person, self.n_bg)
        self.meta = vl.load_metadata(self.dataset)
        self.persons = sorted(n for n, m in self.meta.items() if not vl.is_background(m))
        self.bgs = sorted(n for n, m in self.meta.items() if vl.is_background(m))

    def paths(self, pass_no=1):
        return vl.label_paths(self.dataset, pass_no)

    def backups(self, pass_no=1):
        return ad.list_backups(self.dataset, pass_no)


# ===================================================================== Annotation
class AnnotationStateTests(unittest.TestCase):
    def test_full_annotation_records(self):
        a = h.full_annotation()
        self.assertTrue(a.is_complete())
        rows, status = a.records("x.png")
        self.assertEqual([r["object_type"] for r in rows], ["face", "eye", "eye"])
        self.assertEqual((rows[0]["x"], rows[0]["y"], rows[0]["w"], rows[0]["h"]),
                         ("200", "100", "240", "260"))
        self.assertEqual((rows[1]["side"], rows[1]["x"], rows[1]["w"], rows[1]["h"]),
                         ("left", "270", "", ""))
        self.assertEqual(rows[2]["side"], "right")
        self.assertEqual(status, {"status": "labelled", "n_face": 1, "n_eye": 2})

    def test_targets_follow_the_workflow(self):
        a = ad.Annotation()
        self.assertEqual(a.target(), "face")
        a.set_face(*h.FACE_DRAG)
        self.assertEqual(a.target(), "left")
        a.place_eye(*h.LEFT_EYE)
        self.assertEqual(a.target(), "right")
        a.place_eye(*h.RIGHT_EYE)
        self.assertIsNone(a.target())

    def test_eye_rules(self):
        a = ad.Annotation()
        with self.assertRaises(ad.AnnotationError):
            a.place_eye(270, 190)                       # no face yet
        a.set_face(*h.FACE_DRAG)
        with self.assertRaises(ad.AnnotationError):
            a.place_eye(20, 20)                         # outside the face box
        a.place_eye(300, 190)
        with self.assertRaises(ad.AnnotationError):
            a.place_eye(280, 190)                       # right eye left of the left eye
        a.place_eye(380, 190)
        with self.assertRaises(ad.AnnotationError):
            a.place_eye(330, 190)                       # already complete

    def test_hidden_eye_stores_no_point(self):
        a = ad.Annotation()
        a.set_face(*h.FACE_DRAG)
        a.place_eye(*h.LEFT_EYE)
        a.hide_eye()
        rows, status = a.records("x.png")
        self.assertEqual(status["n_eye"], 1)
        self.assertEqual([r["side"] for r in rows if r["object_type"] == "eye"], ["left"])
        self.assertTrue(a.is_complete())

    def test_both_eyes_hidden_is_complete(self):
        a = ad.Annotation()
        a.set_face(*h.FACE_DRAG)
        a.hide_eye()
        a.hide_eye()
        self.assertTrue(a.is_complete())
        self.assertEqual(a.records("x.png")[1]["n_eye"], 0)
        with self.assertRaises(ad.AnnotationError):
            a.hide_eye()

    def test_uncertain_applies_to_last_placed_object(self):
        a = ad.Annotation()
        with self.assertRaises(ad.AnnotationError):
            a.toggle_uncertain()
        a.set_face(*h.FACE_DRAG)
        a2 = ad.Annotation()
        a2.set_face(*h.FACE_DRAG)
        a2.toggle_uncertain()
        a2.place_eye(*h.LEFT_EYE)
        a2.place_eye(*h.RIGHT_EYE)
        a2.toggle_uncertain()
        rows, _ = a2.records("x.png")
        self.assertEqual([r["uncertain"] for r in rows], ["1", "0", "1"])
        a2.toggle_uncertain()
        self.assertEqual([r["uncertain"] for r in a2.records("x.png")[0]], ["1", "0", "0"])

    def test_uncertain_not_allowed_on_hidden_eye(self):
        a = ad.Annotation()
        a.set_face(*h.FACE_DRAG)
        a.hide_eye()
        with self.assertRaises(ad.AnnotationError):
            a.toggle_uncertain()

    def test_no_face_rules(self):
        a = ad.Annotation()
        a.mark_no_face()
        self.assertTrue(a.is_complete())
        self.assertEqual(a.records("x.png"), ([], {"status": "no_face", "n_face": 0, "n_eye": 0}))
        with self.assertRaises(ad.AnnotationError):
            a.set_face(*h.FACE_DRAG)
        a.undo()
        a.set_face(*h.FACE_DRAG)
        with self.assertRaises(ad.AnnotationError):
            a.mark_no_face()                            # annotations exist: undo first

    def test_undo_restores_each_step(self):
        a = ad.Annotation()
        a.set_face(*h.FACE_DRAG)
        a.place_eye(*h.LEFT_EYE)
        a.hide_eye()
        a.undo()
        self.assertEqual(a.target(), "right")
        a.undo()
        self.assertEqual(a.target(), "left")
        a.undo()
        self.assertEqual(a.target(), "face")
        self.assertTrue(a.is_empty())
        with self.assertRaises(ad.AnnotationError):
            a.undo()

    def test_undo_restores_uncertain(self):
        a = ad.Annotation()
        a.set_face(*h.FACE_DRAG)
        a.toggle_uncertain()
        a.undo()
        self.assertEqual(a.uncertain["face"], 0)

    def test_face_box_rules(self):
        a = ad.Annotation()
        with self.assertRaises(ad.AnnotationError):
            a.set_face(10, 10, 15, 15)                  # too small
        a.set_face(600, 400, 300, 50)                   # corners in any order, clipped to frame
        self.assertEqual(a.face, (300, 50, 300, 350))
        with self.assertRaises(ad.AnnotationError):
            a.set_face(*h.FACE_DRAG)                    # only one face
        b = ad.Annotation()
        b.set_face(-50, -20, 700, 500)
        self.assertEqual(b.face, (0, 0, vl.IMAGE_W, vl.IMAGE_H))

    def test_incomplete_annotation_cannot_be_recorded(self):
        a = ad.Annotation()
        a.set_face(*h.FACE_DRAG)
        with self.assertRaises(ad.AnnotationError):
            a.records("x.png")

    def test_from_saved_round_trip(self):
        a = h.full_annotation()
        a.toggle_uncertain()
        rows, status = a.records("x.png")
        faces = [{"x": 200, "y": 100, "w": 240, "h": 260, "uncertain": 0}]
        eyes = [{"x": 270, "y": 190, "side": "left", "uncertain": 0},
                {"x": 370, "y": 190, "side": "right", "uncertain": 1}]
        faces[0]["uncertain"] = int(rows[0]["uncertain"])
        b = ad.Annotation.from_saved(faces, eyes, {"status": "labelled"})
        self.assertEqual(b.records("x.png")[0], rows)
        c = ad.Annotation.from_saved([], [], {"status": "no_face"})
        self.assertTrue(c.no_face)
        d = ad.Annotation.from_saved(faces, eyes[:1], {"status": "labelled"})
        self.assertTrue(d.hidden["right"] and d.is_complete())


# ======================================================================== storage
class StorageTests(DatasetCase):
    def save(self, name, annotation=None, now=h.T0):
        bg = vl.is_background(self.meta[name])
        return ad.save_image(self.dataset, 1, name, annotation or
                             (h.no_face_annotation() if bg else h.full_annotation()), bg, now)

    def test_save_creates_files_with_headers_and_sorted_rows(self):
        self.save(self.persons[1])
        self.save(self.persons[0])
        self.save(self.bgs[0])
        label_path, status_path = self.paths()
        self.assertEqual(label_path.read_text().splitlines()[0], ",".join(vl.LABEL_FIELDS))
        self.assertEqual(status_path.read_text().splitlines()[0], ",".join(vl.STATUS_FIELDS))
        names = [r.split(",")[0] for r in label_path.read_text().splitlines()[1:]]
        self.assertEqual(names, sorted(names))
        statuses = [r.split(",")[0] for r in status_path.read_text().splitlines()[1:]]
        self.assertEqual(statuses, sorted(statuses))
        data = ad.load_pass(self.dataset, 1)
        self.assertEqual(len(data["status"]), 3)
        self.assertEqual(len(data["faces"][self.persons[0]]), 1)
        self.assertEqual(data["orphans"], [])
        self.assertEqual(vl.validate_dataset(self.dataset, require_complete=False)["errors"], [])

    def test_no_temp_files_left_behind(self):
        self.save(self.persons[0])
        self.assertEqual([p.name for p in self.dataset.iterdir() if p.name.endswith(".tmp")], [])

    def test_identical_resave_changes_nothing_and_makes_no_backup(self):
        self.save(self.persons[0])
        before = [p.read_bytes() for p in self.paths()]
        self.assertEqual(self.save(self.persons[0], now=h.T0 + timedelta(hours=1)), "unchanged")
        self.assertEqual([p.read_bytes() for p in self.paths()], before)
        self.assertEqual(self.backups(), [])

    def test_editing_never_silently_overwrites(self):
        name = self.persons[0]
        self.save(name)
        original = [p.read_bytes() for p in self.paths()]
        edited = h.full_annotation(left=(280, 195))
        self.assertEqual(self.save(name, edited, h.T0 + timedelta(minutes=5)), "saved")
        self.assertNotEqual([p.read_bytes() for p in self.paths()], original)
        backups = self.backups()
        self.assertEqual(len(backups), 2)                               # labels + status
        self.assertEqual({b.read_bytes() for b in backups}, set(original))
        again = h.full_annotation(left=(285, 195))
        self.save(name, again, h.T0 + timedelta(minutes=6))
        self.assertEqual(len(self.backups()), 4)                        # every edit is kept

    def test_invalid_records_are_not_saved(self):
        a = h.full_annotation()
        with self.assertRaises(ad.AnnotationError):
            ad.save_image(self.dataset, 1, self.bgs[0], a, True, h.T0)  # face on a background
        with self.assertRaises(ad.AnnotationError):
            ad.save_image(self.dataset, 1, self.persons[0], ad.Annotation(), False, h.T0)
        self.assertFalse(self.paths()[0].exists())
        self.assertFalse(self.paths()[1].exists())

    def test_background_requires_no_face(self):
        self.assertEqual(self.save(self.bgs[0]), "saved")
        row = ad.load_pass(self.dataset, 1)["status"][self.bgs[0]]
        self.assertEqual((row["status"], row["n_face"], row["n_eye"]), ("no_face", 0, 0))
        self.assertEqual(ad.load_pass(self.dataset, 1)["labels"], [])

    def test_status_row_fields(self):
        self.save(self.persons[0])
        row = ad.load_pass(self.dataset, 1)["status"][self.persons[0]]
        self.assertEqual((row["status"], row["n_face"], row["n_eye"], row["tool_version"]),
                         ("labelled", 1, 2, ad.TOOL_VERSION))
        self.assertEqual(row["annotated_at"], h.T0)

    def test_interruption_between_label_and_status_write_is_recovered(self):
        name = self.persons[0]
        real = ad.write_atomic
        calls = []

        def crash_on_status(path, data):
            calls.append(path.name)
            if len(calls) == 2:
                raise KeyboardInterrupt
            real(path, data)
        with mock.patch.object(ad, "write_atomic", side_effect=crash_on_status):
            with self.assertRaises(KeyboardInterrupt):
                self.save(name)
        data = ad.load_pass(self.dataset, 1)
        self.assertEqual(data["orphans"], [name])
        self.assertNotIn(name, data["status"])                          # not marked complete
        report = vl.validate_dataset(self.dataset, require_complete=False)
        self.assertTrue(any("interrupted save" in e for e in report["errors"]))
        # the next session notices, backs up, and the save repairs the image
        order, notes = ad.start_session_checks(self.dataset, 1, h.T0, check_git=False)
        self.assertTrue(any("interrupted save" in n for n in notes))
        self.save(name, now=h.T0 + timedelta(minutes=1))
        self.assertEqual(ad.load_pass(self.dataset, 1)["orphans"], [])
        self.assertEqual(vl.validate_dataset(self.dataset, require_complete=False)["errors"], [])
        self.assertEqual(len(ad.load_pass(self.dataset, 1)["faces"][name]), 1)

    def test_failed_write_leaves_old_file_intact(self):
        self.save(self.persons[0])
        before = [p.read_bytes() for p in self.paths()]
        with mock.patch.object(ad.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(ad.AnnotationError):
                self.save(self.persons[1], now=h.T0 + timedelta(minutes=1))
        self.assertEqual([p.read_bytes() for p in self.paths()], before)
        self.assertEqual([p.name for p in self.dataset.iterdir() if p.name.endswith(".tmp")], [])

    def test_stale_temp_files_are_removed_at_start(self):
        (self.dataset / ".labels.csv.tmp").write_bytes(b"partial")
        (self.dataset / "labels_keep.tmp").write_bytes(b"not ours")
        self.assertEqual(ad.clean_stale_tmp(self.dataset), [".labels.csv.tmp"])
        self.assertTrue((self.dataset / "labels_keep.tmp").exists())

    def test_session_start_backs_up_existing_files_only(self):
        order, _ = ad.start_session_checks(self.dataset, 1, h.T0, check_git=False)
        self.assertEqual(self.backups(), [])                            # nothing existed yet
        self.save(self.persons[0])
        ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(hours=1), check_git=False)
        names = [b.name for b in self.backups()]
        self.assertEqual(len(names), 2)
        self.assertTrue(all("_session" in n for n in names))
        self.assertTrue(len(order) == self.n_person + self.n_bg)

    def test_corrupted_file_is_refused_and_backup_restores(self):
        name = self.persons[0]
        self.save(name)
        ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(minutes=1), check_git=False)
        good = [p.read_bytes() for p in self.paths()]
        label_path, status_path = self.paths()
        label_path.write_bytes(label_path.read_bytes()[:-9])            # torn / damaged file
        with self.assertRaises(ad.AnnotationError) as caught:
            ad.load_pass(self.dataset, 1)
        self.assertIn("backups", str(caught.exception))
        restored = ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(minutes=2))
        self.assertEqual(sorted(restored), ["annotation_status.csv", "labels.csv"])
        self.assertEqual([p.read_bytes() for p in self.paths()], good)
        self.assertEqual(ad.load_pass(self.dataset, 1)["orphans"], [])
        prerestore = [p for p in ad.backups_dir(self.dataset).iterdir() if "prerestore" in p.name]
        self.assertTrue(prerestore)                                     # the damaged copy is kept

    def test_restore_without_backup_fails_cleanly(self):
        with self.assertRaises(ad.AnnotationError):
            ad.restore_latest_backup(self.dataset, 1, h.T0)

    def test_gitignore_check_uses_git_exit_codes(self):
        ok = subprocess.CompletedProcess([], 0, "", "")
        bad = subprocess.CompletedProcess([], 1, "", "")
        with mock.patch.object(ad.subprocess, "run", return_value=ok):
            ad.check_gitignore(self.root)
        with mock.patch.object(ad.subprocess, "run", return_value=bad):
            with self.assertRaises(ad.AnnotationError):
                ad.check_gitignore(self.root)
        with mock.patch.object(ad.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaises(ad.AnnotationError):
                ad.check_gitignore(self.root)

    def test_real_project_annotation_paths_are_git_ignored(self):
        ad.check_gitignore(vl.PROJECT_ROOT)

    def test_missing_images_folder_is_refused(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(ad.AnnotationError):
                ad.check_layout(empty)

    def test_original_images_and_metadata_are_never_modified(self):
        before = {p.name: p.read_bytes() for p in (self.dataset / "images").iterdir()}
        meta = (self.dataset / "metadata.csv").read_bytes()
        h.annotate_everything(self.dataset)
        ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(hours=1), check_git=False)
        self.assertEqual({p.name: p.read_bytes() for p in (self.dataset / "images").iterdir()},
                         before)
        self.assertEqual((self.dataset / "metadata.csv").read_bytes(), meta)


# ============================================================ ordering and repeat pass
class OrderAndRepeatTests(DatasetCase):
    def test_blind_order_is_fixed_and_covers_every_image(self):
        order = ad.build_order(self.dataset, 1)
        self.assertEqual(sorted(order), sorted(self.meta))
        self.assertEqual(order, ad.build_order(self.dataset, 1))
        self.assertNotEqual(order, sorted(self.meta))

    def test_pass2_refused_when_pass1_incomplete(self):
        h.annotate_everything(self.dataset, names=self.persons[:5])
        with self.assertRaises(ad.AnnotationError) as caught:
            ad.build_order(self.dataset, 2, h.T0 + timedelta(days=3))
        self.assertIn("complete, valid pass 1", str(caught.exception))

    def test_pass2_needs_24_hours_after_pass1_finished(self):
        h.annotate_everything(self.dataset)
        finished = h.T0 + timedelta(seconds=len(self.meta) - 1)
        with self.assertRaises(ad.AnnotationError) as caught:
            ad.build_order(self.dataset, 2, finished + timedelta(hours=23, minutes=59))
        self.assertIn("24 h gap", str(caught.exception))
        order = ad.build_order(self.dataset, 2, finished + timedelta(hours=24))
        self.assertEqual(len(order), 14)

    def test_pass2_sample_is_14_person_images_in_blind_order(self):
        h.annotate_everything(self.dataset)
        later = h.T0 + timedelta(days=2)
        order = ad.build_order(self.dataset, 2, later)
        self.assertEqual(len(set(order)), 14)
        self.assertTrue(set(order) <= set(self.persons))
        self.assertEqual(set(order), set(vl.repeat_sample(self.persons)))
        self.assertEqual(order, ad.build_order(self.dataset, 2, later))

    def test_pass2_uses_separate_files_and_keeps_pass1(self):
        h.annotate_everything(self.dataset)
        pass1 = [p.read_bytes() for p in self.paths(1)]
        later = h.T0 + timedelta(days=2)
        order = ad.build_order(self.dataset, 2, later)
        for i, name in enumerate(order[:3]):
            ad.save_image(self.dataset, 2, name, h.full_annotation(left=(275, 192)), False,
                          later + timedelta(seconds=i))
        self.assertEqual([p.read_bytes() for p in self.paths(1)], pass1)
        self.assertTrue(self.paths(2)[0].name == "labels_pass2.csv" and self.paths(2)[0].exists())
        data2 = ad.load_pass(self.dataset, 2)
        self.assertEqual(len(data2["status"]), 3)
        self.assertEqual({r["tool_version"] for r in data2["status"].values()}, {ad.TOOL_VERSION})
        report = vl.validate_dataset(self.dataset, 2, require_complete=False)
        self.assertEqual(report["errors"], [])
        self.assertEqual(vl.validate_dataset(self.dataset, 1)["errors"], [])
        self.assertEqual(len(ad.load_pass(self.dataset, 1)["status"]), len(self.meta))

    def test_pass2_complete_validation(self):
        h.annotate_everything(self.dataset)
        later = h.T0 + timedelta(days=2)
        order = ad.build_order(self.dataset, 2, later)
        for i, name in enumerate(order):
            ad.save_image(self.dataset, 2, name, h.full_annotation(), False,
                          later + timedelta(seconds=i))
        report = vl.validate_dataset(self.dataset, 2)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["stats"]["annotated_images"], 14)

    def test_pass2_backups_are_separate_from_pass1(self):
        h.annotate_everything(self.dataset)
        later = h.T0 + timedelta(days=2)
        ad.start_session_checks(self.dataset, 1, later, check_git=False)
        self.assertEqual(len(self.backups(1)), 2)
        self.assertEqual(self.backups(2), [])


# ========================================================================= session
class Harness:
    """Replaces OpenCV's window functions by a script of keys and mouse events."""

    def __init__(self, script, visible=1):
        self.script = list(script)
        self.visible = visible if not isinstance(visible, (list, tuple)) else 1
        self.visible_seq = list(visible) if isinstance(visible, (list, tuple)) else None
        self.callback = None
        self.texts = []
        self.frames = []
        self.destroyed = 0
        self.real_put_text = cv2.putText

    def _wait(self, ms):
        if not self.script:
            return 27
        item = self.script.pop(0)
        if item[0] == "key":
            return item[1]
        if item[0] == "raise":
            raise item[1]
        if item[0] == "stop":                    # end the loop without pressing any key
            self.visible = 0
            return -1
        event = {"down": cv2.EVENT_LBUTTONDOWN, "up": cv2.EVENT_LBUTTONUP,
                 "move": cv2.EVENT_MOUSEMOVE}[item[0]]
        self.callback(event, item[1], item[2], 0, None)
        return -1

    def _visible(self, *a):
        if self.visible_seq:                                  # one value per call; last repeats
            return self.visible_seq.pop(0) if len(self.visible_seq) > 1 else self.visible_seq[0]
        return self.visible

    def _put_text(self, img, text, *a, **k):
        self.texts.append(text)
        return self.real_put_text(img, text, *a, **k)

    def __enter__(self):
        def set_cb(name, cb):
            self.callback = cb
        self.patches = [
            mock.patch.object(cv2, "namedWindow"),
            mock.patch.object(cv2, "setMouseCallback", side_effect=set_cb),
            mock.patch.object(cv2, "imshow", side_effect=lambda w, f: self.frames.append(f)),
            mock.patch.object(cv2, "waitKey", side_effect=self._wait),
            mock.patch.object(cv2, "getWindowProperty", side_effect=self._visible),
            mock.patch.object(cv2, "destroyAllWindows",
                              side_effect=lambda: setattr(self, "destroyed", self.destroyed + 1)),
            mock.patch.object(cv2, "putText", side_effect=self._put_text),
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


def px(v):
    return int(round(v * SCALE))


def drag(x0, y0, x1, y1):
    return [("down", px(x0), px(y0)), ("move", px((x0 + x1) / 2), px((y0 + y1) / 2)),
            ("up", px(x1), px(y1))]


def click(x, y):
    return [("down", px(x), px(y)), ("up", px(x), px(y))]


def key(ch):
    return [("key", ord(ch) if isinstance(ch, str) else ch)]


ENTER, ESC = 13, 27
FULL = (drag(*h.FACE_DRAG) + click(*h.LEFT_EYE) + click(*h.RIGHT_EYE))
QUIT = key(ESC) + key(ESC)
STOP = [("stop",)]


class SessionTests(DatasetCase):
    n_person, n_bg = 3, 1

    def session(self, pass_no=1):
        order, _ = ad.start_session_checks(self.dataset, pass_no, h.T0 + timedelta(days=9),
                                           check_git=False, min_gap_hours=0)
        return ad.Session(self.dataset, pass_no, order, SCALE, clock=lambda: h.T0)

    def play(self, session, script, visible=1):
        with Harness(script, visible) as harness:
            session.run()
        return harness

    def first_two(self, session):
        return session.order[0], session.order[1]

    def test_annotating_with_mouse_and_keys_saves_and_advances(self):
        s = self.session()
        first, second = self.first_two(s)
        bg = vl.is_background(self.meta[first])
        script = (key("n") if bg else FULL) + key(ENTER) + QUIT
        self.play(s, script)
        data = ad.load_pass(self.dataset, 1)
        self.assertIn(first, data["status"])
        self.assertNotIn(second, data["status"])
        if not bg:
            face = data["faces"][first][0]
            self.assertEqual((face["x"], face["y"], face["w"], face["h"]), (200, 100, 240, 260))
            eyes = {e["side"]: (e["x"], e["y"]) for e in data["eyes"][first]}
            self.assertEqual(eyes, {"left": h.LEFT_EYE, "right": h.RIGHT_EYE})
        self.assertEqual(s.index, 1)

    def test_display_scaling_preserves_original_coordinates(self):
        s = self.session()
        self.assertEqual(s.to_orig(500, 400), (400, 320))
        self.assertEqual(s.to_orig(0, 0), (0, 0))
        self.assertEqual(s.to_orig(10_000, 10_000), (vl.IMAGE_W - 1, vl.IMAGE_H - 1))
        s2 = ad.Session(self.dataset, 1, s.order, 1.0)
        self.assertEqual(s2.to_orig(500, 400), (500, 400))

    def test_blind_nothing_identifying_is_displayed(self):
        s = self.session()
        harness = self.play(s, FULL + key(ENTER) + key("n") + key(ENTER) + QUIT)
        shown = " ".join(harness.texts)
        for name in self.meta:
            self.assertNotIn(name, shown)
            self.assertNotIn(Path(name).stem, shown)
        for word in ("normal", "frontal", "background", "P01", "NONE", "sub_condition", "bg_"):
            self.assertNotIn(word, shown)
        self.assertTrue(any(re.search(r"Image \d+/4", t) for t in harness.texts))

    def test_hidden_eye_and_uncertain_flow(self):
        s = self.person_first_session()
        name = s.filename()
        script = (drag(*h.FACE_DRAG) + key("u") + click(*h.LEFT_EYE) + key("h") + key(ENTER) + QUIT)
        self.play(s, script)
        data = ad.load_pass(self.dataset, 1)
        self.assertEqual(data["faces"][name][0]["uncertain"], 1)
        self.assertEqual([e["side"] for e in data["eyes"][name]], ["left"])
        self.assertEqual(data["status"][name]["n_eye"], 1)

    def person_first_session(self):
        s = self.session()
        if vl.is_background(self.meta[s.order[0]]):
            s.index = next(i for i, n in enumerate(s.order) if not vl.is_background(self.meta[n]))
            s._load_current()
        return s

    def test_undo_removes_last_actions(self):
        s = self.person_first_session()
        self.play(s, drag(*h.FACE_DRAG) + click(*h.LEFT_EYE) + key("z") + key("z") + QUIT)
        self.assertTrue(s.annotation.is_empty())
        self.play(s, key("z") + QUIT)
        self.assertIn("nothing to undo", s.message)

    def test_incomplete_enter_is_refused_and_nothing_saved(self):
        s = self.person_first_session()
        self.play(s, drag(*h.FACE_DRAG) + key(ENTER) + STOP)
        self.assertEqual(ad.load_pass(self.dataset, 1)["status"], {})
        self.assertIn("not finished", s.message)

    def test_eye_outside_face_is_rejected_with_message(self):
        s = self.person_first_session()
        self.play(s, drag(*h.FACE_DRAG) + click(20, 20) + STOP)
        self.assertIn("outside the face box", s.message)
        self.assertEqual(s.annotation.target(), "left")

    def test_tiny_drag_is_not_a_face(self):
        s = self.person_first_session()
        self.play(s, click(300, 300) + QUIT)
        self.assertIsNone(s.annotation.face)
        self.assertIn("drag", s.message)

    def test_face_on_background_cannot_be_saved(self):
        s = self.session()
        s.index = s.order.index(self.bgs[0])
        s._load_current()
        self.play(s, FULL + key(ENTER) + STOP)
        self.assertEqual(ad.load_pass(self.dataset, 1)["status"], {})
        self.assertIn("nobody", s.message)
        self.play(s, key("z") + key("z") + key("z") + key("n") + key(ENTER) + QUIT)
        self.assertEqual(ad.load_pass(self.dataset, 1)["status"][self.bgs[0]]["status"], "no_face")

    def test_back_returns_to_previous_image_and_edit_creates_backup(self):
        s = self.session()
        first = s.order[0]
        bg = vl.is_background(self.meta[first])
        s.index = 0
        first_script = key("n") if bg else FULL
        self.play(s, first_script + key(ENTER) + key("b") + QUIT)
        self.assertEqual(s.index, 0)
        self.assertEqual(s.filename(), first)
        self.assertEqual(len([b for b in self.backups() if "_edit" in b.name]), 0)
        if not bg:
            self.play(s, key("z") + click(*h.LEFT_EYE) + key(ENTER) + QUIT)   # edit after undo
            self.assertTrue(any("_edit" in b.name for b in self.backups()))

    def test_back_on_first_image_is_refused(self):
        s = self.session()
        s.index = 0
        s._load_current()
        self.assertIsNone(s.handle_key(ord("b")))
        self.assertEqual(s.index, 0)
        self.assertIn("first image", s.message)

    def test_back_with_unsaved_changes_needs_a_second_press(self):
        s = self.session()
        s.index = 1
        s._load_current()
        if vl.is_background(self.meta[s.order[1]]):
            s.annotation.mark_no_face()
        else:
            s.annotation.set_face(*h.FACE_DRAG)
        s.handle_key(ord("b"))
        self.assertEqual(s.index, 1)
        self.assertIn("unsaved", s.message)
        s.handle_key(ord("b"))
        self.assertEqual(s.index, 0)                                  # discarded, went back
        self.assertTrue(s.annotation.is_empty())

    def test_any_other_action_cancels_a_pending_quit(self):
        s = self.person_first_session()
        s.annotation.set_face(*h.FACE_DRAG)
        s.annotation.place_eye(*h.LEFT_EYE)
        self.assertIsNone(s.handle_key(ord("q")))                     # arms the quit
        s.handle_key(ord("u"))                                        # a real action in between
        self.assertIsNone(s.handle_key(ord("q")))                     # warns again, no quit
        self.assertEqual(s.handle_key(ord("q")), "quit")              # now confirmed

    def test_first_quit_press_only_warns(self):
        s = self.person_first_session()
        s.annotation.set_face(*h.FACE_DRAG)
        self.assertIsNone(s.handle_key(ord("q")))
        self.assertIn("unsaved", s.message)
        self.assertEqual(s.handle_key(ord("q")), "quit")

    def test_quit_without_changes_is_immediate_and_progress_is_kept(self):
        s = self.session()
        script = (key("n") if vl.is_background(self.meta[s.order[0]]) else FULL) + key(ENTER) + key("q")
        self.play(s, script)
        self.assertEqual(len(ad.load_pass(self.dataset, 1)["status"]), 1)

    def test_window_close_ends_session(self):
        s = self.session()
        harness = self.play(s, [("move", 10, 10)] * 6, visible=[1, 1, 0])
        self.assertEqual(harness.destroyed, 1)
        self.assertEqual(len(harness.frames), 3)                      # ended on the 3rd frame
        self.assertIn("window was closed", s.exit_reason)

    def test_initial_visibility_glitch_is_ignored(self):
        s = self.session()
        # not visible, not visible, visible, visible, closed -> only the last one may end it
        harness = self.play(s, [("move", 10, 10)] * 10, visible=[-1, -1, 1, 1, 0])
        self.assertEqual(len(harness.frames), 5)
        self.assertIn("window was closed", s.exit_reason)
        self.assertIn("after it had been visible", s.exit_reason)

    def test_window_never_visible_exits_after_grace_period_with_diagnostic(self):
        s = self.session()
        with mock.patch.object(ad, "STARTUP_GRACE_FRAMES", 3):
            harness = self.play(s, [("move", 10, 10)] * 20, visible=[-1])
        self.assertEqual(len(harness.frames), 3)                      # bounded, not indefinite
        self.assertIn("never became visible", s.exit_reason)
        self.assertIn("property -1", s.exit_reason)

    def test_visible_within_grace_period_is_not_cut_short(self):
        s = self.session()
        with mock.patch.object(ad, "STARTUP_GRACE_FRAMES", 3):
            harness = self.play(s, [("move", 10, 10)] * 3 + key("q"), visible=[-1, -1, 1])
        self.assertIn("quit key pressed", s.exit_reason)

    def test_quit_key_works_during_grace_period(self):
        s = self.session()
        with mock.patch.object(ad, "STARTUP_GRACE_FRAMES", 50):
            self.play(s, key("q"), visible=[-1])
        self.assertIn("quit key pressed", s.exit_reason)

    def test_exit_reason_names_the_quit_key(self):
        for code, text in ((ord("q"), "key code 113"), (ESC, "key code 27")):
            with self.subTest(code=code):
                s = self.session()
                self.play(s, key(code))
                self.assertIn(text, s.exit_reason)
                self.assertIn("quit key pressed", s.exit_reason)

    def test_exit_reason_for_keyboard_interrupt(self):
        s = self.session()
        with Harness([("raise", KeyboardInterrupt())]):
            with self.assertRaises(KeyboardInterrupt):
                s.run()
        self.assertIn("KeyboardInterrupt", s.exit_reason)

    def test_main_prints_the_exit_reason(self):
        out = io.StringIO()
        with mock.patch.object(ad, "check_gitignore"), contextlib.redirect_stdout(out):
            with Harness(key(ESC)):
                code = ad.main(["--dataset", str(self.dataset)])
        self.assertEqual(code, 0)
        self.assertIn("Exit reason: quit key pressed (key code 27)", out.getvalue())

    def test_loaded_no_face_can_be_corrected_to_a_face_with_one_eye(self):
        """Regression: reopen a completed no_face image, toggle no_face off, annotate a face."""
        meta = vl.load_metadata(self.dataset)
        order = ad.build_order(self.dataset, 1)
        target = next(n for n in order if not vl.is_background(meta[n]))
        for i, name in enumerate(order):                              # complete pass 1
            bg = vl.is_background(meta[name])
            a = h.no_face_annotation() if (bg or name == target) else h.full_annotation()
            ad.save_image(self.dataset, 1, name, a, bg, h.T0 + timedelta(seconds=i))
        before = {p.name: p.read_bytes() for p in vl.label_paths(self.dataset, 1)}
        ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(days=1), check_git=False)
        session_backups = {b.name: b.read_bytes() for b in self.backups()}
        s = ad.Session(self.dataset, 1, order, SCALE, clock=lambda: h.T0 + timedelta(days=1, minutes=3))
        self.assertIn("reviewing", s.message)
        s.index = order.index(target)
        s._load_current()
        self.assertTrue(s.annotation.no_face)                         # loaded as no_face
        self.assertTrue(s.annotation.is_complete())
        self.assertEqual(ad.Annotation.from_saved([], [], {"status": "no_face"}).target(), None)
        script = (key("n") + drag(*h.FACE_DRAG) + key("h") + click(*h.RIGHT_EYE)
                  + key(ENTER) + STOP)
        self.play(s, script)
        data = ad.load_pass(self.dataset, 1)
        st = data["status"][target]
        self.assertEqual((st["status"], st["n_face"], st["n_eye"]), ("labelled", 1, 1))
        face = data["faces"][target][0]
        self.assertEqual((face["x"], face["y"], face["w"], face["h"]), (200, 100, 240, 260))
        self.assertEqual([(e["side"], e["x"], e["y"]) for e in data["eyes"][target]],
                         [("right", 370, 190)])                       # the hidden eye has no row
        after = {p.name: p.read_bytes() for p in vl.label_paths(self.dataset, 1)}

        def others(blob):
            return [line for line in blob.decode().splitlines() if not line.startswith(target + ",")]
        self.assertEqual(others(before["labels.csv"]), others(after["labels.csv"]))
        self.assertEqual(others(before["annotation_status.csv"]),
                         others(after["annotation_status.csv"]))
        edit = [b for b in self.backups() if "_edit" in b.name]
        self.assertEqual(sorted(b.read_bytes() for b in edit), sorted(before.values()))
        for name, content in session_backups.items():                 # earlier backups untouched
            self.assertEqual((ad.backups_dir(self.dataset) / name).read_bytes(), content)
        self.assertEqual(vl.validate_dataset(self.dataset, 1)["errors"], [])

    def test_toggling_no_face_off_and_on_again_restores_the_original(self):
        a = ad.Annotation.from_saved([], [], {"status": "no_face"})
        a.mark_no_face()
        self.assertFalse(a.no_face)
        self.assertEqual(a.target(), "face")
        a.undo()
        self.assertTrue(a.no_face)
        self.assertEqual(a.fingerprint(), ad.Annotation.from_saved(
            [], [], {"status": "no_face"}).fingerprint())

    def test_keyboard_interrupt_keeps_saved_images_and_closes_window(self):
        s = self.session()
        first_is_bg = vl.is_background(self.meta[s.order[0]])
        script = (key("n") if first_is_bg else FULL) + key(ENTER) + [("raise", KeyboardInterrupt())]
        with Harness(script) as harness:
            with self.assertRaises(KeyboardInterrupt):
                s.run()
        self.assertEqual(harness.destroyed, 1)
        self.assertEqual(len(ad.load_pass(self.dataset, 1)["status"]), 1)
        # an interrupted session resumes at the first unfinished image
        resumed = self.session()
        self.assertEqual(resumed.index, 1)

    def test_resume_starts_at_first_unannotated_image(self):
        order = ad.build_order(self.dataset, 1)
        h.annotate_everything(self.dataset, names=order[:2])
        s = ad.Session(self.dataset, 1, order, SCALE)
        self.assertEqual(s.index, 2)
        self.assertEqual(s.done_count(), 2)

    def test_finishing_the_last_image_ends_the_pass(self):
        order = ad.build_order(self.dataset, 1)
        h.annotate_everything(self.dataset, names=order[:-1])
        s = ad.Session(self.dataset, 1, order, SCALE)
        last_bg = vl.is_background(self.meta[order[-1]])
        self.play(s, (key("n") if last_bg else FULL) + key(ENTER) + QUIT)
        self.assertTrue(s.finished)
        self.assertEqual(vl.validate_dataset(self.dataset)["errors"], [])

    def test_all_annotated_dataset_opens_for_review(self):
        h.annotate_everything(self.dataset)
        s = ad.Session(self.dataset, 1, ad.build_order(self.dataset, 1), SCALE)
        self.assertEqual(s.index, 0)
        self.assertIn("already annotated", s.message)
        self.assertEqual(s.annotation.fingerprint() == s.baseline, True)

    def test_magnifier_draws_only_on_the_display_copy(self):
        s = self.session()
        original = s.image().copy()
        s.cursor = (100, 100)
        canvas = s.render()
        self.assertEqual(canvas.shape, (int(480 * SCALE) + ad.STRIP_HEIGHT, int(640 * SCALE), 3))
        self.assertTrue(np.array_equal(s.image(), original))
        s.cursor = None
        plain = s.render()
        self.assertFalse(np.array_equal(canvas, plain))               # magnifier visible
        top_right = canvas[4:4 + ad.MAG_SIZE, canvas.shape[1] - ad.MAG_SIZE - 4:canvas.shape[1] - 4]
        self.assertEqual(top_right.shape[:2], (ad.MAG_SIZE, ad.MAG_SIZE))

    def test_clicks_on_the_text_strip_are_ignored(self):
        s = self.person_first_session()
        s.on_mouse(cv2.EVENT_LBUTTONDOWN, 100, int(480 * SCALE) + 20)
        self.assertIsNone(s.drag_start)

    def test_repeat_pass_session_shows_only_the_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = h.make_dataset(tmp, 20, 2)
            h.annotate_everything(dataset)
            order = ad.build_order(dataset, 2, h.T0 + timedelta(days=3))
            s = ad.Session(dataset, 2, order, SCALE)
            self.assertEqual(len(s.order), 14)
            canvas = s.render()
            self.assertEqual(canvas.shape[2], 3)
            harness_texts = []
            with Harness(FULL + key(ENTER) + QUIT) as harness:
                s.run()
                harness_texts = harness.texts
            self.assertTrue(any("REPEAT PASS" in t for t in harness_texts))
            self.assertTrue(any(re.search(r"Image \d+/14", t) for t in harness_texts))
            self.assertEqual(len(ad.load_pass(dataset, 2)["status"]), 1)
            self.assertEqual(len(ad.load_pass(dataset, 1)["status"]), 22)

    def test_session_does_not_show_existing_pass1_labels_in_pass2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = h.make_dataset(tmp, 20, 2)
            h.annotate_everything(dataset)
            order = ad.build_order(dataset, 2, h.T0 + timedelta(days=3))
            s = ad.Session(dataset, 2, order, SCALE)
            self.assertTrue(s.annotation.is_empty())                  # blind: nothing pre-filled


class PracticeAndCliTests(unittest.TestCase):
    def test_practice_dataset_is_synthetic_and_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = ad.make_practice_dataset(tmp)
            meta = vl.load_metadata(dataset)
            self.assertEqual(len(meta), 4)
            for name in meta:
                img = cv2.imdecode(np.fromfile(str(dataset / "images" / name), np.uint8),
                                   cv2.IMREAD_COLOR)
                self.assertEqual(img.shape, (480, 640, 3))
            order, _ = ad.start_session_checks(dataset, 1, h.T0, check_git=False)
            self.assertEqual(len(order), 4)

    def test_status_command_prints_progress_without_a_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = h.make_dataset(tmp, 3, 1)
            h.annotate_everything(dataset, names=sorted(vl.load_metadata(dataset))[:2])
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.object(cv2, "imshow") as imshow:
                code = ad.main(["--status", "--dataset", str(dataset)])
            self.assertEqual(code, 0)
            self.assertIn("2/4 images annotated", out.getvalue())
            imshow.assert_not_called()

    def test_pass2_command_refuses_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = h.make_dataset(tmp, 20, 2)
            h.annotate_everything(dataset, now=ad.datetime.now().astimezone())
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.object(ad, "check_gitignore"):
                code = ad.main(["--pass", "2", "--dataset", str(dataset)])
            self.assertEqual(code, 1)
            self.assertIn("24 h gap", out.getvalue())

    def test_neither_tool_uses_a_detector(self):
        for module in (ad, vl):
            source = Path(module.__file__).read_text(encoding="utf-8")
            for word in ("src.detector", "CascadeClassifier", "detectMultiScale",
                         "EyeDetector", "FaceDetector"):
                self.assertNotIn(word, source, module.__name__)


if __name__ == "__main__":
    unittest.main()
