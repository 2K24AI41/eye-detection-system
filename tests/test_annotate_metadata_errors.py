"""The annotator reports an unusable metadata.csv as a clean error, not as a traceback.

`scripts/validate_labels.py` raises ValidationError; `annotate_dataset.main()` shows it through
blind_text() and returns exit code 1. Synthetic temporary datasets only: nothing under the real
dataset/ is touched, no annotation window is opened (OpenCV's window functions are faked) and
the frozen dedup script is not involved.
"""
import contextlib
import io
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotation_helpers as h  # noqa: E402
from test_annotate_dataset import STOP, DatasetCase, Harness  # noqa: E402

ad, vl = h.ad, h.vl

HEADER = "filename,participant_id,condition,sub_condition,capture_time,notes"


def corrupt_duplicate(path):
    lines = path.read_text(encoding="utf-8").splitlines(True)
    path.write_text("".join(lines + [lines[1]]), encoding="utf-8", newline="")


def corrupt_condition(path):
    path.write_text(path.read_text(encoding="utf-8").replace(",normal,frontal,", ",zzz,frontal,", 1),
                    encoding="utf-8", newline="")


def corrupt_short_row(path):
    lines = path.read_text(encoding="utf-8").splitlines(True)
    path.write_text("".join(lines[:2] + ["normal_P01_002.png,P01\n"] + lines[3:]),
                    encoding="utf-8", newline="")


def corrupt_utf8(path):
    path.write_bytes(HEADER.encode() + b"\n\xff\xfe,P01,normal,frontal,t,\n")


def corrupt_nul(path):
    path.write_bytes((HEADER + "\nnormal_P01_001.png,P01,normal,frontal,t,a\x00b\n").encode())


def corrupt_quote(path):
    path.write_bytes((HEADER + '\nnormal_P01_001.png,P01,normal,frontal,t,"open\n').encode())


def corrupt_participant(path):
    path.write_text(path.read_text(encoding="utf-8").replace("bg_NONE_001.png,NONE", "bg_NONE_001.png,P01", 1),
                    encoding="utf-8", newline="")


def corrupt_empty(path):
    path.write_bytes(b"")


BAD_METADATA = {
    "duplicate filename": (corrupt_duplicate, "repeats the one on line"),
    "unknown condition": (corrupt_condition, "condition is not one of"),
    "short row": (corrupt_short_row, "wrong number of values"),
    "invalid UTF-8": (corrupt_utf8, "not valid UTF-8"),
    "NUL byte": (corrupt_nul, "NUL"),
    "malformed quoting": (corrupt_quote, "malformed CSV"),
    "participant/condition mismatch": (corrupt_participant, "does not match the condition type"),
    "empty file": (corrupt_empty, "is empty or lacks the columns"),
}


class MetadataErrorCase(DatasetCase):
    n_person, n_bg = 3, 1

    def setUp(self):
        super().setUp()
        self.names = sorted(self.meta)                         # real names, taken before damage
        self.path = self.dataset / "metadata.csv"

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.object(ad, "check_gitignore"), Harness(STOP) as harness:
            code = ad.main(["--dataset", str(self.dataset), *argv])
        self.frames_shown = len(harness.frames)                 # frames drawn into the FAKE window
        return code, out.getvalue(), err.getvalue()

    def assert_clean_failure(self, code, out, err, fragment):
        self.assertEqual(code, 1)
        self.assertEqual(self.frames_shown, 0)                  # no annotation session was started
        self.assertTrue(out.startswith("Error: "), out)
        self.assertEqual(out.count("Error:"), 1, out)           # one message, not several
        self.assertEqual(len(out.strip().splitlines()), 1, out)
        self.assertIn(fragment, out)
        self.assertNotIn("Traceback", out + err)
        self.assertEqual(err, "")
        self.assert_blind(out)

    def assert_blind(self, text):
        for name in self.names:
            self.assertNotIn(name, text)
            self.assertNotIn(Path(name).stem, text)
        for hidden in (str(self.root), self.root.name, str(self.dataset), self.dataset.as_posix(),
                       ".png", "normal_P01", "bg_NONE"):
            self.assertNotIn(hidden, text)

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.dataset.iterdir() if p.is_file()}


# ===================================================================== startup
class StartupTests(MetadataErrorCase):
    def test_every_kind_of_bad_metadata_is_a_clean_error_at_startup(self):
        for label, (corrupt, fragment) in BAD_METADATA.items():
            with self.subTest(label=label):
                corrupt(self.path)
                code, out, err = self.run_main()
                self.assert_clean_failure(code, out, err, fragment)
                self.setUp()                                       # fresh dataset for the next case

    def test_nothing_is_written_when_the_metadata_is_bad(self):
        corrupt_duplicate(self.path)
        before = self.snapshot()
        self.run_main()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(ad.backups_dir(self.dataset).exists())     # not even a session backup

    def test_the_message_goes_through_blind_text(self):
        corrupt_condition(self.path)
        with mock.patch.object(ad, "blind_text", wraps=ad.blind_text) as wrapped:
            code, out, err = self.run_main()
        self.assertEqual(code, 1)
        wrapped.assert_called_once()
        self.assertIn("condition is not one of", wrapped.call_args[0][0])
        self.assertEqual(wrapped.call_args[0][1], str(self.dataset))

    def test_blind_text_still_removes_a_name_or_path_that_slips_into_an_error(self):
        error = vl.ValidationError(f"boom near {self.names[0]} in {self.dataset}")
        with mock.patch.object(ad, "start_session_checks", side_effect=error):
            code, out, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertNotIn(self.names[0], out)
        self.assertNotIn(str(self.dataset), out)
        self.assertIn("an image", out)
        self.assertIn("the dataset folder", out)


# ===================================================================== call sites
class CallSiteTests(MetadataErrorCase):
    def test_build_order_raises_validation_error_for_pass1_and_pass2(self):
        corrupt_condition(self.path)
        for pass_no in (1, 2):
            with self.subTest(pass_no=pass_no):
                with self.assertRaises(vl.ValidationError):
                    ad.build_order(self.dataset, pass_no, h.T0 + timedelta(days=3))

    def test_session_constructor_raises_validation_error_and_main_handles_it(self):
        h.annotate_everything(self.dataset)
        order = ad.build_order(self.dataset, 1)                     # computed while still valid
        corrupt_duplicate(self.path)
        with self.assertRaises(vl.ValidationError):
            ad.Session(self.dataset, 1, order)
        with mock.patch.object(ad, "start_session_checks", return_value=(order, [])):
            code, out, err = self.run_main()                        # the Session() call fails
        self.assert_clean_failure(code, out, err, "repeats the one on line")

    def test_start_session_checks_raises_validation_error_before_writing_a_backup(self):
        h.annotate_everything(self.dataset)
        corrupt_condition(self.path)
        with self.assertRaises(vl.ValidationError):
            ad.start_session_checks(self.dataset, 1, h.T0, check_git=False)
        self.assertEqual(self.backups(), [])

    def test_pass2_with_bad_metadata_is_a_clean_error(self):
        h.annotate_everything(self.dataset)
        corrupt_condition(self.path)
        code, out, err = self.run_main("--pass", "2")
        self.assert_clean_failure(code, out, err, "condition is not one of")

    def test_pass2_with_valid_metadata_but_too_early_keeps_its_own_message(self):
        h.annotate_everything(self.dataset, now=ad.datetime.now().astimezone())
        code, out, err = self.run_main("--pass", "2")
        self.assertEqual(code, 1)
        self.assertIn("24 h gap", out)
        self.assertEqual(err, "")

    def test_status_with_bad_metadata_is_still_a_clean_error(self):
        corrupt_condition(self.path)
        code, out, err = self.run_main("--status")
        self.assert_clean_failure(code, out, err, "condition is not one of")

    def test_restore_command_does_not_need_the_metadata(self):
        corrupt_condition(self.path)
        with mock.patch("builtins.input", return_value="yes"):
            code, out, err = self.run_main("--restore-latest-backup")
        self.assertEqual(code, 1)                                   # no backup exists: AnnotationError
        self.assertIn("there is no backup to restore", out)


# ======================================================== existing behaviour unchanged
class UnchangedBehaviourTests(MetadataErrorCase):
    def test_annotation_errors_are_still_reported_exactly_as_before(self):
        import shutil
        shutil.rmtree(self.dataset / "images")
        code, out, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertTrue(out.startswith("Error: "))
        self.assertIn("images", out)
        self.assertIn("not found", out)
        self.assertEqual(err, "")

    def test_a_corrupt_label_file_still_gets_its_backup_hint(self):
        h.annotate_everything(self.dataset)
        ad.start_session_checks(self.dataset, 1, h.T0, check_git=False)
        label_path = self.paths()[0]
        label_path.write_bytes(label_path.read_bytes()[:-9])
        code, out, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("Nothing was changed", out)
        self.assertIn("--restore-latest-backup", out)

    def test_other_exceptions_are_not_swallowed(self):
        for exception in (RuntimeError("boom"), ValueError("bad"), OSError("disk")):
            with self.subTest(kind=type(exception).__name__):
                with mock.patch.object(ad, "start_session_checks", side_effect=exception):
                    with self.assertRaises(type(exception)):
                        self.run_main()

    def test_keyboard_interrupt_still_exits_quietly_with_zero(self):
        with mock.patch.object(ad, "start_session_checks", side_effect=KeyboardInterrupt):
            code, out, err = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("Interrupted", out)

    def test_validation_error_is_not_an_annotation_error(self):
        self.assertFalse(issubclass(vl.ValidationError, ad.AnnotationError))   # separate handler

    def test_valid_metadata_still_starts_a_session_and_ends_normally(self):
        code, out, err = self.run_main()
        self.assertEqual((code, err), (0, ""))
        self.assertIn("Exit reason:", out)
        self.assertIn("Session ended. 0/4 images annotated in pass 1.", out)
        self.assertNotIn("Error", out)

    def test_valid_metadata_status_output_is_unchanged(self):
        h.annotate_everything(self.dataset)
        code, out, err = self.run_main("--status")
        self.assertEqual(code, 0)
        self.assertEqual(out, "Pass 1: 4/4 images annotated (3 with a face, 1 no_face, "
                              "0 uncertain objects)\n")

    def test_practice_mode_is_unaffected(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), Harness(STOP):
            code = ad.main(["--practice"])
        self.assertEqual(code, 0)
        self.assertIn("PRACTICE MODE", out.getvalue())


if __name__ == "__main__":
    unittest.main()
