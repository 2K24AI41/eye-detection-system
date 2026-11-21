"""Backup publication, exact restore and rollback, the restore marker, the dataset locks and
consistent read-only snapshots (scripts/annotate_dataset.py, scripts/validate_labels.py).

Synthetic temporary datasets only: nothing under the real dataset/ is read or written, no
window or camera is opened, and the frozen dedup script is not involved. Concurrency tests
use child processes that coordinate through files in a separate temporary control folder;
the locking tests need Windows (msvcrt) and are skipped elsewhere.
"""
import contextlib
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotation_helpers as h  # noqa: E402

ad, vl = h.ad, h.vl
AnnotationError = ad.AnnotationError
TESTS_DIR = Path(__file__).resolve().parent
WINDOWS = os.name == "nt"
needs_windows = unittest.skipUnless(WINDOWS, "the tested lock mechanism is Windows msvcrt")

PRELUDE = textwrap.dedent(f"""
    import os, sys, time
    from pathlib import Path
    sys.path.insert(0, {str(TESTS_DIR)!r})
    import annotation_helpers as h
    ad, vl = h.ad, h.vl
    CTL, DATASET = Path(sys.argv[1]), Path(sys.argv[2])
    ARGS = sys.argv[3:]

    def signal(name, text="1"):
        tmp = CTL / (name + ".part")
        tmp.write_text(str(text))
        os.replace(tmp, CTL / name)

    def wait_for(name, timeout=60):
        end = time.monotonic() + timeout
        while not (CTL / name).exists():
            if time.monotonic() > end:
                raise SystemExit("timeout waiting for " + name)
            time.sleep(0.005)
""")


def tree(folder):
    """Every path under folder -> bytes (files) or 'dir': a complete snapshot. The lock file
    is recorded by size and file id: its bytes cannot be read while another process holds
    a lock on them (Windows locks are enforced)."""
    out = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.name == vl.LOCK_NAME and p.is_file():
            st = p.stat()
            out[str(p.relative_to(folder))] = ("lock", st.st_size, st.st_ino)
        else:
            out[str(p.relative_to(folder))] = p.read_bytes() if p.is_file() else "dir"
    return out


def state(path):
    return Path(path).read_bytes() if os.path.lexists(path) else None


def set_state(path, data):
    if data is None:
        if os.path.lexists(path):
            os.remove(path)
    else:
        Path(path).write_bytes(data)


def hidden_right_eye():
    a = ad.Annotation()
    a.set_face(*h.FACE_DRAG)
    a.place_eye(*h.LEFT_EYE)
    a.hide_eye()
    return a


class SafetyCase(unittest.TestCase):
    n_person, n_bg = 3, 1

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dataset = h.make_dataset(self.root, self.n_person, self.n_bg)
        self.ctl = self.root / "control"                  # outside the dataset folder
        self.ctl.mkdir()
        self.meta = vl.load_metadata(self.dataset)
        self.names = sorted(self.meta)
        self.persons = [n for n in self.names if not vl.is_background(self.meta[n])]
        self.labels, self.status = vl.label_paths(self.dataset, 1)
        self.marker = vl.restore_marker_path(self.dataset)
        self.children = []
        self.addCleanup(self._stop_children)

    # -- child processes
    def child(self, body, *args):
        proc = subprocess.Popen([sys.executable, "-c", PRELUDE + textwrap.dedent(body),
                                 str(self.ctl), str(self.dataset), *map(str, args)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.children.append(proc)
        return proc

    def _stop_children(self):
        for proc in self.children:
            if proc.poll() is None:
                proc.kill()
            proc.communicate()

    def signal(self, name, text="1"):
        (self.ctl / name).write_text(text)

    def wait_for(self, name, timeout=60, proc=None):
        end = time.monotonic() + timeout
        while not (self.ctl / name).exists():
            if proc is not None and proc.poll() is not None:
                out, err = proc.communicate()
                self.fail(f"child ended before '{name}': {out}\n{err}")
            if time.monotonic() > end:
                self.fail(f"timeout waiting for {name}")
            time.sleep(0.005)
        return (self.ctl / name).read_text()

    def finish(self, proc, timeout=120):
        out, err = proc.communicate(timeout=timeout)
        self.assertEqual(proc.returncode, 0, f"child failed:\n{out}\n{err}")
        return out

    # -- helpers
    def files(self):
        return state(self.labels), state(self.status)

    def run_main(self, *args, answer="yes"):
        out = io.StringIO()
        with mock.patch("builtins.input", return_value=answer) as asked, \
                mock.patch.object(ad, "Session", side_effect=AssertionError("no window")), \
                mock.patch.object(ad, "check_gitignore"), contextlib.redirect_stdout(out):
            code = ad.main(["--dataset", str(self.dataset), *args])
        return code, out.getvalue(), asked

    def run_validator(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = vl.main(["--dataset", str(self.dataset), *args])
        return code, out.getvalue()

    def assert_blind(self, text):
        for name in self.names:
            self.assertNotIn(name, text)
            self.assertNotIn(Path(name).stem, text)
        self.assertNotIn(str(self.dataset), text)
        self.assertNotIn(self.dataset.as_posix(), text)


# ================================================================ publication
class PublicationTests(SafetyCase):
    def setUp(self):
        super().setUp()
        h.annotate_everything(self.dataset)
        self.folder = ad.backups_dir(self.dataset)

    def identity(self, now, tag="edit", n=0):
        return f"{ad._stamp(now)}_{tag}_s{n}"

    def manifest(self, identity, pass_no=1):
        return self.folder / ad._manifest_name(pass_no, identity)

    def member(self, kind, identity, pass_no=1):
        stem = (self.labels if kind == "labels" else self.status).stem
        return self.folder / ad._member_name(stem, pass_no, identity)

    def restorable(self):
        return {info["identity"]: info["data"]
                for info in ad._backup_sets(self.dataset, 1).values() if ad._restorable(info)}

    def test_identity_is_claimed_by_one_manifest_per_operation(self):
        now = h.T0 + timedelta(hours=1)
        combos = [(None, b"S-a\n"), (b"L-b\n", b"S-b\n"), (b"L-c\n", None), (b"", b"")]
        for labels, status in combos:                                       # same stamp, same tag
            set_state(self.labels, labels)
            set_state(self.status, status)
            ad.backup_files(self.dataset, 1, now, tag="edit")
        sets = self.restorable()
        self.assertEqual(sorted(sets), [self.identity(now, n=n) for n in range(4)])
        for n, (labels, status) in enumerate(combos):
            identity = self.identity(now, n=n)
            self.assertTrue(self.manifest(identity).is_file())              # one claim each
            self.assertEqual(sets[identity], {"labels": labels, "status": status})
            for kind, data in (("labels", labels), ("status", status)):
                self.assertEqual(os.path.lexists(self.member(kind, identity)), data is not None)
        set_state(self.labels, b"junk")
        set_state(self.status, b"junk")
        ad.restore_latest_backup(self.dataset, 1, now + timedelta(minutes=1))
        self.assertEqual(self.files(), combos[-1])                         # newest: s3

    def test_absent_member_is_recorded_only_in_the_manifest(self):
        self.labels.unlink()
        status = self.status.read_bytes()
        now = h.T0 + timedelta(hours=1)
        made = ad.backup_files(self.dataset, 1, now, tag="edit")
        identity = self.identity(now)
        self.assertEqual(made, [self.member("status", identity)])
        text = self.manifest(identity).read_text(encoding="ascii")
        self.assertIn("labels=absent\n", text)
        self.assertIn(f"status=present {len(status)} {hashlib.sha256(status).hexdigest()}\n",
                      text)
        self.assertFalse(os.path.lexists(self.member("labels", identity)))
        self.assertEqual([p for p in self.folder.iterdir() if p.suffix == ".absent"], [])
        self.labels.write_bytes(b"made later\n")
        ad.restore_latest_backup(self.dataset, 1, now + timedelta(minutes=1))
        self.assertEqual(self.files(), (None, status))                     # absent again

    def test_collision_on_the_second_member_retries_the_whole_pair(self):
        now = h.T0 + timedelta(hours=1)
        current = self.files()
        s0, s1 = self.identity(now), self.identity(now, n=1)
        self.folder.mkdir(exist_ok=True)
        foreign = self.member("status", s0)
        foreign.write_bytes(b"FOREIGN FILE")                               # occupies member 2
        made = ad.backup_files(self.dataset, 1, now, tag="edit")
        self.assertEqual(made, [self.member("labels", s1), self.member("status", s1)])
        self.assertEqual(foreign.read_bytes(), b"FOREIGN FILE")            # never overwritten
        self.assertTrue(self.manifest(s0).is_file())                       # abandoned attempt
        self.assertEqual(self.member("labels", s0).read_bytes(), current[0])  # kept, not deleted
        sets = self.restorable()
        self.assertNotIn(s0, sets)                                         # never restorable
        self.assertEqual(sets[s1], {"labels": current[0], "status": current[1]})
        self.labels.write_bytes(b"damaged")
        ad.restore_latest_backup(self.dataset, 1, now + timedelta(minutes=1))
        self.assertEqual(self.files(), current)
        self.assertEqual(ad.count_temp_leftovers(self.dataset), 0)

    def link_failing_on(self, target_name, exc):
        real = os.link

        def link(src, dst, *a, **kw):
            if Path(dst).name == target_name:
                raise exc
            return real(src, dst, *a, **kw)
        return mock.patch.object(ad.os, "link", side_effect=link)

    def test_failure_after_the_first_link_leaves_an_unrestorable_attempt(self):
        old = ad.backup_files(self.dataset, 1, h.T0 + timedelta(minutes=30), tag="edit")
        good = self.files()
        now = h.T0 + timedelta(hours=1)
        s0, s1 = self.identity(now), self.identity(now, n=1)
        with self.link_failing_on(self.member("status", s0).name, OSError(28, "No space left")):
            with self.assertRaises(AnnotationError) as caught:
                ad.backup_files(self.dataset, 1, now, tag="edit")
        message = str(caught.exception)
        self.assertIn("incomplete and will never be restored", message)
        self.assertNotIn(str(self.dataset), message)
        self.assertTrue(self.manifest(s0).is_file())
        self.assertTrue(self.member("labels", s0).is_file())               # published, kept
        self.assertFalse(os.path.lexists(self.member("status", s0)))
        self.assertEqual(ad.count_temp_leftovers(self.dataset), 0)         # temps removed
        self.assertNotIn(s0, self.restorable())
        s0_files = {p.name: p.read_bytes() for p in (self.manifest(s0), self.member("labels", s0))}
        made = ad.backup_files(self.dataset, 1, now, tag="edit")           # same stamp again
        self.assertEqual([p.name for p in made],
                         [self.member("labels", s1).name, self.member("status", s1).name])
        for name, data in s0_files.items():
            self.assertEqual((self.folder / name).read_bytes(), data)      # s0 untouched
        self.labels.write_bytes(b"damaged")
        ad.restore_latest_backup(self.dataset, 1, now + timedelta(minutes=1))
        self.assertEqual(self.files(), good)
        self.assertTrue(all(p.exists() for p in old))

    def test_crash_between_the_two_links(self):
        ad.backup_files(self.dataset, 1, h.T0 + timedelta(minutes=30), tag="edit")
        good = self.files()
        now = h.T0 + timedelta(hours=1)
        s0 = self.identity(now)
        with self.link_failing_on(self.member("status", s0).name, Crash()), \
                mock.patch.object(ad, "_remove_tmps"):                      # a crash cleans nothing
            with self.assertRaises(Crash):
                ad.backup_files(self.dataset, 1, now, tag="edit")
        self.assertTrue(self.manifest(s0).is_file())
        self.assertNotIn(s0, self.restorable())
        self.assertEqual(ad.count_temp_leftovers(self.dataset), 3)          # 2 members + manifest
        self.assertFalse(any(s0 in p.name for p in ad.list_backups(self.dataset, 1)))
        self.labels.write_bytes(b"damaged")
        ad.restore_latest_backup(self.dataset, 1, now + timedelta(minutes=1))
        self.assertEqual(self.files(), good)

    def test_unverified_or_ambiguous_sets_are_never_selected(self):
        ad.backup_files(self.dataset, 1, h.T0 + timedelta(minutes=1), tag="edit")
        good = self.files()

        def missing_member(identity):
            self.member("status", identity).unlink()

        def wrong_hash(identity):
            p = self.member("labels", identity)
            data = p.read_bytes()
            p.write_bytes(data[:-1] + (b"X" if data[-1:] != b"X" else b"Y"))   # same size

        def wrong_size(identity):
            p = self.member("labels", identity)
            p.write_bytes(p.read_bytes() + b"\n")

        def edit_manifest(old, new):
            def damage(identity):
                p = self.manifest(identity)
                text = p.read_bytes()
                self.assertIn(old, text)
                p.write_bytes(text.replace(old, new, 1))
            return damage

        def duplicate_member(identity):
            p = self.manifest(identity)
            lines = p.read_bytes().split(b"\n")
            p.write_bytes(b"\n".join(lines[:4] + [lines[3]] + lines[5:]))   # labels twice

        def occupied_absent_name(identity):
            self.member("labels", identity).write_bytes(b"not part of the set")

        damages = {
            "missing member": missing_member,
            "incorrect hash": wrong_hash,
            "incorrect size": wrong_size,
            "malformed manifest": edit_manifest(b"backup-set 1", b"backup set 1"),
            "duplicate member": duplicate_member,
            "wrong pass": edit_manifest(b"pass=1", b"pass=2"),
            "wrong identity": edit_manifest(b"identity=", b"identity=x"),
            "CRLF manifest": edit_manifest(b"\n", b"\r\n"),
        }
        for i, (label, damage) in enumerate(damages.items(), start=1):
            with self.subTest(damage=label):
                now = h.T0 + timedelta(hours=i)
                set_state(self.labels, f"newer {i}\n".encode())
                ad.backup_files(self.dataset, 1, now, tag="edit")
                identity = self.identity(now)
                self.assertIn(identity, self.restorable())                 # valid before damage
                damage(identity)
                self.assertNotIn(identity, self.restorable())
                set_state(self.labels, b"junk")
                set_state(self.status, b"junk")
                ad.restore_latest_backup(self.dataset, 1, now + timedelta(minutes=1))
                self.assertEqual(self.files(), good)                       # the older valid set
        with self.subTest(damage="file where an absent member belongs"):
            now = h.T0 + timedelta(hours=20)
            set_state(self.labels, None)
            ad.backup_files(self.dataset, 1, now, tag="edit")
            occupied_absent_name(self.identity(now))
            self.assertNotIn(self.identity(now), self.restorable())

    def test_legacy_pairs_need_identical_stamp_tag_and_suffix(self):
        self.folder.mkdir(exist_ok=True)
        legacy = {
            "labels_pass1_20261120T090000_000000_session.csv": b"A-L\n",       # exact pair A
            "annotation_status_pass1_20261120T090000_000000_session.csv": b"A-S\n",
            "labels_pass1_20261120T091000_000000_edit.csv": b"B-L\n",          # suffix differs
            "annotation_status_pass1_20261120T091000_000000_edit_1.csv": b"B-S\n",
            "labels_pass1_20261120T092000_000000_edit.csv": b"C-L\n",          # tag differs
            "annotation_status_pass1_20261120T092000_000000_session.csv": b"C-S\n",
            "labels_pass1_20261120T093000_000000_edit.csv": b"D-L\n",          # one member
            "labels_pass1_20261120T094000_000000_edit.csv": b"E-L\n",          # interim .absent
            "annotation_status_pass1_20261120T094000_000000_edit.absent": b"",
        }
        for name, data in legacy.items():
            (self.folder / name).write_bytes(data)
        before = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        sets = ad._backup_sets(self.dataset, 1)
        usable = [k for k, info in sets.items() if ad._restorable(info)]
        self.assertEqual(usable, [("20261120T090000_000000", 0, "session", "legacy")])
        ad.restore_latest_backup(self.dataset, 1, h.T0 - timedelta(hours=1))
        self.assertEqual(self.files(), (b"A-L\n", b"A-S\n"))
        for name, data in before.items():                                  # nothing renamed,
            self.assertEqual((self.folder / name).read_bytes(), data)      # converted or deleted
        ad.backup_files(self.dataset, 1, h.T0 + timedelta(hours=1), tag="edit")
        current = self.files()
        self.labels.write_bytes(b"damaged")
        ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=2))
        self.assertEqual(self.files(), current)                            # newer manifest set

    def test_attempts_are_bounded_and_leave_no_temporary_file(self):
        with mock.patch.object(ad, "MAX_BACKUP_ATTEMPTS", 5), \
                mock.patch.object(ad.os, "link", side_effect=FileExistsError(17, "exists")):
            with self.assertRaises(AnnotationError) as caught:
                ad.backup_files(self.dataset, 1, h.T0, tag="edit")
        self.assertIn("no free backup name", str(caught.exception))
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_a_link_error_before_publication_publishes_nothing(self):
        with mock.patch.object(ad.os, "link", side_effect=OSError(1, "no hard links")):
            with self.assertRaises(AnnotationError) as caught:
                ad.backup_files(self.dataset, 1, h.T0, tag="edit")
        self.assertIn("Cannot write a backup", str(caught.exception))
        self.assertNotIn("incomplete", str(caught.exception))
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_unlink_failure_after_publication_is_a_success_with_counted_leftovers(self):
        current = self.files()
        real_unlink = os.unlink

        def refuse_tmp(path, *a, **kw):
            if str(path).endswith(".tmp") and Path(path).parent == self.folder:
                raise PermissionError(13, "in use")
            return real_unlink(path, *a, **kw)

        now = h.T0 + timedelta(hours=1)
        with mock.patch.object(ad.os, "unlink", side_effect=refuse_tmp):
            made = ad.backup_files(self.dataset, 1, now, tag="edit")
        self.assertEqual(tuple(p.read_bytes() for p in made), current)      # complete, intact
        leftovers = [p for p in self.folder.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(len(leftovers), 3)                                # 2 members + manifest
        self.assertEqual(ad.count_temp_leftovers(self.dataset), 3)
        self.assertEqual(sorted(ad.list_backups(self.dataset, 1)), sorted(made))
        _, notes = ad.start_session_checks(self.dataset, 1, now + timedelta(minutes=1),
                                           check_git=False)
        self.assertTrue(any("3 leftover temporary file(s)" in n for n in notes), notes)
        self.labels.write_bytes(b"damaged")
        ad.restore_latest_backup(self.dataset, 1, now + timedelta(minutes=2))
        self.assertEqual(self.files(), current)                             # backup usable
        for p in leftovers:
            p.unlink()                                                      # by hand
        self.assertEqual(tuple(p.read_bytes() for p in made), current)      # still intact

    def test_a_leftover_that_sorts_newest_is_never_restored(self):
        ad.backup_files(self.dataset, 1, h.T0 + timedelta(hours=1), tag="edit")
        good = self.files()
        for name in (".labels_pass1_99991231T235959_999999_edit_s0.csv.tmp",
                     ".backupset_pass1_99991231T235959_999999_edit_s0.txt.tmp",
                     "backupset_pass1_99991231T235959_999999_edit_s0.txt.tmp",
                     "labels_pass1_99991231T235959_999999_edit.csv.tmp"):
            (self.folder / name).write_bytes(b"NOT A BACKUP")
        self.labels.write_bytes(b"damaged")
        ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=2))
        self.assertEqual(self.files(), good)
        self.assertTrue(all("9999" not in p.name for p in ad.list_backups(self.dataset, 1)))

    def test_only_incomplete_sets_give_a_clear_error(self):
        now = h.T0 + timedelta(hours=1)
        ad.backup_files(self.dataset, 1, now, tag="edit")
        self.member("status", self.identity(now)).unlink()
        with self.assertRaises(AnnotationError) as caught:
            ad.restore_latest_backup(self.dataset, 1, now + timedelta(minutes=1))
        self.assertIn("only incomplete or unverifiable backup sets", str(caught.exception))

    @needs_windows
    def test_concurrent_backups_never_overwrite_and_keep_each_pair_together(self):
        body = """
            k = int(ARGS[0])
            wait_for("go")
            now = h.T0 + h.timedelta(hours=5)
            for i in range(25):
                def current(path, k=k, i=i):
                    kind = "L" if Path(path).name.startswith("labels") else "S"
                    if (k + i) % 3 == 1 and kind == "L":
                        return None                          # labels absent
                    if (k + i) % 3 == 2 and kind == "S":
                        return None                          # status absent
                    return f"{k}:{i}:{kind}".encode()
                ad._read_current = current
                ad.backup_files(DATASET, 1, now, tag="edit")
            print("ok")
        """
        procs = [self.child(body, k) for k in range(4)]
        time.sleep(1.5)                                     # let all four import first
        self.signal("go")
        for proc in procs:
            self.finish(proc)
        stamp = ad._stamp(h.T0 + timedelta(hours=5))
        sets = {k: v for k, v in ad._backup_sets(self.dataset, 1).items() if k[0] == stamp}
        self.assertEqual(len(sets), 100)
        self.assertTrue(all(ad._restorable(info) for info in sets.values()))
        self.assertEqual(sorted(k[1] for k in sets), list(range(100)))     # shared suffixes
        seen, absent = set(), 0
        for info in sets.values():
            data = info["data"]
            present = [v for v in data.values() if v is not None]
            absent += 2 - len(present)
            ids = {v.decode()[:-2] for v in present}                       # "k:i" of the writer
            self.assertEqual(len(ids), 1)                                  # one operation per set
            k, i = map(int, ids.pop().split(":"))
            self.assertEqual(data["labels"] is None, (k + i) % 3 == 1)
            self.assertEqual(data["status"] is None, (k + i) % 3 == 2)
            seen.add((k, i))
        self.assertEqual(len(seen), 100)                                   # nothing overwritten
        self.assertGreater(absent, 0)
        self.assertEqual(ad.count_temp_leftovers(self.dataset), 0)


# ====================================================== exact restore and rollback
class ExactRestoreTests(SafetyCase):
    STATES = [(b"L1\n", None), (b"", b"S2\n"), (b"L3\n", b"S3\n"), (None, b""), (b"", b"")]

    def test_backup_and_restore_reproduce_absent_empty_and_bytes_exactly(self):
        for i, wanted in enumerate(self.STATES):
            with self.subTest(state=wanted):
                set_state(self.labels, wanted[0])
                set_state(self.status, wanted[1])
                made = ad.backup_files(self.dataset, 1, h.T0 + timedelta(hours=i), tag="edit")
                self.assertEqual(len(made), sum(d is not None for d in wanted))
                self.assertTrue(all(p.suffix == ".csv" for p in made))
                self.assertEqual(list(ad.backups_dir(self.dataset).glob("*.absent")), [])
                set_state(self.labels, b"junk")
                set_state(self.status, None if wanted[1] is not None else b"junk")
                ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=i, minutes=30))
                self.assertEqual(self.files(), wanted)
                self.assertFalse(os.path.lexists(self.marker))

    def test_nothing_is_backed_up_when_neither_file_exists(self):
        self.assertEqual(ad.backup_files(self.dataset, 1, h.T0, tag="session"), [])
        self.assertFalse(ad.backups_dir(self.dataset).exists())

    def prepare(self, before):
        set_state(self.labels, b"BK-L\n")
        set_state(self.status, b"BK-S\n")
        ad.backup_files(self.dataset, 1, h.T0, tag="edit")
        set_state(self.labels, before[0])
        set_state(self.status, before[1])

    def failing_write(self, fail_path, times=1):
        real = ad.write_atomic
        count = {"n": 0}

        def write(path, data):
            if Path(path) == Path(fail_path) and count["n"] < times:
                count["n"] += 1
                raise AnnotationError("injected write failure")
            return real(path, data)
        return mock.patch.object(ad, "write_atomic", side_effect=write)

    def test_a_failure_in_each_phase_rolls_back_exactly(self):
        cases = {
            "backup": (mock.patch.object(ad, "backup_files",
                                         side_effect=AnnotationError("injected")), None),
            "labels": (None, "labels"),
            "status": (None, "status"),
        }
        for phase, (patcher, fail_kind) in cases.items():
            for before in ((None, b"S0\n"), (b"", None), (b"L0\n", b"S0\n")):
                with self.subTest(phase=phase, before=before):
                    for p in list(ad.backups_dir(self.dataset).glob("*")):
                        p.unlink()
                    self.prepare(before)
                    fail_path = self.labels if fail_kind == "labels" else self.status
                    ctx = patcher or self.failing_write(fail_path)
                    with ctx, self.assertRaises(AnnotationError) as caught:
                        ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=1))
                    self.assertIn("put back exactly", str(caught.exception))
                    self.assertEqual(self.files(), before)                 # exact rollback
                    self.assertFalse(os.path.lexists(self.marker))

    def test_rollback_failure_keeps_the_marker(self):
        self.prepare((None, b"S0\n"))
        real = ad.write_atomic

        def write(path, data):
            if Path(path) == self.status:
                raise AnnotationError("injected")
            return real(path, data)

        with mock.patch.object(ad, "write_atomic", side_effect=write), \
                mock.patch.object(ad, "_apply_state", wraps=ad._apply_state) as apply, \
                mock.patch.object(ad.os, "remove", side_effect=OSError(13, "denied")):
            with self.assertRaises(AnnotationError) as caught:
                ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=1))
        self.assertIn("could not be put back", str(caught.exception))
        self.assertTrue(apply.called)
        self.assertEqual(vl.read_restore_marker(self.dataset), (True, 1, "status"))

    def test_an_earlier_marker_is_kept_after_a_rolled_back_failure(self):
        self.prepare((b"HALF\n", b"S0\n"))
        self.marker.write_bytes(vl.format_restore_marker(1, "labels"))
        with self.failing_write(self.status), self.assertRaises(AnnotationError):
            ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=1))
        self.assertEqual(self.files(), (b"HALF\n", b"S0\n"))
        self.assertTrue(self.marker.exists())                              # still blocked
        ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=2))
        self.assertEqual(self.files(), (b"BK-L\n", b"BK-S\n"))
        self.assertFalse(os.path.lexists(self.marker))

    def remove_failing_for(self, path):
        real = os.remove

        def remove(p, *a, **kw):
            if Path(p) == Path(path):
                raise PermissionError(13, "in use")
            return real(p, *a, **kw)
        return mock.patch.object(ad.os, "remove", side_effect=remove)

    def test_rollback_ok_but_marker_removal_fails(self):
        before = (b"L0\n", None)
        self.prepare(before)
        with self.failing_write(self.status), self.remove_failing_for(self.marker):
            with self.assertRaises(AnnotationError) as caught:
                ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=1))
        message = str(caught.exception)
        for part in ("restore failed", "put back exactly as they were before the restore",
                     ".restore_in_progress could not be removed", "Annotation stays blocked",
                     "--pass 1 --restore-latest-backup"):
            self.assertIn(part, message)
        self.assertNotIn("recovery is complete", message)
        self.assert_blind(message)
        self.assertEqual(self.files(), before)                             # exact rollback
        self.assertEqual(vl.read_restore_marker(self.dataset), (True, 1, "status"))
        with self.assertRaises(AnnotationError):                           # annotation blocked
            ad.check_restore_marker(self.dataset)
        ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=2))  # finishes it
        self.assertEqual(self.files(), (b"BK-L\n", b"BK-S\n"))
        self.assertFalse(os.path.lexists(self.marker))

    def test_restore_ok_but_marker_removal_fails(self):
        self.prepare((b"L0\n", b"S0\n"))
        with self.remove_failing_for(self.marker):
            with self.assertRaises(AnnotationError) as caught:
                ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=1))
        message = str(caught.exception)
        self.assertIn("the files were restored", message)
        self.assertIn("annotation stays blocked", message)
        self.assert_blind(message)
        self.assertEqual(self.files(), (b"BK-L\n", b"BK-S\n"))
        self.assertEqual(vl.read_restore_marker(self.dataset), (True, 1, "status"))
        ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=2))
        self.assertFalse(os.path.lexists(self.marker))
        self.assertEqual(self.files(), (b"BK-L\n", b"BK-S\n"))


# ============================================================ restore marker
class Crash(BaseException):
    """Stands for a power cut: not an Exception, so nothing catches it."""


class MarkerTests(SafetyCase):
    def setUp(self):
        super().setUp()
        h.annotate_everything(self.dataset)
        ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(hours=1), check_git=False)
        self.good = self.files()
        self.status.write_bytes(self.status.read_bytes().replace(b",labelled,1,2", b",labelled,1,1", 1))
        self.damaged = self.files()

    def marker_writes(self, crash_at=None):
        real, seen = ad.write_atomic, []

        def write(path, data):
            result = real(path, data)
            if Path(path) == self.marker:
                seen.append(data)
                if crash_at and data == vl.format_restore_marker(1, crash_at):
                    raise Crash()
            return result
        return mock.patch.object(ad, "write_atomic", side_effect=write), seen

    def test_restore_writes_the_phases_in_order_and_removes_the_marker(self):
        patcher, seen = self.marker_writes()
        with patcher:
            ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=2))
        self.assertEqual(seen, [vl.format_restore_marker(1, p) for p in vl.MARKER_PHASES])
        self.assertFalse(os.path.lexists(self.marker))
        self.assertEqual(self.files(), self.good)

    def test_a_crash_in_each_phase_blocks_annotation_until_the_restore_finishes(self):
        for i, phase in enumerate(vl.MARKER_PHASES):
            with self.subTest(phase=phase):
                set_state(self.labels, self.damaged[0])
                set_state(self.status, self.damaged[1])
                patcher, _ = self.marker_writes(crash_at=phase)
                with patcher, self.assertRaises(Crash):
                    ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=2, minutes=i))
                self.assertEqual(vl.read_restore_marker(self.dataset), (True, 1, phase))
                backups = ad.list_backups(self.dataset, 1)
                with self.assertRaises(AnnotationError) as caught:
                    ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(hours=3),
                                            check_git=False)
                message = str(caught.exception)
                self.assertIn("A restore of pass 1 was interrupted while "
                              + vl.PHASE_TEXT[phase], message)
                self.assertIn("--pass 1 --restore-latest-backup", message)
                self.assertIn("Nothing was changed", message)
                self.assert_blind(message)
                self.assertEqual(ad.list_backups(self.dataset, 1), backups)  # no session backup
                report = vl.validate_dataset(self.dataset, 1)
                self.assertIn("restore of pass 1 was interrupted", report["errors"][0])
                code, out, _ = self.run_main("--status")
                self.assertEqual(code, 1)
                self.assertIn("Pass 1:", out)                               # statistics kept
                self.assertIn(".restore_in_progress", out)
                ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=4, minutes=i))
                self.assertFalse(os.path.lexists(self.marker))
                self.assertEqual(self.files(), self.good)

    def test_malformed_markers_block_startup_with_the_generic_message(self):
        (self.dataset / ".labels.csv.tmp").write_bytes(b"stale")
        before_backups = ad.list_backups(self.dataset, 1)
        for label, data in (("empty", b""), ("truncated", b"pass=1\nph"),
                            ("garbage", b"\xff\xfe\x00"), ("old format", b"pass=1")):
            with self.subTest(marker=label):
                self.marker.write_bytes(data)
                with self.assertRaises(AnnotationError) as caught:
                    ad.start_session_checks(self.dataset, 1, h.T0 + timedelta(hours=3),
                                            check_git=False)
                message = str(caught.exception)
                self.assertIn("affected pass cannot be determined", message)
                self.assertNotIn("pass 1", message)
                self.assertNotIn("pass 2", message)
                self.assert_blind(message)
                self.assertEqual(ad.list_backups(self.dataset, 1), before_backups)
                self.assertTrue((self.dataset / ".labels.csv.tmp").exists())  # not cleaned
                self.assertEqual(self.marker.read_bytes(), data)               # not changed
        self.marker.unlink()
        self.marker.mkdir()
        lock_existed = os.path.lexists(vl.lock_path(self.dataset))
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("affected pass cannot be determined", out)
        self.assertEqual(os.path.lexists(vl.lock_path(self.dataset)), lock_existed)

    def test_a_restore_for_the_other_pass_is_refused(self):
        self.marker.write_bytes(vl.format_restore_marker(2, "labels"))
        before = self.files()
        with self.assertRaises(AnnotationError) as caught:
            ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=2))
        self.assertIn("for pass 2", str(caught.exception))
        self.assertEqual(self.files(), before)
        self.assertEqual(self.marker.read_bytes(), vl.format_restore_marker(2, "labels"))

    # -- malformed markers block automatic restore for both passes
    MALFORMED = {
        "empty": b"",
        "truncated": b"pass=1\nph",
        "malformed": b"\xff\xfe\x00zz",
        "unknown phase": b"pass=1\nphase=finish\n",
        "invalid pass": b"pass=3\nphase=labels\n",
        "CRLF": b"pass=1\r\nphase=labels\r\n",
        "single field": b"pass=1",
    }

    def all_label_files(self):
        return [state(p) for n in (1, 2) for p in vl.label_paths(self.dataset, n)]

    def place_marker(self, data):
        if self.marker.is_dir():
            self.marker.rmdir()
        elif os.path.lexists(self.marker):
            self.marker.unlink()
        if data == "directory":
            self.marker.mkdir()
        else:
            self.marker.write_bytes(data)

    def marker_state(self):
        return "directory" if self.marker.is_dir() else self.marker.read_bytes()

    def assert_manual_recovery_message(self, message):
        self.assertIn("affected pass cannot be determined", message)
        self.assertIn("manual recovery is required", message)
        self.assertIn("Nothing was changed", message)
        self.assertNotIn("pass 1", message)
        self.assertNotIn("pass 2", message)
        self.assert_blind(message)

    def test_malformed_marker_refuses_restore_for_both_passes(self):
        for path in vl.label_paths(self.dataset, 2):
            path.write_bytes(b"pass 2 content " + path.name.encode() + b"\n")
        cases = list(self.MALFORMED.items()) + [("directory", "directory")]
        for label, data in cases:
            for pass_no in (1, 2):
                with self.subTest(marker=label, pass_no=pass_no):
                    self.place_marker(data)
                    before = (self.all_label_files(), tree(ad.backups_dir(self.dataset)),
                              self.marker_state())
                    with mock.patch.object(ad, "_backup_sets",
                                           side_effect=AssertionError("read backups")) as sets, \
                            mock.patch.object(vl, "data_lock",
                                              side_effect=AssertionError("took the lock")) as lk:
                        with self.assertRaises(AnnotationError) as caught:
                            ad.restore_latest_backup(self.dataset, pass_no,
                                                     h.T0 + timedelta(hours=2))
                    self.assert_manual_recovery_message(str(caught.exception))
                    sets.assert_not_called()                               # no candidates read
                    lk.assert_not_called()                                 # no data lock
                    self.assertEqual((self.all_label_files(), tree(ad.backups_dir(self.dataset)),
                                      self.marker_state()), before)        # no prerestore set
        self.place_marker(b"")
        self.marker.unlink()
        target = self.root / "elsewhere"
        with self.subTest(marker="symbolic link"):
            try:
                os.symlink(target, self.marker)
            except (OSError, NotImplementedError) as error:
                self.skipTest("cannot create symbolic links here "         # reported as skipped,
                              f"({type(error).__name__}); the symbolic-link "  # never a silent
                              "marker case was NOT exercised")                 # pass
            for pass_no in (1, 2):
                with self.subTest(marker="symbolic link", pass_no=pass_no):
                    with self.assertRaises(AnnotationError) as caught:
                        ad.restore_latest_backup(self.dataset, pass_no,
                                                 h.T0 + timedelta(hours=2))
                    self.assert_manual_recovery_message(str(caught.exception))
                    self.assertTrue(os.path.islink(self.marker))

    def test_malformed_marker_cli_refuses_before_asking(self):
        for label, data in (("empty", b""), ("CRLF", self.MALFORMED["CRLF"]),
                            ("directory", "directory")):
            for pass_no in (1, 2):
                with self.subTest(marker=label, pass_no=pass_no):
                    self.place_marker(data)
                    before = (self.all_label_files(), self.marker_state())
                    code, out, asked = self.run_main("--restore-latest-backup",
                                                     "--pass", str(pass_no))
                    self.assertEqual(code, 1)
                    asked.assert_not_called()                              # refused before asking
                    self.assert_manual_recovery_message(out)
                    self.assertEqual((self.all_label_files(), self.marker_state()), before)

    def test_manual_recovery_after_removing_the_marker(self):
        self.marker.write_bytes(b"pass=1\nph")                             # cannot be read
        with self.assertRaises(AnnotationError):
            ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=2))
        copies = self.root / "manual-copies"                               # README step: keep
        copies.mkdir()                                                     # copies first
        for path in (self.labels, self.status, self.marker):
            (copies / path.name).write_bytes(path.read_bytes())
        self.marker.unlink()                                               # deliberate, by hand
        ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(hours=2))
        self.assertEqual(self.files(), self.good)
        self.assertFalse(os.path.lexists(self.marker))
        prerestore = [info["data"] for key, info in
                      ad._backup_sets(self.dataset, 1, include_prerestore=True).items()
                      if key[2] == "prerestore" and ad._restorable(info)]
        self.assertIn({"labels": self.damaged[0], "status": self.damaged[1]}, prerestore)
        self.assertEqual((copies / self.status.name).read_bytes(), self.damaged[1])


# ===================================================================== lock file
@needs_windows
class LockFileTests(SafetyCase):
    CREATE = """
        k = int(ARGS[0])
        if len(ARGS) > 1:
            def no_links(*a, **kw):
                raise OSError(1, "no hard links here")
            os.link = no_links
        wait_for("go")
        path = vl.ensure_lock_file(DATASET)
        fd = vl._open_lock_file(path, os.O_RDWR)
        st = os.fstat(fd)
        got = vl._try_lock(fd, vl.INSTANCE_BYTE, False)
        signal(f"res{k}", f"{st.st_dev}:{st.st_ino}:{got}")
        wait_for("release")
    """

    def concurrent_creation(self, n, *extra):
        procs = [self.child(self.CREATE, k, *extra) for k in range(n)]
        time.sleep(2.0)
        self.signal("go")
        results = [self.wait_for(f"res{k}", proc=procs[k]) for k in range(n)]
        ids = {r.rsplit(":", 1)[0] for r in results}
        got = [r.endswith("True") for r in results]
        self.signal("release")
        for proc in procs:
            self.finish(proc)
        return ids, got

    def test_concurrent_first_creation_gives_one_two_byte_file(self):
        ids, got = self.concurrent_creation(8)
        self.assertEqual(len(ids), 1)                                      # one file object
        self.assertEqual(sum(got), 1)                                      # one holder of byte 0
        self.assertEqual(vl.lock_path(self.dataset).read_bytes(), b"\0\0")
        self.assertEqual(ad.count_temp_leftovers(self.dataset), 0)

    def test_exclusive_create_fallback_gives_one_empty_file(self):
        ids, got = self.concurrent_creation(4, "nolink")
        self.assertEqual((len(ids), sum(got)), (1, 1))
        self.assertEqual(vl.lock_path(self.dataset).read_bytes(), b"")
        self.assertEqual(ad.count_temp_leftovers(self.dataset), 0)

    def test_existing_empty_short_or_filled_lock_files_are_never_changed(self):
        for content in (b"", b"\0", b"keep me"):
            with self.subTest(content=content):
                path = vl.lock_path(self.dataset)
                path.write_bytes(content)
                ino = os.stat(path).st_ino
                lock = ad.InstanceLock(self.dataset)
                ad.start_session_checks(self.dataset, 1, h.T0, check_git=False,
                                        instance_lock=lock)
                self.assertIsNotNone(lock.handle)
                h.annotate_everything(self.dataset, names=self.persons[:1])
                ad.save_image(self.dataset, 1, self.persons[0], hidden_right_eye(), False,
                              h.T0 + timedelta(minutes=5))
                ad.restore_latest_backup(self.dataset, 1, h.T0 + timedelta(minutes=10))
                lock.release()
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(os.stat(path).st_ino, ino)

    def test_a_directory_at_the_lock_path_is_refused_and_nothing_is_read_or_written(self):
        h.annotate_everything(self.dataset)
        vl.lock_path(self.dataset).unlink()                                # made by the saves
        vl.lock_path(self.dataset).mkdir()
        before = tree(self.dataset)
        with self.assertRaises(AnnotationError):
            ad.InstanceLock(self.dataset).acquire()
        with self.assertRaises(AnnotationError):
            ad.save_image(self.dataset, 1, self.persons[0], hidden_right_eye(), False, h.T0)
        report = vl.validate_dataset(self.dataset)
        self.assertEqual(report["stats"], {})
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn(".annotation.lock", report["errors"][0])
        code, out, _ = self.run_main("--status")
        self.assertEqual(code, 1)
        self.assertNotIn("Pass 1:", out)
        self.assertEqual(tree(self.dataset), before)

    def test_a_link_at_the_lock_path_is_refused(self):
        target = self.root / "elsewhere.lock"
        target.write_bytes(b"")
        try:
            os.symlink(target, vl.lock_path(self.dataset))
        except (OSError, NotImplementedError):
            self.skipTest("creating symbolic links is not permitted here")
        with self.assertRaises(AnnotationError):
            ad.InstanceLock(self.dataset).acquire()
        self.assertEqual(vl.validate_dataset(self.dataset)["stats"], {})

    HOLD = """
        handle = vl.acquire_lock_byte(DATASET, int(ARGS[0]), 0)
        signal("held")
        wait_for("release")
        handle.release()
    """

    def test_the_two_bytes_are_independent_and_each_is_exclusive(self):
        a = vl.acquire_lock_byte(self.dataset, vl.INSTANCE_BYTE, 0)
        b = vl.acquire_lock_byte(self.dataset, vl.DATA_BYTE, 0)              # same process
        with self.assertRaises(vl.LockBusy):
            vl.acquire_lock_byte(self.dataset, vl.INSTANCE_BYTE, 0)          # same byte, 2nd handle
        a.release()
        b.release()
        proc = self.child(self.HOLD, vl.INSTANCE_BYTE)
        self.wait_for("held", proc=proc)
        b = vl.acquire_lock_byte(self.dataset, vl.DATA_BYTE, 0)              # other process
        b.release()
        with self.assertRaises(vl.LockBusy):
            vl.acquire_lock_byte(self.dataset, vl.INSTANCE_BYTE, 0.2)
        self.signal("release")
        self.finish(proc)

    def test_the_lock_is_released_when_the_holder_is_killed(self):
        proc = self.child(self.HOLD, vl.INSTANCE_BYTE)
        self.wait_for("held", proc=proc)
        with self.assertRaises(vl.LockBusy):
            vl.acquire_lock_byte(self.dataset, vl.INSTANCE_BYTE, 0)
        proc.kill()
        proc.communicate()
        handle = vl.acquire_lock_byte(self.dataset, vl.INSTANCE_BYTE, 5)
        handle.release()

    def test_a_locked_byte_cannot_be_read_by_another_process(self):
        """Windows locks are enforced: this is why the lock file is never written into."""
        vl.ensure_lock_file(self.dataset)
        proc = self.child(self.HOLD, vl.INSTANCE_BYTE)
        self.wait_for("held", proc=proc)
        with open(vl.lock_path(self.dataset), "rb") as handle:
            with self.assertRaises(OSError):
                handle.read(1)
        self.signal("release")
        self.finish(proc)

    def test_the_lock_file_is_never_deleted_replaced_or_truncated_in_the_code(self):
        for module in (ad, vl):
            source = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotIn("O_TRUNC", source)
            for line in source.splitlines():
                if "lock_path(" in line or "LOCK_NAME" in line:
                    for word in ("remove(", "unlink(", "replace(", "rename(", "rmtree("):
                        self.assertNotIn(word, line, line)
        source = Path(vl.__file__).read_text(encoding="utf-8")
        self.assertIn("os.link(tmp, path)", source)
        self.assertIn("os.O_CREAT | os.O_EXCL", source)


# ================================================================ instance lock
@needs_windows
class InstanceTests(SafetyCase):
    def setUp(self):
        super().setUp()
        h.annotate_everything(self.dataset)
        ad.backup_files(self.dataset, 1, h.T0 + timedelta(hours=1), tag="session")

    def test_a_second_session_or_restore_is_refused_while_reads_still_work(self):
        proc = self.child("""
            lock = ad.InstanceLock(DATASET)
            lock.acquire()
            signal("held")
            wait_for("release")
            lock.release()
        """)
        self.wait_for("held", proc=proc)
        before, backups = tree(self.dataset), ad.list_backups(self.dataset, 1)
        with mock.patch.object(ad, "INSTANCE_LOCK_WAIT", 0.2):
            code, out, asked = self.run_main("--restore-latest-backup")
            self.assertEqual(code, 1)
            self.assertIn("Another annotation tool is already using this dataset", out)
            asked.assert_not_called()                                      # refused before asking
            code, out, _ = self.run_main()
            self.assertEqual(code, 1)
            self.assertIn("Another annotation tool", out)
        self.assertEqual(ad.list_backups(self.dataset, 1), backups)          # no session backup
        code, out, _ = self.run_main("--status")
        self.assertEqual(code, 0)
        self.assertIn("Pass 1:", out)
        self.assertEqual(self.run_validator()[0], 0)
        self.assertEqual(tree(self.dataset), before)
        self.signal("release")
        self.finish(proc)
        code, out, _ = self.run_main("--restore-latest-backup")             # free again
        self.assertEqual(code, 0, out)

    TRY_BYTE_0 = """
        try:
            vl.acquire_lock_byte(DATASET, vl.INSTANCE_BYTE, 0).release()
            print("GOT")
        except vl.LockBusy:
            print("BUSY")
    """

    def test_byte_0_stays_locked_through_repeated_byte_1_cycles(self):
        """Byte 1 is always taken and released through its own, separate handles; that can
        never release the instance lock (byte 0) held on the original handle."""
        lock = ad.InstanceLock(self.dataset)
        lock.acquire()
        try:
            for cycle in range(5):
                with vl.data_lock(self.dataset):                           # writer, new handle
                    pass
                if cycle == 2:
                    with self.assertRaises(ValueError):
                        with vl.data_lock(self.dataset):
                            raise ValueError("an error inside the data lock")
                snap = vl.consistent_read(self.dataset,                    # reader, new handle
                                          lambda: vl.read_pass_snapshot(self.dataset, 1))
                self.assertIsNone(snap["error"])
                annotation = hidden_right_eye() if cycle % 2 == 0 else h.full_annotation()
                result = ad.save_image(self.dataset, 1, self.persons[0], annotation, False,
                                       h.T0 + timedelta(hours=2, minutes=cycle))
                self.assertEqual(result, "saved")                          # really wrote
                with self.assertRaises(vl.LockBusy):                       # another handle,
                    vl.acquire_lock_byte(self.dataset, vl.INSTANCE_BYTE, 0)  # timeout zero
            self.assertEqual(self.finish(self.child(self.TRY_BYTE_0)).strip(), "BUSY")
        finally:
            lock.release()
        other = vl.acquire_lock_byte(self.dataset, vl.INSTANCE_BYTE, 0)    # free only now
        other.release()
        self.assertEqual(self.finish(self.child(self.TRY_BYTE_0)).strip(), "GOT")

    def test_no_self_deadlock_and_no_reentry(self):
        lock = ad.InstanceLock(self.dataset)
        lock.acquire()
        try:
            start = time.monotonic()
            ad.check_pass2_allowed(self.dataset, h.T0 + timedelta(days=3))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ad.print_status(self.dataset, 1), 0)
            self.assertLess(time.monotonic() - start, 3.0)
        finally:
            lock.release()
        with vl.data_lock(self.dataset):
            start = time.monotonic()
            with self.assertRaises(RuntimeError):
                with vl.data_lock(self.dataset):
                    pass
            with self.assertRaises(RuntimeError):
                vl.consistent_read(self.dataset, lambda: None)
            self.assertLess(time.monotonic() - start, 1.0)                 # immediate, no wait


# ============================================================ consistent snapshots
@needs_windows
class SnapshotTests(SafetyCase):
    def setUp(self):
        super().setUp()
        h.annotate_everything(self.dataset)

    HOLD_DATA = """
        with vl.data_lock(DATASET):
            signal("held")
            wait_for("release")
    """

    def test_status_without_a_lock_file_is_read_only(self):
        vl.lock_path(self.dataset).unlink()
        before = tree(self.dataset)
        code, out, _ = self.run_main("--status")
        self.assertEqual(code, 0)
        self.assertIn("Pass 1:", out)
        self.assertEqual(self.run_validator()[0], 0)
        self.assertEqual(tree(self.dataset), before)
        self.assertFalse(os.path.lexists(vl.lock_path(self.dataset)))

    def appearing_lock_scenario(self, mutate):
        """The lock file is missing when a read starts; after the labels file was read, a
        writer creates it and saves an edit (labels + status). Returns (report, labels reads)."""
        vl.lock_path(self.dataset).unlink()
        real = vl.read_csv_strict
        st = {"fired": False, "busy": False, "label_reads": 0}

        def read_csv(path, fields):
            rows = real(path, fields)
            if not st["busy"] and Path(path).name == "labels.csv":
                st["label_reads"] += 1
                if not st["fired"]:
                    st["fired"] = st["busy"] = True
                    try:
                        ad.save_image(self.dataset, 1, self.persons[0], hidden_right_eye(),
                                      False, h.T0 + timedelta(hours=1))
                    finally:
                        st["busy"] = False
            return rows

        patches = [mock.patch.object(vl, "read_csv_strict", side_effect=read_csv)]
        if mutate:
            patches.append(mock.patch.object(vl, "_lock_file_exists", return_value=False))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            report = vl.validate_dataset(self.dataset)
        return report, st["label_reads"]

    def test_a_lock_file_appearing_during_a_read_discards_the_first_result(self):
        report, reads = self.appearing_lock_scenario(mutate=False)
        self.assertTrue(os.path.lexists(vl.lock_path(self.dataset)))
        self.assertEqual(reads, 2)                                         # read again, locked
        self.assertEqual(report["errors"], [])
        self.assertEqual(report, vl.validate_dataset(self.dataset))       # the state after

    def test_mutation_without_the_second_existence_check_reads_a_mixed_pair(self):
        report, reads = self.appearing_lock_scenario(mutate=True)
        self.assertEqual(reads, 1)
        self.assertTrue(any("disagree" in e for e in report["errors"]), report["errors"])

    def test_busy_readers_give_no_statistics_and_wait_when_allowed(self):
        proc = self.child(self.HOLD_DATA)
        self.wait_for("held", proc=proc)
        with mock.patch.object(vl, "LOCK_TIMEOUT", 0.3):
            code, out, _ = self.run_main("--status")
            self.assertEqual(code, 1)
            # main() blinds every error; while busy the names cannot be read, so the whole
            # text is replaced by the busy notice (never a partial or unredacted message)
            self.assertEqual(out, f"Error: {ad.BUSY_DETAILS_HIDDEN}\n")
            self.assertIn("the dataset is busy", out)
            self.assertNotIn("Pass 1:", out)
            code, out = self.run_validator()
            self.assertEqual(code, 1)
            self.assertIn("dataset busy", out)
            self.assertNotIn("stats:", out)
            self.assertIn("RESULT: FAILED", out)
            report = vl.validate_dataset(self.dataset)
            self.assertEqual((report["stats"], len(report["errors"])), ({}, 1))
            self.assertEqual(ad.blind_text(f"x {self.persons[0]}", self.dataset),
                             ad.BUSY_DETAILS_HIDDEN)
        timer = threading.Timer(1.0, self.signal, args=("release",))
        timer.start()
        start = time.monotonic()
        code, out, _ = self.run_main("--status")                           # default timeout
        self.assertGreater(time.monotonic() - start, 0.8)
        self.assertEqual(code, 0)
        self.assertIn("Pass 1:", out)
        timer.join()
        self.finish(proc)

    SAVE_PAUSED = """
        real = ad.write_atomic
        def write(path, data):
            real(path, data)
            if Path(path).name == "labels.csv":
                signal("between")
                wait_for("go")
        ad.write_atomic = write
        a = h.ad.Annotation()
        a.set_face(*h.FACE_DRAG)
        a.place_eye(*h.LEFT_EYE)
        a.hide_eye()
        ad.save_image(DATASET, 1, ARGS[0], a, False, h.T0 + h.timedelta(hours=1))
        print("saved")
    """

    def test_a_reader_never_sees_a_save_between_its_two_writes(self):
        proc = self.child(self.SAVE_PAUSED, self.persons[0])
        self.wait_for("between", proc=proc)
        with mock.patch.object(vl, "LOCK_TIMEOUT", 0.3):
            report = vl.validate_dataset(self.dataset)
        self.assertEqual(report["errors"], [vl.BUSY_MESSAGE])
        self.assertEqual(report["stats"], {})
        # mutation: the same read without the protocol sees the mixed pair
        with mock.patch.object(vl, "consistent_read", lambda d, read, timeout=None: read()):
            report = vl.validate_dataset(self.dataset)
        self.assertTrue(any("disagree" in e for e in report["errors"]), report["errors"])
        self.signal("go")
        self.assertIn("saved", self.finish(proc))
        report = vl.validate_dataset(self.dataset)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["stats"]["eyes"], 2 * len(self.persons) - 1)

    def test_repeated_saves_and_readers_never_mix(self):
        proc = self.child("""
            real = ad.write_atomic
            def write(path, data):
                real(path, data)
                if Path(path).name == "labels.csv":
                    time.sleep(0.02)
            ad.write_atomic = write
            failures = 0
            for i in range(30):
                if i % 2:
                    a = h.full_annotation()
                else:
                    a = h.ad.Annotation()
                    a.set_face(*h.FACE_DRAG)
                    a.place_eye(*h.LEFT_EYE)
                    a.hide_eye()
                try:
                    ad.save_image(DATASET, 1, ARGS[0], a, False,
                                  h.T0 + h.timedelta(hours=1, seconds=i))
                except ad.AnnotationError as error:
                    failures += 1
                    print("FAILED", error)
                time.sleep(0.005)
            print("failures", failures)
        """, self.persons[0])
        reads, bad = 0, []
        while proc.poll() is None:
            code_out = io.StringIO()
            with contextlib.redirect_stdout(code_out):
                status_code = ad.print_status(self.dataset, 1)
            report = vl.validate_dataset(self.dataset)
            reads += 1
            if report["errors"] or status_code or "ERROR" in code_out.getvalue():
                bad.append((report["errors"], code_out.getvalue()))
        out = self.finish(proc)
        self.assertIn("failures 0", out)
        self.assertGreater(reads, 5)
        self.assertEqual(bad, [])

    def damage(self):
        ad.backup_files(self.dataset, 1, h.T0 + timedelta(hours=1), tag="session")
        self.good = self.files()
        self.status.write_bytes(self.status.read_bytes().replace(b",labelled,1,2", b",labelled,1,1", 1))

    def test_a_reader_started_just_before_a_restore_makes_the_restore_wait(self):
        self.damage()
        expected = vl.validate_dataset(self.dataset)                       # damaged state
        self.assertTrue(expected["errors"])
        proc = self.child("""
            real = ad.write_atomic
            def write(path, data):
                if Path(path).name == vl.RESTORE_MARKER and not (CTL / "marker_t").exists():
                    signal("marker_t", repr(time.time()))
                return real(path, data)
            ad.write_atomic = write
            wait_for("reader_mid")
            ad.restore_latest_backup(DATASET, 1, h.T0 + h.timedelta(hours=2))
            print("restored")
        """)
        time.sleep(1.5)                                                    # child imports
        real = vl.read_csv_strict
        times = {}

        def read_csv(path, fields):
            rows = real(path, fields)
            if Path(path).name == "labels.csv" and "mid" not in times:
                times["mid"] = True
                self.signal("reader_mid")
                time.sleep(1.0)                                            # restore starts now
            if Path(path).name == "annotation_status.csv":
                times["done"] = time.time()
            return rows

        with mock.patch.object(vl, "read_csv_strict", side_effect=read_csv):
            report = vl.validate_dataset(self.dataset)
        self.assertEqual(report, expected)                                 # wholly before
        self.assertIn("restored", self.finish(proc))
        marker_time = float((self.ctl / "marker_t").read_text())
        self.assertGreater(marker_time, times["done"])                     # restore waited
        self.assertEqual(vl.validate_dataset(self.dataset)["errors"], [])  # wholly after
        self.assertEqual(self.files(), self.good)

    RESTORE_PAUSED = """
        real = ad._apply_state
        def apply(path, data):
            real(path, data)
            if Path(path).name == "labels.csv":
                signal("mid")
                wait_for("go")
        ad._apply_state = apply
        ad.restore_latest_backup(DATASET, 1, h.T0 + h.timedelta(hours=2))
        print("restored")
    """

    def test_readers_wait_for_a_running_restore(self):
        self.damage()
        proc = self.child(self.RESTORE_PAUSED)
        self.wait_for("mid", proc=proc)
        with mock.patch.object(vl, "LOCK_TIMEOUT", 0.3):
            self.assertEqual(vl.validate_dataset(self.dataset)["errors"], [vl.BUSY_MESSAGE])
            code, out, _ = self.run_main("--status")
            self.assertEqual(code, 1)
            self.assertNotIn("Pass 1:", out)
        self.signal("go")
        self.assertIn("restored", self.finish(proc))
        report = vl.validate_dataset(self.dataset)
        self.assertEqual(report["errors"], [])                             # no marker, no mix
        self.assertEqual(self.files(), self.good)

    def test_a_killed_restore_is_reported_at_once_and_can_be_finished(self):
        self.damage()
        proc = self.child(self.RESTORE_PAUSED)
        self.wait_for("mid", proc=proc)
        proc.kill()
        proc.communicate()
        report = vl.validate_dataset(self.dataset)
        self.assertIn("restore of pass 1 was interrupted while restoring the labels file",
                      report["errors"][0])
        self.assertTrue(report["stats"])
        code, out, _ = self.run_main("--status")
        self.assertEqual(code, 1)
        self.assertIn("Pass 1:", out)
        self.assertIn(".restore_in_progress", out)
        self.assert_blind(out)
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("A restore of pass 1 was interrupted", out)
        code, out, _ = self.run_main("--restore-latest-backup")
        self.assertEqual(code, 0, out)
        self.assertFalse(os.path.lexists(self.marker))
        self.assertEqual(self.files(), self.good)
        self.assertEqual(vl.validate_dataset(self.dataset)["errors"], [])

    def test_a_save_gives_up_cleanly_while_a_reader_holds_the_data_lock(self):
        proc = self.child(self.HOLD_DATA)
        self.wait_for("held", proc=proc)
        before, backups = self.files(), ad.list_backups(self.dataset, 1)
        with mock.patch.object(vl, "LOCK_TIMEOUT", 0.3):
            with self.assertRaises(AnnotationError) as caught:
                ad.save_image(self.dataset, 1, self.persons[0], hidden_right_eye(), False,
                              h.T0 + timedelta(hours=1))
        self.assertIn("busy", str(caught.exception))
        self.assertEqual(self.files(), before)
        self.assertEqual(ad.list_backups(self.dataset, 1), backups)
        self.signal("release")
        self.finish(proc)

    def test_status_redaction_uses_the_snapshot_names_only(self):
        self.status.write_bytes(self.status.read_bytes().replace(b",labelled,1,2", b",labelled,1,1", 1))
        with mock.patch.object(vl, "load_metadata", wraps=vl.load_metadata) as loads:
            code, out, _ = self.run_main("--status")
        self.assertEqual(code, 0)
        self.assertIn("ERROR", out)
        self.assertEqual(loads.call_count, 1)                              # inside the snapshot
        self.assert_blind(out)
        with mock.patch.object(vl, "consistent_read", wraps=vl.consistent_read) as reads:
            text = ad.blind_text(f"see {self.persons[0]}", self.dataset)
        self.assertTrue(reads.called)
        self.assertEqual(text, "see an image")


# ===================================================================== redaction
class RedactionTests(SafetyCase):
    """Real OSErrors carry full paths (with doubled backslashes in their text on Windows):
    none of that may reach the terminal or the annotation window."""

    def setUp(self):
        super().setUp()
        h.annotate_everything(self.dataset)

    def path_forms(self):
        forms = set()
        for p in (self.dataset, self.dataset.resolve(), self.root, self.root.resolve()):
            text = str(p)
            forms |= {text, p.as_posix(), text.replace("\\", "\\\\"), repr(text)[1:-1]}
        return forms

    def assert_no_paths(self, text):
        for form in self.path_forms():
            self.assertNotIn(form, text)
        self.assertNotIn("\\\\", text)                                    # no escaped path at all

    def replace_failing_for(self, name):
        """os.replace raising a real Windows-style OSError that names both paths."""
        real = os.replace

        def replace(src, dst, *a, **kw):
            if Path(dst).name == name:
                raise PermissionError(13, "Access is denied", str(src), None, str(dst))
            return real(src, dst, *a, **kw)
        return mock.patch.object(ad.os, "replace", side_effect=replace)

    def test_a_real_os_error_during_restore_never_shows_a_path(self):
        ad.backup_files(self.dataset, 1, h.T0 + timedelta(hours=1), tag="session")
        self.labels.write_bytes(b"damaged")
        for failing, expected in ((self.labels.name, "Cannot write labels.csv: Access is denied"),
                                  (vl.RESTORE_MARKER,
                                   "Cannot write .restore_in_progress: Access is denied")):
            with self.subTest(failing=failing):
                before = self.files()
                with self.replace_failing_for(failing):
                    # the injected error really does carry the paths (escaped, as on Windows)
                    with self.assertRaises(PermissionError) as raw:
                        os.replace(str(self.dataset / "x"), str(self.dataset / failing))
                    self.assertIn(str(self.dataset).replace("\\", "\\\\")
                                  if os.name == "nt" else str(self.dataset), str(raw.exception))
                    code, out, _ = self.run_main("--restore-latest-backup")
                self.assertEqual(code, 1)
                self.assertIn(expected, out)
                self.assert_no_paths(out)
                self.assert_blind(out)
                self.assertEqual(self.files(), before)                     # rolled back
                self.assertFalse(os.path.lexists(self.marker))

    def test_a_real_os_error_during_a_save_never_shows_a_path_in_the_window(self):
        person = self.persons[0]
        others = [n for n in self.names if n != person]
        set_state(self.labels, None)
        set_state(self.status, None)
        session = ad.Session(self.dataset, 1, [person] + others)          # no window is opened
        self.assertEqual(session.filename(), person)
        session.annotation = h.full_annotation()
        before = self.files()
        with self.replace_failing_for(self.labels.name):
            session.handle_key(ad.KEYS_ENTER[0])
        message = session.message
        self.assertTrue(message.startswith("not saved:"), message)
        self.assertIn("Cannot write labels.csv: Access is denied", message)
        self.assert_no_paths(message)
        self.assert_blind(message)
        self.assertEqual(self.files(), before)                             # nothing written

    def test_a_save_validation_error_in_the_window_names_no_image(self):
        person = self.persons[0]
        others = [n for n in self.names if n != person]
        session = ad.Session(self.dataset, 1, [person] + others)          # no window is opened
        session.annotation = hidden_right_eye()                            # a real change
        before = self.files()
        failure = [f"{person}: has 2 face boxes (at most one allowed)",
                   f"{person}: left eye point is not inside the face box "
                   f"(see {self.dataset / 'images' / person})"]
        with mock.patch.object(ad.vl, "validate_image", return_value=failure) as check:
            session.handle_key(ad.KEYS_ENTER[0])
        self.assertTrue(check.called)
        message = session.message
        self.assertEqual(message.split(";")[0],
                         "not saved: an image: has 2 face boxes (at most one allowed)")
        self.assertIn("left eye point is not inside the face box", message)  # still useful
        hidden = set(self.names) | {Path(n).stem for n in self.names}
        hidden |= {m["condition"] for m in self.meta.values()}
        hidden |= {m["participant_id"] for m in self.meta.values()}
        for word in hidden:
            self.assertNotIn(word, message)
        self.assert_no_paths(message)
        self.assert_blind(message)
        self.assertEqual(self.files(), before)                             # nothing saved
        self.assertEqual(session.filename(), person)                       # still on the image

    def test_an_unreadable_image_error_names_no_image(self):
        for name in self.names:
            (self.dataset / "images" / name).write_bytes(b"not an image")
        session = ad.Session(self.dataset, 1, list(self.names))
        hidden = set(self.names) | {Path(n).stem for n in self.names}
        hidden |= {m["condition"] for m in self.meta.values()}
        hidden |= {m["participant_id"] for m in self.meta.values()}
        hidden |= {m["sub_condition"] for m in self.meta.values()}
        with self.assertRaises(AnnotationError) as caught:
            session.image()
        message = str(caught.exception)
        self.assertEqual(message, "an image is unreadable or not 640x480; check the image file "
                                  "and confirm it is a readable 640x480 image")
        self.assertNotIn("validate_labels", message)
        for word in hidden:
            self.assertNotIn(word, message)
        out = io.StringIO()                                                # through main()
        with mock.patch.object(ad.Session, "run", lambda self: self.image()), \
                mock.patch.object(ad, "check_gitignore"), contextlib.redirect_stdout(out):
            code = ad.main(["--dataset", str(self.dataset)])
        self.assertEqual(code, 1)
        self.assertIn("an image is unreadable or not 640x480", out.getvalue())
        for word in hidden:
            self.assertNotIn(word, out.getvalue())
        self.assert_no_paths(out.getvalue())

    def test_main_passes_annotation_errors_through_blind_text(self):
        error = AnnotationError(f"boom near {self.names[0]} in {self.dataset}")
        with mock.patch.object(ad, "start_session_checks", side_effect=error), \
                mock.patch.object(ad, "blind_text", wraps=ad.blind_text) as wrapped:
            code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        wrapped.assert_called_once()
        self.assertNotIn(self.names[0], out)
        self.assertNotIn(str(self.dataset), out)
        self.assertIn("an image", out)
        self.assertIn("the dataset folder", out)

    def test_practice_mode_errors_never_read_the_real_dataset(self):
        with mock.patch.object(ad, "start_session_checks",
                               side_effect=AnnotationError("practice problem")), \
                mock.patch.object(vl, "consistent_read",
                                  side_effect=AssertionError("read a dataset")) as reads:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = ad.main(["--practice"])
        self.assertEqual(code, 1)
        self.assertIn("practice problem", out.getvalue())
        reads.assert_not_called()


if __name__ == "__main__":
    unittest.main()
