"""Validate the ground-truth annotation files (Milestone 7).

Follows docs/EVALUATION_PROTOCOL.md section 6. This module only reads files: it never
changes images, metadata or labels, and it never runs a detector.

    python scripts/validate_labels.py                 pass 1, every image must be annotated
    python scripts/validate_labels.py --partial       pass 1, allow images not annotated yet
    python scripts/validate_labels.py --pass 2        the repeat check (14 images)

Files (inside the Git-ignored dataset/ folder):
    labels.csv / labels_pass2.csv                    one row per face / eye
    annotation_status.csv / annotation_status_pass2.csv   one row per annotated image
    .restore_in_progress     exists only while a restore runs, or after one was interrupted:
                             it is always reported as an error (never changed here)
    .annotation.lock         lock file of scripts/annotate_dataset.py (byte 0: one running
                             instance; byte 1: data lock around every write). The validator
                             takes the data lock only to read one consistent snapshot; it never
                             creates the lock file or writes anything. The writer-side helpers
                             below (ensure_lock_file, data_lock, acquire_lock_byte) are used by
                             scripts/annotate_dataset.py only.
"""
import argparse
import contextlib
import csv
import errno
import hashlib
import os
import secrets
import stat
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import msvcrt                               # Windows: the tested locking mechanism
except ImportError:                             # pragma: no cover - other systems
    msvcrt = None
    import fcntl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

IMAGE_W, IMAGE_H = 640, 480
LABEL_FIELDS = ["filename", "object_type", "x", "y", "w", "h", "side", "uncertain"]
STATUS_FIELDS = ["filename", "status", "n_face", "n_eye", "annotated_at", "pass", "tool_version"]
STATUSES = ("labelled", "no_face")
SIDES = ("left", "right")                      # relative to the image (left = image left)
EYE_MARGIN_FRACTION = 0.05                     # eye may lie this fraction of the face width
                                               # outside the face box ("small margin", protocol 6.5)
# metadata.csv (written by scripts/capture_dataset.py). All six columns must exist.
METADATA_FIELDS = ["filename", "participant_id", "condition", "sub_condition",
                   "capture_time", "notes"]
METADATA_REQUIRED_VALUES = ("filename", "participant_id", "condition", "sub_condition",
                            "capture_time")                    # notes may be empty
# Allowed condition -> sub-conditions (docs/EVALUATION_PROTOCOL.md section 3; the same sets as
# capture_dataset.CONDITIONS, and the same conditions as build_dedup_manifest.CONDITION_ORDER:
# tests/test_validate_metadata.py checks that the three copies agree).
CONDITION_SUBS = {
    "normal": ("frontal",),
    "dim": ("dim",),
    "strong": ("side", "back"),
    "turn": ("30deg", "60deg"),
    "tilt": ("roll", "down"),
    "dist": ("near", "far"),
    "glasses": ("glasses",),
    "bg": ("background",),
}
PARTICIPANTS = ("P01", "P02", "NONE")          # NONE = background-only images (condition bg)
PASS_FILES = {1: ("labels.csv", "annotation_status.csv"),
              2: ("labels_pass2.csv", "annotation_status_pass2.csv")}
# Fixed seeds (protocol 6.2 and 6.9). Changing them changes the blind order / repeat sample,
# so they must stay fixed once labelling has started.
ORDER_SEED = 20261120
REPEAT_SEED = 20261121
REPEAT_SAMPLE_SIZE = 14


class ValidationError(Exception):
    """A file is unreadable or malformed (a structural problem, not a rule violation)."""


# ------------------------------------------------------- restore-in-progress marker
RESTORE_MARKER = ".restore_in_progress"
MARKER_PHASES = ("backup", "labels", "status")
MARKER_READ_LIMIT = 64
PHASE_TEXT = {
    "backup": "making the pre-restore backups (the label files were not changed yet)",
    "labels": "restoring the labels file (the status file was not changed yet)",
    "status": "restoring the status file (the labels file was already restored)",
}


def restore_marker_path(dataset_dir):
    return Path(dataset_dir) / RESTORE_MARKER


def format_restore_marker(pass_no, phase):
    """The exact bytes of a marker: 'pass=<1|2>' and 'phase=<word>', ASCII, LF line ends."""
    if pass_no not in PASS_FILES or phase not in MARKER_PHASES:
        raise ValueError(f"bad marker fields: pass {pass_no!r}, phase {phase!r}")
    return f"pass={pass_no}\nphase={phase}\n".encode("ascii")


def read_restore_marker(dataset_dir):
    """(present, pass_no or None, phase or None). Read-only.

    Presence depends only on existence (a broken link, a directory or an unreadable file is
    present too). The pass and phase are returned only when the whole marker is exactly the
    format written by format_restore_marker; anything else (empty, truncated, unknown phase,
    CRLF, too long, not ASCII) gives (True, None, None)."""
    path = restore_marker_path(dataset_dir)
    if not os.path.lexists(path):
        return False, None, None
    if os.path.islink(path) or not path.is_file():
        return True, None, None
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MARKER_READ_LIMIT + 1)
    except OSError:
        return True, None, None
    for pass_no in PASS_FILES:
        for phase in MARKER_PHASES:
            if raw == format_restore_marker(pass_no, phase):
                return True, pass_no, phase
    return True, None, None


def restore_marker_message(pass_no, phase):
    """Blind text (no image names, no paths) about an interrupted restore."""
    if pass_no is None:
        return ("a restore was interrupted, but .restore_in_progress cannot be read, so the "
                "affected pass cannot be determined. The label files of either pass may mix "
                "old and restored data. Automatic restore is blocked for both passes; manual "
                "recovery is required (see the README section \"Interrupted restore\").")
    return (f"a restore of pass {pass_no} was interrupted while {PHASE_TEXT[phase]} "
            "(.restore_in_progress in the dataset folder). The pass "
            f"{pass_no} label files may mix old and restored data. Run "
            f"scripts/annotate_dataset.py --pass {pass_no} --restore-latest-backup to finish it.")


def marker_error(marker_state):
    present, pass_no, phase = marker_state
    if not present:
        return None
    return f"{RESTORE_MARKER}: " + restore_marker_message(pass_no, phase)


def restore_marker_error(dataset_dir):
    return marker_error(read_restore_marker(dataset_dir))


# --------------------------------------------------------------- dataset locks
LOCK_NAME = ".annotation.lock"
INSTANCE_BYTE, DATA_BYTE = 0, 1
LOCK_TIMEOUT = 10.0                  # seconds to wait for the data lock (read at call time)
LOCK_POLL = 0.05
BUSY_MESSAGE = ("dataset busy: a save or restore is running (or another check is reading); "
                "nothing was checked, try again")
_HELD = threading.local()


class LockBusy(Exception):
    """The lock is held by someone else and did not become free in time."""


class LockUnusable(Exception):
    """The lock file exists but cannot be used (not a regular file, a link, unreadable)."""


def lock_path(dataset_dir):
    return Path(dataset_dir) / LOCK_NAME


def ensure_lock_file(dataset_dir):
    """Create the lock file if it does not exist yet. Never truncates, replaces or pads an
    existing one (empty and one-byte files stay as they are: Windows locks bytes past the
    end of a file).

    A new file appears fully formed (two zero bytes) through a hard link, which fails if the
    name already exists; on a file system without hard links an exclusive create gives an
    empty file instead. Either way every process ends up opening the same single file."""
    path = lock_path(dataset_dir)
    if os.path.lexists(path):
        return path
    tmp = path.with_name(f"{LOCK_NAME}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    made_tmp = False
    try:
        with open(tmp, "xb") as handle:
            made_tmp = True
            handle.write(b"\0\0")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    except OSError:
        pass                         # exists already (another process won) or no hard links
    if not os.path.lexists(path):
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR
                             | getattr(os, "O_BINARY", 0)))
        except FileExistsError:
            pass
        except OSError as error:
            if made_tmp:
                _try_unlink(tmp)
            raise LockUnusable(f"cannot create the lock file "
                               f"({error.strerror or type(error).__name__})")
    if made_tmp:
        _try_unlink(tmp)             # a failure only leaves a counted, harmless leftover
    return path


def _try_unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _open_lock_file(path, flags):
    """Open the existing lock file (never creates). Raises FileNotFoundError if missing and
    LockUnusable if it is a link, a directory or not the file at that name."""
    fd = os.open(path, flags | getattr(os, "O_BINARY", 0))
    try:
        st, lst = os.fstat(fd), os.lstat(path)
        if (not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(lst.st_mode)
                or (st.st_dev, st.st_ino) != (lst.st_dev, lst.st_ino)):
            raise LockUnusable("the lock file is not a regular file")
    except OSError as error:
        os.close(fd)
        raise LockUnusable(f"cannot check the lock file "
                           f"({error.strerror or type(error).__name__})")
    except LockUnusable:
        os.close(fd)
        raise
    return fd


def _try_lock(fd, byte, shared):
    """One non-blocking attempt. True if locked, False if someone else holds it."""
    try:
        if msvcrt is not None:
            os.lseek(fd, byte, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)        # exclusive only on Windows
        else:                                             # pragma: no cover - untested
            fcntl.lockf(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB,
                        1, byte, os.SEEK_SET)
        return True
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            return False
        raise LockUnusable(f"cannot lock the lock file "
                           f"({error.strerror or type(error).__name__})")


def _unlock(fd, byte):
    if msvcrt is not None:
        os.lseek(fd, byte, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:                                                 # pragma: no cover - untested
        fcntl.lockf(fd, fcntl.LOCK_UN, 1, byte, os.SEEK_SET)


def _wait_lock(fd, byte, timeout, shared=False):
    end = time.monotonic() + max(0.0, timeout)
    while True:
        if _try_lock(fd, byte, shared):
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(LOCK_POLL)


class LockHandle:
    """One locked byte of the lock file; release() (or closing the process) frees it."""

    def __init__(self, fd, byte):
        self.fd, self.byte = fd, byte

    def release(self):
        if self.fd is None:
            return
        try:
            _unlock(self.fd, self.byte)
        except OSError:
            pass                                          # closing the handle frees it too
        finally:
            os.close(self.fd)
            self.fd = None


def acquire_lock_byte(dataset_dir, byte, timeout):
    """Writer side: create the lock file if needed, then lock one byte. Raises LockBusy."""
    path = ensure_lock_file(dataset_dir)
    try:
        fd = _open_lock_file(path, os.O_RDWR)
    except FileNotFoundError:
        raise LockUnusable("the lock file disappeared")
    except OSError as error:
        raise LockUnusable(f"cannot open the lock file "
                           f"({error.strerror or type(error).__name__})")
    try:
        if not _wait_lock(fd, byte, timeout):
            raise LockBusy(BUSY_MESSAGE)
    except BaseException:
        os.close(fd)
        raise
    return LockHandle(fd, byte)


def _held():
    return getattr(_HELD, "data", False)


@contextlib.contextmanager
def data_lock(dataset_dir, timeout=None):
    """Writer side: hold the data lock (byte 1) around a group of writes."""
    if _held():
        raise RuntimeError("the data lock is already held by this thread")
    handle = acquire_lock_byte(dataset_dir, DATA_BYTE,
                               LOCK_TIMEOUT if timeout is None else timeout)
    _HELD.data = True
    try:
        yield
    finally:
        _HELD.data = False
        handle.release()


def _lock_file_exists(path):
    return os.path.lexists(path)


def consistent_read(dataset_dir, read, timeout=None):
    """Reader side: run read() so that it sees one consistent state of the dataset files.

    If the lock file exists, read() runs while holding the data lock (exclusive on Windows,
    so readers take turns). If it does not exist, read() runs unlocked and the result is
    kept only if the lock file is still missing afterwards: writers create it before their
    first write and never delete it, so no writer changed anything meanwhile. Otherwise the
    result is discarded and read() runs again under the lock. Never creates or writes
    anything. Raises LockBusy (timeout) or LockUnusable."""
    if _held():
        raise RuntimeError("a consistent read was requested while holding the data lock")
    timeout = LOCK_TIMEOUT if timeout is None else timeout
    path = lock_path(dataset_dir)
    for attempt in (1, 2):
        try:
            fd = _open_lock_file(path, os.O_RDONLY)
        except FileNotFoundError:
            if _lock_file_exists(path) or attempt == 2:   # broken link, or vanished
                raise LockUnusable("the lock file cannot be opened")
            result = read()
            if not _lock_file_exists(path):
                return result
            continue                                      # a writer started: read again
        except LockUnusable:
            raise
        except OSError as error:
            raise LockUnusable(f"cannot open the lock file "
                               f"({error.strerror or type(error).__name__})")
        try:
            if not _wait_lock(fd, DATA_BYTE, timeout, shared=True):
                raise LockBusy(BUSY_MESSAGE)
            try:
                return read()
            finally:
                try:
                    _unlock(fd, DATA_BYTE)
                except OSError:
                    pass
        finally:
            os.close(fd)
    raise LockUnusable("the lock file cannot be opened")  # pragma: no cover


# ---------------------------------------------------------------- helpers
def label_paths(dataset_dir, pass_no):
    labels, status = PASS_FILES[pass_no]
    return Path(dataset_dir) / labels, Path(dataset_dir) / status


def stable_order(filenames, seed):
    """Deterministic pseudo-random order that does not depend on the input order or on
    the Python version: sort by SHA-256 of '<seed>:<filename>'."""
    return sorted(filenames, key=lambda n: hashlib.sha256(f"{seed}:{n}".encode()).hexdigest())


def annotation_order(filenames, seed=ORDER_SEED):
    return stable_order(filenames, seed)


def repeat_sample(person_filenames, seed=REPEAT_SEED, size=REPEAT_SAMPLE_SIZE):
    """The 14 person images of the repeat check (random but fixed)."""
    return stable_order(person_filenames, seed)[:size]


def _check_metadata_row(row, line, seen_filenames):
    """Raise ValidationError if one metadata row is unusable.

    Messages are blind: they give the line number and the column, never a filename, a
    filename stem or a path, and they do not repeat the offending value. Nothing is
    stripped or normalised: ' normal' is simply not a known condition."""
    where = f"metadata.csv line {line}"
    if None in row or any(value is None for value in row.values()):
        raise ValidationError(f"{where}: wrong number of values (incomplete or damaged row)")
    if any("\x00" in value for value in row.values()):
        raise ValidationError(f"{where}: contains a NUL character (damaged file)")
    for column in METADATA_REQUIRED_VALUES:
        if row[column].strip() == "":
            raise ValidationError(f"{where}: {column} is empty")
    condition, participant = row["condition"], row["participant_id"]
    if condition not in CONDITION_SUBS:
        raise ValidationError(f"{where}: condition is not one of {sorted(CONDITION_SUBS)}")
    if row["sub_condition"] not in CONDITION_SUBS[condition]:
        raise ValidationError(f"{where}: sub_condition is not allowed for this condition")
    if participant not in PARTICIPANTS:
        raise ValidationError(f"{where}: participant_id must be one of {list(PARTICIPANTS)}")
    if (condition == "bg") != (participant == "NONE"):
        raise ValidationError(f"{where}: participant_id does not match the condition type "
                              "(NONE only with bg; P01 or P02 only with person conditions)")
    first = seen_filenames.get(row["filename"])
    if first is not None:
        raise ValidationError(f"{where}: filename repeats the one on line {first}")
    seen_filenames[row["filename"]] = line


def load_metadata(dataset_dir):
    """filename -> metadata row. Raises ValidationError if unusable.

    Checks the six-column schema (no duplicate column names), every row (number of values,
    empty values, NUL characters, known condition and sub-condition, participant that fits
    the condition) and duplicate filenames. Valid files give exactly the rows a plain
    csv.DictReader gives (an optional UTF-8 BOM, LF or CRLF line ends and quoted notes are
    fine); no value is stripped or changed. Error messages never contain filenames or paths.
    """
    path = Path(dataset_dir) / "metadata.csv"
    rows, seen_filenames, reader = {}, {}, None
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            header = reader.fieldnames
            if header is not None and len(set(header)) != len(header):
                repeated = sorted({c for c in header if header.count(c) > 1})
                raise ValidationError(f"metadata.csv: duplicate column name(s) {repeated}")
            if header is None or any(column not in header for column in METADATA_FIELDS):
                raise ValidationError(f"metadata.csv is empty or lacks the columns "
                                      f"{sorted(METADATA_FIELDS)}")
            if any("\x00" in column for column in header):
                raise ValidationError("metadata.csv line 1: contains a NUL character "
                                      "(damaged file)")
            for row in reader:
                _check_metadata_row(row, reader.line_num, seen_filenames)
                rows[row["filename"]] = row
    except OSError as error:                    # error.strerror has no path in it
        raise ValidationError(f"Cannot read metadata.csv: "
                              f"{error.strerror or type(error).__name__}")
    except UnicodeDecodeError as error:
        after = f" (after line {reader.line_num})" if reader is not None and reader.line_num else ""
        raise ValidationError(f"Cannot read metadata.csv{after}: not valid UTF-8 "
                              f"({error.reason} at byte {error.start})")
    except csv.Error as error:
        after = f" after line {reader.line_num}" if reader is not None and reader.line_num else ""
        raise ValidationError(f"Cannot read metadata.csv: malformed CSV{after} ({error})")
    if not rows:
        raise ValidationError("metadata.csv is empty or lacks the columns "
                              f"{sorted(METADATA_FIELDS)}")
    return rows


def is_background(meta_row):
    return meta_row["condition"] == "bg" or meta_row["participant_id"] == "NONE"


def read_csv_strict(path, fields):
    """Rows of a label/status file ([] if the file is missing). Rejects wrong headers and
    rows with the wrong number of values, naming the line."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fields:
                raise ValidationError(f"{path.name}: columns {reader.fieldnames} differ from "
                                      f"the required {fields}")
            rows = []
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValidationError(f"{path.name} line {reader.line_num}: wrong number "
                                          "of values (incomplete or damaged row)")
                row["_line"] = reader.line_num
                rows.append(row)
            return rows
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValidationError(f"Cannot read {path.name}: {error}")


def _int(value, what, where):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{where}: {what} '{value}' is not an integer")


def parse_label_row(row, name="labels.csv"):
    """Typed copy of one labels row; raises ValidationError for malformed fields."""
    where = f"{name} line {row['_line']}"
    kind = row["object_type"]
    uncertain = row["uncertain"]
    if uncertain not in ("0", "1"):
        raise ValidationError(f"{where}: uncertain '{uncertain}' must be 0 or 1")
    out = {"filename": row["filename"], "object_type": kind, "uncertain": int(uncertain),
           "_line": row["_line"]}
    if kind == "face":
        out.update(x=_int(row["x"], "x", where), y=_int(row["y"], "y", where),
                   w=_int(row["w"], "w", where), h=_int(row["h"], "h", where))
        if row["side"] != "":
            raise ValidationError(f"{where}: a face row must have an empty side")
    elif kind == "eye":
        out.update(x=_int(row["x"], "x", where), y=_int(row["y"], "y", where), side=row["side"])
        if row["w"] != "" or row["h"] != "":
            raise ValidationError(f"{where}: an eye row must have empty w and h")
        if row["side"] not in SIDES:
            raise ValidationError(f"{where}: side '{row['side']}' must be left or right")
    else:
        raise ValidationError(f"{where}: object_type '{kind}' must be face or eye")
    return out


def parse_status_row(row, pass_no, name="annotation_status.csv"):
    where = f"{name} line {row['_line']}"
    if row["status"] not in STATUSES:
        raise ValidationError(f"{where}: status '{row['status']}' must be one of {STATUSES}")
    out = {"filename": row["filename"], "status": row["status"],
           "n_face": _int(row["n_face"], "n_face", where),
           "n_eye": _int(row["n_eye"], "n_eye", where),
           "annotated_at_text": row["annotated_at"], "tool_version": row["tool_version"],
           "_line": row["_line"]}
    try:
        stamp = datetime.fromisoformat(row["annotated_at"])
    except ValueError:
        raise ValidationError(f"{where}: annotated_at '{row['annotated_at']}' is not an ISO date-time")
    if stamp.tzinfo is None:
        raise ValidationError(f"{where}: annotated_at must include the UTC offset")
    out["annotated_at"] = stamp
    if row["pass"] != str(pass_no):
        raise ValidationError(f"{where}: pass '{row['pass']}' should be {pass_no}")
    if not row["tool_version"]:
        raise ValidationError(f"{where}: tool_version is empty")
    return out


# --------------------------------------------------------- per-image rules
def eye_in_face(eye, face):
    margin = EYE_MARGIN_FRACTION * face["w"]
    return (face["x"] - margin <= eye["x"] <= face["x"] + face["w"] + margin
            and face["y"] - margin <= eye["y"] <= face["y"] + face["h"] + margin)


def validate_image(filename, faces, eyes, status, background):
    """Rule violations for one image (list of strings; empty = valid).

    faces / eyes: typed label dicts of this image. status: typed status dict or None.
    """
    errors = []
    tag = f"{filename}: "
    if status is None:
        return [tag + "has no status row (not annotated)"]
    if len(faces) > 1:
        errors.append(tag + f"has {len(faces)} face boxes (at most one allowed)")
    if len(eyes) > 2:
        errors.append(tag + f"has {len(eyes)} eye points (at most two allowed)")
    for face in faces:
        if face["w"] <= 0 or face["h"] <= 0:
            errors.append(tag + "face box has non-positive size")
        if (face["x"] < 0 or face["y"] < 0 or face["x"] + face["w"] > IMAGE_W
                or face["y"] + face["h"] > IMAGE_H):
            errors.append(tag + "face box is outside the 640x480 image")
    sides = [e["side"] for e in eyes]
    if len(set(sides)) != len(sides):
        errors.append(tag + "the same eye side is used more than once")
    for eye in eyes:
        if not (0 <= eye["x"] < IMAGE_W and 0 <= eye["y"] < IMAGE_H):
            errors.append(tag + f"{eye['side']} eye point is outside the image")
    if eyes and not faces:
        errors.append(tag + "has eye points but no face")
    if len(faces) == 1:
        for eye in eyes:
            if not eye_in_face(eye, faces[0]):
                errors.append(tag + f"{eye['side']} eye point is not inside the face box")
    by_side = {e["side"]: e for e in eyes}
    if "left" in by_side and "right" in by_side and by_side["left"]["x"] >= by_side["right"]["x"]:
        errors.append(tag + "the left eye must be left of the right eye in the image")
    if background and (faces or eyes):
        errors.append(tag + "is a background-only image but has face or eye annotations")
    if background and status["status"] != "no_face":
        errors.append(tag + "background-only images must have status no_face")
    if status["status"] == "no_face" and (faces or eyes):
        errors.append(tag + "has status no_face but also has annotations")
    if status["status"] == "labelled" and len(faces) != 1:
        errors.append(tag + "has status labelled but not exactly one face")
    if status["n_face"] != len(faces) or status["n_eye"] != len(eyes):
        errors.append(tag + f"status counts (face {status['n_face']}, eye {status['n_eye']}) "
                            f"disagree with the labels (face {len(faces)}, eye {len(eyes)})")
    return errors


# ------------------------------------------------------------ whole dataset
def read_pass_snapshot(dataset_dir, pass_no):
    """Everything one validation needs, read in one go (call it through consistent_read):
    the marker state, metadata, the labels and status rows of the pass and the image list.
    A structural problem is recorded in 'error' instead of being raised."""
    dataset_dir = Path(dataset_dir)
    snap = {"marker": read_restore_marker(dataset_dir), "names": [], "meta": None,
            "label_rows": None, "status_rows": None, "on_disk": set(), "error": None}
    try:
        meta = load_metadata(dataset_dir)
        snap["meta"], snap["names"] = meta, list(meta)
        label_path, status_path = label_paths(dataset_dir, pass_no)
        snap["label_rows"] = [parse_label_row(r, label_path.name)
                              for r in read_csv_strict(label_path, LABEL_FIELDS)]
        snap["status_rows"] = [parse_status_row(r, pass_no, status_path.name)
                               for r in read_csv_strict(status_path, STATUS_FIELDS)]
    except ValidationError as error:
        snap["error"] = str(error)
        return snap
    image_dir = dataset_dir / "images"
    snap["on_disk"] = {p.name for p in image_dir.iterdir()} if image_dir.is_dir() else set()
    return snap


def validate_dataset(dataset_dir, pass_no=1, require_complete=True):
    """Check one pass. Returns {'errors': [...], 'warnings': [...], 'stats': {...}}."""
    return validate_dataset_snapshot(dataset_dir, pass_no, require_complete)[0]


def validate_dataset_snapshot(dataset_dir, pass_no=1, require_complete=True):
    """Check one pass from one consistent snapshot. Returns (report, info) where info is
    {'names': metadata filenames of that snapshot (for blind messages), 'marker': (present,
    pass, phase)}. If the dataset is busy or the lock file is unusable, the report holds only
    that error and no statistics."""
    dataset_dir = Path(dataset_dir)
    report = {"errors": [], "warnings": [], "stats": {}}
    info = {"names": [], "marker": (False, None, None)}
    errors, warnings = report["errors"], report["warnings"]
    try:
        snap = consistent_read(dataset_dir, lambda: read_pass_snapshot(dataset_dir, pass_no))
    except LockBusy:
        errors.append(BUSY_MESSAGE)
        return report, info
    except LockUnusable as error:
        errors.append(f"{LOCK_NAME}: {error}; nothing was checked")
        return report, info
    info.update(names=snap["names"], marker=snap["marker"])
    marker = marker_error(snap["marker"])
    if marker:
        errors.append(marker)
    if snap["error"] is not None:
        errors.append(snap["error"])
        return report, info
    meta, label_rows, status_rows = snap["meta"], snap["label_rows"], snap["status_rows"]
    on_disk = snap["on_disk"]
    if on_disk != set(meta):
        errors.append("images folder and metadata.csv do not list the same files")
    status_by = {}
    for r in status_rows:
        if r["filename"] in status_by:
            errors.append(f"{r['filename']}: more than one status row (line {r['_line']})")
        status_by[r["filename"]] = r
    for name in status_by:
        if name not in meta:
            errors.append(f"{name}: status row for a file that is not in the dataset")
    faces, eyes = {}, {}
    for r in label_rows:
        if r["filename"] not in meta:
            errors.append(f"{r['filename']}: label row (line {r['_line']}) references a file "
                          "that is not in the dataset")
            continue
        (faces if r["object_type"] == "face" else eyes).setdefault(r["filename"], []).append(r)
    for name in sorted(set(faces) | set(eyes)):
        if name not in status_by:
            errors.append(f"{name}: has label rows but no status row (interrupted save); "
                          "re-annotate this image")
    expected = set(meta)
    if pass_no == 2:
        person = [n for n, m in meta.items() if not is_background(m)]
        expected = set(repeat_sample(person))
        for name in status_by:
            if name in meta and name not in expected:
                errors.append(f"{name}: is not one of the {REPEAT_SAMPLE_SIZE} repeat-check images")
    for name in sorted(status_by):
        if name in meta:
            errors.extend(validate_image(name, faces.get(name, []), eyes.get(name, []),
                                         status_by[name], is_background(meta[name])))
    missing = sorted(expected - set(status_by))
    if missing and require_complete:
        errors.append(f"{len(missing)} of {len(expected)} images have no status row "
                      f"(for example {missing[0]})")
    # warnings for the author to review (protocol 6.5); none of them uses a detector
    person_done = [n for n in status_by if n in meta and not is_background(meta[n])]
    for name in sorted(person_done):
        st = status_by[name]
        if st["status"] == "no_face":
            warnings.append(f"{name}: person image marked no_face")
        elif st["n_eye"] < 2:
            warnings.append(f"{name}: face labelled with {st['n_eye']} eye point(s)")
    by_cond = {}
    for name in person_done:
        for f in faces.get(name, []):
            by_cond.setdefault(meta[name]["condition"], []).append((name, f["w"]))
    for cond, items in by_cond.items():
        widths = [w for _, w in items]
        if len(widths) >= 3:
            med = statistics.median(widths)
            mad = statistics.median(abs(w - med) for w in widths) * 1.4826
            limit = max(3 * mad, 0.25 * med)
            for name, w in items:
                if abs(w - med) > limit:
                    warnings.append(f"{name}: face width {w}px is unusual for condition "
                                    f"'{cond}' (median {med:.0f}px)")
    for r in label_rows:
        if r["uncertain"]:
            warnings.append(f"{r['filename']}: uncertain {r['object_type']}"
                            + (f" ({r['side']})" if r["object_type"] == "eye" else ""))
    report["stats"] = {"pass": pass_no, "expected_images": len(expected),
                       "annotated_images": len([n for n in status_by if n in expected]),
                       "labelled": sum(s["status"] == "labelled" for s in status_by.values()),
                       "no_face": sum(s["status"] == "no_face" for s in status_by.values()),
                       "faces": sum(len(v) for v in faces.values()),
                       "eyes": sum(len(v) for v in eyes.values()),
                       "uncertain_objects": sum(r["uncertain"] for r in label_rows),
                       "missing": len(missing)}
    return report, info


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate annotation files (read-only).")
    parser.add_argument("--pass", dest="pass_no", type=int, choices=(1, 2), default=1)
    parser.add_argument("--partial", action="store_true",
                        help="do not require every image to be annotated yet")
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    args = parser.parse_args(argv)
    report = validate_dataset(args.dataset, args.pass_no, require_complete=not args.partial)
    for line in report["errors"]:
        print("ERROR  ", line)
    for line in report["warnings"]:
        print("WARNING", line)
    if report["stats"]:                         # none after a busy or unusable-lock result
        print("stats:", report["stats"])
    print("RESULT:", "FAILED" if report["errors"] else "OK")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
