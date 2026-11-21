"""Manual ground-truth annotation tool (Milestone 7).

Follows docs/EVALUATION_PROTOCOL.md section 6. It never runs or shows a detector, never
pre-fills labels, and hides filenames and condition names (blind, fixed-seed random order).

    python scripts/annotate_dataset.py --practice     learn the controls on synthetic images
    python scripts/annotate_dataset.py --status       progress only (no window)
    python scripts/annotate_dataset.py                annotate (pass 1); resumes automatically
    python scripts/annotate_dataset.py --pass 2       repeat check: 14 person images, only
                                                      24 h or more after pass 1 is finished

Mouse and keys (the window must be in focus):
    drag with the left button   draw the face box (top-left to bottom-right)
    click                       place an eye centre: first the eye on the LEFT of the image,
                                then the one on the RIGHT
    h        the next eye is hidden (not visible): no point is stored
    u        toggle "uncertain" on the last placed face/eye
    n        no face visible in this image (status no_face)
    f / l / r  redo the face box / the left eye / the right eye of a saved or drawn
             annotation: that one object is cleared and must be placed again
    z        undo the last action (repeat to undo more); also restores a redone object
    Enter    check the annotation and save it, then go to the next image
    b        go back to the previous image (press twice if this image has unsaved changes)
    q / ESC  quit (press twice if this image has unsaved changes). Saved images are kept.

Files (inside the Git-ignored dataset/ folder): labels.csv, annotation_status.csv and
timestamped copies in backups/. Pass 2 uses labels_pass2.csv / annotation_status_pass2.csv.
Only one session or restore may use a dataset at a time (.annotation.lock); a second one is
refused. .restore_in_progress exists only while a restore runs or after one was interrupted,
and blocks annotation until the restore is run again.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_labels as vl  # noqa: E402

TOOL_VERSION = "1.0"
DEFAULT_SCALE = 1.25                 # 640x480 shown at 800x600 (fits a 1366x768 screen)
STRIP_HEIGHT = 96                    # text area below the image
HELP_SCALE = 0.45                    # font size of the key list, so the whole line fits 800 px
MAG_REGION, MAG_SIZE = 41, 164       # magnifier: 41x41 original pixels shown 4x
MIN_FACE_PX = 10
DRAG_THRESHOLD_PX = 6
STARTUP_GRACE_FRAMES = 150           # ~4.5 s (30 ms per frame) for OpenCV to report the window visible
WINDOW_NAME = "Annotation  (Enter=save  z=undo  b=back  q=quit)"
KEYS_ENTER = (13, 10)
KEY_ESC = 27
GITIGNORE_PROBES = ("dataset/labels.csv", "dataset/annotation_status.csv",
                    "dataset/backups/probe.csv")


class AnnotationError(Exception):
    """A problem the annotator can understand and fix."""


# ================================================================ annotation data
class Annotation:
    """The annotation of ONE image, with undo. Pure data: no window, no files."""

    def __init__(self):
        self.face = None                               # (x, y, w, h) in original pixels
        self.eyes = {"left": None, "right": None}      # (x, y) or None
        self.hidden = {"left": False, "right": False}
        self.uncertain = {"face": 0, "left": 0, "right": 0}
        self.no_face = False
        self.last_object = None                        # "face" / "left" / "right"
        self._history = []

    # -- state helpers
    def fingerprint(self):
        return (self.face, tuple(sorted(self.eyes.items())), tuple(sorted(self.hidden.items())),
                tuple(sorted(self.uncertain.items())), self.no_face)

    def _snapshot(self):
        return (self.face, dict(self.eyes), dict(self.hidden), dict(self.uncertain),
                self.no_face, self.last_object)

    def _push(self):
        self._history.append(self._snapshot())

    def is_empty(self):
        return not (self.no_face or self.face or any(self.hidden.values())
                    or any(self.eyes.values()))

    def target(self):
        """What the next click means: 'face', 'left', 'right' or None (all decided)."""
        if self.no_face:
            return None
        if self.face is None:
            return "face"
        for side in vl.SIDES:
            if self.eyes[side] is None and not self.hidden[side]:
                return side
        return None

    def is_complete(self):
        return self.no_face or (self.face is not None and self.target() is None)

    def missing(self):
        t = self.target()
        if t == "face":
            return "draw the face box (drag), or press n if no face is visible"
        if t in vl.SIDES:
            return f"click the {t} eye (image {t}) or press h if it is hidden"
        return ""

    # -- actions
    def set_face(self, x0, y0, x1, y1):
        if self.no_face:
            raise AnnotationError("image is marked no_face: press z to undo that first")
        if self.face is not None:
            raise AnnotationError("a face box already exists: press z to undo it first")
        x0, x1 = sorted((max(0, min(vl.IMAGE_W, x0)), max(0, min(vl.IMAGE_W, x1))))
        y0, y1 = sorted((max(0, min(vl.IMAGE_H, y0)), max(0, min(vl.IMAGE_H, y1))))
        if x1 - x0 < MIN_FACE_PX or y1 - y0 < MIN_FACE_PX:
            raise AnnotationError(f"face box too small (at least {MIN_FACE_PX} px each way)")
        box = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
        for side in vl.SIDES:
            if self.eyes[side] is not None and not vl.eye_in_face(
                    {"x": self.eyes[side][0], "y": self.eyes[side][1]}, box):
                raise AnnotationError(f"the {side} eye would be outside the new face box: press "
                                      f"{side[0]} to redo that eye first, or z to undo")
        self._push()
        self.face = (x0, y0, x1 - x0, y1 - y0)
        self.last_object = "face"

    def place_eye(self, x, y):
        side = self.target()
        if side not in vl.SIDES:
            raise AnnotationError("nothing to place: " + (self.missing() or
                                  "press Enter to save, or z to undo"))
        x, y = int(x), int(y)
        if not (0 <= x < vl.IMAGE_W and 0 <= y < vl.IMAGE_H):
            raise AnnotationError("point is outside the image")
        fx, fy, fw, fh = self.face
        if not vl.eye_in_face({"x": x, "y": y}, {"x": fx, "y": fy, "w": fw, "h": fh}):
            raise AnnotationError("eye point is outside the face box")
        other = self.eyes["left" if side == "right" else "right"]
        if side == "right" and self.eyes["left"] is not None and x <= self.eyes["left"][0]:
            raise AnnotationError("the right eye must be to the right of the left eye "
                                  "(as seen in the image)")
        if side == "left" and other is not None and x >= other[0]:
            raise AnnotationError("the left eye must be to the left of the right eye")
        self._push()
        self.eyes[side] = (x, y)
        self.last_object = side

    def hide_eye(self):
        side = self.target()
        if side not in vl.SIDES:
            raise AnnotationError("no eye is waiting: " + (self.missing() or "already complete"))
        self._push()
        self.hidden[side] = True
        self.last_object = None

    def toggle_uncertain(self):
        obj = self.last_object
        if obj is None:
            raise AnnotationError("nothing to flag: place a face or an eye first")
        self._push()
        self.uncertain[obj] ^= 1

    def mark_no_face(self):
        if self.face is not None or any(self.eyes.values()) or any(self.hidden.values()):
            raise AnnotationError("this image already has annotations: press z to undo them "
                                  "first, then n")
        self._push()
        self.no_face = not self.no_face
        self.last_object = None

    def reopen(self, obj):
        """Clear ONE object ("face", "left" or "right") so that it can be placed again.

        This is how a saved or already drawn annotation is corrected. The old value is not
        lost: the full state is pushed first, so undo (z) restores it exactly, including the
        uncertain flag. Nothing else is touched. The annotation is incomplete until the object
        is placed again, so it cannot be saved half-corrected.
        """
        if self.no_face:
            raise AnnotationError("marked no_face: press n to turn that off first")
        if obj == "face":
            if self.face is None:
                raise AnnotationError("there is no face box to redo")
            self._push()
            self.face = None
            self.uncertain["face"] = 0
        elif obj in vl.SIDES:
            if self.eyes[obj] is None and not self.hidden[obj]:
                raise AnnotationError(f"the {obj} eye is not placed yet: just click it "
                                      "(or press h)")
            self._push()
            self.eyes[obj] = None
            self.hidden[obj] = False
            self.uncertain[obj] = 0
        else:
            raise AnnotationError(f"cannot redo '{obj}'")
        self.last_object = None

    def undo(self):
        if not self._history:
            raise AnnotationError("nothing to undo")
        (self.face, self.eyes, self.hidden, self.uncertain,
         self.no_face, self.last_object) = self._history.pop()

    # -- conversion to file records
    def records(self, filename):
        """(label rows as dicts of strings, status dict). Requires a complete annotation."""
        if not self.is_complete():
            raise AnnotationError("annotation incomplete: " + self.missing())
        rows = []
        if self.face and not self.no_face:
            x, y, w, h = self.face
            rows.append({"filename": filename, "object_type": "face", "x": str(x), "y": str(y),
                         "w": str(w), "h": str(h), "side": "",
                         "uncertain": str(self.uncertain["face"])})
            for side in vl.SIDES:
                if self.eyes[side] is not None:
                    ex, ey = self.eyes[side]
                    rows.append({"filename": filename, "object_type": "eye", "x": str(ex),
                                 "y": str(ey), "w": "", "h": "", "side": side,
                                 "uncertain": str(self.uncertain[side])})
        status = {"status": "no_face" if self.no_face else "labelled",
                  "n_face": 1 if rows else 0, "n_eye": max(len(rows) - 1, 0)}
        return rows, status

    @classmethod
    def from_saved(cls, faces, eyes, status):
        """Rebuild an annotation from saved rows (typed dicts). Missing eyes = hidden."""
        a = cls()
        if status["status"] == "no_face":
            a.no_face = True
            return a
        if len(faces) != 1 or len(eyes) > 2 or len({e["side"] for e in eyes}) != len(eyes):
            raise AnnotationError("the saved rows of this image are inconsistent "
                                  f"({len(faces)} face rows, {len(eyes)} eye rows); nothing "
                                  "was changed. Run scripts/validate_labels.py for details.")
        f = faces[0]
        a.face = (f["x"], f["y"], f["w"], f["h"])
        a.uncertain["face"] = f["uncertain"]
        for side in vl.SIDES:
            row = next((e for e in eyes if e["side"] == side), None)
            if row is None:
                a.hidden[side] = True
            else:
                a.eyes[side] = (row["x"], row["y"])
                a.uncertain[side] = row["uncertain"]
        return a


# ================================================================== file storage
def pass_paths(dataset_dir, pass_no):
    return vl.label_paths(dataset_dir, pass_no)


def backups_dir(dataset_dir):
    return Path(dataset_dir) / "backups"


def _csv_bytes(fields, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_atomic(path, data):
    """Write through a temporary file, fsync, then replace: a crash leaves either the old
    file or the new file, never half of one."""
    path = Path(path)
    if os.path.islink(path):
        raise AnnotationError(f"{path.name} is a link; refusing to write")
    tmp = path.with_name("." + path.name + ".tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as error:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise AnnotationError(f"Cannot write {path.name}: {_why(error)}")   # no folder path


def clean_stale_tmp(dataset_dir):
    """Remove leftover temporary files of an interrupted save (our own .tmp files only).
    Called only while this instance holds the dataset locks."""
    removed = []
    for name in (".labels.csv.tmp", ".annotation_status.csv.tmp",
                 ".labels_pass2.csv.tmp", ".annotation_status_pass2.csv.tmp",
                 "." + vl.RESTORE_MARKER + ".tmp"):
        path = Path(dataset_dir) / name
        if path.exists() and not os.path.islink(path):
            os.remove(path)
            removed.append(name)
    return removed


# ----------------------------------------------------------------- dataset locks
INSTANCE_LOCK_WAIT = 2.0             # seconds; covers a delayed release after a crash
SECOND_INSTANCE_MESSAGE = ("Another annotation tool is already using this dataset (a session "
                           "or a restore). Close it first. Nothing was changed.")


class InstanceLock:
    """Byte 0 of dataset/.annotation.lock, held by a session or a restore for its whole run.
    Not acquired until acquire() is called, so creating the object writes nothing."""

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.handle = None

    def acquire(self):
        if self.handle is not None:
            return
        try:
            self.handle = vl.acquire_lock_byte(self.dataset_dir, vl.INSTANCE_BYTE,
                                               INSTANCE_LOCK_WAIT)
        except vl.LockBusy:
            raise AnnotationError(SECOND_INSTANCE_MESSAGE)
        except vl.LockUnusable as error:
            raise AnnotationError(f"{vl.LOCK_NAME}: {error}. Nothing was changed.")

    def release(self):
        if self.handle is not None:
            self.handle.release()
            self.handle = None


@contextlib.contextmanager
def writing(dataset_dir, what="nothing was saved"):
    """Hold the data lock (byte 1) around a group of writes; busy -> AnnotationError."""
    stack = contextlib.ExitStack()
    try:
        stack.enter_context(vl.data_lock(dataset_dir))
    except vl.LockBusy:
        raise AnnotationError(f"the dataset is busy (validate_labels.py or --status is "
                              f"reading it); {what}. Try again.")
    except vl.LockUnusable as error:
        raise AnnotationError(f"{vl.LOCK_NAME}: {error}; {what}.")
    with stack:
        yield


def count_temp_leftovers(dataset_dir):
    """Temporary files left by a backup or lock-file creation whose temporary name could not
    be removed. They are never backups and never restored; they may be deleted by hand."""
    folder = backups_dir(dataset_dir)
    count = 0
    if folder.is_dir() and not os.path.islink(folder):
        count += sum(1 for p in folder.iterdir()
                     if p.name.startswith(".") and p.name.endswith(".tmp"))
    count += sum(1 for p in Path(dataset_dir).glob(vl.LOCK_NAME + ".*.tmp"))
    return count


def leftover_note(dataset_dir):
    n = count_temp_leftovers(dataset_dir)
    if not n:
        return None
    return (f"{n} leftover temporary file(s) from earlier backups in the dataset folder; they "
            "are not backups, are never restored, and may be deleted by hand")


# ---------------------------------------------------------------------- backups
# A backup SET records the exact state of the labels and status files of one pass.
#
# New format: the identity <stamp>[_<tag>]_s<n> is claimed by publishing the manifest
# backupset_pass<N>_<identity>.txt first (a hard link, which never replaces a file). The
# manifest lists each member as 'present <size> <sha256>' or 'absent'; present members are
# published as <stem>_pass<N>_<identity>.csv. A set is restorable only when its manifest
# parses strictly and every member matches it exactly; an interrupted or abandoned attempt
# never does. If any name is taken, the whole set moves to the next suffix.
#
# Legacy format (older backups, no manifest): <stem>_pass<N>_<stamp>[_<tag>][_<n>].csv. Only
# an exact pair (one labels and one status file with the same stamp, tag and suffix) is
# restorable; its contents cannot be verified (no hashes).
BACKUP_KINDS = ("labels", "status")
IDENTITY_PATTERN = r"(?P<stamp>\d{8}T\d{6}_\d{6})(?:_(?P<tag>[a-z]+))?_s(?P<n>0|[1-9]\d*)"
MANIFEST_NAME = re.compile(r"backupset_pass(?P<pass>[12])_" + IDENTITY_PATTERN + r"\.txt")
LEGACY_REST = re.compile(r"(?P<stamp>\d{8}T\d{6}_\d{6})(?:_(?P<tag>[a-z]+))?"
                         r"(?:_(?P<n>\d+))?\.csv")
MANIFEST_HEADER = "backup-set 1"
MANIFEST_LIMIT = 1024
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
SIZE_TEXT = re.compile(r"0|[1-9]\d*")
MAX_BACKUP_ATTEMPTS = 1000


def _stamp(now):
    return now.strftime("%Y%m%dT%H%M%S_%f")


def _read_current(path):
    """Exact current state of a label/status file: its bytes, or None if it does not exist."""
    path = Path(path)
    if os.path.islink(path):
        raise AnnotationError(f"{path.name} is a link; refusing to use it")
    if not os.path.lexists(path):
        return None
    if not path.is_file():
        raise AnnotationError(f"{path.name} is not a regular file")
    return path.read_bytes()


def _why(error):
    """The reason of an error without any path in it."""
    if isinstance(error, OSError):
        return error.strerror or type(error).__name__
    return str(error)


def _stage_tmp(folder, stem, data):
    tmp = folder / f".{stem}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    with open(tmp, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return tmp


def _remove_tmps(tmps):
    """Remove temporary names only. A temporary name hard-linked to a published file is just
    a second name: removing it never affects the published backup. A name that cannot be
    removed stays as a counted leftover (it starts with '.', so it is never a backup)."""
    for tmp in tmps:
        if os.path.lexists(tmp):
            vl._try_unlink(tmp)


def _member_name(stem, pass_no, identity):
    return f"{stem}_pass{pass_no}_{identity}.csv"


def _manifest_name(pass_no, identity):
    return f"backupset_pass{pass_no}_{identity}.txt"


def _manifest_bytes(pass_no, identity, states):
    lines = [MANIFEST_HEADER, f"pass={pass_no}", f"identity={identity}"]
    for kind, data in zip(BACKUP_KINDS, states):
        if data is None:
            lines.append(f"{kind}=absent")
        else:
            lines.append(f"{kind}=present {len(data)} {hashlib.sha256(data).hexdigest()}")
    return ("\n".join(lines) + "\n").encode("ascii")


def _parse_manifest(raw, pass_no, identity):
    """{kind: None (absent) or (size, sha256)} or None if the manifest is not exactly the
    format written by _manifest_bytes for this pass and identity."""
    if len(raw) > MANIFEST_LIMIT:
        return None
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return None
    lines = text.split("\n")
    if len(lines) != 6 or lines[5] != "":
        return None
    if lines[:3] != [MANIFEST_HEADER, f"pass={pass_no}", f"identity={identity}"]:
        return None
    members = {}
    for kind, line in zip(BACKUP_KINDS, lines[3:5]):
        prefix = kind + "="
        if not line.startswith(prefix):
            return None                       # wrong order, a duplicate or an unknown member
        value = line[len(prefix):]
        if value == "absent":
            members[kind] = None
            continue
        parts = value.split(" ")
        if (len(parts) != 3 or parts[0] != "present" or not SIZE_TEXT.fullmatch(parts[1])
                or not SHA256_HEX.fullmatch(parts[2])):
            return None
        members[kind] = (int(parts[1]), parts[2])
    if all(v is None for v in members.values()):
        return None                           # never written: nothing to back up
    return members


def _regular_file_bytes(path):
    """Bytes of a regular file that is not a link, else None."""
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
        return Path(path).read_bytes()
    except OSError:
        return None


def backup_files(dataset_dir, pass_no, now, tag=""):
    """Back up the current label and status files of this pass as one manifest set.

    The manifest is published first and claims the identity; then each existing file is
    published under that identity (an absent file is recorded only in the manifest). Never
    overwrites: if the manifest name or a member name is taken, that identity is abandoned
    and the whole set is published again with the next suffix. Already published files are
    never deleted; an abandoned or interrupted attempt is simply never restorable. Nothing is
    written when neither file exists. Returns the published member paths."""
    paths = pass_paths(dataset_dir, pass_no)
    states = [_read_current(path) for path in paths]
    if all(data is None for data in states):
        return []
    folder = backups_dir(dataset_dir)
    if os.path.islink(folder):
        raise AnnotationError("the backups folder is a link; refusing to use it")
    base = _stamp(now) + (f"_{tag}" if tag else "")
    tmps, started = [], False
    try:
        folder.mkdir(parents=True, exist_ok=True)
        member_tmps = {}
        for kind, path, data in zip(BACKUP_KINDS, paths, states):
            if data is not None:
                member_tmps[kind] = _stage_tmp(folder, path.stem, data)
                tmps.append(member_tmps[kind])
        for n in range(MAX_BACKUP_ATTEMPTS):
            identity = f"{base}_s{n}"
            manifest_tmp = _stage_tmp(folder, "backupset",
                                      _manifest_bytes(pass_no, identity, states))
            tmps.append(manifest_tmp)
            try:
                os.link(manifest_tmp, folder / _manifest_name(pass_no, identity))
            except FileExistsError:
                continue                                  # identity claimed by another set
            started = True
            published, abandoned = [], False
            for kind, path in zip(BACKUP_KINDS, paths):
                if kind not in member_tmps:
                    continue
                target = folder / _member_name(path.stem, pass_no, identity)
                try:
                    os.link(member_tmps[kind], target)
                except FileExistsError:
                    abandoned = True                      # occupied: retry the whole set
                    break
                published.append(target)
            if not abandoned:
                return published
        raise AnnotationError(f"no free backup name after {MAX_BACKUP_ATTEMPTS} attempts")
    except OSError as error:
        if started:
            raise AnnotationError(f"Cannot write a backup ({_why(error)}); the interrupted "
                                  "backup attempt is incomplete and will never be restored")
        raise AnnotationError(f"Cannot write a backup ({_why(error)})")
    finally:
        _remove_tmps(tmps)


def _restorable(info):
    return info["data"] is not None


def _backup_sets(dataset_dir, pass_no, include_prerestore=False):
    """All backup sets of this pass: identity key -> info.

    key = (stamp, n, tag, format). info = {'format': 'manifest' or 'legacy', 'identity',
    'tag', 'members': {kind: path or None}, 'data': {kind: bytes or None} when restorable
    else None}. For a manifest set, 'data' holds the verified bytes, so a restore writes
    exactly what was checked."""
    folder = backups_dir(dataset_dir)
    sets = {}
    if not folder.is_dir() or os.path.islink(folder):
        return sets
    names = {p.name for p in folder.iterdir()}
    label_path, status_path = pass_paths(dataset_dir, pass_no)
    stems = {"labels": label_path.stem, "status": status_path.stem}
    for name in names:
        m = MANIFEST_NAME.fullmatch(name)
        if not m or int(m["pass"]) != pass_no:
            continue
        tag = m["tag"] or ""
        if tag == "prerestore" and not include_prerestore:
            continue
        identity = name[len(f"backupset_pass{pass_no}_"):-len(".txt")]
        key = (m["stamp"], int(m["n"]), tag, "manifest")
        members = {kind: folder / _member_name(stems[kind], pass_no, identity)
                   for kind in BACKUP_KINDS}
        info = {"format": "manifest", "identity": identity, "tag": tag, "members": {},
                "data": None, "manifest": folder / name}
        raw = _regular_file_bytes(folder / name)
        expected = _parse_manifest(raw, pass_no, identity) if raw is not None else None
        if expected is not None:
            data, ok = {}, True
            for kind in BACKUP_KINDS:
                member = members[kind]
                if expected[kind] is None:
                    info["members"][kind] = None
                    data[kind] = None
                    if os.path.lexists(member):
                        ok = False                        # ambiguous: a file where none belongs
                    continue
                info["members"][kind] = member
                content = _regular_file_bytes(member)
                size, digest = expected[kind]
                if (content is None or len(content) != size
                        or hashlib.sha256(content).hexdigest() != digest):
                    ok = False
                    continue
                data[kind] = content
            if ok:
                info["data"] = data
        sets[key] = info
    legacy = {}
    for name in names:
        for kind in BACKUP_KINDS:
            prefix = f"{stems[kind]}_pass{pass_no}_"
            if not name.startswith(prefix):
                continue
            m = LEGACY_REST.fullmatch(name[len(prefix):])
            if not m:
                continue
            tag = m["tag"] or ""
            if tag == "prerestore" and not include_prerestore:
                continue
            key = (m["stamp"], int(m["n"] or 0), tag, "legacy")
            legacy.setdefault(key, {}).setdefault(kind, []).append(folder / name)
    for key, found in legacy.items():
        info = {"format": "legacy", "identity": None, "tag": key[2], "members": {},
                "data": None}
        if all(len(found.get(kind, [])) == 1 for kind in BACKUP_KINDS):
            data = {kind: _regular_file_bytes(found[kind][0]) for kind in BACKUP_KINDS}
            info["members"] = {kind: found[kind][0] for kind in BACKUP_KINDS}
            if all(v is not None for v in data.values()):
                info["data"] = data
        else:
            info["members"] = {kind: paths[0] for kind, paths in found.items()}
        sets[key] = info
    return sets


def _newest_restorable(dataset_dir, pass_no):
    sets = _backup_sets(dataset_dir, pass_no)
    usable = [key for key, info in sets.items() if _restorable(info)]
    if not usable:
        return None, bool(sets)
    # newest stamp, then the highest shared suffix; a verified manifest set wins a tie
    key = max(usable, key=lambda k: (k[0], k[1], k[3] == "manifest"))
    return sets[key], True


def list_backups(dataset_dir, pass_no):
    """Member files of the restorable backup sets of this pass (no pre-restore copies, no
    manifests, never incomplete attempts or temporary leftovers)."""
    return sorted(p for info in _backup_sets(dataset_dir, pass_no).values()
                  if _restorable(info) for p in info["members"].values() if p is not None)


def _apply_state(path, data):
    """Make a label/status file exactly `data` (bytes) or absent (None)."""
    if data is None:
        if os.path.lexists(path):
            if os.path.islink(path) or not Path(path).is_file():
                raise AnnotationError(f"{Path(path).name} is not a regular file")
            os.remove(path)
    else:
        write_atomic(path, data)


def _write_marker(dataset_dir, pass_no, phase):
    write_atomic(vl.restore_marker_path(dataset_dir), vl.format_restore_marker(pass_no, phase))


def restore_precheck(dataset_dir, pass_no):
    """Refuse a restore that the marker does not allow. A marker that cannot be read blocks
    automatic restore for both passes (it is never changed here); a readable marker for the
    other pass is refused too. Reads nothing else and writes nothing."""
    present, marker_pass, _ = vl.read_restore_marker(dataset_dir)
    if not present:
        return
    if marker_pass is None:
        message = vl.restore_marker_message(None, None)
        raise AnnotationError(message[0].upper() + message[1:] + " Nothing was changed.")
    if marker_pass != pass_no:
        raise AnnotationError(f"the interrupted restore was for pass {marker_pass}: run "
                              f"--pass {marker_pass} --restore-latest-backup first. Nothing "
                              "was changed.")


def restore_latest_backup(dataset_dir, pass_no, now):
    """Put the newest restorable backup set back exactly (bytes, empty or absent).

    While it runs, dataset/.restore_in_progress names the pass and the phase (backup ->
    labels -> status); it is removed only when the restore is complete. If a step fails, the
    files are rolled back exactly to their state before this run; a marker that existed
    before (an earlier interrupted restore of this pass) is kept. A crash leaves the marker,
    which blocks annotation until the restore is run again. The current files are kept as a
    prerestore backup set."""
    restore_precheck(dataset_dir, pass_no)
    chosen, any_sets = _newest_restorable(dataset_dir, pass_no)
    if chosen is None:
        raise AnnotationError("there is no backup to restore" +
                              (" (only incomplete or unverifiable backup sets were found)"
                               if any_sets else ""))
    label_path, status_path = pass_paths(dataset_dir, pass_no)
    targets = {"labels": label_path, "status": status_path}
    wanted = chosen["data"]
    marker = vl.restore_marker_path(dataset_dir)
    with writing(dataset_dir, "nothing was restored"):
        before = {kind: _read_current(path) for kind, path in targets.items()}
        marker_existed = os.path.lexists(marker)
        _write_marker(dataset_dir, pass_no, "backup")            # fails -> nothing changed
        try:
            backup_files(dataset_dir, pass_no, now, tag="prerestore")
            for kind in BACKUP_KINDS:
                _write_marker(dataset_dir, pass_no, kind)
                _apply_state(targets[kind], wanted[kind])
        except (AnnotationError, OSError) as error:
            try:
                for kind, path in targets.items():
                    if _read_current(path) != before[kind]:
                        _apply_state(path, before[kind])
            except (AnnotationError, OSError) as rollback_error:
                raise AnnotationError(f"restore failed ({_why(error)}) and the files could not "
                                      f"be put back ({_why(rollback_error)}); the restore "
                                      "marker .restore_in_progress stays. Run the restore "
                                      "again.")
            if not marker_existed:
                try:
                    os.remove(marker)
                except OSError as remove_error:
                    raise AnnotationError(
                        f"restore failed ({_why(error)}). The label files were put back exactly "
                        "as they were before the restore, but the restore marker "
                        f".restore_in_progress could not be removed ({_why(remove_error)}). "
                        "Annotation stays blocked until the marker is resolved; running "
                        f"--pass {pass_no} --restore-latest-backup again finishes the "
                        "recovery.")
            raise AnnotationError(f"restore failed ({_why(error)}); the files were put back "
                                  "exactly as they were before the restore")
        try:
            os.remove(marker)
        except OSError as error:
            raise AnnotationError(f"the files were restored, but the restore marker "
                                  f".restore_in_progress could not be removed ({_why(error)}); "
                                  "annotation stays blocked until it is removed")
    return [label_path.name, status_path.name]


def saved_data_problems(status_rows, faces, eyes):
    """Status rows that disagree with their label rows (an interrupted edit, a hand edit).

    Filenames are not quoted: the tool is blind, so a problem is located by its line in the
    status file instead (line 1 is the header)."""
    problems = []
    for index, st in enumerate(status_rows):
        name, line = st["filename"], index + 2
        n_face, n_eye = len(faces.get(name, [])), len(eyes.get(name, []))
        if st["status"] == "no_face":
            if n_face or n_eye:
                problems.append(f"line {line}: status no_face but {n_face} face and {n_eye} "
                                "eye label rows")
            if st["n_face"] or st["n_eye"]:
                problems.append(f"line {line}: status no_face but it says {st['n_face']} face "
                                f"and {st['n_eye']} eye rows (must be 0 and 0)")
            continue
        sides = [e["side"] for e in eyes.get(name, [])]
        if n_face != 1:
            problems.append(f"line {line}: status labelled but {n_face} face rows (need 1)")
        elif n_eye > 2 or len(set(sides)) != n_eye:
            problems.append(f"line {line}: {n_eye} eye rows with sides {sides}")
        elif (st["n_face"], st["n_eye"]) != (n_face, n_eye):
            problems.append(f"line {line}: status says {st['n_face']} face and {st['n_eye']} "
                            f"eye rows, the label file has {n_face} and {n_eye}")
    return problems


def load_pass(dataset_dir, pass_no):
    """Read and structurally check the files of one pass.

    Returns dict(status={filename: row}, faces={filename: [rows]}, eyes={...}, labels=[rows],
    orphans=[filenames with label rows but no status row = interrupted save]).
    """
    label_path, status_path = pass_paths(dataset_dir, pass_no)
    try:
        labels = [vl.parse_label_row(r, label_path.name)
                  for r in vl.read_csv_strict(label_path, vl.LABEL_FIELDS)]
        status_rows = [vl.parse_status_row(r, pass_no, status_path.name)
                       for r in vl.read_csv_strict(status_path, vl.STATUS_FIELDS)]
    except vl.ValidationError as error:
        names = [p.name for p in list_backups(dataset_dir, pass_no)[-4:]]
        raise AnnotationError(f"{error}. Nothing was changed. Latest backups: "
                              f"{names or 'none'}. Use --restore-latest-backup to restore.")
    status = {}
    for r in status_rows:
        if r["filename"] in status:
            raise AnnotationError(f"{status_path.name}: duplicate status for one image")
        status[r["filename"]] = r
    faces, eyes = {}, {}
    for r in labels:
        (faces if r["object_type"] == "face" else eyes).setdefault(r["filename"], []).append(r)
    problems = saved_data_problems(status_rows, faces, eyes)
    if problems:
        names = [p.name for p in list_backups(dataset_dir, pass_no)[-4:]]
        more = f"; and {len(problems) - 3} more" if len(problems) > 3 else ""
        raise AnnotationError(f"{status_path.name} does not match {label_path.name}: "
                              + "; ".join(problems[:3]) + more
                              + ". Nothing was changed. Run scripts/validate_labels.py for "
                              f"details. Latest backups: {names or 'none'}; "
                              "--restore-latest-backup restores them.")
    orphans = sorted((set(faces) | set(eyes)) - set(status))
    return {"status": status, "faces": faces, "eyes": eyes, "labels": labels,
            "orphans": orphans}


def _label_text_rows(typed_rows):
    out = []
    for r in typed_rows:
        out.append({"filename": r["filename"], "object_type": r["object_type"],
                    "x": str(r["x"]), "y": str(r["y"]),
                    "w": str(r["w"]) if r["object_type"] == "face" else "",
                    "h": str(r["h"]) if r["object_type"] == "face" else "",
                    "side": r.get("side", "") if r["object_type"] == "eye" else "",
                    "uncertain": str(r["uncertain"])})
    return out


def _sort_key(row):
    order = {"face": 0, "eye": 1}
    return (row["filename"], order[row["object_type"]], row["side"])


def save_image(dataset_dir, pass_no, filename, annotation, background, now):
    """Validate and save one image. Returns 'saved' or 'unchanged'.

    Order of writes: labels first, then the status row. The status row is the commit marker,
    so an interruption in between leaves label rows without status; those are detected on
    the next start and redone. An existing annotation is backed up before being replaced.
    """
    rows, status = annotation.records(filename)
    typed_faces = [{"x": int(r["x"]), "y": int(r["y"]), "w": int(r["w"]), "h": int(r["h"]),
                    "uncertain": int(r["uncertain"])} for r in rows if r["object_type"] == "face"]
    typed_eyes = [{"x": int(r["x"]), "y": int(r["y"]), "side": r["side"],
                   "uncertain": int(r["uncertain"])} for r in rows if r["object_type"] == "eye"]
    errors = vl.validate_image(filename, typed_faces, typed_eyes,
                               dict(status, annotated_at=now), background)
    if errors:
        raise AnnotationError("; ".join(errors))
    current = load_pass(dataset_dir, pass_no)
    existing = current["status"].get(filename)
    if existing is not None:
        old_rows = _label_text_rows(current["faces"].get(filename, []) +
                                    current["eyes"].get(filename, []))
        key = lambda r: _sort_key(r)    # noqa: E731
        same = (sorted(old_rows, key=key) == sorted(rows, key=key)
                and existing["status"] == status["status"])
        if same:
            return "unchanged"
    label_path, status_path = pass_paths(dataset_dir, pass_no)
    kept_labels = [r for r in _label_text_rows(current["labels"]) if r["filename"] != filename]
    all_labels = sorted(kept_labels + rows, key=_sort_key)
    status_rows = []
    for name, r in current["status"].items():
        if name != filename:
            status_rows.append({"filename": name, "status": r["status"], "n_face": r["n_face"],
                                "n_eye": r["n_eye"], "annotated_at": r["annotated_at_text"],
                                "pass": pass_no, "tool_version": r["tool_version"]})
    status_rows.append({"filename": filename, "status": status["status"],
                        "n_face": status["n_face"], "n_eye": status["n_eye"],
                        "annotated_at": now.isoformat(timespec="seconds"), "pass": pass_no,
                        "tool_version": TOOL_VERSION})
    status_rows.sort(key=lambda r: r["filename"])
    with writing(dataset_dir):          # readers never see the pair between the two writes
        if existing is not None or filename in current["orphans"]:
            backup_files(dataset_dir, pass_no, now, tag="edit")
        write_atomic(label_path, _csv_bytes(vl.LABEL_FIELDS, all_labels))
        write_atomic(status_path, _csv_bytes(vl.STATUS_FIELDS, status_rows))
    return "saved"


# =================================================================== start-up checks
def check_gitignore(project_root=vl.PROJECT_ROOT):
    """Ask Git whether the annotation paths are ignored; stop if they are not."""
    for probe in GITIGNORE_PROBES:
        try:
            result = subprocess.run(["git", "-C", str(project_root), "check-ignore", "-q",
                                     "--", probe], capture_output=True, text=True, timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            raise AnnotationError(f"Cannot verify Git ignore rules ({error}). Nothing started.")
        if result.returncode != 0:
            raise AnnotationError(f"Git does not report '{probe}' as ignored. Fix .gitignore "
                                  "first so annotation files cannot be committed by accident.")


def check_layout(dataset_dir):
    dataset_dir = Path(dataset_dir)
    for path in (dataset_dir, dataset_dir / "images", dataset_dir / "metadata.csv",
                 backups_dir(dataset_dir)):
        if os.path.islink(path):
            raise AnnotationError(f"{path} is a link; refusing to use it")
    if not (dataset_dir / "images").is_dir():
        raise AnnotationError(f"{dataset_dir / 'images'} not found")


BUSY_DETAILS_HIDDEN = ("details hidden: the dataset is busy, so the image names could not be "
                       "read; run scripts/validate_labels.py for details")


def blind_text(text, dataset_dir, names=None):
    """Remove image filenames and the dataset folder path from a message shown to the
    annotator, keeping the rest (the kind of problem and its counts).

    `names` are the metadata filenames of the snapshot the message came from. Without them
    the names are read through vl.consistent_read (never outside the data lock); if the
    dataset is busy the whole text is replaced, so no name can slip through."""
    if names is None:
        try:
            names = vl.consistent_read(dataset_dir, lambda: list(vl.load_metadata(dataset_dir)))
        except vl.ValidationError:
            names = []
        except (vl.LockBusy, vl.LockUnusable):
            return BUSY_DETAILS_HIDDEN
    names = list(names)
    for name in sorted(names, key=len, reverse=True):          # whole names before their stems
        text = text.replace(name, "an image")
    for name in sorted(names, key=len, reverse=True):
        text = text.replace(Path(name).stem, "an image")
    folder = Path(dataset_dir)
    for form in sorted({str(folder), str(folder.resolve()), folder.as_posix(),
                        folder.resolve().as_posix()}, key=len, reverse=True):
        text = text.replace(form, "the dataset folder")
    # also any other image-looking name (for example a status row for a file that is not in
    # the dataset)
    return re.sub(r"[\w.\-]+\.(?:png|jpe?g|bmp|tiff?)\b", "an image", text, flags=re.IGNORECASE)


def check_pass2_allowed(dataset_dir, now, min_gap_hours=24.0):
    """Pass 2 only after pass 1 is complete, valid, and finished at least 24 h ago.

    The message does not name any image: the tool is blind. scripts/validate_labels.py, run
    outside the annotation window, lists the affected images."""
    report, info = vl.validate_dataset_snapshot(dataset_dir, 1, require_complete=True)
    if report["errors"]:
        raise AnnotationError("pass 2 needs a complete, valid pass 1 first (run "
                              "scripts/validate_labels.py to see which images). First problem: "
                              + blind_text(report["errors"][0], dataset_dir, info["names"]))
    try:
        data = vl.consistent_read(dataset_dir, lambda: load_pass(dataset_dir, 1))
    except (vl.LockBusy, vl.LockUnusable) as error:
        raise AnnotationError(f"cannot read pass 1: {error}")
    finished = max(r["annotated_at"] for r in data["status"].values())
    ready = finished + timedelta(hours=min_gap_hours)
    if now < ready:
        raise AnnotationError(f"pass 2 may start at {ready.isoformat(timespec='minutes')} "
                              f"(pass 1 finished {finished.isoformat(timespec='minutes')}; "
                              f"{min_gap_hours:g} h gap required)")


def build_order(dataset_dir, pass_no, now=None, min_gap_hours=24.0):
    """Blind order of the images of a pass (filenames only; never shown to the annotator)."""
    meta = vl.load_metadata(dataset_dir)
    if pass_no == 1:
        names = sorted(meta)
    else:
        check_pass2_allowed(dataset_dir, now or datetime.now().astimezone(), min_gap_hours)
        names = vl.repeat_sample([n for n, m in meta.items() if not vl.is_background(m)])
    return vl.annotation_order(names)


def check_restore_marker(dataset_dir):
    """Refuse to annotate while .restore_in_progress exists (whatever it contains)."""
    present, pass_no, phase = vl.read_restore_marker(dataset_dir)
    if present:
        message = vl.restore_marker_message(pass_no, phase)
        raise AnnotationError(message[0].upper() + message[1:] + " Nothing was changed.")


def start_session_checks(dataset_dir, pass_no, now, check_git=True, min_gap_hours=24.0,
                         instance_lock=None):
    """Everything that must hold before a window opens. Returns (order, notes).

    The read-only checks run first, so a refused start writes nothing (not even the lock
    file). Then `instance_lock` (if given) is acquired and the restore marker is checked
    again, because a restore may have started or been interrupted meanwhile."""
    notes = []
    check_layout(dataset_dir)
    if check_git:
        check_gitignore()
    check_restore_marker(dataset_dir)
    order = build_order(dataset_dir, pass_no, now, min_gap_hours)
    if instance_lock is not None:
        instance_lock.acquire()
        check_restore_marker(dataset_dir)
    data = load_pass(dataset_dir, pass_no)
    if data["orphans"]:
        notes.append(f"{len(data['orphans'])} image(s) have label rows from an interrupted save "
                     "and will be annotated again")
    with writing(dataset_dir, "nothing was started"):
        removed = clean_stale_tmp(dataset_dir)
        backup_files(dataset_dir, pass_no, now, tag="session")
    if removed:
        notes.insert(0, f"removed leftover temporary files from an interrupted save: {removed}")
    leftovers = leftover_note(dataset_dir)
    if leftovers:
        notes.append(leftovers)
    return order, notes


# ================================================================== the session
class Session:
    """The interactive window logic. `on_mouse` and `handle_key` can be driven by tests."""

    def __init__(self, dataset_dir, pass_no, order, scale=DEFAULT_SCALE, clock=None):
        self.dataset_dir = Path(dataset_dir)
        self.pass_no = pass_no
        self.order = list(order)
        self.scale = scale
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.meta = vl.load_metadata(self.dataset_dir)
        self.data = load_pass(self.dataset_dir, pass_no)
        self.images = {}
        self.message = ""
        self.finished = False
        self.exit_reason = ""
        self.cursor = None
        self.drag_start = None
        self.drag_now = None
        self.armed = None                         # "quit" / "back" waiting for a 2nd press
        done = [i for i, n in enumerate(self.order) if n in self.data["status"]]
        first_open = next((i for i, n in enumerate(self.order) if n not in self.data["status"]),
                          None)
        self.index = first_open if first_open is not None else 0
        if first_open is None and done:
            self.message = "All images in this pass are already annotated (reviewing)."
        self._load_current()

    # -- image and annotation handling
    def filename(self):
        return self.order[self.index]

    def image(self, name=None):
        name = name or self.filename()
        if name not in self.images:
            path = self.dataset_dir / "images" / name
            img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (vl.IMAGE_H, vl.IMAGE_W):
                raise AnnotationError("an image is unreadable or not 640x480; check the image "
                                      "file and confirm it is a readable 640x480 image")
            self.images[name] = img
        return self.images[name]

    def _load_current(self):
        name = self.filename()
        st = self.data["status"].get(name)
        if st is None:
            self.annotation = Annotation()
        else:
            self.annotation = Annotation.from_saved(self.data["faces"].get(name, []),
                                                    self.data["eyes"].get(name, []), st)
        self.baseline = self.annotation.fingerprint()
        self.drag_start = self.drag_now = None
        self.armed = None

    def dirty(self):
        return self.annotation.fingerprint() != self.baseline

    def done_count(self):
        return sum(1 for n in self.order if n in self.data["status"])

    def to_orig(self, x, y):
        return (max(0, min(vl.IMAGE_W - 1, int(round(x / self.scale)))),
                max(0, min(vl.IMAGE_H - 1, int(round(y / self.scale)))))

    # -- mouse
    def on_mouse(self, event, x, y, flags=0, param=None):
        if y >= int(vl.IMAGE_H * self.scale):               # clicks on the text strip
            return
        ox, oy = self.to_orig(x, y)
        self.cursor = (ox, oy)
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.armed = None
                if self.annotation.target() == "face":
                    self.drag_start = (x, y)
                    self.drag_now = (x, y)
                elif self.annotation.target() in vl.SIDES:
                    self.annotation.place_eye(ox, oy)
                    self.message = ""
                elif self.annotation.no_face:
                    raise AnnotationError("marked no_face: press z to undo first")
                else:
                    raise AnnotationError("all points are placed: Enter saves, z undoes")
            elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
                self.drag_now = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
                sx, sy = self.drag_start
                self.drag_start = self.drag_now = None
                if abs(x - sx) < DRAG_THRESHOLD_PX and abs(y - sy) < DRAG_THRESHOLD_PX:
                    raise AnnotationError("drag with the button held to draw the face box")
                x0, y0 = self.to_orig(sx, sy)
                self.annotation.set_face(x0, y0, ox, oy)
                self.message = ""
        except AnnotationError as error:
            self.message = str(error)

    # -- keys
    def handle_key(self, key):
        """Returns 'quit' when the session should end, else None."""
        k = chr(key).lower() if 0 <= key < 128 else ""
        try:
            if key in KEYS_ENTER:
                self._enter()
            elif k == "h":
                self.annotation.hide_eye()
                self.message = ""
            elif k == "u":
                self.annotation.toggle_uncertain()
                self.message = "uncertain: " + ("ON" if self.annotation.uncertain.get(
                    self.annotation.last_object, 0) else "off")
            elif k == "n":
                self.annotation.mark_no_face()
                self.message = "marked no_face" if self.annotation.no_face else ""
            elif k in ("f", "l", "r"):
                obj = {"f": "face", "l": "left", "r": "right"}[k]
                self.annotation.reopen(obj)
                self.message = (f"{obj} {'box' if obj == 'face' else 'eye'} cleared: place it "
                                "again (z restores the previous one)")
            elif k == "z":
                self.annotation.undo()
                self.message = "undone"
            elif k == "b":
                self._back()
                return None
            elif k == "q" or key == KEY_ESC:
                return self._quit()
            else:
                return None
            self.armed = None                      # any real action cancels a pending q / b
        except AnnotationError as error:
            self.message = str(error)
        return None

    def _confirm_discard(self, what):
        if self.dirty() and self.armed != what:
            self.armed = what
            self.message = f"unsaved changes on this image: press {what[0]} again to " \
                           f"{what} and discard them"
            return False
        return True

    def _quit(self):
        if not self._confirm_discard("quit"):
            return None
        return "quit"

    def _back(self):
        if self.index == 0:
            self.message = "this is the first image"
            return None
        if not self._confirm_discard("back"):
            return None
        self.index -= 1
        self._load_current()
        self.message = ""
        return None

    def _enter(self):
        if not self.annotation.is_complete():
            raise AnnotationError("not finished: " + self.annotation.missing())
        name = self.filename()
        background = vl.is_background(self.meta[name])
        now = self.clock()
        had_saved_version = name in self.data["status"]
        try:
            result = save_image(self.dataset_dir, self.pass_no, name, self.annotation,
                                background, now)
        except AnnotationError as error:
            text = str(error)
            if "background-only" in text:
                text = ("this image should contain nobody (no face allowed): press z to undo, "
                        "then n. If a person really is visible, stop and report it.")
            # validation errors start with the image filename: keep the window blind
            raise AnnotationError("not saved: " + blind_text(text, self.dataset_dir,
                                                             list(self.meta)))
        self.data = load_pass(self.dataset_dir, self.pass_no)
        self.message = ("no change" if result != "saved" else
                        "saved (previous version kept in backups)" if had_saved_version
                        else "saved")
        if self.index + 1 >= len(self.order):
            self.finished = True
            self.message = "all images of this pass are annotated - press q to close"
            self._load_current()
            return
        self.index += 1
        self._load_current()

    # -- drawing
    def render(self):
        h, w = int(vl.IMAGE_H * self.scale), int(vl.IMAGE_W * self.scale)
        canvas = np.full((h + STRIP_HEIGHT, w, 3), 40, np.uint8)
        base = cv2.resize(self.image(), (w, h), interpolation=cv2.INTER_LINEAR)
        s = self.scale
        a = self.annotation
        if a.face is not None:
            x, y, fw, fh = a.face
            cv2.rectangle(base, (int(x * s), int(y * s)), (int((x + fw) * s), int((y + fh) * s)),
                          (0, 255, 0), 2)
            if a.uncertain["face"]:
                cv2.putText(base, "?", (int(x * s) + 4, int(y * s) + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        colors = {"left": (255, 160, 0), "right": (255, 0, 200)}
        for side, pt in a.eyes.items():
            if pt is not None:
                cx, cy = int(pt[0] * s), int(pt[1] * s)
                cv2.drawMarker(base, (cx, cy), colors[side], cv2.MARKER_CROSS, 18, 2)
                cv2.putText(base, side[0].upper() + ("?" if a.uncertain[side] else ""),
                            (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[side], 2)
        if self.drag_start is not None and self.drag_now is not None:
            cv2.rectangle(base, self.drag_start, self.drag_now, (0, 200, 255), 1)
        if self.cursor is not None:
            self._draw_magnifier(base)
        canvas[:h] = base
        total = len(self.order)
        head = f"Image {self.index + 1}/{total}   annotated {self.done_count()}/{total}"
        if self.pass_no == 2:
            head += "   (REPEAT PASS)"
        nxt = "complete: Enter saves" if a.is_complete() else "next: " + a.missing()
        if a.no_face:
            nxt = "no_face: Enter saves (z undoes)"
        lines = [head, nxt, self.message,
                 "drag=face click=eye h=hidden u=uncertain n=no face f/l/r=redo "
                 "z=undo Enter=save b=back q=quit"]
        for i, text in enumerate(lines):
            cv2.putText(canvas, text, (8, h + 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        HELP_SCALE if i == 3 else 0.5,
                        (230, 230, 230) if i != 2 else (0, 220, 255), 1)
        return canvas

    def _draw_magnifier(self, img):
        cx, cy = self.cursor
        half = MAG_REGION // 2
        padded = cv2.copyMakeBorder(self.image(), half, half, half, half, cv2.BORDER_REPLICATE)
        crop = padded[cy:cy + MAG_REGION, cx:cx + MAG_REGION]       # centred on (cx, cy)
        mag = cv2.resize(crop, (MAG_SIZE, MAG_SIZE), interpolation=cv2.INTER_CUBIC)
        c = MAG_SIZE // 2
        cv2.drawMarker(mag, (c, c), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
        cv2.rectangle(mag, (0, 0), (MAG_SIZE - 1, MAG_SIZE - 1), (255, 255, 255), 1)
        x0 = img.shape[1] - MAG_SIZE - 4
        img[4:4 + MAG_SIZE, x0:x0 + MAG_SIZE] = mag                  # only the display copy

    # -- main loop
    def run(self):
        """Show the window until the user quits. `self.exit_reason` says why it ended.

        OpenCV can report "not visible" for a moment while a window is still being created.
        So a window that is not visible only ends the session AFTER it has been confirmed
        visible at least once. q / ESC always work, even if visibility is never reported.
        """
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        self.exit_reason = ""
        seen_visible = False
        frames = 0
        try:
            while True:
                cv2.imshow(WINDOW_NAME, self.render())
                key = cv2.waitKey(30) & 0xFF
                frames += 1
                visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                if visible >= 1:
                    seen_visible = True
                elif seen_visible:
                    self.exit_reason = (f"window was closed (visibility property {visible} "
                                        "after it had been visible)")
                    break
                elif frames >= STARTUP_GRACE_FRAMES:
                    self.exit_reason = (f"window never became visible within the startup grace "
                                        f"period ({STARTUP_GRACE_FRAMES} frames, visibility "
                                        f"property {visible}); no annotation was lost, every "
                                        "saved image is already on disk")
                    break
                if key != 255 and self.handle_key(key) == "quit":
                    self.exit_reason = f"quit key pressed (key code {key})"
                    break
        except BaseException as error:
            self.exit_reason = f"interrupted by {type(error).__name__}"
            raise
        finally:
            cv2.destroyAllWindows()
        return self.done_count()


# ================================================================== practice mode
def make_practice_dataset(root):
    """Three synthetic cartoon faces and one empty image (no real data) for learning."""
    dataset = Path(root) / "dataset"
    (dataset / "images").mkdir(parents=True)
    rng = np.random.RandomState(0)
    rows = []
    for i in range(1, 5):
        img = np.full((480, 640, 3), 190, np.uint8)
        img = np.clip(img + rng.randint(-6, 6, img.shape), 0, 255).astype(np.uint8)
        if i < 4:
            cx, cy = 320 + 30 * (i - 2), 250
            cv2.ellipse(img, (cx, cy), (95, 125), 0, 0, 360, (150, 175, 215), -1)
            for dx in (-40, 40):
                cv2.ellipse(img, (cx + dx, cy - 25), (18, 9 if i != 3 else 3), 0, 0, 360,
                            (255, 255, 255), -1)
                cv2.circle(img, (cx + dx, cy - 25), 5 if i != 3 else 2, (40, 40, 40), -1)
            cv2.ellipse(img, (cx, cy + 55), (30, 10), 0, 0, 360, (80, 80, 190), -1)
        name = f"practice_{i}.png"
        ok, buf = cv2.imencode(".png", img)
        (dataset / "images" / name).write_bytes(buf.tobytes())
        rows.append({"filename": name, "participant_id": "NONE" if i == 4 else "P01",
                     "condition": "bg" if i == 4 else "normal",
                     "sub_condition": "background" if i == 4 else "frontal",
                     "capture_time": "2026-01-01T10:00:00+00:00", "notes": ""})
    with open(dataset / "metadata.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dataset


# ====================================================================== CLI
def print_status(dataset_dir, pass_no):
    """Progress and validation problems. Like every message of this blind tool it does not name
    images or folders: scripts/validate_labels.py lists the affected images.

    Everything comes from one consistent snapshot (read-only; the lock file is never
    created). Busy or an unusable lock file: only that error, no statistics (exit 1). An
    interrupted restore: statistics and the marker error are shown, exit 1. Returns the
    exit code."""
    report, info = vl.validate_dataset_snapshot(dataset_dir, pass_no, require_complete=False)
    names = info["names"]
    st = report["stats"]
    if not st:
        raise AnnotationError(blind_text("; ".join(report["errors"]), dataset_dir, names))
    print(f"Pass {pass_no}: {st['annotated_images']}/{st['expected_images']} images annotated "
          f"({st['labelled']} with a face, {st['no_face']} no_face, {st['uncertain_objects']} "
          "uncertain objects)")
    for line in report["errors"]:
        print("ERROR  ", blind_text(line, dataset_dir, names))
    return 1 if info["marker"][0] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Blind manual annotation (see docs).")
    parser.add_argument("--pass", dest="pass_no", type=int, choices=(1, 2), default=1)
    parser.add_argument("--status", action="store_true", help="show progress, no window")
    parser.add_argument("--practice", action="store_true",
                        help="learn the controls on synthetic images (nothing real is touched)")
    parser.add_argument("--restore-latest-backup", action="store_true")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--dataset", default=str(vl.DATASET_DIR))
    args = parser.parse_args(argv)
    now = datetime.now().astimezone()
    try:
        if args.practice:
            with tempfile.TemporaryDirectory() as tmp:
                dataset = make_practice_dataset(tmp)
                order, _ = start_session_checks(dataset, 1, now, check_git=False)
                print("PRACTICE MODE: synthetic images; everything is deleted afterwards.")
                Session(dataset, 1, order, args.scale).run()
            return 0
        dataset = Path(args.dataset)
        if args.status:
            return print_status(dataset, args.pass_no)
        lock = InstanceLock(dataset)
        try:
            if args.restore_latest_backup:
                lock.acquire()                 # before the question: a second one is refused
                restore_precheck(dataset, args.pass_no)      # refuse before asking
                answer = input("Restore the newest backup over the current files? Type 'yes': ")
                if answer.strip().lower() != "yes":
                    raise AnnotationError("not restored")
                print("restored:", restore_latest_backup(dataset, args.pass_no, now))
                note = leftover_note(dataset)
                if note:
                    print("Note:", note)
                return 0
            order, notes = start_session_checks(dataset, args.pass_no, now,
                                                instance_lock=lock)
            for note in notes:
                print("Note:", note)
            session = Session(dataset, args.pass_no, order, args.scale)
            done = session.run()
            print(f"Exit reason: {session.exit_reason}")
            print(f"Session ended. {done}/{len(order)} images annotated in pass "
                  f"{args.pass_no}. Saved progress is kept; run again to resume.")
            return 0
        finally:
            lock.release()
    except AnnotationError as error:
        # Blind like every other message. Practice mode never looks at the real dataset.
        names = [] if args.practice else None
        print(f"Error: {blind_text(str(error), args.dataset, names)}")
        return 1
    except vl.ValidationError as error:
        # An unusable metadata.csv (raised by vl.load_metadata in build_order and in Session).
        # Its text is already blind (line numbers and columns); blind_text() makes sure that no
        # filename, stem or dataset path can slip through.
        print(f"Error: {blind_text(str(error), args.dataset)}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Everything saved so far is kept; run again to resume.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
