"""Controlled webcam capture for the evaluation dataset (Milestone 5, Task 2).

Follows docs/EVALUATION_PROTOCOL.md. It only SAVES images: it does not run the
detector and does not draw anything on the saved image.

Typical use (run from the project folder, venv active):

    python scripts/capture_dataset.py --status
    python scripts/capture_dataset.py                       (asks questions)
    python scripts/capture_dataset.py --condition turn --participant P01 --sub 30deg

Keys in the video window:
    SPACE      save one 640x480 PNG (at most one every 2 seconds)
    Q or ESC   quit safely (every image is already saved when you press SPACE)
Ctrl+C in the terminal also quits safely.

Everything is written inside the Git-ignored `dataset/` folder:
    dataset/images/<condition>_<participant>_<nnn>.png
    dataset/metadata.csv

Safety rules enforced here (see the functions named in brackets):
  * images are created exclusively and never overwritten            [save_capture]
  * a failed or interrupted save removes its own partial image      [save_capture]
  * metadata rows are validated; duplicates/mismatches are errors   [read_metadata]
  * the CSV is never appended onto an incomplete last row; a failed
    append restores the previous file length                        [append_metadata]
  * Git itself must confirm that dataset paths are ignored          [check_gitignore]
  * symbolic links / junctions in dataset paths are refused         [check_layout]
  * metadata rows reserve their image numbers: never delete a row   [load_state]
"""
import argparse
import csv
import io
import os
import re
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import Camera, CameraError  # noqa: E402  (needs the path line above)

DATASET_DIR = PROJECT_ROOT / "dataset"
EXPECTED_SHAPE = (480, 640, 3)          # height, width, channels
MIN_INTERVAL_SECONDS = 2.0
METADATA_FIELDS = ["filename", "participant_id", "condition",
                   "sub_condition", "capture_time", "notes"]
PARTICIPANTS = ("P01", "P02", "NONE")   # NONE = background-only images
MAX_NOTE_LENGTH = 200
# Paths that Git must report as ignored before any capture is allowed.
GITIGNORE_PROBES = ("dataset/images/probe_P01_001.png", "dataset/metadata.csv",
                    "dataset/labels.csv")

# condition code -> label, total target, {sub_condition: target or None}, needs a person.
# Totals and splits come from docs/EVALUATION_PROTOCOL.md section 3.
CONDITIONS = {
    "normal":  {"label": "Normal indoor lighting", "target": 20,
                "subs": {"frontal": None}, "person": True},
    "dim":     {"label": "Dim lighting", "target": 15,
                "subs": {"dim": None}, "person": True},
    "strong":  {"label": "Strong side or back lighting", "target": 15,
                "subs": {"side": None, "back": None}, "person": True},
    "turn":    {"label": "Head turned", "target": 20,
                "subs": {"30deg": 10, "60deg": 10}, "person": True},
    "tilt":    {"label": "Head tilted / looking down", "target": 15,
                "subs": {"roll": None, "down": None}, "person": True},
    "dist":    {"label": "Near / far distance", "target": 20,
                "subs": {"near": 10, "far": 10}, "person": True},
    "glasses": {"label": "Glasses (only if available)", "target": 15,
                "subs": {"glasses": None}, "person": True},
    "bg":      {"label": "Background-only (nobody in view)", "target": 20,
                "subs": {"background": None}, "person": False},
}

FILENAME_PATTERN = re.compile(r"^([a-z]+)_(P01|P02|NONE)_(\d{3})\.png$")


class CaptureError(Exception):
    """A problem the user can understand and fix (bad input, existing file, ...)."""


# ----------------------------------------------------------------- validation
def validate_selection(condition, participant, sub_condition):
    """Raise CaptureError if the combination is not allowed by the protocol."""
    if condition not in CONDITIONS:
        raise CaptureError(f"Unknown condition '{condition}'. "
                           f"Choose one of: {', '.join(CONDITIONS)}.")
    if participant not in PARTICIPANTS:
        raise CaptureError(f"Unknown participant '{participant}'. "
                           f"Choose one of: {', '.join(PARTICIPANTS)}.")
    info = CONDITIONS[condition]
    if info["person"] and participant == "NONE":
        raise CaptureError(f"Condition '{condition}' needs a participant (P01 or P02).")
    if not info["person"] and participant != "NONE":
        raise CaptureError("Background-only images must use participant NONE "
                           "(nobody may be in view).")
    if sub_condition not in info["subs"]:
        raise CaptureError(f"Unknown sub-condition '{sub_condition}' for '{condition}'. "
                           f"Choose one of: {', '.join(info['subs'])}.")


def clean_note(note):
    """One line, limited length. Notes must never contain names or personal details."""
    note = " ".join(str(note or "").split())
    if len(note) > MAX_NOTE_LENGTH:
        raise CaptureError(f"Note is too long ({len(note)} characters, "
                           f"maximum {MAX_NOTE_LENGTH}).")
    return note


def build_filename(condition, participant, index):
    """<condition>_<participant>_<nnn>.png, for example dim_P02_007.png."""
    if not 1 <= index <= 999:
        raise CaptureError("Image number must be between 1 and 999.")
    return f"{condition}_{participant}_{index:03d}.png"


# ----------------------------------------------------- paths, links and Git
def is_link(path):
    """True for symbolic links AND Windows junctions/reparse points (never follows them)."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    reparse = getattr(info, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(info.st_mode) or bool(reparse)


def images_dir(dataset_dir):
    return Path(dataset_dir) / "images"


def metadata_path(dataset_dir):
    return Path(dataset_dir) / "metadata.csv"


def ensure_inside(dataset_dir, path):
    """Refuse to touch any path whose real location is outside the dataset folder."""
    try:
        Path(path).resolve().relative_to(Path(dataset_dir).resolve())
    except ValueError:
        raise CaptureError(f"Refusing to write outside the dataset folder: {path}")


def check_layout(dataset_dir):
    """Reject links/junctions/odd types for dataset/, dataset/images and metadata.csv.

    Nothing is created or followed. A link could silently send private images
    to another folder, so the session stops instead. (DATASET_DIR is a direct child
    of the already-resolved project folder, so these three paths are the only places
    a link could redirect writes; image destinations are also checked by
    ensure_inside() and are created exclusively.)
    """
    dataset_dir = Path(dataset_dir)
    parts = ((dataset_dir, "dataset folder"), (images_dir(dataset_dir), "images folder"),
             (metadata_path(dataset_dir), "metadata file"))
    for path, kind in parts:
        if is_link(path):
            raise CaptureError(f"The {kind} ({path}) is a symbolic link or junction. "
                               "Refusing to use it: it could redirect private images "
                               "elsewhere.")
    for path, kind in parts[:2]:
        if os.path.lexists(path) and not os.path.isdir(path):
            raise CaptureError(f"The {kind} ({path}) exists but is not a folder.")
    meta = parts[2][0]
    if os.path.lexists(meta) and not os.path.isfile(meta):
        raise CaptureError(f"The metadata file ({meta}) exists but is not a regular file.")


def check_gitignore(project_root=PROJECT_ROOT):
    """Ask Git itself whether dataset paths are ignored (handles every pattern form).

    `git check-ignore -q` exits 0 if the path is ignored and 1 if it is not. A later
    negation rule (!dataset/) or an already tracked file makes it exit 1, so the
    script stops instead of risking a commit of private images.
    """
    for probe in GITIGNORE_PROBES:
        try:
            result = subprocess.run(
                ["git", "-C", str(project_root), "check-ignore", "-q", "--", probe],
                capture_output=True, text=True, timeout=15)
        except FileNotFoundError:
            raise CaptureError("Git was not found, so the ignore rules cannot be verified. "
                               "Install Git or fix PATH. Nothing was captured.")
        except subprocess.TimeoutExpired:
            raise CaptureError("'git check-ignore' timed out. Nothing was captured.")
        if result.returncode == 1:
            raise CaptureError(
                f"Git does NOT ignore '{probe}' (missing or negated rule in .gitignore, "
                "or the path is already tracked). Fix .gitignore first so images "
                "cannot be committed by accident.")
        if result.returncode != 0:
            raise CaptureError(f"Could not verify Git ignore rules (git exit code "
                               f"{result.returncode}: {result.stderr.strip()}). "
                               f"Is {project_root} a Git repository?")


# ------------------------------------------------------------------- metadata
def check_csv_tail(path):
    """The file must end with a line break; otherwise the last row may be incomplete."""
    if os.path.getsize(path) == 0:
        raise CaptureError(f"{path} is empty (no header). If no images exist yet, "
                           "delete this empty file; otherwise restore it from a backup.")
    with open(path, "rb") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) != b"\n":
            raise CaptureError(
                f"{path} does not end with a line break, so its last row may be "
                "incomplete. Not appending onto it. If the last row is complete, add "
                "a line break at the end of the file; otherwise restore a backup.")


def validate_row(row, line_number, label):
    """Check one metadata row; returns (condition, number). Raises CaptureError."""
    where = f"{label} line {line_number}"
    if None in row:
        raise CaptureError(f"{where}: more values than columns (row is malformed).")
    if any(value is None for value in row.values()):
        raise CaptureError(f"{where}: fewer values than columns (incomplete row).")
    match = FILENAME_PATTERN.match(row["filename"])
    if not match:
        raise CaptureError(f"{where}: filename '{row['filename']}' does not match "
                           "<condition>_<participant>_<nnn>.png.")
    condition, participant, number = match.group(1), match.group(2), int(match.group(3))
    if row["condition"] != condition:
        raise CaptureError(f"{where}: condition '{row['condition']}' disagrees with "
                           f"filename '{row['filename']}'.")
    if row["participant_id"] != participant:
        raise CaptureError(f"{where}: participant '{row['participant_id']}' disagrees "
                           f"with filename '{row['filename']}'.")
    try:
        validate_selection(condition, participant, row["sub_condition"])
    except CaptureError as error:
        raise CaptureError(f"{where}: {error}")
    try:
        datetime.fromisoformat(row["capture_time"])
    except ValueError:
        raise CaptureError(f"{where}: capture_time '{row['capture_time']}' is not an "
                           "ISO date-time.")
    if number < 1:
        raise CaptureError(f"{where}: image number must be at least 001.")
    return condition, number


def read_metadata(dataset_dir):
    """Return all metadata rows, fully validated. Missing file = no rows.

    Rejects: wrong header, incomplete/odd rows, filename/condition/participant
    disagreement, duplicate filenames, duplicate numbers within a condition,
    and a missing final line break. A UTF-8 BOM (added by some editors) is accepted.
    """
    check_layout(dataset_dir)
    path = metadata_path(dataset_dir)
    if not os.path.lexists(path):
        return []
    check_csv_tail(path)
    rows, seen_names, seen_numbers = [], {}, {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != METADATA_FIELDS:
                raise CaptureError(
                    f"{path} has unexpected columns {reader.fieldnames}; expected "
                    f"{METADATA_FIELDS}. Not touching it. Fix or move the file.")
            for row in reader:
                condition, number = validate_row(row, reader.line_num, path.name)
                name, line = row["filename"], reader.line_num
                if name in seen_names:
                    raise CaptureError(f"{path.name} line {line}: duplicate row for "
                                       f"'{name}' (first seen on line {seen_names[name]}).")
                if (condition, number) in seen_numbers:
                    first = seen_numbers[(condition, number)]
                    raise CaptureError(f"{path.name} line {line}: image number {number:03d} "
                                       f"of '{condition}' is used twice (line {first}).")
                seen_names[name] = seen_numbers[(condition, number)] = line
                rows.append(row)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CaptureError(f"Cannot read {path}: {error}")
    return rows


def _truncate_file(path, size):
    with open(path, "r+b") as handle:
        handle.truncate(size)
        handle.flush()
        os.fsync(handle.fileno())


def _remove_file(path):
    os.remove(path)


def append_metadata(dataset_dir, row):
    """Append one row safely.

    Refuses to append onto an incomplete last row. The write is flushed and fsync'd;
    if anything goes wrong (including Ctrl+C) the file is restored to its previous
    length (or removed if this call created it) and the original problem is reported.
    """
    check_layout(dataset_dir)
    path = metadata_path(dataset_dir)
    ensure_inside(dataset_dir, path)
    existed = os.path.lexists(path)
    previous_size = 0
    if existed:
        check_csv_tail(path)
        previous_size = os.path.getsize(path)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=METADATA_FIELDS)
    if not existed:
        writer.writeheader()
    writer.writerow(row)
    data = buffer.getvalue().encode("utf-8")

    opened = False
    try:
        # "xb" = create (fails if it appeared meanwhile); "ab" = append to the existing file.
        with open(path, "ab" if existed else "xb") as handle:
            opened = True
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException as error:
        problem = None
        if opened:                                  # something may have been written
            try:
                if existed:
                    _truncate_file(path, previous_size)
                else:
                    _remove_file(path)
            except OSError as rollback_error:
                problem = str(rollback_error)
        if problem:
            raise CaptureError(
                f"Writing {path} failed ({error}) AND restoring it failed ({problem}). "
                "The file may end with a partial row. Fix it by hand (it should end "
                "after the last complete row) before capturing again.") from error
        if isinstance(error, OSError):
            raise CaptureError(f"Cannot write {path}: {error}. The file was restored "
                               "to its previous state.") from error
        raise


# ---------------------------------------------------------- folder and state
def scan_images(dataset_dir):
    """Look at images/ without following anything.

    Returns (names, problems, unexpected):
      names      - entries named like dataset images (these numbers are in use)
      problems   - those entries that are links/folders instead of regular files
      unexpected - other entries (Thumbs.db, notes, ...): reported, never touched
    """
    check_layout(dataset_dir)
    folder = images_dir(dataset_dir)
    names, problems, unexpected = set(), [], []
    if not os.path.lexists(folder):
        return names, problems, unexpected
    try:
        entries = sorted(os.listdir(folder))
    except OSError as error:
        raise CaptureError(f"Cannot list {folder}: {error}")
    for entry in entries:
        path = folder / entry
        if FILENAME_PATTERN.match(entry):
            names.add(entry)
            if is_link(path) or not os.path.isfile(path):
                problems.append(entry)
        else:
            unexpected.append(entry)
    return names, problems, unexpected


def load_state(dataset_dir):
    """Validate everything on disk and return counts. Raises CaptureError on problems.

    Returns a dict: rows, names, warnings, by_condition, by_sub, by_participant, used
    (used = {condition: set of numbers that must never be reused}).
    Only rows whose image file exists are counted. Duplicate rows cannot inflate the
    counts because read_metadata() rejects them.
    """
    rows = read_metadata(dataset_dir)
    names, problems, unexpected = scan_images(dataset_dir)
    if problems:
        raise CaptureError(f"{problems[0]} (and {len(problems) - 1} more) in "
                           f"{images_dir(dataset_dir)} is a link or folder, not an image "
                           "file. Remove it first.")
    recorded = {r["filename"] for r in rows}
    unrecorded = sorted(names - recorded)
    if unrecorded:
        raise CaptureError(
            f"{len(unrecorded)} image(s) in {images_dir(dataset_dir)} have no metadata row "
            f"(first: {unrecorded[0]}). This usually follows a crash. Not capturing, to "
            "avoid mixing up the data. Delete that image file (only the PNG) or "
            "re-create its row.")
    warnings = []
    if unexpected:
        shown = ", ".join(unexpected[:3]) + (", ..." if len(unexpected) > 3 else "")
        warnings.append(f"Ignoring {len(unexpected)} unexpected item(s) in the images "
                        f"folder (not dataset images): {shown}")
    for name in sorted(recorded - names):
        warnings.append(f"Metadata row without an image file (number stays reserved, "
                        f"image not counted): {name}")
    used = {}
    for name in recorded | names:
        match = FILENAME_PATTERN.match(name)
        if match:
            used.setdefault(match.group(1), set()).add(int(match.group(3)))
    for condition, numbers in sorted(used.items()):
        gaps = sorted(set(range(1, max(numbers) + 1)) - numbers)
        if gaps:
            warnings.append(f"Numbers {gaps} of '{condition}' have neither an image nor a "
                            "metadata row. Metadata rows reserve numbers: never delete "
                            "rows from metadata.csv.")
    by_condition, by_sub, by_participant = {}, {}, {}
    for row in rows:
        if row["filename"] not in names:
            continue
        cond, sub, pid = row["condition"], row["sub_condition"], row["participant_id"]
        by_condition[cond] = by_condition.get(cond, 0) + 1
        by_sub[(cond, sub)] = by_sub.get((cond, sub), 0) + 1
        by_participant[pid] = by_participant.get(pid, 0) + 1
    return {"rows": rows, "names": names, "warnings": warnings, "by_condition": by_condition,
            "by_sub": by_sub, "by_participant": by_participant, "used": used}


def check_consistency(dataset_dir):
    """Raise on problems; return a list of warnings."""
    return load_state(dataset_dir)["warnings"]


def progress(dataset_dir):
    """Counts of saved images: ({condition: n}, {(condition, sub): n}, {participant: n})."""
    state = load_state(dataset_dir)
    return state["by_condition"], state["by_sub"], state["by_participant"]


def next_index(dataset_dir, condition):
    """Next free running number for this condition (never reuses a reserved number)."""
    used = load_state(dataset_dir)["used"].get(condition, set())
    return max(used, default=0) + 1


def check_target_not_reached(dataset_dir, condition, sub_condition, state=None):
    state = state or load_state(dataset_dir)
    info = CONDITIONS[condition]
    if state["by_condition"].get(condition, 0) >= info["target"]:
        raise CaptureError(f"Target reached for '{condition}' "
                           f"({info['target']} images). No more images needed.")
    sub_target = info["subs"][sub_condition]
    saved_sub = state["by_sub"].get((condition, sub_condition), 0)
    if sub_target is not None and saved_sub >= sub_target:
        raise CaptureError(f"Target reached for '{condition}/{sub_condition}' "
                           f"({sub_target} images). Pick the other sub-condition.")


# ----------------------------------------------------------- interval control
class CaptureGate:
    """Allows a capture only if MIN_INTERVAL_SECONDS passed since the last one."""

    def __init__(self, min_interval=MIN_INTERVAL_SECONDS, clock=time.monotonic):
        self.min_interval = min_interval
        self.clock = clock          # injectable so tests need no real waiting
        self._last = None

    def seconds_remaining(self):
        if self._last is None:
            return 0.0
        return max(0.0, self.min_interval - (self.clock() - self._last))

    def ready(self):
        return self.seconds_remaining() == 0.0

    def mark(self):
        """Call after a capture was saved successfully."""
        self._last = self.clock()


# ------------------------------------------------------------------- saving
def save_capture(dataset_dir, frame, condition, participant, sub_condition,
                 notes="", now=None):
    """Validate, save one PNG without overwriting, record metadata. Returns the filename.

    Order: PNG first (created exclusively), then the CSV row. If anything fails or is
    interrupted after the PNG was created, that PNG is removed again; if the removal
    itself fails the problem is reported explicitly.
    """
    validate_selection(condition, participant, sub_condition)
    notes = clean_note(notes)
    if (frame is None or tuple(getattr(frame, "shape", ())) != EXPECTED_SHAPE
            or str(getattr(frame, "dtype", "")) != "uint8"):
        raise CaptureError(f"Frame must be a 640x480 8-bit colour image (shape "
                           f"{EXPECTED_SHAPE}, uint8), got {getattr(frame, 'shape', None)} "
                           f"{getattr(frame, 'dtype', None)}. Image not saved.")
    state = load_state(dataset_dir)                       # validates everything first
    check_target_not_reached(dataset_dir, condition, sub_condition, state)

    number = max(state["used"].get(condition, set()), default=0) + 1
    filename = build_filename(condition, participant, number)
    folder = images_dir(dataset_dir)
    path = folder / filename
    ensure_inside(dataset_dir, path)
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise CaptureError("OpenCV could not encode the frame as PNG.")
    stamp = now or datetime.now()
    if stamp.tzinfo is None:
        stamp = stamp.astimezone()                        # attach the local UTC offset
    row = {"filename": filename, "participant_id": participant, "condition": condition,
           "sub_condition": sub_condition,
           "capture_time": stamp.isoformat(timespec="seconds"), "notes": notes}

    try:
        folder.mkdir(parents=True, exist_ok=True)
        check_layout(dataset_dir)                         # images/ may just have been created
        # Mode "xb" = create new file, FAIL if it already exists (never overwrites).
        handle = open(path, "xb")
    except FileExistsError:
        raise CaptureError(f"{filename} already exists; refusing to overwrite it.")
    except OSError as error:
        raise CaptureError(f"Cannot save {path}: {error}")
    try:
        with handle:                                      # we created it: we may remove it
            handle.write(encoded.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        append_metadata(dataset_dir, row)
    except BaseException as error:
        failure = None
        try:
            _remove_file(path)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            failure = str(cleanup_error)
        extra = ""
        if failure:
            extra = (f" ALSO: could not remove the incomplete image {path} ({failure}); "
                     "delete that file by hand before capturing again.")
            print(f"WARNING:{extra}", file=sys.stderr)
        if isinstance(error, OSError):
            raise CaptureError(f"Cannot save {path}: {error}.{extra}") from error
        if failure and isinstance(error, CaptureError):
            raise CaptureError(f"{error}{extra}") from error
        raise                                             # KeyboardInterrupt etc. unchanged
    return filename


# --------------------------------------------------------- terminal handling
def print_status(dataset_dir):
    state = load_state(dataset_dir)
    for warning in state["warnings"]:
        print(f"Warning: {warning}")
    by_condition, by_sub = state["by_condition"], state["by_sub"]
    print("Dataset progress (saved images / target)")
    total = 0
    for cond, info in CONDITIONS.items():
        n = by_condition.get(cond, 0)
        total += n
        parts = []
        for sub, sub_target in info["subs"].items():
            if len(info["subs"]) > 1:
                shown = f"{by_sub.get((cond, sub), 0)}" + (f"/{sub_target}" if sub_target else "")
                parts.append(f"{sub}: {shown}")
        extra = f"   ({', '.join(parts)})" if parts else ""
        print(f"  {cond:8s} {n:3d}/{info['target']:<3d} {info['label']}{extra}")
    print(f"  total    {total:3d}/140  (glasses may be dropped: then 125)")
    print("  per participant: " + ", ".join(
        f"{pid}={state['by_participant'].get(pid, 0)}" for pid in PARTICIPANTS))


def ask_choice(title, options):
    """options: {key: description}. Re-asks until valid. Raises CaptureError on EOF."""
    print(f"\n{title}")
    for key, description in options.items():
        print(f"  {key:10s} {description}")
    while True:
        try:
            answer = input("> ").strip()
        except EOFError:
            raise CaptureError("No input available; use the command-line options instead.")
        if answer in options:
            return answer
        print(f"  '{answer}' is not valid. Type one of: {', '.join(options)}")


CONSENT_NOTICE = (
    "This confirmation is a SELF-DECLARATION only. It does not replace the signed consent\n"
    "form, which must exist before any image of P02 is taken and is kept outside the\n"
    "project. Permission to show images in the report or slides is a SEPARATE yes/no on\n"
    "that form and is not recorded here.")


def confirm_consent(participant, already_confirmed):
    """P02 is a volunteer: signed consent must exist before any image is taken."""
    if participant != "P02":
        return
    print(f"\nP02 is the volunteer.\n{CONSENT_NOTICE}")
    if already_confirmed:
        print("Consent was declared with --consent-confirmed (not checked by this script).")
        return
    try:
        answer = input("Has P02 signed the consent form? Type 'yes' to continue: ")
    except EOFError:
        answer = ""
    if answer.strip().lower() != "yes":
        raise CaptureError("Consent not confirmed. Nothing was captured.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture evaluation images (see docs/EVALUATION_PROTOCOL.md).")
    parser.add_argument("--status", action="store_true",
                        help="show progress and exit (no camera is opened)")
    parser.add_argument("--condition", choices=list(CONDITIONS))
    parser.add_argument("--participant", choices=list(PARTICIPANTS))
    parser.add_argument("--sub", dest="sub_condition", help="sub-condition, e.g. 30deg")
    parser.add_argument("--notes", default="", help="optional note saved with every image "
                        "(no names or personal details)")
    parser.add_argument("--consent-confirmed", action="store_true",
                        help="declare that P02 has signed the consent form "
                             "(self-declaration; not verified)")
    return parser.parse_args(argv)


def resolve_selection(args):
    """Fill in anything not given on the command line by asking the user."""
    condition = args.condition or ask_choice(
        "Choose a condition:",
        {c: f"{i['label']} (target {i['target']})" for c, i in CONDITIONS.items()})
    info = CONDITIONS[condition]
    if info["person"]:
        participant = args.participant or ask_choice(
            "Choose the participant:", {"P01": "project author", "P02": "volunteer"})
    else:
        participant = "NONE"
        if args.participant not in (None, "NONE"):
            raise CaptureError("Background-only images must use participant NONE.")
    sub = args.sub_condition
    if sub is None:
        subs = list(info["subs"])
        sub = subs[0] if len(subs) == 1 else ask_choice(
            "Choose the sub-condition:", {s: "" for s in subs})
    validate_selection(condition, participant, sub)
    return condition, participant, sub, clean_note(args.notes)


def overlay(frame, lines, color=(0, 255, 0)):
    shown = frame.copy()       # draw on a COPY: the saved image stays untouched
    for i, text in enumerate(lines):
        cv2.putText(shown, text, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
    return shown


def run_capture(dataset_dir, condition, participant, sub, notes, state):
    """Live window. `state` was loaded once at session start (no disk scans per frame).

    The on-screen counter is updated after each successful save; save_capture() still
    re-validates the disk, so target limits are always enforced from the real files.
    """
    window = "Dataset capture - SPACE = save, Q/ESC = quit"
    camera = Camera(index=0)
    gate = CaptureGate()
    message, message_until = "", 0.0
    saved_now = 0
    saved_total = state["by_condition"].get(condition, 0)
    target = CONDITIONS[condition]["target"]
    try:
        camera.open()
        props = camera.properties()
        if (props["width"], props["height"]) != (EXPECTED_SHAPE[1], EXPECTED_SHAPE[0]):
            raise CaptureError(f"Camera resolution is {props['width']}x{props['height']}, "
                               "but the protocol needs 640x480. Nothing captured.")
        print("\nCamera ready. Click the video window, then press SPACE to save, Q/ESC to quit.")
        while True:
            frame = camera.read()
            lines = [f"{condition}/{sub}  {participant}   {saved_total}/{target} saved"]
            wait = gate.seconds_remaining()
            lines.append(f"wait {wait:.1f}s" if wait > 0 else "ready - press SPACE")
            if time.monotonic() < message_until:
                lines.append(message)
            cv2.imshow(window, overlay(frame, lines))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:                                   # SPACE
                if not gate.ready():
                    message = f"too soon: wait {gate.seconds_remaining():.1f}s"
                else:
                    try:
                        name = save_capture(dataset_dir, frame, condition, participant,
                                            sub, notes)
                        gate.mark()
                        saved_now += 1
                        saved_total += 1
                        message = f"saved {name}"
                        print(f"Saved {name}")
                    except CaptureError as error:
                        message = str(error)[:60]
                        print(f"Not saved: {error}")
                message_until = time.monotonic() + 2.0
        print(f"\nStopped. Saved {saved_now} image(s) this session.")
        return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.status:
            print_status(DATASET_DIR)
            return 0
        check_gitignore()
        state = load_state(DATASET_DIR)
        for warning in state["warnings"]:
            print(f"Warning: {warning}")
        condition, participant, sub, notes = resolve_selection(args)
        confirm_consent(participant, args.consent_confirmed)
        check_target_not_reached(DATASET_DIR, condition, sub, state)
        return run_capture(DATASET_DIR, condition, participant, sub, notes, state)
    except CaptureError as error:
        print(f"Error: {error}")
        return 1
    except CameraError as error:
        print(f"Camera error: {error}")
        return 1
    except KeyboardInterrupt:
        print("\nCancelled by user. Images already saved are kept.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
