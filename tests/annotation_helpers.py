"""Shared helpers for the annotation tests (synthetic images, temporary datasets only)."""
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import annotate_dataset as ad  # noqa: E402
import validate_labels as vl  # noqa: E402

T0 = datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone(timedelta(hours=5)))
FACE_DRAG = (200, 100, 440, 360)          # original pixels: x0, y0, x1, y1
LEFT_EYE, RIGHT_EYE = (270, 190), (370, 190)


def make_dataset(root, n_person=20, n_bg=4):
    """A tiny synthetic dataset: n_person person images + n_bg background images."""
    dataset = Path(root) / "dataset"
    (dataset / "images").mkdir(parents=True)
    rng = np.random.RandomState(1)
    rows = []
    for kind, count, cond, pid, sub in (("normal", n_person, "normal", "P01", "frontal"),
                                        ("bg", n_bg, "bg", "NONE", "background")):
        for i in range(1, count + 1):
            name = f"{kind}_{pid}_{i:03d}.png"
            img = rng.randint(0, 255, (480, 640, 3)).astype(np.uint8)
            ok, buf = cv2.imencode(".png", img)
            (dataset / "images" / name).write_bytes(buf.tobytes())
            rows.append({"filename": name, "participant_id": pid, "condition": cond,
                         "sub_condition": sub,
                         "capture_time": (T0 + timedelta(seconds=i)).isoformat(), "notes": ""})
    with open(dataset / "metadata.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dataset


def full_annotation(face=FACE_DRAG, left=LEFT_EYE, right=RIGHT_EYE):
    a = ad.Annotation()
    a.set_face(*face)
    a.place_eye(*left)
    a.place_eye(*right)
    return a


def no_face_annotation():
    a = ad.Annotation()
    a.mark_no_face()
    return a


def annotate_everything(dataset, pass_no=1, now=T0, names=None):
    """Annotate all images (or `names`) through save_image: persons with a face and two
    eyes, backgrounds as no_face."""
    meta = vl.load_metadata(dataset)
    for i, name in enumerate(sorted(names if names is not None else meta)):
        bg = vl.is_background(meta[name])
        ad.save_image(dataset, pass_no, name, no_face_annotation() if bg else full_annotation(),
                      bg, now + timedelta(seconds=i))


def write_labels(dataset, rows, pass_no=1):
    path, _ = vl.label_paths(dataset, pass_no)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=vl.LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_status(dataset, rows, pass_no=1):
    _, path = vl.label_paths(dataset, pass_no)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=vl.STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def face_row(name, x=200, y=100, w=240, h=260, unc=0):
    return {"filename": name, "object_type": "face", "x": x, "y": y, "w": w, "h": h,
            "side": "", "uncertain": unc}


def eye_row(name, side, x, y, unc=0):
    return {"filename": name, "object_type": "eye", "x": x, "y": y, "w": "", "h": "",
            "side": side, "uncertain": unc}


def status_row(name, status="labelled", n_face=1, n_eye=2, pass_no=1,
               at="2026-11-20T10:00:00+05:00"):
    return {"filename": name, "status": status, "n_face": n_face, "n_eye": n_eye,
            "annotated_at": at, "pass": pass_no, "tool_version": "1.0"}
