"""Build and verify the de-duplicated evaluation subset (Milestone 6).

Approved rule (docs/EVALUATION_PROTOCOL.md, section 11): "Method B" with T = 2.5.

  1. Convert each image to grayscale (cv2.COLOR_BGR2GRAY) and resize it to 64 x 48 with
     cv2.INTER_AREA (area averaging); keep float32.
  2. Group images by recorded (condition, sub_condition).
  3. Inside a group, walk through the images in capture order.
  4. Compare each image with the FIRST image of the current cluster using the mean
     absolute difference (MAD) of the small grayscale images.
  5. If MAD > T the image starts a new cluster, otherwise it joins the current one.
  6. The earliest image of each cluster is its representative.

Only pixel similarity is used: no detector, no face/eye labels. The script never changes
an image or metadata.csv; it only writes three files into dataset/deduplication/.

    python scripts/build_dedup_manifest.py            create the files (or confirm they match)
    python scripts/build_dedup_manifest.py --verify   re-derive everything and check the files
"""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

THRESHOLD = 2.5
SMALL_SIZE = (64, 48)                    # (width, height) for cv2.resize
EXPECTED_SHAPE = (480, 640, 3)
CONDITION_ORDER = ["normal", "dim", "strong", "turn", "tilt", "dist", "glasses", "bg"]
EXPECTED = {"images": 125, "representatives": 73, "person_representatives": 68,
            "background_representatives": 5}
MANIFEST_FIELDS = [
    "filename", "condition", "sub_condition", "image_type", "participant_id", "capture_time",
    "cluster_id", "cluster_size", "representative_filename", "status",
    "mad_to_representative", "in_dedup_subset", "person_metrics_full", "person_metrics_dedup",
    "background_fp_full", "background_fp_dedup", "source_sha256"]
SUMMARY_FIELDS = ["scope", "condition", "sub_condition", "images", "clusters_representatives",
                  "excluded", "largest_cluster", "person_images", "background_images"]

# The seven merges flagged "questionable" or "unsafe" in the visual audit of Method B.
# Each entry: description, image that must NOT share a cluster with the first image listed.
FLAGGED_MERGES = [
    ("TURN-03", "turn_P01_005.png", "turn_P01_002.png"),
    ("TURN-05", "turn_P01_010.png", "turn_P01_006.png"),
    ("TURN-06", "turn_P01_015.png", "turn_P01_012.png"),
    ("TURN-11", "turn_P01_019.png", "turn_P01_016.png"),
    ("STRO-06", "strong_P01_013.png", "strong_P01_011.png"),
    ("DIM-04", "dim_P01_008.png", "dim_P01_005.png"),
    ("DIM-10", "dim_P01_014.png", "dim_P01_013.png"),
]
VISUAL_AUDIT = {"clusters_reviewed": 63, "safe": 56, "questionable": 6, "unsafe": 1,
                "unreviewed": 0, "thresholds_audited": [2.0, 2.5, 3.0, 4.0]}


class DedupError(Exception):
    """Inconsistent input or output; nothing is guessed or repaired."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_metadata(dataset_dir):
    path = Path(dataset_dir) / "metadata.csv"
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise DedupError(f"Cannot read {path}: {error}")
    need = {"filename", "participant_id", "condition", "sub_condition", "capture_time"}
    if not rows or not need <= set(rows[0]):
        raise DedupError(f"{path} is empty or lacks the columns {sorted(need)}")
    return rows


def load_small(path):
    """Return (sha256 of the file, 64x48 float32 grayscale image)."""
    data = Path(path).read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != EXPECTED_SHAPE:
        raise DedupError(f"{path} is unreadable or not 640x480 colour")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, SMALL_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    return sha256_bytes(data), small


def capture_order(rows):
    """Rows of one group in capture order; fails if filename numbers disagree with time."""
    ordered = sorted(rows, key=lambda r: (datetime.fromisoformat(r["capture_time"]),
                                          r["filename"]))
    by_name = sorted(rows, key=lambda r: r["filename"])
    if [r["filename"] for r in ordered] != [r["filename"] for r in by_name]:
        raise DedupError("capture-time order differs from filename-number order in "
                         f"{rows[0]['condition']}/{rows[0]['sub_condition']}")
    return ordered


def mad(a, b):
    """Mean absolute difference of two small grayscale images (gray levels)."""
    return float(np.abs(a - b).mean())


def cluster_group(names, small, threshold=THRESHOLD):
    """Clusters of one group: list of lists of (filename, mad_to_first). First = earliest."""
    clusters = []
    for name in names:
        if clusters:
            d = mad(small[name], small[clusters[-1][0][0]])
            if d <= threshold:
                clusters[-1].append((name, d))
                continue
        clusters.append([(name, 0.0)])
    return clusters


def build(dataset_dir=DATASET_DIR, threshold=THRESHOLD):
    """Return (manifest_rows, summary_rows, methodology dict). Pure function of the data."""
    dataset_dir = Path(dataset_dir)
    rows = read_metadata(dataset_dir)
    meta = {r["filename"]: r for r in rows}
    if len(meta) != len(rows):
        raise DedupError("duplicate filenames in metadata.csv")
    image_dir = dataset_dir / "images"
    hashes, small = {}, {}
    for name in sorted(meta):
        hashes[name], small[name] = load_small(image_dir / name)
    groups = {}
    for r in rows:
        groups.setdefault((r["condition"], r["sub_condition"]), []).append(r)
    unknown = {c for c, _ in groups} - set(CONDITION_ORDER)
    if unknown:
        raise DedupError(f"unknown conditions in metadata: {sorted(unknown)}")
    ordered_groups = sorted(groups, key=lambda k: (CONDITION_ORDER.index(k[0]),
                                                   min(r["filename"] for r in groups[k])))
    manifest = []
    for cond, sub in ordered_groups:
        names = [r["filename"] for r in capture_order(groups[(cond, sub)])]
        for k, cluster in enumerate(cluster_group(names, small, threshold), start=1):
            rep = cluster[0][0]
            for name, d in cluster:
                m = meta[name]
                is_rep = name == rep
                person = cond != "bg"
                manifest.append({
                    "filename": name, "condition": cond, "sub_condition": sub,
                    "image_type": "person" if person else "background",
                    "participant_id": m["participant_id"], "capture_time": m["capture_time"],
                    "cluster_id": f"{cond}-{sub}-C{k:02d}", "cluster_size": len(cluster),
                    "representative_filename": rep,
                    "status": "representative" if is_rep else "excluded",
                    "mad_to_representative": f"{d:.4f}",
                    "in_dedup_subset": int(is_rep),
                    "person_metrics_full": int(person), "person_metrics_dedup": int(is_rep and person),
                    "background_fp_full": int(not person), "background_fp_dedup": int(is_rep and not person),
                    "source_sha256": hashes[name]})
    summary = summarise(manifest)
    aggregate = sha256_bytes("\n".join(f"{n}:{hashes[n]}" for n in sorted(hashes)).encode())
    meta_hash = sha256_bytes((dataset_dir / "metadata.csv").read_bytes())
    methodology = describe(threshold, manifest, aggregate, meta_hash)
    return manifest, summary, methodology


def summarise(manifest):
    out = []

    def line(scope, cond, sub, items):
        sizes = {}
        for r in items:
            sizes[r["cluster_id"]] = r["cluster_size"]
        out.append({"scope": scope, "condition": cond, "sub_condition": sub,
                    "images": len(items), "clusters_representatives": len(sizes),
                    "excluded": sum(r["status"] == "excluded" for r in items),
                    "largest_cluster": max(sizes.values()) if sizes else 0,
                    "person_images": sum(r["image_type"] == "person" for r in items),
                    "background_images": sum(r["image_type"] == "background" for r in items)})
    line("overall", "all", "all", manifest)
    line("person", "all", "all", [r for r in manifest if r["image_type"] == "person"])
    line("background", "bg", "all", [r for r in manifest if r["image_type"] == "background"])
    seen = []
    for r in manifest:
        key = (r["condition"], r["sub_condition"])
        if key not in seen:
            seen.append(key)
    for cond in CONDITION_ORDER:
        items = [r for r in manifest if r["condition"] == cond]
        if items:
            line("condition", cond, "all", items)
    for cond, sub in seen:
        line("condition_sub", cond, sub,
             [r for r in manifest if (r["condition"], r["sub_condition"]) == (cond, sub)])
    return out


def describe(threshold, manifest, images_hash, metadata_hash):
    reps = [r for r in manifest if r["status"] == "representative"]
    return {
        "method": "B: sub-condition-aware raw pixel comparison",
        "status": "approved final rule (Milestone 6)",
        "threshold_T_gray_levels": threshold,
        "grouping": "separately for each recorded (condition, sub_condition)",
        "order": "capture order (capture_time; must equal filename-number order)",
        "preprocessing": {"grayscale": "cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)",
                          "resize_width_height": list(SMALL_SIZE),
                          "interpolation": "cv2.INTER_AREA", "dtype": "float32",
                          "normalisation": "none (raw gray levels 0-255)"},
        "difference": "MAD = mean(abs(image - anchor)) over the 64x48 small images",
        "anchor": "the first (earliest) image of the current cluster",
        "new_cluster_rule": "MAD > T starts a new cluster; MAD <= T joins the current one",
        "representative": "the earliest image of each cluster",
        "inputs_used": "pixels and recorded condition/sub_condition/capture_time only; "
                       "no detector output and no ground-truth labels",
        "background_images": "kept in the manifest; flagged background_fp_*; never used "
                             "in person-image metrics",
        "counts": {"images": len(manifest), "representatives": len(reps),
                   "person_representatives": sum(r["image_type"] == "person" for r in reps),
                   "background_representatives": sum(r["image_type"] == "background" for r in reps),
                   "expected": EXPECTED},
        "visual_audit_of_method_B": VISUAL_AUDIT,
        "audit_flagged_merges_absent_at_T": [f[0] for f in FLAGGED_MERGES],
        "thresholds_not_selected": {"3.0": "forms merges the visual audit judged questionable "
                                           "(turn, TURN-05/06/11)",
                                    "4.0": "forms the merges judged unsafe or questionable "
                                           "(TURN-03, STRO-06, DIM-04, DIM-10 and the others)"},
        "inputs": {"metadata_csv_sha256": metadata_hash, "all_images_aggregate_sha256": images_hash,
                   "aggregate_definition": "sha256 of lines 'filename:sha256' sorted by filename "
                                           "joined with newline"},
        "software": {"python": platform.python_version(), "opencv": cv2.__version__,
                     "numpy": np.__version__},
        "script": "scripts/build_dedup_manifest.py",
    }


def render(manifest, summary, methodology):
    """Exact bytes of the three output files (deterministic: no timestamps)."""
    def csv_bytes(fields, rows):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")
    script_hash = sha256_bytes(Path(__file__).read_bytes())
    methodology = dict(methodology, script_sha256=script_hash)
    return {"manifest.csv": csv_bytes(MANIFEST_FIELDS, manifest),
            "summary.csv": csv_bytes(SUMMARY_FIELDS, summary),
            "methodology.json": (json.dumps(methodology, indent=2, sort_keys=True) + "\n").encode()}


def check_invariants(manifest, metadata_rows, threshold=THRESHOLD, expect_counts=True):
    """Raise DedupError listing every violated rule; return a list of passed checks."""
    problems, passed = [], []

    def check(ok, text):
        (passed if ok else problems).append(text)
    names = [r["filename"] for r in manifest]
    meta = {r["filename"]: r for r in metadata_rows}
    check(sorted(names) == sorted(meta) and len(set(names)) == len(names),
          f"every one of the {len(meta)} source images appears exactly once")
    reps = [r for r in manifest if r["status"] == "representative"]
    if expect_counts:
        check(len(manifest) == EXPECTED["images"], "125 source images accounted for")
        check(len(reps) == EXPECTED["representatives"], "73 representatives")
        check(sum(r["image_type"] == "person" for r in reps) == EXPECTED["person_representatives"],
              "68 person representatives")
        check(sum(r["image_type"] == "background" for r in reps)
              == EXPECTED["background_representatives"], "5 background representatives")
    clusters = {}
    for r in manifest:
        clusters.setdefault(r["cluster_id"], []).append(r)
    check(all(len({(m["condition"], m["sub_condition"]) for m in c}) == 1
              and all((m["condition"], m["sub_condition"]) == (meta[m["filename"]]["condition"],
                                                               meta[m["filename"]]["sub_condition"])
                      for m in c) for c in clusters.values()),
          "no cluster mixes conditions or recorded sub-conditions")
    ok_rep = True
    for c in clusters.values():
        order = sorted(c, key=lambda m: (datetime.fromisoformat(m["capture_time"]), m["filename"]))
        reps_in = [m for m in c if m["status"] == "representative"]
        ok_rep &= len(reps_in) == 1 and reps_in[0] is order[0]
        ok_rep &= all(m["representative_filename"] == order[0]["filename"] for m in c)
    check(ok_rep, "each cluster has exactly one representative and it is the earliest image")
    ok_ex = True
    for r in manifest:
        rep = next((m for m in clusters[r["cluster_id"]] if m["status"] == "representative"), None)
        d = float(r["mad_to_representative"])
        if r["status"] == "excluded":
            ok_ex &= rep is not None and rep["filename"] == r["representative_filename"] \
                and 0.0 <= d <= threshold + 5e-5
        else:
            ok_ex &= d == 0.0
    check(ok_ex, "every excluded image has a valid representative and a MAD <= T")
    check(all(r["background_fp_full"] + r["person_metrics_full"] == 1 for r in manifest)
          and all(r["person_metrics_full"] == int(r["image_type"] == "person") for r in manifest)
          and all(r["person_metrics_dedup"] == int(r["image_type"] == "person"
                                                   and r["status"] == "representative")
                  for r in manifest),
          "background images are never in the person-image metric sets")
    cid = {r["filename"]: r["cluster_id"] for r in manifest}
    if all(f[1] in cid and f[2] in cid for f in FLAGGED_MERGES):
        gone = [f[0] for f in FLAGGED_MERGES if cid[f[1]] != cid[f[2]]]
        check(len(gone) == len(FLAGGED_MERGES),
              f"all {len(FLAGGED_MERGES)} audit-flagged merges are absent at T={threshold}")
    if problems:
        raise DedupError("; ".join(problems))
    return passed


def write_new(path, data):
    """Create a file; if it exists it must already be byte-identical (never overwritten)."""
    if os.path.islink(path):
        raise DedupError(f"{path} is a link; refusing")
    if os.path.exists(path):
        if Path(path).read_bytes() != data:
            raise DedupError(f"{path} already exists with different content; not overwriting")
        return "unchanged"
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return "created"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verify", action="store_true", help="check existing output files")
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--out", default=None, help="output folder (default dataset/deduplication)")
    args = parser.parse_args(argv)
    dataset = Path(args.dataset)
    out_dir = Path(args.out) if args.out else dataset / "deduplication"
    try:
        manifest, summary, methodology = build(dataset)
        passed = check_invariants(manifest, read_metadata(dataset))
        files = render(manifest, summary, methodology)
        if args.verify:
            for name, data in files.items():
                path = out_dir / name
                if not path.is_file() or path.read_bytes() != data:
                    raise DedupError(f"{path} is missing or differs from the re-derived output")
            print("verify: output files match a fresh derivation byte for byte")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, data in files.items():
                print(f"{write_new(out_dir / name, data):9s} {out_dir / name}")
        for text in passed:
            print("PASS", text)
        return 0
    except DedupError as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
