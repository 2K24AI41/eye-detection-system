"""Tests for scripts/build_dedup_manifest.py using small synthetic datasets in temporary
folders. No real images, no webcam, no detector. The real dataset/ is never touched.

Run:  python -m unittest discover -s tests -v
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_dedup_manifest as dd  # noqa: E402


def gray_image(level):
    """A plain 640x480 colour image; its 64x48 gray version is exactly `level`."""
    return np.full((480, 640, 3), level, np.uint8)


def make_dataset(root, items):
    """items: (filename, condition, sub, gray_level, capture_time). Returns dataset dir."""
    dataset = Path(root) / "dataset"
    (dataset / "images").mkdir(parents=True)
    with open(dataset / "metadata.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "participant_id", "condition", "sub_condition",
                         "capture_time", "notes"])
        for name, cond, sub, level, when in items:
            ok, buf = cv2.imencode(".png", gray_image(level))
            (dataset / "images" / name).write_bytes(buf.tobytes())
            writer.writerow([name, "NONE" if cond == "bg" else "P01", cond, sub, when, ""])
    return dataset


def t(n):
    return f"2026-01-01T10:00:{n:02d}+05:00"


ITEMS = [
    ("tilt_P01_001.png", "tilt", "roll", 100, t(1)),
    ("tilt_P01_002.png", "tilt", "roll", 101, t(2)),
    ("tilt_P01_003.png", "tilt", "down", 100, t(3)),   # same pixels as 001 but other sub-condition
    ("tilt_P01_004.png", "tilt", "down", 101, t(4)),
    ("tilt_P01_005.png", "tilt", "down", 120, t(5)),
    ("bg_NONE_001.png", "bg", "background", 60, t(6)),
    ("bg_NONE_002.png", "bg", "background", 61, t(7)),
]


class ClusterRuleTests(unittest.TestCase):
    def small(self, levels):
        return {f"i{k}": np.full((48, 64), v, np.float32) for k, v in enumerate(levels)}

    def names(self, levels):
        return list(self.small(levels))

    def groups(self, levels, threshold=2.5):
        s = self.small(levels)
        return [[n for n, _ in c] for c in dd.cluster_group(list(s), s, threshold)]

    def test_compares_with_first_image_of_cluster_not_previous(self):
        # 100, 102 (d=2) join; 103 is 3 away from the FIRST image -> new cluster even though
        # it is only 1 away from 102; 105 is 2 away from 103 and joins that cluster.
        self.assertEqual(self.groups([100, 102, 103, 105]), [["i0", "i1"], ["i2", "i3"]])

    def test_boundary_equal_to_threshold_joins(self):
        self.assertEqual(self.groups([100, 102.5]), [["i0", "i1"]])
        self.assertEqual(self.groups([100, 102.6]), [["i0"], ["i1"]])

    def test_representative_is_first_and_mad_is_to_it(self):
        s = self.small([100, 101, 102])
        cluster = dd.cluster_group(list(s), s, 2.5)[0]
        self.assertEqual([n for n, _ in cluster], ["i0", "i1", "i2"])
        self.assertEqual([round(d, 3) for _, d in cluster], [0.0, 1.0, 2.0])

    def test_mad_function(self):
        a, b = np.full((48, 64), 10, np.float32), np.full((48, 64), 13, np.float32)
        self.assertEqual(dd.mad(a, b), 3.0)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dataset = make_dataset(self._tmp.name, ITEMS)
        self.manifest, self.summary, self.method = dd.build(self.dataset)
        self.by_name = {r["filename"]: r for r in self.manifest}

    def tearDown(self):
        self._tmp.cleanup()

    def test_sub_conditions_are_never_mixed(self):
        # 001 and 003 have identical pixels but different recorded sub-conditions.
        self.assertNotEqual(self.by_name["tilt_P01_001.png"]["cluster_id"],
                            self.by_name["tilt_P01_003.png"]["cluster_id"])
        self.assertEqual(self.by_name["tilt_P01_002.png"]["representative_filename"],
                         "tilt_P01_001.png")
        self.assertEqual(self.by_name["tilt_P01_004.png"]["representative_filename"],
                         "tilt_P01_003.png")

    def test_threshold_splits_and_status(self):
        self.assertEqual(self.by_name["tilt_P01_005.png"]["status"], "representative")  # 20 away
        self.assertEqual(self.by_name["tilt_P01_002.png"]["status"], "excluded")
        self.assertEqual(self.by_name["tilt_P01_002.png"]["mad_to_representative"], "1.0000")
        self.assertEqual(sum(r["status"] == "representative" for r in self.manifest), 4)

    def test_cluster_ids_and_sizes(self):
        self.assertEqual(self.by_name["tilt_P01_001.png"]["cluster_id"], "tilt-roll-C01")
        self.assertEqual(self.by_name["tilt_P01_005.png"]["cluster_id"], "tilt-down-C02")
        self.assertEqual(self.by_name["tilt_P01_001.png"]["cluster_size"], 2)

    def test_background_flags(self):
        for name in ("bg_NONE_001.png", "bg_NONE_002.png"):
            r = self.by_name[name]
            self.assertEqual((r["image_type"], r["person_metrics_full"],
                              r["person_metrics_dedup"], r["background_fp_full"]),
                             ("background", 0, 0, 1))
        self.assertEqual(self.by_name["bg_NONE_001.png"]["background_fp_dedup"], 1)
        self.assertEqual(self.by_name["bg_NONE_002.png"]["background_fp_dedup"], 0)
        self.assertEqual(self.by_name["tilt_P01_002.png"]["person_metrics_dedup"], 0)
        self.assertEqual(self.by_name["tilt_P01_001.png"]["person_metrics_dedup"], 1)

    def test_invariants_pass(self):
        rows = dd.read_metadata(self.dataset)
        passed = dd.check_invariants(self.manifest, rows, expect_counts=False)
        # count-specific and audit-flagged-merge checks do not apply to this toy dataset
        self.assertEqual(len(passed), 5)

    def test_invariants_catch_wrong_representative(self):
        rows = dd.read_metadata(self.dataset)
        bad = [dict(r) for r in self.manifest]
        for r in bad:                                    # make the later image the representative
            if r["filename"] == "tilt_P01_001.png":
                r["status"] = "excluded"
            if r["filename"] == "tilt_P01_002.png":
                r["status"] = "representative"
        with self.assertRaises(dd.DedupError):
            dd.check_invariants(bad, rows, expect_counts=False)

    def test_invariants_catch_mixed_clusters_and_missing_image(self):
        rows = dd.read_metadata(self.dataset)
        mixed = [dict(r) for r in self.manifest]
        mixed[2]["cluster_id"] = mixed[0]["cluster_id"]  # tilt/down image into tilt/roll cluster
        with self.assertRaises(dd.DedupError):
            dd.check_invariants(mixed, rows, expect_counts=False)
        with self.assertRaises(dd.DedupError):
            dd.check_invariants(self.manifest[:-1], rows, expect_counts=False)

    def test_invariants_catch_bad_mad(self):
        rows = dd.read_metadata(self.dataset)
        bad = [dict(r) for r in self.manifest]
        for r in bad:
            if r["filename"] == "tilt_P01_002.png":
                r["mad_to_representative"] = "9.0000"
        with self.assertRaises(dd.DedupError):
            dd.check_invariants(bad, rows, expect_counts=False)

    def test_summary_counts(self):
        overall = self.summary[0]
        self.assertEqual((overall["images"], overall["clusters_representatives"],
                          overall["excluded"]), (7, 4, 3))
        person = next(r for r in self.summary if r["scope"] == "person")
        self.assertEqual((person["images"], person["background_images"]), (5, 0))

    def test_output_is_deterministic(self):
        first = dd.render(self.manifest, self.summary, self.method)
        again = dd.build(self.dataset)
        self.assertEqual(first, dd.render(*again))

    def test_methodology_records_rule(self):
        self.assertEqual(self.method["threshold_T_gray_levels"], 2.5)
        self.assertEqual(self.method["preprocessing"]["resize_width_height"], [64, 48])
        self.assertEqual(self.method["preprocessing"]["interpolation"], "cv2.INTER_AREA")
        self.assertIn("no detector", self.method["inputs_used"])


class SafetyTests(unittest.TestCase):
    def test_capture_order_must_match_filename_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = [("dim_P01_001.png", "dim", "dim", 100, t(5)),
                     ("dim_P01_002.png", "dim", "dim", 100, t(1))]   # numbered 2nd but earlier
            dataset = make_dataset(tmp, items)
            with self.assertRaises(dd.DedupError):
                dd.build(dataset)

    def test_unknown_condition_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = make_dataset(tmp, [("zzz_P01_001.png", "zzz", "x", 100, t(1))])
            with self.assertRaises(dd.DedupError):
                dd.build(dataset)

    def test_wrong_image_size_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = make_dataset(tmp, [("dim_P01_001.png", "dim", "dim", 100, t(1))])
            ok, buf = cv2.imencode(".png", np.zeros((240, 320, 3), np.uint8))
            (dataset / "images" / "dim_P01_001.png").write_bytes(buf.tobytes())
            with self.assertRaises(dd.DedupError):
                dd.build(dataset)

    def test_write_never_overwrites_different_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            self.assertEqual(dd.write_new(path, b"abc"), "created")
            self.assertEqual(dd.write_new(path, b"abc"), "unchanged")
            with self.assertRaises(dd.DedupError):
                dd.write_new(path, b"different")
            self.assertEqual(path.read_bytes(), b"abc")

    def test_main_creates_and_verifies_in_temp_folder_without_touching_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = make_dataset(tmp, ITEMS)
            before = {p.name: p.read_bytes() for p in (dataset / "images").iterdir()}
            meta_before = (dataset / "metadata.csv").read_bytes()
            out = Path(tmp) / "out"
            # expected real-dataset counts do not apply to the toy data, so call the pieces
            manifest, summary, method = dd.build(dataset)
            files = dd.render(manifest, summary, method)
            out.mkdir()
            for name, data in files.items():
                dd.write_new(out / name, data)
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["manifest.csv", "methodology.json", "summary.csv"])
            self.assertEqual({p.name: p.read_bytes() for p in (dataset / "images").iterdir()},
                             before)
            self.assertEqual((dataset / "metadata.csv").read_bytes(), meta_before)

    def test_script_never_uses_a_detector(self):
        source = (Path(dd.__file__)).read_text(encoding="utf-8")
        self.assertNotIn("src.detector", source)
        self.assertNotIn("CascadeClassifier", source)
        self.assertNotIn("detectMultiScale", source)


if __name__ == "__main__":
    unittest.main()
