"""Unit tests for the eye duplicate-suppression rule.

Run from the project folder:  python -m unittest discover -s tests -v
"""
import unittest

from src.detector import EyeDetector, DetectorError, suppress_duplicate_eyes

# Face of 200 x 200 px. Boxes are face-relative (x, y, w, h).
FW = FH = 200


class SuppressDuplicateEyesTests(unittest.TestCase):
    def test_empty_detections(self):
        self.assertEqual(suppress_duplicate_eyes([], FW, FH), [])

    def test_overlapping_duplicates_become_one(self):
        # Two boxes on the same left eye (centres ~30% across, ~35% down).
        boxes = [(40, 55, 40, 40), (45, 56, 44, 44)]
        self.assertEqual(len(suppress_duplicate_eyes(boxes, FW, FH)), 1)

    def test_smaller_box_is_preferred(self):
        tight, loose = (45, 60, 30, 30), (35, 45, 55, 55)  # tight is inside loose
        self.assertEqual(suppress_duplicate_eyes([loose, tight], FW, FH), [tight])
        self.assertEqual(suppress_duplicate_eyes([tight, loose], FW, FH), [tight])

    def test_genuine_small_eye_is_kept(self):
        # 24 px wide = 12% of the face; must NOT be rejected (no min-width rule).
        small = (48, 62, 24, 24)
        right = (120, 60, 40, 40)
        result = suppress_duplicate_eyes([small, right], FW, FH)
        self.assertEqual(sorted(result), sorted([small, right]))

    def test_left_and_right_eyes_are_separated(self):
        left, right = (40, 60, 40, 40), (120, 60, 40, 40)
        result = suppress_duplicate_eyes([right, left], FW, FH)
        self.assertEqual(sorted(result), [left, right])

    def test_non_overlapping_boxes_in_same_half_keep_closest_to_typical(self):
        # Two separate left-half boxes: typical eye centre is (30%, 33%) = (60, 66).
        near = (40, 46, 40, 40)   # centre (60, 66) -> exactly typical
        far = (80, 30, 24, 24)    # centre (92, 42) -> further away, no overlap
        self.assertEqual(suppress_duplicate_eyes([far, near], FW, FH), [near])

    def test_implausible_boxes_are_rejected(self):
        too_low = (40, 130, 40, 40)      # centre y = 75% (mouth area)
        too_wide = (10, 40, 85, 85)      # wider than 40% of the face
        centre = (80, 55, 40, 40)        # centre x = 50%, between the zones
        outside = (0, 55, 10, 30)        # centre x = 2.5%, at the face edge
        self.assertEqual(
            suppress_duplicate_eyes([too_low, too_wide, centre, outside], FW, FH), [])

    def test_at_most_two_outputs(self):
        boxes = [(40, 60, 40, 40), (45, 61, 38, 38), (100, 58, 30, 30),
                 (120, 60, 40, 40), (125, 62, 36, 36), (150, 55, 25, 25)]
        self.assertLessEqual(len(suppress_duplicate_eyes(boxes, FW, FH)), 2)


class EyeDetectorTests(unittest.TestCase):
    def test_missing_cascade_raises(self):
        with self.assertRaises(DetectorError):
            EyeDetector(r"C:\definitely\missing.xml")

    def test_production_configuration_unchanged(self):
        det = EyeDetector()
        self.assertEqual((det.scale_factor, det.min_neighbors, det.min_size,
                          det.search_fraction), (1.1, 5, (20, 20), 0.6))

    def test_blank_face_returns_no_eyes(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertEqual(EyeDetector().detect(frame, (200, 100, 200, 200)), [])


if __name__ == "__main__":
    unittest.main()
