import json
import os
import tempfile
import unittest

from fetch_negatives import existing_negative_offsets, sample_negative_offsets
from window_geometry import (
    WINDOW_GEOMETRY_NAME,
    WINDOW_GEOMETRY_VERSION,
    has_current_geometry,
    window_bounds,
)


class WindowGeometryTest(unittest.TestCase):
    def test_standard_clip_start_window(self):
        self.assertEqual(window_bounds(100), (95.0, 130.0))

    def test_near_vod_start_keeps_conceptual_pre_window(self):
        self.assertEqual(window_bounds(2), (-3.0, 32.0))

    def test_current_geometry_requires_version_and_bounds(self):
        record = {
            "target_offset": 100,
            "window_start": 95,
            "window_end": 130,
            "window_geometry": WINDOW_GEOMETRY_NAME,
            "window_geometry_version": WINDOW_GEOMETRY_VERSION,
        }
        self.assertTrue(has_current_geometry(record))

        record["window_start"] = 70
        record["window_end"] = 105
        self.assertFalse(has_current_geometry(record))

    def test_old_geometry_without_version_is_stale(self):
        record = {
            "target_offset": 100,
            "window_start": 70,
            "window_end": 105,
        }
        self.assertFalse(has_current_geometry(record))

    def test_negative_sampling_leaves_space_for_new_window(self):
        offsets = sample_negative_offsets(
            vod_duration=100,
            clip_offsets=[],
            count=20,
        )
        self.assertEqual(len(offsets), 20)
        self.assertTrue(all(5 <= offset <= 70 for offset in offsets))

    def test_negative_inventory_separates_current_and_stale_files(self):
        with tempfile.TemporaryDirectory() as directory:
            current = {
                "target_offset": 100,
                "window_start": 95,
                "window_end": 130,
                "window_geometry": WINDOW_GEOMETRY_NAME,
                "window_geometry_version": WINDOW_GEOMETRY_VERSION,
            }
            stale = {
                "target_offset": 200,
                "window_start": 170,
                "window_end": 205,
            }
            for offset, record in ((100, current), (200, stale)):
                path = os.path.join(directory, f"123_{offset}.json")
                with open(path, "w", encoding="utf-8") as file:
                    json.dump(record, file)

            current_offsets, stale_offsets = existing_negative_offsets(
                directory,
                "123",
            )
            self.assertEqual(current_offsets, {100})
            self.assertEqual(stale_offsets, {200})


if __name__ == "__main__":
    unittest.main()
