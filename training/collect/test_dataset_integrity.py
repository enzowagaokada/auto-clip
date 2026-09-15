import os
import unittest

from build_dataset import (
    apply_review_annotations,
    assign_event_groups,
    content_hash,
    deduplicate_examples,
    example_id,
    exclude_clip_collisions,
    require_annotation_partition,
    require_training_partition,
)


def make_example(target, label=0, source="sampled_negative", path=None):
    return {
        "label": label,
        "streamer_name": "example",
        "vod_id": "123",
        "target_offset": target,
        "source": source,
        "_source_path": path or f"{source}-{target}.json",
        "message_count": 1,
        "messages_per_second": 1.0 / 35.0,
        "unique_users": 1,
        "message_rate_buckets": [0.2, 0, 0, 0, 0, 0, 0],
        "message_rate_change": -0.1,
        "peak_5s_rate": 0.2,
        "repeat_message_ratio": 0.0,
        "window_start": target - 5,
        "window_end": target + 30,
        "window_geometry": "clip_start_minus_5_plus_30",
        "window_geometry_version": 2,
        "messages": ["hello"],
    }


class DatasetIntegrityTest(unittest.TestCase):
    def test_locked_confirmation_window_is_rejected_from_training(self):
        require_training_partition(
            {"review_partition": "calibration"},
            "calibration.json",
        )
        with self.assertRaisesRegex(ValueError, "locked confirmation"):
            require_training_partition(
                {"review_partition": "confirmation"},
                "confirmation.json",
            )
        with self.assertRaisesRegex(ValueError, "Locked confirmation review"):
            require_annotation_partition({
                "streamer_name": "example",
                "vod_id": "123",
                "target_offset": "100",
                "review_partition": "confirmation",
                "review_identity": "confirmation:episode:e",
            })

    def test_review_resolves_conflict_before_deduplication(self):
        positive = make_example(100, label=1, source="historical_positive")
        negative = make_example(100, label=0)
        annotation = {
            ("example", "123", 100): {
                "review_label": "positive",
                "training_label": "1",
                "review_notes": "confirmed",
            }
        }
        reviewed, _ = apply_review_annotations([positive, negative], annotation)
        rows, duplicate_keys, duplicate_rows = deduplicate_examples(reviewed)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], 1)
        self.assertEqual(rows[0]["review_label"], "positive")
        self.assertEqual((duplicate_keys, duplicate_rows), (1, 1))

    def test_unreviewed_label_conflict_fails(self):
        with self.assertRaisesRegex(ValueError, "unresolved label conflict"):
            deduplicate_examples(
                [
                    make_example(100, label=1, source="historical_positive"),
                    make_example(100, label=0, source="live_review"),
                ]
            )

    def test_sampled_negative_collision_is_removed_before_deduplication(self):
        positive = make_example(100, label=1, source="historical_positive")
        negative = make_example(100, label=0)
        kept, excluded, _ = exclude_clip_collisions(
            [positive, negative],
            {"123": [100]},
        )
        rows, _, _ = deduplicate_examples(kept)
        self.assertEqual(rows, [positive])
        self.assertEqual(excluded, [negative])

    def test_clip_collision_boundary_and_review_override(self):
        inside = make_example(159)
        boundary = make_example(160)
        reviewed = make_example(159)
        reviewed["review_label"] = "hard_negative"
        kept, excluded, overridden = exclude_clip_collisions(
            [inside, boundary, reviewed],
            {"123": [100]},
        )
        self.assertEqual(kept, [boundary, reviewed])
        self.assertEqual(excluded, [inside])
        self.assertEqual(overridden, [reviewed])

    def test_event_groups_are_fixed_anchor_not_transitive(self):
        rows = [make_example(target) for target in (0, 34, 68)]
        for row in rows:
            row["example_id"] = example_id(row)
        group_count, cross_group_overlaps = assign_event_groups(rows)
        self.assertEqual(group_count, 2)
        self.assertEqual(cross_group_overlaps, 1)
        self.assertEqual(rows[0]["event_group_id"], rows[1]["event_group_id"])
        self.assertNotEqual(rows[1]["event_group_id"], rows[2]["event_group_id"])
        self.assertAlmostEqual(rows[0]["base_sample_weight"], 0.5)
        self.assertAlmostEqual(rows[1]["base_sample_weight"], 0.5)
        self.assertAlmostEqual(rows[2]["base_sample_weight"], 1.0)

    def test_identity_and_content_hash_are_deterministic(self):
        first = make_example(100)
        second = make_example(100, path=os.path.join("different", "path.json"))
        first["example_id"] = example_id(first)
        second["example_id"] = example_id(second)
        self.assertEqual(first["example_id"], second["example_id"])
        self.assertEqual(content_hash(first), content_hash(second))


if __name__ == "__main__":
    unittest.main()
