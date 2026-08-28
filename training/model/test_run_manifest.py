import json
import os
import tempfile
import unittest

from data import vod_group_split
from run_manifest import (
    build_run_manifest,
    ensure_dataset_snapshot,
    indices_from_manifest,
    load_manifest_dataset,
    sha256_file,
    write_run_manifest,
)


def rows():
    result = []
    for index, (vod_id, label) in enumerate(
        [("1", 1), ("1", 0), ("2", 1), ("2", 0), ("3", 1), ("3", 0)]
    ):
        result.append(
            {
                "example_id": f"example-{index}",
                "content_hash": f"content-{index}",
                "event_group_id": f"event-{index}",
                "base_sample_weight": 1.0,
                "streamer_name": "example",
                "vod_id": vod_id,
                "label": label,
            }
        )
    return result


def write_jsonl(path, values):
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        for value in values:
            file.write(json.dumps(value, sort_keys=True) + "\n")


class RunManifestTest(unittest.TestCase):
    def test_snapshot_is_content_addressed_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = os.path.join(directory, "dataset.jsonl")
            snapshots = os.path.join(directory, "snapshots")
            write_jsonl(dataset, rows())
            first_path, first_hash = ensure_dataset_snapshot(dataset, snapshots)
            second_path, second_hash = ensure_dataset_snapshot(dataset, snapshots)
            self.assertEqual(first_path, second_path)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(sha256_file(first_path), first_hash)

    def test_manifest_uses_exact_rows_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = os.path.join(directory, "dataset.jsonl")
            snapshots = os.path.join(directory, "snapshots")
            run_dir = os.path.join(directory, "run")
            values = rows()
            write_jsonl(dataset, values)
            snapshot, digest = ensure_dataset_snapshot(dataset, snapshots)
            train_idx = [0, 1, 2, 3]
            val_idx = [4, 5]
            manifest = build_run_manifest(
                values,
                train_idx,
                val_idx,
                snapshot,
                digest,
                "fixed test split",
                {"root_config": {"training": {}}},
                0,
            )
            manifest_path = write_run_manifest(run_dir, manifest)
            with open(manifest_path, "rb") as file:
                first_bytes = file.read()
            write_run_manifest(run_dir, manifest)
            with open(manifest_path, "rb") as file:
                self.assertEqual(file.read(), first_bytes)
            loaded = load_manifest_dataset(manifest)
            self.assertEqual(
                indices_from_manifest(loaded, manifest, "validation"),
                val_idx,
            )

            loaded[4]["content_hash"] = "changed"
            with self.assertRaisesRegex(ValueError, "row changed"):
                indices_from_manifest(loaded, manifest, "validation")

    def test_same_seed_has_fixed_vod_membership(self):
        values = rows()
        first_train, first_val = vod_group_split(values, val_frac=0.34, seed=7)
        second_train, second_val = vod_group_split(values, val_frac=0.34, seed=7)
        self.assertEqual(first_train.tolist(), second_train.tolist())
        self.assertEqual(first_val.tolist(), second_val.tolist())


if __name__ == "__main__":
    unittest.main()
