"""Immutable dataset snapshots and row-level manifests for saved training runs."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


RUN_MANIFEST_FILENAME = "run_manifest.json"
SNAPSHOT_SCHEMA_VERSION = 1
RUN_MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset_snapshot(
    dataset_path,
    snapshots_dir="data/processed/snapshots",
):
    """Copy a dataset to its content-addressed path and return path plus digest."""
    dataset_path = Path(dataset_path)
    digest = sha256_file(dataset_path)
    snapshots_dir = Path(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshots_dir / f"dataset-{digest}.jsonl"
    if snapshot_path.exists():
        if sha256_file(snapshot_path) != digest:
            raise ValueError(f"Existing dataset snapshot is corrupt: {snapshot_path}")
    else:
        temporary = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
        shutil.copyfile(dataset_path, temporary)
        os.replace(temporary, snapshot_path)
    return snapshot_path, digest


def validate_dataset_identity(rows):
    seen = set()
    for index, row in enumerate(rows):
        example_id = str(row.get("example_id") or "")
        content_hash = str(row.get("content_hash") or "")
        if not example_id or not content_hash:
            raise ValueError(
                f"Dataset row {index} is missing example_id/content_hash; "
                "run build_dataset.py before training."
            )
        if example_id in seen:
            raise ValueError(f"Duplicate example_id in dataset: {example_id}")
        if not row.get("event_group_id"):
            raise ValueError(f"Dataset row {index} is missing event_group_id.")
        if float(row.get("base_sample_weight", 0.0)) <= 0:
            raise ValueError(
                f"Dataset row {index} has invalid base_sample_weight."
            )
        seen.add(example_id)


def partition_rows(rows, indices):
    return [
        {
            "example_id": rows[int(index)]["example_id"],
            "content_hash": rows[int(index)]["content_hash"],
        }
        for index in indices
    ]


def partition_summary(rows, indices):
    selected = [rows[int(index)] for index in indices]
    return {
        "rows": len(selected),
        "positives": sum(int(row["label"]) == 1 for row in selected),
        "negatives": sum(int(row["label"]) == 0 for row in selected),
        "effective_positives": sum(
            float(row.get("base_sample_weight", 1.0))
            for row in selected
            if int(row["label"]) == 1
        ),
        "effective_negatives": sum(
            float(row.get("base_sample_weight", 1.0))
            for row in selected
            if int(row["label"]) == 0
        ),
        "event_groups": len(
            {str(row.get("event_group_id") or row["example_id"]) for row in selected}
        ),
    }


def _git_metadata():
    def run(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "working_tree_dirty": None if status is None else bool(status),
    }


def build_run_manifest(
    rows,
    train_idx,
    val_idx,
    snapshot_path,
    dataset_sha256,
    split_description,
    config,
    seed,
):
    validate_dataset_identity(rows)
    train_vods = sorted({str(rows[int(index)]["vod_id"]) for index in train_idx})
    val_vods = sorted({str(rows[int(index)]["vod_id"]) for index in val_idx})
    overlap = sorted(set(train_vods) & set(val_vods))
    if overlap:
        raise ValueError(f"Train/validation VOD overlap: {overlap[:10]}")
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "dataset": {
            "snapshot_path": os.path.normpath(str(snapshot_path)),
            "sha256": dataset_sha256,
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        },
        "split": {
            "description": split_description,
            "seed": int(seed),
            "train_vod_ids": train_vods,
            "validation_vod_ids": val_vods,
            "train_rows": partition_rows(rows, train_idx),
            "validation_rows": partition_rows(rows, val_idx),
            "train_summary": partition_summary(rows, train_idx),
            "validation_summary": partition_summary(rows, val_idx),
        },
        "configuration": config,
        "code": _git_metadata(),
    }


def write_run_manifest(run_dir, manifest):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_MANIFEST_FILENAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, ensure_ascii=False, sort_keys=True, indent=2)
        file.write("\n")
    os.replace(temporary, path)
    return path


def load_run_manifest(run_dir):
    path = Path(run_dir) / RUN_MANIFEST_FILENAME
    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run manifest schema: {manifest.get('schema_version')!r}"
        )
    return manifest


def load_manifest_dataset(manifest):
    snapshot_path = Path(manifest["dataset"]["snapshot_path"])
    expected_digest = manifest["dataset"]["sha256"]
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Saved dataset snapshot is missing: {snapshot_path}")
    actual_digest = sha256_file(snapshot_path)
    if actual_digest != expected_digest:
        raise ValueError(
            f"Saved dataset snapshot changed: expected {expected_digest}, "
            f"got {actual_digest}"
        )
    rows = []
    with open(snapshot_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    validate_dataset_identity(rows)
    return rows


def indices_from_manifest(rows, manifest, partition="validation"):
    expected_rows = manifest["split"][f"{partition}_rows"]
    by_id = {row["example_id"]: (index, row) for index, row in enumerate(rows)}
    indices = []
    for expected in expected_rows:
        match = by_id.get(expected["example_id"])
        if match is None:
            raise ValueError(
                f"Run manifest {partition} row is missing: {expected['example_id']}"
            )
        index, row = match
        if row["content_hash"] != expected["content_hash"]:
            raise ValueError(
                f"Run manifest {partition} row changed: {expected['example_id']}"
            )
        indices.append(index)
    return indices
