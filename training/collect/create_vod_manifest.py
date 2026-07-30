"""Snapshot processed VOD IDs or create a manifest containing only new VODs.

Before collecting:
    python training/collect/create_vod_manifest.py \
        --output data/splits/vods_before_collection.txt

After collecting and rebuilding:
    python training/collect/create_vod_manifest.py \
        --exclude data/splits/vods_before_collection.txt \
        --output data/splits/untouched_vods.txt
"""

import argparse
import json
import os


DATASET_FILE = "data/processed/dataset.jsonl"


def dataset_vod_ids(path):
    vod_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            vod_id = str(row.get("vod_id", "")).strip()
            if vod_id and vod_id not in {"None", "unknown"}:
                vod_ids.add(vod_id)
    return vod_ids


def load_manifest(path):
    vod_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.partition("#")[0].strip()
            if value:
                vod_ids.add(value)
    return vod_ids


def vod_sort_key(vod_id):
    try:
        return 0, int(vod_id)
    except ValueError:
        return 1, vod_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_FILE)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exclude",
        default=None,
        help="Write only VOD IDs not present in this earlier snapshot.",
    )
    args = parser.parse_args()

    current_vods = dataset_vod_ids(args.dataset)
    if not current_vods:
        raise ValueError(f"No VOD IDs found in {args.dataset}")

    if args.exclude:
        baseline_vods = load_manifest(args.exclude)
        selected_vods = current_vods - baseline_vods
        description = (
            f"VODs added after baseline {os.path.normpath(args.exclude)}"
        )
        if not selected_vods:
            raise ValueError(
                "No new VOD IDs were found. Collect new streams, rebuild the "
                "dataset, and run this command again."
            )
    else:
        selected_vods = current_vods
        description = f"Baseline snapshot from {os.path.normpath(args.dataset)}"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    temporary_output = args.output + ".tmp"
    with open(temporary_output, "w", encoding="utf-8") as f:
        f.write(f"# {description}\n")
        f.write(f"# count={len(selected_vods)}\n")
        for vod_id in sorted(selected_vods, key=vod_sort_key):
            f.write(f"{vod_id}\n")
    os.replace(temporary_output, args.output)

    print(f"Wrote {len(selected_vods)} VOD IDs -> {args.output}")
    if args.exclude:
        print(
            f"Current={len(current_vods)} "
            f"baseline={len(baseline_vods)} "
            f"new={len(selected_vods)}"
        )


if __name__ == "__main__":
    main()
