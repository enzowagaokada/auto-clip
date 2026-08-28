"""Create the fixed validation-VOD manifest used by remediation experiments."""

import argparse
import os

from data import DATASET_FILE, load_dataset_rows, vod_group_split
from run_manifest import (
    ensure_dataset_snapshot,
    sha256_file,
    validate_dataset_identity,
)


DEFAULT_OUTPUT = "data/splits/remediation_validation_vods.txt"


def vod_sort_key(vod_id):
    try:
        return 0, int(vod_id)
    except ValueError:
        return 1, vod_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise FileExistsError(
            f"Fixed remediation manifest already exists: {args.output}. "
            "Do not replace it during an experiment series."
        )

    rows = load_dataset_rows(args.dataset)
    validate_dataset_identity(rows)
    snapshot_path, dataset_digest = ensure_dataset_snapshot(args.dataset)
    rows = load_dataset_rows(str(snapshot_path))
    _, validation = vod_group_split(
        rows,
        val_frac=args.validation_fraction,
        seed=args.seed,
    )
    validation_vods = sorted(
        {str(rows[int(index)]["vod_id"]) for index in validation},
        key=vod_sort_key,
    )
    if not validation_vods:
        raise ValueError("No validation VODs were selected.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    temporary = args.output + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as file:
        file.write("# Fixed remediation validation VODs; do not regenerate mid-series.\n")
        file.write(f"# dataset_snapshot={os.path.normpath(str(snapshot_path))}\n")
        file.write(f"# dataset_sha256={dataset_digest}\n")
        file.write(f"# validation_fraction={args.validation_fraction}\n")
        file.write(f"# seed={args.seed}\n")
        file.write(f"# count={len(validation_vods)}\n")
        for vod_id in validation_vods:
            file.write(f"{vod_id}\n")
    os.replace(temporary, args.output)
    print(
        f"Wrote {len(validation_vods)} fixed validation VODs -> {args.output} "
        f"(dataset {sha256_file(snapshot_path)})"
    )


if __name__ == "__main__":
    main()
