"""Verify the committed Go preprocessing fixture against the Python pipeline.

Run from the repository root:
    python training/export/verify_go_fixture.py
"""

import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_dir in (
    REPO_ROOT / "training" / "collect",
    REPO_ROOT / "training" / "features",
):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from build_dataset import compute_features
from preprocessing import raw_feature_vector
from tokenizer import encode_window


FIXTURE_PATH = (
    REPO_ROOT / "clipper" / "preprocess" / "testdata" / "python_golden.json"
)


def main():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as file:
        expected = json.load(file)

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[SEP]": 2,
        "Hello": 3,
        "HELLO": 4,
        "world": 5,
    }
    messages = [
        {"offset_seconds": 1, "user": "a", "message": "Hello world"},
        {"offset_seconds": 6, "user": "b", "message": "HELLO"},
        {"offset_seconds": 31, "user": "a", "message": "hello"},
    ]
    record = {
        "window_start": 0,
        "window_end": 35,
        "messages": messages,
    }
    computed = compute_features(record)
    row = {
        **computed,
        "target_offset": 21600,
    }

    tokens = encode_window(
        [message["message"] for message in messages],
        vocab,
        max_seq_len=8,
    )
    raw_features = raw_feature_vector(row, stream_time_scale_seconds=43200)

    if tokens != expected["tokens"]:
        raise AssertionError(
            f"Python tokens differ from Go fixture: {tokens} != {expected['tokens']}"
        )
    np.testing.assert_allclose(
        np.asarray(raw_features, dtype=np.float32),
        np.asarray(expected["raw_features"], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    print(f"Python preprocessing matches {FIXTURE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
