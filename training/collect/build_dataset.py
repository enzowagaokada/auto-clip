import csv
import hashlib
import json
import os
from collections import defaultdict

import pandas as pd

from window_geometry import (
    WINDOW_GEOMETRY_NAME,
    WINDOW_GEOMETRY_VERSION,
    has_current_geometry,
)


CLIPS_FILE = "data/raw/clips.csv"
POSITIVE_DIR = "data/raw/chat"
NEGATIVE_DIR = "data/raw/chat_negatives"
LIVE_DIR = "data/raw/chat_live"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.jsonl")
AUDIT_FILE = os.path.join(OUTPUT_DIR, "dataset_audit.json")
REVIEW_LABELS_FILE = "data/reviews/window_labels.csv"
TEMPORAL_BUCKET_SECONDS = 5
TEMPORAL_BUCKET_COUNT = 7
CLIP_EXCLUSION_SECONDS = 60
EVENT_GROUP_SECONDS = 35
SOURCE_PRIORITY = {
    "historical_positive": 0,
    "sampled_negative": 1,
    "live_review": 2,
}


def load_clip_metadata():
    """Return clip streamer lookup and known clip anchors grouped by VOD."""
    df = pd.read_csv(CLIPS_FILE)
    streamers = dict(zip(df["clip_id"].astype(str), df["streamer_name"].astype(str)))
    offsets_by_vod = defaultdict(list)
    for row in df.to_dict("records"):
        offsets_by_vod[str(row["vod_id"])].append(int(float(row["vod_offset"])))
    return streamers, {
        vod_id: sorted(set(offsets))
        for vod_id, offsets in offsets_by_vod.items()
    }


def example_key(record):
    return (
        str(record.get("streamer_name", "unknown")).strip().lower(),
        str(record.get("vod_id")),
        int(float(record.get("target_offset") or 0)),
    )


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def example_id(example):
    return stable_hash(
        {
            "streamer_name": example_key(example)[0],
            "vod_id": example_key(example)[1],
            "target_offset": example_key(example)[2],
            "window_geometry_version": int(example["window_geometry_version"]),
        }
    )


def content_hash(example):
    return stable_hash(
        {
            "example_id": example["example_id"],
            "label": int(example["label"]),
            "messages": example["messages"],
            "message_count": int(example["message_count"]),
            "messages_per_second": float(example["messages_per_second"]),
            "unique_users": int(example["unique_users"]),
            "message_rate_buckets": example["message_rate_buckets"],
            "message_rate_change": float(example["message_rate_change"]),
            "peak_5s_rate": float(example["peak_5s_rate"]),
            "repeat_message_ratio": float(example["repeat_message_ratio"]),
            "window_start": float(example["window_start"]),
            "window_end": float(example["window_end"]),
        }
    )


def load_review_annotations():
    if not os.path.exists(REVIEW_LABELS_FILE):
        return {}
    with open(REVIEW_LABELS_FILE, "r", encoding="utf-8", newline="") as f:
        annotations = {}
        for row in csv.DictReader(f):
            require_annotation_partition(row)
            annotations[example_key(row)] = row
        return annotations


def require_annotation_partition(row):
    if row.get("review_partition") == "confirmation":
        raise ValueError(
            "Locked confirmation review entered durable training "
            f"annotations: {row.get('review_identity') or example_key(row)}"
        )


def apply_review_annotations(examples, annotations):
    """Override reviewed labels and exclude genuinely uncertain reviewed rows."""
    reviewed_examples = []
    counts = {"positive": 0, "hard_negative": 0, "uncertain": 0}
    for example in examples:
        annotation = annotations.get(example_key(example))
        if annotation is None:
            reviewed_examples.append(example)
            continue

        review_label = annotation["review_label"]
        counts[review_label] = counts.get(review_label, 0) + 1
        if review_label == "uncertain":
            continue

        reviewed = dict(example)
        reviewed["label"] = int(annotation["training_label"])
        reviewed["review_label"] = review_label
        reviewed["review_notes"] = annotation.get("review_notes", "")
        if annotation.get("review_partition"):
            reviewed["review_partition"] = annotation["review_partition"]
        if annotation.get("review_identity"):
            reviewed["review_identity"] = annotation["review_identity"]
        reviewed_examples.append(reviewed)
    return reviewed_examples, counts


def is_clip_collision(example, clip_offsets_by_vod):
    if example["source"] != "sampled_negative":
        return False
    target = example_key(example)[2]
    return any(
        abs(target - clip_offset) < CLIP_EXCLUSION_SECONDS
        for clip_offset in clip_offsets_by_vod.get(str(example["vod_id"]), [])
    )


def exclude_clip_collisions(examples, clip_offsets_by_vod):
    kept = []
    excluded = []
    overridden = []
    for example in examples:
        if not is_clip_collision(example, clip_offsets_by_vod):
            kept.append(example)
        elif example.get("review_label") == "hard_negative":
            kept.append(example)
            overridden.append(example)
        else:
            excluded.append(example)
    return kept, excluded, overridden


def deduplicate_examples(examples):
    """Resolve exact keys deterministically and fail on unresolved label conflicts."""
    grouped = defaultdict(list)
    for example in examples:
        grouped[example_key(example)].append(example)

    deduplicated = []
    duplicate_rows = 0
    duplicate_keys = 0
    conflicts = []
    for key in sorted(grouped):
        candidates = grouped[key]
        if len(candidates) > 1:
            duplicate_keys += 1
            duplicate_rows += len(candidates) - 1
        labels = {int(candidate["label"]) for candidate in candidates}
        if len(labels) > 1:
            conflicts.append(
                {
                    "key": key,
                    "sources": [
                        {
                            "path": candidate["_source_path"],
                            "source": candidate["source"],
                            "label": int(candidate["label"]),
                        }
                        for candidate in candidates
                    ],
                }
            )
            continue
        selected = min(
            candidates,
            key=lambda candidate: (
                SOURCE_PRIORITY[candidate["source"]],
                candidate["_source_path"],
                stable_hash(candidate["messages"]),
            ),
        )
        deduplicated.append(selected)

    if conflicts:
        details = []
        for conflict in conflicts:
            sources = ", ".join(
                f"{item['path']} ({item['source']}, label={item['label']})"
                for item in conflict["sources"]
            )
            details.append(f"{conflict['key']}: {sources}")
        raise ValueError(
            f"{len(conflicts)} unresolved label conflict(s) remain after reviews:\n"
            + "\n".join(details)
        )
    return deduplicated, duplicate_keys, duplicate_rows


def assign_event_groups(examples):
    """Assign bounded, fixed-anchor event groups and normalized base weights."""
    by_stream = defaultdict(list)
    for example in examples:
        by_stream[(example_key(example)[0], str(example["vod_id"]))].append(example)

    for (streamer, vod_id), group_rows in sorted(by_stream.items()):
        ordered = sorted(
            group_rows,
            key=lambda row: (example_key(row)[2], row["example_id"]),
        )
        group_anchor = None
        for row in ordered:
            target = example_key(row)[2]
            if group_anchor is None or target - group_anchor >= EVENT_GROUP_SECONDS:
                group_anchor = target
            row["event_group_id"] = stable_hash(
                {
                    "streamer_name": streamer,
                    "vod_id": vod_id,
                    "anchor_offset": group_anchor,
                    "window_geometry_version": int(row["window_geometry_version"]),
                }
            )

    cross_group_overlaps = 0
    for group_rows in by_stream.values():
        ordered = sorted(group_rows, key=lambda row: example_key(row)[2])
        for left_index, left in enumerate(ordered):
            left_target = example_key(left)[2]
            for right in ordered[left_index + 1:]:
                difference = example_key(right)[2] - left_target
                if difference >= EVENT_GROUP_SECONDS:
                    break
                if right["event_group_id"] != left["event_group_id"]:
                    cross_group_overlaps += 1

    group_sizes = defaultdict(int)
    for example in examples:
        group_sizes[example["event_group_id"]] += 1
    for example in examples:
        example["base_sample_weight"] = 1.0 / group_sizes[example["event_group_id"]]
    return len(group_sizes), cross_group_overlaps


def compute_features(record):
    """Compute window-level features shared by positive and negative examples."""
    messages = record.get("messages", [])
    message_count = len(messages)

    window_start = record.get("window_start", 0)
    window_end = record.get("window_end", 0)
    duration = max(1, window_end - window_start)

    unique_users = len({m.get("user") for m in messages if m.get("user")})
    bucket_counts = [0] * TEMPORAL_BUCKET_COUNT
    normalized_messages = []
    for message in messages:
        offset = float(message.get("offset_seconds", window_start))
        relative_offset = max(0.0, offset - window_start)
        bucket_index = min(
            int(relative_offset // TEMPORAL_BUCKET_SECONDS),
            TEMPORAL_BUCKET_COUNT - 1,
        )
        bucket_counts[bucket_index] += 1
        text = message.get("message", "").strip().casefold()
        if text:
            normalized_messages.append(text)

    bucket_rates = [
        round(count / TEMPORAL_BUCKET_SECONDS, 4) for count in bucket_counts
    ]
    early_rate = sum(bucket_rates[:2]) / 2
    recent_rate = sum(bucket_rates[-2:]) / 2
    repeat_ratio = (
        1.0 - len(set(normalized_messages)) / len(normalized_messages)
        if normalized_messages
        else 0.0
    )

    return {
        "message_count": message_count,
        "messages_per_second": round(message_count / duration, 4),
        "unique_users": unique_users,
        "message_rate_buckets": bucket_rates,
        "message_rate_change": round(recent_rate - early_rate, 4),
        "peak_5s_rate": max(bucket_rates, default=0.0),
        "repeat_message_ratio": round(repeat_ratio, 4),
        "window_start": window_start,
        "window_end": window_end,
    }


def build_example(record, label, streamer_name, source, source_path):
    """Assemble one dataset row from a raw chat window."""
    features = compute_features(record)

    example = {
        "label": label,
        "streamer_name": streamer_name,
        "vod_id": str(record.get("vod_id")),
        "target_offset": int(float(record.get("target_offset") or 0)),
        "source": source,
        "_source_path": os.path.normpath(source_path),
        "message_count": features["message_count"],
        "messages_per_second": features["messages_per_second"],
        "unique_users": features["unique_users"],
        "message_rate_buckets": features["message_rate_buckets"],
        "message_rate_change": features["message_rate_change"],
        "peak_5s_rate": features["peak_5s_rate"],
        "repeat_message_ratio": features["repeat_message_ratio"],
        "window_start": features["window_start"],
        "window_end": features["window_end"],
        "window_geometry": WINDOW_GEOMETRY_NAME,
        "window_geometry_version": WINDOW_GEOMETRY_VERSION,
        "messages": [m.get("message", "") for m in record.get("messages", [])],
    }
    for name in (
        "review_partition",
        "review_identity",
        "episode_id",
        "record_type",
        "peak_target_at",
    ):
        if record.get(name) is not None:
            example[name] = record[name]
    return example


def iter_json_files(directory):
    if not os.path.isdir(directory):
        return
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            yield os.path.join(directory, name)


def require_current_geometry(record, path):
    if not has_current_geometry(record):
        raise ValueError(
            f"{path} uses stale or invalid window geometry. "
            "Rebuild from current-geometry sources before continuing."
        )


def require_training_partition(record, path):
    if record.get("review_partition") not in (None, "", "calibration"):
        raise ValueError(
            f"{path} belongs to locked confirmation partition and must "
            "never enter a training dataset"
        )


def main():
    if not os.path.exists(CLIPS_FILE):
        print(f"Error: {CLIPS_FILE} not found. Run fetch_clips.py first.")
        return

    clip_streamers, clip_offsets_by_vod = load_clip_metadata()
    review_annotations = load_review_annotations()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    examples = []
    skipped_empty = 0
    missing_streamer = 0
    processed_files = 0

    # Positives (label = 1). These lack label/streamer_name, so we join on clip_id.
    for path in iter_json_files(POSITIVE_DIR):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        require_current_geometry(record, path)
        if not record.get("messages"):
            skipped_empty += 1
            continue

        clip_id = str(record.get("clip_id"))
        streamer_name = clip_streamers.get(clip_id)
        if streamer_name is None:
            missing_streamer += 1
            streamer_name = "unknown"

        examples.append(
            build_example(
                record,
                label=1,
                streamer_name=streamer_name,
                source="historical_positive",
                source_path=path,
            )
        )
        processed_files += 1
        if processed_files % 500 == 0:
            print(f"Processed {processed_files} non-empty chat files...")

    # Negatives (label = 0). These already carry label and streamer_name.
    for path in iter_json_files(NEGATIVE_DIR):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        require_current_geometry(record, path)
        if not record.get("messages"):
            skipped_empty += 1
            continue

        streamer_name = record.get("streamer_name", "unknown")
        examples.append(
            build_example(
                record,
                label=0,
                streamer_name=streamer_name,
                source="sampled_negative",
                source_path=path,
            )
        )
        processed_files += 1
        if processed_files % 500 == 0:
            print(f"Processed {processed_files} non-empty chat files...")

    live_added = 0
    for path in iter_json_files(LIVE_DIR):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        require_current_geometry(record, path)
        require_training_partition(record, path)
        if not record.get("messages"):
            skipped_empty += 1
            continue

        streamer_name = record.get("streamer_name", "unknown")
        label = int(record.get("label", 0))
        examples.append(
            build_example(
                record,
                label=label,
                streamer_name=streamer_name,
                source="live_review",
                source_path=path,
            )
        )
        live_added += 1
        processed_files += 1
        if processed_files % 500 == 0:
            print(f"Processed {processed_files} non-empty chat files...")

    raw_examples = len(examples)
    examples, review_counts = apply_review_annotations(
        examples,
        review_annotations,
    )
    examples, proximity_excluded, proximity_overridden = exclude_clip_collisions(
        examples,
        clip_offsets_by_vod,
    )
    examples, duplicate_keys, duplicate_rows = deduplicate_examples(examples)

    for example in examples:
        example["example_id"] = example_id(example)
    event_groups, cross_group_overlaps = assign_event_groups(examples)
    for example in examples:
        example["content_hash"] = content_hash(example)
        example.pop("_source_path", None)
    examples.sort(key=lambda example: example["example_id"])

    temporary_output = OUTPUT_FILE + ".tmp"
    with open(temporary_output, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    os.replace(temporary_output, OUTPUT_FILE)

    positives = sum(1 for e in examples if e["label"] == 1)
    negatives = sum(1 for e in examples if e["label"] == 0)
    effective_positives = sum(
        e["base_sample_weight"] for e in examples if e["label"] == 1
    )
    effective_negatives = sum(
        e["base_sample_weight"] for e in examples if e["label"] == 0
    )
    audit = {
        "dataset_sha256": file_hash(OUTPUT_FILE),
        "raw_json_files": processed_files + skipped_empty,
        "raw_non_empty_examples": raw_examples,
        "skipped_empty": skipped_empty,
        "output_examples": len(examples),
        "positives": positives,
        "negatives": negatives,
        "effective_positives": effective_positives,
        "effective_negatives": effective_negatives,
        "review_applications": review_counts,
        "missing_positive_streamer": missing_streamer,
        "live_windows_read": live_added,
        "duplicate_keys": duplicate_keys,
        "duplicate_rows_removed": duplicate_rows,
        "unresolved_conflicts": 0,
        "negative_clip_collisions_excluded": len(proximity_excluded),
        "negative_clip_collisions_review_overridden": len(proximity_overridden),
        "event_groups": event_groups,
        "cross_group_direct_overlaps": cross_group_overlaps,
    }
    temporary_audit = AUDIT_FILE + ".tmp"
    with open(temporary_audit, "w", encoding="utf-8", newline="\n") as f:
        json.dump(audit, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    os.replace(temporary_audit, AUDIT_FILE)

    print("\n--- Summary ---")
    print(f"Raw JSON files: {processed_files + skipped_empty}")
    print(f"Raw non-empty examples: {raw_examples}")
    print(f"Total examples: {len(examples)}")
    print(f"Positives (label=1): {positives}")
    print(f"Negatives (label=0): {negatives}")
    print(
        "Effective event-weighted counts: "
        f"positive={effective_positives:.3f} negative={effective_negatives:.3f}"
    )
    if positives:
        print(f"Negative:positive ratio: {negatives / positives:.2f}:1")
    print(f"Skipped (no messages): {skipped_empty}")
    if review_annotations:
        print(
            "Applied reviews: "
            f"{review_counts['positive']} positive, "
            f"{review_counts['hard_negative']} hard negative, "
            f"{review_counts['uncertain']} uncertain excluded"
        )
    if missing_streamer:
        print(f"Positives with no streamer match in clips.csv: {missing_streamer}")
    print(f"Live windows read: {live_added}")
    print(
        f"Exact duplicates removed: {duplicate_rows} rows across "
        f"{duplicate_keys} keys"
    )
    print(
        f"Negative clip collisions: {len(proximity_excluded)} excluded, "
        f"{len(proximity_overridden)} retained by durable hard-negative review"
    )
    print(
        f"Event groups: {event_groups}; "
        f"cross-group direct overlaps: {cross_group_overlaps}"
    )
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Audit saved to: {AUDIT_FILE}")


if __name__ == "__main__":
    main()
