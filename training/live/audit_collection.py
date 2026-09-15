"""Audit Checkpoint 3 calibration/confirmation collection gates."""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict


TARGET_STREAMERS = {"arky", "jynxzi", "marlon", "lacy"}
DECIDED_LABELS = {"positive", "hard_negative", "negative"}
VALID_LABELS = DECIDED_LABELS | {"uncertain", ""}
PARTITIONS = ("calibration", "confirmation")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_partition(rows, partition, identity):
    for row in rows:
        value = str(row.get("review_partition") or "").strip()
        if value != partition:
            raise ValueError(
                f"{identity} {row.get(identity)} in {partition} logs declares "
                f"review_partition={value!r}"
            )


def assert_confirmation_not_in_training(annotation_path, live_dir):
    leaked = []
    for row in read_csv(annotation_path):
        if row.get("review_partition") == "confirmation":
            leaked.append(f"annotation:{row.get('review_identity')}")
    if os.path.isdir(live_dir):
        for name in sorted(os.listdir(live_dir)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(live_dir, name)
            with open(path, "r", encoding="utf-8") as handle:
                row = json.load(handle)
            if row.get("review_partition") == "confirmation":
                leaked.append(path)
    if leaked:
        raise ValueError(
            "Locked confirmation rows entered training inputs: " + ", ".join(leaked)
        )


def audit_collection(root, annotation_path, live_dir):
    useful_hours = {}
    all_streamer_hours = {}
    decided = {}
    session_ids = {}
    episode_ids = {}
    per_streamer_hours = defaultdict(float)
    label_counts = {}

    for partition in PARTITIONS:
        partition_dir = os.path.join(root, partition)
        sessions = read_jsonl(os.path.join(partition_dir, "sessions.jsonl"))
        episodes = read_jsonl(os.path.join(partition_dir, "episodes.jsonl"))
        reviews = read_csv(os.path.join(partition_dir, "episodes_review.csv"))
        assert_partition(sessions, partition, "session_id")
        assert_partition(episodes, partition, "episode_id")
        assert_partition(reviews, partition, "episode_id")

        current_sessions = {str(row["session_id"]) for row in sessions}
        current_episodes = {str(row["episode_id"]) for row in episodes}
        if session_ids.keys() & current_sessions:
            raise ValueError("A session_id appears in both review partitions")
        if episode_ids.keys() & current_episodes:
            raise ValueError("An episode_id appears in both review partitions")
        session_ids.update({value: partition for value in current_sessions})
        episode_ids.update({value: partition for value in current_episodes})

        all_streamer_hours[partition] = (
            sum(float(row.get("useful_seconds", 0)) for row in sessions) / 3600
        )
        target_seconds = 0
        for row in sessions:
            streamer = str(row.get("streamer") or "").strip().lower()
            if streamer in TARGET_STREAMERS:
                seconds = float(row.get("useful_seconds", 0))
                target_seconds += seconds
                per_streamer_hours[streamer] += seconds / 3600
        useful_hours[partition] = target_seconds / 3600

        review_by_id = {
            str(row.get("episode_id") or "").strip(): row for row in reviews
        }
        invalid_labels = {
            str(row.get("review_label") or "").strip().lower()
            for row in review_by_id.values()
        } - VALID_LABELS
        if invalid_labels:
            raise ValueError(
                f"{partition} reviews contain invalid labels: "
                f"{sorted(invalid_labels)}"
            )
        unknown_reviews = set(review_by_id) - current_episodes
        if unknown_reviews:
            raise ValueError(
                f"{partition} reviews reference unknown episodes: "
                f"{sorted(unknown_reviews)}"
            )
        counts = Counter(
            str(row.get("review_label") or "").strip().lower()
            for row in review_by_id.values()
            if str(row.get("review_label") or "").strip().lower()
        )
        label_counts[partition] = dict(counts)
        decided[partition] = sum(counts[label] for label in DECIDED_LABELS)
        label_counts[partition]["unreviewed"] = (
            len(current_episodes - set(review_by_id))
            + sum(
                not str(row.get("review_label") or "").strip()
                for row in review_by_id.values()
            )
        )

    assert_confirmation_not_in_training(annotation_path, live_dir)
    total_hours = sum(useful_hours.values())
    total_decided = sum(decided.values())
    gates = {
        "calibration_hours_at_least_8": useful_hours["calibration"] >= 8,
        "confirmation_hours_at_least_8": useful_hours["confirmation"] >= 8,
        "total_hours_at_least_16": total_hours >= 16,
        "each_target_streamer_at_least_2_hours": all(
            per_streamer_hours[name] >= 2 for name in TARGET_STREAMERS
        ),
        "decided_episodes_at_least_100": total_decided >= 100,
        "all_sampled_episodes_reviewed": all(
            label_counts[partition].get("unreviewed", 0) == 0
            for partition in PARTITIONS
        ),
        "confirmation_absent_from_training": True,
    }
    return {
        "useful_hours": useful_hours,
        "total_useful_hours": total_hours,
        "all_streamer_useful_hours": all_streamer_hours,
        "all_streamer_total_useful_hours": sum(all_streamer_hours.values()),
        "per_streamer_useful_hours": {
            name: per_streamer_hours[name] for name in sorted(TARGET_STREAMERS)
        },
        "decided_reviews": decided,
        "total_decided_reviews": total_decided,
        "label_counts": label_counts,
        "gates": gates,
        "ready_for_checkpoint_4": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data/live/shadow/window-v2"
    )
    parser.add_argument(
        "--annotations", default="data/reviews/window_labels.csv"
    )
    parser.add_argument("--live-dir", default="data/raw/chat_live")
    parser.add_argument(
        "--output",
        default="data/live/shadow/window-v2/checkpoint3_audit.json",
    )
    args = parser.parse_args()
    summary = audit_collection(args.root, args.annotations, args.live_dir)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary["gates"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
