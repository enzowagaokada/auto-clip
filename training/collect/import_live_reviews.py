"""Import live shadow reviews into durable labels and raw training windows.

Run from the repository root:
    python training/collect/import_live_reviews.py --partition calibration
"""

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone

from import_reviews import (
    DEFAULT_OUTPUT,
    TRAINING_LABELS,
    annotation_key,
    load_existing_annotations,
    load_jsonl,
    normalize_label,
    write_annotations,
)
from window_geometry import (
    WINDOW_GEOMETRY_NAME,
    WINDOW_GEOMETRY_VERSION,
    window_bounds,
)


LIVE_LOG_ROOT = "data/live/shadow/window-v2"
SUPPORTED_LIVE_SCHEMA_VERSIONS = {1, 2}
DEFAULT_LIVE_DIR = "data/raw/chat_live"
DEFAULT_SOURCE_RUN = "live-shadow-window-v2"
PARTITIONS = {"calibration", "confirmation"}


def parse_datetime(value):
    """Parse live ISO timestamps, including Z and extra fractional digits."""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        if "." in text:
            head, rest = text.split(".", 1)
            frac_end = 0
            while frac_end < len(rest) and rest[frac_end].isdigit():
                frac_end += 1
            fraction = rest[:frac_end][:6].ljust(6, "0")
            text = f"{head}.{fraction}{rest[frac_end:]}"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def offset_to_stamp(seconds):
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h{minutes}m{secs}s"


def twitch_url(vod_id, stamp=None, target_offset=None):
    if not stamp:
        stamp = offset_to_stamp(target_offset or 0)
    return f"https://www.twitch.tv/videos/{vod_id}?t={stamp}"


def sessions_by_id(path):
    sessions = {}
    for row in load_jsonl(path):
        session_id = str(row.get("session_id") or "").strip()
        if session_id:
            sessions[session_id] = row
    return sessions


def candidates_by_id(path):
    return {
        str(row.get("candidate_id") or "").strip(): row
        for row in load_jsonl(path)
        if row.get("candidate_id")
    }


def episodes_by_id(path, partition):
    episodes = {}
    for row in load_jsonl(path):
        schema_version = int(row.get("schema_version", 0))
        if schema_version not in SUPPORTED_LIVE_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported episode schema_version {schema_version}; "
                f"expected one of {sorted(SUPPORTED_LIVE_SCHEMA_VERSIONS)}"
            )
        row_partition = str(row.get("review_partition") or partition).strip()
        if row_partition != partition:
            raise ValueError(
                f"Episode {row.get('episode_id')} belongs to {row_partition}, "
                f"not requested partition {partition}"
            )
        episode_id = str(row.get("episode_id") or "").strip()
        if episode_id:
            episodes[episode_id] = row
    return episodes


def partition_path(partition, filename):
    if partition not in PARTITIONS:
        raise ValueError(
            f"Unknown partition {partition!r}; expected calibration or confirmation"
        )
    return os.path.join(LIVE_LOG_ROOT, partition, filename)


def convert_live_messages(messages, target_offset, target_at):
    converted = []
    target_at = parse_datetime(target_at)
    for message in messages or []:
        text = str(message.get("text") or message.get("message") or "")
        created_at = message.get("time") or message.get("created_at")
        if created_at:
            delta = (parse_datetime(created_at) - target_at).total_seconds()
            offset_seconds = float(target_offset) + delta
        else:
            offset_seconds = float(target_offset)
        converted.append(
            {
                "offset_seconds": offset_seconds,
                "created_at": created_at or "",
                "user": message.get("user") or "Unknown",
                "message": text,
            }
        )
    return converted


def live_window_record(
    candidate,
    vod_id,
    target_offset,
    streamer_name,
    label,
    *,
    source=DEFAULT_SOURCE_RUN,
    review_partition="calibration",
    review_label=None,
    review_identity=None,
):
    start_time, end_time = window_bounds(target_offset)
    messages = convert_live_messages(
        candidate.get("messages") or [],
        target_offset,
        candidate["target_at"],
    )
    return {
        "label": label,
        "streamer_name": streamer_name,
        "vod_id": str(vod_id),
        "target_offset": int(target_offset),
        "window_start": start_time,
        "window_end": end_time,
        "window_geometry": WINDOW_GEOMETRY_NAME,
        "window_geometry_version": WINDOW_GEOMETRY_VERSION,
        "message_count": len(messages),
        "messages": messages,
        "candidate_id": candidate.get("candidate_id"),
        "episode_id": candidate.get("episode_id"),
        "record_type": candidate.get("record_type"),
        "review_partition": review_partition,
        "review_label": review_label,
        "review_identity": review_identity,
        "model_manifest_sha256": candidate.get("model_manifest_sha256"),
        "peak_target_at": candidate.get("peak_target_at"),
        "source": source,
    }


def write_live_window(live_dir, record):
    os.makedirs(live_dir, exist_ok=True)
    filename = f"{record['vod_id']}_{record['target_offset']}.json"
    output_file = os.path.join(live_dir, filename)
    temporary_output = output_file + ".tmp"
    with open(temporary_output, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    os.replace(temporary_output, output_file)
    return output_file


def validate_episode_geometry(episode):
    target = parse_datetime(episode["peak_target_at"])
    start = parse_datetime(episode["peak_window_start"])
    end = parse_datetime(episode["peak_window_end"])
    if abs((target - start).total_seconds() - 5) > 1e-3:
        raise ValueError(f"Episode {episode.get('episode_id')} peak window starts incorrectly")
    if abs((end - target).total_seconds() - 30) > 1e-3:
        raise ValueError(f"Episode {episode.get('episode_id')} peak window ends incorrectly")


def import_live_reviews(
    *,
    partition,
    input_kind,
    review_file,
    records_file,
    sessions_file,
    output,
    live_dir,
    source_run,
    validate_only=False,
):
    if partition not in PARTITIONS:
        raise ValueError("partition must be calibration or confirmation")
    if partition == "confirmation" and not validate_only:
        raise ValueError(
            "Locked confirmation reviews cannot be imported into training; "
            "use --validate-only to audit them."
        )
    sessions = sessions_by_id(sessions_file)
    records = (
        episodes_by_id(records_file, partition)
        if input_kind == "episode"
        else candidates_by_id(records_file)
    )
    annotations = {} if validate_only else load_existing_annotations(output)
    imported = Counter()
    skipped_unlabeled = 0
    skipped_no_vod = 0
    skipped_missing = 0
    windows_written = 0

    with open(review_file, "r", encoding="utf-8", newline="") as f:
        for review_row in csv.DictReader(f):
            review_label = normalize_label(review_row.get("review_label"))
            if review_label is None:
                skipped_unlabeled += 1
                continue

            id_field = "episode_id" if input_kind == "episode" else "candidate_id"
            record_id = str(review_row.get(id_field) or "").strip()
            record = records.get(record_id)
            if record is None:
                skipped_missing += 1
                print(f"Skipping {record_id or '(missing id)'}: no {input_kind} log")
                continue

            session_id = str(review_row.get("session_id") or "").strip()
            session = sessions.get(session_id, {})
            for owner, row_partition in (
                ("review", review_row.get("review_partition")),
                ("record", record.get("review_partition")),
                ("session", session.get("review_partition")),
            ):
                if row_partition and str(row_partition).strip() != partition:
                    raise ValueError(
                        f"{owner} for {record_id} belongs to {row_partition}, "
                        f"not {partition}"
                    )
            vod_id = str(session.get("vod_id") or "").strip()
            if not vod_id:
                skipped_no_vod += 1
                print(
                    f"Skipping {record_id}: session {session_id} has no vod_id"
                )
                continue

            try:
                offset_field = (
                    "peak_stream_offset_seconds"
                    if input_kind == "episode"
                    else "stream_offset_seconds"
                )
                target_offset = int(round(float(record[offset_field])))
            except (KeyError, TypeError, ValueError):
                skipped_missing += 1
                print(f"Skipping {record_id}: missing {offset_field}")
                continue
            if input_kind == "episode":
                validate_episode_geometry(record)

            streamer_name = str(
                record.get("streamer") or review_row.get("streamer") or "unknown"
            ).strip()
            training_label = TRAINING_LABELS[review_label]
            stamp = str(review_row.get("stream_offset_stamp") or "").strip()
            try:
                score_field = "peak_score" if input_kind == "episode" else "score"
                score = float(review_row.get(score_field) or record.get(score_field))
            except (TypeError, ValueError):
                score = ""

            annotation = {
                "streamer_name": streamer_name,
                "vod_id": vod_id,
                "target_offset": target_offset,
                "review_label": review_label,
                "training_label": training_label,
                "review_notes": str(review_row.get("reason") or "").strip(),
                "source_run": f"{source_run}-{partition}",
                "dataset_index": "",
                "score": score,
                "twitch_url": twitch_url(vod_id, stamp=stamp, target_offset=target_offset),
                "review_partition": partition,
                "review_identity": f"{partition}:{input_kind}:{record_id}",
            }
            imported[review_label] += 1

            if validate_only:
                continue

            annotations[annotation_key(annotation)] = annotation
            window_source = dict(record)
            if input_kind == "episode":
                window_source["target_at"] = record["peak_target_at"]
            record_label = int(training_label) if training_label else 0
            window_record = live_window_record(
                window_source,
                vod_id,
                target_offset,
                streamer_name,
                record_label,
                source=f"{source_run}-episode-peak"
                if input_kind == "episode"
                else source_run,
                review_partition=partition,
                review_label=review_label,
                review_identity=f"{partition}:{input_kind}:{record_id}",
            )
            write_live_window(live_dir, window_record)
            windows_written += 1

    if not validate_only:
        write_annotations(output, annotations)

    action = "Validated" if validate_only else "Imported"
    print(
        f"{action} {sum(imported.values())} {partition} {input_kind} reviews: "
        f"positive={imported['positive']} "
        f"hard_negative={imported['hard_negative']} "
        f"uncertain={imported['uncertain']}"
    )
    if not validate_only:
        print(f"Wrote {windows_written} windows -> {live_dir}")
        print(f"Durable annotations: {len(annotations)} -> {output}")
    print(
        "Skipped: "
        f"unlabeled={skipped_unlabeled} "
        f"no_vod_id={skipped_no_vod} "
        f"missing_record={skipped_missing}"
    )
    return {
        "partition": partition,
        "input_kind": input_kind,
        "imported": dict(imported),
        "windows_written": windows_written,
        "skipped_unlabeled": skipped_unlabeled,
        "skipped_no_vod": skipped_no_vod,
        "skipped_missing": skipped_missing,
        "validate_only": validate_only,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=sorted(PARTITIONS), default="calibration")
    parser.add_argument("--input-kind", choices=("episode", "candidate"), default="episode")
    parser.add_argument("--review-file", default=None)
    parser.add_argument("--records-file", default=None)
    parser.add_argument("--sessions-file", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--live-dir", default=DEFAULT_LIVE_DIR)
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate labels and joins without writing annotations or raw windows.",
    )
    args = parser.parse_args()

    review_name = (
        "episodes_review.csv"
        if args.input_kind == "episode"
        else "candidates_review.csv"
    )
    records_name = (
        "episodes.jsonl"
        if args.input_kind == "episode"
        else "candidates.jsonl"
    )
    import_live_reviews(
        partition=args.partition,
        input_kind=args.input_kind,
        review_file=args.review_file or partition_path(args.partition, review_name),
        records_file=args.records_file or partition_path(args.partition, records_name),
        sessions_file=args.sessions_file
        or partition_path(args.partition, "sessions.jsonl"),
        output=args.output,
        live_dir=args.live_dir,
        source_run=args.source_run,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    main()
