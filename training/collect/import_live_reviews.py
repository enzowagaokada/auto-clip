"""Import live shadow reviews into durable labels and raw training windows.

Run from the repository root:
    python training/collect/import_live_reviews.py
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


DEFAULT_REVIEW_FILE = "data/live/shadow/window-v2/candidates_review.csv"
DEFAULT_CANDIDATES_FILE = "data/live/shadow/window-v2/candidates.jsonl"
DEFAULT_SESSIONS_FILE = "data/live/shadow/window-v2/sessions.jsonl"
DEFAULT_LIVE_DIR = "data/raw/chat_live"
DEFAULT_SOURCE_RUN = "live-shadow-window-v2"


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


def live_window_record(candidate, vod_id, target_offset, streamer_name, label):
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
        "source": DEFAULT_SOURCE_RUN,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-file", default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--candidates-file", default=DEFAULT_CANDIDATES_FILE)
    parser.add_argument("--sessions-file", default=DEFAULT_SESSIONS_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--live-dir", default=DEFAULT_LIVE_DIR)
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    args = parser.parse_args()

    sessions = sessions_by_id(args.sessions_file)
    candidates = candidates_by_id(args.candidates_file)
    annotations = load_existing_annotations(args.output)

    imported = Counter()
    skipped_unlabeled = 0
    skipped_no_vod = 0
    skipped_missing = 0
    windows_written = 0

    with open(args.review_file, "r", encoding="utf-8", newline="") as f:
        for review_row in csv.DictReader(f):
            review_label = normalize_label(review_row.get("review_label"))
            if review_label is None:
                skipped_unlabeled += 1
                continue

            candidate_id = str(review_row.get("candidate_id") or "").strip()
            candidate = candidates.get(candidate_id)
            if candidate is None:
                skipped_missing += 1
                print(f"Skipping {candidate_id or '(missing id)'}: no candidate log")
                continue

            session = sessions.get(str(review_row.get("session_id") or "").strip(), {})
            vod_id = str(session.get("vod_id") or "").strip()
            if not vod_id:
                skipped_no_vod += 1
                print(
                    f"Skipping {candidate_id}: session "
                    f"{review_row.get('session_id')} has no vod_id"
                )
                continue

            try:
                target_offset = int(round(float(candidate["stream_offset_seconds"])))
            except (KeyError, TypeError, ValueError):
                skipped_missing += 1
                print(f"Skipping {candidate_id}: missing stream_offset_seconds")
                continue

            streamer_name = str(
                candidate.get("streamer") or review_row.get("streamer") or "unknown"
            ).strip()
            training_label = TRAINING_LABELS[review_label]
            stamp = str(review_row.get("stream_offset_stamp") or "").strip()
            try:
                score = float(review_row.get("score") or candidate.get("score"))
            except (TypeError, ValueError):
                score = ""

            annotation = {
                "streamer_name": streamer_name,
                "vod_id": vod_id,
                "target_offset": target_offset,
                "review_label": review_label,
                "training_label": training_label,
                "review_notes": str(review_row.get("reason") or "").strip(),
                "source_run": args.source_run,
                "dataset_index": "",
                "score": score,
                "twitch_url": twitch_url(vod_id, stamp=stamp, target_offset=target_offset),
            }
            annotations[annotation_key(annotation)] = annotation
            imported[review_label] += 1

            record_label = int(training_label) if training_label else 0
            record = live_window_record(
                candidate,
                vod_id,
                target_offset,
                streamer_name,
                record_label,
            )
            write_live_window(args.live_dir, record)
            windows_written += 1

    write_annotations(args.output, annotations)

    print(
        f"Imported {sum(imported.values())} live reviews: "
        f"positive={imported['positive']} "
        f"hard_negative={imported['hard_negative']} "
        f"uncertain={imported['uncertain']}"
    )
    print(f"Wrote {windows_written} windows -> {args.live_dir}")
    print(f"Durable annotations: {len(annotations)} -> {args.output}")
    print(
        "Skipped: "
        f"unlabeled={skipped_unlabeled} "
        f"no_vod_id={skipped_no_vod} "
        f"missing_candidate={skipped_missing}"
    )


if __name__ == "__main__":
    main()
