"""Resolve live shadow session vod_ids from Helix archives.

Run from the repository root:
    python training/live/resolve_session_vods.py --partition calibration
    python training/live/resolve_session_vods.py --all
"""

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone

from dotenv import load_dotenv
import requests

LIVE_LOG_ROOT = "data/live/shadow/window-v2"
PARTITIONS = ("calibration", "confirmation")
PAGE_SIZE = 100
CREATED_AT_SLACK_SECONDS = 120
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
VIDEOS_URL = "https://api.twitch.tv/helix/videos"
REVIEW_URL_FIELDS = ("vod_id", "twitch_url")
EPISODE_CSV = "episodes_review.csv"
CANDIDATE_CSV = "candidates_review.csv"
SESSIONS_JSONL = "sessions.jsonl"


def load_env():
    load_dotenv()
    client_id = os.getenv("TWITCH_CLIENT_ID")
    client_secret = os.getenv("TWITCH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET are required")
    return client_id, client_secret


def get_app_access_token(client_id, client_secret, post=requests.post):
    response = post(
        TOKEN_URL,
        params={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def parse_datetime(value):
    if not value:
        return None
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
    return parsed.astimezone(timezone.utc)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path, rows):
    directory = os.path.dirname(path) or "."
    handle, temp_path = tempfile.mkstemp(prefix="sessions.", suffix=".jsonl", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def twitch_url(vod_id, stamp):
    stamp = str(stamp or "0h0m0s").strip() or "0h0m0s"
    return f"https://www.twitch.tv/videos/{vod_id}?t={stamp}"


def fetch_archives(user_id, headers, get=requests.get):
    videos = []
    cursor = None
    while True:
        params = {
            "user_id": user_id,
            "type": "archive",
            "first": PAGE_SIZE,
        }
        if cursor:
            params["after"] = cursor
        response = get(VIDEOS_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        page = payload.get("data") or []
        videos.extend(page)
        cursor = (payload.get("pagination") or {}).get("cursor")
        if not page or not cursor:
            break
    return videos


def match_vod(session, videos):
    stream_id = str(session.get("stream_id") or "").strip()
    user_id = str(session.get("broadcaster_id") or "").strip()
    started = parse_datetime(session.get("stream_started_at"))
    stream_matches = []
    time_matches = []
    for video in videos:
        video_id = str(video.get("id") or "").strip()
        if not video_id:
            continue
        video_stream = str(video.get("stream_id") or "").strip()
        if stream_id and video_stream and video_stream == stream_id:
            stream_matches.append(video_id)
            continue
        if stream_id and video_stream:
            continue
        if user_id and str(video.get("user_id") or "").strip() != user_id:
            continue
        created = parse_datetime(video.get("created_at"))
        if started is None or created is None:
            continue
        delta = abs((created - started).total_seconds())
        if delta <= CREATED_AT_SLACK_SECONDS:
            time_matches.append(video_id)
    unique_stream = list(dict.fromkeys(stream_matches))
    if len(unique_stream) == 1:
        return unique_stream[0]
    if len(unique_stream) > 1:
        return None
    unique_time = list(dict.fromkeys(time_matches))
    if len(unique_time) == 1:
        return unique_time[0]
    return None


def apply_session_vods(rows, resolved):
    updated = []
    already = 0
    filled = 0
    conflicts = []
    for row in rows:
        session_id = str(row.get("session_id") or "").strip()
        existing = str(row.get("vod_id") or "").strip()
        incoming = resolved.get(session_id)
        if existing:
            already += 1
            if incoming and incoming != existing:
                conflicts.append((session_id, existing, incoming))
            updated.append(row)
            continue
        if incoming:
            row = dict(row)
            row["vod_id"] = incoming
            filled += 1
        updated.append(row)
    return updated, filled, already, conflicts


def pending_sessions(rows):
    pending = []
    for row in rows:
        if str(row.get("vod_id") or "").strip():
            continue
        if not str(row.get("session_id") or "").strip():
            continue
        pending.append(row)
    return pending


def rewrite_review_csv(path, vod_by_session):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for field in REVIEW_URL_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    changed = 0
    for row in rows:
        session_id = str(row.get("session_id") or "").strip()
        vod_id = vod_by_session.get(session_id) or str(row.get("vod_id") or "").strip()
        if not vod_id:
            continue
        stamp = str(row.get("stream_offset_stamp") or "").strip()
        url = twitch_url(vod_id, stamp)
        if row.get("vod_id") != vod_id or row.get("twitch_url") != url:
            changed += 1
        row["vod_id"] = vod_id
        row["twitch_url"] = url
    directory = os.path.dirname(path) or "."
    handle, temp_path = tempfile.mkstemp(prefix="review.", suffix=".csv", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return changed


def resolve_partition(partition_dir, fetch_archives_fn, dry_run=False):
    sessions_path = os.path.join(partition_dir, SESSIONS_JSONL)
    rows = load_jsonl(sessions_path)
    pending = pending_sessions(rows)
    videos_by_user = {}
    resolved = {}
    unresolved = []
    for session in pending:
        user_id = str(session.get("broadcaster_id") or "").strip()
        session_id = str(session.get("session_id") or "").strip()
        if not user_id:
            unresolved.append(session)
            continue
        if user_id not in videos_by_user:
            videos_by_user[user_id] = fetch_archives_fn(user_id)
        vod_id = match_vod(session, videos_by_user[user_id])
        if vod_id:
            resolved[session_id] = vod_id
        else:
            unresolved.append(session)
    updated, filled, already, conflicts = apply_session_vods(rows, resolved)
    if not dry_run and filled:
        write_jsonl_atomic(sessions_path, updated)
    vod_by_session = {
        str(row.get("session_id") or "").strip(): str(row.get("vod_id") or "").strip()
        for row in updated
        if str(row.get("vod_id") or "").strip()
    }
    csv_changed = 0
    if not dry_run:
        csv_changed += rewrite_review_csv(
            os.path.join(partition_dir, EPISODE_CSV), vod_by_session
        )
        csv_changed += rewrite_review_csv(
            os.path.join(partition_dir, CANDIDATE_CSV), vod_by_session
        )
    return {
        "sessions": len(rows),
        "pending": len(pending),
        "filled": filled,
        "already": already,
        "unresolved": unresolved,
        "conflicts": conflicts,
        "csv_changed": csv_changed,
        "resolved": resolved,
    }


def report_result(partition, result):
    print(
        f"{partition}: filled={result['filled']} already={result['already']} "
        f"pending={result['pending']} unresolved={len(result['unresolved'])} "
        f"csv_rows={result['csv_changed']}"
    )
    for session in result["unresolved"]:
        print(
            "  unresolved "
            f"streamer={session.get('streamer')} "
            f"session_id={session.get('session_id')} "
            f"stream_id={session.get('stream_id')}"
        )
    for session_id, existing, incoming in result["conflicts"]:
        print(
            f"  conflict session_id={session_id} existing={existing} matched={incoming}"
        )


def main():
    parser = argparse.ArgumentParser(description="Resolve live session vod_ids from Helix")
    parser.add_argument("--partition", choices=PARTITIONS, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--live-root", default=LIVE_LOG_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if bool(args.partition) == bool(args.all):
        parser.error("specify exactly one of --partition or --all")

    partitions = PARTITIONS if args.all else (args.partition,)
    client_id, client_secret = load_env()
    token = get_app_access_token(client_id, client_secret)
    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {token}",
    }

    def fetch(user_id):
        return fetch_archives(user_id, headers)

    for partition in partitions:
        partition_dir = os.path.join(args.live_root, partition)
        result = resolve_partition(partition_dir, fetch, dry_run=args.dry_run)
        report_result(partition, result)


if __name__ == "__main__":
    main()
