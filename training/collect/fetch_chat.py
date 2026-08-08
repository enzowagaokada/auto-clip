import os
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yaml

from window_geometry import (
    WINDOW_GEOMETRY_NAME,
    WINDOW_GEOMETRY_VERSION,
    has_current_geometry,
    window_bounds,
)

# The standard Twitch Web Client ID used for public GraphQL requests
TWITCH_GQL_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
TWITCH_GQL_URL = "https://gql.twitch.tv/gql"

DEFAULT_MAX_WORKERS = 6


def get_max_workers():
    """Read max_workers from config.yaml (falls back to a safe default)."""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return int(config.get("twitch", {}).get("clips", {}).get("max_workers", DEFAULT_MAX_WORKERS))
    except Exception:
        return DEFAULT_MAX_WORKERS


def request_with_backoff(payload, headers, max_retries=5):
    """POST to the GQL endpoint with exponential backoff on transient failures.

    Retries on network errors and HTTP 429/5xx. Returns the parsed JSON, or None
    if all retries are exhausted.
    """
    delay = 1.0
    for attempt in range(max_retries):
        try:
            response = requests.post(TWITCH_GQL_URL, json=payload, headers=headers, timeout=20)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    Request failed after {max_retries} attempts: {e}")
                return None
            # Exponential backoff with jitter to avoid thundering-herd retries.
            time.sleep(delay + random.uniform(0, 0.5))
            delay *= 2
    return None

def fetch_chat_window(vod_id, start_offset_seconds, end_offset_seconds):
    """
    Fetches chat messages for a specific window in a VOD using Twitch's GraphQL API.

    Pagination is done by content offset, NOT by cursor. Twitch's anti-bot
    integrity check (IntegrityCheckFailed) fires on cursor-based pagination but
    not on offset-based pagination, so we always re-query with the offset of the
    last message seen and dedupe overlapping pages by message id.
    """
    messages = []
    seen_ids = set()

    # GraphQL Query for VOD comments. `id` is requested so we can dedupe pages.
    query = """
    query VideoCommentsByOffsetOrCursor($videoID: ID!, $contentOffsetSeconds: Int) {
        video(id: $videoID) {
            comments(contentOffsetSeconds: $contentOffsetSeconds, first: 100) {
                edges {
                    node {
                        id
                        createdAt
                        contentOffsetSeconds
                        commenter {
                            displayName
                        }
                        message {
                            fragments {
                                text
                            }
                        }
                    }
                }
                pageInfo {
                    hasNextPage
                }
            }
        }
    }
    """

    headers = {
        "Client-Id": TWITCH_GQL_CLIENT_ID,
        "Content-Type": "application/json"
    }

    next_offset = max(0, start_offset_seconds)

    while True:
        payload = {
            "query": query,
            "variables": {
                "videoID": str(vod_id),
                "contentOffsetSeconds": int(next_offset),
            },
        }

        try:
            data = request_with_backoff(payload, headers)
            if data is None:
                break

            # Surface GraphQL errors instead of silently treating them as empty
            if data.get("errors"):
                print(f"    GraphQL error: {data['errors']}")
                break

            # Handle missing data (e.g. deleted/unavailable VOD)
            if not data.get("data") or not data["data"].get("video") or not data["data"]["video"].get("comments"):
                break

            comments_data = data["data"]["video"]["comments"]
            edges = comments_data.get("edges", [])

            if not edges:
                break

            page_max_offset = next_offset
            new_in_page = 0

            for edge in edges:
                node = edge["node"]
                offset = node["contentOffsetSeconds"]
                page_max_offset = max(page_max_offset, offset)

                # If we've passed our window, stop fetching entirely
                if offset > end_offset_seconds:
                    return messages

                msg_id = node.get("id")
                if msg_id in seen_ids:
                    continue
                seen_ids.add(msg_id)
                new_in_page += 1

                # Twitch may return a page that begins before the requested
                # content offset. Live inference never includes messages before
                # its rolling-window boundary, so historical collection must
                # filter them for parity.
                if offset < start_offset_seconds:
                    continue

                # Assemble the message text from fragments
                msg_fragments = node.get("message", {}).get("fragments", [])
                full_text = "".join([frag.get("text", "") for frag in msg_fragments]).strip()

                commenter = node.get("commenter")
                display_name = commenter.get("displayName") if commenter else "Unknown"

                messages.append({
                    "offset_seconds": offset,
                    "created_at": node["createdAt"],
                    "user": display_name,
                    "message": full_text
                })

            # Stop if Twitch says there are no more pages
            page_info = comments_data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break

            # Advance by offset. Move to the latest offset seen; if the page did
            # not advance the offset (a single second with >100 messages), force
            # progress by +1 to avoid an infinite loop.
            if page_max_offset > next_offset:
                next_offset = page_max_offset
            else:
                next_offset = next_offset + 1

        except Exception as e:
            print(f"    Error fetching chunk: {e}")
            break

    return messages

def process_clip(row, chat_dir):
    """Fetch and save one clip's chat window. Returns a status string."""
    clip_id = row["clip_id"]
    vod_id = row["vod_id"]
    vod_offset = float(row["vod_offset"])

    output_file = os.path.join(chat_dir, f"{clip_id}.json")
    existed = os.path.exists(output_file)

    # Skip only records already fetched with the current geometry. This makes
    # the window migration resumable without silently retaining stale files.
    if existed:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if has_current_geometry(existing, expected_target=vod_offset):
                return "skipped"
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    # Helix vod_offset is the start of the clip video, not the time the viewer
    # pressed Clip. Keep a fixed 35-second model contract while covering five
    # seconds before that start and the first 30 seconds of clipped video.
    start_time, end_time = window_bounds(vod_offset)

    messages = fetch_chat_window(vod_id, start_time, end_time)

    if not messages:
        if existed:
            os.replace(output_file, output_file + ".window-v1-stale")
            return "quarantined"
        return "empty"

    duration = row.get("duration")
    clip_duration = None if pd.isna(duration) else float(duration)
    temporary_output = output_file + ".tmp"
    with open(temporary_output, "w", encoding="utf-8") as f:
        json.dump({
            "clip_id": clip_id,
            "vod_id": vod_id,
            "target_offset": vod_offset,
            "clip_start_offset": vod_offset,
            "clip_duration_seconds": clip_duration,
            "window_start": start_time,
            "window_end": end_time,
            "window_geometry": WINDOW_GEOMETRY_NAME,
            "window_geometry_version": WINDOW_GEOMETRY_VERSION,
            "message_count": len(messages),
            "messages": messages,
        }, f, indent=2, ensure_ascii=False)
    os.replace(temporary_output, output_file)
    return "refetched" if existed else "fetched"


def main():
    clips_file = "data/raw/clips.csv"
    chat_dir = "data/raw/chat"

    if not os.path.exists(clips_file):
        print(f"Error: {clips_file} not found. Please run fetch_clips.py first.")
        return

    os.makedirs(chat_dir, exist_ok=True)

    print(f"Loading clips from {clips_file}...")
    df = pd.read_csv(clips_file)
    total_clips = len(df)
    max_workers = get_max_workers()
    print(f"Found {total_clips} clips. Fetching with {max_workers} workers...")

    counts = {
        "fetched": 0,
        "refetched": 0,
        "quarantined": 0,
        "skipped": 0,
        "empty": 0,
    }
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_clip, row, chat_dir): row["clip_id"]
            for _, row in df.iterrows()
        }
        for future in as_completed(futures):
            clip_id = futures[future]
            try:
                status = future.result()
            except Exception as e:
                print(f"    Error on {clip_id}: {e}")
                status = "empty"
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if done % 50 == 0 or done == total_clips:
                print(
                    f"[{done}/{total_clips}] fetched={counts['fetched']} "
                    f"refetched={counts['refetched']} "
                    f"quarantined={counts['quarantined']} "
                    f"skipped={counts['skipped']} empty={counts['empty']}"
                )

    print("\n--- Summary ---")
    print(f"Successfully fetched: {counts['fetched']}")
    print(f"Stale geometry refetched: {counts['refetched']}")
    print(f"Unavailable stale files quarantined: {counts['quarantined']}")
    print(f"Already existed (skipped): {counts['skipped']}")
    print(f"No messages / Errors: {counts['empty']}")


if __name__ == "__main__":
    main()