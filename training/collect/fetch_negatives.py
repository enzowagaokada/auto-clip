import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from fetch_chat import fetch_chat_window, get_max_workers
from fetch_clips import CLIENT_ID, CLIENT_SECRET, get_app_access_token
from window_geometry import (
    WINDOW_AFTER_SECONDS,
    WINDOW_BEFORE_SECONDS,
    WINDOW_GEOMETRY_NAME,
    WINDOW_GEOMETRY_VERSION,
    has_current_geometry,
    window_bounds,
)


NEGATIVE_RATIO = 2
CLIP_EXCLUSION_SECONDS = 60


def parse_twitch_duration(duration):
    """Convert Twitch durations like '3h12m5s' or '45m2s' into seconds."""
    total = 0

    for value, unit in re.findall(r"(\d+)([hms])", duration):
        value = int(value)

        if unit == "h":
            total += value * 3600
        elif unit == "m":
            total += value * 60
        elif unit == "s":
            total += value

    return total


def fetch_vod_duration(vod_id, headers):
    """Fetch a VOD duration from Twitch Helix."""
    url = "https://api.twitch.tv/helix/videos"
    params = {"id": str(vod_id)}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    videos = response.json().get("data", [])
    if not videos:
        return None

    return parse_twitch_duration(videos[0]["duration"])


def existing_negative_offsets(output_dir, vod_id):
    """Return current and stale offsets saved as `{vod_id}_{offset}.json`."""
    current = set()
    stale = set()
    prefix = f"{vod_id}_"
    suffix = ".json"

    for name in os.listdir(output_dir):
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        stem = name[len(prefix) : -len(suffix)]
        try:
            offset = int(stem)
        except ValueError:
            continue
        path = os.path.join(output_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
            destination = (
                current
                if has_current_geometry(record, expected_target=offset)
                else stale
            )
        except (OSError, ValueError, json.JSONDecodeError):
            destination = stale
        destination.add(offset)

    return current, stale


def is_far_from_clips(candidate_offset, clip_offsets):
    """Return True if the candidate is outside the exclusion buffer for every clip."""
    for clip_offset in clip_offsets:
        if abs(candidate_offset - clip_offset) < CLIP_EXCLUSION_SECONDS:
            return False

    return True


def sample_negative_offsets(vod_duration, clip_offsets, count, exclude_offsets=None):
    """Sample random VOD offsets that are not close to known clip moments."""
    offsets = []
    attempts = 0
    max_attempts = max(count * 100, 100)
    exclude_offsets = exclude_offsets or set()

    min_offset = WINDOW_BEFORE_SECONDS
    max_offset = vod_duration - WINDOW_AFTER_SECONDS

    if count <= 0 or max_offset <= min_offset:
        return offsets

    while len(offsets) < count and attempts < max_attempts:
        attempts += 1
        candidate_offset = random.randint(min_offset, max_offset)

        if candidate_offset in exclude_offsets or candidate_offset in offsets:
            continue

        if is_far_from_clips(candidate_offset, clip_offsets):
            offsets.append(candidate_offset)

    return offsets


def process_negative(task):
    """Fetch and save one negative window. Returns its status."""
    streamer_name, vod_id, offset, output_file, is_refetch = task

    start_time, end_time = window_bounds(offset)

    messages = fetch_chat_window(vod_id, start_time, end_time)
    if not messages:
        if is_refetch and os.path.exists(output_file):
            os.replace(output_file, output_file + ".window-v1-stale")
            return "quarantined"
        return "empty"

    payload = {
        "label": 0,
        "streamer_name": streamer_name,
        "vod_id": str(vod_id),
        "target_offset": offset,
        "window_start": start_time,
        "window_end": end_time,
        "window_geometry": WINDOW_GEOMETRY_NAME,
        "window_geometry_version": WINDOW_GEOMETRY_VERSION,
        "message_count": len(messages),
        "messages": messages,
    }
    temporary_output = output_file + ".tmp"
    with open(temporary_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(temporary_output, output_file)
    return "refetched" if is_refetch else "saved"


def main():
    clips_file = "data/raw/clips.csv"
    output_dir = "data/raw/chat_negatives"

    if not os.path.exists(clips_file):
        print(f"Error: {clips_file} not found. Run fetch_clips.py first.")
        return

    if not CLIENT_ID or not CLIENT_SECRET:
        print("Error: TWITCH_CLIENT_ID or TWITCH_CLIENT_SECRET is not set.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("Authenticating with Twitch...")
    token = get_app_access_token()
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {token}",
    }

    clips_df = pd.read_csv(clips_file)
    grouped = clips_df.groupby(["streamer_name", "vod_id"])
    max_workers = get_max_workers()

    # Phase 1: top up each VOD to clips * NEGATIVE_RATIO using existing files.
    print(
        f"Building negative task list (target {NEGATIVE_RATIO}:1 per VOD, "
        "fetching durations)..."
    )
    tasks = []
    total_existing = 0
    total_target = 0
    total_shortfall = 0

    for (streamer_name, vod_id), group in grouped:
        clip_offsets = group["vod_offset"].astype(int).tolist()
        target_count = len(clip_offsets) * NEGATIVE_RATIO
        current_offsets, stale_offsets = existing_negative_offsets(
            output_dir,
            vod_id,
        )
        existing_offsets = current_offsets | stale_offsets
        need = max(0, target_count - len(existing_offsets))

        total_target += target_count
        total_existing += len(existing_offsets)

        negative_offsets = []
        if need > 0:
            try:
                vod_duration = fetch_vod_duration(vod_id, headers)
            except requests.exceptions.RequestException as e:
                print(f"  -> {streamer_name} VOD {vod_id}: could not fetch duration: {e}")
                vod_duration = None

            if vod_duration:
                negative_offsets = sample_negative_offsets(
                    vod_duration=vod_duration,
                    clip_offsets=clip_offsets,
                    count=need,
                    exclude_offsets=existing_offsets,
                )
                total_shortfall += need - len(negative_offsets)
            else:
                total_shortfall += need

        # Preserve sampled negative anchors across geometry migrations by
        # refetching stale files in place before adding any shortfall.
        for offset in sorted(stale_offsets):
            output_file = os.path.join(output_dir, f"{vod_id}_{offset}.json")
            tasks.append(
                (streamer_name, str(vod_id), offset, output_file, True)
            )

        for offset in negative_offsets:
            output_file = os.path.join(output_dir, f"{vod_id}_{offset}.json")
            tasks.append(
                (streamer_name, str(vod_id), offset, output_file, False)
            )

    print(
        f"Target={total_target} existing={total_existing} "
        f"to_fetch={len(tasks)} undersampled={total_shortfall}"
    )
    if not tasks:
        print("Already at target ratio for every VOD. Nothing to fetch.")
        return

    print(f"Fetching {len(tasks)} negative windows with {max_workers} workers...")

    # Phase 2: fetch chat windows concurrently.
    total_written = 0
    total_refetched = 0
    total_quarantined = 0
    total_empty = 0
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_negative, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                status = future.result()
                if status == "saved":
                    total_written += 1
                elif status == "refetched":
                    total_refetched += 1
                elif status == "quarantined":
                    total_quarantined += 1
                else:
                    total_empty += 1
            except Exception as e:
                print(f"    Error: {e}")
                total_empty += 1
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(
                    f"[{done}/{len(tasks)}] saved={total_written} "
                    f"refetched={total_refetched} "
                    f"quarantined={total_quarantined} empty={total_empty}"
                )

    print("\n--- Summary ---")
    print(f"Saved negatives: {total_written}")
    print(f"Stale geometry refetched: {total_refetched}")
    print(f"Unavailable stale files quarantined: {total_quarantined}")
    print(f"No messages / Errors: {total_empty}")
    print(f"Already on disk before run: {total_existing}")
    print(f"Per-VOD target total: {total_target}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
