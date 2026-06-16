import json
import os
import random
import re
import time

import pandas as pd
import requests

from fetch_chat import fetch_chat_window
from fetch_clips import CLIENT_ID, CLIENT_SECRET, get_app_access_token


NEGATIVE_RATIO = 3
CLIP_EXCLUSION_SECONDS = 60
WINDOW_BEFORE_SECONDS = 30
WINDOW_AFTER_SECONDS = 5


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


def is_far_from_clips(candidate_offset, clip_offsets):
    """Return True if the candidate is outside the exclusion buffer for every clip."""
    for clip_offset in clip_offsets:
        if abs(candidate_offset - clip_offset) < CLIP_EXCLUSION_SECONDS:
            return False

    return True


def sample_negative_offsets(vod_duration, clip_offsets, count):
    """Sample random VOD offsets that are not close to known clip moments."""
    offsets = []
    attempts = 0
    max_attempts = count * 100

    min_offset = WINDOW_BEFORE_SECONDS
    max_offset = vod_duration - WINDOW_AFTER_SECONDS

    if max_offset <= min_offset:
        return offsets

    while len(offsets) < count and attempts < max_attempts:
        attempts += 1
        candidate_offset = random.randint(min_offset, max_offset)

        if is_far_from_clips(candidate_offset, clip_offsets):
            offsets.append(candidate_offset)

    return offsets


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

    total_written = 0
    total_skipped = 0

    for (streamer_name, vod_id), group in grouped:
        clip_offsets = group["vod_offset"].astype(int).tolist()
        negative_count = len(clip_offsets) * NEGATIVE_RATIO

        print(
            f"Sampling {negative_count} negatives for {streamer_name} "
            f"(VOD {vod_id}, {len(clip_offsets)} clips)..."
        )

        try:
            vod_duration = fetch_vod_duration(vod_id, headers)
        except requests.exceptions.RequestException as e:
            print(f"  -> Could not fetch VOD duration: {e}")
            continue

        if not vod_duration:
            print("  -> Missing VOD duration, skipping.")
            continue

        negative_offsets = sample_negative_offsets(
            vod_duration=vod_duration,
            clip_offsets=clip_offsets,
            count=negative_count,
        )

        if len(negative_offsets) < negative_count:
            print(f"  -> Only sampled {len(negative_offsets)} valid negative offsets.")

        for offset in negative_offsets:
            output_file = os.path.join(output_dir, f"{vod_id}_{offset}.json")

            if os.path.exists(output_file):
                total_skipped += 1
                continue

            start_time = max(0, offset - WINDOW_BEFORE_SECONDS)
            end_time = offset + WINDOW_AFTER_SECONDS

            print(f"  -> Fetching negative window at {offset}s")

            messages = fetch_chat_window(vod_id, start_time, end_time)

            if not messages:
                print("     No messages found, skipping.")
                continue

            payload = {
                "label": 0,
                "streamer_name": streamer_name,
                "vod_id": str(vod_id),
                "target_offset": offset,
                "window_start": start_time,
                "window_end": end_time,
                "message_count": len(messages),
                "messages": messages,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            total_written += 1
            time.sleep(0.5)

    print("\n--- Summary ---")
    print(f"Saved negatives: {total_written}")
    print(f"Already existed: {total_skipped}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
