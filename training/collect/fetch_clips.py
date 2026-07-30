import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")


def get_app_access_token():
    """Fetches an App Access Token from Twitch."""
    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    response = requests.post(url, params=params)
    response.raise_for_status()
    return response.json()["access_token"]


def build_date_windows(now, days_back, window_days):
    """Split the last `days_back` days into consecutive windows of `window_days`.

    Fetching top clips per small window yields far more unique clips than one big
    range (Twitch returns top-by-views per range, and deep pagination is unreliable).
    Returns a list of (start, end) datetime tuples, most recent last.
    """
    if window_days <= 0:
        window_days = days_back

    windows = []
    span_start = now - timedelta(days=days_back)
    cursor = span_start
    while cursor < now:
        win_end = min(cursor + timedelta(days=window_days), now)
        windows.append((cursor, win_end))
        cursor = win_end
    return windows


def fetch_clips_in_range(broadcaster_id, headers, started_at, ended_at, max_clips=100):
    """Fetches top clips for a broadcaster within a date range."""
    url = "https://api.twitch.tv/helix/clips"
    all_clips = []
    cursor = None

    while len(all_clips) < max_clips:
        params = {
            "broadcaster_id": broadcaster_id,
            "first": min(100, max_clips - len(all_clips)),
            "started_at": started_at,
            "ended_at": ended_at,
        }
        if cursor:
            params["after"] = cursor

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        payload = response.json()
        clips = payload.get("data", [])
        if not clips:
            break

        all_clips.extend(clips)
        cursor = payload.get("pagination", {}).get("cursor")
        if not cursor:
            break

    return all_clips


def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("Error: TWITCH_CLIENT_ID or TWITCH_CLIENT_SECRET not set in environment.")
        return

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["data"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    clips_config = config["twitch"].get("clips", {})
    days_back = clips_config.get("days_back", 30)
    max_per_streamer = clips_config.get("max_per_streamer", 100)
    window_days = clips_config.get("window_days", days_back)

    now = datetime.now(timezone.utc)
    windows = build_date_windows(now, days_back, window_days)

    print("Authenticating with Twitch...")
    token = get_app_access_token()
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {token}",
    }

    print(
        f"Fetching up to {max_per_streamer} clips per {window_days}-day window "
        f"across the last {days_back} days ({len(windows)} windows per streamer)..."
    )

    all_clips = []

    for streamer in config["twitch"]["streamers"]:
        if not streamer.get("active", False):
            continue

        name = streamer["name"]
        broadcaster_id = streamer["broadcaster_id"]
        print(f"Fetching clips for {name} ({broadcaster_id})...")

        kept_before = len(all_clips)
        seen_ids = set()

        for win_start, win_end in windows:
            started_at_str = win_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            ended_at_str = win_end.strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                clips = fetch_clips_in_range(
                    broadcaster_id,
                    headers,
                    started_at_str,
                    ended_at_str,
                    max_clips=max_per_streamer,
                )
            except requests.exceptions.RequestException as e:
                print(f"  -> Error for window {started_at_str[:10]}..{ended_at_str[:10]}: {e}")
                continue

            for clip in clips:
                if clip["id"] in seen_ids:
                    continue
                seen_ids.add(clip["id"])
                if clip.get("video_id") and clip.get("vod_offset") is not None:
                    all_clips.append(
                        {
                            "streamer_name": name,
                            "clip_id": clip["id"],
                            "vod_id": clip["video_id"],
                            "vod_offset": clip["vod_offset"],
                            "view_count": clip["view_count"],
                            "created_at": clip["created_at"],
                            "duration": clip["duration"],
                        }
                    )

        kept = len(all_clips) - kept_before
        print(f"  -> Kept {kept} clips with VOD data across {len(windows)} windows.")

    if all_clips:
        output_file = os.path.join(raw_dir, "clips.csv")
        new_df = pd.DataFrame(all_clips)

        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            existing_count = len(existing_df)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["clip_id"], keep="first")
            added_count = len(combined_df) - existing_count
        else:
            combined_df = new_df
            added_count = len(combined_df)

        combined_df.to_csv(output_file, index=False)

        print(f"\nSaved {len(combined_df)} total unique clips to {output_file}")
        print(f"Added {added_count} new clips")
    else:
        print("\nNo clips with VOD data found.")


if __name__ == "__main__":
    main()
