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

    ended_at = datetime.now(timezone.utc)
    started_at = ended_at - timedelta(days=days_back)
    started_at_str = started_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    ended_at_str = ended_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    print("Authenticating with Twitch...")
    token = get_app_access_token()
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {token}",
    }

    print(
        f"Fetching up to {max_per_streamer} clips per streamer "
        f"from the last {days_back} days ({started_at_str} to {ended_at_str})..."
    )

    all_clips = []

    for streamer in config["twitch"]["streamers"]:
        if not streamer.get("active", False):
            continue

        name = streamer["name"]
        broadcaster_id = streamer["broadcaster_id"]

        print(f"Fetching clips for {name} ({broadcaster_id})...")
        try:
            clips = fetch_clips_in_range(
                broadcaster_id,
                headers,
                started_at_str,
                ended_at_str,
                max_clips=max_per_streamer,
            )
            kept_before = len(all_clips)
            for clip in clips:
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
            print(f"  -> Found {len(clips)} clips, kept {kept} with VOD data.")
        except requests.exceptions.RequestException as e:
            print(f"  -> Error fetching clips for {name}: {e}")

    if all_clips:
        df = pd.DataFrame(all_clips)
        output_file = os.path.join(raw_dir, "clips.csv")
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(all_clips)} total clips to {output_file}")
    else:
        print("\nNo clips with VOD data found.")


if __name__ == "__main__":
    main()
