import os
import requests
import yaml
import pandas as pd
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
        "grant_type": "client_credentials"
    }
    response = requests.post(url, params=params)
    response.raise_for_status()
    return response.json()["access_token"]

def fetch_top_clips(broadcaster_id, headers, first=100):
    """Fetches the top clips for a given broadcaster ID."""
    url = "https://api.twitch.tv/helix/clips"
    params = {
        "broadcaster_id": broadcaster_id,
        "first": first
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json().get("data", [])

def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("Error: TWITCH_CLIENT_ID or TWITCH_CLIENT_SECRET not set in environment.")
        return

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Setup directories
    raw_dir = config["data"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    print("Authenticating with Twitch...")
    token = get_app_access_token()
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {token}"
    }

    all_clips = []

    for streamer in config["twitch"]["streamers"]:
        if not streamer.get("active", False):
            continue
            
        name = streamer["name"]
        broadcaster_id = streamer["broadcaster_id"]
        
        print(f"Fetching clips for {name} ({broadcaster_id})...")
        try:
            clips = fetch_top_clips(broadcaster_id, headers)
            for clip in clips:
                # We only want clips that are attached to a VOD (video_id) so we can fetch chat
                if clip.get("video_id") and clip.get("vod_offset") is not None:
                    all_clips.append({
                        "streamer_name": name,
                        "clip_id": clip["id"],
                        "vod_id": clip["video_id"],
                        "vod_offset": clip["vod_offset"],
                        "view_count": clip["view_count"],
                        "created_at": clip["created_at"],
                        "duration": clip["duration"]
                    })
            print(f"  -> Found {len(clips)} clips, kept {len([c for c in all_clips if c['streamer_name'] == name])} with VOD data.")
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