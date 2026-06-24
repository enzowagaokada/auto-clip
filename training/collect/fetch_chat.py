import os
import json
import time
import pandas as pd
import requests

# The standard Twitch Web Client ID used for public GraphQL requests
TWITCH_GQL_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
TWITCH_GQL_URL = "https://gql.twitch.tv/gql"

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
            response = requests.post(TWITCH_GQL_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

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

            # Be polite to the undocumented API
            time.sleep(0.1)

        except Exception as e:
            print(f"    Error fetching chunk: {e}")
            break

    return messages

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
    print(f"Found {total_clips} clips to process.")
    
    fetched_count = 0
    skipped_count = 0
    error_count = 0

    for index, row in df.iterrows():
        clip_id = row['clip_id']
        vod_id = row['vod_id']
        vod_offset = row['vod_offset']
        
        output_file = os.path.join(chat_dir, f"{clip_id}.json")
        
        # Skip if we already downloaded this chat window (resumability)
        if os.path.exists(output_file):
            skipped_count += 1
            continue
            
        print(f"[{index + 1}/{total_clips}] Fetching chat for Clip: {clip_id} (VOD: {vod_id}, Offset: {vod_offset}s)")
        
        # Target window: 30 seconds before the clip starts, up to 5 seconds into it
        start_time = max(0, vod_offset - 30)
        end_time = vod_offset + 5
        
        messages = fetch_chat_window(vod_id, start_time, end_time)
        
        if messages:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "clip_id": clip_id,
                    "vod_id": vod_id,
                    "target_offset": vod_offset,
                    "window_start": start_time,
                    "window_end": end_time,
                    "message_count": len(messages),
                    "messages": messages
                }, f, indent=2, ensure_ascii=False)
            fetched_count += 1
        else:
            print(f"    Warning: No messages found in window for clip {clip_id}")
            error_count += 1
            
        # Rate limiting protection
        time.sleep(0.5)

    print("\n--- Summary ---")
    print(f"Successfully fetched: {fetched_count}")
    print(f"Already existed (skipped): {skipped_count}")
    print(f"No messages / Errors: {error_count}")

if __name__ == "__main__":
    main()