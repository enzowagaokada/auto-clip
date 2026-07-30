"""Resolve Twitch usernames to broadcaster IDs via the Helix Get Users endpoint.

Usage (from repo root):
    python training/collect/get_broadcaster_ids.py adapt lacy marlon

Prints a ready-to-paste config.yaml snippet for each resolved user.
"""

import sys

import requests

from fetch_clips import CLIENT_ID, get_app_access_token


def get_users(logins, headers):
    """Look up up to 100 users by login name."""
    url = "https://api.twitch.tv/helix/users"
    params = [("login", login) for login in logins]
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json().get("data", [])


def main():
    logins = [arg.strip().lower() for arg in sys.argv[1:] if arg.strip()]
    if not logins:
        print("Usage: python training/collect/get_broadcaster_ids.py <name1> <name2> ...")
        return

    if not CLIENT_ID:
        print("Error: TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET not set in .env")
        return

    token = get_app_access_token()
    headers = {"Client-ID": CLIENT_ID, "Authorization": f"Bearer {token}"}

    users = get_users(logins, headers)
    found = {u["login"]: u for u in users}

    print("\n--- Results ---")
    for login in logins:
        user = found.get(login)
        if user:
            print(f"{login}: broadcaster_id = {user['id']}  (display: {user['display_name']})")
        else:
            print(f"{login}: NOT FOUND (check spelling of the exact Twitch login)")

    print("\n--- config.yaml snippet ---")
    for login in logins:
        user = found.get(login)
        if user:
            print(f'    - name: "{user["login"]}"')
            print(f'      broadcaster_id: "{user["id"]}"')
            print(f"      active: true")


if __name__ == "__main__":
    main()
