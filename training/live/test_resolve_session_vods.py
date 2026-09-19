import csv
import importlib.util
import json
import os
import tempfile
import unittest


MODULE_PATH = os.path.join(os.path.dirname(__file__), "resolve_session_vods.py")
SPEC = importlib.util.spec_from_file_location("resolve_session_vods", MODULE_PATH)
RESOLVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESOLVE)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class ResolveSessionVodsTest(unittest.TestCase):
    def test_fetch_archives_paginates_until_cursor_ends(self):
        calls = []

        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, headers=None, params=None, timeout=None):
            calls.append(params.copy())
            if params.get("after") == "c1":
                return FakeResponse(
                    {
                        "data": [{"id": "v2", "stream_id": "s2", "user_id": "u1"}],
                        "pagination": {},
                    }
                )
            return FakeResponse(
                {
                    "data": [{"id": "v1", "stream_id": "s1", "user_id": "u1"}],
                    "pagination": {"cursor": "c1"},
                }
            )

        videos = RESOLVE.fetch_archives("u1", {"Authorization": "Bearer x"}, get=fake_get)
        self.assertEqual([video["id"] for video in videos], ["v1", "v2"])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("after", calls[0])
        self.assertEqual(calls[1]["after"], "c1")

    def test_match_prefers_stream_id_and_skips_ambiguous_time(self):
        session = {
            "stream_id": "live-1",
            "broadcaster_id": "u1",
            "stream_started_at": "2026-01-01T00:00:00Z",
        }
        self.assertEqual(
            RESOLVE.match_vod(
                session,
                [
                    {"id": "old", "stream_id": "other", "user_id": "u1"},
                    {"id": "wanted", "stream_id": "live-1", "user_id": "u1"},
                ],
            ),
            "wanted",
        )
        self.assertIsNone(
            RESOLVE.match_vod(
                {
                    "stream_id": "",
                    "broadcaster_id": "u1",
                    "stream_started_at": "2026-01-01T00:00:00Z",
                },
                [
                    {
                        "id": "a",
                        "user_id": "u1",
                        "created_at": "2026-01-01T00:00:10Z",
                    },
                    {
                        "id": "b",
                        "user_id": "u1",
                        "created_at": "2026-01-01T00:00:20Z",
                    },
                ],
            )
        )
        self.assertEqual(
            RESOLVE.match_vod(
                {
                    "stream_id": "live-9",
                    "broadcaster_id": "u1",
                    "stream_started_at": "2026-01-01T00:00:00Z",
                },
                [
                    {
                        "id": "time-only",
                        "user_id": "u1",
                        "created_at": "2026-01-01T00:01:00Z",
                    }
                ],
            ),
            "time-only",
        )

    def test_resolve_partition_rewrites_one_session_row_and_keeps_csv_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration = os.path.join(directory, "calibration")
            confirmation = os.path.join(directory, "confirmation")
            os.makedirs(calibration)
            os.makedirs(confirmation)
            write_jsonl(
                os.path.join(calibration, "sessions.jsonl"),
                [
                    {
                        "session_id": "s-old",
                        "review_partition": "calibration",
                        "streamer": "arky",
                        "broadcaster_id": "u1",
                        "stream_id": "live-old",
                        "useful_seconds": 100,
                    },
                    {
                        "session_id": "s-keep",
                        "review_partition": "calibration",
                        "streamer": "arky",
                        "broadcaster_id": "u1",
                        "stream_id": "live-keep",
                        "vod_id": "already",
                        "useful_seconds": 50,
                    },
                ],
            )
            write_jsonl(
                os.path.join(confirmation, "sessions.jsonl"),
                [
                    {
                        "session_id": "s-conf",
                        "review_partition": "confirmation",
                        "streamer": "lacy",
                        "broadcaster_id": "u2",
                        "stream_id": "live-conf",
                    }
                ],
            )
            episode_csv = os.path.join(calibration, "episodes_review.csv")
            with open(episode_csv, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "episode_id",
                        "record_type",
                        "review_partition",
                        "session_id",
                        "streamer",
                        "onset_score",
                        "peak_score",
                        "stream_offset_stamp",
                        "review_label",
                        "reason",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "episode_id": "e1",
                        "record_type": "triggered",
                        "review_partition": "calibration",
                        "session_id": "s-old",
                        "streamer": "arky",
                        "onset_score": "0.5",
                        "peak_score": "0.6",
                        "stream_offset_stamp": "1h1m4s",
                        "review_label": "positive",
                        "reason": "real hype",
                    }
                )

            def fake_fetch(user_id):
                self.assertEqual(user_id, "u1")
                return [
                    {"id": "recent", "stream_id": "live-new", "user_id": "u1"},
                    {"id": "wanted", "stream_id": "live-old", "user_id": "u1"},
                ]

            result = RESOLVE.resolve_partition(calibration, fake_fetch)
            sessions = RESOLVE.load_jsonl(os.path.join(calibration, "sessions.jsonl"))
            self.assertEqual(len(sessions), 2)
            self.assertEqual(sessions[0]["vod_id"], "wanted")
            self.assertEqual(sessions[0]["useful_seconds"], 100)
            self.assertEqual(sessions[1]["vod_id"], "already")
            self.assertEqual(result["filled"], 1)
            confirmation_sessions = RESOLVE.load_jsonl(
                os.path.join(confirmation, "sessions.jsonl")
            )
            self.assertNotIn("vod_id", confirmation_sessions[0])

            with open(episode_csv, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["review_label"], "positive")
            self.assertEqual(rows[0]["reason"], "real hype")
            self.assertEqual(rows[0]["vod_id"], "wanted")
            self.assertEqual(
                rows[0]["twitch_url"],
                "https://www.twitch.tv/videos/wanted?t=1h1m4s",
            )

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sessions.jsonl")
            write_jsonl(
                path,
                [
                    {
                        "session_id": "s1",
                        "broadcaster_id": "u1",
                        "stream_id": "live-1",
                    }
                ],
            )
            result = RESOLVE.resolve_partition(
                directory,
                lambda user_id: [{"id": "v1", "stream_id": "live-1", "user_id": "u1"}],
                dry_run=True,
            )
            self.assertEqual(result["filled"], 1)
            self.assertNotIn("vod_id", RESOLVE.load_jsonl(path)[0])


if __name__ == "__main__":
    unittest.main()
