import csv
import json
import os
import tempfile
import unittest

from import_live_reviews import (
    convert_live_messages,
    episodes_by_id,
    import_live_reviews,
    live_window_record,
    offset_to_stamp,
    parse_datetime,
    twitch_url,
)
from window_geometry import WINDOW_GEOMETRY_NAME, WINDOW_GEOMETRY_VERSION, has_current_geometry


class ImportLiveReviewsTest(unittest.TestCase):
    def test_parse_datetime_truncates_extra_fraction(self):
        parsed = parse_datetime("2026-08-07T23:04:12.0310019Z")
        self.assertEqual(parsed.tzinfo.utcoffset(parsed).total_seconds(), 0)
        self.assertEqual(parsed.microsecond, 31001)

    def test_message_offsets_use_target_clock(self):
        messages = convert_live_messages(
            [
                {
                    "time": "2026-08-07T23:04:07.0310019Z",
                    "user": "alice",
                    "text": "SON",
                },
                {
                    "time": "2026-08-07T23:04:42.0310019Z",
                    "user": "bob",
                    "text": "LOL",
                },
            ],
            target_offset=3664,
            target_at="2026-08-07T23:04:12.0310019Z",
        )
        self.assertEqual(messages[0]["message"], "SON")
        self.assertAlmostEqual(messages[0]["offset_seconds"], 3659.0, places=3)
        self.assertAlmostEqual(messages[1]["offset_seconds"], 3694.0, places=3)

    def test_live_window_has_current_geometry(self):
        candidate = {
            "candidate_id": "abc",
            "target_at": "2026-08-07T23:04:12.0310019Z",
            "messages": [
                {
                    "time": "2026-08-07T23:04:12.0310019Z",
                    "user": "alice",
                    "text": "SON",
                }
            ],
        }
        record = live_window_record(candidate, "2840042393", 3664, "jasontheween", 1)
        self.assertTrue(has_current_geometry(record, expected_target=3664))
        self.assertEqual(record["window_geometry"], WINDOW_GEOMETRY_NAME)
        self.assertEqual(record["window_geometry_version"], WINDOW_GEOMETRY_VERSION)
        self.assertEqual(record["window_start"], 3659.0)
        self.assertEqual(record["window_end"], 3694.0)

    def test_twitch_url_uses_stamp(self):
        self.assertEqual(
            twitch_url("2840042393", stamp="1h1m4s"),
            "https://www.twitch.tv/videos/2840042393?t=1h1m4s",
        )
        self.assertEqual(offset_to_stamp(3664), "1h1m4s")

    def write_episode_fixture(self, directory, partition="calibration"):
        episode = {
            "schema_version": 2,
            "episode_id": "episode-1",
            "record_type": "triggered",
            "review_partition": partition,
            "session_id": "session-1",
            "streamer": "arky",
            "model_manifest_sha256": "manifest",
            "peak_target_at": "2026-09-06T20:00:00Z",
            "peak_window_start": "2026-09-06T19:59:55Z",
            "peak_window_end": "2026-09-06T20:00:30Z",
            "peak_stream_offset_seconds": 123,
            "peak_score": 0.81,
            "messages": [
                {
                    "time": "2026-09-06T19:59:58Z",
                    "user": "alice",
                    "text": "NO WAY",
                }
            ],
        }
        episodes_path = os.path.join(directory, "episodes.jsonl")
        with open(episodes_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(episode) + "\n")
        sessions_path = os.path.join(directory, "sessions.jsonl")
        with open(sessions_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "session_id": "session-1",
                "review_partition": partition,
                "vod_id": "999",
            }) + "\n")
        review_path = os.path.join(directory, "episodes_review.csv")
        with open(review_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
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
            ])
            writer.writeheader()
            writer.writerow({
                "episode_id": "episode-1",
                "record_type": "triggered",
                "review_partition": partition,
                "session_id": "session-1",
                "streamer": "arky",
                "peak_score": "0.81",
                "stream_offset_stamp": "0h2m3s",
                "review_label": "positive",
                "reason": "real moment",
            })
        return episodes_path, sessions_path, review_path

    def test_episode_import_materializes_peak_with_review_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            episodes_path, sessions_path, review_path = self.write_episode_fixture(
                directory
            )
            output = os.path.join(directory, "window_labels.csv")
            live_dir = os.path.join(directory, "chat_live")
            result = import_live_reviews(
                partition="calibration",
                input_kind="episode",
                review_file=review_path,
                records_file=episodes_path,
                sessions_file=sessions_path,
                output=output,
                live_dir=live_dir,
                source_run="checkpoint-3",
            )
            self.assertEqual(result["windows_written"], 1)
            with open(
                os.path.join(live_dir, "999_123.json"),
                "r",
                encoding="utf-8",
            ) as handle:
                record = json.load(handle)
            self.assertTrue(has_current_geometry(record, expected_target=123))
            self.assertEqual(record["messages"][0]["offset_seconds"], 121)
            self.assertEqual(record["episode_id"], "episode-1")
            self.assertEqual(record["review_partition"], "calibration")
            self.assertEqual(
                record["review_identity"], "calibration:episode:episode-1"
            )
            self.assertEqual(record["source"], "checkpoint-3-episode-peak")
            self.assertEqual(
                record["peak_target_at"], "2026-09-06T20:00:00Z"
            )
            with open(output, "r", encoding="utf-8", newline="") as handle:
                annotation = next(csv.DictReader(handle))
            self.assertEqual(annotation["review_partition"], "calibration")
            self.assertEqual(
                annotation["review_identity"], "calibration:episode:episode-1"
            )

    def test_confirmation_partition_is_validation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            episodes_path, sessions_path, review_path = self.write_episode_fixture(
                directory,
                partition="confirmation",
            )
            arguments = {
                "partition": "confirmation",
                "input_kind": "episode",
                "review_file": review_path,
                "records_file": episodes_path,
                "sessions_file": sessions_path,
                "output": os.path.join(directory, "window_labels.csv"),
                "live_dir": os.path.join(directory, "chat_live"),
                "source_run": "checkpoint-3",
            }
            with self.assertRaisesRegex(ValueError, "cannot be imported"):
                import_live_reviews(**arguments)
            result = import_live_reviews(**arguments, validate_only=True)
            self.assertEqual(result["imported"]["positive"], 1)
            self.assertFalse(os.path.exists(arguments["output"]))
            self.assertFalse(os.path.exists(arguments["live_dir"]))

    def test_episode_schema_and_partition_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            episodes_path, _, _ = self.write_episode_fixture(directory)
            with open(episodes_path, "r", encoding="utf-8") as handle:
                episode = json.loads(handle.readline())
            episode["schema_version"] = 99
            with open(episodes_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(episode) + "\n")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                episodes_by_id(episodes_path, "calibration")


if __name__ == "__main__":
    unittest.main()
