import unittest

from import_live_reviews import (
    convert_live_messages,
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


if __name__ == "__main__":
    unittest.main()
