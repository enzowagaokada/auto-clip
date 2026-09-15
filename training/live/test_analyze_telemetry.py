import importlib.util
import json
import os
import tempfile
import unittest


MODULE_PATH = os.path.join(os.path.dirname(__file__), "analyze_telemetry.py")
SPEC = importlib.util.spec_from_file_location("analyze_telemetry", MODULE_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def telemetry_row(session_id, streamer, second, score, threshold=0.5):
    stamp = f"2026-01-01T00:00:{second:02d}+00:00"
    return {
        "schema_version": 1,
        "session_id": session_id,
        "streamer": streamer,
        "model_manifest_sha256": "abc",
        "inference_at": stamp,
        "target_at": stamp,
        "score": score,
        "threshold": threshold,
        "triggered": False,
        "raw_features": [0.0] * 13,
        "cumulative_dropped_chat": 0,
    }


class TelemetryAnalyzerTest(unittest.TestCase):
    def test_committed_episode_fixture(self):
        fixture_dir = os.path.join(os.path.dirname(__file__), "testdata")
        rows = ANALYZER.read_jsonl(
            os.path.join(fixture_dir, "telemetry_episode.jsonl"),
            "telemetry",
        )
        with open(
            os.path.join(fixture_dir, "expected_episode.json"),
            "r",
            encoding="utf-8",
        ) as handle:
            expected = json.load(handle)
        episodes = ANALYZER.replay_session(rows, 0.5, 75)
        self.assertEqual(len(episodes), expected["episode_count"])
        self.assertEqual(episodes[0]["onset_score"], expected["onset_score"])
        self.assertEqual(episodes[0]["peak_score"], expected["peak_score"])
        self.assertEqual(episodes[0]["close_reason"], expected["close_reason"])

    def test_replay_peak_differs_from_onset_and_closes(self):
        rows = [
            telemetry_row("s", "arky", 0, 0.4),
            telemetry_row("s", "arky", 1, 0.51),
            telemetry_row("s", "arky", 2, 0.9),
            telemetry_row("s", "arky", 3, 0.49),
            telemetry_row("s", "arky", 4, 0.48),
        ]
        episodes = ANALYZER.replay_session(rows, 0.5, 75)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["onset_score"], 0.51)
        self.assertEqual(episodes[0]["peak_score"], 0.9)
        self.assertEqual(
            episodes[0]["close_reason"], "consecutive_below_threshold"
        )

    def test_replay_respects_cooldown_and_rearm(self):
        rows = [
            telemetry_row("s", "arky", 0, 0.6),
            telemetry_row("s", "arky", 1, 0.4),
            telemetry_row("s", "arky", 2, 0.4),
            telemetry_row("s", "arky", 3, 0.7),
            telemetry_row("s", "arky", 4, 0.4),
            telemetry_row("s", "arky", 5, 0.4),
        ]
        self.assertEqual(len(ANALYZER.replay_session(rows, 0.5, 10)), 1)
        self.assertEqual(len(ANALYZER.replay_session(rows, 0.5, 0)), 2)

    def test_analysis_is_deterministic_and_guards_acceptance_support(self):
        telemetry = [
            telemetry_row("s1", "arky", 0, 0.4),
            telemetry_row("s1", "arky", 1, 0.6),
            telemetry_row("s1", "arky", 2, 0.8),
            telemetry_row("s1", "arky", 3, 0.4),
            telemetry_row("s1", "arky", 4, 0.4),
        ]
        sessions = [{
            "session_id": "s1",
            "streamer": "arky",
            "useful_seconds": 3600,
            "messages_seen": 100,
            "dropped_chat": 2,
        }]
        episodes = [
            {
                "schema_version": 1,
                "episode_id": "positive",
                "record_type": "triggered",
                "session_id": "s1",
                "streamer": "arky",
                "peak_score": 0.8,
            },
            {
                "schema_version": 1,
                "episode_id": "negative",
                "record_type": "triggered",
                "session_id": "s1",
                "streamer": "arky",
                "peak_score": 0.6,
            },
        ]
        reviews = {"positive": "positive", "negative": "hard_negative"}
        first = ANALYZER.analyze(
            telemetry, sessions, episodes, reviews, [0.4, 0.5], 75, 50, 7
        )
        second = ANALYZER.analyze(
            telemetry, sessions, episodes, reviews, [0.4, 0.5], 75, 50, 7
        )
        self.assertEqual(first, second)
        low = first["threshold_results"][0]["global"]
        active = first["threshold_results"][1]["global"]
        self.assertFalse(low["acceptance_supported"])
        self.assertIsNone(low["acceptance"])
        self.assertTrue(active["acceptance_supported"])
        self.assertEqual(active["peak_score_roc_auc"], 1.0)

    def test_schema_validation_rejects_wrong_feature_count(self):
        row = telemetry_row("s", "arky", 0, 0.5)
        row["raw_features"] = [0.0]
        with self.assertRaisesRegex(ValueError, "13 raw features"):
            ANALYZER.validate_telemetry([row])

    def test_schema_v2_requires_review_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "telemetry.jsonl")
            row = telemetry_row("s", "arky", 0, 0.5)
            row["schema_version"] = 2
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "review_partition"):
                ANALYZER.read_jsonl(path, "telemetry")


if __name__ == "__main__":
    unittest.main()
