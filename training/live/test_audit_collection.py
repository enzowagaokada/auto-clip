import csv
import importlib.util
import json
import os
import tempfile
import unittest


MODULE_PATH = os.path.join(os.path.dirname(__file__), "audit_collection.py")
SPEC = importlib.util.spec_from_file_location("audit_collection", MODULE_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class CollectionAuditTest(unittest.TestCase):
    def make_partition(self, root, partition):
        directory = os.path.join(root, partition)
        os.makedirs(directory)
        sessions = []
        for index, streamer in enumerate(sorted(AUDIT.TARGET_STREAMERS)):
            sessions.append({
                "session_id": f"{partition}-session-{index}",
                "review_partition": partition,
                "streamer": streamer,
                "useful_seconds": 7200,
            })
        with open(
            os.path.join(directory, "sessions.jsonl"), "w", encoding="utf-8"
        ) as handle:
            for row in sessions:
                handle.write(json.dumps(row) + "\n")

        episodes = []
        for index in range(50):
            episodes.append({
                "episode_id": f"{partition}-episode-{index}",
                "review_partition": partition,
                "session_id": sessions[index % len(sessions)]["session_id"],
            })
        with open(
            os.path.join(directory, "episodes.jsonl"), "w", encoding="utf-8"
        ) as handle:
            for row in episodes:
                handle.write(json.dumps(row) + "\n")
        with open(
            os.path.join(directory, "episodes_review.csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "episode_id", "review_partition", "review_label"
            ])
            writer.writeheader()
            for row in episodes:
                writer.writerow({
                    "episode_id": row["episode_id"],
                    "review_partition": partition,
                    "review_label": "positive"
                    if int(row["episode_id"].rsplit("-", 1)[1]) % 2
                    else "hard_negative",
                })

    def test_ready_gate_and_partition_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.join(directory, "live")
            self.make_partition(root, "calibration")
            self.make_partition(root, "confirmation")
            with open(
                os.path.join(root, "calibration", "sessions.jsonl"),
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps({
                    "session_id": "calibration-extra",
                    "review_partition": "calibration",
                    "streamer": "stableronaldo",
                    "useful_seconds": 36000,
                }) + "\n")
            summary = AUDIT.audit_collection(
                root,
                os.path.join(directory, "missing_annotations.csv"),
                os.path.join(directory, "chat_live"),
            )
            self.assertTrue(summary["ready_for_checkpoint_4"])
            self.assertEqual(summary["total_decided_reviews"], 100)
            self.assertEqual(summary["total_useful_hours"], 16)
            self.assertEqual(summary["all_streamer_total_useful_hours"], 26)

    def test_confirmation_training_leak_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations = os.path.join(directory, "window_labels.csv")
            with open(annotations, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["review_partition", "review_identity"],
                )
                writer.writeheader()
                writer.writerow({
                    "review_partition": "confirmation",
                    "review_identity": "confirmation:episode:e",
                })
            with self.assertRaisesRegex(ValueError, "entered training"):
                AUDIT.assert_confirmation_not_in_training(
                    annotations,
                    os.path.join(directory, "chat_live"),
                )


if __name__ == "__main__":
    unittest.main()
