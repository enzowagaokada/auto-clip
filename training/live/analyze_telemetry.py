"""Analyze schema-v1 live inference telemetry and episode reviews.

Run from the repository root:
    python training/live/analyze_telemetry.py
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import yaml


SCHEMA_VERSION = 1
DECIDED_LABELS = {"positive", "hard_negative", "negative"}


def parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def read_jsonl(path, kind):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("schema_version", 0)) != SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{line_number}: unsupported {kind} schema_version "
                    f"{row.get('schema_version')!r}"
                )
            rows.append(row)
    return rows


def validate_telemetry(rows):
    required = {
        "session_id", "streamer", "model_manifest_sha256", "inference_at",
        "target_at", "score", "threshold", "triggered", "raw_features",
        "cumulative_dropped_chat",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"telemetry row {index} is missing {missing}")
        if len(row["raw_features"]) != 13:
            raise ValueError(f"telemetry row {index} must contain 13 raw features")
        score = float(row["score"])
        if not 0 <= score <= 1:
            raise ValueError(f"telemetry row {index} score is outside [0, 1]")


def load_reviews(path):
    reviews = {}
    if not os.path.exists(path):
        return reviews
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            episode_id = row.get("episode_id", "").strip()
            label = row.get("review_label", "").strip().lower()
            if episode_id and label:
                reviews[episode_id] = label
    return reviews


def detector_decisions(rows, threshold, cooldown_seconds):
    above = False
    armed = True
    cooldown_until = datetime.min.replace(tzinfo=timezone.utc)
    decisions = []
    for row in rows:
        at = parse_time(row["inference_at"])
        score = float(row["score"])
        is_above = score >= threshold
        crossed = is_above and not above
        if not is_above:
            armed = True
        triggered = crossed and armed and at >= cooldown_until
        if triggered:
            armed = False
            cooldown_until = at + timedelta(seconds=cooldown_seconds)
        above = is_above
        decisions.append((row, at, score, triggered))
    return decisions


def replay_session(rows, threshold, cooldown_seconds, close_below_ticks=2, max_seconds=60):
    rows = sorted(rows, key=lambda row: parse_time(row["inference_at"]))
    episodes = []
    active = None
    for row, at, score, triggered in detector_decisions(rows, threshold, cooldown_seconds):
        if active is None and triggered:
            active = {
                "onset_at": at,
                "onset_score": score,
                "peak_at": at,
                "peak_score": score,
                "minimum_score": score,
                "maximum_score": score,
                "below_ticks": 0,
            }
        if active is None:
            continue
        if score > active["peak_score"]:
            active["peak_score"] = score
            active["peak_at"] = at
        active["minimum_score"] = min(active["minimum_score"], score)
        active["maximum_score"] = max(active["maximum_score"], score)
        active["below_ticks"] = active["below_ticks"] + 1 if score < threshold else 0
        reason = None
        if active["below_ticks"] >= close_below_ticks:
            reason = "consecutive_below_threshold"
        elif (at - active["onset_at"]).total_seconds() >= max_seconds:
            reason = "maximum_duration"
        if reason:
            active["closed_at"] = at
            active["close_reason"] = reason
            episodes.append(active)
            active = None
    if active is not None:
        active["closed_at"] = parse_time(rows[-1]["inference_at"])
        active["close_reason"] = "telemetry_end"
        episodes.append(active)
    return episodes


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def percentile_interval(values):
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    low, high = np.percentile(finite, [2.5, 97.5])
    return [float(low), float(high)]


def score_distribution(values):
    scores = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(scores)),
        "median": float(np.median(scores)),
        "p90": float(np.percentile(scores, 90)),
        "maximum": float(np.max(scores)),
    }


def bootstrap_acceptance(labels, rng, samples):
    if not labels:
        return None
    values = np.asarray([label == "positive" for label in labels], dtype=np.float64)
    estimates = [
        float(rng.choice(values, size=len(values), replace=True).mean())
        for _ in range(samples)
    ]
    return percentile_interval(estimates)


def bootstrap_auc(labels, scores, rng, samples):
    if len(labels) < 2:
        return None
    estimates = []
    indices = np.arange(len(labels))
    for _ in range(samples):
        sample = rng.choice(indices, size=len(indices), replace=True)
        estimates.append(roc_auc(np.asarray(labels)[sample], np.asarray(scores)[sample]))
    return percentile_interval(estimates)


def reviewed_metrics(episode_rows, reviews, threshold, active_threshold, rng, samples):
    selected = [
        row for row in episode_rows
        if float(row["peak_score"]) >= threshold and reviews.get(row["episode_id"]) in DECIDED_LABELS
    ]
    labels = [
        "hard_negative" if reviews[row["episode_id"]] == "negative" else reviews[row["episode_id"]]
        for row in selected
    ]
    binary = [1 if label == "positive" else 0 for label in labels]
    scores = [float(row["peak_score"]) for row in selected]
    local_support = sum(
        1 for row in selected
        if row.get("record_type") == "local_maximum" and float(row["peak_score"]) < active_threshold
    )
    supported = threshold >= active_threshold or local_support >= 5
    return {
        "decided_reviews": len(labels),
        "positives": sum(binary),
        "hard_negatives": len(binary) - sum(binary),
        "acceptance_supported": supported,
        "acceptance": float(np.mean(binary)) if labels and supported else None,
        "acceptance_95_ci": bootstrap_acceptance(labels, rng, samples) if supported else None,
        "peak_score_roc_auc": roc_auc(binary, scores),
        "peak_score_roc_auc_95_ci": bootstrap_auc(binary, scores, rng, samples),
        "below_threshold_local_peak_support": local_support,
    }


def session_hours(session_rows, streamer=None):
    selected = [
        row for row in session_rows
        if streamer is None or row.get("streamer") == streamer
    ]
    return sum(float(row.get("useful_seconds", 0)) for row in selected) / 3600


def rate_interval(session_rows, counts, rng, samples, streamer=None):
    selected = [
        row for row in session_rows
        if streamer is None or row.get("streamer") == streamer
    ]
    if not selected:
        return None
    estimates = []
    for _ in range(samples):
        sampled = rng.choice(len(selected), size=len(selected), replace=True)
        hours = sum(float(selected[i].get("useful_seconds", 0)) for i in sampled) / 3600
        episodes = sum(counts.get(selected[i]["session_id"], 0) for i in sampled)
        if hours > 0:
            estimates.append(episodes / hours)
    return percentile_interval(estimates)


def analyze(telemetry, sessions, episode_rows, reviews, thresholds, cooldown_seconds, samples, seed):
    validate_telemetry(telemetry)
    if not telemetry:
        raise ValueError("telemetry input is empty")
    by_session = defaultdict(list)
    for row in telemetry:
        by_session[row["session_id"]].append(row)
    telemetry_session_ids = set(by_session)
    sessions = [
        row for row in sessions if row.get("session_id") in telemetry_session_ids
    ]
    episode_rows = [
        row for row in episode_rows
        if row.get("session_id") in telemetry_session_ids
    ]
    active_threshold = max(float(row["threshold"]) for row in telemetry)
    rng = np.random.default_rng(seed)
    results = []
    streamers = sorted({row["streamer"] for row in telemetry})
    for threshold in thresholds:
        replayed = {
            session_id: replay_session(
                rows,
                threshold,
                cooldown_seconds.get(rows[0]["streamer"], cooldown_seconds["*"])
                if isinstance(cooldown_seconds, dict)
                else cooldown_seconds,
            )
            for session_id, rows in by_session.items()
        }
        counts = {session_id: len(rows) for session_id, rows in replayed.items()}
        hours = session_hours(sessions)
        dropped = sum(int(row.get("dropped_chat", 0)) for row in sessions)
        seen = sum(int(row.get("messages_seen", 0)) for row in sessions)
        global_result = {
            "useful_streamer_hours": hours,
            "episodes": sum(counts.values()),
            "episodes_per_hour": sum(counts.values()) / hours if hours else None,
            "episodes_per_hour_95_ci": rate_interval(sessions, counts, rng, samples),
            "dropped_chat": dropped,
            "dropped_chat_rate": dropped / (seen + dropped) if seen + dropped else 0.0,
            **reviewed_metrics(
                episode_rows, reviews, threshold, active_threshold, rng, samples
            ),
        }
        per_streamer = {}
        for streamer in streamers:
            streamer_sessions = {
                row["session_id"] for row in sessions if row.get("streamer") == streamer
            }
            streamer_counts = {
                session_id: count for session_id, count in counts.items()
                if session_id in streamer_sessions
            }
            streamer_hours = session_hours(sessions, streamer)
            streamer_episode_rows = [
                row for row in episode_rows if row.get("streamer") == streamer
            ]
            streamer_telemetry = [
                row for row in telemetry if row.get("streamer") == streamer
            ]
            streamer_active_threshold = max(
                float(row["threshold"]) for row in streamer_telemetry
            )
            session_subset = [
                row for row in sessions if row.get("streamer") == streamer
            ]
            dropped = sum(int(row.get("dropped_chat", 0)) for row in session_subset)
            seen = sum(int(row.get("messages_seen", 0)) for row in session_subset)
            episode_count = sum(streamer_counts.values())
            per_streamer[streamer] = {
                "active_threshold": streamer_active_threshold,
                "useful_streamer_hours": streamer_hours,
                "episodes": episode_count,
                "episodes_per_hour": episode_count / streamer_hours if streamer_hours else None,
                "episodes_per_hour_95_ci": rate_interval(
                    sessions, streamer_counts, rng, samples, streamer
                ),
                "dropped_chat": dropped,
                "dropped_chat_rate": dropped / (seen + dropped) if seen + dropped else 0.0,
                "score_distribution": score_distribution(
                    [float(row["score"]) for row in streamer_telemetry]
                ),
                **reviewed_metrics(
                    streamer_episode_rows, reviews, threshold,
                    streamer_active_threshold, rng, samples,
                ),
            }
        results.append({
            "threshold": threshold,
            "active_threshold": active_threshold,
            "global": global_result,
            "per_streamer": per_streamer,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "telemetry_rows": len(telemetry),
        "sessions": len(sessions),
        "score_distribution": score_distribution(
            [float(row["score"]) for row in telemetry]
        ),
        "threshold_results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", default="data/live/shadow/window-v2/telemetry.jsonl")
    parser.add_argument("--sessions", default="data/live/shadow/window-v2/sessions.jsonl")
    parser.add_argument("--episodes", default="data/live/shadow/window-v2/episodes.jsonl")
    parser.add_argument(
        "--reviews", default="data/live/shadow/window-v2/episodes_review.csv"
    )
    parser.add_argument("--thresholds", default=None, help="Comma-separated scores")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=None,
        help="Override the configured global and per-streamer cooldowns.",
    )
    parser.add_argument(
        "--streamer-cooldowns",
        default="",
        help="Optional comma-separated overrides such as arky=60,lacy=90",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", default="data/live/shadow/window-v2/telemetry_analysis.json"
    )
    args = parser.parse_args()
    telemetry = read_jsonl(args.telemetry, "telemetry")
    sessions = []
    with open(args.sessions, "r", encoding="utf-8") as handle:
        sessions = [json.loads(line) for line in handle if line.strip()]
    episodes = read_jsonl(args.episodes, "episode")
    reviews = load_reviews(args.reviews)
    if args.thresholds:
        thresholds = [float(value) for value in args.thresholds.split(",")]
    else:
        thresholds = sorted({float(row["threshold"]) for row in telemetry})
    if not telemetry:
        raise ValueError("telemetry input is empty")
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("all thresholds must be in [0, 1]")
    if args.cooldown_seconds is not None and args.cooldown_seconds < 0:
        raise ValueError("--cooldown-seconds must be non-negative")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    configured_global = int(config.get("clipper", {}).get("cooldown_seconds", 75))
    if configured_global < 0:
        raise ValueError("configured clipper.cooldown_seconds must be non-negative")
    cooldowns = {
        "*": args.cooldown_seconds
        if args.cooldown_seconds is not None
        else configured_global
    }
    if args.cooldown_seconds is None:
        for streamer in config.get("twitch", {}).get("streamers", []):
            if streamer.get("cooldown_seconds") is not None:
                value = int(streamer["cooldown_seconds"])
                if value < 0:
                    raise ValueError(
                        f"configured cooldown for {streamer['name']} must be non-negative"
                    )
                cooldowns[str(streamer["name"])] = value
    for item in filter(None, args.streamer_cooldowns.split(",")):
        name, separator, value = item.partition("=")
        if not separator or not name.strip() or int(value) < 0:
            raise ValueError("invalid --streamer-cooldowns entry")
        cooldowns[name.strip()] = int(value)
    summary = analyze(
        telemetry, sessions, episodes, reviews, thresholds,
        cooldowns, args.bootstrap_samples, args.seed,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Analyzed {summary['telemetry_rows']} telemetry rows")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
