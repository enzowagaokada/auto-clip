# Standalone Go live shadow clipper

`cmd/autoclip` runs the exported chat model against either Twitch EventSub chat
or existing raw historical chat JSON. It is strictly shadow-only: it records
review candidates and never calls Twitch Create Clip.

Run commands from the repository root. Configuration comes from the root
`config.yaml`: `clipper` contains runtime settings and `twitch.streamers`
selects channels. If `clip_threshold` is absent, the saved threshold in
`inference_meta.json` is used. A global `clipper.clip_threshold` or per-streamer
`twitch.streamers[].clip_threshold` can override it.

`clipper.mode` is required to remain `shadow`. Startup rejects any other value;
this milestone contains no Twitch Create Clip path.

The model contract is strict: 35-second windows spanning
`[clip-start target - 5s, target + 30s]`, seven 5-second buckets, 13 features in
metadata order, case-preserving whitespace tokenization, `[SEP]` between
message records, recent-token truncation, and left padding. Set
`window_seconds: 35` and `target_lag_seconds: 30`; startup rejects other values
and stale model bundles. The production bundle defaults to
`models/exports/window-v2-vod-seed0`.

At startup, `manifest.json` is required and its SHA-256 entries for
`chat_classifier.onnx`, `vocab.json`, and `inference_meta.json` are verified
before ONNX Runtime is initialized.

Live mode loads `.env` and requires `TWITCH_CLIENT_ID` plus a
`TWITCH_USER_ACCESS_TOKEN` with `user:read:chat`. It polls Helix for stream
identity, uses one shared EventSub socket, and writes append-only candidate,
candidate-review, telemetry, episode, episode-review, and session records.
`telemetry.jsonl` contains one schema-v2 row per successful inference with
score, threshold, detector/cooldown state, raw features, model checksum, and
cumulative per-session dropped-chat count; it does not duplicate full chat.
Full candidate records contain the
model manifest checksum, raw/scaled features, and source chat; companions
`candidates_review.jsonl` and `candidates_review.csv` keep id, session_id,
streamer, score, and seek stamp (CSV also has empty `review_label`/`reason`).
Join `session_id` to `sessions.jsonl` for `vod_id` (filled on close when Helix
has the archive, or later with `python training/live/resolve_session_vods.py`).

Triggered episodes retain the highest-scoring full window and close after two
consecutive below-threshold ticks, 60 seconds, or session close. Deterministic
below-threshold local maxima are marked separately and capped at five per
useful-hour bucket. Set `clipper.review_partition` before startup; calibration
and confirmation files are physically separated and carry that partition.
Review `episodes_review.csv`; analyze threshold volume and
supported acceptance ranges with:

```powershell
python training/live/analyze_telemetry.py --partition calibration --thresholds 0.48,0.52,0.56
```

The window-v2 logs live under
`data/live/shadow/window-v2/{calibration|confirmation}/`; legacy-geometry logs
remain in their original parent directory for comparison.

Replay mode does not require Twitch credentials and never writes candidate or
session JSONL:

```powershell
go -C clipper run ./cmd/autoclip -repo .. -replay data/raw/chat/example.json
go -C clipper run ./cmd/autoclip -repo .. -replay first.json -replay second.json
```

Live mode:

```powershell
go -C clipper run ./cmd/autoclip -repo .. -config config.yaml
```

The Windows setup script installs ONNX Runtime 1.26.0, the native ABI targeted
by `onnxruntime_go` v1.31.0, after verifying the pinned archive SHA-256.
