# ONNX and Go Shadow Clipper Runbook

Run every command from the repository root unless the command uses `go -C
clipper`. The Go application is hard-gated to `clipper.mode: shadow` and has no
Twitch Create Clip implementation.

> **Window-v2 is active:** keep using `models/exports/window-v2-vod-seed0/`
> with the `[target - 5s, target + 30s]` contract. Legacy five-second-lag logs
> remain historical evidence only.

## 1. Rebuild the clip-start-aligned model

The fetchers automatically replace stale raw windows in place. Run the full
pipeline from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python training/collect/fetch_clips.py
python training/collect/fetch_chat.py
python training/collect/fetch_negatives.py
python training/collect/build_dataset.py
python training/features/encode.py
python training/model/train.py --output-dir models/runs/window-v2-vod-seed0
```

`fetch_clips.py` is not required merely to migrate existing clip IDs, but it is
recommended here because expired VODs may make some legacy windows unavailable;
fresh clips replace some of that lost training coverage.

The fixed v2 contract is `[clip start - 5s, clip start + 30s]`, seven
five-second buckets, and a 30-second live target lag. `build_dataset.py` rejects
stale raw files instead of silently mixing geometries. When an expired VOD or
empty replacement prevents migration, the fetcher preserves the old file with
a `.window-v1-stale` suffix; it is not included in the rebuilt dataset.

## 2. Export and verify the model

```powershell
python training/export/export_onnx.py --overwrite
python training/export/verify_onnx.py
python training/export/verify_go_fixture.py
```

The first command creates `models/exports/window-v2-vod-seed0/`. The parity
command must finish with zero threshold-decision mismatches. The fixture check
proves the committed Go tokenizer/feature expectations came from the Python
pipeline.

Window-v2 verification passed on 2026-08-07 across all 3,078 saved validation
rows: zero logit/threshold-decision mismatches, maximum absolute logit error
`2.38e-7`, Python/Go preprocessing parity passed, and the Go executable built
successfully. Positive/negative historical replay also passed. The first
authenticated window-v2 shadow session also passed: JasonTheWeen, about
2 minutes 27 seconds, 386 messages, 45 inferences, 1 candidate at score
`0.5053` / threshold `0.480`, zero inference errors.

## 3. Install the native ONNX Runtime library

The Go binding and setup script are pinned to ONNX Runtime 1.26.0. The script
verifies Microsoft's published SHA-256 before extracting the DLL.

```powershell
powershell -ExecutionPolicy Bypass -File .\clipper\scripts\setup-onnxruntime-windows.ps1
```

The generated DLL is stored at `clipper/runtime/onnxruntime.dll` and is ignored
by Git.

## 4. Verify and build the Go module

The Go clipper uses CGO (ONNX Runtime). On Windows that means a C compiler
(`gcc`) is required for **normal** `go test` / `go build` as well as
`go test -race` — plain PowerShell without `gcc` on `PATH` fails.

Install [MSYS2](https://www.msys2.org/), open an **MSYS2 UCRT64** shell, install
the toolchain, then run the Go commands from that shell (or any terminal where
`gcc` is on `PATH`):

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc
cd /c/Users/<you>/Documents/auto-clip

export PATH="/c/Program Files/Go/bin:$PATH"
go -C clipper mod tidy
go -C clipper fmt ./...
go -C clipper test ./...
go -C clipper test -race ./...
go -C clipper build ./cmd/autoclip
```

These are the terminal checks reserved for the user. The former window contract
passed normal/race tests, replay, and a live smoke test on 2026-08-04. All checks
must be rerun for window v2. The window-v2 Python unit tests and Go normal/race
tests were reported passing on 2026-08-06; replay remains blocked on the new
data, trained model, and export.

## 5. Replay a historical window

Replay uses the exported ONNX model but needs no Twitch credentials and writes
no shadow logs:

Run at least one historical positive and one historical negative:

In Powershell
```powershell
$PositiveReplay = (Get-ChildItem .\data\raw\chat -Filter *.json | Select-Object -First 1).FullName
$NegativeReplay = (Get-ChildItem .\data\raw\chat_negatives -Filter *.json | Select-Object -First 1).FullName
go -C clipper run ./cmd/autoclip -repo .. -replay $PositiveReplay -replay $NegativeReplay
```

In Bash
```bash
positive_files=(data/raw/chat/*.json)
negative_files=(data/raw/chat_negatives/*.json)

go -C clipper run ./cmd/autoclip -repo .. \
  -replay "${positive_files[0]}" \
  -replay "${negative_files[0]}"
```

The command prints one JSON object per file containing the target offset, score,
saved threshold, and `candidate` decision. Confirm that both files load, ONNX
inference completes, scores are in `[0, 1]`, and the threshold matches the newly
saved model threshold. A positive may score below threshold and a negative may
score above it; either result can be a model error rather than a replay failure.
Replay never writes live shadow logs.

To inspect more windows, repeat `-replay` or pass comma-separated paths.

## 6. Configure Twitch user authorization

Copy `.env.example` to `.env` if needed. Live mode requires:

```dotenv
TWITCH_CLIENT_ID=your_registered_application_client_id
TWITCH_USER_ACCESS_TOKEN=your_user_token
```

The user token must:

- be issued for the same client ID;
- include `user:read:chat`;
- represent a Twitch user allowed to read the configured channels.

For prototype testing, Twitch documents generating the token with the Twitch
CLI:

```powershell
twitch token --user-token --scopes "user:read:chat"
```

Configure the Twitch CLI with the same application first. The clipper validates
the token at startup and hourly, and never prints the token.

## 7. Start live shadow mode

All configured streamers may run concurrently so collection is not blocked
when one target channel is offline. Checkpoint 3's required 16-hour gate still
counts only Arky, Jynxzi, Marlon, and Lacy; extra streamers are reported
separately and create additional episodes to review. Before each run, predeclare
the session by setting exactly one value:

```yaml
clipper:
  review_partition: "calibration"  # first eight useful streamer-hours
```

After calibration reaches eight useful streamer-hours, stop the clipper,
change the value to `confirmation`, and restart. Never change a session's
partition after collection. Then run:

```powershell
go -C clipper run ./cmd/autoclip -repo .. -config config.yaml
```

The process uses one EventSub WebSocket for all active channels. It waits 35
seconds after observing a live stream before inference, scores every 2.5
seconds, and scores the clip-start-equivalent target 30 seconds before each
inference time. It appends lightweight telemetry for every successful
inference, preserves immediate candidate logs on threshold triggers, and
finalizes peak-window episodes for review. Stop it with `Ctrl+C`.

For the first smoke test, run while at least one configured active streamer is
live. Confirm:

1. startup reports that the Twitch user token was validated;
2. a shadow session starts for each configured streamer that is live;
3. the process remains connected for at least several minutes without a fatal
   EventSub error;
4. `Ctrl+C` exits cleanly; and
5. the selected partition's `sessions.jsonl` receives a session record.

It is valid for a short smoke test to produce no candidate. If a threshold
crossing occurs, confirm that it is appended to
the selected partition's `candidates.jsonl`, `candidates_review.jsonl`, and
`candidates_review.csv`. Do not lower the
threshold merely to force a candidate during the smoke test.

The legacy-geometry authenticated smoke test passed on 2026-08-04: one StableRonaldo
session ran for about 23 minutes 53 seconds, processed 9,948 messages over 560
inferences, recorded 15 candidates, and reported zero inference errors.
It proves the transport/runtime path worked but must not be used as window-v2
quality evidence.

Generated records are physically separated under
`data/live/shadow/window-v2/{calibration|confirmation}/`. Schema-v2 telemetry,
episodes, and sessions also carry `review_partition`:

- `telemetry.jsonl` — one row per
  successful inference with score, threshold, detector/cooldown state, raw
  features, model manifest checksum, and cumulative per-session dropped-chat
  count; it intentionally contains no full chat;
- `candidates.jsonl` — full candidate windows,
  scores, messages, exact features, and model manifest checksum;
- `candidates_review.jsonl` — scrollable companion
  written automatically on each candidate (`candidate_id`, `session_id`,
  `streamer`, `score`, `stream_offset_stamp`);
- `candidates_review.csv` — same companion fields plus
  empty `review_label` / `reason` columns for human notes;
- `sessions.jsonl` — immutable per-stream counters
  and useful durations, episode/local-peak counts, dropped-chat totals, plus
  optional `vod_id` once known; join reviews via `session_id`;
- `episodes.jsonl` — finalized schema-v2 triggered
  episodes and below-threshold local maxima. Triggered episodes retain the
  highest-scoring full chat window and close after two consecutive
  below-threshold ticks, 60 seconds, or session close;
- `episodes_review.jsonl` and
  `episodes_review.csv` — episode review companions. `record_type` distinguishes
  `triggered` from `local_maximum`; fill only the CSV review columns.

Below-threshold sampling uses a deterministic three-tick rule: the middle score
must be strictly greater than the prior score, at least the following score,
and below threshold. At most five are persisted in each useful-hour bucket.

Window-v2 uses a separate directory so its acceptance metrics cannot be
accidentally mixed with the legacy five-second-lag shadow session.

Do not interpret scores as confidence percentages. Review candidates against
the stream/VOD context before changing thresholds or enabling any future clip
creation behavior.

## 8. Analyze telemetry and replay sensitivities

After episode reviews are filled, replay one or more thresholds without running
the model again:

```powershell
python training/live/analyze_telemetry.py --partition calibration --thresholds 0.48,0.52,0.56
python training/live/analyze_telemetry.py --partition confirmation --thresholds 0.48,0.52,0.56
```

The analyzer writes `telemetry_analysis.json` inside the selected partition
with useful streamer-hours,
episodes/hour, score distributions, decided acceptance, reviewed
positive-versus-hard-negative peak-score ROC AUC, deterministic bootstrap
intervals, per-streamer results, and dropped-chat rates.

Telemetry supports volume estimates at alternative thresholds. Acceptance is
reported only when reviewed peak windows support that score range. For a lower
threshold, the analyzer requires at least five decided below-threshold local
maxima in range; otherwise `acceptance_supported` is false and acceptance is
left null.

Checkpoint 2 verification commands (user-run):

```powershell
python -m unittest discover -s training/live -p "test_*.py"
go -C clipper test ./...
go -C clipper test -race ./...
```

Checkpoint 3 adds collection/import/audit tests. After changing this branch,
rerun:

```powershell
python -m unittest discover -s training/collect -p "test_*.py"
python -m unittest discover -s training/live -p "test_*.py"
go -C clipper test ./...
go -C clipper test -race ./...
```

Then run representative replay/synthetic score sequences with the current
bundle and confirm a finalized episode can have `peak_score > onset_score` and
the analyzer replay episode count matches the emitted sequence. Do not start
new Checkpoint 3 collection until these checks pass.

## 9. Review shadow episodes against the VOD

Keep each partition's append-only `episodes.jsonl` for peak features and chat.
Every finalized triggered episode or sampled local maximum appends a companion
row to that partition's `episodes_review.csv` with episode/session identity,
onset/peak scores, and the peak seek stamp. Fill `review_label` and `reason`
after watching the VOD (`positive` / `hard_negative` / `uncertain`). Review
every row in both partitions, but never import confirmation into training. Do
not edit append-only JSONL. If you edit CSV in Excel while the clipper runs,
close it before the next episode write or Excel may lock the append.

Join `session_id` → `sessions.jsonl`. When that session has `vod_id`, open:

`https://www.twitch.tv/videos/{vod_id}?t={stream_offset_stamp}`

The episode stamp is the peak window's clip-start-equivalent target. Start
roughly five seconds earlier to inspect its full scored window. Judge the video
moment, not only chat or score.

If `vod_id` is missing, resolve once from Helix archives by matching the
session `stream_id` (Twitch CLI example):

```powershell
twitch api get /videos -q user_id=100869214 -q type=archive -q first=20
```

Automatic resolve-on-session-close / review-time refresh is deferred; see
`docs/project_status.md`.

After calibration episode labels are filled, import peak windows into training
(does not edit append-only JSONL logs):

```powershell
python training/collect/import_live_reviews.py --partition calibration
python training/collect/build_dataset.py
python training/features/encode.py
```

The importer defaults to schema-versioned episode reviews and materializes each
episode's peak target/window, not its onset candidate. It writes
`review_partition` and `review_identity` to durable annotations and raw live
windows. Confirmation is locked: a normal confirmation import fails closed.
Audit its joins without writing training data:

```powershell
python training/collect/import_live_reviews.py --partition confirmation --validate-only
python training/live/audit_collection.py
```

The audit requires eight useful hours in each partition, at least two total
hours for every target streamer, at least 100 decided episode reviews, all
sampled episodes reviewed, and no confirmation row in annotations/raw training
windows. Rebuild and freeze the Checkpoint 4 snapshot only after those gates
pass.

Rows without `vod_id` are skipped. Live VODs that enter training must not be
reused as an untouched test. See `docs/training_playbook.md`.
