# ONNX and Go Shadow Clipper Runbook

Run every command from the repository root unless the command uses `go -C
clipper`. The Go application is hard-gated to `clipper.mode: shadow` and has no
Twitch Create Clip implementation.

> **Window-v2 migration:** the existing raw chat, trained run, ONNX export, and
> recorded shadow session use the superseded `[target - 30s, target + 5s]`
> geometry. Do not restart live shadow mode until the v2 data has been fetched,
> retrained, exported, and verified.

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

```powershell
$PositiveReplay = (Get-ChildItem .\data\raw\chat -Filter *.json | Select-Object -First 1).FullName
$NegativeReplay = (Get-ChildItem .\data\raw\chat_negatives -Filter *.json | Select-Object -First 1).FullName
go -C clipper run ./cmd/autoclip -repo .. -replay $PositiveReplay -replay $NegativeReplay
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

Review the active streamers and `clipper` section in `config.yaml`, then run:

```powershell
go -C clipper run ./cmd/autoclip -repo .. -config config.yaml
```

The process uses one EventSub WebSocket for all active channels. It waits 35
seconds after observing a live stream before inference, scores every 2.5
seconds, and scores the clip-start-equivalent target 30 seconds before each
inference time. It logs only below-to-above threshold crossings. Stop it with
`Ctrl+C`.

For the first smoke test, run while at least one configured active streamer is
live. Confirm:

1. startup reports that the Twitch user token was validated;
2. a shadow session starts for each configured streamer that is live;
3. the process remains connected for at least several minutes without a fatal
   EventSub error;
4. `Ctrl+C` exits cleanly; and
5. `data/live/shadow/window-v2/sessions.jsonl` receives a session record.

It is valid for a short smoke test to produce no candidate. If a threshold
crossing occurs, confirm that it is appended to
`data/live/shadow/window-v2/candidates.jsonl`,
`data/live/shadow/window-v2/candidates_review.jsonl`, and
`data/live/shadow/window-v2/candidates_review.csv`. Do not lower the
threshold merely to force a candidate during the smoke test.

The legacy-geometry authenticated smoke test passed on 2026-08-04: one StableRonaldo
session ran for about 23 minutes 53 seconds, processed 9,948 messages over 560
inferences, recorded 15 candidates, and reported zero inference errors.
It proves the transport/runtime path worked but must not be used as window-v2
quality evidence.

Generated records:

- `data/live/shadow/window-v2/candidates.jsonl` — full candidate windows,
  scores, messages, exact features, and model manifest checksum;
- `data/live/shadow/window-v2/candidates_review.jsonl` — scrollable companion
  written automatically on each candidate (`candidate_id`, `streamer`, `score`,
  `stream_offset_stamp`);
- `data/live/shadow/window-v2/candidates_review.csv` — same companion fields plus
  empty `review_label` / `reason` columns for human notes;
- `data/live/shadow/window-v2/sessions.jsonl` — immutable per-stream counters
  and durations, sufficient to calculate candidates per stream-hour.

Window-v2 uses a separate directory so its acceptance metrics cannot be
accidentally mixed with the legacy five-second-lag shadow session.

Do not interpret scores as confidence percentages. Review candidates against
the stream/VOD context before changing thresholds or enabling any future clip
creation behavior.

## 8. Review shadow candidates against the VOD

Keep the full append-only `candidates.jsonl` for features and chat. While the
clipper runs, each candidate also appends one companion line to
`data/live/shadow/window-v2/candidates_review.jsonl` and
`data/live/shadow/window-v2/candidates_review.csv` with
`candidate_id`, `streamer`, `score`, and `stream_offset_stamp` (for example
`1h1m4s`). The CSV also has empty `review_label` and `reason` columns — fill
those after watching the VOD (`positive` / `hard_negative` / `uncertain`, plus
a short reason). Do not edit the append-only JSONL logs to store decisions.
If you edit the CSV in Excel while the clipper is running, close the file
before the next candidate write or Excel may lock the append.

Copy the stamp into Twitch's seek box or a URL of the form
`https://www.twitch.tv/videos/VOD_ID?t=1h1m4s`.

Under window v2, the stamp is the clip-start-equivalent moment 30 seconds
before `detected_at`. To find the VOD:

1. Query the broadcaster's recent archives with the configured Twitch CLI:

   ```powershell
   twitch api get /videos -q user_id=107117952 -q type=archive -q first=20
   ```

2. Find the returned video whose `stream_id` matches the candidate/session
   `stream_id`. Its `id` is the VOD ID.
3. Seek to `stream_offset_stamp`, starting roughly five seconds earlier to
   inspect the complete scored window.
4. Judge the actual video moment, not only the chat messages or score.

The logged `stream_id` identifies the broadcast but differs from the archive
VOD `id`. If an archive is not returned yet, wait for Twitch to finish
processing it and retry.
