# ONNX and Go Shadow Clipper Runbook

Run every command from the repository root unless the command uses `go -C
clipper`. The Go application is hard-gated to `clipper.mode: shadow` and has no
Twitch Create Clip implementation.

## 1. Export and verify the model

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python training/export/export_onnx.py
python training/export/verify_onnx.py
python training/export/verify_go_fixture.py
```

The first command creates `models/exports/reviewed-vod-seed0/`. The parity
command must finish with zero threshold-decision mismatches. The fixture check
proves the committed Go tokenizer/feature expectations came from the Python
pipeline.

If rerunning an existing export intentionally:

```powershell
python training/export/export_onnx.py --overwrite
```

## 2. Install the native ONNX Runtime library

The Go binding and setup script are pinned to ONNX Runtime 1.26.0. The script
verifies Microsoft's published SHA-256 before extracting the DLL.

```powershell
powershell -ExecutionPolicy Bypass -File .\clipper\scripts\setup-onnxruntime-windows.ps1
```

The generated DLL is stored at `clipper/runtime/onnxruntime.dll` and is ignored
by Git.

## 3. Verify and build the Go module

These are the terminal checks reserved for the user:

```powershell
go -C clipper mod tidy
go -C clipper fmt ./...
go -C clipper test ./...
go -C clipper test -race ./...
go -C clipper build ./cmd/autoclip
```

The normal and race-detector test suites were reported passing on 2026-08-04.
The historical replay and authenticated live smoke test remain pending.

`-race` requires CGO support and may require a supported C compiler on Windows.
If that command cannot start because the race toolchain is unavailable, record
the toolchain error separately; the normal test command is still required.

## 4. Replay a historical window

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
inference completes, scores are in `[0, 1]`, and the threshold is the saved
model threshold (currently about `0.570`). A positive may score below threshold
and a negative may score above it; either result can be a model error rather
than a replay failure. Replay never writes live shadow logs.

To inspect more windows, repeat `-replay` or pass comma-separated paths.

## 5. Configure Twitch user authorization

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

## 6. Start live shadow mode

Review the active streamers and `clipper` section in `config.yaml`, then run:

```powershell
go -C clipper run ./cmd/autoclip -repo .. -config config.yaml
```

The process uses one EventSub WebSocket for all active channels. It waits 35
seconds after observing a live stream before inference, scores every 2.5
seconds, and logs only below-to-above threshold crossings. Stop it with
`Ctrl+C`.

For the first smoke test, run while at least one configured active streamer is
live. Confirm:

1. startup reports that the Twitch user token was validated;
2. a shadow session starts for each configured streamer that is live;
3. the process remains connected for at least several minutes without a fatal
   EventSub error;
4. `Ctrl+C` exits cleanly; and
5. `data/live/shadow/sessions.jsonl` receives a session record.

It is valid for a short smoke test to produce no candidate. If a threshold
crossing occurs, confirm that it is appended to
`data/live/shadow/candidates.jsonl`. Do not lower the threshold merely to force
a candidate during the smoke test.

Generated records:

- `data/live/shadow/candidates.jsonl` — reviewable candidate windows, scores,
  messages, exact features, and model manifest checksum;
- `data/live/shadow/sessions.jsonl` — immutable per-stream counters and
  durations, sufficient to calculate candidates per stream-hour.

Do not interpret scores as confidence percentages. Review candidates against
the stream/VOD context before changing thresholds or enabling any future clip
creation behavior.
