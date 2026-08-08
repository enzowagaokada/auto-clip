# Project Status (Living Doc)

**Last updated:** 2026-08-08

Agents and humans: read this first for current state and next actions.
Deep methodology lives in `docs/twitch_classifier_brief.md`.
Training details live in `docs/training_playbook.md`.

---

## One-line status

Window-v2 end-to-end path is live: train, export, parity, replay, and authenticated shadow all passed. Live shadow dual-writes review companions to `candidates_review.jsonl` and `candidates_review.csv` (empty label/reason for human fill). Next: review window-v2 candidates and measure acceptance quality.

---

## What this product is

- Rank chat windows by how much they look like historically clippable hype.
- Ship as **candidate generator + human review**, not guaranteed auto-clips.
- User-facing control: **clipping sensitivity** (Strict / Balanced / Discovery).
- Chat alone cannot perfectly separate joke spam (`67`, music, stream start) from real moments.

---

## Current phase

| Layer | Status |
|---|---|
| Data collection pipeline | Window-v2 refetch completed; unavailable legacy windows quarantined |
| Dataset + temporal features | Rebuilt: 15,376 examples, 4,800 positive / 10,576 negative |
| Train / evaluate / holdout / review loop | Window-v2 candidate trained; saved at best validation AP |
| Untouched-VOD evaluation | Legacy baseline recorded; new untouched evaluation required after retraining |
| Hard-negative sample weighting | Not built yet |
| ONNX export | Window-v2 bundle exported; 3,078-row parity passed with zero mismatches |
| Go live clipper | Window-v2 unit/race tests, build, replay, and authenticated shadow passed |
| Shadow-mode acceptance tracking | First window-v2 session recorded; human review pending |
| Paid product / UI | Later |

---

## Window-v2 live shadow smoke

- Streamer: `jasontheween`
- Session: about 2m 27s, 386 messages, 45 inferences
- Candidates: 1 at score `0.5053` / threshold `0.480`
- Inference errors: 0
- Logs: `data/live/shadow/window-v2/`

This proves the window-v2 live path works. Candidate quality still needs human
VOD review; do not mix these records with the legacy five-second-lag smoke.

## Legacy live shadow smoke

- Streamer: `stableronaldo`
- Session: about 23m 53s, 9,948 messages, 560 inferences
- Candidates: 15 (about 37.7 per stream-hour in this short sample)
- Score range: 0.5760–0.8239 at threshold 0.570
- Inference errors: 0

This proved the transport/runtime path under the superseded
`[target - 30s, target + 5s]` geometry. Do not use these candidates as
window-v2 quality evidence.

---

## Window-v2 model candidate

- Run dir: `models/runs/window-v2-vod-seed0`
- Best epoch: **1** (early-stopped after epoch 4)
- Saved threshold: **0.480**
- VOD-grouped validation: precision **0.509**, recall **0.626**, F1 **0.562**
- Validation AUC: **0.727**
- Validation AP: **0.545** (positive prevalence ≈ **0.312**)

Training metrics continued improving while validation AP fell after epoch 1,
showing rapid overfitting. Early stopping correctly restored the epoch-1
checkpoint.

---

## Previous model (superseded by window-v2 migration)

- Run dir: `models/runs/reviewed-vod-seed0`
- Trained after StableRonaldo false-positive reviews were imported
- Saved threshold: **0.570**
- Artifacts: `chat_classifier_params.msgpack`, `vocab.json`, `inference_meta.json`
- Status: legacy geometry; do not export or run live after the v2 code change

### Untouched VOD test (honest generalization)

Command used:

```powershell
python training/model/analyze_run.py --run-dir models/runs/reviewed-vod-seed0 --vod-manifest data/splits/untouched_vods.txt
```

Results (`analysis-untouched_vods/`):

| Metric | Value |
|---|---|
| Windows | 3137 across 39 new VODs |
| Threshold | 0.570 (saved; not retuned) |
| Precision | 0.480 |
| Recall | 0.722 |
| F1 | 0.577 |
| AUC | 0.713 |
| AP | 0.545 |

Random AP baseline ≈ positive prevalence ≈ **0.33**. This is meaningfully better and transfers to new streams.

---

## Key decisions already made

1. Default validation is **whole-VOD split**, not random windows.
2. True untouched test = evaluate a **saved** model with `analyze_run.py --vod-manifest` at its **saved** threshold. Do not retrain or retune first.
3. `train.py --holdout-vods` is validation/tuning, **not** an untouched test.
4. Manual reviews go through `import_reviews.py` → `data/reviews/window_labels.csv`. Never only edit `dataset.jsonl`.
5. Product framing is **clipping sensitivity + review queue**, not raw “confidence %”.
6. Current model capacity (embed 32 / GRU 64 / vocab 10k) is intentional for current data size.
7. Export directly with `jax2onnx`; the old `jax2tf -> tf2onnx` route is deprecated.
8. Export uses explicit saved GRU equations because `jax2onnx` cannot trace the
   current Flax lifted `nn.RNN`; the exporter first asserts exact Flax parity.
9. Helix `vod_offset` is clip-video start. Historical parity is
   `[target - 5s, target + 30s]`; live uses `[now - 35s, now]` and scores
   target `now - 30s`.
10. The first Go release is hard-gated to **shadow mode** and contains no Create Clip path.
11. Live chat uses one Twitch EventSub WebSocket and a user token with `user:read:chat`.
12. Raw windows and model metadata carry geometry version 2; builders/export/live
    must reject stale geometry instead of mixing contracts.

---

## Important paths

| Path | Purpose |
|---|---|
| `config.yaml` | Streamers, model, training hyperparams |
| `data/processed/dataset.jsonl` | Labeled windows |
| `data/reviews/window_labels.csv` | Durable manual reviews |
| `data/splits/vods_before_collection.txt` | Baseline VOD snapshot |
| `data/splits/untouched_vods.txt` | New VODs used for the recorded untouched test |
| `models/runs/reviewed-vod-seed0/` | Superseded legacy-geometry model |
| `models/runs/reviewed-vod-seed0/analysis-untouched_vods/` | Untouched-test outputs |
| `models/runs/window-v2-vod-seed0/` | Planned window-v2 trained run |
| `training/export/` | Direct ONNX export and parity tools |
| `models/exports/window-v2-vod-seed0/` | Planned window-v2 deployment bundle |
| `clipper/` | Standalone Go shadow clipper |
| `data/live/shadow/window-v2/` | Window-v2 candidate/session/review logs (`candidates_review.jsonl` + `.csv`) |
| `docs/live_clipper.md` | Setup, verification, replay, and live runbook |
| `docs/training_playbook.md` | How to train / interpret metrics |
| `docs/twitch_classifier_brief.md` | Full product/ML brief |

---

## Next steps (ordered)

### 1. Complete window-v2 migration (do this next)

- [x] Implement direct ONNX export from the current JAX/Flax model
- [x] Export legacy `models/exports/reviewed-vod-seed0/`
- [x] Verify ONNX outputs match JAX logits/sigmoid decisions
- [x] Verify the Go preprocessing fixture against the Python pipeline
- [x] Build Go 35s rolling buffer + 5s target lag + inference loop
- [x] Add threshold-crossing rearm and cooldown
- [x] Log candidates and session counters to JSONL in hard-gated shadow mode
- [x] Run normal Go tests and race-detector tests
- [x] Run historical replay against representative positive and negative windows
- [x] Run the first authenticated live shadow-mode smoke test
- [x] Correct historical geometry to `[clip start - 5s, clip start + 30s]`
- [x] Update Go target lag to 30s and add stale-bundle rejection
- [x] Run window-v2 Python unit tests and Go normal/race tests
- [x] Refetch all positive and negative raw chat windows
- [x] Rebuild dataset and inspection encoding
- [x] Train `models/runs/window-v2-vod-seed0`
- [x] Analyze the saved window-v2 validation split
- [x] Export and verify the window-v2 ONNX bundle
- [x] Verify Python/Go preprocessing parity and build the Go executable
- [x] Run positive/negative replay against the new bundle
- [x] Run a new authenticated window-v2 shadow session
- [ ] Review window-v2 live candidates and track acceptance rate / bad suggestions per hour

### 2. Improve model in parallel / after shadow data

- [ ] Optionally review more untouched false positives
- [ ] Implement hard-negative sample weighting (~3×)
- [ ] Retrain only after new labels or live feedback exist
- [ ] Collect a **new** untouched VOD set after the next retrain cycle

### 3. Productization later

- [ ] Strict / Balanced / Discovery sensitivity presets
- [ ] Approval queue UI or Discord alerts
- [ ] Per-streamer calibration from acceptance rates
- [ ] Outside-community streamers for broader generalization
- [ ] Fully automatic clipping only after live acceptance is high enough

---

## Metric targets (reminder)

| Goal | Target |
|---|---|
| Start Go/ONNX prototype | Current untouched results are enough |
| Closed beta suggestion product | Live acceptance ~60–70%+, AUC ≥ ~0.75 preferred |
| Full auto-clipping | Live acceptance ~80–85%+, stable across streamers |

Offline precision overstates live precision because the dataset is enriched (~1:2 pos:neg), while real streams have far fewer clip moments.

---

## How to resume quickly

```powershell
# Activate env
.\.venv\Scripts\Activate.ps1

# Re-check untouched analysis summary
# models/runs/reviewed-vod-seed0/analysis-untouched_vods/summary.json
```

Tell a new agent:

> Read `docs/project_status.md` first, then only the docs it points to if needed.

---

## Update rules

Whenever meaningful progress happens, update **this file** in the same session:

1. Change **Last updated**
2. Refresh **One-line status** and **Current phase**
3. Record new best metrics / model path
4. Check off finished next steps and add new ones
5. Note any decision that future agents must not reverse casually
