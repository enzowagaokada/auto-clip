# Project Status (Living Doc)

**Last updated:** 2026-08-21

Agents and humans: read this first for current state and next actions.
Deep methodology lives in `docs/twitch_classifier_brief.md`.
Training details live in `docs/training_playbook.md`.

---

## One-line status

Harvest retrain `window-v2-harvest-seed0` finished: val AP **0.506** (better than live-hn 0.454, still below vod-seed0’s 0.545 on an easier in-community val split). Same epoch-1 overfit. Keep `window-v2-vod-seed0` live. Next: `analyze_run.py` on the harvest run, then optional FP review.

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
| Dataset + temporal features | Post-harvest rebuild: 23,706 examples, 7,603 positive / 16,103 negative |
| Train / evaluate / holdout / review loop | `window-v2-harvest-seed0` trained (val AP 0.506); `window-v2-vod-seed0` still the live bundle |
| Untouched-VOD evaluation | Harvest eval recorded for `window-v2-vod-seed0` (see below) |
| Hard-negative sample weighting | Built: `training.hard_negative_weight: 3.0` on reviewed hard negatives |
| ONNX export | Window-v2 bundle exported; 3,078-row parity passed with zero mismatches |
| Go live clipper | Window-v2 unit/race tests, build, replay, and authenticated shadow passed |
| Shadow-mode acceptance tracking | ~100/198 window-v2 candidates labeled; first acceptance readout below |
| Paid product / UI | Later |

---

## Window-v2 live shadow review (2026-08-11)

Logs: `data/live/shadow/window-v2/`. Threshold: **0.480**.

### Volume (~9.9 useful streamer-hours; concurrent sessions overlap)

| Streamer | Hours | Candidates | Cand/hour |
|---|---:|---:|---:|
| arky | 4.07 | 87 | 21.4 |
| lacy | 2.29 | 73 | 31.8 |
| jynxzi | 2.29 | 32 | 14.0 |
| marlon | 1.20 | 5 | 4.2 |
| jasontheween | 0.04 | 1 | (smoke) |
| **Total** | **9.90** | **198** | **20.0** |

### Acceptance (labeled rows only; `negative` counted as `hard_negative`)

- Labeled: **100** / 198 (unlabeled mostly arky 53 + lacy 42)
- Labels: 46 positive / 20 hard_negative / 34 uncertain
- **Acceptance among decided** (`pos / (pos+hn)`): **69.7%**
- Positive share of all labeled (uncertain counts against): **46.0%**
- In-train streamers (lacy+marlon): decided acc **76.0%** (small marlon n=5, all non-positive)
- Never-trained (arky+jynxzi): decided acc **65.9%**
- Per-streamer decided acc: lacy **90.5%**, arky **83.3%**, jynxzi **52.2%**, marlon **0%** (n=4 decided)

### Score compression (matches reviewer notes)

- All candidate scores ≈ **0.48–0.60** (median **0.501**, p90 **0.544**, max **0.601**)
- Positive median **0.505** vs hard_negative median **0.503** — almost no ranking separation
- Common hard-negative themes in reasons: WW/gift spam, stream start, empty mix, emote-only

First Jason smoke still only proves transport. Do not mix with legacy five-second-lag logs.

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

## Window-v2 harvest retrain (2026-08-21)

- Run dir: `models/runs/window-v2-harvest-seed0`
- Dataset: **23,706** (7,603 pos / 16,103 neg)
- Hard-negative 3× applied to **36** train rows
- Best epoch: **1** (early-stopped after epoch 4)
- Saved threshold: **0.460**
- VOD-grouped validation: precision **0.436**, recall **0.780**, F1 **0.560**
- Validation AUC: **0.707**
- Validation AP: **0.506**

Better than live-hn (AP 0.454) on a larger mixed-streamer split. Not a clear replacement for `window-v2-vod-seed0` (val AP 0.545) because that older val set had no Arky/Jynxzi/PBM. Same overfitting: train AP 0.508→0.833 while val AP 0.506→0.450. Harvest VODs are no longer an untouched test for this run.

---

## Window-v2 live-hn retrain (2026-08-18)

- Run dir: `models/runs/window-v2-live-hn-seed0`
- Dataset: **15,442** (4,846 pos / 10,596 neg); +66 vs prior rebuild matches 46 live positives + 20 live hard-negatives
- Hard-negative 3× applied to **34** train rows
- Best epoch: **1** (early-stopped after epoch 4)
- Saved threshold: **0.520**
- VOD-grouped validation: precision **0.412**, recall **0.709**, F1 **0.521**
- Validation AUC: **0.664**
- Validation AP: **0.454**

Worse than `window-v2-vod-seed0` on this split (AP 0.545 / AUC 0.727 / P 0.509). The split is not identical (live VODs are now in the pool). Same overfitting: train AP rose while val AP fell. Do not replace the live bundle until an untouched eval and/or a new shadow pass say otherwise.

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

### Untouched harvest test (2026-08-21)

Manifest reconstructed from pre-Aug-19 raw mtimes after the snapshot was skipped:
`data/splits/untouched_after_harvest.txt` (106 new VODs; live-session VODs excluded).

```powershell
python training/model/analyze_run.py --run-dir models/runs/window-v2-vod-seed0 --vod-manifest data/splits/untouched_after_harvest.txt
```

Results (`analysis-untouched_after_harvest/`):

| Metric | Value |
|---|---|
| Windows | 7317 |
| Threshold | 0.480 (saved; not retuned) |
| Precision | 0.477 |
| Recall | 0.580 |
| F1 | 0.523 |
| AUC | 0.677 |
| AP | 0.485 |
| Prevalence | 0.333 |

Random AP ≈ **0.33**. This is weaker than the same run’s VOD-grouped val AP **0.545** (expected: most of these VODs are Arky/Jynxzi/PBM, never in training) but still better than chance. Do not retune on this manifest. These VODs may now be used in a later retrain; collect a new untouched set after that.

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
4. Manual reviews go through `import_reviews.py` or `import_live_reviews.py` → `data/reviews/window_labels.csv`. Live reviews also write `data/raw/chat_live/`. Never only edit `dataset.jsonl`.
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
13. Raw windows and model metadata carry geometry version 2; builders/export/live
    must reject stale geometry instead of mixing contracts.
14. Live shadow reviews become training data only via `import_live_reviews.py`.
    Training applies `hard_negative_weight` (default 3×) to reviewed hard
    negatives; live VODs used in that retrain are not an untouched test.

---

## Important paths

| Path | Purpose |
|---|---|
| `config.yaml` | Streamers, model, training hyperparams |
| `data/processed/dataset.jsonl` | Labeled windows |
| `data/reviews/window_labels.csv` | Durable manual reviews |
| `data/raw/chat_live/` | Geometry-v2 windows materialized from live shadow reviews |
| `models/runs/window-v2-live-hn-seed0/` | Planned retrain after live-review import + hard-negative weighting |
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
- [x] Review a first pass of window-v2 live candidates and measure acceptance / candidates per hour
- [x] Build live-review → `window_labels` + `data/raw/chat_live/` import so shadow labels can train
- [x] Implement hard-negative sample weighting (~3×) as part of the next retrain
- [ ] Optionally finish remaining ~98 unlabeled rows (mostly arky/lacy) to tighten estimates
- [x] Run live import, rebuild, and train `models/runs/window-v2-live-hn-seed0` (do not overwrite `window-v2-vod-seed0`)
- [ ] After the live-hn retrain, run a **new** untouched VOD eval that excludes imported live VODs; do not export over the previous bundle on val AP alone

```powershell
python training/collect/import_live_reviews.py
python training/collect/build_dataset.py
python training/features/encode.py
python training/model/train.py --output-dir models/runs/window-v2-live-hn-seed0
```

### 2. Improve model after this shadow readout

- [x] Add never-trained live streamers (arky/jynxzi) into collection when ready for broader coverage
- [x] After the live-hn retrain, run a **new** untouched VOD eval that excludes imported live VODs
- [x] Retrain a **new** run dir on the harvest dataset (keep `window-v2-vod-seed0` as the live bundle until the new run beats it)
- [ ] `python training/model/analyze_run.py --run-dir models/runs/window-v2-harvest-seed0` then optional top-FP review
- [ ] After that retrain, snapshot VODs first, then collect a **new** untouched set before any later retrain

### 3. Shadow review linking (partial now / auto later)

- [x] Put `session_id` on `candidates_review.jsonl` / `.csv` so rows join to `sessions.jsonl`
- [x] Add optional `vod_id` on session records; backfill existing Arky/Marlon/Jason sessions
- [ ] Later: on session close, poll Helix archives until `stream_id` matches and persist `vod_id` (retry/backoff; archives often appear minutes–hours after offline)
- [ ] Later: optional review-time refresh for still-pending `vod_id`s; paginate beyond `first=20` for backfill of old sessions
- [ ] Later: surface ready Twitch URLs in the review CSV (`https://www.twitch.tv/videos/{vod_id}?t={stamp}`)

### 4. Productization later

- [ ] Strict / Balanced / Discovery sensitivity presets
- [ ] Approval queue UI or Discord alerts
- [ ] Per-streamer calibration from acceptance rates
- [ ] Outside-community streamers for broader generalization
- [ ] Fully automatic clipping only after live acceptance is high enough
- [ ] Ensure broadcast ids can automatically be inserted without the user needing to search it
      up themselves
- [ ] Figure out how to make persistent tokens for the user without needed them to login/
      get another user token consistently.

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

# Import live reviews, rebuild, retrain (do not overwrite window-v2-vod-seed0)
python training/collect/import_live_reviews.py
python training/collect/build_dataset.py
python training/features/encode.py
python training/model/train.py --output-dir models/runs/window-v2-live-hn-seed0
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
