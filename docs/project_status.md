# Project Status (Living Doc)

**Last updated:** 2026-07-29

Agents and humans: read this first for current state and next actions.
Deep methodology lives in `docs/twitch_classifier_brief.md`.
Training details live in `docs/training_playbook.md`.

---

## One-line status

Chat-only GRU clip-candidate model works on new VODs. Next product step is ONNX export + live Go clipper in shadow/suggestion mode. Do not wait for perfect auto-clipping.

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
| Data collection pipeline | Done / ongoing |
| Dataset + temporal features | Done |
| Train / evaluate / holdout / review loop | Done |
| Untouched-VOD evaluation | Done (baseline recorded) |
| Hard-negative sample weighting | Not built yet |
| ONNX export | Not started |
| Go live clipper | Not started |
| Shadow-mode acceptance tracking | Not started |
| Paid product / UI | Later |

---

## Best current model

- Run dir: `models/runs/reviewed-vod-seed0`
- Trained after StableRonaldo false-positive reviews were imported
- Saved threshold: **0.570**
- Artifacts: `chat_classifier_params.msgpack`, `vocab.json`, `inference_meta.json`

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
7. Next major engineering priority: **ONNX + Go live path**, not another big retrain.

---

## Important paths

| Path | Purpose |
|---|---|
| `config.yaml` | Streamers, model, training hyperparams |
| `data/processed/dataset.jsonl` | Labeled windows |
| `data/reviews/window_labels.csv` | Durable manual reviews |
| `data/splits/vods_before_collection.txt` | Baseline VOD snapshot |
| `data/splits/untouched_vods.txt` | New VODs used for the recorded untouched test |
| `models/runs/reviewed-vod-seed0/` | Current candidate production model |
| `models/runs/reviewed-vod-seed0/analysis-untouched_vods/` | Untouched-test outputs |
| `docs/training_playbook.md` | How to train / interpret metrics |
| `docs/twitch_classifier_brief.md` | Full product/ML brief |

---

## Next steps (ordered)

### 1. Build live path (do this next)

- [ ] Implement ONNX export from current JAX/Flax model
- [ ] Verify ONNX outputs match JAX logits/sigmoid closely
- [ ] Build Go rolling 30s chat buffer + inference loop
- [ ] Add cooldown + post-detection delay
- [ ] Log candidates to JSONL/CSV in **shadow mode** (no public auto-clip required)
- [ ] Track: candidates/hour, acceptance rate, bad suggestions/hour

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
