# Training Playbook

Use this checklist whenever the chat classifier is rebuilt. The goal is not to
drive training loss to zero; the goal is to rank unseen clippable moments above
normal moments from VODs and streamers the model did not memorize.

## Rebuild and train

Run scripts from the repository root:

```powershell
python training/collect/build_dataset.py
python training/features/encode.py
python training/model/train.py
```

`encode.py` creates full-dataset arrays for inspection. `train.py` reads
`dataset.jsonl` directly and deliberately rebuilds its vocabulary and numeric
feature scaler from the training split only. The model must ship with
`models/vocab.json` and `models/inference_meta.json`; do not substitute the
full-dataset vocabulary from `data/processed/`.

The default validation split keeps complete VODs together. For the stronger
generalization test, hold out each streamer in turn:

```powershell
python training/model/train.py --holdout-streamer jasontheween --seed 0 --output-dir models/runs/jasontheween-seed0
python training/model/train.py --holdout-streamer stableronaldo --seed 0 --output-dir models/runs/stableronaldo-seed0
```

Repeat important runs with seeds 0 through 4. Compare means and variation, not
only the best seed. Always give experiments separate output directories so they
do not overwrite the candidate production model in `models/`.

## Reading the output

- **BCE**: lower is better, but compare train and validation together.
- **Precision**: fraction of predicted moments that were clips.
- **Recall**: fraction of known clips detected.
- **F1**: balance between precision and recall at the displayed threshold.
- **AUC**: threshold-independent ranking quality.
- **AP**: average precision; the primary ranking metric for this imbalanced
  problem. Its no-skill baseline is the positive fraction.
- **t**: validation-selected decision threshold. It is a deployment setting,
  not a universal constant.

Healthy learning has improving train and validation metrics. Falling training
BCE combined with rising validation BCE or falling validation AP/AUC is
overfitting. Early stopping retains the best validation-AP checkpoint.

Do not compare F1 values from runs with different validation populations as if
they were identical experiments. Always record the split, seed, class counts,
threshold, and per-streamer results.

## Analyze a saved run

Export individual predictions and a threshold report without retraining:

```powershell
python training/model/analyze_run.py --run-dir models/runs/stableronaldo
```

This creates `<run-dir>/analysis/` containing:

- `summary.json` — metrics plus attainable 50/60/70/80% precision targets;
- `threshold_report.csv` — precision/recall/F1 and confusion counts by threshold;
- `validation_predictions.jsonl` — every scored validation window;
- `false_positives.jsonl` and `false_negatives.jsonl` — full error details/chat;
- `false_positive_review.csv` — the highest-scoring false positives to review.

Precision targets apply to the sampled validation ratio and will be lower on a
live stream where true clip moments are much rarer. Confirm deployment behavior
in shadow mode as false detections per stream-hour.

For each row in `false_positive_review.csv`, set `review_label` to:

- `positive` if the moment was genuinely worth clipping and the dataset label is
  incomplete;
- `hard_negative` if it was correctly negative despite high chat activity;
- `uncertain` if the VOD/chat context is insufficient.

Do not edit `dataset.jsonl` directly with review decisions because rebuilding it
will erase them. Import a completed review into durable annotations:

```powershell
python training/collect/import_reviews.py --review-file models/runs/stableronaldo/analysis/false_positive_review.csv
```

This merges reviews into `data/reviews/window_labels.csv` using
streamer/VOD/offset as the stable identity. It also repairs the common
`unertain` typo and can recover mildly malformed review rows by matching them to
the source `false_positives.jsonl`.

Then rebuild and retrain:

```powershell
python training/collect/build_dataset.py
python training/features/encode.py
python training/model/train.py --output-dir models/runs/window-v2-vod-seed0
```

Reviewed positives override the original negative label, reviewed hard
negatives remain explicit negatives, and reviewed uncertain windows are
excluded. Once reviews are incorporated into training, evaluate the resulting
model on new untouched VODs; the reviewed windows are no longer an unbiased
test set.

The analyzer preserves an existing `false_positive_review.csv` by default. Pass
`--overwrite-review` only when intentionally discarding completed review work.

## One-time untouched VOD test

An untouched test must evaluate an already-trained model at its already-selected
threshold. Do not train, early-stop, or tune thresholds on these VODs before
recording the result.

Before collecting new streams, snapshot every VOD currently in the processed
dataset:

```powershell
python training/collect/create_vod_manifest.py --output data/splits/vods_before_collection.txt
```

After new streams occur, append data through the normal pipeline:

```powershell
python training/collect/fetch_clips.py
python training/collect/fetch_chat.py
python training/collect/fetch_negatives.py
python training/collect/build_dataset.py
```

Create a manifest containing only VOD IDs absent from the baseline:

```powershell
python training/collect/create_vod_manifest.py --exclude data/splits/vods_before_collection.txt --output data/splits/untouched_vods.txt
```

Evaluate the already-trained window-v2 model at its saved threshold using a
fresh manifest that was not used by the legacy model:

```powershell
python training/model/analyze_run.py --run-dir models/runs/window-v2-vod-seed0 --vod-manifest data/splits/window_v2_untouched_vods.txt
```

External-test output goes to
`models/runs/window-v2-vod-seed0/analysis-window_v2_untouched_vods/`. Threshold
reports are deliberately omitted so the test is not silently used for tuning.
Record the fixed-threshold metrics before inspecting errors. The previously
recorded `reviewed-vod-seed0` untouched result remains historical evidence for
the legacy geometry and must not be recomputed against the rebuilt dataset.

Passing `--explore-thresholds` creates threshold reports, but doing so consumes
the manifest as a validation/tuning set. It must no longer be described as an
untouched test afterward.

`train.py --holdout-vods <manifest>` is also supported for explicit VOD
validation. Because training uses those metrics for early stopping and threshold
selection, that command is not a final untouched test.

After recording the one-time result, the VODs may be reviewed and folded into
future training. Create a fresh baseline and collect newer untouched VODs for
the next evaluation cycle.

## Before accepting a model

1. Confirm no VOD appears in both training and validation.
2. Confirm vocabulary and feature scaling came only from training rows.
3. Compare against simple message-rate and unique-user baselines.
4. Inspect errors, especially high-confidence false positives and false
   negatives.
5. Run streamer holdouts. A random-window score is not evidence of
   generalization.
6. Prefer AP/AUC for checkpoint selection, then choose a threshold for the
   product's precision/recall tradeoff.
7. Keep a final test set untouched by model and threshold tuning before a paid
   release.

For automatic clipping, precision usually matters more than maximum F1. For an
approval queue, lower precision can be acceptable in exchange for recall.
Weighted BCE scores are not calibrated probabilities, so validate the
threshold in shadow mode rather than interpreting `0.8` as an 80% guarantee.

## Capacity and data growth

The current 32-dimensional embedding, 64-dimensional GRU, and 10,000-token
vocabulary are sized for the current small dataset. Increase capacity only when
train and validation metrics plateau close together. If training improves while
validation stalls or declines, more capacity will make memorization worse.

Data diversity matters more than duplicate volume:

- add independent VODs and streamers;
- include different categories and audience sizes;
- collect high-activity non-clip moments as hard negatives;
- avoid many overlapping windows from the same event;
- manually inspect label quality.

Plot learning curves using 25%, 50%, 75%, and 100% of the training data. If
validation AP still rises consistently, more diverse data is likely useful. If
it plateaus, improve labels/features before scaling the model.

## Temporal features and inference parity

`build_dataset.py` derives seven five-second message-rate buckets, rate change,
peak rate, and repeated-message ratio. Re-run it after collecting chat so these
fields are present. The Go inference implementation must reproduce:

- case-preserving whitespace tokenization;
- left padding and most-recent-token truncation;
- all numeric features in the exact order from `inference_meta.json`;
- the saved feature means and standard deviations;
- the fixed 43,200-second stream-time scale;
- sigmoid applied to the model logit;
- the saved threshold.

Any mismatch between Python and Go preprocessing invalidates model scores.

## Export and verify the deployment model

Export only from an accepted saved run. After the geometry migration, the
deployment source is `models/runs/window-v2-vod-seed0`; the former
`reviewed-vod-seed0` run and older root-level files use stale geometry.

Run these commands from the repository root:

```powershell
python training/export/export_onnx.py
python training/export/verify_onnx.py
```

The exporter writes a generated bundle to
`models/exports/window-v2-vod-seed0/` containing the ONNX graph, saved
vocabulary, inference metadata, and checksum manifest. Verification checks the
graph contract and compares JAX and ONNX logits/sigmoid decisions over the
reconstructed saved validation split. Do not start the Go clipper if parity or
manifest verification fails.

Helix `vod_offset` is the start of the clip video. Historical collection uses
the fixed 35-second window `[clip start - 5s, clip start + 30s]`; clip duration
is retained as metadata but does not vary the model window. Live inference uses
the same `[now - 35s, now]` buffer and scores the clip-start-equivalent target
at `now - 30s`.

Window geometry is versioned in raw data and saved inference metadata. After a
geometry change, refetch positives and negatives, rebuild the dataset, retrain,
and re-export. Never change only the Go target lag against an older model.

## Release gate

Before charging users, run shadow mode on unseen streams and have humans rate
candidate moments. Track accepted suggestions, bad suggestions per stream-hour,
missed known clips, and results per streamer. A useful initial beta target is
roughly 70% accepted suggestions; fully automatic clipping should require
substantially higher precision.

Keep collecting approvals and rejections. They are more valuable future labels
than “a Twitch clip existed” versus “no clip existed.”
