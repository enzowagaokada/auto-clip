# Final Draft: Overfitting and Live-Scoring Remediation

## Purpose

This is the prioritized implementation roadmap for reducing rapid
train/validation divergence and measuring live ranking quality correctly. Work
through the checkpoints in order and verify each checkpoint before starting the
next one. Do not implement the whole plan as one change.

Keep `models/runs/window-v2-vod-seed0` deployed in shadow mode until a challenger
passes Checkpoint 6. Keep JAX/Flax/Optax, direct `jax2onnx`, the standalone Go
clipper, and window geometry v2 (`[target - 5s, target + 30s]`).

The current evidence supports two hypotheses, not proven root causes:

1. uniformly sampled historical negatives do not represent the high-activity
   windows the live candidate system must rank;
2. the 10,000-token embedding table may memorize community- and VOD-specific
   text faster than it learns transferable signal.

Telemetry and controlled ablations below are required to test those hypotheses.

## Test execution ownership

The coding agent may create or update tests and must provide the exact terminal
commands needed to run them, but it must **not execute any tests, training,
export, replay, build, parity, lint, or authenticated shadow commands without
the user's explicit permission**. The user will run terminal verification and
share the results. Record a check as passed only after the user reports its
output.

---

## Checkpoint 1 — Data integrity and reproducible experiments

**Priority:** Highest. This blocks trustworthy retraining and comparison.

**Status (passed 2026-08-27):** All 19 collection and 4 model tests passed, two
cleaned rebuilds produced byte-identical dataset/audit hashes, and the fixed
60-VOD remediation validation manifest was created against dataset SHA-256
`270ba115f78149026fbb9ae50b995836eabdc30585e5b94163ee67b0905283be`.

### 1.1 Reconcile and clean the dataset

Current audit findings:

- `dataset.jsonl` has 23,698 rows while `project_status.md` records 23,706;
- 64 extra rows occur across 60 duplicate stable keys;
- one unreviewed key has contradictory labels;
- 202 negative anchors are within 60 seconds of a known clip;
- 6,269 same-VOD window pairs overlap.

Update `training/collect/build_dataset.py` so every row has:

- `example_id`: deterministic hash of normalized streamer, VOD, integer target
  offset, and geometry version;
- `source`: historical positive, sampled negative, or live review;
- `content_hash`: deterministic hash of label, messages, features, and identity;
- `event_group_id`: deterministic identifier for nearby windows representing
  substantially the same event;
- `base_sample_weight`: `1 / group_size` for training rows in an event group.

Deduplicate exact stable keys before writing the dataset. Durable reviews take
precedence over inferred labels. If contradictory labels remain after applying
reviews, fail the build and print every source path involved.

Do not create event groups using unrestricted transitive overlap: a long chain
of windows could merge unrelated moments. Within each streamer/VOD, sort by
target offset, anchor a group on its first target, and include later targets
only while they remain within one 35-second window of that fixed anchor. Start
a new group afterward. Report direct overlaps that cross group boundaries so
this policy remains auditable.

The build summary must report raw files, empty windows, exact duplicates,
conflicts, proximity exclusions, event groups, raw class counts, and effective
weighted class counts.

### 1.2 Repair negative-label hygiene

Update `training/collect/fetch_negatives.py` to revalidate every existing
negative against the latest `clips.csv` before it counts toward the per-VOD
target. An anchor inside the 60-second exclusion radius must be quarantined with
a descriptive suffix and replaced through the resumable top-up process.

Add the same validation defensively to `build_dataset.py`. Exclude unreviewed
collisions. A durable `hard_negative` review may explicitly override proximity.
Preserve all geometry-v2 checks and stale-window quarantine behavior.

### 1.3 Make runs self-contained

Create a deterministic, content-addressed dataset snapshot for each distinct
experiment dataset. Each run writes `run_manifest.json` containing:

- dataset snapshot path and SHA-256;
- exact train and validation VOD IDs;
- ordered train/validation `example_id` and `content_hash` values;
- raw and effective class/event counts;
- model, optimizer, vocabulary, feature, geometry, augmentation, and split
  configuration;
- seed and code/version metadata available locally.

Update `analyze_run.py` and ONNX verification to use the saved snapshot and run
manifest rather than reconstructing rows from mutable `dataset.jsonl`.
Missing or changed rows must fail clearly.

Create one fixed remediation validation-VOD manifest. All later ablations use
the same dataset snapshot and validation VODs; seeds may change initialization,
dropout, and batch order, but not membership.

### 1.4 Weight handling

Combine `base_sample_weight` with the existing 3× reviewed-hard-negative
multiplier. Normalize each batch loss by the sum of its effective weights so
event grouping does not unintentionally change the optimizer's overall
gradient scale. Keep validation metrics unweighted and add event-level metrics.

### Verification and exit gate

Add tests for:

- exact deduplication and durable-review precedence;
- unresolved conflict failure;
- deterministic, bounded event grouping and group weights summing to one;
- newly discovered clips invalidating stored negatives;
- 60-second boundary behavior, reviewed overrides, quarantine, and top-up;
- deterministic snapshots/manifests and row-drift rejection;
- fixed validation membership across seeds;
- weighted-loss normalization.

Rebuild twice and require byte-identical dataset output, snapshot identity, and
audit counts. Record the reconciled totals in `project_status.md`.

**Stop condition:** Do not retrain if any contradiction remains, a proximity
violation enters without an explicit review override, or repeated builds differ.

---

## Checkpoint 2 — Correct live measurement

**Priority:** Very high. This must precede conclusions about live score
compression.

**Status (implemented, verification pending):** Schema-v1 per-inference
telemetry, finalized triggered episodes, bounded below-threshold local maxima,
episode review companions, and deterministic threshold replay/analyzer code
are implemented on `1-checkpoint-2-live-telemetry`. The user-owned Go/Python
tests and synthetic/replay verification must pass before this checkpoint is
recorded as complete or new collection starts.

The current Go clipper records only the first threshold-crossing score. Scores
therefore naturally cluster near the threshold and cannot establish whether
underlying live ranking has collapsed.

### 2.1 Lightweight inference telemetry

Add a schema-versioned append-only telemetry JSONL row for every inference:

- session, streamer, stream ID, and model-manifest checksum;
- inference and target timestamps;
- score, active threshold, cooldown/rearm state, and trigger decision;
- raw numeric features;
- cumulative dropped-chat count.

Do not duplicate full chat every 2.5 seconds.

### 2.2 Finalized episodes

Keep the existing immediate candidate logs unchanged for compatibility. Add a
shadow-only episode aggregator that:

- opens at the first below-to-above threshold crossing;
- retains the highest-scoring window, full chat, and features as its peak;
- closes after two consecutive below-threshold ticks or 60 seconds;
- flushes open episodes when a session closes;
- writes versioned episode JSONL and review CSV records.

Store onset/peak score and time, duration, score range, peak target/window,
threshold, streamer/session identity, and model-manifest checksum.

Also retain a bounded sample of below-threshold local maxima with full peak
windows—at most five per useful streamer-hour—so lower sensitivity settings and
missed hard moments can be reviewed. Define the local-maximum rule
deterministically and mark these records separately from triggered episodes.

### 2.3 Analyzer

Create an analyzer that replays telemetry through configurable thresholds,
rearm logic, and cooldowns. It reports:

- useful streamer-hours and episodes per hour;
- score distributions and peak-score ROC AUC for reviewed positives versus hard
  negatives;
- acceptance among decided reviews;
- bootstrap confidence intervals;
- per-streamer results and dropped-chat rates.

Telemetry can estimate volume at alternative thresholds. Acceptance at an
alternative threshold is valid only where that score range has representative
reviewed peak windows; the analyzer must not imply otherwise.

### Verification and exit gate

Add Go tests for telemetry persistence, schema/config validation, episode peak
retention, both closure rules, session-close flushing, cooldown/rearm
interaction, local-peak limits, dropped-chat accounting, and compatibility with
existing candidate logs. Add deterministic analyzer fixtures.

Run replay/synthetic score sequences using the current bundle. Confirm episode
peaks can differ from onset scores and telemetry replay reproduces emitted
episodes.

**Stop condition:** Do not use candidate-onset medians to judge a challenger and
do not start collection until episodes survive restart/error and closure tests.

---

## Checkpoint 3 — Representative live labels

**Priority:** High. Improve the operating-distribution data before broad model
changes.

1. Finish the approximately 98 existing unlabeled window-v2 candidate reviews.
2. Update `import_live_reviews.py` to understand versioned episode review files
   and materialize the episode's peak window, not merely the onset candidate.
3. Collect 16 useful streamer-hours across Arky, Jynxzi, Marlon, and Lacy, with
   at least two hours per streamer.
4. Predeclare sessions as:
   - **calibration:** first eight useful hours;
   - **locked confirmation:** second eight useful hours.
5. Review every triggered episode plus the bounded below-threshold local-peak
   sample.

Keep calibration and confirmation review files physically separate. Import only
calibration decisions before challenger training. Confirmation decisions remain
locked evaluation data until the final model decision is recorded.

Keep `hard_negative_weight: 3.0` and event-group base weights. Do not duplicate,
oversample, or stratify the small reviewed-hard-negative pool. Reconsider batch
stratification only after at least 500 distinct reviewed hard-negative event
groups exist.

If 16 hours produce fewer than 100 new decided episodes, extend shadow
collection. Build full-VOD activity-matched negative mining only if live active
learning remains too sparse; if needed, retain a configurable mixture of
uniform and high-activity negatives.

### Verification and exit gate

Verify episode imports preserve geometry, target timestamps, chat, source,
review identity, and calibration/confirmation partition. Rebuild and audit the
dataset, then freeze the snapshot used by Checkpoint 4.

**Stop condition:** Do not proceed if confirmation rows entered the training
snapshot or if fewer than 100 representative decided episodes exist without a
documented decision to collect longer.

---

## Checkpoint 4 — Controlled diagnosis and regularization

**Priority:** Medium-high. Run only on the frozen snapshot and validation split.

### 4.1 Diagnostic models

Make architecture selection explicit while reusing identical preprocessing,
splits, loss, manifests, and metrics:

- simple message-rate baseline;
- message-rate plus unique-user baseline;
- numeric-features-only classifier;
- text-only GRU;
- current combined model.

Report unweighted window-level and event-level AP/AUC plus per-streamer metrics.
Use these results to determine whether text adds transferable ranking signal or
mostly train-set memorization.

### 4.2 Seed-0 one-factor screen

Start with the cleaned current configuration, then test one factor at a time:

1. cap vocabulary at 6,000 and require each included non-special token to occur
   in at least three distinct training VODs;
2. apply training-only 10% token dropout by replacing non-special tokens with
   `[UNK]`, preserving `[PAD]` and `[SEP]`;
3. reduce learning rate from `1e-3` to `3e-4`;
4. increase AdamW weight decay from `1e-4` to `1e-3`.

Evaluate and checkpoint every 150 optimizer steps, including during epoch 1.
Express early-stopping patience in evaluation intervals and save full learning
curves.

Keep a factor only if it improves validation AP, or materially reduces the
train/validation gap without lowering validation AP by more than 0.01. Compose
only retained factors.

Do not add:

- per-streamer feature z-scoring;
- hard-negative duplication;
- message deletion augmentation;
- embedding-specific optimizer transforms;
- a larger GRU or Transformer.

If numeric-only nearly matches the combined model, next test window-local,
scale-independent ratios such as peak/mean message rate and
unique-users/messages. Treat this as a new feature-contract experiment requiring
Python/Go parity work—not an automatic addition.

### Verification and exit gate

Tests must prove VOD-frequency vocabulary filtering is train-only and
deterministic, token dropout never changes validation/export inputs, step-level
checkpoint restoration works, and every run has identical frozen membership.

**Stop condition:** Do not combine factors that only improve training metrics or
that were compared on different rows.

---

## Checkpoint 5 — Multi-seed and cross-streamer confirmation

**Priority:** Medium. Confirm that a seed-0 improvement is stable.

Run the selected composition with seeds 0–4 on the same frozen train/validation
membership. Compare mean, variance, and worst seed—not only the best run.

Then run leave-one-streamer-out evaluations for all configured streamers using
the selected configuration. Report AP/AUC, prevalence, example/event counts,
and uncertainty per streamer. Do not tune separate model weights per streamer.

Advance one challenger only if it:

- maintains the seed-0 gain across seeds;
- does not create a severe regression for an individual streamer;
- improves or preserves event-level ranking;
- shows slower train/validation divergence than the current setup.

Per-streamer threshold overrides remain a later calibration option, but require
at least five useful hours and 20 decided episodes for that streamer.

---

## Checkpoint 6 — Locked evaluation and deployment gate

**Priority:** Final gate. Passing internal validation alone is insufficient.

### 6.1 Offline comparison

Snapshot VODs before collecting a fresh untouched set. Compare the current
bundle and challenger on the same fresh untouched-VOD manifest at their saved
thresholds. Do not retune thresholds or inspect errors before recording metrics.

The challenger may lose no more than 0.01 AP or AUC relative to the current
bundle unless a larger live benefit is explicitly reviewed and accepted.

### 6.2 Locked live comparison

Score the exact same locked confirmation peak windows with both models offline,
using each model's saved preprocessing. This avoids confounding model quality
with different live sessions. Use confirmation labels only for this decision.

Use telemetry replay to select sensitivity thresholds:

| Sensitivity | Maximum episodes/hour | Minimum decided acceptance |
|---|---:|---:|
| Strict | 2 | 80% |
| Balanced | 5 | 70% |
| Discovery | 10 | 60% |

The challenger must:

- improve positive-versus-hard-negative peak-score AUC on locked windows;
- meet Balanced acceptance and volume targets globally;
- have no streamer with at least 20 decisions below 50% acceptance;
- have confidence intervals and sample counts shown beside point estimates.

### 6.3 Deployment verification

Only after the model gates pass:

- export the challenger to a new bundle directory;
- verify JAX/ONNX logits and threshold decisions;
- verify Python/Go preprocessing parity;
- run Go unit and race tests;
- run representative historical replay;
- run an authenticated shadow smoke test.

Do not overwrite or repoint the current bundle until all gates pass. Remain
shadow-only; automatic Twitch clip creation is out of scope.

---

## Documentation updates during implementation

After every checkpoint:

1. update `docs/project_status.md` with verified counts, commands reported by the
   user, metrics, decisions, and the next checkpoint;
2. update `docs/training_playbook.md` when snapshots, manifests, event metrics,
   or experiment workflows become real;
3. update `docs/live_clipper.md` when telemetry, episode review, or sensitivity
   replay becomes real;
4. update `docs/twitch_classifier_brief.md` to remove the obsolete fixed
   `F1 > 0.75` completion target and clarify that sigmoid bounds the score but
   does not calibrate it as a probability;
5. update `AGENTS.md` only when the implemented pipeline or deployment contract
   changes.

The older plans remain historical rationale. This document is the ordered
implementation authority once work begins.
