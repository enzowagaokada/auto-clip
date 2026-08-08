# Twitch Viral Clip Classifier — Project Brief

## What I Am Building

A binary classifier that watches a rolling window of Twitch chat messages and outputs
a ranking score (0–1) representing how strongly the chat resembles reactions around
historically clippable moments. The raw score is not assumed to be a calibrated
probability.

The classifier replaces hardcoded emote detection entirely. It learns what hype chat
*feels like* from historical data — so it generalizes to emotes, slang, and reaction
patterns I am not even aware of.

The product exposes the threshold as **clipping sensitivity**. A strict sensitivity
produces fewer, higher-confidence candidates; balanced and discovery sensitivities
produce progressively more candidates for the user to review. When a candidate crosses
the configured sensitivity threshold, the Go clipper can create a clip or send it to an
approval queue.

This is intentionally a candidate-ranking system rather than a guarantee that every
detection is worth clipping. Chat alone cannot always distinguish a meaningful
on-stream event from routine spam, raids, repeated jokes, or community memes that cause
the same reaction pattern. User approvals and rejections are therefore part of the
product and future training loop.

---



## The Core Problem

Given a 30-second window of Twitch chat messages, predict:

- **1** = this is a viral/clippable moment
- **0** = this is a normal/boring moment

This is **binary classification** on **sequential text data**.

---



## Why Sequence-Based (Not Just Counts)

A simple feature vector (emote counts, message rate) misses the temporal structure of
hype. Real viral moments have a shape:

- Slow baseline chat
- Sudden acceleration
- Burst of repeated reactions
- Then decay

A sequence model (GRU or small Transformer) sees the ordered stream of messages over
time and learns this shape — regardless of which specific emotes or words appear.
This means it generalizes to new emotes automatically.

---



## Tech Stack


| Layer                  | Tool                                            |
| ---------------------- | ----------------------------------------------- |
| Language               | Python 3.11+                                    |
| ML Framework           | JAX                                             |
| Neural Network Library | Flax (JAX-native, explicit parameter handling)  |
| Optimizer Library      | Optax                                           |
| Data processing        | Pandas, NumPy                                   |
| Tokenization           | Simple custom vocab or SentencePiece            |
| Experiment tracking    | Weights & Biases (wandb) — optional             |
| Model export           | ONNX (direct via `jax2onnx`) for Go integration |


---



## Project Structure

```
/data
  /raw                    ← raw clip metadata and chat logs from Twitch API
  /processed              ← tokenized, windowed, labeled dataset ready for training

/training                 ← Python ML environment
  /collect
    fetch_clips.py        ← pull top clips from Twitch API (positive examples)
    fetch_chat.py         ← fetch chat replay logs for each clip timestamp
    fetch_negatives.py    ← pull random non-clip VOD timestamps (negative examples)
    build_dataset.py      ← combine positives + negatives, window, label, save

  /features
    tokenizer.py          ← build vocab from chat, tokenize messages
    windowing.py          ← slice chat logs into fixed 30-second windows
    encode.py             ← encode each window into a sequence tensor

  /model
    architecture.py       ← GRU or small Transformer in Flax
    loss.py               ← weighted binary cross entropy
    train.py              ← training loop with JAX + Optax
    evaluate.py           ← precision, recall, F1, confusion matrix

  /export
    export_onnx.py        ← export a saved run to an ONNX deployment bundle
    verify_onnx.py        ← compare JAX and ONNX outputs

/clipper                  ← standalone Go module
  /cmd/autoclip           ← live shadow-clipper entrypoint
  /internal               ← Twitch, preprocessing, inference, candidate logging

/models/exports           ← generated ONNX deployment bundles

config.yaml               ← hyperparameters, streamer list, data paths
requirements.txt
```

---



## Project Roadmap / Phases

**Current phase:** Phase 4/5 — Baseline Model and Generalization  
**Current next step:** Rebuild the processed dataset, train with whole-VOD
validation, then run streamer-held-out evaluations. See
`docs/training_playbook.md`.

### Phase 1 — Raw Data Collection

Goal: collect positive and negative chat windows from Twitch VODs.

Status: **Implemented; collection remains an ongoing data-growth task**

Completed:

- `fetch_clips.py` fetches recent top clips from Twitch Helix.
- `fetch_clips.py` appends to `data/raw/clips.csv` and deduplicates by `clip_id`
instead of overwriting prior collection runs.
- `fetch_negatives.py` exists for sampling non-clip moments from the same VODs.
- The latest observed `fetch_clips.py` run reported 343 unique clips and 153 newly
added clips for `stableronaldo` and `jasontheween`.

Next steps:

- Run `python training/collect/fetch_chat.py` to fetch positive chat windows into
`data/raw/chat/`.
- Run `python training/collect/fetch_negatives.py` to fetch negative chat windows
into `data/raw/chat_negatives/`.
- Inspect the resulting JSON counts before building the processed dataset.



### Phase 2 — Processed Dataset

Goal: turn raw chat JSON into a labeled ML dataset.

Implemented:

- Create `training/collect/build_dataset.py`.
- Combine `data/raw/chat/` as `label = 1`.
- Combine `data/raw/chat_negatives/` as `label = 0`.
- Save processed examples to `data/processed/dataset.jsonl`.
- Include basic metadata and features such as streamer name, VOD ID, target offset,
message count, messages per second, unique users, and label.



### Phase 3 — Tokenization and Encoding

Goal: convert chat text into model-ready tensors.

Implemented:

- Create a tokenizer/vocabulary from the collected chat corpus.
- Encode messages with `[PAD]`, `[UNK]`, and `[SEP]`.
- Compute extra features:
  - messages per second
  - unique users
  - normalized stream time
- Save encoded arrays under `data/processed/`.



### Phase 4 — Baseline Model

Goal: train the first JAX/Flax GRU classifier.

Implemented:

- Implement `training/model/architecture.py`.
- Implement weighted binary cross entropy.
- Implement the training loop with Optax.
- Track precision, recall, F1, confusion matrix, and AUC.



### Phase 5 — Evaluation and Generalization

Goal: prove the model works beyond one streamer.

Planned:

- Run streamer-held-out validation.
- Track metrics per streamer.
- Tune `clip_threshold` per streamer.
- Add calibration/suggestion mode for new streamers.



### Phase 6 — Export and Inference

Goal: make the model usable outside Python.

Planned:

- Export the trained model to ONNX.
- Verify ONNX output matches JAX output.
- Export the vocabulary file alongside the model.
- Build `training/inference/predict.py`.



### Phase 7 — Go Live Clipper

Goal: use the trained ONNX model in a real-time Go clipper.

Planned:

- Connect to live Twitch chat.
- Maintain a rolling 35-second buffer and score the clip-start-equivalent target
at `now - 30s`.
- Run ONNX inference every 2-3 seconds.
- Log deduplicated candidates in shadow mode.
- Respect cooldown and per-streamer thresholds.



### Phase 8 — Product / Business Layer

Goal: turn the classifier into a commercial clipping product.

Planned:

- Add per-streamer clipping-sensitivity calibration.
- Offer user-facing presets such as **Strict**, **Balanced**, and **Discovery** rather
than presenting the raw model score as a probability.
- Add an approval queue or Discord alerts so users can review candidate clips.
- Track expected candidates per stream-hour and accepted-candidate rate for each
sensitivity preset.
- Feed approvals, rejections, and ambiguous candidates into persistent review data for
retraining and optional streamer-specific personalization.
- Add vertical clip formatting and captions.
- Add managed streamer/agency workflow.

---



## Data Collection Pipeline



### Step 1 — Positive Examples (Viral Moments)

Use the Twitch API to pull the most viewed clips for each target streamer within a
recent time window (older top clips often have expired VODs and cannot be used for
chat replay):

```
GET https://api.twitch.tv/helix/clips?broadcaster_id={id}&first=100&started_at={iso}&ended_at={iso}
```

`fetch_clips.py` reads `twitch.clips` from `config.yaml`, paginates until
`max_per_streamer` is reached, and only keeps clips that still have VOD data.

For each clip, record:

- `clip_id`
- `vod_offset` — the timestamp where the clip video starts on the VOD; it is
**not** the time the viewer pressed Clip
- `vod_id` — which VOD it came from
- `view_count` — proxy for how viral it was
- `duration` — clip-video duration, retained as metadata



#### Tuning clip collection (`config.yaml` → `twitch.clips`)


| Parameter          | Default | What it does                                                              | When to change it                                                                                                                                                                        |
| ------------------ | ------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `days_back`        | `30`    | Only fetch clips created in the last N days                               | **Lower** (e.g. `14`) if many clips are missing VODs — VODs expire on Twitch, so recent clips survive longer. **Raise** (e.g. `60`) if you want more history, but expect more dead VODs. |
| `max_per_streamer` | `100`   | Cap how many clips to fetch per active streamer (paginates automatically) | **Raise** (e.g. `200`) when you need more training positives. **Lower** if you want a quick test run or less chat to download.                                                           |


**Symptoms and fixes:**

- `Found 100 clips, kept 3 with VOD data` → window is too wide or clips are too old; **lower** `days_back`
- `Found 20 clips, kept 20` → streamer had few clips in that window; **raise** `days_back` or `**max_per_streamer`**
- Script runs fine but dataset feels small → **raise** `max_per_streamer` across more streamers

Then fetch the chat replay for a window around each clip's `vod_offset`. Because the Helix `/comments` endpoint is deprecated, use the Twitch public GraphQL API (`https://gql.twitch.tv/gql`) with the `VideoCommentsByOffsetOrCursor` query.

```graphql
query VideoCommentsByOffsetOrCursor($videoID: ID!, $contentOffsetSeconds: Int) {
    video(id: $videoID) {
        comments(contentOffsetSeconds: $contentOffsetSeconds, first: 100) {
            edges {
                node {
                    id
                    createdAt
                    contentOffsetSeconds
                    commenter { displayName }
                    message { fragments { text } }
                }
            }
            pageInfo { hasNextPage }
        }
    }
}
```

Fetch the fixed 35-second range `[vod_offset - 5s, vod_offset + 30s]`. This
captures five seconds before the clip video starts plus the first 30 seconds of
the clipped video, where the highlight and chat reaction actually occur.

Clip durations in the collected data range from roughly 5–60 seconds. Duration
is retained in raw metadata, but it does not change model-window length: a
variable `[offset - 5s, offset + duration]` range would break the seven fixed
5-second buckets and could not exactly match one fixed live inference buffer.

Label these windows: **y = 1**

#### Gotchas with the GQL chat endpoint (learned the hard way)

This is an **undocumented, unstable** internal API. Two failure modes hit during
development, both of which `fetch_chat.py` now handles:

1. **Variable types:** the schema expects `contentOffsetSeconds: Int` (not `Float`)
  and, if you use it, `cursor: Cursor` (not `String`). Wrong types make *every*
   request fail GraphQL validation, which looks like "no messages" unless you print
   the `errors` array. `fetch_chat.py` now logs GraphQL errors explicitly.
2. **Integrity check (**`IntegrityCheckFailed`**):** **cursor-based** pagination trips
  Twitch's anti-bot challenge (KPSDK/Kasada), which needs a real browser to solve.
   **Offset-based** pagination does not. So `fetch_chat.py` paginates by re-querying
   with the last message's `contentOffsetSeconds` and dedupes overlapping pages by
   message `id` — never using the cursor.



#### Alternative: `chat-downloader` (pip)

We are **currently using the raw GQL function** above. A maintained alternative worth
considering if the GQL approach keeps breaking is the `chat-downloader` pip package
(`pip install chat-downloader`). It is pure Python, importable as a library, supports
Twitch VOD chat with offset windows, and offloads the maintenance of Twitch's internal
quirks to its maintainers.

Trade-off: it hits the *same* private Twitch endpoints, so it is not immune to Twitch
changes — the difference is who owns the fix. If we swap, replace only the internals of
`fetch_chat_window()` (keep the same inputs/outputs and JSON schema) so the rest of the
`clips.csv`-driven pipeline and any already-downloaded files stay valid.

`TwitchDownloaderCLI` is also excellent but is an external .NET binary oriented toward
full-VOD chat dumps, which is a heavier fit for our windowed (35s-per-clip) sampling.

### Step 2 — Negative Examples (Normal Moments)

For each VOD that had clips, sample random timestamps that are:

- Not within 60 seconds of any clip timestamp (add buffer so you don't accidentally
label a viral moment as negative)
- From parts of the stream with roughly average chat activity

Fetch the same 35-second chat window for each negative timestamp.

Label these windows: **y = 0**

### Step 3 — Class Balance

You will have far more negatives than positives. A streamer might clip 20 moments
in an 8-hour stream — that's maybe 1% of all possible windows being positive.

Handle this two ways:

1. **Undersample negatives** at data collection time — don't collect 10,000 negatives
  if you only have 200 positives. Aim for roughly 3:1 or 4:1 negative:positive ratio.
2. **Weighted loss** at training time — penalize false negatives more (see Loss section).



### Scaling Data Collection (Future Work)

Collection is currently slow because it is fully sequential: one clip at a time, one
100-message page at a time, each a separate network round trip, with deliberate sleeps
between requests. For the current dataset size (hundreds of clips) this is fine — it is
a one-time, resumable, offline batch job. **Do not optimize this yet.** Let it run in
the background.

The real wall at scale is not code speed but Twitch's **~10,000-messages-per-IP rate
limit** and bot detection. When much more training data is needed, escalate in this
order (cheapest/lowest-risk first):

1. **Let it run unattended.** Resumable + skips existing files, so start it and walk
  away. Gets you to low thousands of clips without any code changes.
2. **Concurrency + backoff.** Process 3–4 clips in parallel with exponential-backoff
  retry on `429`/integrity errors. ~3–4× wall-clock improvement, but does not raise the
   per-IP ceiling — it just reaches it faster.
3. **Whole-VOD download + local windowing (highest leverage).** Instead of many tiny
  35s windows (which re-request overlapping regions of the same VOD and waste the
   rate-limit budget), download each VOD's full chat once and slice many positive and
   negative windows out of it locally. Far better messages-per-request efficiency.
4. **Spread across IPs.** Residential/rotating proxies or sharding the clip list across
  multiple machines/VMs. This is the actual lever for large volume.
5. **Offload to a paid scraper.** The Apify "Twitch VOD Chat Archive" actor (~$1.05 per
  1,000 messages) handles TLS fingerprinting, proxy rotation, and the integrity dance.
   Worth it for a big one-time pull; not for routine top-ups.

Recommended trajectory: **now** → option 1; **low thousands** → options 2 + 3;
**tens of thousands+** → option 4 or 5. Do not build any of this until the model is
proven on the current dataset.

---



## Input Representation

Each training sample is a **sequence of chat messages** plus numeric temporal
features from a fixed 35-second window spanning five seconds before through 30
seconds after the clip-start/negative anchor.

### Tokenization

Build a vocabulary from the collected chat corpus. Include:

- Common words and slang
- Emote names (they appear as plain text in IRC — `MINIONLAUGH`, `KEKW`, etc.)
- Special tokens: `[PAD]`, `[UNK]`, `[SEP]` (separator between messages)

Each message in the window gets tokenized and concatenated with `[SEP]` between them:

```
["MINIONLAUGH", "[SEP]", "no", "way", "bro", "[SEP]", "MINIONLAUGH", "MINIONLAUGH", ...]
```

Cap the total sequence length (e.g. 512 tokens). Pad shorter windows.

#### `[UNK]`, new emotes, and future-proofing

Any token not in the vocabulary (rare words, typos, or emotes too new/infrequent to
clear `min_freq`) maps to `[UNK]`. This is not just error handling — it has real
implications for how the model handles emotes that go viral *after* training:

- **The structural signal survives even when an emote is** `[UNK]`**.** This is precisely
why a sequence model "generalizes to emotes we are not aware of." When a brand-new
emote trends and hundreds of users spam it, the model sees `[UNK] [SEP] [UNK] [SEP] [UNK] ...` — but the *shape* (rapid repetition, high message velocity, many unique
users) is the exact hype pattern the GRU learns. So a new viral emote can still fire
the classifier immediately, as `[UNK]`, because the model keys off the burst pattern,
not the specific token identity.
- **The specific identity is lost until we rebuild.** `[UNK]` cannot learn that one
particular new emote is individually high-signal. To "promote" a new emote from
`[UNK]` into its own learned token, periodically re-run the pipeline
(`build_dataset.py` -> `encode.py` rebuilds the vocab from fresh data) and retrain.
This is a normal maintenance loop, not a redesign — the data pipeline already
supports it.

Takeaway: today's `[UNK]` may be tomorrow's high-signal emote. The structural features
keep us covered in the short term; a periodic vocab-rebuild + retrain captures new
emotes' specific identities over time.

### Additional Features (concatenated after sequence encoding)

- Messages per second (velocity)
- Unique users in window
- Time since stream started (divided by a fixed 12-hour scale and clipped)
- Seven 5-second message-rate buckets
- Late-window versus early-window rate change
- Peak 5-second rate
- Repeated-message ratio

These get concatenated onto the final hidden state before the classification head.
The bucket features are required for learning acceleration and decay: token order
alone contains no message timestamps.

---



## Model Architecture

Use a **GRU** (Gated Recurrent Unit) as the sequence encoder. It's simpler than a
Transformer, trains faster on smaller datasets, and is very well suited to
sequential chat data where order and timing matter.

Upgrade to a small Transformer later if GRU performance plateaus.

```python
import flax.linen as nn
import jax
import jax.numpy as jnp

class ChatClassifier(nn.Module):
    vocab_size: int
    embed_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, tokens, extra_features, training=False):
        # embed tokens
        x = nn.Embed(self.vocab_size, self.embed_dim)(tokens)  # (seq_len, embed_dim)

        # run GRU over sequence
        gru = nn.RNN(nn.GRUCell(self.hidden_dim))
        x = gru(x)                    # (seq_len, hidden_dim)
        hidden = x[:, -1, :]          # take final hidden state (batch, hidden_dim)

        # concatenate extra features
        combined = jnp.concatenate([hidden, extra_features], axis=-1)

        # classification head
        out = nn.Dense(64)(combined)
        out = nn.relu(out)
        out = nn.Dropout(rate=0.3)(out, deterministic=not training)
        out = nn.Dense(1)(out)
        return out.squeeze(-1)        # logits; apply sigmoid only for evaluation/inference


# Flax requires explicit param init — weights live in a separate dict, not the model
key = jax.random.PRNGKey(0)
model = ChatClassifier(vocab_size=10000, embed_dim=64, hidden_dim=128)
dummy_tokens = jnp.zeros((1, 512), dtype=jnp.int32)
dummy_features = jnp.zeros((1, 3), dtype=jnp.float32)
params = model.init(key, dummy_tokens, dummy_features)  # params is a dict

# inference
logits = model.apply(params, dummy_tokens, dummy_features)
probability = jax.nn.sigmoid(logits)
```



### Why These Activation Functions

- **ReLU** in hidden layers — introduces non-linearity, lets the model learn complex
patterns, kills negative values (prevents vanishing gradients better than tanh)
- **Sigmoid** on the output layer — squashes output to [0, 1] so it's a valid
probability score

---



## Loss Function

**Weighted Binary Cross Entropy**

```python
def weighted_bce_loss(params, model, x_tokens, x_features, y, key):
    # run batch through model — Flax passes params separately via apply()
    logits = model.apply(
        params, x_tokens, x_features, training=False,
        rngs={"dropout": key}
    )
    loss = optax.sigmoid_binary_cross_entropy(logits, y)
    weights = 1.0 + y * (pos_weight - 1.0)
    return jnp.mean(loss * weights)
```

Calculate `pos_weight` as negative/positive count from the training split only.
Logit-space BCE is numerically stable and avoids clipping saturated
probabilities.

---



## Training Loop

```python
import optax
import flax.linen as nn
from flax.training import train_state

# hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 20
POS_WEIGHT = 4.0

# Flax uses a TrainState to bundle params + optimizer state together
optimizer = optax.adam(LEARNING_RATE)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

@jax.jit
def train_step(state, x_tokens, x_features, y, key):
    def loss_fn(params):
        preds = state.apply_fn(
            params, x_tokens, x_features, training=True,
            rngs={"dropout": key}
        )
        loss = -(
            POS_WEIGHT * y * jnp.log(preds + 1e-7) +
            (1 - y) * jnp.log(1 - preds + 1e-7)
        )
        return jnp.mean(loss)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# training loop
for epoch in range(EPOCHS):
    for batch in dataloader:
        key, subkey = jax.random.split(key)
        state, loss = train_step(
            state,
            batch["tokens"], batch["features"], batch["label"], subkey
        )
    print(f"Epoch {epoch} loss: {loss:.4f}")
```

---



## Evaluation Metrics

Do NOT use accuracy — it's meaningless with class imbalance.

Use:

- **Precision** — of moments the model called viral, how many actually were?
- **Recall** — of actual viral moments, how many did the model catch?
- **F1 Score** — threshold-specific balance of precision and recall
- **Average precision (AP)** — prevalence-aware ranking metric used for checkpoint selection
- **Confusion matrix** — visualize false positives vs false negatives
- **AUC-ROC** — threshold-independent measure of classifier quality

Tune the classification threshold based on the user's clipping sensitivity:

- **Strict / higher threshold** → fewer candidates and less review work, but more
potentially good moments are missed.
- **Balanced / middle threshold** → compromise between candidate quality and coverage.
- **Discovery / lower threshold** → more candidates and higher recall, but more review
work.

Do not display the raw threshold as a literal confidence percentage unless the model is
calibrated on representative live-stream data. Prefer user-facing sensitivity presets
and expected candidates per hour. Offline precision is measured on an intentionally
enriched positive/negative dataset and will not equal live production precision, where
true clip moments are much rarer.

### Streamer-Held-Out Validation

Do not only split windows randomly across the whole dataset. That can make the
model look better than it really is because clips from the same streamer, VOD, and
chat culture may appear in both train and validation.

The key generalization test is:

> Can the model work on a streamer it did not see during training?

Evaluate this by holding out one or more entire streamers from training and reporting
precision, recall, F1, and false-positive rate on those unseen channels. Also track
metrics per streamer, because a single global F1 can hide that the model works well
for one community and poorly for another.

---



## Generalization Across Streamers

Streamer chats differ heavily by community, emotes, inside jokes, baseline message
speed, sarcasm, and what the audience considers clippable. A classifier trained on
only one or two streamers may overfit to those communities and fail on unseen
channels.

Use a two-layer strategy:

1. **Global base model** — train on clips and chat windows from many streamers across
  categories. This model learns universal clippability signals such as chat
   acceleration, repeated reactions, user participation bursts, and hype decay.
2. **Per-streamer sensitivity calibration** — tune lightweight settings per streamer
  instead of retraining the full model by default. Examples include the thresholds
  behind Strict/Balanced/Discovery, baseline chat velocity, minimum unique users,
  cooldown duration, post-detection delay, and streamer-specific emote vocabulary.

For new streamers, start in calibration/suggestion mode:

1. Run the global model for several streams without fully trusting automation.
2. Save candidate high-score moments.
3. Compare predictions against actual Twitch clips, manual approvals, and rejected
  candidates.
4. Measure candidates per stream-hour and accepted-candidate rate at each sensitivity.
5. Adjust streamer-specific thresholds, cooldowns, and minimum activity requirements.
6. Feed approved/rejected moments back into future training data.

For high-value customers, offer optional custom fine-tuning on that streamer's
historical clips and chat logs. This becomes a paid product feature: generic AI
clippers treat every stream the same, while this system learns the streamer's
specific chat culture.

---



## Hyperparameters to Tune


| Parameter        | Starting Value | Notes                                                  |
| ---------------- | -------------- | ------------------------------------------------------ |
| `embed_dim`      | 32             | Raise only after learning curves justify more capacity |
| `hidden_dim`     | 64             | Raise only after learning curves justify more capacity |
| `learning_rate`  | 1e-3           | Adam default, reduce if unstable                       |
| `dropout`        | 0.3            | Regularization, increase if overfitting                |
| `pos_weight`     | auto           | Training-split negative:positive ratio                 |
| `batch_size`     | 32             | Increase if training is slow                           |
| `window_seconds` | 35             | Fixed chat window size                                 |
| `max_seq_len`    | 512            | Token sequence cap                                     |
| `clip_threshold` | 0.75           | Inference threshold for triggering clip                |


---



## Model Export to ONNX (for Go integration)

`jax2tf(..., enable_xla=False) -> tf2onnx` is no longer the supported path:
JAX deprecated conversion to TensorFlow ops and recommends StableHLO for its
official export surface. This project instead uses `jax2onnx`, whose Flax Linen
coverage includes `RNN` and `GRUCell`.

`training/export/export_onnx.py` loads a complete saved run and exports an
inference-only forward function. The function applies the exact saved
`flax.linen.GRUCell` equations explicitly with `jax.lax.scan`, because current
`jax2onnx` tracing turns a shape inside Linen's lifted `nn.RNN` into a
`JitTracer`. Before export, the script asserts this explicit forward matches
the original `model.apply(..., training=False)` logits.

The exported contract is:

- `tokens`: `(batch, 512)` `int32`;
- `features`: `(batch, 13)` `float32`;
- `logits`: `(batch,)` `float32`.

The generated bundle also contains the saved vocabulary, inference metadata,
and a checksum manifest. `training/export/verify_onnx.py` must compare real
validation rows in JAX and ONNX Runtime before the bundle is used by Go.

---



## Go Integration (how the classifier plugs into the clipper)



### Shared connection, isolated streamer processors

Use one Twitch EventSub WebSocket and subscribe it to every active configured
channel. A router dispatches notifications to per-streamer processors, each
with its own chat buffer, inference state, cooldown, and counters. Twitch
explicitly recommends one WebSocket connection until another is needed.

### Choosing which streamers to watch

Streamers are configured in `config.yaml` — not hardcoded. To add or remove a
streamer, edit the file and restart the app. No recompile needed.

```yaml
twitch:
  streamers:
    - name: stableronaldo
      broadcaster_id: "123456789"
      active: true        # set to false to pause without removing
      clip_threshold: 0.82
      cooldown_seconds: 75

    - name: jasontheween
      broadcaster_id: "987654321"
      active: true
      clip_threshold: 0.78
      cooldown_seconds: 60

    - name: someotherstreamer
      broadcaster_id: "111222333"
      active: false       # not watching this one right now
      clip_threshold: 0.90
      cooldown_seconds: 120
```

Only streamers with `active: true` are subscribed and receive a per-stream
session. One shared EventSub/event loop owns those isolated buffers, detector
state, cooldowns, and counters; it does not spawn one connection or inference
goroutine per streamer.

Per-streamer threshold and cooldown overrides let the same global model adapt to
different chat cultures.

### Per-streamer inference state

On each shared inference tick, every live session:

1. Maintains a rolling 35-second buffer ending at the current notification time
2. Every 2–3 seconds, scores the clip-start-equivalent target at `now - 30s`
3. Runs inference via onnxruntime-go
4. On a below-to-above threshold crossing, writes a shadow candidate and starts
  the cooldown; the detector must fall below threshold before it can rearm

The vocabulary file (token → int mapping) gets shipped alongside the ONNX model so
Go can tokenize identically to how Python tokenized during training.

### Why live inference has a 30-second target lag

Helix `vod_offset` identifies where the clip video starts, not when Clip was
pressed. Historical positives therefore span
`[clip start - 5s, clip start + 30s]`. At live time, the same fixed 35-second
buffer `[now - 35s, now]` scores the clip-start-equivalent target at
`now - 30s`. This alignment is about covering the clipped video and reaction,
not an assumption that chat always peaks exactly five seconds after a target.

The first live milestone is shadow-only. A later Clip API sink can add a
separate post-detection delay after live acceptance is high enough. Twitch's
current Create Clip behavior uses a 90-second capture window and publishes up
to the last 30 seconds by default; the old `has_delay` parameter has been
removed and must not be used.

---



## Definition of Done

- [x] Data collection scripts pull top clips and chat replays for stableronaldo and jasontheween
- [x] Negative examples collected and dataset balanced
- [x] Tokenizer built from collected chat corpus
- [x] Chat windows encoded into sequence tensors
- [x] GRU classifier implemented in Flax
- [x] Weighted logit-space BCE loss implemented
- [x] Training loop runs without NaN loss
- [ ] F1 score > 0.75 on held-out validation set
- [ ] Streamer-held-out validation confirms the model generalizes to unseen channels
- [ ] Strict/Balanced/Discovery sensitivity presets calibrated in shadow mode
- [ ] Per-streamer sensitivity settings documented and loaded by the Go clipper
- [ ] Approval/rejection feedback persists for future retraining
- [x] Model exported to ONNX successfully
- [x] Inference script confirms ONNX output matches JAX model output
- [x] Vocabulary file exported alongside ONNX model for Go tokenization

---



## What NOT to Build

- No Twitch IRC connection in the Python project — this project only consumes historical VOD chat data
- No real-time inference in Python — that happens in Go via ONNX at runtime
- No dashboard or UI
- Do not use PyTorch — use JAX + Flax throughout
- Do not use Equinox

