# Twitch Viral Clip Classifier — Project Brief

## What I Am Building

A binary classifier that watches a rolling window of Twitch chat messages and outputs
a probability score (0–1) representing how likely the current moment is clippable/viral.

The classifier replaces hardcoded emote detection entirely. It learns what hype chat
*feels like* from historical data — so it generalizes to emotes, slang, and reaction
patterns I am not even aware of.

When the score crosses a threshold, the Go clipper fires the Twitch Clip API.

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

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| ML Framework | JAX |
| Neural Network Library | Flax (JAX-native, explicit parameter handling) |
| Optimizer Library | Optax |
| Data processing | Pandas, NumPy |
| Tokenization | Simple custom vocab or SentencePiece |
| Experiment tracking | Weights & Biases (wandb) — optional |
| Model export | ONNX (via jax2tf → tf2onnx) for Go integration |

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
    export.py             ← export trained model to ONNX

  /inference
    predict.py            ← load model, run inference on a chat window

/cmd
  /autoclip               ← Go application entrypoint

/models                   ← Exported ONNX and vocab files

config.yaml               ← hyperparameters, streamer list, data paths
requirements.txt
```

---

## Project Roadmap / Phases

**Current phase:** Phase 1 — Raw Data Collection  
**Current next step:** Run `python training/collect/fetch_chat.py` from the repository
root to fetch positive chat windows for the collected clips.

### Phase 1 — Raw Data Collection

Goal: collect positive and negative chat windows from Twitch VODs.

Status: **In progress**

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

Planned:
- Create `training/collect/build_dataset.py`.
- Combine `data/raw/chat/` as `label = 1`.
- Combine `data/raw/chat_negatives/` as `label = 0`.
- Save processed examples to `data/processed/dataset.jsonl`.
- Include basic metadata and features such as streamer name, VOD ID, target offset,
  message count, messages per second, unique users, and label.

### Phase 3 — Tokenization and Encoding

Goal: convert chat text into model-ready tensors.

Planned:
- Create a tokenizer/vocabulary from the collected chat corpus.
- Encode messages with `[PAD]`, `[UNK]`, and `[SEP]`.
- Compute extra features:
  - messages per second
  - unique users
  - normalized stream time
- Save encoded arrays under `data/processed/`.

### Phase 4 — Baseline Model

Goal: train the first JAX/Flax GRU classifier.

Planned:
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
- Maintain a rolling 30-second buffer per streamer.
- Run ONNX inference every 2-3 seconds.
- Trigger the Twitch Clip API after the configured delay.
- Respect cooldown and per-streamer thresholds.

### Phase 8 — Product / Business Layer

Goal: turn the classifier into a commercial clipping product.

Planned:
- Add per-streamer calibration.
- Add approval queue or Discord alerts.
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
- `vod_offset` — the timestamp in the VOD where the clip starts
- `vod_id` — which VOD it came from
- `view_count` — proxy for how viral it was

#### Tuning clip collection (`config.yaml` → `twitch.clips`)

| Parameter | Default | What it does | When to change it |
|---|---|---|---|
| `days_back` | `30` | Only fetch clips created in the last N days | **Lower** (e.g. `14`) if many clips are missing VODs — VODs expire on Twitch, so recent clips survive longer. **Raise** (e.g. `60`) if you want more history, but expect more dead VODs. |
| `max_per_streamer` | `100` | Cap how many clips to fetch per active streamer (paginates automatically) | **Raise** (e.g. `200`) when you need more training positives. **Lower** if you want a quick test run or less chat to download. |

**Symptoms and fixes:**
- `Found 100 clips, kept 3 with VOD data` → window is too wide or clips are too old; **lower `days_back`**
- `Found 20 clips, kept 20` → streamer had few clips in that window; **raise `days_back`** or **`max_per_streamer`**
- Script runs fine but dataset feels small → **raise `max_per_streamer`** across more streamers

Then fetch the chat replay for a window around each clip's `vod_offset`. Because the Helix `/comments` endpoint is deprecated, use the Twitch public GraphQL API (`https://gql.twitch.tv/gql`) with the `VideoCommentsByOffsetOrCursor` query.

```graphql
query VideoCommentsByOffsetOrCursor($videoID: ID!, $contentOffsetSeconds: Float, $cursor: String) {
    video(id: $videoID) {
        comments(contentOffsetSeconds: $contentOffsetSeconds, after: $cursor, first: 100) {
            edges {
                cursor
                node {
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

Fetch from `offset - 30s` to `offset + 5s` — you want the chat leading up to and
during the moment, not after.

Label these windows: **y = 1**

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

---

## Input Representation

Each training sample is a **sequence of chat messages** in a 30-second window.

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

### Additional Features (concatenated after sequence encoding)

- Messages per second (velocity)
- Unique users in window
- Time since stream started (normalized)

These get concatenated onto the final hidden state before the classification head.

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
        out = nn.sigmoid(out)         # output: probability 0-1
        return out.squeeze(-1)


# Flax requires explicit param init — weights live in a separate dict, not the model
key = jax.random.PRNGKey(0)
model = ChatClassifier(vocab_size=10000, embed_dim=64, hidden_dim=128)
dummy_tokens = jnp.zeros((1, 512), dtype=jnp.int32)
dummy_features = jnp.zeros((1, 3), dtype=jnp.float32)
params = model.init(key, dummy_tokens, dummy_features)  # params is a dict

# inference
output = model.apply(params, dummy_tokens, dummy_features)
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
    preds = model.apply(
        params, x_tokens, x_features, training=False,
        rngs={"dropout": key}
    )
    preds = jnp.squeeze(preds)

    # weighted BCE: penalize missing a viral moment more than false alarming
    loss = -(
        pos_weight * y * jnp.log(preds + 1e-7) +
        (1 - y) * jnp.log(1 - preds + 1e-7)
    )
    return jnp.mean(loss)
```

`pos_weight` should reflect your class ratio. If you have 4x more negatives than
positives, start with `pos_weight = 4.0` and tune from there.

The `1e-7` epsilon prevents `log(0)` from producing NaN or -inf.

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
- **F1 Score** — harmonic mean of precision and recall, the primary metric
- **Confusion matrix** — visualize false positives vs false negatives
- **AUC-ROC** — threshold-independent measure of classifier quality

Tune the classification threshold (default 0.5) based on your preference:
- Lower threshold → more clips, more false positives
- Higher threshold → fewer clips, might miss things

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
2. **Per-streamer calibration** — tune lightweight settings per streamer instead of
   retraining the full model by default. Examples include `clip_threshold`, baseline
   chat velocity, minimum unique users, cooldown duration, post-detection delay, and
   streamer-specific emote vocabulary.

For new streamers, start in calibration/suggestion mode:

1. Run the global model for several streams without fully trusting automation.
2. Save candidate high-score moments.
3. Compare predictions against actual Twitch clips, manual approvals, and rejected
   candidates.
4. Adjust streamer-specific thresholds, cooldowns, and minimum activity requirements.
5. Feed approved/rejected moments back into future training data.

For high-value customers, offer optional custom fine-tuning on that streamer's
historical clips and chat logs. This becomes a paid product feature: generic AI
clippers treat every stream the same, while this system learns the streamer's
specific chat culture.

---

## Hyperparameters to Tune

| Parameter | Starting Value | Notes |
|---|---|---|
| `embed_dim` | 64 | Embedding size per token |
| `hidden_dim` | 128 | GRU hidden state size |
| `learning_rate` | 1e-3 | Adam default, reduce if unstable |
| `dropout` | 0.3 | Regularization, increase if overfitting |
| `pos_weight` | 4.0 | Match your negative:positive ratio |
| `batch_size` | 32 | Increase if training is slow |
| `window_seconds` | 30 | Chat window size |
| `max_seq_len` | 512 | Token sequence cap |
| `clip_threshold` | 0.75 | Inference threshold for triggering clip |

---

## Model Export to ONNX (for Go integration)

After training, export the model so the Go clipper can run inference without Python:

```python
# export.py
import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
import tensorflow as tf
import tf2onnx

# trace the model
dummy_tokens = jnp.zeros((1, MAX_SEQ_LEN), dtype=jnp.int32)
dummy_features = jnp.zeros((1, 3), dtype=jnp.float32)

tf_fn = jax2tf.convert(
    lambda tok, feat: model(tok, feat, jax.random.PRNGKey(0), inference=True),
    enable_xla=False
)

# save as ONNX
onnx_model, _ = tf2onnx.convert.from_function(
    tf_fn,
    input_signature=[
        tf.TensorSpec((1, MAX_SEQ_LEN), tf.int32),
        tf.TensorSpec((1, 3), tf.float32)
    ]
)

with open("chat_classifier.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

The Go clipper then loads `chat_classifier.onnx` via `onnxruntime-go` and runs
inference locally — no Python needed at runtime.

---

## Go Integration (how the classifier plugs into the clipper)

### One goroutine per streamer

Each streamer gets its own goroutine — a lightweight concurrent process in Go. They
all run in parallel, completely independently. If you are watching 5 streamers, 5
goroutines are running simultaneously, each with its own chat buffer, inference loop,
and cooldown timer. They do not share state or interfere with each other.

```go
func main() {
    streamers := loadStreamersFromConfig()  // reads config.yaml

    var wg sync.WaitGroup
    for _, streamer := range streamers {
        wg.Add(1)
        go func(s StreamerConfig) {
            defer wg.Done()
            watchStreamer(s)  // blocks forever, runs the full loop for this streamer
        }(streamer)
    }
    wg.Wait()
}
```

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
      min_unique_users: 30
      cooldown_seconds: 75
      post_detection_delay_seconds: 25

    - name: jasontheween
      broadcaster_id: "987654321"
      active: true
      clip_threshold: 0.78
      min_unique_users: 40
      cooldown_seconds: 60
      post_detection_delay_seconds: 25

    - name: someotherstreamer
      broadcaster_id: "111222333"
      active: false       # not watching this one right now
      clip_threshold: 0.90
      min_unique_users: 80
      cooldown_seconds: 120
      post_detection_delay_seconds: 30
```

Only streamers with `active: true` get a goroutine spawned at startup. This lets you
control exactly who you are watching without touching any Go code.

Per-streamer settings let the same global model adapt to different chat cultures. A
chaotic chat may need a higher `clip_threshold` and `min_unique_users`, while a quieter
chat may need a lower threshold so the model does not miss genuinely important moments.

### Per-streamer inference loop

Each goroutine runs this loop independently:
1. Maintains a rolling 30-second buffer of chat messages for its streamer
2. Every 2–3 seconds, encodes the current buffer into tokens + feature vector
3. Runs inference via onnxruntime-go
4. If score > the streamer's configured `clip_threshold` → wait `post_detection_delay_seconds` → fire Twitch Clip API → set `cooldown_seconds`

The vocabulary file (token → int mapping) gets shipped alongside the ONNX model so
Go can tokenize identically to how Python tokenized during training.

### Why the Delay Before Clipping

Twitch clips save approximately the last 2 minutes of stream content when triggered.
If you clip immediately when the model fires, you capture the buildup but miss the
aftermath — the streamer's delayed reaction, chat spam peaking, the follow-up moment.

By waiting 20-30 seconds after detection before firing the clip API, you capture:
- ~90 seconds of buildup leading into the moment
- The moment itself
- 20-30 seconds of streamer and chat reaction afterward

The aftermath is often the funniest part. The delay is worth it.

---

## Definition of Done

- [ ] Data collection scripts pull top clips and chat replays for stableronaldo and jasontheween
- [ ] Negative examples collected and dataset balanced
- [ ] Tokenizer built from collected chat corpus
- [ ] Chat windows encoded into sequence tensors
- [ ] GRU classifier implemented in Flax
- [ ] Weighted BCE loss implemented
- [ ] Training loop runs without NaN loss
- [ ] F1 score > 0.75 on held-out validation set
- [ ] Streamer-held-out validation confirms the model generalizes to unseen channels
- [ ] Per-streamer calibration settings documented and loaded by the Go clipper
- [ ] Model exported to ONNX successfully
- [ ] Inference script confirms ONNX output matches JAX model output
- [ ] Vocabulary file exported alongside ONNX model for Go tokenization

---

## What NOT to Build

- No Twitch IRC connection in the Python project — this project only consumes historical VOD chat data
- No real-time inference in Python — that happens in Go via ONNX at runtime
- No dashboard or UI
- Do not use PyTorch — use JAX + Flax throughout
- Do not use Equinox
