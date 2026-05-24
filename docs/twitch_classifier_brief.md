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

/src
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

config.yaml               ← hyperparameters, streamer list, data paths
requirements.txt
```

---

## Data Collection Pipeline

### Step 1 — Positive Examples (Viral Moments)

Use the Twitch API to pull the most viewed clips for each target streamer:

```
GET https://api.twitch.tv/helix/clips?broadcaster_id={id}&first=100
```

For each clip, record:
- `clip_id`
- `vod_offset` — the timestamp in the VOD where the clip starts
- `vod_id` — which VOD it came from
- `view_count` — proxy for how viral it was

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
streamers:
  - name: stableronaldo
    broadcaster_id: "123456789"
    active: true        # set to false to pause without removing

  - name: jasontheween
    broadcaster_id: "987654321"
    active: true

  - name: someotherstreamer
    broadcaster_id: "111222333"
    active: false       # not watching this one right now
```

Only streamers with `active: true` get a goroutine spawned at startup. This lets you
control exactly who you are watching without touching any Go code.

### Per-streamer inference loop

Each goroutine runs this loop independently:
1. Maintains a rolling 30-second buffer of chat messages for its streamer
2. Every 2–3 seconds, encodes the current buffer into tokens + feature vector
3. Runs inference via onnxruntime-go
4. If score > 0.75 → wait 20-30 seconds → fire Twitch Clip API → set 45 second cooldown

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
