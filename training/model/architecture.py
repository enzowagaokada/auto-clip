"""Flax GRU classifier for Twitch chat windows.

embed tokens -> GRU over the sequence -> take final hidden state ->
concatenate the numeric features -> dense head -> sigmoid probability.

Sequences are left-padded (see training/features/tokenizer.py), so the final
timestep is always a real token, which is what we read for classification.
"""

import flax.linen as nn
import jax.numpy as jnp


class ChatClassifier(nn.Module):
    vocab_size: int
    embed_dim: int = 64
    hidden_dim: int = 128
    num_features: int = 3
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, tokens, features, training: bool = False):
        # tokens: (batch, seq_len) int32 ; features: (batch, num_features) float32
        x = nn.Embed(self.vocab_size, self.embed_dim, name="embed")(tokens)

        # GRU over the sequence; nn.RNN returns all hidden states.
        gru = nn.RNN(nn.GRUCell(features=self.hidden_dim), name="gru")
        hidden_states = gru(x)              # (batch, seq_len, hidden_dim)
        hidden = hidden_states[:, -1, :]    # final timestep/hidden state after reading entire window (batch, hidden_dim)

        combined = jnp.concatenate([hidden, features], axis=-1)

        out = nn.Dense(64, name="dense_1")(combined)
        out = nn.relu(out)
        out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(out)
        out = nn.Dense(1, name="dense_out")(out)
        out = nn.sigmoid(out)
        return out.squeeze(-1)              # (batch,)
