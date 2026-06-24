"""Whitespace tokenizer + vocabulary for Twitch chat windows.

Kept deliberately simple so the Go clipper can replicate it exactly at runtime:
split each message on whitespace (case preserved, since ALL-CAPS hype and
case-sensitive emote names like KEKW are real signal), join messages in a window
with [SEP], map tokens to ids via the saved vocab.
"""

import json
from collections import Counter

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
SEP_TOKEN = "[SEP]"

# Reserved ids. Keep PAD at 0 so zero-padding == [PAD].
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SEP_TOKEN]


def tokenize_message(text):
    """Split a single chat message into tokens (whitespace, case-preserved)."""
    return text.split()


def build_vocab(messages, min_freq=2, max_size=20000):
    """Build a token->id vocab from an iterable of raw message strings."""
    counter = Counter()
    for message in messages:
        counter.update(tokenize_message(message))

    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

    for token, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def encode_window(messages, vocab, max_seq_len):
    """Encode a window's messages into a fixed-length list of token ids.

    Messages are assumed to already be in chronological order. The sequence is
    truncated/left-padded so the most recent tokens (closest to the clip moment)
    are always at the end — this matches the GRU using its final hidden state.
    """
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab[UNK_TOKEN]
    sep_id = vocab[SEP_TOKEN]

    ids = []
    for i, message in enumerate(messages):
        if i > 0:
            ids.append(sep_id)
        for token in tokenize_message(message):
            ids.append(vocab.get(token, unk_id))

    if len(ids) > max_seq_len:
        ids = ids[-max_seq_len:]
    else:
        ids = [pad_id] * (max_seq_len - len(ids)) + ids

    return ids


def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
