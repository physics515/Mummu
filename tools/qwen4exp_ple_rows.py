#!/usr/bin/env python3
"""Independent oracle for the qwen4exp PLE n-gram row ids.

Transcribed from the transformers reference (`modeling_qwen4_exp.py`:
`_splitmix64`, `_build_layer_multipliers`, `_find_nth_prime_after`,
`Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos` and `.forward`), NOT
from llama.cpp, so it is a second opinion on the Rust port's
`ple::PleHash`, which follows llama.cpp's `llm_graph_input_ple::set_input`.
torch is replaced by numpy int64 (torch.long); every product stays below
2**63 because the multipliers are drawn below `(2**63-1) // vocab_size`.

It checks three things and writes the fixture the Rust unit test replays:

1. the hash PARAMETERS derived the transformers way (seed 1234, base 20 M,
   ple_layer_index 0, vocab 248320) equal the ones the shipped GGUF header
   carries (`qwen4exp.ple.*`, copied below from the NVMe shard 1 header);
2. one-shot row ids for a 40-token sequence with one mid-stream EOS;
3. the same ids when the sequence is fed as a cached prefill followed by
   single-token steps (transformers keeps the last `ngram_size - 1` ids in a
   conv-state slot), so the window carried across calls is exercised.

Usage: python3 tools/qwen4exp_ple_rows.py > crates/mummu/tests/fixtures/qwen4exp_ple_rows.json
"""

import json
import math
import sys

import numpy as np

# ---- transformers config defaults (configuration_qwen4_exp.py) ----------
VOCAB_SIZE = 248320
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 8
NGRAM_VOCAB_SIZE_BASE = 20_000_000
MAKE_DIVISIBLE_BY = 128
SEED = 1234
PLE_LAYER_INDEX = 0  # ple_layer_ids == [2] (one-indexed) -> the first PLE layer

# ---- the shipped GGUF header (qwen4exp.ple.*, shard 1) ------------------
GGUF_MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]
GGUF_VOCAB_SIZES = [
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
]
GGUF_OFFSETS = [
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
]
GGUF_PLE_EOS = 248044
GGUF_TABLE_ROWS = 320001536

# ---- transcribed helpers -------------------------------------------------
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value):
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed):
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        out.append(2 * (_splitmix64(value) % half_bound) + 1)
    return out


def _is_prime(value):
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start, count):
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class NGram:
    """numpy twin of Qwen4ExpTextNGramEmbedding's id path (batch size 1)."""

    def __init__(self, eos):
        self.ngram_size = NGRAM_SIZE
        self.context_len = NGRAM_SIZE - 1
        self.heads_per_ngram = HEADS_PER_NGRAM
        self.ngram_heads = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
        self.eos_token_id = eos
        self.head_vocab_sizes = []
        self.head_offsets = []
        self.total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = PLE_LAYER_INDEX * self.ngram_heads + head_idx
            size = _find_nth_prime_after(NGRAM_VOCAB_SIZE_BASE - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size
        self.layer_multipliers = np.array(
            _build_layer_multipliers(VOCAB_SIZE, NGRAM_SIZE, PLE_LAYER_INDEX, SEED), dtype=np.int64
        )
        self.padded_vocab_size = math.ceil(self.total_vocab_size / MAKE_DIVISIBLE_BY) * MAKE_DIVISIBLE_BY
        self.cache = None  # conv_states[2]: the last context_len ids

    def _shift_right_ignore_eos(self, token_ids, shift):
        if shift == 0:
            return token_ids
        (seq_len,) = token_ids.shape
        positions = np.arange(seq_len, dtype=np.int64)
        eos_positions = np.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = np.maximum.accumulate(eos_positions)
        previous_eos = np.concatenate([[-1], previous_eos_inclusive[:-1]])
        segment_start = previous_eos + 1
        position_in_segment = positions - segment_start
        source_positions = positions - shift
        shifted = token_ids[np.clip(source_positions, 0, None)]
        valid = (position_in_segment >= shift) & (source_positions >= 0)
        return np.where(valid, shifted, np.int64(self.eos_token_id))

    def forward(self, input_ids, use_cache):
        input_ids = np.asarray(input_ids, dtype=np.int64)
        if use_cache and self.cache is not None:
            previous_context = self.cache.copy()
        else:
            previous_context = np.full(self.context_len, self.eos_token_id, dtype=np.int64)
        if use_cache:
            # update_conv_state keeps the newest context_len ids; on the first
            # call a short input is left-padded with EOS (the reference pads
            # explicitly because the cache would pad with zeros).
            if self.cache is None:
                hist = input_ids
                if hist.shape[0] < self.context_len:
                    hist = np.concatenate(
                        [np.full(self.context_len - hist.shape[0], self.eos_token_id, dtype=np.int64), hist]
                    )
            else:
                hist = np.concatenate([self.cache, input_ids])
            self.cache = hist[-self.context_len:].copy()

        token_history = np.concatenate([previous_context, input_ids])
        shifted_tokens = [self._shift_right_ignore_eos(token_history, s) for s in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = np.bitwise_xor(mixed_ids, shifted_tokens[position] * self.layer_multipliers[position])
            assert (mixed_ids >= 0).all(), "products stay below 2**63"
            vocab = np.array(self.head_vocab_sizes[start_idx:end_idx], dtype=np.int64)
            offs = np.array(self.head_offsets[start_idx:end_idx], dtype=np.int64)
            ngram_ids = np.remainder(mixed_ids[:, None], vocab[None, :])
            blocks.append(ngram_ids + offs[None, :])
        return np.concatenate(blocks, axis=-1)[-input_ids.shape[0]:]


def main():
    ng = NGram(GGUF_PLE_EOS)

    # 1. the header's hash parameters are the transformers derivation
    assert [int(m) for m in ng.layer_multipliers] == GGUF_MULTIPLIERS, ng.layer_multipliers
    assert ng.head_vocab_sizes == GGUF_VOCAB_SIZES, ng.head_vocab_sizes
    assert ng.head_offsets == GGUF_OFFSETS, ng.head_offsets
    assert ng.padded_vocab_size == GGUF_TABLE_ROWS, ng.padded_vocab_size

    # 2. one-shot ids: 40 random tokens, one EOS mid-stream
    rng = np.random.default_rng(20260916)
    tokens = rng.integers(0, VOCAB_SIZE, size=40, dtype=np.int64)
    tokens[tokens == GGUF_PLE_EOS] = 7  # exactly one EOS, placed below
    eos_at = 17
    tokens[eos_at] = GGUF_PLE_EOS
    assert int((tokens == GGUF_PLE_EOS).sum()) == 1
    one_shot = ng.forward(tokens, use_cache=False)
    assert one_shot.shape == (40, 16)
    assert (one_shot < ng.total_vocab_size).all()

    # 3. cached: a 23-token prefill (the EOS inside it), then 17 single steps
    ng.cache = None
    chunks = [ng.forward(tokens[:23], use_cache=True)]
    for i in range(23, 40):
        chunks.append(ng.forward(tokens[i:i + 1], use_cache=True))
    cached = np.concatenate(chunks, axis=0)
    assert (cached == one_shot).all(), "cached window reproduces the one-shot ids"

    json.dump(
        {
            "source": "tools/qwen4exp_ple_rows.py (numpy transcription of transformers modeling_qwen4_exp.py)",
            "ngram_size": NGRAM_SIZE,
            "heads_per_ngram": HEADS_PER_NGRAM,
            "eos": GGUF_PLE_EOS,
            "multipliers": GGUF_MULTIPLIERS,
            "head_vocab_sizes": GGUF_VOCAB_SIZES,
            "head_offsets": GGUF_OFFSETS,
            "eos_position": eos_at,
            "tokens": [int(t) for t in tokens],
            "rows": [[int(r) for r in row] for row in one_shot],
        },
        sys.stdout,
        indent=None,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
