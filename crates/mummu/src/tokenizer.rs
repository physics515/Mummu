//! Tokenizer from GGUF metadata — the piece that makes a GGUF file fully
//! self-contained (no sibling `tokenizer.json` needed).
//!
//! llama.cpp stores the tokenizer as `tokenizer.ggml.*` metadata: the vocab
//! (`tokens`, index = token id), per-token types, BPE `merges`, and a `pre`
//! identifier naming the pre-tokenizer regex (the ecosystem hardcodes the
//! regex per model family, exactly as llama.cpp's `llama_vocab` does). This
//! module rebuilds the equivalent HF `tokenizers` pipeline:
//! NFC → Split(pre regex) → ByteLevel → BPE, with control/user-defined
//! tokens re-added as special/non-special added tokens.
//!
//! Faithfulness is verified against the same checkpoint's `tokenizer.json`
//! (byte-identical ids over a battery of prompts) in `tests/real_gguf.rs`.

use tokenizers::models::bpe::{BPE, Merges, Vocab};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

use crate::gguf::{GgufFile, GgufValue};

/// llama.cpp token types (`llama_token_type`).
const TOKEN_TYPE_NORMAL: i64 = 1;
const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;
const TOKEN_TYPE_UNUSED: i64 = 5;

/// The GPT-2-style byte-level BPE pre-tokenizer regexes, keyed by
/// `tokenizer.ggml.pre` — the same registry llama.cpp keeps in its vocab
/// loader. Only families we actually run are listed; unknown ids are a loud
/// error (a wrong regex silently produces wrong token ids).
fn pre_tokenizer_regex(pre: &str) -> Option<&'static str> {
    match pre {
        // Qwen2/2.5 (matches the checkpoint's tokenizer.json byte for byte).
        "qwen2" => Some(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ),
        _ => None,
    }
}

/// A required `tokenizer.ggml.*` metadata array.
fn required_array<'f>(f: &'f GgufFile, key: &str) -> Result<&'f [GgufValue], String> {
    f.get(key)
        .and_then(GgufValue::as_array)
        .ok_or_else(|| format!("missing or non-array GGUF metadata '{key}'"))
}

/// Build an HF [`Tokenizer`] from a GGUF file's `tokenizer.ggml.*` metadata.
///
/// Supports the `gpt2` (byte-level BPE) tokenizer model with a known `pre`
/// regex. Token ids are the `tokens` array indexes; control (type 3) tokens
/// become special added tokens, user-defined (type 4) become non-special
/// added tokens, unused (type 5) padding entries are skipped. Every added
/// token's id is verified after construction — a drifted id would silently
/// corrupt every prompt, so it fails loudly instead.
pub fn tokenizer_from_gguf(f: &GgufFile) -> Result<Tokenizer, String> {
    let model = f
        .get("tokenizer.ggml.model")
        .and_then(GgufValue::as_str)
        .ok_or("missing GGUF metadata 'tokenizer.ggml.model'")?;
    if model != "gpt2" {
        return Err(format!(
            "tokenizer model '{model}' is not supported yet (only gpt2 byte-level BPE)"
        ));
    }
    let pre = f
        .get("tokenizer.ggml.pre")
        .and_then(GgufValue::as_str)
        .ok_or("missing GGUF metadata 'tokenizer.ggml.pre'")?;
    let Some(regex) = pre_tokenizer_regex(pre) else {
        return Err(format!("unknown pre-tokenizer id '{pre}'"));
    };

    let tokens = required_array(f, "tokenizer.ggml.tokens")?;
    let types = required_array(f, "tokenizer.ggml.token_type")?;
    if tokens.len() != types.len() {
        return Err(format!(
            "tokens ({}) and token_type ({}) lengths differ",
            tokens.len(),
            types.len()
        ));
    }

    // The BPE vocab: every NORMAL token, id = array index.
    let mut vocab = Vocab::default();
    // (index, content, is_special) for everything re-added post-BPE.
    let mut added: Vec<(usize, String, bool)> = Vec::new();
    for (index, (token, ty)) in tokens.iter().zip(types).enumerate() {
        let text = token
            .as_str()
            .ok_or_else(|| format!("token {index} is not a string"))?;
        let ty = ty
            .as_i64()
            .ok_or_else(|| format!("token_type {index} is not an integer"))?;
        #[allow(clippy::cast_possible_truncation)] // bounded by MAX_ARRAY_LEN
        match ty {
            TOKEN_TYPE_NORMAL => {
                vocab.insert(text.to_string(), index as u32);
            }
            TOKEN_TYPE_CONTROL => added.push((index, text.to_string(), true)),
            TOKEN_TYPE_USER_DEFINED => added.push((index, text.to_string(), false)),
            TOKEN_TYPE_UNUSED => {} // vocab-padding entries ([PADn])
            other => return Err(format!("token {index} has unsupported type {other}")),
        }
    }
    assert!(!vocab.is_empty(), "a tokenizer must have normal tokens");

    let mut merges = Merges::with_capacity(required_array(f, "tokenizer.ggml.merges")?.len());
    for (i, m) in required_array(f, "tokenizer.ggml.merges")?
        .iter()
        .enumerate()
    {
        let m = m
            .as_str()
            .ok_or_else(|| format!("merge {i} is not a string"))?;
        let (a, b) = m
            .split_once(' ')
            .ok_or_else(|| format!("merge {i} ('{m}') is not 'left right'"))?;
        merges.push((a.to_string(), b.to_string()));
    }

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| format!("BPE build: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    tok.with_normalizer(Some(NFC));
    let split = Split::new(
        SplitPattern::Regex(regex.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| format!("pre-tokenizer regex: {e}"))?;
    // ByteLevel exactly as the HF checkpoints configure it: no prefix space,
    // no offset trimming, regex handled by the Split stage above.
    let byte_level = ByteLevel::new(false, false, false);
    tok.with_pre_tokenizer(Some(Sequence::new(vec![split.into(), byte_level.into()])));
    tok.with_decoder(Some(byte_level));
    tok.with_post_processor(Some(byte_level));

    // Re-add control/user-defined tokens in id order, then verify every id
    // landed where the GGUF says it lives.
    for (_, text, special) in &added {
        let t = AddedToken::from(text.clone(), *special);
        if *special {
            tok.add_special_tokens(&[t]);
        } else {
            tok.add_tokens(&[t]);
        }
    }
    for (index, text, _) in &added {
        let got = tok.token_to_id(text);
        #[allow(clippy::cast_possible_truncation)] // bounded by MAX_ARRAY_LEN
        if got != Some(*index as u32) {
            return Err(format!(
                "added token '{text}' resolved to id {got:?}, GGUF says {index} — \
                 non-contiguous added-token ids are not supported"
            ));
        }
    }
    Ok(tok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufValue;

    /// A minimal synthetic GGUF header carrying a 4-token BPE tokenizer.
    fn toy_gguf() -> GgufFile {
        let strs = |v: &[&str]| {
            GgufValue::Array(v.iter().map(|s| GgufValue::Str((*s).to_string())).collect())
        };
        let ints = |v: &[i32]| GgufValue::Array(v.iter().map(|&i| GgufValue::I32(i)).collect());
        GgufFile {
            path: std::path::PathBuf::new(),
            version: 3,
            metadata: vec![
                ("tokenizer.ggml.model".into(), GgufValue::Str("gpt2".into())),
                ("tokenizer.ggml.pre".into(), GgufValue::Str("qwen2".into())),
                (
                    "tokenizer.ggml.tokens".into(),
                    strs(&["a", "b", "ab", "<|stop|>", "[PAD4]"]),
                ),
                ("tokenizer.ggml.token_type".into(), ints(&[1, 1, 1, 3, 5])),
                ("tokenizer.ggml.merges".into(), strs(&["a b"])),
            ],
            tensors: vec![],
            alignment: 32,
            data_offset: 0,
        }
    }

    #[test]
    fn toy_gguf_tokenizer_encodes_merges_and_specials() {
        let tok = tokenizer_from_gguf(&toy_gguf()).expect("builds");
        let ids = tok.encode("ab", true).expect("encodes");
        assert_eq!(ids.get_ids(), &[2], "the merge applies");
        assert_eq!(tok.token_to_id("<|stop|>"), Some(3), "control token id");
        let ids = tok.encode("<|stop|>ab", true).expect("encodes");
        assert_eq!(ids.get_ids(), &[3, 2], "special token is never split");
        // Unused padding entries are skipped, not part of the vocab.
        assert_eq!(tok.token_to_id("[PAD4]"), None);
    }

    #[test]
    fn unknown_model_pre_and_bad_merges_fail_loudly() {
        let mut f = toy_gguf();
        f.metadata[0].1 = GgufValue::Str("sentencepiece".into());
        assert!(tokenizer_from_gguf(&f).unwrap_err().contains("gpt2"));

        let mut f = toy_gguf();
        f.metadata[1].1 = GgufValue::Str("llama-bpe".into());
        assert!(
            tokenizer_from_gguf(&f)
                .unwrap_err()
                .contains("pre-tokenizer")
        );

        let mut f = toy_gguf();
        f.metadata[4].1 = GgufValue::Array(vec![GgufValue::Str("nospace".into())]);
        assert!(tokenizer_from_gguf(&f).unwrap_err().contains("merge"));
    }
}
