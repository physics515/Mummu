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

use std::path::Path;

use tokenizers::models::bpe::{BPE, Merges, Vocab};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

use crate::gguf::{GgufFile, GgufValue};
use crate::import::ImportError;
use crate::tok_config::{self, TokenizerConfig, ToolCallConvention};

/// llama.cpp token types (`llama_token_type`).
const TOKEN_TYPE_NORMAL: i64 = 1;
const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;
const TOKEN_TYPE_UNUSED: i64 = 5;

/// How a model family's byte-level BPE pipeline is configured — the part a
/// GGUF names by `tokenizer.ggml.pre` id instead of carrying explicitly.
struct PreSpec {
    /// The pre-tokenizer split regex (from the family's `tokenizer.json`).
    regex: &'static str,
    /// Whether the pipeline NFC-normalizes first.
    nfc: bool,
}

/// The per-family registry, keyed by `tokenizer.ggml.pre` — the same registry
/// llama.cpp keeps in its vocab loader. Only families we actually run are
/// listed; unknown ids are a loud error (a wrong regex silently produces
/// wrong token ids).
fn pre_spec(pre: &str) -> Option<PreSpec> {
    match pre {
        // Qwen2/2.5 (matches the checkpoint's tokenizer.json byte for byte).
        "qwen2" => Some(PreSpec {
            regex: r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            nfc: true,
        }),
        // LFM2/LFM2.5: digits split in groups of ≤3, no normalizer.
        "lfm2" => Some(PreSpec {
            regex: r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            nfc: false,
        }),
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
    let Some(spec) = pre_spec(pre) else {
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

    // The BPE vocab: every non-padding token at id = array index. Control/
    // user-defined tokens go in TOO — `add_tokens` below then reuses the
    // model id it finds, which is what lets added tokens live at LOW ids
    // (LFM2 puts its 500+ specials at 0..) instead of only after the vocab.
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
            TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED => {
                vocab.insert(text.to_string(), index as u32);
                added.push((index, text.to_string(), ty == TOKEN_TYPE_CONTROL));
            }
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
    if spec.nfc {
        tok.with_normalizer(Some(NFC))
            .map_err(|e| format!("NFC normalizer: {e}"))?;
    }
    let split = Split::new(
        SplitPattern::Regex(spec.regex.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| format!("pre-tokenizer regex: {e}"))?;
    // ByteLevel exactly as the HF checkpoints configure it: no prefix space,
    // no offset trimming, regex handled by the Split stage above.
    let byte_level = ByteLevel::new(false, false, false);
    tok.with_pre_tokenizer(Some(Sequence::new(vec![split.into(), byte_level.into()])));
    tok.with_decoder(Some(byte_level));

    // `tokenizer.ggml.add_bos_token` → a BOS-prepending template processor
    // (what the family's tokenizer.json does); otherwise the offsets-only
    // ByteLevel processor.
    let add_bos = f
        .get("tokenizer.ggml.add_bos_token")
        .and_then(|v| match *v {
            GgufValue::Bool(b) => Some(b),
            _ => None,
        })
        .unwrap_or(false);
    if add_bos {
        let bos_id = f
            .get("tokenizer.ggml.bos_token_id")
            .and_then(GgufValue::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("add_bos_token is set but tokenizer.ggml.bos_token_id is missing")?;
        let bos = tokens
            .get(bos_id as usize)
            .and_then(GgufValue::as_str)
            .ok_or_else(|| format!("bos_token_id {bos_id} is out of vocab range"))?;
        let template = TemplateProcessing::builder()
            .try_single(format!("{bos} $A"))
            .map_err(|e| format!("BOS template: {e}"))?
            .special_tokens(vec![(bos.to_string(), bos_id)])
            .build()
            .map_err(|e| format!("BOS template: {e}"))?;
        tok.with_post_processor(Some(template));
    } else {
        tok.with_post_processor(Some(byte_level));
    }

    // Re-add control/user-defined tokens in id order, then verify every id
    // landed where the GGUF says it lives.
    for (_, text, special) in &added {
        let t = AddedToken::from(text.clone(), *special);
        if *special {
            tok.add_special_tokens([t])
                .map_err(|e| format!("add special token '{text}': {e}"))?;
        } else {
            tok.add_tokens([t])
                .map_err(|e| format!("add token '{text}': {e}"))?;
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

/// The HF fast-tokenizer file a checkpoint ships beside its weights.
pub const TOKENIZER_JSON: &str = "tokenizer.json";

/// Cap on the number of individual mismatches spelled out in an
/// [`ImportError::Inconsistent`] message — the rest are summarized as a count so
/// a pathologically-broken checkpoint can't produce an unbounded error string.
const MAX_LISTED_MISMATCHES: usize = 8;

/// The full checkpoint-metadata gate a safetensors loader runs after parsing
/// `config.json` and **before** reading any weight bytes. It layers the
/// tokenizer-opening id cross-check on top of the tokenizer-free
/// [`tok_config::validate_dir`] gate (EOS agreement + tool-call convention),
/// giving the loaders one call that fails loudly on *any* metadata
/// disagreement at load time rather than at generate time.
///
/// Both sibling files are optional, and each absence is "nothing to check", not
/// an error:
///   * no `tokenizer_config.json` (a GGUF-derived or minimal dir) → `Ok(None)`;
///   * a `tokenizer_config.json` present but no `tokenizer.json` beside it → the
///     EOS/convention checks still run, the id cross-check is skipped.
///
/// A present, well-formed `tokenizer_config.json` whose declared added-token ids
/// disagree with the real `tokenizer.json` is an [`ImportError::Inconsistent`]
/// (a repackaging bug a checked *weight* load cannot see). On success the parsed
/// config is returned so a loader can reuse it (e.g. future config-driven BOS).
pub fn validate_checkpoint_dir(
    dir: &Path,
    config_eos_ids: &[u32],
    expected_convention: Option<ToolCallConvention>,
) -> Result<Option<TokenizerConfig>, ImportError> {
    assert!(
        !dir.as_os_str().is_empty(),
        "validate_checkpoint_dir: empty dir"
    );
    assert!(
        config_eos_ids.len() <= 256,
        "config.json eos_token_id sets are small; got {}",
        config_eos_ids.len()
    );
    let cfg = tok_config::validate_dir(dir, config_eos_ids, expected_convention)?;
    if let Some(cfg) = &cfg {
        check_added_token_ids(dir, cfg)?;
    }
    Ok(cfg)
}

/// Cross-check every added-token id `cfg` declares against the id the sibling
/// `dir/tokenizer.json` assigns that content. `Ok` when the tokenizer file is
/// absent (nothing to cross-check) or every id agrees; an
/// [`ImportError::Inconsistent`] naming the disagreements otherwise. A
/// `tokenizer.json` that is present but unreadable/malformed is a loud
/// [`ImportError::Parse`], never a panic.
fn check_added_token_ids(dir: &Path, cfg: &TokenizerConfig) -> Result<(), ImportError> {
    assert!(
        !dir.as_os_str().is_empty(),
        "check_added_token_ids: empty dir"
    );
    let path = dir.join(TOKENIZER_JSON);
    if !path.is_file() {
        return Ok(()); // no fast tokenizer beside the checkpoint — nothing to check
    }
    let tok = Tokenizer::from_file(&path).map_err(|e| ImportError::Parse {
        file: path.clone(),
        reason: format!("load {TOKENIZER_JSON}: {e}"),
    })?;
    let mismatches = match cfg.check_ids_against(|t| tok.token_to_id(t)) {
        Ok(()) => return Ok(()),
        Err(m) => m,
    };
    assert!(
        !mismatches.is_empty(),
        "the Err branch lists at least one mismatch"
    );
    debug_assert!(
        mismatches.len() <= cfg.added_tokens.len(),
        "at most one mismatch per declared added token"
    );
    let shown = mismatches.len().min(MAX_LISTED_MISMATCHES);
    let mut reason = format!(
        "{} added-token id(s) in {} disagree with {}:",
        mismatches.len(),
        tok_config::FILE_NAME,
        TOKENIZER_JSON,
    );
    for m in mismatches.iter().take(shown) {
        reason.push_str(&format!(
            " {:?} (config={:?}, tokenizer={:?});",
            m.content, m.found, m.expected
        ));
    }
    if mismatches.len() > shown {
        reason.push_str(&format!(" (+{} more)", mismatches.len() - shown));
    }
    Err(ImportError::Inconsistent { file: path, reason })
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
