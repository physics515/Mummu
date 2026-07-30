//! Tokenizer construction from non-`tokenizer.json` sources.
//!
//! **GGUF metadata** ([`tokenizer_from_gguf`]) — llama.cpp stores the
//! tokenizer as `tokenizer.ggml.*` metadata: the vocab (`tokens`, index =
//! token id), per-token types, BPE `merges`, and a `pre` identifier naming
//! the pre-tokenizer regex (the ecosystem hardcodes the regex per model
//! family, exactly as llama.cpp's `llama_vocab` does). Rebuilt as:
//! NFC → Split(pre regex) → ByteLevel → BPE, with control/user-defined
//! tokens re-added as special/non-special added tokens. Verified
//! byte-identical vs the checkpoint's `tokenizer.json` in `tests/real_gguf.rs`.
//!
//! **SentencePiece `tokenizer.model`** ([`tokenizer_from_spm`]) — the SPM
//! proto the Llama/Gemma/T5 families ship. A bounded hand-rolled protobuf
//! reader (the `gguf.rs` approach — the schema is tiny and frozen) feeds the
//! same pipeline HF's `convert_slow_tokenizer` assembles: Precompiled
//! charsmap (+ multi-space collapse) → Metaspace → **Unigram**. Verified
//! byte-identical vs the checkpoint's `tokenizer.json` in `tests/real_spm.rs`.

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

// ---------------------------------------------------------------------------
// SentencePiece `tokenizer.model` import (Llama/Gemma/T5-family checkpoints
// that ship the SPM proto instead of — or beside — a `tokenizer.json`).
// ---------------------------------------------------------------------------

/// Largest `tokenizer.model` file the reader will load — an order of magnitude
/// past the largest real proto (Gemma's 256k-piece model is ~4 MiB).
const MAX_SPM_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// Most pieces a proto may declare (Gemma: 256k; bound leaves headroom).
const MAX_SPM_PIECES: usize = 1_048_576;

/// SentencePiece `ModelProto.SentencePiece.Type` values.
const SPM_TYPE_NORMAL: i64 = 1;
const SPM_TYPE_UNKNOWN: i64 = 2;
const SPM_TYPE_CONTROL: i64 = 3;
const SPM_TYPE_USER_DEFINED: i64 = 4;
const SPM_TYPE_UNUSED: i64 = 5;
const SPM_TYPE_BYTE: i64 = 6;

/// `TrainerSpec.model_type` values.
const SPM_MODEL_UNIGRAM: i64 = 1;
const SPM_MODEL_BPE: i64 = 2;

/// One vocab piece out of the proto: `(text, score, type)` at index = id.
struct SpmPiece {
    text: String,
    score: f64,
    kind: i64,
}

/// The slice of a SentencePiece `ModelProto` that tokenizer assembly needs.
struct SpmProto {
    pieces: Vec<SpmPiece>,
    model_type: i64,
    unk_id: i64,
    precompiled_charsmap: Vec<u8>,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

/// A bounded protobuf wire-format reader (the same hand-rolled-parser approach
/// as `gguf.rs` — the proto schema is tiny and frozen, a protobuf codegen
/// dependency would be heavier than the format).
struct ProtoReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        debug_assert!(self.pos <= self.buf.len(), "pos never overshoots");
        self.pos >= self.buf.len()
    }

    /// A base-128 varint, bounded to 10 bytes (the u64 maximum).
    fn varint(&mut self) -> Result<u64, String> {
        let mut out: u64 = 0;
        for shift in 0..10u32 {
            let Some(&b) = self.buf.get(self.pos) else {
                return Err("varint runs past the end of the buffer".into());
            };
            self.pos += 1;
            out |= u64::from(b & 0x7f) << (7 * shift).min(63);
            if b & 0x80 == 0 {
                return Ok(out);
            }
        }
        Err("varint longer than 10 bytes".into())
    }

    /// The next field key as `(field_number, wire_type)`.
    fn key(&mut self) -> Result<(u64, u8), String> {
        let key = self.varint()?;
        #[allow(clippy::cast_possible_truncation)] // wire type is 3 bits
        Ok((key >> 3, (key & 0x7) as u8))
    }

    /// A length-delimited payload (wire type 2).
    fn bytes(&mut self) -> Result<&'a [u8], String> {
        let len = usize::try_from(self.varint()?).map_err(|_| "length overflows usize")?;
        let end = self
            .pos
            .checked_add(len)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| format!("length-delimited field of {len} B runs past the end"))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// Skip one field of the given wire type (unknown/uninteresting fields).
    fn skip(&mut self, wire_type: u8) -> Result<(), String> {
        match wire_type {
            0 => self.varint().map(|_| ()),
            1 => self.advance(8),
            2 => self.bytes().map(|_| ()),
            5 => self.advance(4),
            other => Err(format!("unsupported protobuf wire type {other}")),
        }
    }

    fn advance(&mut self, n: usize) -> Result<(), String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or("fixed-width field runs past the end")?;
        self.pos = end;
        Ok(())
    }

    /// A fixed32 float (wire type 5).
    fn float32(&mut self) -> Result<f32, String> {
        let end = self.pos + 4;
        let bytes: [u8; 4] = self
            .buf
            .get(self.pos..end)
            .and_then(|s| s.try_into().ok())
            .ok_or("float32 runs past the end")?;
        self.pos = end;
        Ok(f32::from_le_bytes(bytes))
    }
}

/// Parse one `ModelProto.SentencePiece` message: `piece`(1), `score`(2),
/// `type`(3, default NORMAL).
fn parse_spm_piece(buf: &[u8]) -> Result<SpmPiece, String> {
    let mut r = ProtoReader::new(buf);
    let mut piece = SpmPiece {
        text: String::new(),
        score: 0.0,
        kind: SPM_TYPE_NORMAL,
    };
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, 2) => {
                piece.text = String::from_utf8(r.bytes()?.to_vec())
                    .map_err(|_| "piece text is not UTF-8".to_string())?;
            }
            (2, 5) => piece.score = f64::from(r.float32()?),
            (3, 0) => {
                piece.kind = i64::try_from(r.varint()?).map_err(|_| "piece type overflows")?;
            }
            (_, w) => r.skip(w)?,
        }
    }
    Ok(piece)
}

/// Parse a SentencePiece `ModelProto`: `pieces`(1, repeated),
/// `trainer_spec`(2) for `model_type`(3)/`unk_id`(40), `normalizer_spec`(3)
/// for `precompiled_charsmap`(2)/`add_dummy_prefix`(3)/
/// `remove_extra_whitespaces`(4). Unknown fields are skipped, truncation is a
/// loud error, and the piece count is bounded.
fn parse_model_proto(buf: &[u8]) -> Result<SpmProto, String> {
    let mut proto = SpmProto {
        pieces: Vec::new(),
        model_type: SPM_MODEL_UNIGRAM,
        // The sentencepiece defaults (overridden by every real trainer_spec).
        unk_id: 0,
        precompiled_charsmap: Vec::new(),
        add_dummy_prefix: true,
        remove_extra_whitespaces: true,
        escape_whitespaces: true,
    };
    let mut r = ProtoReader::new(buf);
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, 2) => {
                if proto.pieces.len() >= MAX_SPM_PIECES {
                    return Err(format!("more than {MAX_SPM_PIECES} pieces"));
                }
                proto.pieces.push(parse_spm_piece(r.bytes()?)?);
            }
            (2, 2) => {
                let mut t = ProtoReader::new(r.bytes()?);
                while !t.done() {
                    let (f, w) = t.key()?;
                    match (f, w) {
                        (3, 0) => {
                            proto.model_type =
                                i64::try_from(t.varint()?).map_err(|_| "model_type overflows")?;
                        }
                        (40, 0) => {
                            proto.unk_id =
                                i64::try_from(t.varint()?).map_err(|_| "unk_id overflows")?;
                        }
                        (_, w) => t.skip(w)?,
                    }
                }
            }
            (3, 2) => {
                let mut n = ProtoReader::new(r.bytes()?);
                while !n.done() {
                    let (f, w) = n.key()?;
                    match (f, w) {
                        (2, 2) => proto.precompiled_charsmap = n.bytes()?.to_vec(),
                        (3, 0) => proto.add_dummy_prefix = n.varint()? != 0,
                        (4, 0) => proto.remove_extra_whitespaces = n.varint()? != 0,
                        (5, 0) => proto.escape_whitespaces = n.varint()? != 0,
                        (_, w) => n.skip(w)?,
                    }
                }
            }
            (_, w) => r.skip(w)?,
        }
    }
    if proto.pieces.is_empty() {
        return Err("proto declares no pieces".into());
    }
    Ok(proto)
}

/// Build an HF [`Tokenizer`] from a SentencePiece `tokenizer.model` proto
/// (the Llama/Gemma/T5-family format) — the same pipelines HF's own
/// `convert_slow_tokenizer` assembles, dispatched on the proto's
/// `model_type`:
///
/// - **UNIGRAM** (T5/ALBERT/Gemma): `Precompiled` charsmap normalizer (+ a
///   `{2,}`-space collapse when `remove_extra_whitespaces`), a Metaspace
///   pre-tokenizer/decoder driven by `add_dummy_prefix`, and a `Unigram`
///   model over the proto's pieces (index = token id).
/// - **BPE** (Llama-2 family): merges reconstructed from the vocab + scores
///   (HF's `SentencePieceExtractor` algorithm, literally), a `▁`-prepend +
///   space→`▁` normalizer, no pre-tokenizer, and the
///   Replace/ByteFallback/Fuse/Strip decoder chain.
///
/// In both, CONTROL/UNKNOWN pieces become special added tokens, USER_DEFINED
/// become plain added tokens, and every added id is verified post-build
/// exactly like the GGUF path. Tokens a checkpoint adds *beyond* the proto
/// (T5's `<extra_id_*>`, chat specials) live in sibling metadata
/// (`tokenizer_config.json`), not the proto — add them via
/// [`Tokenizer::add_special_tokens`] after this returns.
///
/// Faithfulness is verified against the same checkpoints' `tokenizer.json`
/// (byte-identical ids over a battery of prompts) in `tests/real_spm.rs`.
pub fn tokenizer_from_spm(path: &Path) -> Result<Tokenizer, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
    if meta.len() > MAX_SPM_FILE_BYTES {
        return Err(format!(
            "{} is {} B — over the {MAX_SPM_FILE_BYTES} B tokenizer.model bound",
            path.display(),
            meta.len()
        ));
    }
    let buf = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let proto = parse_model_proto(&buf).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut tok = match proto.model_type {
        SPM_MODEL_UNIGRAM => assemble_spm_unigram(&proto)?,
        SPM_MODEL_BPE => assemble_spm_bpe(&proto)?,
        other => {
            return Err(format!(
                "SentencePiece model_type {other} is not supported (UNIGRAM and BPE are)"
            ));
        }
    };
    add_and_verify_spm_specials(&mut tok, &proto)?;
    Ok(tok)
}

/// The proto's declared `unk_id`, range-checked against the pieces.
fn spm_unk_index(proto: &SpmProto) -> Result<usize, String> {
    debug_assert!(!proto.pieces.is_empty(), "parse rejects empty protos");
    usize::try_from(proto.unk_id)
        .ok()
        .filter(|&u| u < proto.pieces.len())
        .ok_or_else(|| format!("unk_id {} is out of vocab range", proto.unk_id))
}

/// The UNIGRAM assembly (T5/ALBERT/Gemma family) — see [`tokenizer_from_spm`].
fn assemble_spm_unigram(proto: &SpmProto) -> Result<Tokenizer, String> {
    use tokenizers::decoders::metaspace::{Metaspace, PrependScheme};
    use tokenizers::models::unigram::Unigram;
    use tokenizers::normalizers::replace::ReplacePattern;
    use tokenizers::normalizers::{Precompiled, Replace, Sequence as NormSequence};

    let unk = spm_unk_index(proto)?;
    let byte_fallback = proto.pieces.iter().any(|p| p.kind == SPM_TYPE_BYTE);
    let vocab: Vec<(String, f64)> = proto
        .pieces
        .iter()
        .map(|p| (p.text.clone(), p.score))
        .collect();
    debug_assert!(vocab.len() == proto.pieces.len(), "vocab keeps every index");
    let model =
        Unigram::from(vocab, Some(unk), byte_fallback).map_err(|e| format!("unigram: {e}"))?;
    let mut tok = Tokenizer::new(model);

    let mut normalizers: Vec<tokenizers::normalizers::NormalizerWrapper> = Vec::with_capacity(2);
    if !proto.precompiled_charsmap.is_empty() {
        let pre = Precompiled::from(&proto.precompiled_charsmap)
            .map_err(|e| format!("precompiled charsmap: {e}"))?;
        normalizers.push(pre.into());
    }
    if proto.remove_extra_whitespaces {
        // A REGEX pattern (multi-space collapse), exactly as the reference
        // tokenizer.json spells it — a string pattern would match literally.
        let collapse = Replace::new(ReplacePattern::Regex(" {2,}".into()), " ")
            .map_err(|e| format!("replace: {e}"))?;
        normalizers.push(collapse.into());
    }
    if !normalizers.is_empty() {
        tok.with_normalizer(Some(NormSequence::new(normalizers)))
            .map_err(|e| format!("normalizer: {e}"))?;
    }

    let prepend = if proto.add_dummy_prefix {
        PrependScheme::Always
    } else {
        PrependScheme::Never
    };
    let metaspace = Metaspace::new('\u{2581}', prepend, true);
    tok.with_pre_tokenizer(Some(metaspace.clone()));
    tok.with_decoder(Some(metaspace));
    Ok(tok)
}

/// The BPE assembly (Llama-2 family) — see [`tokenizer_from_spm`]. The
/// pipeline shape is pinned by the family's own `tokenizer.json`:
/// `Prepend(▁)` + `Replace(" "→"▁")` normalizers, NO pre-tokenizer, and the
/// `Replace("▁"→" ")` / `ByteFallback` / `Fuse` / `Strip(" ")` decoder chain.
fn assemble_spm_bpe(proto: &SpmProto) -> Result<Tokenizer, String> {
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::decoders::fuse::Fuse;
    use tokenizers::decoders::sequence::Sequence as DecoderSequence;
    use tokenizers::decoders::strip::Strip;
    use tokenizers::normalizers::replace::ReplacePattern;
    use tokenizers::normalizers::{Precompiled, Prepend, Replace, Sequence as NormSequence};

    let unk = spm_unk_index(proto)?;
    let byte_fallback = proto.pieces.iter().any(|p| p.kind == SPM_TYPE_BYTE);
    let mut vocab = Vocab::default();
    for (index, p) in proto.pieces.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)] // bounded by MAX_SPM_PIECES
        vocab.insert(p.text.clone(), index as u32);
    }
    if vocab.len() != proto.pieces.len() {
        return Err("duplicate piece text in the proto".into());
    }
    let merges = extract_spm_merges(&proto.pieces, &vocab);
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token(proto.pieces[unk].text.clone())
        .fuse_unk(true)
        .byte_fallback(byte_fallback)
        .build()
        .map_err(|e| format!("BPE build: {e}"))?;
    let mut tok = Tokenizer::new(bpe);

    let mut normalizers: Vec<tokenizers::normalizers::NormalizerWrapper> = Vec::with_capacity(3);
    if !proto.precompiled_charsmap.is_empty() {
        let pre = Precompiled::from(&proto.precompiled_charsmap)
            .map_err(|e| format!("precompiled charsmap: {e}"))?;
        normalizers.push(pre.into());
    }
    if proto.add_dummy_prefix {
        normalizers.push(Prepend::new("\u{2581}".into()).into());
    }
    if proto.escape_whitespaces {
        let escape = Replace::new(ReplacePattern::String(" ".into()), "\u{2581}")
            .map_err(|e| format!("replace: {e}"))?;
        normalizers.push(escape.into());
    }
    if !normalizers.is_empty() {
        tok.with_normalizer(Some(NormSequence::new(normalizers)))
            .map_err(|e| format!("normalizer: {e}"))?;
    }

    let unescape = Replace::new(ReplacePattern::String("\u{2581}".into()), " ")
        .map_err(|e| format!("replace: {e}"))?;
    let mut decoders: Vec<tokenizers::DecoderWrapper> = vec![unescape.into()];
    if byte_fallback {
        decoders.push(ByteFallback::new().into());
    }
    decoders.push(Fuse::new().into());
    if proto.add_dummy_prefix {
        // Strip the ONE leading space the dummy prefix injected.
        decoders.push(Strip::new(' ', 1, 0).into());
    }
    tok.with_decoder(Some(DecoderSequence::new(decoders)));
    Ok(tok)
}

/// Reconstruct BPE merges from a SentencePiece BPE proto's vocab + scores —
/// HF's `SentencePieceExtractor.extract(vocab_scores)`, literally: every
/// piece contributes each split `(l, r)` whose halves are both pieces, local
/// candidates ordered by `(id(l), id(r))`; the whole list is then
/// STABLE-sorted by score **descending** (a higher score is an earlier
/// merge; ties keep piece-id order, exactly like Python's stable sort over
/// an insertion-ordered dict).
fn extract_spm_merges(pieces: &[SpmPiece], vocab: &Vocab) -> Merges {
    debug_assert!(!pieces.is_empty(), "parse rejects empty protos");
    debug_assert!(vocab.len() == pieces.len(), "one vocab entry per piece");
    let mut scored: Vec<(String, String, f64)> = Vec::new();
    for p in pieces {
        let mut local: Vec<(&str, &str)> = Vec::new();
        for (split, _) in p.text.char_indices().skip(1) {
            let (l, r) = p.text.split_at(split);
            if vocab.contains_key(l) && vocab.contains_key(r) {
                local.push((l, r));
            }
        }
        local.sort_by_key(|&(l, r)| (vocab[l], vocab[r]));
        scored.extend(
            local
                .into_iter()
                .map(|(l, r)| (l.to_string(), r.to_string(), p.score)),
        );
    }
    scored.sort_by(|a, b| b.2.total_cmp(&a.2));
    scored.into_iter().map(|(l, r, _)| (l, r)).collect()
}

/// CONTROL/UNKNOWN pieces are the proto's specials (`<s>`, `</s>`, `<unk>`,
/// …); USER_DEFINED are plain added tokens. Re-add + verify every id like
/// the GGUF path — a drifted id would silently corrupt every prompt.
fn add_and_verify_spm_specials(tok: &mut Tokenizer, proto: &SpmProto) -> Result<(), String> {
    let mut added: Vec<(usize, &str, bool)> = Vec::new();
    for (index, p) in proto.pieces.iter().enumerate() {
        match p.kind {
            SPM_TYPE_CONTROL | SPM_TYPE_UNKNOWN => added.push((index, &p.text, true)),
            SPM_TYPE_USER_DEFINED => added.push((index, &p.text, false)),
            SPM_TYPE_NORMAL | SPM_TYPE_UNUSED | SPM_TYPE_BYTE => {}
            other => return Err(format!("piece {index} has unsupported type {other}")),
        }
    }
    for &(_, text, special) in &added {
        let t = AddedToken::from(text.to_string(), special);
        if special {
            tok.add_special_tokens([t])
                .map_err(|e| format!("add special token '{text}': {e}"))?;
        } else {
            tok.add_tokens([t])
                .map_err(|e| format!("add token '{text}': {e}"))?;
        }
    }
    for &(index, text, _) in &added {
        let got = tok.token_to_id(text);
        #[allow(clippy::cast_possible_truncation)] // bounded by MAX_SPM_PIECES
        if got != Some(index as u32) {
            return Err(format!(
                "added token '{text}' resolved to id {got:?}, the proto says {index}"
            ));
        }
    }
    Ok(())
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

    // ---- SentencePiece proto reader + assembly ----------------------------

    fn pb_varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    fn pb_field(field: u64, wire: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = pb_varint((field << 3) | u64::from(wire));
        if wire == 2 {
            out.extend(pb_varint(payload.len() as u64));
        }
        out.extend_from_slice(payload);
        out
    }

    /// One `SentencePiece` message: piece(1)=text, score(2)=f32, type(3).
    fn pb_piece(text: &str, score: f32, kind: Option<i64>) -> Vec<u8> {
        let mut msg = pb_field(1, 2, text.as_bytes());
        msg.extend(pb_field(2, 5, &score.to_le_bytes()));
        if let Some(k) = kind {
            msg.extend(pb_field(3, 0, &pb_varint(k as u64)));
        }
        msg
    }

    /// A tiny UNIGRAM proto: `<unk>`(UNKNOWN) `<s>`(CONTROL) then normal
    /// pieces, `unk_id = 0`, `add_dummy_prefix` + `remove_extra_whitespaces`.
    fn toy_spm(model_type: i64) -> Vec<u8> {
        let mut buf = Vec::new();
        for msg in [
            pb_piece("<unk>", 0.0, Some(SPM_TYPE_UNKNOWN)),
            pb_piece("<s>", 0.0, Some(SPM_TYPE_CONTROL)),
            pb_piece("\u{2581}", -2.0, None),
            pb_piece("\u{2581}hello", -1.0, None),
            pb_piece("hello", -3.0, None),
        ] {
            buf.extend(pb_field(1, 2, &msg));
        }
        let mut trainer = pb_field(3, 0, &pb_varint(model_type as u64));
        trainer.extend(pb_field(40, 0, &pb_varint(0))); // unk_id
        buf.extend(pb_field(2, 2, &trainer));
        let mut norm = pb_field(3, 0, &pb_varint(1)); // add_dummy_prefix
        norm.extend(pb_field(4, 0, &pb_varint(1))); // remove_extra_whitespaces
        buf.extend(pb_field(3, 2, &norm));
        buf
    }

    fn spm_file(bytes: &[u8]) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        // Unique per call: parallel tests in this process must never share a
        // file (same-length protos would otherwise collide and race).
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("mummu-spm-test-{}-{n}.model", std::process::id()));
        std::fs::write(&path, bytes).expect("temp proto writes");
        path
    }

    #[test]
    fn toy_spm_tokenizer_encodes_with_metaspace_and_specials() {
        let path = spm_file(&toy_spm(SPM_MODEL_UNIGRAM));
        let tok = tokenizer_from_spm(&path).expect("builds");
        let _ = std::fs::remove_file(&path);

        // add_dummy_prefix: "hello" → "▁hello" → piece 3.
        let ids = tok.encode("hello", false).expect("encodes");
        assert_eq!(ids.get_ids(), &[3], "metaspace prefix + unigram pick");
        // remove_extra_whitespaces collapses the run before metaspace.
        let ids = tok.encode("hello  hello", false).expect("encodes");
        assert_eq!(ids.get_ids(), &[3, 3], "space run collapses to one");
        // The CONTROL piece is a special added token at its proto index.
        assert_eq!(tok.token_to_id("<s>"), Some(1));
        let ids = tok.encode("<s>hello", false).expect("encodes");
        assert_eq!(ids.get_ids()[0], 1, "special token is never split");
    }

    #[test]
    fn spm_rejects_unknown_type_truncation_and_bad_unk() {
        // model_type 3 = WORD — neither UNIGRAM nor BPE.
        let path = spm_file(&toy_spm(3));
        let err = tokenizer_from_spm(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(
            err.contains("not supported"),
            "WORD protos are a loud error: {err}"
        );

        let full = toy_spm(SPM_MODEL_UNIGRAM);
        let path = spm_file(&full[..full.len() - 3]);
        let err = tokenizer_from_spm(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(
            err.contains("end") || err.contains("varint"),
            "truncation is a loud error, not a panic: {err}"
        );

        // unk_id past the vocab is rejected.
        let mut buf = Vec::new();
        buf.extend(pb_field(1, 2, &pb_piece("x", 0.0, None)));
        let mut trainer = pb_field(3, 0, &pb_varint(SPM_MODEL_UNIGRAM as u64));
        trainer.extend(pb_field(40, 0, &pb_varint(7)));
        buf.extend(pb_field(2, 2, &trainer));
        let path = spm_file(&buf);
        let err = tokenizer_from_spm(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("unk_id"), "out-of-range unk_id: {err}");
    }

    /// A BPE-shaped toy proto: single chars + the merged pieces, scores
    /// encoding merge order (higher = earlier merge, the SPM convention).
    fn toy_spm_bpe() -> Vec<u8> {
        let mut buf = Vec::new();
        for msg in [
            pb_piece("<unk>", 0.0, Some(SPM_TYPE_UNKNOWN)),
            pb_piece("<s>", 0.0, Some(SPM_TYPE_CONTROL)),
            pb_piece("\u{2581}", 0.0, None),
            pb_piece("h", 0.0, None),
            pb_piece("e", 0.0, None),
            pb_piece("l", 0.0, None),
            pb_piece("o", 0.0, None),
            pb_piece("he", -1.0, None),
            pb_piece("ll", -2.0, None),
            pb_piece("llo", -3.0, None),
            pb_piece("hello", -4.0, None),
            pb_piece("\u{2581}hello", -5.0, None),
        ] {
            buf.extend(pb_field(1, 2, &msg));
        }
        let mut trainer = pb_field(3, 0, &pb_varint(SPM_MODEL_BPE as u64));
        trainer.extend(pb_field(40, 0, &pb_varint(0)));
        buf.extend(pb_field(2, 2, &trainer));
        let mut norm = pb_field(3, 0, &pb_varint(1)); // add_dummy_prefix
        norm.extend(pb_field(5, 0, &pb_varint(1))); // escape_whitespaces
        buf.extend(pb_field(3, 2, &norm));
        buf
    }

    #[test]
    fn toy_spm_bpe_merges_chain_and_decode_round_trips() {
        let path = spm_file(&toy_spm_bpe());
        let tok = tokenizer_from_spm(&path).expect("BPE proto builds");
        let _ = std::fs::remove_file(&path);

        // "hello" → prepend+escape "▁hello" → chars merge all the way up the
        // score-ordered chain (h+e, l+l, ll+o, he+llo, ▁+hello) → one piece.
        let ids = tok.encode("hello", false).expect("encodes");
        assert_eq!(ids.get_ids(), &[11], "the merge chain reaches ▁hello");
        // Decode chain strips the dummy prefix back off.
        let text = tok.decode(&[11], true).expect("decodes");
        assert_eq!(text, "hello", "Replace/Fuse/Strip decode chain");
        // The CONTROL piece is a special added token at its proto index.
        assert_eq!(tok.token_to_id("<s>"), Some(1));
    }

    #[test]
    fn spm_proto_reader_skips_unknown_fields() {
        // An unknown length-delimited field (99) + an unknown varint field
        // (98) must be skipped, leaving the known fields intact.
        let mut buf = pb_field(99, 2, b"ignored");
        buf.extend(pb_field(98, 0, &pb_varint(12345)));
        buf.extend(toy_spm(SPM_MODEL_UNIGRAM));
        let proto = parse_model_proto(&buf).expect("parses around unknown fields");
        assert_eq!(proto.pieces.len(), 5);
        assert_eq!(proto.unk_id, 0);
        assert!(proto.add_dummy_prefix);
    }
}
