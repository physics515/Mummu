//! `tokenizer_config.json` import — the **special-tokens map + chat template**
//! half of P3's "tokenizer + chat-template import".
//!
//! HF checkpoints carry the tokenizer *pipeline* in `tokenizer.json` (loaded
//! directly by the `tokenizers` crate) but keep the *conventions* a runner
//! needs — which tokens are BOS/EOS/PAD/UNK, whether to prepend BOS, the full
//! id→token map of added specials, and the Jinja **chat template** — in a
//! sibling `tokenizer_config.json`. This module parses that file into a
//! [`TokenizerConfig`] so an app can discover a model's declared special
//! tokens (and their ids) and its embedded chat template without hardcoding.
//!
//! It does **not** render the Jinja template — Mummu's prompt wrapping stays
//! the byte-verified code in [`crate::chat`]. The imported template is the
//! source of truth to check those renderers against and to detect a model's
//! tool-call convention; the imported special-token *ids* are a cross-check
//! that the config and the tokenizer agree (verified on real weights in
//! `tests/real_tokenizer_config.rs`).
//!
//! Union-typed fields are handled faithfully: a special-token field is `null`,
//! a bare string, or an `AddedToken` object `{ "content": … }`; `chat_template`
//! is a string or a list of `{ "name", "template" }` (the `default` entry, else
//! the first, is chosen). Every parse is total and bounded — malformed input is
//! a loud [`ImportError::Parse`], never a panic.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::import::ImportError;

/// The file this module reads.
pub const FILE_NAME: &str = "tokenizer_config.json";

/// A standalone chat-template file recent `transformers` `save_pretrained`
/// writes *instead of* the `chat_template` key of [`FILE_NAME`] (some
/// checkpoints — e.g. Gemma4 — ship it only here). [`TokenizerConfig::from_dir`]
/// falls back to it when the JSON key is absent, so the template-dependent
/// checks keep working for those checkpoints.
pub const CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";

/// Hard cap on the config file size. Chat templates (esp. tool-calling ones)
/// are the large part — a few tens of KiB in practice; 4 MiB is generous
/// headroom while still refusing a pathological or wrong file outright.
const MAX_CONFIG_BYTES: u64 = 4 * 1024 * 1024;

/// Hard cap on `added_tokens_decoder` entries — bounds the id space we ingest
/// (real vocabs top out in the low hundreds of thousands of *tokens*, of which
/// only the added specials appear here).
const MAX_ADDED_TOKENS: usize = 1_048_576;

/// One entry of `added_tokens_decoder`: a token added on top of the base BPE
/// vocab, at a known id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedTokenInfo {
    /// The token id (the JSON map key).
    pub id: u32,
    /// The token's literal text.
    pub content: String,
    /// Whether it is a special (control) token, never split out of text.
    pub special: bool,
}

/// A resolved special-token slot (BOS/EOS/PAD/UNK). `id` is `Some` when the
/// token's content was found in `added_tokens_decoder` — i.e. the config's
/// own tables agree on the id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialToken {
    /// The token's literal text.
    pub content: String,
    /// The id `added_tokens_decoder` assigns this content, if present.
    pub id: Option<u32>,
    /// Whether it is flagged special (authoritatively from the added-token
    /// entry when found, else the field's own flag, else `true`).
    pub special: bool,
}

/// The tool-call convention a chat template speaks, detected from its marker
/// tokens. Lets an app pick the matching [`crate::chat`] render style
/// (`render_with_tools` Hermes vs LFM) from the checkpoint's own template
/// instead of hardcoding it per model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallConvention {
    /// Hermes / Qwen (`<tool_call>{json}</tool_call>` with a `<tools>` block) —
    /// [`crate::chat`]'s Hermes-style renderer + `parse_tool_calls`.
    Hermes,
    /// LFM2.5 (a Pythonic call list between `<|tool_call_start|>` /
    /// `<|tool_call_end|>`) — [`crate::chat`]'s LFM-style renderer +
    /// `parse_tool_calls_lfm`.
    Lfm,
}

/// A tokenizer id disagreement found by [`TokenizerConfig::check_ids_against`]
/// or [`TokenizerConfig::check_eos_agrees`]: the config recorded `found` for
/// `content`, but the authoritative source says `expected`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdMismatch {
    /// Which slot disagreed, for the message (e.g. `"eos_token_id"`).
    pub what: &'static str,
    /// The token text at issue.
    pub content: String,
    /// The authoritative id (from the tokenizer or `config.json`); `None` when
    /// the token is absent there.
    pub expected: Option<u32>,
    /// The id `tokenizer_config.json` recorded.
    pub found: Option<u32>,
}

/// The conventions imported from a `tokenizer_config.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenizerConfig {
    /// Prepend BOS on encode (`add_bos_token`, default `false`).
    pub add_bos_token: bool,
    /// Append EOS on encode (`add_eos_token`, default `false`).
    pub add_eos_token: bool,
    pub bos_token: Option<SpecialToken>,
    pub eos_token: Option<SpecialToken>,
    pub pad_token: Option<SpecialToken>,
    pub unk_token: Option<SpecialToken>,
    /// `model_max_length`, when present and an integer.
    pub model_max_length: Option<u64>,
    /// The raw Jinja chat template (not rendered here). From the JSON
    /// `chat_template` key, or — via [`Self::from_dir`] when that key is absent —
    /// a standalone sibling [`CHAT_TEMPLATE_FILE`].
    pub chat_template: Option<String>,
    /// The full `added_tokens_decoder` map, sorted by ascending id.
    pub added_tokens: Vec<AddedTokenInfo>,
}

/// Build an [`ImportError::Parse`] for `file` with `reason`.
fn parse_err(file: &Path, reason: impl Into<String>) -> ImportError {
    ImportError::Parse {
        file: file.to_path_buf(),
        reason: reason.into(),
    }
}

/// Read `dir/chat_template.jinja` if present — the standalone template file
/// recent `transformers` writes instead of the `chat_template` JSON key. Absent,
/// a non-file, or an empty/whitespace-only body → `Ok(None)` (as good as no
/// template); a present but oversized or non-UTF-8 file is a loud
/// [`ImportError::Parse`], never a panic. Bounded by the same [`MAX_CONFIG_BYTES`]
/// cap as the JSON file (a real template is tens of KiB).
fn read_chat_template_file(dir: &Path) -> Result<Option<String>, ImportError> {
    assert!(
        !dir.as_os_str().is_empty(),
        "read_chat_template_file: empty dir"
    );
    let path = dir.join(CHAT_TEMPLATE_FILE);
    let Ok(meta) = std::fs::metadata(&path) else {
        return Ok(None); // no standalone template beside the checkpoint
    };
    if !meta.is_file() {
        return Ok(None);
    }
    if meta.len() > MAX_CONFIG_BYTES {
        return Err(parse_err(
            &path,
            format!(
                "{CHAT_TEMPLATE_FILE} is {} bytes (> {MAX_CONFIG_BYTES} cap)",
                meta.len()
            ),
        ));
    }
    let bytes = std::fs::read(&path).map_err(|e| parse_err(&path, format!("read: {e}")))?;
    let text = String::from_utf8(bytes).map_err(|e| parse_err(&path, format!("not utf-8: {e}")))?;
    if text.trim().is_empty() {
        return Ok(None);
    }
    debug_assert!(
        !text.trim().is_empty(),
        "a returned template has non-whitespace content"
    );
    Ok(Some(text))
}

impl TokenizerConfig {
    /// Read + parse `dir/tokenizer_config.json`.
    ///
    /// [`ImportError::MissingFile`] if absent, [`ImportError::Parse`] if it is
    /// oversized, unreadable, or malformed.
    pub fn from_dir(dir: &Path) -> Result<Self, ImportError> {
        assert!(!dir.as_os_str().is_empty(), "from_dir: empty dir");
        let path = dir.join(FILE_NAME);
        let meta = std::fs::metadata(&path).map_err(|_| ImportError::MissingFile(path.clone()))?;
        if !meta.is_file() {
            return Err(ImportError::MissingFile(path));
        }
        if meta.len() > MAX_CONFIG_BYTES {
            return Err(parse_err(
                &path,
                format!(
                    "{FILE_NAME} is {} bytes (> {MAX_CONFIG_BYTES} cap)",
                    meta.len()
                ),
            ));
        }
        let bytes = std::fs::read(&path).map_err(|e| parse_err(&path, format!("read: {e}")))?;
        debug_assert!(
            bytes.len() as u64 <= MAX_CONFIG_BYTES,
            "size checked before read"
        );
        let mut cfg = Self::from_json(&bytes, &path)?;
        // Fall back to a standalone `chat_template.jinja` only when the JSON key
        // was absent — a checkpoint that ships the template in the file (Gemma4)
        // otherwise reads as having no template, silently disabling the
        // template-dependent checks. A present JSON key wins (never overridden).
        if cfg.chat_template.is_none() {
            cfg.chat_template = read_chat_template_file(dir)?;
        }
        Ok(cfg)
    }

    /// Parse `tokenizer_config.json` bytes (`file` only labels errors).
    pub fn from_json(bytes: &[u8], file: &Path) -> Result<Self, ImportError> {
        assert!(!file.as_os_str().is_empty(), "from_json: empty file label");
        let root: Value =
            serde_json::from_slice(bytes).map_err(|e| parse_err(file, format!("json: {e}")))?;
        let obj = root
            .as_object()
            .ok_or_else(|| parse_err(file, "top-level is not a JSON object"))?;

        let added = parse_added_tokens(obj.get("added_tokens_decoder"), file)?;
        assert!(
            added.len() <= MAX_ADDED_TOKENS,
            "added tokens bounded by parser"
        );
        let index: HashMap<&str, (u32, bool)> = added
            .iter()
            .map(|a| (a.content.as_str(), (a.id, a.special)))
            .collect();

        let cfg = Self {
            add_bos_token: bool_field(obj, "add_bos_token"),
            add_eos_token: bool_field(obj, "add_eos_token"),
            bos_token: resolve_special(obj.get("bos_token"), &index),
            eos_token: resolve_special(obj.get("eos_token"), &index),
            pad_token: resolve_special(obj.get("pad_token"), &index),
            unk_token: resolve_special(obj.get("unk_token"), &index),
            model_max_length: obj.get("model_max_length").and_then(Value::as_u64),
            chat_template: obj.get("chat_template").and_then(parse_chat_template),
            added_tokens: added,
        };
        debug_assert!(
            cfg.added_tokens.windows(2).all(|w| w[0].id < w[1].id),
            "added tokens are sorted and unique after parse"
        );
        Ok(cfg)
    }

    /// The EOS token id, if the config resolved one.
    #[must_use]
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_token.as_ref().and_then(|t| t.id)
    }

    /// The BOS token id, if the config resolved one.
    #[must_use]
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_token.as_ref().and_then(|t| t.id)
    }

    /// The PAD token id, if the config resolved one.
    #[must_use]
    pub fn pad_id(&self) -> Option<u32> {
        self.pad_token.as_ref().and_then(|t| t.id)
    }

    /// Whether an embedded chat template was imported.
    #[must_use]
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// Detect the tool-call convention the imported `chat_template` implies,
    /// by its marker tokens. `None` when there is no template or it names no
    /// tool markers we recognize (a chat-only model). LFM's
    /// `<|tool_call_start|>` / `<|tool_list_start|>` are checked first because
    /// they are unambiguous; a Hermes template is identified by `<tool_call>`.
    #[must_use]
    pub fn tool_call_convention(&self) -> Option<ToolCallConvention> {
        let template = self.chat_template.as_deref()?;
        if template.contains("tool_call_start") || template.contains("tool_list_start") {
            return Some(ToolCallConvention::Lfm);
        }
        if template.contains("<tool_call>") {
            return Some(ToolCallConvention::Hermes);
        }
        None
    }

    /// Cross-check every added-token id against an authoritative `token_to_id`
    /// (the loaded HF [`tokenizers::Tokenizer`]): each `added_tokens_decoder`
    /// entry must resolve, in the real tokenizer, to exactly the id the config
    /// recorded. Catches a checkpoint whose `tokenizer_config.json` disagrees
    /// with its `tokenizer.json` (a repackaging bug that would silently corrupt
    /// special-token handling). The resolved BOS/EOS/PAD/UNK slots need no
    /// separate check — each carries an id only because its content was found
    /// in `added_tokens_decoder`, so it is already covered here.
    ///
    /// Takes a closure, not a `Tokenizer`, so this module stays free of a
    /// tokenizer dependency and any id source can be validated. `Ok` when all
    /// agree; `Err` lists every mismatch (bounded by the added-token count).
    pub fn check_ids_against(
        &self,
        token_to_id: impl Fn(&str) -> Option<u32>,
    ) -> Result<(), Vec<IdMismatch>> {
        assert!(
            self.added_tokens.len() <= MAX_ADDED_TOKENS,
            "added tokens bounded by the parser"
        );
        let mut bad = Vec::new();
        for a in &self.added_tokens {
            let found = token_to_id(&a.content);
            if found != Some(a.id) {
                bad.push(IdMismatch {
                    what: "added token",
                    content: a.content.clone(),
                    expected: found,
                    found: Some(a.id),
                });
            }
        }
        if bad.is_empty() {
            return Ok(());
        }
        debug_assert!(
            !bad.is_empty(),
            "the Err branch reports at least one mismatch"
        );
        Err(bad)
    }

    /// The fail-loud consistency gate a safetensors loader runs after parsing
    /// `config.json`: the imported `tokenizer_config.json` must not contradict
    /// (a) `config.json`'s EOS set, nor (b) this family's byte-verified
    /// [`crate::chat`] renderer, whose tool-call convention is
    /// `expected_convention` (`None` = the family has no tool renderer, so the
    /// template's convention is not checked). Returns a human-readable reason on
    /// the first disagreement; the loader wraps it as
    /// [`ImportError::Inconsistent`].
    ///
    /// The template check only fires when the imported template *declares* a
    /// convention (`tool_call_convention()` is `Some`): a base/chat-only
    /// checkpoint whose template names no tool markers is not forced to match.
    /// This catches the real bug — a checkpoint packaged with a *different*
    /// tool-call style than the loader's renderer emits (e.g. an LFM template
    /// dropped into a Qwen dir) — without rejecting tool-less templates.
    pub fn check_consistency(
        &self,
        config_eos_ids: &[u32],
        expected_convention: Option<ToolCallConvention>,
    ) -> Result<(), String> {
        if let Err(m) = self.check_eos_agrees(config_eos_ids) {
            return Err(format!(
                "tokenizer_config EOS {:?} ({:?}) is not in config.json eos_token_id {config_eos_ids:?}",
                m.found, m.content
            ));
        }
        if let (Some(expected), Some(found)) = (expected_convention, self.tool_call_convention())
            && found != expected
        {
            return Err(format!(
                "chat_template tool-call convention {found:?} contradicts this model's {expected:?} renderer",
            ));
        }
        Ok(())
    }

    /// Cross-check the config's declared EOS against `config.json`'s
    /// `eos_token_id` set: if `tokenizer_config.json` resolved an EOS id, it
    /// must be one of the ids `config.json` names. Catches the packaging bug
    /// where the two files disagree on which token ends a turn (the model then
    /// never stops, or stops on the wrong id). `Ok` when they agree or when no
    /// EOS id was resolved (nothing to check); `Err` names the disagreement.
    pub fn check_eos_agrees(&self, config_eos_ids: &[u32]) -> Result<(), IdMismatch> {
        assert!(
            config_eos_ids.len() <= 256,
            "config.json eos_token_id sets are small; got {}",
            config_eos_ids.len()
        );
        let Some(eos) = self.eos_token.as_ref() else {
            return Ok(());
        };
        let Some(id) = eos.id else {
            return Ok(());
        };
        assert!(
            !eos.content.is_empty(),
            "a resolved EOS has non-empty content"
        );
        if config_eos_ids.contains(&id) {
            Ok(())
        } else {
            Err(IdMismatch {
                what: "eos_token_id",
                content: eos.content.clone(),
                expected: config_eos_ids.first().copied(),
                found: Some(id),
            })
        }
    }
}

/// Run the [`TokenizerConfig::check_consistency`] gate against the
/// `tokenizer_config.json` beside a checkpoint's other files, if one is present.
///
/// `tokenizer_config.json` is **optional** — a GGUF-derived dir or a minimal
/// checkpoint may not ship one — so a missing file is `Ok(None)` (nothing to
/// validate). A file that is *present but malformed* propagates its parse error,
/// and a present, well-formed file that *disagrees* with `config_eos_ids` or the
/// family renderer becomes [`ImportError::Inconsistent`]. On success the parsed
/// config is returned (`Some`) so a loader can reuse it (e.g. future BOS wiring).
pub fn validate_dir(
    dir: &Path,
    config_eos_ids: &[u32],
    expected_convention: Option<ToolCallConvention>,
) -> Result<Option<TokenizerConfig>, ImportError> {
    assert!(!dir.as_os_str().is_empty(), "validate_dir: empty dir");
    let cfg = match TokenizerConfig::from_dir(dir) {
        Ok(cfg) => cfg,
        // The file is optional: absence is not an error, it is "nothing to check".
        Err(ImportError::MissingFile(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    cfg.check_consistency(config_eos_ids, expected_convention)
        .map_err(|reason| ImportError::Inconsistent {
            file: dir.join(FILE_NAME),
            reason,
        })?;
    Ok(Some(cfg))
}

/// A boolean field, defaulting to `false` when absent or non-boolean.
fn bool_field(obj: &serde_json::Map<String, Value>, key: &str) -> bool {
    obj.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// Parse `added_tokens_decoder` (a `"id" -> { content, special, … }` map) into
/// an id-sorted [`AddedTokenInfo`] list. Absent → empty. Malformed keys,
/// missing content, an oversize map, or a duplicate id are loud errors.
fn parse_added_tokens(v: Option<&Value>, file: &Path) -> Result<Vec<AddedTokenInfo>, ImportError> {
    let Some(v) = v else {
        return Ok(Vec::new());
    };
    let map = v
        .as_object()
        .ok_or_else(|| parse_err(file, "added_tokens_decoder is not an object"))?;
    if map.len() > MAX_ADDED_TOKENS {
        return Err(parse_err(
            file,
            format!(
                "added_tokens_decoder has {} entries (> {MAX_ADDED_TOKENS} cap)",
                map.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(map.len());
    for (key, entry) in map {
        let id = key.parse::<u32>().map_err(|_| {
            parse_err(
                file,
                format!("added_tokens_decoder key {key:?} is not a token id"),
            )
        })?;
        let content = entry
            .as_object()
            .and_then(|o| o.get("content"))
            .and_then(Value::as_str)
            .ok_or_else(|| parse_err(file, format!("added token {id} has no string content")))?;
        let special = entry
            .as_object()
            .and_then(|o| o.get("special"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        out.push(AddedTokenInfo {
            id,
            content: content.to_string(),
            special,
        });
    }
    out.sort_by_key(|a| a.id);
    // Distinct JSON keys can still collide numerically ("1" vs "01"); reject
    // rather than silently keep an ambiguous id.
    for w in out.windows(2) {
        if w[0].id == w[1].id {
            return Err(parse_err(
                file,
                format!("duplicate added-token id {}", w[0].id),
            ));
        }
    }
    Ok(out)
}

/// A special-token field's `(content, own_special_flag)`. `null`/absent/other
/// shapes → `None`; a bare string assumes special; an object reads `content`
/// (+ optional `special`).
fn special_content(v: &Value) -> Option<(String, bool)> {
    match v {
        Value::String(s) => Some((s.clone(), true)),
        Value::Object(o) => {
            let content = o.get("content").and_then(Value::as_str)?;
            let special = o.get("special").and_then(Value::as_bool).unwrap_or(true);
            Some((content.to_string(), special))
        }
        _ => None,
    }
}

/// Resolve a special-token field to a [`SpecialToken`], looking its id up in
/// the added-token index (which is authoritative for the id and the special
/// flag when the content is found there).
fn resolve_special(v: Option<&Value>, index: &HashMap<&str, (u32, bool)>) -> Option<SpecialToken> {
    let (content, own_special) = special_content(v?)?;
    let (id, special) = match index.get(content.as_str()) {
        Some(&(id, sp)) => (Some(id), sp),
        None => (None, own_special),
    };
    Some(SpecialToken {
        content,
        id,
        special,
    })
}

/// Extract the chat template: a bare string, or the `default` (else first)
/// entry of a `[{ "name", "template" }]` list. Other shapes → `None`.
fn parse_chat_template(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Array(arr) => {
            let mut first: Option<String> = None;
            for item in arr {
                let Some(tmpl) = item.get("template").and_then(Value::as_str) else {
                    continue;
                };
                if item.get("name").and_then(Value::as_str) == Some("default") {
                    return Some(tmpl.to_string());
                }
                if first.is_none() {
                    first = Some(tmpl.to_string());
                }
            }
            first
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> TokenizerConfig {
        TokenizerConfig::from_json(json.as_bytes(), Path::new("tokenizer_config.json"))
            .expect("valid config parses")
    }

    #[test]
    fn string_form_specials_resolve_ids_from_added_tokens() {
        let cfg = parse(
            r#"{
                "add_bos_token": false,
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
                "unk_token": null,
                "added_tokens_decoder": {
                    "151643": {"content": "<|endoftext|>", "special": true},
                    "151645": {"content": "<|im_end|>", "special": true}
                }
            }"#,
        );
        assert_eq!(cfg.eos_id(), Some(151645));
        assert_eq!(cfg.pad_id(), Some(151643));
        assert!(cfg.eos_token.as_ref().unwrap().special);
        assert_eq!(cfg.unk_token, None, "null token is absent");
        assert!(!cfg.add_bos_token);
        assert_eq!(cfg.added_tokens.len(), 2);
        // added_tokens are sorted by id.
        assert_eq!(cfg.added_tokens[0].id, 151643);
        assert_eq!(cfg.added_tokens[1].id, 151645);
    }

    #[test]
    fn object_form_special_reads_content_and_flag() {
        let cfg = parse(
            r#"{
                "bos_token": {"content": "<s>", "special": true, "lstrip": false},
                "added_tokens_decoder": {"1": {"content": "<s>", "special": true}}
            }"#,
        );
        let bos = cfg.bos_token.expect("bos present");
        assert_eq!(bos.content, "<s>");
        assert_eq!(bos.id, Some(1));
        assert!(bos.special);
    }

    #[test]
    fn unresolved_special_has_no_id_but_keeps_content() {
        // eos content is not in added_tokens_decoder → id None, content kept.
        let cfg = parse(r#"{ "eos_token": "</s>" }"#);
        let eos = cfg.eos_token.expect("eos present");
        assert_eq!(eos.content, "</s>");
        assert_eq!(eos.id, None);
        assert!(eos.special, "bare-string tokens default special");
    }

    #[test]
    fn chat_template_string_and_list_forms() {
        let s = parse(r#"{ "chat_template": "{{ hi }}" }"#);
        assert_eq!(s.chat_template.as_deref(), Some("{{ hi }}"));
        assert!(s.has_chat_template());

        let list = parse(
            r#"{ "chat_template": [
                {"name": "tool_use", "template": "TOOLS"},
                {"name": "default", "template": "DEFAULT"}
            ] }"#,
        );
        assert_eq!(
            list.chat_template.as_deref(),
            Some("DEFAULT"),
            "default entry wins"
        );

        let first = parse(
            r#"{ "chat_template": [
                {"name": "a", "template": "FIRST"},
                {"name": "b", "template": "SECOND"}
            ] }"#,
        );
        assert_eq!(
            first.chat_template.as_deref(),
            Some("FIRST"),
            "no default → first"
        );
    }

    #[test]
    fn tool_call_convention_is_detected_from_template_markers() {
        // Hermes: <tool_call> + <tools> block (Qwen-style).
        let hermes = parse(
            r#"{ "chat_template": "…<tools>{{sig}}</tools>…<tool_call>\n{json}\n</tool_call>…" }"#,
        );
        assert_eq!(
            hermes.tool_call_convention(),
            Some(ToolCallConvention::Hermes)
        );

        // LFM: the distinctive <|tool_call_start|> markers win even if the
        // template also mentions a generic tool word.
        let lfm = parse(
            r#"{ "chat_template": "…List of tools: […]…<|tool_call_start|>[f(x=1)]<|tool_call_end|>…" }"#,
        );
        assert_eq!(lfm.tool_call_convention(), Some(ToolCallConvention::Lfm));

        // A chat-only template names no tool markers.
        let chat_only = parse(r#"{ "chat_template": "<|im_start|>{{content}}<|im_end|>" }"#);
        assert_eq!(chat_only.tool_call_convention(), None);

        // No template at all, and an empty template, are both None (no panic).
        assert_eq!(parse("{}").tool_call_convention(), None);
        assert_eq!(
            parse(r#"{ "chat_template": "" }"#).tool_call_convention(),
            None
        );
    }

    #[test]
    fn check_ids_against_passes_when_agreeing_and_lists_mismatches() {
        let cfg = parse(
            r#"{
                "eos_token": "<|im_end|>",
                "added_tokens_decoder": {
                    "100": {"content": "<|im_end|>", "special": true},
                    "200": {"content": "<|extra|>", "special": true}
                }
            }"#,
        );
        // Agreeing tokenizer: both ids match.
        let agree = |t: &str| match t {
            "<|im_end|>" => Some(100),
            "<|extra|>" => Some(200),
            _ => None,
        };
        assert!(cfg.check_ids_against(agree).is_ok());

        // A tokenizer that disagrees on one id and is missing the other.
        let disagree = |t: &str| match t {
            "<|im_end|>" => Some(999),
            _ => None,
        };
        let bad = cfg.check_ids_against(disagree).unwrap_err();
        assert_eq!(bad.len(), 2, "both added tokens mismatch");
        let end = bad.iter().find(|m| m.content == "<|im_end|>").unwrap();
        assert_eq!(end.found, Some(100), "config recorded 100");
        assert_eq!(end.expected, Some(999), "tokenizer says 999");
        let extra = bad.iter().find(|m| m.content == "<|extra|>").unwrap();
        assert_eq!(extra.expected, None, "tokenizer is missing it");
    }

    #[test]
    fn check_eos_agrees_matches_config_json_eos_set() {
        let cfg = parse(
            r#"{
                "eos_token": "<|im_end|>",
                "added_tokens_decoder": {"151645": {"content": "<|im_end|>", "special": true}}
            }"#,
        );
        // config.json eos_token_id lists the same id → agree.
        assert!(cfg.check_eos_agrees(&[151_645]).is_ok());
        // A list that includes it (Qwen lists multiple) → agree.
        assert!(cfg.check_eos_agrees(&[151_643, 151_645]).is_ok());
        // A disagreeing config.json → loud mismatch naming both sides.
        let m = cfg.check_eos_agrees(&[151_643]).unwrap_err();
        assert_eq!(m.what, "eos_token_id");
        assert_eq!(m.found, Some(151_645), "tokenizer_config's eos id");
        assert_eq!(m.expected, Some(151_643), "config.json's first eos id");

        // No resolved EOS id → nothing to check, always Ok.
        let no_eos = parse(r#"{ "eos_token": "</s>" }"#); // unresolved (no added_tokens)
        assert!(no_eos.check_eos_agrees(&[7]).is_ok());
        assert!(parse("{}").check_eos_agrees(&[]).is_ok());
    }

    #[test]
    fn check_consistency_passes_agreeing_and_flags_each_disagreement() {
        // A well-formed Hermes checkpoint: EOS agrees, template is Hermes.
        let cfg = parse(
            r#"{
                "eos_token": "<|im_end|>",
                "chat_template": "…<tools>{{s}}</tools>…<tool_call>\n{j}\n</tool_call>…",
                "added_tokens_decoder": {"151645": {"content": "<|im_end|>", "special": true}}
            }"#,
        );
        assert!(
            cfg.check_consistency(&[151_645], Some(ToolCallConvention::Hermes))
                .is_ok()
        );
        // EOS not in config.json's set → loud reason mentioning the id.
        let eos_err = cfg
            .check_consistency(&[151_643], Some(ToolCallConvention::Hermes))
            .unwrap_err();
        assert!(eos_err.contains("151645"), "reason names the stray id");
        // Right EOS, but the loader expected the LFM convention → contradiction.
        let conv_err = cfg
            .check_consistency(&[151_645], Some(ToolCallConvention::Lfm))
            .unwrap_err();
        assert!(conv_err.contains("Hermes") && conv_err.contains("Lfm"));
    }

    #[test]
    fn check_consistency_skips_template_check_when_none_expected_or_declared() {
        // A tool-less (base) template declares no convention: any expectation is
        // fine, only the EOS is checked.
        let base = parse(
            r#"{
                "eos_token": "<|end|>",
                "chat_template": "<|im_start|>{{content}}<|im_end|>",
                "added_tokens_decoder": {"9": {"content": "<|end|>", "special": true}}
            }"#,
        );
        assert!(
            base.check_consistency(&[9], Some(ToolCallConvention::Hermes))
                .is_ok(),
            "no declared convention → template not forced to match"
        );
        // expected None → the template convention is never checked even if set.
        let lfm = parse(r#"{ "chat_template": "…<|tool_call_start|>[f()]<|tool_call_end|>…" }"#);
        assert!(lfm.check_consistency(&[], None).is_ok());
    }

    #[test]
    fn validate_dir_is_ok_when_file_absent_and_loud_on_mismatch() {
        use crate::import::ImportError;
        let dir = std::env::temp_dir().join("mummu_tokcfg_validate_test");
        std::fs::create_dir_all(&dir).unwrap();
        let _ = std::fs::remove_file(dir.join(FILE_NAME));
        // Absent file → Ok(None): the file is optional.
        assert!(matches!(
            validate_dir(&dir, &[1, 2], Some(ToolCallConvention::Hermes)),
            Ok(None)
        ));
        // Present + agreeing → Ok(Some(cfg)).
        std::fs::write(
            dir.join(FILE_NAME),
            br#"{"eos_token":"<|im_end|>","added_tokens_decoder":{"5":{"content":"<|im_end|>","special":true}}}"#,
        )
        .unwrap();
        assert!(matches!(
            validate_dir(&dir, &[5], Some(ToolCallConvention::Hermes)),
            Ok(Some(_))
        ));
        // Present + disagreeing EOS → loud Inconsistent (not a silent pass).
        assert!(matches!(
            validate_dir(&dir, &[7], Some(ToolCallConvention::Hermes)),
            Err(ImportError::Inconsistent { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_config_defaults_everything() {
        let cfg = parse("{}");
        assert!(!cfg.add_bos_token && !cfg.add_eos_token);
        assert_eq!(cfg.eos_token, None);
        assert_eq!(cfg.model_max_length, None);
        assert!(!cfg.has_chat_template());
        assert!(cfg.added_tokens.is_empty());
    }

    #[test]
    fn model_max_length_and_add_eos_are_read() {
        let cfg = parse(r#"{ "add_eos_token": true, "model_max_length": 131072 }"#);
        assert!(cfg.add_eos_token);
        assert_eq!(cfg.model_max_length, Some(131072));
    }

    #[test]
    fn malformed_inputs_fail_loudly() {
        let file = Path::new("tokenizer_config.json");
        // Non-object top level.
        assert!(TokenizerConfig::from_json(b"[1,2,3]", file).is_err());
        // Invalid JSON.
        assert!(TokenizerConfig::from_json(b"{ not json", file).is_err());
        // Non-numeric added-token key.
        assert!(
            TokenizerConfig::from_json(
                br#"{"added_tokens_decoder": {"oops": {"content": "x"}}}"#,
                file
            )
            .is_err()
        );
        // Added token without content.
        assert!(
            TokenizerConfig::from_json(
                br#"{"added_tokens_decoder": {"1": {"special": true}}}"#,
                file
            )
            .is_err()
        );
        // Duplicate numeric id ("1" vs "01").
        assert!(
            TokenizerConfig::from_json(
                br#"{"added_tokens_decoder": {"1": {"content": "a"}, "01": {"content": "b"}}}"#,
                file
            )
            .is_err()
        );
    }

    #[test]
    fn from_dir_reads_and_reports_missing() {
        let dir = std::env::temp_dir().join("mummu_tokcfg_test");
        std::fs::create_dir_all(&dir).unwrap();
        let _ = std::fs::remove_file(dir.join(FILE_NAME));
        assert!(matches!(
            TokenizerConfig::from_dir(&dir),
            Err(ImportError::MissingFile(_))
        ));
        std::fs::write(dir.join(FILE_NAME), br#"{"eos_token": "<|end|>"}"#).unwrap();
        let cfg = TokenizerConfig::from_dir(&dir).expect("present config parses");
        assert_eq!(cfg.eos_token.unwrap().content, "<|end|>");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn from_dir_falls_back_to_standalone_chat_template_jinja() {
        let dir = std::env::temp_dir().join("mummu_tokcfg_jinja_fallback");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // tokenizer_config.json has NO chat_template key; the template lives in a
        // sibling chat_template.jinja (the Gemma4-style layout).
        std::fs::write(dir.join(FILE_NAME), br#"{"eos_token": "<|im_end|>"}"#).unwrap();
        std::fs::write(
            dir.join(CHAT_TEMPLATE_FILE),
            b"<|im_start|>system\n{{x}}<|im_end|>\n<tools>{{s}}</tools>\n<tool_call>\n{j}\n</tool_call>",
        )
        .unwrap();
        let cfg = TokenizerConfig::from_dir(&dir).expect("config + jinja fallback parses");
        assert!(cfg.has_chat_template(), "the .jinja template was picked up");
        assert!(
            cfg.chat_template
                .as_deref()
                .unwrap()
                .contains("<|im_start|>")
        );
        // The convention gate now works for this checkpoint (it would have been
        // None — silently unchecked — without the fallback).
        assert_eq!(
            cfg.tool_call_convention(),
            Some(ToolCallConvention::Hermes),
            "the standalone template's tool-call convention is detected"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn from_dir_json_chat_template_wins_over_the_jinja_file() {
        let dir = std::env::temp_dir().join("mummu_tokcfg_jinja_precedence");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Both present: the JSON key is authoritative and must not be overridden.
        std::fs::write(
            dir.join(FILE_NAME),
            br#"{"chat_template": "FROM_JSON_KEY"}"#,
        )
        .unwrap();
        std::fs::write(dir.join(CHAT_TEMPLATE_FILE), b"FROM_JINJA_FILE").unwrap();
        let cfg = TokenizerConfig::from_dir(&dir).expect("parses");
        assert_eq!(
            cfg.chat_template.as_deref(),
            Some("FROM_JSON_KEY"),
            "a present JSON chat_template wins over the standalone file"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn from_dir_treats_an_empty_jinja_file_as_absent() {
        let dir = std::env::temp_dir().join("mummu_tokcfg_jinja_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(FILE_NAME), b"{}").unwrap();
        std::fs::write(dir.join(CHAT_TEMPLATE_FILE), b"   \n\t  ").unwrap();
        let cfg = TokenizerConfig::from_dir(&dir).expect("parses");
        assert!(
            !cfg.has_chat_template(),
            "a whitespace-only .jinja is as good as no template"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
