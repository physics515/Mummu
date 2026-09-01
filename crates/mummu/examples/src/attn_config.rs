//! Attention-shaping configuration a checkpoint may declare that the shared
//! blocks do not implement: **RoPE frequency scaling** (`rope_scaling`) and
//! **sliding-window attention** (`sliding_window`).
//!
//! [`crate::nn::rope_tables`] computes plain rotary frequencies and
//! [`crate::nn::causal_mask`] a full causal mask. Every model currently in the
//! zoo is fine — their configs ship `rope_scaling: null` and no *enabled*
//! window — but this is the silent-wrong-answer class of gap: a checkpoint
//! that carries `rope_scaling` (Qwen2.5 past its 32 k native context via YaRN)
//! or an enabled `sliding_window` (Mistral-family; Gemma 2/3's alternating
//! layers) would load clean, pass the short-prompt parity probes, and degrade
//! numerically only far out in the context — exactly where nothing looks.
//!
//! So the cheap half first: **parse the fields and refuse the load, naming the
//! mode**. That converts silent degradation into an error a consumer can act
//! on. Implementing the modes is the second half (scaled `rope_tables` + a
//! windowed mask as config-driven variants of the same blocks), gated on a
//! long-context parity leg the short probes cannot see by construction.
//!
//! The trap this module exists to not fall into: **`sliding_window` being
//! present does not mean it is on**. Every Qwen2.5 checkpoint ships
//! `"sliding_window": 32768` together with `"use_sliding_window": false` — the
//! window is inert, and rejecting on the field's presence would refuse two
//! models Mummu has parity-verified for a month. Each family therefore decides
//! *enabled* by its own convention and passes the answer here; see
//! [`check_sliding_window`].

use crate::gguf::{GgufFile, GgufValue};

/// The `rope_scaling` object of an HF `config.json`, in both spellings that
/// appear in the wild: transformers ≥ 4.38 writes `rope_type`, older
/// checkpoints write `type`, and some carry both.
///
/// Deliberately lenient about the *payload* (`factor` and friends stay
/// `Option`) and strict about the *mode*: an unknown mode must be refused, and
/// refusing it does not require understanding its parameters.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct RopeScaling {
    /// transformers ≥ 4.38 spelling.
    #[serde(default)]
    pub rope_type: Option<String>,
    /// Pre-4.38 spelling. Kept separate rather than aliased so a checkpoint
    /// carrying both disagreeing values is visible instead of arbitrated.
    #[serde(default, rename = "type")]
    pub legacy_type: Option<String>,
    /// Context-extension factor (YaRN / linear / dynamic-NTK). Unused until
    /// the scaled tables land; parsed so the eventual implementation reads the
    /// same struct the rejection does.
    #[serde(default)]
    pub factor: Option<f64>,
    /// The context length the checkpoint was trained at, before scaling.
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    /// Everything else the object carries. Kept, not discarded, because the
    /// *shape* of the leftovers is load-bearing: transformers lets a
    /// Gemma-3-style config nest one RoPE object **per layer type**
    /// (`{"full_attention": {…}, "sliding_attention": {…}}`), and a nested map
    /// deserializes into this struct with every named field absent — which
    /// would read as "plain rotary" and sail through [`Self::check`]. Seeing
    /// the nested objects is what lets that be refused instead.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_json::Value>,
}

/// The spellings that mean "no scaling — plain rotary". `default` is what
/// transformers writes for an unscaled `rope_parameters`; `none`/`null` show
/// up in hand-written and converted configs.
const PLAIN_ROPE_TYPES: [&str; 3] = ["default", "none", "null"];

impl RopeScaling {
    /// The declared mode, lowercased, or `"default"` when the object names
    /// none. Both spellings are consulted; `rope_type` wins when they agree in
    /// meaning, and a disagreement is reported by [`Self::check`] rather than
    /// silently resolved.
    #[must_use]
    pub fn kind(&self) -> String {
        self.rope_type
            .as_deref()
            .or(self.legacy_type.as_deref())
            .unwrap_or("default")
            .trim()
            .to_ascii_lowercase()
    }

    /// Is this object just "plain rotary, no scaling"?
    #[must_use]
    pub fn is_plain(&self) -> bool {
        let k = self.kind();
        PLAIN_ROPE_TYPES.contains(&k.as_str())
    }

    /// Refuse anything that is not plain rotary, naming the mode.
    ///
    /// `whose` labels the source in the message (`"qwen2 config.json"`,
    /// `"GGUF qwen2.rope.scaling.type"`) so a consumer knows which file to
    /// look at. Returns `Ok(())` for a plain/absent mode — the only case the
    /// shared blocks actually compute.
    pub fn check(&self, whose: &str) -> Result<(), String> {
        debug_assert!(!whose.is_empty(), "check: `whose` must name a source");
        // A per-layer-type map (Gemma 3 and friends) names no mode of its own,
        // so it would otherwise read as plain. Refuse on the nesting itself —
        // whatever the sub-objects say, per-layer RoPE is not implemented.
        let nested: Vec<&str> = self
            .extra
            .iter()
            .filter(|(_, v)| v.is_object())
            .map(|(k, _)| k.as_str())
            .collect();
        if !nested.is_empty() {
            return Err(format!(
                "{whose}: rope_scaling nests a RoPE object per layer type ({}) — Mummu computes                  one rotary table for every layer, so a per-layer-type map cannot be honored",
                nested.join(", ")
            ));
        }
        // Both spellings present and disagreeing: neither is safe to believe.
        if let (Some(new), Some(old)) = (self.rope_type.as_deref(), self.legacy_type.as_deref())
            && !new.trim().eq_ignore_ascii_case(old.trim())
        {
            return Err(format!(
                "{whose}: rope_scaling names two different modes — rope_type '{new}' vs type \
                 '{old}'; refusing rather than guessing which one the weights were trained with"
            ));
        }
        if self.is_plain() {
            return Ok(());
        }
        let kind = self.kind();
        let factor = self
            .factor
            .map_or_else(|| "unspecified".to_string(), |f| format!("{f}"));
        Err(format!(
            "{whose}: rope_scaling mode '{kind}' (factor {factor}) is not implemented — Mummu \
             computes plain rotary frequencies, so this checkpoint would load clean and degrade \
             numerically past its original context instead of failing. Run it at its unscaled \
             context, or wait for scaled RoPE (ROADMAP P2)"
        ))
    }

    /// Read `<arch>.rope.scaling.*` out of a GGUF header. `None` when the
    /// header declares no scaling at all — llama.cpp omits the keys for an
    /// unscaled model rather than writing `"none"`.
    #[must_use]
    pub fn from_gguf(f: &GgufFile, arch: &str) -> Option<Self> {
        debug_assert!(!arch.is_empty(), "from_gguf: arch must be named");
        let ty = f
            .get(&format!("{arch}.rope.scaling.type"))
            .and_then(GgufValue::as_str)
            .map(str::to_owned);
        let factor = f
            .get(&format!("{arch}.rope.scaling.factor"))
            .and_then(GgufValue::as_f32)
            .map(f64::from);
        let original = f
            .get(&format!("{arch}.rope.scaling.original_context_length"))
            .and_then(GgufValue::as_u64)
            .and_then(|v| usize::try_from(v).ok());
        if ty.is_none() && factor.is_none() && original.is_none() {
            return None;
        }
        Some(Self {
            rope_type: ty,
            legacy_type: None,
            factor,
            original_max_position_embeddings: original,
            extra: std::collections::BTreeMap::new(),
        })
    }
}

/// Refuse an **enabled** sliding window, naming the span.
///
/// `enabled` is the family's own answer, not a guess from `window.is_some()`:
/// Qwen2/Qwen3 gate their `sliding_window` behind `use_sliding_window` and
/// ship it `false`, while a Gemma-style config has no such flag and a present
/// window is live. Passing `enabled: false` with a `Some(window)` is the
/// normal, correct call for every Qwen checkpoint in the zoo.
///
/// A window at least as long as the trained context is also inert — it can
/// never clip a position the model is allowed to reach — so it is accepted
/// even when enabled, with `max_positions` supplying that ceiling.
pub fn check_sliding_window(
    window: Option<usize>,
    enabled: bool,
    max_positions: Option<usize>,
    whose: &str,
) -> Result<(), String> {
    debug_assert!(!whose.is_empty(), "check_sliding_window: name a source");
    let Some(window) = window else {
        return Ok(());
    };
    if !enabled {
        return Ok(());
    }
    if window == 0 {
        return Err(format!(
            "{whose}: sliding_window is enabled but zero — no query could attend to any key"
        ));
    }
    // A window that spans the whole trained context masks nothing.
    if max_positions.is_some_and(|max| window >= max) {
        return Ok(());
    }
    Err(format!(
        "{whose}: sliding-window attention is enabled (window {window} tokens) but Mummu builds a \
         full causal mask — past {window} tokens this checkpoint would attend to keys the trained \
         model masks, and answer wrong without failing. Run it under {window} tokens of context, \
         or wait for windowed attention (ROADMAP P2)"
    ))
}

/// The sliding-window span a GGUF header declares (`<arch>.attention.
/// sliding_window`), if any. llama.cpp writes the key only for architectures
/// that use it, so `None` means "full attention" — the same convention the
/// `rope.scaling.*` keys follow.
#[must_use]
pub fn sliding_window_from_gguf(f: &GgufFile, arch: &str) -> Option<usize> {
    debug_assert!(
        !arch.is_empty(),
        "sliding_window_from_gguf: arch must be named"
    );
    f.get(&format!("{arch}.attention.sliding_window"))
        .and_then(GgufValue::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&w| w > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> RopeScaling {
        serde_json::from_str(json).expect("rope_scaling parses")
    }

    #[test]
    fn absent_mode_reads_as_plain_and_passes() {
        let s = RopeScaling::default();
        assert_eq!(s.kind(), "default");
        assert!(s.is_plain());
        assert!(s.check("test").is_ok());
    }

    #[test]
    fn plain_spellings_all_pass() {
        for ty in ["default", "none", "NULL", " Default "] {
            let s = parse(&format!(r#"{{"rope_type": "{ty}"}}"#));
            assert!(s.is_plain(), "{ty} should read as plain");
            assert!(s.check("test").is_ok());
        }
    }

    #[test]
    fn yarn_is_refused_and_the_message_names_the_mode_and_factor() {
        let s = parse(
            r#"{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}"#,
        );
        assert_eq!(s.kind(), "yarn");
        assert_eq!(s.factor, Some(4.0));
        assert_eq!(s.original_max_position_embeddings, Some(32768));
        let err = s
            .check("qwen2 config.json")
            .expect_err("yarn must be refused");
        assert!(err.contains("qwen2 config.json"), "{err}");
        assert!(err.contains("yarn"), "{err}");
        assert!(err.contains('4'), "{err}");
    }

    #[test]
    fn the_legacy_type_spelling_is_read_too() {
        let s = parse(r#"{"type": "linear", "factor": 2.0}"#);
        assert_eq!(s.kind(), "linear");
        assert!(s.check("test").is_err());
    }

    #[test]
    fn llama3_and_dynamic_are_refused_by_name() {
        for ty in ["llama3", "dynamic", "longrope", "someone_invented_this"] {
            let s = parse(&format!(r#"{{"rope_type": "{ty}"}}"#));
            let err = s.check("test").expect_err("non-plain must be refused");
            assert!(err.contains(ty), "message should name '{ty}': {err}");
        }
    }

    #[test]
    fn two_disagreeing_spellings_are_refused_rather_than_arbitrated() {
        let s = parse(r#"{"rope_type": "yarn", "type": "linear"}"#);
        let err = s.check("test").expect_err("disagreement must be refused");
        assert!(err.contains("yarn") && err.contains("linear"), "{err}");
        // The same mode written twice is not a disagreement.
        let same = parse(r#"{"rope_type": "Linear", "type": "linear"}"#);
        let err = same
            .check("test")
            .expect_err("linear is still unimplemented");
        assert!(!err.contains("two different modes"), "{err}");
    }

    /// The hole this `extra` map exists to close: a Gemma-3-style per-layer
    /// map names no mode, so without the nesting check it would deserialize to
    /// an all-default struct and read as plain rotary.
    #[test]
    fn a_per_layer_type_rope_map_is_refused_rather_than_read_as_plain() {
        let s = parse(
            r#"{"full_attention": {"rope_type": "dynamic", "factor": 8.0},
                "sliding_attention": {"rope_type": "default"}}"#,
        );
        assert_eq!(s.rope_type, None, "the nested map names no top-level mode");
        let err = s
            .check("gemma3 config.json")
            .expect_err("nesting must refuse");
        assert!(err.contains("full_attention"), "{err}");
        assert!(err.contains("sliding_attention"), "{err}");
    }

    /// Scalar extras (YaRN's `beta_fast`, an `attention_factor`) are NOT
    /// nesting, and must not be mistaken for it — the mode still decides.
    #[test]
    fn scalar_extras_do_not_trip_the_nesting_check() {
        let s = parse(r#"{"rope_type": "default", "beta_fast": 32, "attention_factor": 1.0}"#);
        assert!(s.check("test").is_ok(), "scalar extras are harmless");
        let yarn = parse(r#"{"rope_type": "yarn", "factor": 4.0, "beta_slow": 1}"#);
        let err = yarn.check("test").expect_err("yarn is still refused");
        assert!(err.contains("yarn"), "{err}");
    }

    #[test]
    fn an_inert_window_loads_and_an_enabled_one_does_not() {
        // The Qwen2.5 shape: present, but `use_sliding_window: false`.
        assert!(check_sliding_window(Some(32768), false, Some(32768), "qwen2").is_ok());
        // Enabled and shorter than the trained context: refused, span named.
        let err = check_sliding_window(Some(4096), true, Some(131_072), "mistral")
            .expect_err("an enabled window must be refused");
        assert!(err.contains("4096"), "{err}");
        assert!(err.contains("mistral"), "{err}");
        // No window at all is the common case.
        assert!(check_sliding_window(None, true, Some(4096), "x").is_ok());
    }

    #[test]
    fn a_window_spanning_the_whole_context_masks_nothing() {
        assert!(check_sliding_window(Some(4096), true, Some(4096), "x").is_ok());
        assert!(check_sliding_window(Some(8192), true, Some(4096), "x").is_ok());
        assert!(check_sliding_window(Some(4095), true, Some(4096), "x").is_err());
        // Unknown ceiling: cannot prove it is inert, so it is refused.
        assert!(check_sliding_window(Some(4096), true, None, "x").is_err());
    }

    #[test]
    fn a_zero_window_is_refused_as_degenerate() {
        let err = check_sliding_window(Some(0), true, Some(4096), "x")
            .expect_err("a zero window is nonsense");
        assert!(err.contains("zero"), "{err}");
    }
}
