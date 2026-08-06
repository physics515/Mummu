//! Render a checkpoint's **own** imported chat template — the general
//! fallback for models whose family has no hardcoded [`crate::chat`]
//! renderer. Behind the non-default `jinja-template` feature.
//!
//! The zoo's prompt wrapping is from-scratch and byte-verified: one
//! [`ChatMl`] constructor per family, each proven byte-identical to
//! `transformers.apply_chat_template` on the real checkpoint's template by
//! `tests/template_gate.rs`. That is the right shape for a model Mummu has
//! ported — the bytes are pinned by a test, not by a template file that can
//! change under us. It is no shape at all for a model Mummu has *not* ported,
//! which is exactly what the import suite (P3) is for: a checkpoint arrives
//! with a `chat_template` nobody has written a renderer for.
//!
//! So the rule this module encodes, and the one a consumer should follow:
//!
//! - **A family renderer exists → use it.** [`ChatMl::qwen2`],
//!   [`ChatMl::qwen3`], [`ChatMl::lfm2`]. Byte-pinned, no Jinja at runtime.
//! - **No family renderer → use [`ImportedTemplate`].** The checkpoint's own
//!   template is the authority on its own prompt format, and rendering it is
//!   strictly better than guessing ChatML.
//!
//! [`Renderer`] is that rule as a value, for a consumer that holds one
//! renderer and does not want to branch at every call site.
//!
//! What this module does NOT do is replace the family renderers. The gate
//! proved the two agree today on Qwen2.5/Qwen3/LFM2.5; that agreement is a
//! *result*, and the from-scratch path stays the shipping one.

use std::path::Path;

use hf_chat_template::{
    ChatTemplate, ChatTemplateField, Message, RenderInput, TokenField,
    TokenizerConfig as HfTokenizerConfig,
};

use crate::chat::{ChatMl, MAX_TOOLS, MAX_TURNS, Role, ToolSpec, Turn};
use crate::tok_config::{SpecialToken, TokenizerConfig};

/// Largest prompt a render will return. A chat template is a Jinja program
/// from an untrusted checkpoint; a loop over a long history can produce far
/// more than it was handed. Real prompts are kilobytes — a 8 MiB result is a
/// runaway template, not a conversation.
const MAX_RENDERED_BYTES: usize = 8 * 1024 * 1024;

/// What went wrong rendering a checkpoint's imported template.
#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    /// The checkpoint declares no chat template at all — neither the
    /// `chat_template` key of `tokenizer_config.json` nor a standalone
    /// `chat_template.jinja` beside it. There is nothing to render with.
    #[error("checkpoint declares no chat_template")]
    Absent,
    /// The template is not valid Jinja, or failed while rendering (a
    /// `raise_exception` in the template lands here too, with its message).
    #[error("chat template: {0}")]
    Jinja(String),
    /// The render produced more than [`MAX_RENDERED_BYTES`].
    #[error("chat template rendered {got} bytes, over the {MAX_RENDERED_BYTES} byte bound")]
    TooLarge { got: usize },
    /// A tool signature could not be serialized to JSON for the template.
    #[error("tool {name:?}: {reason}")]
    BadTool { name: String, reason: String },
}

impl From<hf_chat_template::Error> for TemplateError {
    fn from(e: hf_chat_template::Error) -> Self {
        TemplateError::Jinja(e.to_string())
    }
}

/// A compiled chat template imported from a checkpoint.
///
/// Compiling is the expensive half (Jinja parse); hold one per model and
/// render many prompts from it.
pub struct ImportedTemplate {
    inner: ChatTemplate,
}

impl std::fmt::Debug for ImportedTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ImportedTemplate")
    }
}

/// Our resolved special-token slot in the shape the Jinja context wants.
fn token_field(slot: &Option<SpecialToken>) -> Option<TokenField> {
    slot.as_ref().map(|t| TokenField::Str(t.content.clone()))
}

impl ImportedTemplate {
    /// Compile the template a parsed [`TokenizerConfig`] carries.
    ///
    /// The config's BOS/EOS/PAD/UNK slots go into the render context under
    /// the names templates use (`bos_token`, …) — many families' templates
    /// end a turn with `{{ eos_token }}` rather than a literal, so a template
    /// compiled without them renders a prompt the model never saw in
    /// training.
    pub fn from_config(config: &TokenizerConfig) -> Result<Self, TemplateError> {
        let source = config
            .chat_template
            .as_deref()
            .ok_or(TemplateError::Absent)?;
        assert!(
            !source.trim().is_empty(),
            "TokenizerConfig never stores a blank chat_template"
        );
        let hf = HfTokenizerConfig {
            chat_template: Some(ChatTemplateField::Single(source.to_string())),
            bos_token: token_field(&config.bos_token),
            eos_token: token_field(&config.eos_token),
            pad_token: token_field(&config.pad_token),
            unk_token: token_field(&config.unk_token),
            extra: Default::default(),
        };
        let inner = ChatTemplate::from_tokenizer_config(&hf)?;
        Ok(Self { inner })
    }

    /// Read a checkpoint directory's `tokenizer_config.json` (falling back to
    /// a standalone `chat_template.jinja`, per [`TokenizerConfig::from_dir`])
    /// and compile what it declares.
    pub fn from_dir(dir: &Path) -> Result<Self, TemplateError> {
        assert!(!dir.as_os_str().is_empty(), "from_dir: empty dir");
        let config =
            TokenizerConfig::from_dir(dir).map_err(|e| TemplateError::Jinja(e.to_string()))?;
        Self::from_config(&config)
    }

    /// Render a conversation, with the assistant generation prefix appended —
    /// the same contract as [`ChatMl::render`].
    pub fn render(&self, turns: &[Turn]) -> Result<String, TemplateError> {
        self.render_with_tools(&[], turns)
    }

    /// Render a conversation that advertises `tools`, with the assistant
    /// generation prefix appended — the same contract as
    /// [`ChatMl::render_with_tools`]. The template decides *where* and *how*
    /// the signatures appear; that is the whole point of using it.
    ///
    /// Each tool is handed over in the shape `transformers` itself produces
    /// (`get_json_schema`): `{"type": "function", "function": {name,
    /// description, parameters}}`. That is what the mainstream templates
    /// (Hermes/Qwen and everything modelled on them) unpack — but the key is
    /// *open*: a template runs `tool | tojson` on whatever it is given, and
    /// some families want the signature bare (LFM2.5 does; its own renderer
    /// covers it). Use [`Self::render_with_tools_json`] when a checkpoint's
    /// template wants a different shape.
    pub fn render_with_tools(
        &self,
        tools: &[ToolSpec],
        turns: &[Turn],
    ) -> Result<String, TemplateError> {
        let json: Vec<serde_json::Value> = tools.iter().map(tool_json).collect::<Result<_, _>>()?;
        self.render_with_tools_json(&json, turns)
    }

    /// Render with tool signatures given as raw JSON, for a template whose
    /// `tools` shape is not the `transformers` default (see
    /// [`Self::render_with_tools`]).
    pub fn render_with_tools_json(
        &self,
        tools: &[serde_json::Value],
        turns: &[Turn],
    ) -> Result<String, TemplateError> {
        assert!(
            turns.len() <= MAX_TURNS,
            "imported render: {} turns exceeds the {MAX_TURNS} bound",
            turns.len()
        );
        assert!(
            tools.len() <= MAX_TOOLS,
            "imported render: {} tools exceeds the {MAX_TOOLS} bound",
            tools.len()
        );
        let input = RenderInput {
            messages: turns.iter().map(message_from).collect(),
            tools: tools.to_vec(),
            add_generation_prompt: true,
            ..RenderInput::default()
        };
        let out = self.inner.render(&input)?;
        if out.len() > MAX_RENDERED_BYTES {
            return Err(TemplateError::TooLarge { got: out.len() });
        }
        Ok(out)
    }
}

/// One [`Turn`] as the message shape `transformers` hands a template.
///
/// An assistant turn carrying structured `tool_calls` passes them as data and
/// drops its `content` — the calls' rendered markers in [`Turn::content`] are
/// the *family* renderer's wire format, and re-emitting them here would
/// double-wrap under a template that writes its own. A turn is either a
/// tool-call turn or a text turn, never both, which is what the constructors
/// in [`crate::chat`] build.
fn message_from(turn: &Turn) -> Message {
    let role = match turn.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    if turn.role == Role::Assistant && !turn.tool_calls.is_empty() {
        let mut m = Message::new(role, "");
        m.content = None;
        m.tool_calls = turn
            .tool_calls
            .iter()
            .map(|c| serde_json::to_value(c).unwrap_or(serde_json::Value::Null))
            .collect();
        debug_assert_eq!(m.tool_calls.len(), turn.tool_calls.len());
        return m;
    }
    Message::new(role, turn.content.clone())
}

/// One [`ToolSpec`] in the Hermes wire shape templates expect under `tools`:
/// `{"type": "function", "function": {name, description, parameters}}`.
fn tool_json(spec: &ToolSpec) -> Result<serde_json::Value, TemplateError> {
    let function = serde_json::to_value(spec).map_err(|e| TemplateError::BadTool {
        name: spec.name.clone(),
        reason: e.to_string(),
    })?;
    let mut wire = serde_json::Map::new();
    wire.insert("type".into(), serde_json::Value::String("function".into()));
    wire.insert("function".into(), function);
    Ok(serde_json::Value::Object(wire))
}

/// Which renderer a model uses, as a value.
///
/// Construct [`Renderer::Family`] whenever the architecture is one Mummu has
/// ported (its bytes are pinned by the template gate); fall back to
/// [`Renderer::Imported`] for anything else. [`Renderer::for_checkpoint`]
/// applies exactly that rule.
#[derive(Debug)]
pub enum Renderer {
    /// A byte-verified from-scratch renderer.
    Family(ChatMl),
    /// The checkpoint's own Jinja template.
    Imported(ImportedTemplate),
}

impl Renderer {
    /// Take `family` when the caller has one for this architecture, else
    /// compile the checkpoint's own template from `dir`.
    ///
    /// The family renderer is not second-guessed: passing `Some` never reads
    /// the template. That is deliberate — the gate proves the family
    /// renderers match, and a checkpoint repackaged with a foreign template
    /// is caught at load by the consistency gate in `tokenizer.rs`, not
    /// silently obeyed here.
    pub fn for_checkpoint(family: Option<ChatMl>, dir: &Path) -> Result<Self, TemplateError> {
        match family {
            Some(chat_ml) => Ok(Self::Family(chat_ml)),
            None => ImportedTemplate::from_dir(dir).map(Self::Imported),
        }
    }

    /// Render a conversation with the generation prefix appended.
    pub fn render(&self, turns: &[Turn]) -> Result<String, TemplateError> {
        match self {
            Self::Family(c) => Ok(c.render(turns)),
            Self::Imported(t) => t.render(turns),
        }
    }

    /// Render a tool-advertising conversation with the generation prefix.
    pub fn render_with_tools(
        &self,
        tools: &[ToolSpec],
        turns: &[Turn],
    ) -> Result<String, TemplateError> {
        match self {
            Self::Family(c) => Ok(c.render_with_tools(tools, turns)),
            Self::Imported(t) => t.render_with_tools(tools, turns),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ToolCall;

    /// A minimal ChatML-shaped template, in the same Jinja dialect a real
    /// checkpoint ships: enough to exercise roles, tools and tool calls
    /// without needing a multi-GB checkpoint on disk.
    const TOY_TEMPLATE: &str = concat!(
        "{%- if tools %}<tools>{{ tools | tojson }}</tools>\n{%- endif %}",
        "{%- for m in messages %}",
        "<|{{ m.role }}|>",
        "{%- if m.tool_calls %}",
        "{%- for c in m.tool_calls %}[call {{ c.name }} {{ c.arguments | tojson }}]{%- endfor %}",
        "{%- else %}{{ m.content }}{%- endif %}",
        "{{ eos_token }}",
        "{%- endfor %}",
        "{%- if add_generation_prompt %}<|assistant|>{%- endif %}"
    );

    fn config_with(template: Option<&str>) -> TokenizerConfig {
        TokenizerConfig {
            chat_template: template.map(str::to_string),
            eos_token: Some(SpecialToken {
                content: "<END>".into(),
                id: Some(7),
                special: true,
            }),
            ..TokenizerConfig::default()
        }
    }

    #[test]
    fn renders_roles_and_the_generation_prompt_from_the_imported_template() {
        let t = ImportedTemplate::from_config(&config_with(Some(TOY_TEMPLATE)))
            .expect("toy template compiles");
        let out = t
            .render(&[Turn::system("be brief"), Turn::user("hi")])
            .expect("renders");
        assert_eq!(
            out, "<|system|>be brief<END><|user|>hi<END><|assistant|>",
            "roles, the config's eos_token and the generation prefix all reach the template"
        );
    }

    /// Tools reach the template as structured JSON in the Hermes wire shape,
    /// and the template — not us — decides where they land.
    #[test]
    fn tools_reach_the_template_as_hermes_wire_json() {
        let t = ImportedTemplate::from_config(&config_with(Some(TOY_TEMPLATE))).expect("compiles");
        let spec = ToolSpec {
            name: "get_weather".into(),
            description: "weather".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let out = t
            .render_with_tools(&[spec], &[Turn::user("weather?")])
            .expect("renders");
        assert!(out.starts_with("<tools>["), "tools block leads: {out}");
        // `tojson` spells separators the Python way (`": "`, `", "`) — the
        // same spelling `chat::python_json` pins for the from-scratch path.
        assert!(
            out.contains(r#""type": "function""#) && out.contains(r#""name": "get_weather""#),
            "the tool arrives wrapped as a function spec: {out}"
        );
    }

    /// The load-bearing difference from the family renderers: an assistant
    /// tool-call turn passes its calls as DATA, so the template writes its
    /// own markers instead of inheriting Hermes' `<tool_call>` wrapping.
    #[test]
    fn assistant_tool_calls_pass_structurally_not_as_hermes_markers() {
        let t = ImportedTemplate::from_config(&config_with(Some(TOY_TEMPLATE))).expect("compiles");
        let calls = [ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }];
        let out = t
            .render(&[
                Turn::user("weather?"),
                Turn::assistant_tool_calls(&calls),
                Turn::tool_response("{\"c\": 21}"),
            ])
            .expect("renders");
        assert!(
            out.contains("[call get_weather "),
            "the template wrote its own call markers: {out}"
        );
        assert!(
            !out.contains("<tool_call>"),
            "Hermes markers must NOT leak through as content: {out}"
        );
        assert!(out.contains("<|tool|>{\"c\": 21}"), "tool role turn: {out}");
    }

    #[test]
    fn a_checkpoint_without_a_template_is_a_loud_absent() {
        let err = ImportedTemplate::from_config(&config_with(None)).unwrap_err();
        assert!(matches!(err, TemplateError::Absent), "got {err:?}");
    }

    #[test]
    fn a_broken_template_is_a_loud_jinja_error_not_a_panic() {
        let err = ImportedTemplate::from_config(&config_with(Some("{% for x in %}")))
            .expect_err("unbalanced Jinja must not compile");
        assert!(matches!(err, TemplateError::Jinja(_)), "got {err:?}");
    }

    /// A runaway template (a loop that multiplies its input) must hit the
    /// byte bound rather than return a prompt nothing can tokenize.
    #[test]
    fn a_runaway_render_trips_the_byte_bound() {
        let bomb = "{%- for _ in range(4000) %}{{ messages[0].content }}{%- endfor %}";
        let t = ImportedTemplate::from_config(&config_with(Some(bomb))).expect("compiles");
        let big = "x".repeat(4096);
        let err = t
            .render(&[Turn::user(big)])
            .expect_err("must trip the bound");
        assert!(matches!(err, TemplateError::TooLarge { .. }), "got {err:?}");
    }

    /// `Renderer` never second-guesses a family renderer: given one, it does
    /// not read the checkpoint dir at all (here: a dir that does not exist).
    #[test]
    fn renderer_prefers_the_family_renderer_without_touching_the_checkpoint() {
        let r = Renderer::for_checkpoint(Some(ChatMl::qwen3()), Path::new("no/such/dir"))
            .expect("a family renderer needs no files");
        assert!(matches!(r, Renderer::Family(_)));
        let ours = r.render(&[Turn::user("hi")]).expect("renders");
        assert_eq!(ours, ChatMl::qwen3().render(&[Turn::user("hi")]));
    }
}
