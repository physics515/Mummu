//! Explicit chat templates. Prompt wrapping is part of a model's contract —
//! an implicit or slightly-wrong template silently ruins output quality — so
//! templates are code here, never guessed: each per-model constructor is
//! byte-verified against the parity references (the Qwen2 template renders
//! the exact prompt committed in the Candle logits fixture).
//!
//! Both zoo LLMs speak ChatML; LFM2.5 additionally prefixes `<|startoftext|>`.
//! Tool use follows the Hermes convention Qwen2.5/Qwen3 ship in their chat
//! template: tool signatures in a `<tools>` block of the system turn, calls
//! emitted as `<tool_call>{json}</tool_call>`, results returned inside
//! `<tool_response>` blocks of a user turn. LFM2.5's bracket notation is a
//! P4 follow-up.

/// Who is speaking in a [`Turn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    /// A tool result going back to the model. Hermes-style templates render
    /// these inside a *user* turn as `<tool_response>` blocks.
    Tool,
}

impl Role {
    fn tag(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "user", // Hermes: tool results ride in a user turn
        }
    }
}

/// One message in a conversation.
#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

impl Turn {
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }

    /// An assistant turn that invokes tools: each call becomes a Hermes
    /// `<tool_call>` block in the turn body (what the model itself would
    /// have emitted), so histories containing calls re-render faithfully.
    #[must_use]
    pub fn assistant_tool_calls(calls: &[ToolCall]) -> Self {
        assert!(!calls.is_empty(), "assistant_tool_calls: no calls");
        assert!(
            calls.len() <= MAX_TOOL_CALLS,
            "assistant_tool_calls: {} calls exceeds the {MAX_TOOL_CALLS} bound",
            calls.len()
        );
        let blocks: Vec<String> = calls
            .iter()
            .map(|c| {
                let json = serde_json::to_string(c).unwrap_or_default();
                debug_assert!(!json.is_empty(), "a ToolCall always serializes");
                format!("<tool_call>\n{json}\n</tool_call>")
            })
            .collect();
        Self {
            role: Role::Assistant,
            content: blocks.join("\n"),
        }
    }

    /// A tool's result going back to the model; renders as a
    /// `<tool_response>` block (consecutive ones merge into one user turn).
    #[must_use]
    pub fn tool_response(content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
        }
    }
}

/// A callable tool signature, serialized into the system prompt exactly as
/// the Hermes-style templates expect: `{"type": "function", "function":
/// {"name": …, "description": …, "parameters": <json-schema>}}`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON schema of the arguments object.
    pub parameters: serde_json::Value,
}

/// The Hermes wire shape for one tool (field order matters for byte-stable
/// rendering, so this is a struct, not a `json!` map).
#[derive(serde::Serialize)]
struct ToolWire<'a> {
    r#type: &'static str,
    function: &'a ToolSpec,
}

/// One tool invocation, as emitted by the model inside `<tool_call>` tags.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Most tool calls a single response may contain (or a single assistant
/// history turn may carry) — far past anything a small model emits.
pub const MAX_TOOL_CALLS: usize = 64;

/// Most tools one render will advertise.
const MAX_TOOLS: usize = 128;

/// What went wrong extracting tool calls from a model response.
#[derive(Debug, thiserror::Error)]
pub enum ToolCallError {
    #[error("tool call {index}: unclosed <tool_call> tag")]
    Unclosed { index: usize },
    #[error("tool call {index}: {reason}")]
    BadJson { index: usize, reason: String },
    #[error("more than {MAX_TOOL_CALLS} tool calls in one response")]
    TooMany,
}

/// Extract Hermes-style tool calls from a model response: every
/// `<tool_call>…</tool_call>` block parses as a [`ToolCall`]; the text
/// outside the blocks (the model's prose, trimmed) comes back alongside.
/// Text with no blocks is simply `(vec![], text)` — not an error.
pub fn parse_tool_calls(text: &str) -> Result<(Vec<ToolCall>, String), ToolCallError> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut calls = Vec::new();
    let mut prose = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        if calls.len() == MAX_TOOL_CALLS {
            return Err(ToolCallError::TooMany);
        }
        prose.push_str(&rest[..start]);
        let after_open = &rest[start + OPEN.len()..];
        let Some(end) = after_open.find(CLOSE) else {
            return Err(ToolCallError::Unclosed { index: calls.len() });
        };
        let body = after_open[..end].trim();
        let call: ToolCall = serde_json::from_str(body).map_err(|e| ToolCallError::BadJson {
            index: calls.len(),
            reason: e.to_string(),
        })?;
        calls.push(call);
        rest = &after_open[end + CLOSE.len()..];
    }
    prose.push_str(rest);
    debug_assert!(calls.len() <= MAX_TOOL_CALLS, "bound enforced in the loop");
    Ok((calls, prose.trim().to_string()))
}

/// Longest conversation a single render will wrap — a generous bound that
/// still catches an unbounded history being passed by mistake.
const MAX_TURNS: usize = 1024;

/// The ChatML template family: `<|im_start|>role\ncontent<|im_end|>\n` per
/// turn, then an open assistant turn for the model to complete. `bos` is
/// prepended once when a model requires a start-of-text token.
#[derive(Debug, Clone)]
pub struct ChatMl {
    bos: Option<&'static str>,
}

impl ChatMl {
    /// Qwen2 / Qwen2.5-Instruct: plain ChatML, no BOS.
    #[must_use]
    pub fn qwen2() -> Self {
        Self { bos: None }
    }

    /// LFM2 / LFM2.5-Instruct: ChatML behind `<|startoftext|>`.
    #[must_use]
    pub fn lfm2() -> Self {
        Self {
            bos: Some("<|startoftext|>"),
        }
    }

    /// Render a conversation into the raw prompt string, ending with the open
    /// assistant turn the model completes. The caller tokenizes the result
    /// with special tokens enabled by the tokenizer itself, not re-added.
    #[must_use]
    pub fn render(&self, turns: &[Turn]) -> String {
        assert!(!turns.is_empty(), "chat render: no turns");
        assert!(
            turns.len() <= MAX_TURNS,
            "chat render: {} turns exceeds the {MAX_TURNS} bound",
            turns.len()
        );
        assert!(
            turns.last().map(|t| t.role) != Some(Role::Assistant),
            "chat render: the template opens the assistant turn itself; \
             a trailing assistant turn would double it"
        );
        let mut out = String::from(self.bos.unwrap_or(""));
        let mut i = 0;
        while i < turns.len() {
            if turns[i].role == Role::Tool {
                // Hermes: consecutive tool results merge into ONE user turn,
                // each wrapped in its own <tool_response> block.
                out.push_str("<|im_start|>user");
                while i < turns.len() && turns[i].role == Role::Tool {
                    out.push_str("\n<tool_response>\n");
                    out.push_str(&turns[i].content);
                    out.push_str("\n</tool_response>");
                    i += 1;
                }
                out.push_str("<|im_end|>\n");
            } else {
                out.push_str("<|im_start|>");
                out.push_str(turns[i].role.tag());
                out.push('\n');
                out.push_str(&turns[i].content);
                out.push_str("<|im_end|>\n");
                i += 1;
            }
        }
        out.push_str("<|im_start|>assistant\n");
        debug_assert!(out.ends_with("assistant\n"), "render must open a turn");
        out
    }

    /// [`render`](Self::render) with Hermes-style function calling: the tool
    /// signatures are advertised in a `# Tools` section of the system turn —
    /// the exact wording and tag structure Qwen2.5/Qwen3 ship in their chat
    /// template. An existing leading system turn provides the preamble; a
    /// conversation without one gets a neutral "You are a helpful assistant."
    #[must_use]
    pub fn render_with_tools(&self, tools: &[ToolSpec], turns: &[Turn]) -> String {
        assert!(
            !tools.is_empty(),
            "render_with_tools: no tools — use render()"
        );
        assert!(
            tools.len() <= MAX_TOOLS,
            "render_with_tools: {} tools exceeds the {MAX_TOOLS} bound",
            tools.len()
        );
        assert!(
            tools.iter().all(|t| !t.name.is_empty()),
            "render_with_tools: every tool needs a name"
        );

        let (preamble, rest) = match turns.first() {
            Some(t) if t.role == Role::System => (t.content.as_str(), &turns[1..]),
            _ => ("You are a helpful assistant.", turns),
        };
        let mut system = String::from(preamble);
        system.push_str(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
        );
        for tool in tools {
            let json = serde_json::to_string(&ToolWire {
                r#type: "function",
                function: tool,
            })
            .unwrap_or_default();
            debug_assert!(!json.is_empty(), "a ToolSpec always serializes");
            system.push('\n');
            system.push_str(&json);
        }
        system.push_str(
            "\n</tools>\n\nFor each function call, return a json object with function name and \
             arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": \
             <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
        );

        let mut wrapped = Vec::with_capacity(rest.len() + 1);
        wrapped.push(Turn::system(system));
        wrapped.extend_from_slice(rest);
        self.render(&wrapped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen2_render_matches_the_parity_verified_shape() {
        // This exact string (with this system prompt + user text) is what the
        // Candle fixture and the Ollama fp16 greedy leg were verified against.
        let raw = ChatMl::qwen2().render(&[
            Turn::system("You are a helpful assistant."),
            Turn::user("List the first five prime numbers."),
        ]);
        assert_eq!(
            raw,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nList the first five prime numbers.<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn lfm2_render_prefixes_bos_and_matches_the_parity_shape() {
        let raw = ChatMl::lfm2().render(&[Turn::user("List the first five prime numbers.")]);
        assert_eq!(
            raw,
            "<|startoftext|><|im_start|>user\nList the first five prime numbers.<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn multi_turn_history_renders_in_order() {
        let raw = ChatMl::qwen2().render(&[
            Turn::user("Hi."),
            Turn::assistant("Hello!"),
            Turn::user("Bye."),
        ]);
        assert_eq!(
            raw,
            "<|im_start|>user\nHi.<|im_end|>\n\
             <|im_start|>assistant\nHello!<|im_end|>\n\
             <|im_start|>user\nBye.<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    #[should_panic(expected = "no turns")]
    fn empty_conversation_is_rejected() {
        let _ = ChatMl::qwen2().render(&[]);
    }

    fn weather_tool() -> ToolSpec {
        ToolSpec {
            name: "get_weather".into(),
            description: "Get the current weather for a city.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        }
    }

    /// The tools section must match the Qwen2.5/Qwen3 chat template's wording
    /// and tag structure byte-for-byte (the model was trained on this text).
    /// Inside a tool's `parameters` schema, keys serialize in serde_json's
    /// canonical (sorted) order — key order isn't part of the trained text.
    #[test]
    fn tools_render_matches_the_hermes_template_shape() {
        let raw = ChatMl::qwen2().render_with_tools(
            &[weather_tool()],
            &[
                Turn::system("You are a helpful assistant."),
                Turn::user("Weather in Paris?"),
            ],
        );
        let expected = "<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\n\
             You may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>\n\
             {\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get the current weather for a city.\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"}}}\n\
             </tools>\n\n\
             For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n\
             <tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n\
             <|im_start|>user\nWeather in Paris?<|im_end|>\n\
             <|im_start|>assistant\n";
        assert_eq!(raw, expected);
    }

    #[test]
    fn tools_render_without_a_system_turn_injects_a_neutral_preamble() {
        let raw = ChatMl::qwen2().render_with_tools(&[weather_tool()], &[Turn::user("hi")]);
        assert!(raw.starts_with("<|im_start|>system\nYou are a helpful assistant.\n\n# Tools"));
        // The user turn survives un-consumed.
        assert!(raw.contains("<|im_start|>user\nhi<|im_end|>"));
    }

    #[test]
    fn consecutive_tool_responses_merge_into_one_user_turn() {
        let calls = [ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }];
        let raw = ChatMl::qwen2().render(&[
            Turn::user("Weather in Paris and Lyon?"),
            Turn::assistant_tool_calls(&calls),
            Turn::tool_response("{\"temp_c\": 21}"),
            Turn::tool_response("{\"temp_c\": 24}"),
        ]);
        // The assistant history turn carries the <tool_call> block it emitted.
        assert!(raw.contains(
            "<|im_start|>assistant\n<tool_call>\n\
             {\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n\
             </tool_call><|im_end|>\n"
        ));
        // Both results ride in ONE user turn, each in its own block.
        assert!(raw.contains(
            "<|im_start|>user\n\
             <tool_response>\n{\"temp_c\": 21}\n</tool_response>\n\
             <tool_response>\n{\"temp_c\": 24}\n</tool_response><|im_end|>\n"
        ));
        assert_eq!(raw.matches("<|im_start|>user").count(), 2);
    }

    #[test]
    fn parse_extracts_calls_and_prose() {
        let text = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \
                    \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n\
                    <tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
                    {\"city\": \"Lyon\"}}\n</tool_call>";
        let (calls, prose) = parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Paris");
        assert_eq!(calls[1].arguments["city"], "Lyon");
        assert_eq!(prose, "Let me check.");
    }

    #[test]
    fn parse_of_plain_text_is_empty_not_an_error() {
        let (calls, prose) = parse_tool_calls("The answer is 4.").unwrap();
        assert!(calls.is_empty());
        assert_eq!(prose, "The answer is 4.");
    }

    #[test]
    fn parse_rejects_unclosed_and_bad_json() {
        assert!(matches!(
            parse_tool_calls("<tool_call>\n{\"name\": \"x\"}"),
            Err(ToolCallError::Unclosed { index: 0 })
        ));
        assert!(matches!(
            parse_tool_calls("<tool_call>\nnot json\n</tool_call>"),
            Err(ToolCallError::BadJson { index: 0, .. })
        ));
    }

    /// The whole loop: a rendered history turn re-parses to the same calls.
    #[test]
    fn tool_calls_round_trip_through_render_and_parse() {
        let calls = vec![ToolCall {
            name: "lookup".into(),
            arguments: serde_json::json!({"q": "primes", "k": 5}),
        }];
        let turn = Turn::assistant_tool_calls(&calls);
        let (parsed, prose) = parse_tool_calls(&turn.content).unwrap();
        assert_eq!(parsed, calls);
        assert!(prose.is_empty());
    }

    #[test]
    #[should_panic(expected = "double it")]
    fn trailing_assistant_turn_is_rejected() {
        let _ = ChatMl::qwen2().render(&[Turn::user("q"), Turn::assistant("half-done")]);
    }
}
