//! Explicit chat templates. Prompt wrapping is part of a model's contract —
//! an implicit or slightly-wrong template silently ruins output quality — so
//! templates are code here, never guessed: each per-model constructor is
//! byte-verified against the parity references (the Qwen2 template renders
//! the exact prompt committed in the Candle logits fixture).
//!
//! Both zoo LLMs speak ChatML; LFM2.5 additionally prefixes `<|startoftext|>`.
//! Tool use comes in two conventions, selected by the per-model constructor:
//!
//! - **Hermes** (Qwen2.5/Qwen3): tool signatures in a `<tools>` block of the
//!   system turn, calls emitted as `<tool_call>{json}</tool_call>`, results
//!   returned inside `<tool_response>` blocks of a user turn.
//! - **LFM** (LFM2.5, per its `chat_template.jinja` + model card): tool
//!   signatures as bare JSON in a `List of tools: […]` line of the system
//!   turn, calls emitted as a *Pythonic call list* between the
//!   `<|tool_call_start|>`/`<|tool_call_end|>` special tokens — e.g.
//!   `[get_weather(city="Paris")]` — results returned in a dedicated `tool`
//!   role turn, and `</think>`-prefixed reasoning stripped from every
//!   assistant history turn but the last.

/// Who is speaking in a [`Turn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    /// A tool result going back to the model. Hermes-style templates render
    /// these inside a *user* turn as `<tool_response>` blocks; LFM-style
    /// templates give each one its own `tool` role turn.
    Tool,
}

/// Which tool-use convention a template speaks (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolCallStyle {
    Hermes,
    Lfm,
}

impl Role {
    fn tag(self, style: ToolCallStyle) -> &'static str {
        match (self, style) {
            (Self::System, _) => "system",
            (Self::User, _) => "user",
            (Self::Assistant, _) => "assistant",
            // Hermes: tool results ride in a user turn; LFM: a real tool turn.
            (Self::Tool, ToolCallStyle::Hermes) => "user",
            (Self::Tool, ToolCallStyle::Lfm) => "tool",
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

    /// An assistant turn that invokes tools in LFM's Pythonic convention:
    /// the calls render as one bracketed call list between the
    /// `<|tool_call_start|>`/`<|tool_call_end|>` special tokens — exactly
    /// what an LFM2.5 model emits — so histories re-render faithfully.
    #[must_use]
    pub fn assistant_tool_calls_lfm(calls: &[ToolCall]) -> Self {
        assert!(!calls.is_empty(), "assistant_tool_calls_lfm: no calls");
        assert!(
            calls.len() <= MAX_TOOL_CALLS,
            "assistant_tool_calls_lfm: {} calls exceeds the {MAX_TOOL_CALLS} bound",
            calls.len()
        );
        Self {
            role: Role::Assistant,
            content: format!(
                "<|tool_call_start|>{}<|tool_call_end|>",
                pythonic_calls(calls)
            ),
        }
    }

    /// A tool's result going back to the model. Hermes templates render it as
    /// a `<tool_response>` block (consecutive ones merge into one user turn);
    /// LFM templates give it its own `tool` role turn.
    #[must_use]
    pub fn tool_response(content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
        }
    }
}

/// Deepest literal nesting the Pythonic renderer/parser will follow — far
/// past any real argument payload, and the recursion bound for both.
const MAX_VALUE_DEPTH: usize = 8;

/// Render tool calls as LFM's Pythonic call list: `[name(k=v, …), …]`.
/// JSON scalars map to Python spellings (`true`→`True`, `null`→`None`);
/// strings/lists/objects render as Python literals.
fn pythonic_calls(calls: &[ToolCall]) -> String {
    assert!(!calls.is_empty(), "pythonic_calls: no calls");
    assert!(calls.len() <= MAX_TOOL_CALLS, "pythonic_calls: over bound");
    let mut out = String::from("[");
    for (i, call) in calls.iter().enumerate() {
        assert!(!call.name.is_empty(), "pythonic_calls: unnamed call");
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&call.name);
        out.push('(');
        match &call.arguments {
            serde_json::Value::Object(args) => {
                for (j, (key, value)) in args.iter().enumerate() {
                    if j > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(key);
                    out.push('=');
                    python_literal(value, &mut out, 0);
                }
            }
            serde_json::Value::Null => {}
            other => panic!("pythonic_calls: arguments must be an object or null, got {other}"),
        }
        out.push(')');
    }
    out.push(']');
    debug_assert!(out.starts_with('[') && out.ends_with(']'), "list shape");
    out
}

/// Append one JSON value as a Python literal. Strings render double-quoted
/// with JSON-style escapes (valid Python), scalars as Python spellings.
fn python_literal(value: &serde_json::Value, out: &mut String, depth: usize) {
    assert!(depth <= MAX_VALUE_DEPTH, "python_literal: value too deep");
    match value {
        serde_json::Value::Null => out.push_str("None"),
        serde_json::Value::Bool(true) => out.push_str("True"),
        serde_json::Value::Bool(false) => out.push_str("False"),
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::String(s) => python_string_literal(s, out),
        serde_json::Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                python_literal(item, out, depth + 1);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            out.push('{');
            for (i, (key, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                python_string_literal(key, out);
                out.push_str(": ");
                python_literal(val, out, depth + 1);
            }
            out.push('}');
        }
    }
}

/// Double-quoted Python string literal with JSON-compatible escapes.
fn python_string_literal(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    debug_assert!(out.ends_with('"'), "string literal closes");
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
    #[error("tool call {index}: unclosed tool-call tag")]
    Unclosed { index: usize },
    #[error("tool call {index}: {reason}")]
    BadJson { index: usize, reason: String },
    #[error("tool call block {index}, byte {offset}: {reason}")]
    Syntax {
        /// Which `<|tool_call_start|>` block (0-based) failed to parse.
        index: usize,
        /// Byte offset *inside the block* where parsing stopped.
        offset: usize,
        reason: String,
    },
    #[error("tool call arguments nested deeper than {MAX_VALUE_DEPTH} levels")]
    TooDeep,
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

/// Extract LFM-style tool calls from a model response: every
/// `<|tool_call_start|>…<|tool_call_end|>` block parses as a *Pythonic call
/// list* (`[name(k=v, …), …]`); the text outside the blocks (the model's
/// prose, trimmed) comes back alongside. Text with no blocks is simply
/// `(vec![], text)` — not an error.
pub fn parse_tool_calls_lfm(text: &str) -> Result<(Vec<ToolCall>, String), ToolCallError> {
    const OPEN: &str = "<|tool_call_start|>";
    const CLOSE: &str = "<|tool_call_end|>";
    let mut calls = Vec::new();
    let mut prose = String::new();
    let mut rest = text;
    let mut block = 0;
    while let Some(start) = rest.find(OPEN) {
        prose.push_str(&rest[..start]);
        let after_open = &rest[start + OPEN.len()..];
        let Some(end) = after_open.find(CLOSE) else {
            return Err(ToolCallError::Unclosed { index: block });
        };
        let parsed = PythonicParser::new(after_open[..end].trim(), block).parse_call_list()?;
        if calls.len() + parsed.len() > MAX_TOOL_CALLS {
            return Err(ToolCallError::TooMany);
        }
        calls.extend(parsed);
        rest = &after_open[end + CLOSE.len()..];
        block += 1;
    }
    prose.push_str(rest);
    debug_assert!(calls.len() <= MAX_TOOL_CALLS, "bound enforced per block");
    Ok((calls, prose.trim().to_string()))
}

/// A bounded recursive-descent parser for LFM's Pythonic call list.
///
/// Grammar (whitespace-tolerant, trailing commas allowed):
/// ```text
/// calls  := '[' [ call (',' call)* [','] ] ']'
/// call   := ident '(' [ kwarg (',' kwarg)* [','] ] ')'
/// kwarg  := ident '=' value
/// value  := 'True' | 'False' | 'None' | 'true' | 'false' | 'null'
///         | number | string | '[' … ']' | '{' string ':' value, … '}'
/// ```
/// Lowercase JSON spellings are accepted because LFM2.5 documents a JSON
/// fallback mode and small models mix the two.
struct PythonicParser<'a> {
    src: &'a str,
    pos: usize,
    block: usize,
}

impl<'a> PythonicParser<'a> {
    fn new(src: &'a str, block: usize) -> Self {
        Self { src, pos: 0, block }
    }

    fn fail(&self, reason: impl Into<String>) -> ToolCallError {
        ToolCallError::Syntax {
            index: self.block,
            offset: self.pos,
            reason: reason.into(),
        }
    }

    fn skip_ws(&mut self) {
        while self.src[self.pos..].starts_with(|c: char| c.is_ascii_whitespace()) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += c.len_utf8();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: char, what: &str) -> Result<(), ToolCallError> {
        self.skip_ws();
        if self.eat(c) {
            Ok(())
        } else {
            Err(self.fail(format!("expected '{c}' {what}")))
        }
    }

    /// The whole block: a bracketed list of calls, then end of input.
    fn parse_call_list(mut self) -> Result<Vec<ToolCall>, ToolCallError> {
        assert!(self.pos == 0, "parse_call_list: parser already consumed");
        self.expect('[', "to open the call list")?;
        let mut calls = Vec::new();
        loop {
            self.skip_ws();
            if self.eat(']') {
                break;
            }
            if calls.len() == MAX_TOOL_CALLS {
                return Err(ToolCallError::TooMany);
            }
            calls.push(self.parse_call()?);
            self.skip_ws();
            if !self.eat(',') && self.peek() != Some(']') {
                return Err(self.fail("expected ',' or ']' after a call"));
            }
        }
        self.skip_ws();
        if self.pos != self.src.len() {
            return Err(self.fail("trailing text after the call list"));
        }
        debug_assert!(calls.len() <= MAX_TOOL_CALLS, "bound enforced in loop");
        Ok(calls)
    }

    fn parse_call(&mut self) -> Result<ToolCall, ToolCallError> {
        let name = self.parse_ident("a function name")?;
        self.expect('(', "to open the arguments")?;
        let mut args = serde_json::Map::new();
        loop {
            self.skip_ws();
            if self.eat(')') {
                break;
            }
            let key = self.parse_ident("an argument name")?;
            self.expect('=', "between argument name and value")?;
            self.skip_ws();
            let value = self.parse_value(0)?;
            args.insert(key, value);
            self.skip_ws();
            if !self.eat(',') && self.peek() != Some(')') {
                return Err(self.fail("expected ',' or ')' after an argument"));
            }
        }
        debug_assert!(!name.is_empty(), "parse_ident never returns empty");
        Ok(ToolCall {
            name,
            arguments: serde_json::Value::Object(args),
        })
    }

    fn parse_ident(&mut self, what: &str) -> Result<String, ToolCallError> {
        self.skip_ws();
        let start = self.pos;
        if self
            .peek()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        {
            self.pos += 1;
            while self
                .peek()
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                self.pos += 1;
            }
        }
        if self.pos == start {
            return Err(self.fail(format!("expected {what}")));
        }
        Ok(self.src[start..self.pos].to_string())
    }

    fn parse_value(&mut self, depth: usize) -> Result<serde_json::Value, ToolCallError> {
        if depth > MAX_VALUE_DEPTH {
            return Err(ToolCallError::TooDeep);
        }
        self.skip_ws();
        match self.peek() {
            Some('"') | Some('\'') => Ok(serde_json::Value::String(self.parse_string()?)),
            Some('[') => self.parse_list(depth),
            Some('{') => self.parse_dict(depth),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) if c.is_ascii_alphabetic() => {
                let word = self.parse_ident("a literal")?;
                match word.as_str() {
                    "True" | "true" => Ok(serde_json::Value::Bool(true)),
                    "False" | "false" => Ok(serde_json::Value::Bool(false)),
                    "None" | "null" => Ok(serde_json::Value::Null),
                    other => Err(self.fail(format!("unknown literal '{other}'"))),
                }
            }
            _ => Err(self.fail("expected a value")),
        }
    }

    fn parse_list(&mut self, depth: usize) -> Result<serde_json::Value, ToolCallError> {
        assert!(self.peek() == Some('['), "parse_list: caller checked '['");
        self.pos += 1;
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.eat(']') {
                break;
            }
            items.push(self.parse_value(depth + 1)?);
            self.skip_ws();
            if !self.eat(',') && self.peek() != Some(']') {
                return Err(self.fail("expected ',' or ']' in a list"));
            }
        }
        Ok(serde_json::Value::Array(items))
    }

    fn parse_dict(&mut self, depth: usize) -> Result<serde_json::Value, ToolCallError> {
        assert!(self.peek() == Some('{'), "parse_dict: caller checked brace");
        self.pos += 1;
        let mut map = serde_json::Map::new();
        loop {
            self.skip_ws();
            if self.eat('}') {
                break;
            }
            self.skip_ws();
            if !matches!(self.peek(), Some('"') | Some('\'')) {
                return Err(self.fail("dict keys must be strings"));
            }
            let key = self.parse_string()?;
            self.expect(':', "between dict key and value")?;
            let value = self.parse_value(depth + 1)?;
            map.insert(key, value);
            self.skip_ws();
            if !self.eat(',') && self.peek() != Some('}') {
                return Err(self.fail("expected ',' or '}' in a dict"));
            }
        }
        Ok(serde_json::Value::Object(map))
    }

    /// A single- or double-quoted string with Python/JSON escapes.
    fn parse_string(&mut self) -> Result<String, ToolCallError> {
        let quote = self.peek().expect("parse_string: caller checked a quote");
        assert!(quote == '"' || quote == '\'', "caller checked the quote");
        self.pos += 1;
        let mut out = String::new();
        loop {
            let Some(c) = self.peek() else {
                return Err(self.fail("unterminated string"));
            };
            self.pos += c.len_utf8();
            match c {
                c if c == quote => break,
                '\\' => out.push(self.parse_escape()?),
                c => out.push(c),
            }
        }
        Ok(out)
    }

    fn parse_escape(&mut self) -> Result<char, ToolCallError> {
        let Some(c) = self.peek() else {
            return Err(self.fail("dangling escape at end of string"));
        };
        self.pos += c.len_utf8();
        match c {
            '"' | '\'' | '\\' | '/' => Ok(c),
            'n' => Ok('\n'),
            't' => Ok('\t'),
            'r' => Ok('\r'),
            'u' => {
                let hex = self
                    .src
                    .get(self.pos..self.pos + 4)
                    .ok_or_else(|| self.fail("truncated \\u escape"))?;
                let code = u32::from_str_radix(hex, 16).map_err(|_| self.fail("bad \\u escape"))?;
                self.pos += 4;
                char::from_u32(code).ok_or_else(|| self.fail("\\u escape is not a scalar"))
            }
            other => Err(self.fail(format!("unsupported escape '\\{other}'"))),
        }
    }

    fn parse_number(&mut self) -> Result<serde_json::Value, ToolCallError> {
        let start = self.pos;
        self.eat('-');
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))
        {
            self.pos += 1;
        }
        let text = &self.src[start..self.pos];
        debug_assert!(!text.is_empty(), "caller checked a digit or '-'");
        if let Ok(i) = text.parse::<i64>() {
            return Ok(serde_json::Value::Number(i.into()));
        }
        let f = text
            .parse::<f64>()
            .map_err(|_| self.fail(format!("bad number '{text}'")))?;
        serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .ok_or_else(|| self.fail(format!("non-finite number '{text}'")))
    }
}

/// Longest conversation a single render will wrap — a generous bound that
/// still catches an unbounded history being passed by mistake.
const MAX_TURNS: usize = 1024;

/// The ChatML template family: `<|im_start|>role\ncontent<|im_end|>\n` per
/// turn, then an open assistant turn for the model to complete. `bos` is
/// prepended once when a model requires a start-of-text token; `tool_style`
/// picks the tool-use convention (see the module docs).
#[derive(Debug, Clone)]
pub struct ChatMl {
    bos: Option<&'static str>,
    tool_style: ToolCallStyle,
}

impl ChatMl {
    /// Qwen2 / Qwen2.5-Instruct: plain ChatML, no BOS, Hermes tool use.
    #[must_use]
    pub fn qwen2() -> Self {
        Self {
            bos: None,
            tool_style: ToolCallStyle::Hermes,
        }
    }

    /// LFM2 / LFM2.5-Instruct: ChatML behind `<|startoftext|>`, LFM
    /// (Pythonic) tool use.
    #[must_use]
    pub fn lfm2() -> Self {
        Self {
            bos: Some("<|startoftext|>"),
            tool_style: ToolCallStyle::Lfm,
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
        let last_assistant = turns.iter().rposition(|t| t.role == Role::Assistant);
        let mut out = String::from(self.bos.unwrap_or(""));
        let mut i = 0;
        while i < turns.len() {
            if self.tool_style == ToolCallStyle::Hermes && turns[i].role == Role::Tool {
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
                out.push_str(turns[i].role.tag(self.tool_style));
                out.push('\n');
                out.push_str(self.turn_content(&turns[i], i, last_assistant));
                out.push_str("<|im_end|>\n");
                i += 1;
            }
        }
        out.push_str("<|im_start|>assistant\n");
        debug_assert!(out.ends_with("assistant\n"), "render must open a turn");
        out
    }

    /// What a turn's body renders as. LFM templates strip `</think>`-prefixed
    /// reasoning from every assistant history turn but the last (the
    /// `keep_past_thinking=false` default of LFM2.5's `chat_template.jinja`);
    /// everything else passes through.
    fn turn_content<'a>(
        &self,
        turn: &'a Turn,
        index: usize,
        last_assistant: Option<usize>,
    ) -> &'a str {
        debug_assert!(index <= MAX_TURNS, "index bounded by the render assert");
        debug_assert!(
            turn.role != Role::Assistant || last_assistant.is_some(),
            "an assistant turn implies a last-assistant index"
        );
        let is_past_assistant =
            turn.role == Role::Assistant && last_assistant.is_some_and(|l| index != l);
        if self.tool_style == ToolCallStyle::Lfm
            && is_past_assistant
            && let Some(end) = turn.content.rfind("</think>")
        {
            return turn.content[end + "</think>".len()..].trim();
        }
        &turn.content
    }

    /// [`render`](Self::render) with function calling in this template's
    /// convention.
    ///
    /// - **Hermes** (Qwen2.5/Qwen3): tool signatures advertised in a
    ///   `# Tools` section of the system turn — the exact wording and tag
    ///   structure those models ship in their chat template. A conversation
    ///   without a system turn gets a neutral "You are a helpful assistant."
    /// - **LFM** (LFM2.5): bare tool JSON on a `List of tools: […]` line
    ///   appended to the system turn — the exact shape of LFM2.5's
    ///   `chat_template.jinja` + model card, which injects *no* default
    ///   preamble: without a system turn the tools line stands alone.
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
        match self.tool_style {
            ToolCallStyle::Hermes => self.render_with_tools_hermes(tools, turns),
            ToolCallStyle::Lfm => self.render_with_tools_lfm(tools, turns),
        }
    }

    fn render_with_tools_hermes(&self, tools: &[ToolSpec], turns: &[Turn]) -> String {
        debug_assert!(!tools.is_empty(), "checked by render_with_tools");
        debug_assert!(self.tool_style == ToolCallStyle::Hermes, "hermes only");
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

    fn render_with_tools_lfm(&self, tools: &[ToolSpec], turns: &[Turn]) -> String {
        debug_assert!(!tools.is_empty(), "checked by render_with_tools");
        debug_assert!(self.tool_style == ToolCallStyle::Lfm, "lfm only");
        let (preamble, rest) = match turns.first() {
            Some(t) if t.role == Role::System => (t.content.as_str(), &turns[1..]),
            _ => ("", turns),
        };
        let mut system = String::from(preamble);
        if !system.is_empty() {
            system.push('\n');
        }
        system.push_str("List of tools: [");
        for (i, tool) in tools.iter().enumerate() {
            if i > 0 {
                system.push_str(", ");
            }
            let json = serde_json::to_string(tool).unwrap_or_default();
            debug_assert!(!json.is_empty(), "a ToolSpec always serializes");
            system.push_str(&json);
        }
        system.push(']');

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

    // ---- LFM (Pythonic) tool use ----------------------------------------

    /// The tools line must match LFM2.5's `chat_template.jinja` byte shape:
    /// `List of tools: [{bare tool json}, …]` appended to the system turn
    /// with a `\n`, tools comma-joined, bare (no Hermes `"type":"function"`
    /// wrapper — the model card's examples show `{"name": …}` directly).
    #[test]
    fn lfm_tools_render_matches_the_lfm25_template_shape() {
        let raw = ChatMl::lfm2().render_with_tools(
            &[weather_tool()],
            &[
                Turn::system("You are a helpful assistant."),
                Turn::user("Weather in Paris?"),
            ],
        );
        let expected = "<|startoftext|><|im_start|>system\nYou are a helpful assistant.\n\
             List of tools: [{\"name\":\"get_weather\",\"description\":\"Get the current weather for a city.\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"}}]<|im_end|>\n\
             <|im_start|>user\nWeather in Paris?<|im_end|>\n\
             <|im_start|>assistant\n";
        assert_eq!(raw, expected);
    }

    /// LFM2.5's template injects NO default preamble: without a system turn
    /// the tools line stands alone as the whole system prompt.
    #[test]
    fn lfm_tools_render_without_a_system_turn_has_no_preamble() {
        let raw = ChatMl::lfm2().render_with_tools(&[weather_tool()], &[Turn::user("hi")]);
        assert!(raw.starts_with("<|startoftext|><|im_start|>system\nList of tools: ["));
        assert!(raw.contains("<|im_start|>user\nhi<|im_end|>"));
    }

    #[test]
    fn lfm_tools_comma_join_in_one_list() {
        let mut second = weather_tool();
        second.name = "get_time".into();
        let raw = ChatMl::lfm2().render_with_tools(&[weather_tool(), second], &[Turn::user("hi")]);
        assert!(raw.contains("\"name\":\"get_weather\""));
        assert!(raw.contains("}, {\"name\":\"get_time\""));
        assert_eq!(raw.matches("List of tools: [").count(), 1);
    }

    /// LFM tool results are real `tool` role turns — one each, no Hermes
    /// merging, no `<tool_response>` wrapper.
    #[test]
    fn lfm_tool_responses_render_as_tool_turns() {
        let calls = [ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }];
        let raw = ChatMl::lfm2().render(&[
            Turn::user("Weather in Paris and Lyon?"),
            Turn::assistant_tool_calls_lfm(&calls),
            Turn::tool_response("{\"temp_c\": 21}"),
            Turn::tool_response("{\"temp_c\": 24}"),
        ]);
        assert!(raw.contains(
            "<|im_start|>assistant\n<|tool_call_start|>[get_weather(city=\"Paris\")]\
             <|tool_call_end|><|im_end|>\n"
        ));
        assert!(raw.contains("<|im_start|>tool\n{\"temp_c\": 21}<|im_end|>\n"));
        assert!(raw.contains("<|im_start|>tool\n{\"temp_c\": 24}<|im_end|>\n"));
        assert!(!raw.contains("<tool_response>"));
        assert_eq!(raw.matches("<|im_start|>tool\n").count(), 2);
    }

    /// Hermes rendering is untouched by the LFM additions: tool turns still
    /// merge into a user turn.
    #[test]
    fn hermes_tool_turns_still_merge_after_the_style_split() {
        let raw = ChatMl::qwen2().render(&[
            Turn::user("q"),
            Turn::tool_response("r1"),
            Turn::tool_response("r2"),
        ]);
        assert_eq!(raw.matches("<|im_start|>user").count(), 2);
        assert!(!raw.contains("<|im_start|>tool"));
    }

    /// The LFM2.5 template strips `</think>` reasoning from every assistant
    /// history turn but the LAST (keep_past_thinking=false default).
    #[test]
    fn lfm_strips_past_thinking_but_keeps_the_last() {
        let raw = ChatMl::lfm2().render(&[
            Turn::user("a?"),
            Turn::assistant("<think>hmm</think>\n\nAlpha."),
            Turn::user("b?"),
            Turn::assistant("<think>later thoughts</think>\n\nBeta."),
            Turn::user("c?"),
        ]);
        assert!(raw.contains("<|im_start|>assistant\nAlpha.<|im_end|>"));
        assert!(raw.contains("<think>later thoughts</think>"));
        assert!(!raw.contains("hmm"));
    }

    /// Qwen2 (Hermes) does no thinking-stripping — not part of its template.
    #[test]
    fn hermes_keeps_past_thinking_verbatim() {
        let raw = ChatMl::qwen2().render(&[
            Turn::user("a?"),
            Turn::assistant("<think>hmm</think>Alpha."),
            Turn::user("b?"),
        ]);
        assert!(raw.contains("<think>hmm</think>Alpha."));
    }

    #[test]
    fn pythonic_rendering_covers_the_scalar_spellings() {
        let calls = [ToolCall {
            name: "f".into(),
            arguments: serde_json::json!({
                "s": "he said \"hi\"\n",
                "i": -3,
                "x": 1.5,
                "yes": true,
                "no": false,
                "nothing": null,
                "list": [1, "two"],
                "map": {"k": true}
            }),
        }];
        let turn = Turn::assistant_tool_calls_lfm(&calls);
        // serde_json object keys iterate in sorted order.
        assert_eq!(
            turn.content,
            "<|tool_call_start|>[f(i=-3, list=[1, \"two\"], map={\"k\": True}, \
             no=False, nothing=None, s=\"he said \\\"hi\\\"\\n\", x=1.5, \
             yes=True)]<|tool_call_end|>"
        );
    }

    /// The model card's own example parses to the exact call.
    #[test]
    fn lfm_parse_handles_the_model_card_example() {
        let text = "<|tool_call_start|>[get_candidate_status(candidate_id=\"12345\")]\
                    <|tool_call_end|>";
        let (calls, prose) = parse_tool_calls_lfm(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_candidate_status");
        assert_eq!(calls[0].arguments["candidate_id"], "12345");
        assert!(prose.is_empty());
    }

    #[test]
    fn lfm_parse_extracts_multiple_calls_and_prose() {
        let text = "Checking both.\n<|tool_call_start|>[get_weather(city=\"Paris\"), \
                    get_weather(city='Lyon', units=None)]<|tool_call_end|> done";
        let (calls, prose) = parse_tool_calls_lfm(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments["city"], "Paris");
        assert_eq!(calls[1].arguments["city"], "Lyon");
        assert_eq!(calls[1].arguments["units"], serde_json::Value::Null);
        assert_eq!(prose, "Checking both.\n done");
    }

    #[test]
    fn lfm_parse_of_plain_text_is_empty_not_an_error() {
        let (calls, prose) = parse_tool_calls_lfm("The answer is 4.").unwrap();
        assert!(calls.is_empty());
        assert_eq!(prose, "The answer is 4.");
    }

    #[test]
    fn lfm_parse_accepts_python_and_json_literal_spellings() {
        let text = "<|tool_call_start|>[f(a=True, b=false, c=null, d=None, \
                    e=[1, 2.5, -3], g={\"k\": \"v\", 'k2': True,})]<|tool_call_end|>";
        let (calls, _) = parse_tool_calls_lfm(text).unwrap();
        let args = &calls[0].arguments;
        assert_eq!(args["a"], true);
        assert_eq!(args["b"], false);
        assert_eq!(args["c"], serde_json::Value::Null);
        assert_eq!(args["d"], serde_json::Value::Null);
        assert_eq!(args["e"], serde_json::json!([1, 2.5, -3]));
        assert_eq!(args["g"], serde_json::json!({"k": "v", "k2": true}));
    }

    #[test]
    fn lfm_parse_handles_string_escapes() {
        let text = "<|tool_call_start|>[f(s='it\\'s \\\"q\\\" \\u00e9\\n')]<|tool_call_end|>";
        let (calls, _) = parse_tool_calls_lfm(text).unwrap();
        assert_eq!(calls[0].arguments["s"], "it's \"q\" \u{e9}\n");
    }

    #[test]
    fn lfm_parse_rejects_malformed_blocks_loudly() {
        // Unclosed special token.
        assert!(matches!(
            parse_tool_calls_lfm("<|tool_call_start|>[f()]"),
            Err(ToolCallError::Unclosed { index: 0 })
        ));
        // Not a call list.
        assert!(matches!(
            parse_tool_calls_lfm("<|tool_call_start|>f()<|tool_call_end|>"),
            Err(ToolCallError::Syntax { index: 0, .. })
        ));
        // Positional args are not in the grammar.
        assert!(matches!(
            parse_tool_calls_lfm("<|tool_call_start|>[f(\"paris\")]<|tool_call_end|>"),
            Err(ToolCallError::Syntax { .. })
        ));
        // Unterminated string.
        assert!(matches!(
            parse_tool_calls_lfm("<|tool_call_start|>[f(a=\"oops)]<|tool_call_end|>"),
            Err(ToolCallError::Syntax { .. })
        ));
        // Trailing junk after the list.
        assert!(matches!(
            parse_tool_calls_lfm("<|tool_call_start|>[f()] junk<|tool_call_end|>"),
            Err(ToolCallError::Syntax { .. })
        ));
    }

    #[test]
    fn lfm_parse_bounds_depth_and_call_count() {
        // 9 levels of list nesting exceeds MAX_VALUE_DEPTH = 8.
        let deep = format!(
            "<|tool_call_start|>[f(a={}1{})]<|tool_call_end|>",
            "[".repeat(9),
            "]".repeat(9)
        );
        assert!(matches!(
            parse_tool_calls_lfm(&deep),
            Err(ToolCallError::TooDeep)
        ));
        let many = format!(
            "<|tool_call_start|>[{}]<|tool_call_end|>",
            vec!["f()"; MAX_TOOL_CALLS + 1].join(", ")
        );
        assert!(matches!(
            parse_tool_calls_lfm(&many),
            Err(ToolCallError::TooMany)
        ));
    }

    /// The whole LFM loop: a rendered history turn re-parses to the same
    /// calls, through the Pythonic spelling and back.
    #[test]
    fn lfm_tool_calls_round_trip_through_render_and_parse() {
        let calls = vec![ToolCall {
            name: "lookup".into(),
            arguments: serde_json::json!({"q": "primes", "k": 5, "deep": {"a": [true, null]}}),
        }];
        let turn = Turn::assistant_tool_calls_lfm(&calls);
        let (parsed, prose) = parse_tool_calls_lfm(&turn.content).unwrap();
        assert_eq!(parsed, calls);
        assert!(prose.is_empty());
    }
}
