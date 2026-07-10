//! Explicit chat templates. Prompt wrapping is part of a model's contract —
//! an implicit or slightly-wrong template silently ruins output quality — so
//! templates are code here, never guessed: each per-model constructor is
//! byte-verified against the parity references (the Qwen2 template renders
//! the exact prompt committed in the Candle logits fixture).
//!
//! Both zoo LLMs speak ChatML; LFM2.5 additionally prefixes `<|startoftext|>`.
//! Tool-use templates (Hermes-style, LFM2.5 bracket notation) are P4 follow-ups.

/// Who is speaking in a [`Turn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn tag(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
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
        for turn in turns {
            out.push_str("<|im_start|>");
            out.push_str(turn.role.tag());
            out.push('\n');
            out.push_str(&turn.content);
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        debug_assert!(out.ends_with("assistant\n"), "render must open a turn");
        out
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

    #[test]
    #[should_panic(expected = "double it")]
    fn trailing_assistant_turn_is_rejected() {
        let _ = ChatMl::qwen2().render(&[Turn::user("q"), Turn::assistant("half-done")]);
    }
}
