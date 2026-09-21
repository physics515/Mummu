//! Separating a reasoning model's thinking from its answer.
//!
//! Qwen3.8 opens a reply with a `<think>…</think>` block and only then says
//! the thing the user asked for. mummu used to stream that through verbatim,
//! which is wrong in two ways for a client that did not ask for it: the
//! thinking is not the answer, and it competes for the same token budget.
//! Measured live on 2026-09-20 — a 40-token request came back as forty
//! tokens of deliberation and no reply at all, and the calorie tracker this
//! was written for caps replies at 600.
//!
//! So thinking is suppressed unless a request opts in (ollama's `think`,
//! OpenAI's `reasoning_effort`). Opting in passes it through unchanged.
//!
//! The filter is streaming, because the deltas it sees are token-shaped and
//! a tag can be split across any number of them — `<`, `th`, `ink>` is three
//! deltas and one tag. Anything that could still turn into a tag is held
//! back rather than emitted and regretted.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// Streaming remover of `<think>…</think>` spans.
#[derive(Default)]
pub(crate) struct Filter {
    /// Text held back: either a partial tag, or the inside of a block.
    pending: String,
    /// Inside a `<think>` block.
    inside: bool,
    /// Everything that was suppressed, in order.
    thought: String,
}

/// The longest proper prefix of `tag` that `s` ends with — the part that
/// might still become a tag once more text arrives.
fn dangling(s: &str, tag: &str) -> usize {
    let max = (tag.len() - 1).min(s.len());
    (1..=max)
        .rev()
        .find(|&n| s.is_char_boundary(s.len() - n) && tag.starts_with(&s[s.len() - n..]))
        .unwrap_or(0)
}

impl Filter {
    /// Feed one delta; returns the part that may be shown now.
    pub(crate) fn push(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let mut out = String::new();
        loop {
            if self.inside {
                let Some(at) = self.pending.find(CLOSE) else {
                    // Hold everything that could still be part of the
                    // closing tag; the rest is thinking and is recorded.
                    let keep = dangling(&self.pending, CLOSE);
                    let split = self.pending.len() - keep;
                    self.thought.push_str(&self.pending[..split]);
                    self.pending.drain(..split);
                    break;
                };
                self.thought.push_str(&self.pending[..at]);
                self.pending.drain(..at + CLOSE.len());
                self.inside = false;
            } else {
                let Some(at) = self.pending.find(OPEN) else {
                    let keep = dangling(&self.pending, OPEN);
                    let split = self.pending.len() - keep;
                    out.push_str(&self.pending[..split]);
                    self.pending.drain(..split);
                    break;
                };
                out.push_str(&self.pending[..at]);
                self.pending.drain(..at + OPEN.len());
                self.inside = true;
            }
        }
        out
    }

    /// Whatever is still held back at the end of a generation.
    ///
    /// A model that opens `<think>` and is then cut off by `max_tokens`
    /// never closes it, so the block is dropped rather than leaked as a
    /// half-tag — but an unterminated *partial tag* outside a block was
    /// ordinary text all along and is released.
    pub(crate) fn finish(&mut self) -> String {
        if self.inside {
            let tail = std::mem::take(&mut self.pending);
            self.thought.push_str(&tail);
            String::new()
        } else {
            std::mem::take(&mut self.pending)
        }
    }

    /// Did the model open a block that never closed?
    pub(crate) fn truncated(&self) -> bool {
        self.inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(deltas: &[&str]) -> String {
        let mut f = Filter::default();
        let mut out: String = deltas.iter().map(|d| f.push(d)).collect();
        out.push_str(&f.finish());
        out
    }

    #[test]
    fn strips_a_whole_block() {
        assert_eq!(run(&["<think>reasoning</think>The answer"]), "The answer");
    }

    #[test]
    fn passes_text_with_no_block() {
        assert_eq!(run(&["just ", "an ", "answer"]), "just an answer");
    }

    #[test]
    fn handles_a_tag_split_across_deltas() {
        // The way it actually arrives: one token at a time.
        assert_eq!(
            run(&["<", "th", "ink", ">", "hmm", "</", "think", ">", "Hi"]),
            "Hi"
        );
    }

    #[test]
    fn emits_nothing_early_for_a_partial_open_tag() {
        let mut f = Filter::default();
        assert_eq!(f.push("<th"), "", "must not leak a maybe-tag");
        assert_eq!(f.push("ink>x</think>done"), "done");
    }

    #[test]
    fn a_lone_less_than_is_ordinary_text() {
        assert_eq!(run(&["5 < 7 and 8 > 2"]), "5 < 7 and 8 > 2");
    }

    #[test]
    fn an_unclosed_block_is_dropped_not_leaked() {
        let mut f = Filter::default();
        assert_eq!(f.push("<think>still going"), "");
        assert_eq!(f.finish(), "", "a truncated block yields no answer text");
        assert!(f.truncated(), "and says so");
    }

    #[test]
    fn text_before_a_block_survives() {
        assert_eq!(run(&["before <think>mid</think> after"]), "before  after");
    }

    #[test]
    fn records_what_it_suppressed() {
        let mut f = Filter::default();
        let _ = f.push("<think>the reasoning</think>answer");
        assert_eq!(f.thought, "the reasoning");
    }

    #[test]
    fn multibyte_text_is_not_split_mid_character() {
        assert_eq!(run(&["café <think>x</think> ☕"]), "café  ☕");
        // A dangling byte of a multi-byte char must not be mistaken for a tag.
        let mut f = Filter::default();
        let out: String = ["caf", "é ☕"].iter().map(|d| f.push(d)).collect();
        assert_eq!(out + &f.finish(), "café ☕");
    }
}
