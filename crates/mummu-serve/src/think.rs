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
//!
//! The same machinery holds a family's tool-call markup back from a streamed
//! answer on both surfaces, whose calls go out structured once they are
//! whole (see `crate::shim` and `crate::openai`): [`Filter::spans`] with the
//! family's tags, and [`Filter::settle`] for what is owed at the end.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// Streaming remover of `<think>…</think>` spans (or of any other pair of
/// tags, see [`Filter::spans`]).
pub(crate) struct Filter {
    /// The tag a span opens with.
    open: &'static str,
    /// The tag that closes it.
    close: &'static str,
    /// Text held back: either a partial tag, or the inside of a block.
    pending: String,
    /// Inside a block.
    inside: bool,
    /// Everything that was suppressed, in order.
    thought: String,
    /// The same, tags and all: what a caller that cannot use the spans after
    /// all hands back as ordinary text (see [`Filter::settle`]).
    withheld: String,
}

impl Default for Filter {
    fn default() -> Self {
        Self::spans(OPEN, CLOSE)
    }
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
    /// A filter for the spans between `open` and `close`.
    pub(crate) fn spans(open: &'static str, close: &'static str) -> Self {
        Self {
            open,
            close,
            pending: String::new(),
            inside: false,
            thought: String::new(),
            withheld: String::new(),
        }
    }

    /// Feed one delta; returns the part that may be shown now.
    pub(crate) fn push(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let mut out = String::new();
        loop {
            if self.inside {
                let Some(at) = self.pending.find(self.close) else {
                    // Hold everything that could still be part of the
                    // closing tag; the rest is thinking and is recorded.
                    let keep = dangling(&self.pending, self.close);
                    let split = self.pending.len() - keep;
                    self.thought.push_str(&self.pending[..split]);
                    self.withheld.push_str(&self.pending[..split]);
                    self.pending.drain(..split);
                    break;
                };
                self.thought.push_str(&self.pending[..at]);
                self.withheld
                    .push_str(&self.pending[..at + self.close.len()]);
                self.pending.drain(..at + self.close.len());
                self.inside = false;
            } else {
                let Some(at) = self.pending.find(self.open) else {
                    let keep = dangling(&self.pending, self.open);
                    let split = self.pending.len() - keep;
                    out.push_str(&self.pending[..split]);
                    self.pending.drain(..split);
                    break;
                };
                out.push_str(&self.pending[..at]);
                self.withheld.push_str(self.open);
                self.pending.drain(..at + self.open.len());
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
            self.withheld.push_str(&tail);
            String::new()
        } else {
            std::mem::take(&mut self.pending)
        }
    }

    /// Did the model open a block that never closed?
    pub(crate) fn truncated(&self) -> bool {
        self.inside
    }

    /// End a stream that held spans back to use them: the text still owed.
    ///
    /// That is the partial tag at the very end, which was ordinary text all
    /// along, and, when the spans went unused (`used` false: nothing in them
    /// parsed), every span held back, verbatim and tags included, so nothing
    /// the model wrote goes missing.
    pub(crate) fn settle(&mut self, used: bool) -> String {
        let rest = self.finish();
        if used {
            rest
        } else {
            format!("{}{rest}", self.withheld)
        }
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
    fn other_tags_are_held_back_the_same_way_and_kept_verbatim() {
        let mut f = Filter::spans("<tool_call>", "</tool_call>");
        let shown: String = [
            "Sure. <tool",
            "_call>{\"name\": ",
            "\"x\"}</tool_",
            "call> done",
        ]
        .iter()
        .map(|d| f.push(d))
        .collect();
        assert_eq!(shown + &f.finish(), "Sure.  done");
        assert_eq!(f.withheld, "<tool_call>{\"name\": \"x\"}</tool_call>");
        // A <think> tag means nothing to a filter for other tags.
        assert_eq!(f.push("<think>kept</think>"), "<think>kept</think>");
    }

    #[test]
    fn an_unclosed_span_is_withheld_whole() {
        let mut f = Filter::spans("<tool_call>", "</tool_call>");
        assert_eq!(f.push("<tool_call>{\"name\": \"cut off"), "");
        assert_eq!(f.finish(), "");
        assert_eq!(f.withheld, "<tool_call>{\"name\": \"cut off");
    }

    /// What a stream that held calls back still owes at the end: when the
    /// spans were used, only a partial tag left dangling; when they were not,
    /// every span as well, so the client still sees all the model wrote.
    #[test]
    fn settling_releases_the_spans_only_when_they_went_unused() {
        let deltas = ["a <tool_call>{\"name\": \"x\"}</tool_call> b <tool_"];
        let mut used = Filter::spans("<tool_call>", "</tool_call>");
        assert_eq!(used.push(deltas[0]), "a  b ");
        assert_eq!(
            used.settle(true),
            "<tool_",
            "a dangling partial tag is text"
        );
        let mut unused = Filter::spans("<tool_call>", "</tool_call>");
        assert_eq!(unused.push(deltas[0]), "a  b ");
        assert_eq!(
            unused.settle(false),
            "<tool_call>{\"name\": \"x\"}</tool_call><tool_"
        );
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
