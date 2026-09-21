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
//! OpenAI's `reasoning_effort`). Opting in passes it through unchanged, and
//! the surface decides what to do with it: OpenAI's gets it inline, and the
//! ollama shim splits it into the `thinking` field ollama clients read
//! ([`Filter::push_split`]).
//!
//! The filter is streaming, because the deltas it sees are token-shaped and
//! a tag can be split across any number of them — `<`, `th`, `ink>` is three
//! deltas and one tag. Anything that could still turn into a tag is held
//! back rather than emitted and regretted.
//!
//! Qwen3 writes `</think>\n\n` before its answer, so a filter that only cut
//! the block out left every answer opening on a blank line (measured live on
//! 2026-09-21: `"\n\n42"`). The think filter drops the whitespace around a
//! block the way ollama's thinking parser (`thinking/parser.go`) does: after
//! it, and before it when nothing but whitespace came first. An answer with
//! no block keeps its leading whitespace, and the whitespace around a block
//! in the middle of an answer is the answer's own. The rule is decided in
//! the order the text arrives, so an answer streamed one token at a time
//! and the same answer filtered whole come out the same. It is the same
//! rule whether the thinking is dropped or, on the shim, split out into
//! `thinking`: the answer is `content` either way.
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
    /// Drop the whitespace around a span that opens the text (see
    /// [`Filter::show`]). The think filter's rule only: the whitespace
    /// around a tool call is the answer's.
    trim: bool,
    /// Text other than whitespace has been let through: its whitespace is
    /// its own from here on.
    begun: bool,
    /// Whitespace the text opened with, held until it is known whether it
    /// was the gap around a leading span (dropped) or the text's own (kept).
    gap: String,
}

impl Default for Filter {
    /// The think filter: `<think>…</think>` spans, and the whitespace around
    /// one that opens the answer.
    fn default() -> Self {
        Self {
            trim: true,
            ..Self::spans(OPEN, CLOSE)
        }
    }
}

/// One piece of an answer, split: what may be shown as the answer, and what
/// was inside a span.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Split {
    pub(crate) visible: String,
    pub(crate) thought: String,
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
            trim: false,
            begun: false,
            gap: String::new(),
        }
    }

    /// Feed one delta; returns the part that may be shown now.
    pub(crate) fn push(&mut self, delta: &str) -> String {
        self.push_split(delta).visible
    }

    /// Feed one delta; returns the part that may be shown now AND the part
    /// of the span it completed — for a caller that shows the thinking
    /// somewhere else instead of dropping it.
    pub(crate) fn push_split(&mut self, delta: &str) -> Split {
        let before = self.thought.len();
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
                    let text: String = self.pending.drain(..self.pending.len() - keep).collect();
                    self.show(&text, &mut out);
                    break;
                };
                let text: String = self.pending.drain(..at).collect();
                self.show(&text, &mut out);
                self.withheld.push_str(self.open);
                self.pending.drain(..self.open.len());
                self.inside = true;
            }
        }
        Split {
            visible: out,
            thought: self.thought[before..].to_owned(),
        }
    }

    /// Has a span opened?
    fn opened(&self) -> bool {
        !self.withheld.is_empty()
    }

    /// Let `text` — outside any span, in the order it came — through to
    /// `out`, minus the whitespace around a span that opened the text.
    ///
    /// Until something other than whitespace arrives, whitespace is held in
    /// `gap`, because what it is depends on what comes next. Text arriving
    /// after a span has opened means the gap was the space around the span —
    /// before it, or Qwen's `\n\n` after `</think>` — and it is dropped,
    /// along with the whitespace this text opens with. Text arriving before
    /// any span means there is none to trim around, and the gap was the
    /// text's own. From then on everything passes as is.
    fn show(&mut self, text: &str, out: &mut String) {
        if !self.trim || self.begun {
            out.push_str(text);
            return;
        }
        let body = text.trim_start();
        if body.is_empty() {
            self.gap.push_str(text);
            return;
        }
        self.begun = true;
        let gap = std::mem::take(&mut self.gap);
        if self.opened() {
            out.push_str(body);
        } else {
            out.push_str(&gap);
            out.push_str(text);
        }
    }

    /// Whatever is still held back at the end of a generation.
    ///
    /// A model that opens `<think>` and is then cut off by `max_tokens`
    /// never closes it, so the block is dropped rather than leaked as a
    /// half-tag — but an unterminated *partial tag* outside a block was
    /// ordinary text all along and is released. So is whitespace still held
    /// when no span ever opened: an answer of nothing else, which the
    /// model meant.
    pub(crate) fn finish(&mut self) -> String {
        self.finish_split().visible
    }

    /// [`Self::finish`], with the unclosed span's tail as `thought`: the
    /// reasoning a token cap cut off is still reasoning.
    pub(crate) fn finish_split(&mut self) -> Split {
        let tail = std::mem::take(&mut self.pending);
        if self.inside {
            self.thought.push_str(&tail);
            self.withheld.push_str(&tail);
            return Split {
                visible: String::new(),
                thought: tail,
            };
        }
        let mut visible = String::new();
        self.show(&tail, &mut visible);
        if !self.opened() {
            visible.push_str(&std::mem::take(&mut self.gap));
        }
        Split {
            visible,
            thought: String::new(),
        }
    }

    /// Did the model open a block that never closed?
    pub(crate) fn truncated(&self) -> bool {
        self.inside
    }

    /// Every span held back so far, verbatim — tags included, and an
    /// unclosed one's tail once [`Self::finish`] has run. For a caller that
    /// puts a think block back where it was (`engine::lift_tool_calls`).
    pub(crate) fn withheld(&self) -> &str {
        &self.withheld
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

    fn split(visible: &str, thought: &str) -> Split {
        Split {
            visible: visible.into(),
            thought: thought.into(),
        }
    }

    /// The ollama shim shows the thinking instead of dropping it, so each
    /// push hands back the thought it completed — and holds a maybe-tag on
    /// either side exactly as `push` does.
    #[test]
    fn push_split_hands_back_the_thought_as_it_arrives() {
        let mut f = Filter::default();
        assert_eq!(f.push_split("<th"), split("", ""), "a maybe-open is held");
        assert_eq!(f.push_split("ink>step "), split("", "step "));
        assert_eq!(
            f.push_split("one</th"),
            split("", "one"),
            "a maybe-close is held"
        );
        assert_eq!(f.push_split("ink>Answer"), split("Answer", ""));
        assert_eq!(f.finish_split(), split("", ""));
        assert_eq!(f.thought, "step one");
    }

    /// A block the token cap cut off ends as thought, half-tag and all;
    /// a half-tag outside a block ends as text, as [`Filter::finish`] has it.
    #[test]
    fn finish_split_hands_back_a_cut_off_block_as_thought() {
        let mut f = Filter::default();
        assert_eq!(f.push_split("<think>still go").thought, "still go");
        assert_eq!(f.push_split("ing</thi").thought, "ing");
        assert_eq!(f.finish_split(), split("", "</thi"));
        assert!(f.truncated());

        let mut f = Filter::default();
        assert_eq!(f.push_split("x <thi"), split("x ", ""));
        assert_eq!(f.finish_split(), split("<thi", ""));
    }

    /// Qwen3's shape, token by token: the blank line after `</think>` is
    /// not the answer's (measured live: `"\n\n42"` before this).
    #[test]
    fn drops_the_blank_line_after_the_block() {
        let deltas = ["<think>", "\n", "Okay.", "\n", "</think>", "\n\n", "4", "2"];
        assert_eq!(run(&deltas), "42");
        assert_eq!(run(&[deltas.concat().as_str()]), "42");
    }

    #[test]
    fn drops_the_whitespace_before_a_block_that_opens_the_answer() {
        assert_eq!(run(&["\n ", "<think>x</think>", "\n\nanswer"]), "answer");
        // More than one block in front of the answer: every gap goes.
        assert_eq!(
            run(&["<think>a</think>\n<think>b</think>\n\nanswer"]),
            "answer"
        );
    }

    #[test]
    fn an_answer_with_no_block_keeps_its_leading_whitespace() {
        assert_eq!(run(&["\n\n", "  indented"]), "\n\n  indented");
        assert_eq!(run(&[" \n"]), " \n", "whitespace alone was the answer");
    }

    #[test]
    fn the_answer_keeps_its_own_whitespace_once_it_has_begun() {
        assert_eq!(run(&["<think>x</think>\n\nA\n\n  B "]), "A\n\n  B ");
        // A block in mid-answer is cut out; the space around it is the
        // answer's, and dropping it would run the words together.
        assert_eq!(
            run(&["Sure.", "<think>x</think>", "\n\nMore"]),
            "Sure.\n\nMore"
        );
    }

    #[test]
    fn a_block_and_nothing_but_whitespace_is_an_empty_answer() {
        assert_eq!(run(&["<think>x</think>", "\n\n"]), "");
    }

    #[test]
    fn whitespace_is_not_emitted_before_it_is_known_to_be_the_answers() {
        let mut f = Filter::default();
        assert_eq!(f.push("\n\n"), "", "could still be the gap before a block");
        assert_eq!(f.push("hi"), "\n\nhi");
    }

    /// The last characters of an answer that could still start a tag are
    /// held back until the end — and are the answer's once it comes.
    #[test]
    fn a_trailing_partial_tag_is_released_at_the_end() {
        for tail in ["<", "<th", "<think"] {
            let mut f = Filter::default();
            assert_eq!(f.push(&format!("<think>x</think>\n\n5 {tail}")), "5 ");
            assert_eq!(f.finish(), tail);
        }
    }

    /// However the answer is cut into deltas, it comes out the same as when
    /// it is filtered whole — which is what keeps a streamed answer and a
    /// buffered one from disagreeing.
    #[test]
    fn any_split_of_an_answer_filters_the_same_as_the_whole() {
        let answers = [
            "<think>\nOkay.\n</think>\n\n42",
            "  \n<think>x</think>\n\n  answer ",
            "\n\n  no block at all",
            " \t\n",
            "hi <think>x</think>\n\nthere",
            "<think>a</think> <think>b</think>\n c <",
            "<think>cut off by the token cap",
            "5 < 7 <th",
        ];
        for text in answers {
            let whole = run(&[text]);
            let chars: Vec<String> = text.chars().map(String::from).collect();
            for size in 1..=chars.len() {
                let deltas: Vec<String> = chars.chunks(size).map(|c| c.concat()).collect();
                let deltas: Vec<&str> = deltas.iter().map(String::as_str).collect();
                assert_eq!(run(&deltas), whole, "{text:?} in deltas of {size} chars");
            }
        }
    }

    #[test]
    fn the_whitespace_around_a_tool_call_is_the_answers() {
        let mut f = Filter::spans("<tool_call>", "</tool_call>");
        let shown = f.push("\n<tool_call>{}</tool_call>\n\nok");
        assert_eq!(shown + &f.finish(), "\n\n\nok");
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
