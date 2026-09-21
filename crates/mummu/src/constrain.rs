//! Grammar-constrained decoding: a token-level validator the decode loop
//! consults before it accepts a sampled id.
//!
//! The problem this solves is ollama's `format: "json"`. A client that asks
//! for JSON and gets prose has no way to tell that the option was ignored —
//! it just fails to parse, with nothing in any log naming the cause. So the
//! constraint is enforced *during* decoding rather than hoped for in the
//! prompt: a token whose bytes would break JSON validity is never sampled,
//! which makes "the answer parses" a property of the decoder instead of a
//! property of the model's mood.
//!
//! **Cost.** The interesting design point is that constraining is nearly
//! free on the happy path. A model that was going to emit valid JSON anyway
//! has its argmax accepted with one automaton replay over the token's bytes
//! — no change to the on-device argmax, no vocabulary readback. Only when
//! the model's preferred token would break the grammar does the loop pay a
//! full readback and walk the logits in descending order for the best legal
//! token. See `decode::generate_loop`.

/// What the decode loop needs from a constraint.
///
/// `Send + Sync` are supertraits, not decoration. A constraint is held
/// across the awaits in `decode::generate_loop`, and mummu-serve spawns that
/// future onto a multi-threaded runtime, so the future must be `Send`.
///
/// `Sync` is the half that is easy to miss: the loop takes a *shared*
/// reference to the constraint for its candidate tests, and `&T: Send`
/// requires `T: Sync`. With only `Send` here, `&dyn Constraint` stays
/// non-`Send`, every caller of `run_chat` fails to compile, and the error
/// names the whole generation future instead of this line.
pub trait Constraint: Send + Sync {
    /// May `id` be appended to what has been accepted so far?
    fn allows(&self, id: u32) -> bool;

    /// Commit `id`. Only ever called with an id [`Self::allows`] accepted.
    fn accept(&mut self, id: u32);

    /// Is the constrained value complete — may generation stop here? While
    /// this is false the loop suppresses EOS, which is what stops a model
    /// from bailing out halfway through an object and handing the client
    /// `{"calories":`.
    fn is_complete(&self) -> bool;
}

// ---------------------------------------------------------------------------
// The JSON automaton
// ---------------------------------------------------------------------------

/// Nesting cap. The container stack is one bit per level packed into a
/// `u128` so the whole automaton state stays `Copy` — testing a candidate
/// token is then a register copy and a byte replay, with no allocation in
/// the inner loop over candidates. Real documents are a handful deep; a
/// model that has opened 128 containers is looping, not answering.
const MAX_DEPTH: u32 = 128;

const LIT_TRUE: u8 = 0;
const LIT_FALSE: u8 = 1;
const LIT_NULL: u8 = 2;

fn literal(kind: u8) -> &'static [u8] {
    match kind {
        LIT_TRUE => b"true",
        LIT_FALSE => b"false",
        _ => b"null",
    }
}

/// Where the byte scanner is inside the document.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum St {
    /// Before the top-level value.
    Start,
    /// Just after `{`: a key, or the closing `}` of an empty object.
    NeedKeyOrEnd,
    /// After `,` in an object: a key, and no longer a `}` (no trailing
    /// commas in JSON).
    NeedKey,
    KeyStr,
    KeyEsc,
    /// Remaining hex digits of a `\uXXXX` escape in a key.
    KeyUni(u8),
    NeedColon,
    /// After `:`, or after `,` in an array.
    NeedValue,
    /// Just after `[`: a value, or the closing `]` of an empty array.
    NeedValueOrEnd,
    Str,
    StrEsc,
    StrUni(u8),
    /// After `-`: a digit is mandatory.
    NumSign,
    /// After a leading `0`: no more integer digits may follow.
    NumZero,
    NumInt,
    /// After `.`: a digit is mandatory.
    NumFracFirst,
    NumFrac,
    /// After `e`/`E`: a sign or a digit.
    NumExpSign,
    /// After an exponent sign: a digit is mandatory.
    NumExpFirst,
    NumExp,
    /// Partway through `true` / `false` / `null`: (which, bytes matched).
    Lit(u8, u8),
    /// A value just ended; expect `,` or a closing bracket.
    AfterValue,
    /// The top-level value closed. Only trailing whitespace may follow.
    Done,
}

/// A byte-level JSON validator: `step` accepts a byte iff the document so
/// far remains a valid *prefix* of some JSON document.
///
/// **The top-level value must be an object or an array.** JSON permits a
/// bare scalar, but a bare scalar has no unambiguous end — `1` is a complete
/// document and so is `12`, so nothing can decide when to stop without
/// waiting for a token the grammar cannot supply. Every consumer of
/// `format: "json"` wants an object anyway, and the restriction is what
/// makes [`Constraint::is_complete`] exact rather than a guess.
#[derive(Clone, Copy, Debug)]
pub struct Json {
    /// One bit per open container, LSB = outermost. 1 = array, 0 = object.
    stack: u128,
    depth: u32,
    st: St,
}

impl Default for Json {
    fn default() -> Self {
        Self::new()
    }
}

impl Json {
    #[must_use]
    pub fn new() -> Self {
        Self {
            stack: 0,
            depth: 0,
            st: St::Start,
        }
    }

    /// The whole document has been read and closed.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.st == St::Done
    }

    fn push(&mut self, is_array: bool) -> bool {
        if self.depth >= MAX_DEPTH {
            return false;
        }
        let bit = 1u128 << self.depth;
        if is_array {
            self.stack |= bit;
        } else {
            self.stack &= !bit;
        }
        self.depth += 1;
        true
    }

    fn top_is_array(&self) -> Option<bool> {
        (self.depth > 0).then(|| (self.stack >> (self.depth - 1)) & 1 == 1)
    }

    /// Pop the innermost container, which must be of the expected kind — a
    /// `]` may not close a `{`.
    fn pop(&mut self, is_array: bool) -> bool {
        if self.top_is_array() == Some(is_array) {
            self.depth -= 1;
            true
        } else {
            false
        }
    }

    /// A value (scalar, string, or a container that just closed) ended.
    fn value_ended(&mut self) {
        self.st = if self.depth == 0 {
            St::Done
        } else {
            St::AfterValue
        };
    }

    /// Start a value from its first byte. `closer` is the bracket that may
    /// appear here instead (an empty container), if any.
    fn begin_value(&mut self, b: u8, closer: Option<u8>) -> bool {
        match b {
            b'"' => {
                self.st = St::Str;
                true
            }
            b'{' => {
                self.push(false) && {
                    self.st = St::NeedKeyOrEnd;
                    true
                }
            }
            b'[' => {
                self.push(true) && {
                    self.st = St::NeedValueOrEnd;
                    true
                }
            }
            b'-' => {
                self.st = St::NumSign;
                true
            }
            b'0' => {
                self.st = St::NumZero;
                true
            }
            b'1'..=b'9' => {
                self.st = St::NumInt;
                true
            }
            b't' => {
                self.st = St::Lit(LIT_TRUE, 1);
                true
            }
            b'f' => {
                self.st = St::Lit(LIT_FALSE, 1);
                true
            }
            b'n' => {
                self.st = St::Lit(LIT_NULL, 1);
                true
            }
            b']' if closer == Some(b']') => {
                self.pop(true) && {
                    self.value_ended();
                    true
                }
            }
            _ => false,
        }
    }

    /// A byte that ends a number without being part of it.
    fn num_terminator(b: u8) -> bool {
        is_ws(b) || b == b',' || b == b'}' || b == b']'
    }

    /// Consume one byte. Returns false if it would break JSON validity, in
    /// which case `self` is left in an unspecified state — callers test on a
    /// copy (see [`Self::accepts`]).
    pub fn step(&mut self, b: u8) -> bool {
        match self.st {
            St::Start => match b {
                _ if is_ws(b) => true,
                b'{' => {
                    self.push(false) && {
                        self.st = St::NeedKeyOrEnd;
                        true
                    }
                }
                b'[' => {
                    self.push(true) && {
                        self.st = St::NeedValueOrEnd;
                        true
                    }
                }
                // A bare top-level scalar is legal JSON but has no decidable
                // end; see the type comment.
                _ => false,
            },

            St::NeedKeyOrEnd => match b {
                _ if is_ws(b) => true,
                b'"' => {
                    self.st = St::KeyStr;
                    true
                }
                b'}' => {
                    self.pop(false) && {
                        self.value_ended();
                        true
                    }
                }
                _ => false,
            },

            St::NeedKey => match b {
                _ if is_ws(b) => true,
                b'"' => {
                    self.st = St::KeyStr;
                    true
                }
                _ => false,
            },

            St::KeyStr => match b {
                b'"' => {
                    self.st = St::NeedColon;
                    true
                }
                b'\\' => {
                    self.st = St::KeyEsc;
                    true
                }
                // Unescaped controls are the only forbidden string content;
                // everything from 0x20 up (including UTF-8 continuation
                // bytes, which is why this works a byte at a time) is fine.
                0x20.. => true,
                _ => false,
            },
            St::KeyEsc => escape(b).is_some_and(|next| {
                self.st = match next {
                    Esc::Simple => St::KeyStr,
                    Esc::Unicode => St::KeyUni(4),
                };
                true
            }),
            St::KeyUni(n) => {
                b.is_ascii_hexdigit() && {
                    self.st = if n == 1 {
                        St::KeyStr
                    } else {
                        St::KeyUni(n - 1)
                    };
                    true
                }
            }

            St::NeedColon => match b {
                _ if is_ws(b) => true,
                b':' => {
                    self.st = St::NeedValue;
                    true
                }
                _ => false,
            },

            St::NeedValue => is_ws(b) || self.begin_value(b, None),
            St::NeedValueOrEnd => is_ws(b) || self.begin_value(b, Some(b']')),

            St::Str => match b {
                b'"' => {
                    self.value_ended();
                    true
                }
                b'\\' => {
                    self.st = St::StrEsc;
                    true
                }
                0x20.. => true,
                _ => false,
            },
            St::StrEsc => escape(b).is_some_and(|next| {
                self.st = match next {
                    Esc::Simple => St::Str,
                    Esc::Unicode => St::StrUni(4),
                };
                true
            }),
            St::StrUni(n) => {
                b.is_ascii_hexdigit() && {
                    self.st = if n == 1 { St::Str } else { St::StrUni(n - 1) };
                    true
                }
            }

            St::NumSign => match b {
                b'0' => {
                    self.st = St::NumZero;
                    true
                }
                b'1'..=b'9' => {
                    self.st = St::NumInt;
                    true
                }
                _ => false,
            },
            // A leading zero admits no further integer digits ("01" is not
            // JSON), so only a fraction, an exponent, or the end.
            St::NumZero => self.number_tail(b, false),
            St::NumInt => self.number_tail(b, true),
            St::NumFracFirst => {
                b.is_ascii_digit() && {
                    self.st = St::NumFrac;
                    true
                }
            }
            St::NumFrac => self.number_tail(b, true),
            St::NumExpSign => match b {
                b'+' | b'-' => {
                    self.st = St::NumExpFirst;
                    true
                }
                b'0'..=b'9' => {
                    self.st = St::NumExp;
                    true
                }
                _ => false,
            },
            St::NumExpFirst => {
                b.is_ascii_digit() && {
                    self.st = St::NumExp;
                    true
                }
            }
            St::NumExp => {
                if b.is_ascii_digit() {
                    true
                } else if Self::num_terminator(b) {
                    self.value_ended();
                    self.step(b)
                } else {
                    false
                }
            }

            St::Lit(kind, at) => {
                let want = literal(kind);
                if want.get(at as usize) != Some(&b) {
                    return false;
                }
                let at = at + 1;
                if at as usize == want.len() {
                    self.value_ended();
                } else {
                    self.st = St::Lit(kind, at);
                }
                true
            }

            St::AfterValue => match b {
                _ if is_ws(b) => true,
                b',' => match self.top_is_array() {
                    Some(true) => {
                        self.st = St::NeedValue;
                        true
                    }
                    Some(false) => {
                        self.st = St::NeedKey;
                        true
                    }
                    None => false,
                },
                b'}' => {
                    self.pop(false) && {
                        self.value_ended();
                        true
                    }
                }
                b']' => {
                    self.pop(true) && {
                        self.value_ended();
                        true
                    }
                }
                _ => false,
            },

            St::Done => is_ws(b),
        }
    }

    /// The shared tail of every number state that may legally end: more
    /// digits (when `digits_ok`), a fraction, an exponent, or a terminator
    /// that closes the number and is then re-dispatched.
    fn number_tail(&mut self, b: u8, digits_ok: bool) -> bool {
        match b {
            b'0'..=b'9' if digits_ok => true,
            b'.' => {
                self.st = St::NumFracFirst;
                true
            }
            b'e' | b'E' => {
                self.st = St::NumExpSign;
                true
            }
            _ if Self::num_terminator(b) => {
                self.value_ended();
                self.step(b)
            }
            _ => false,
        }
    }

    /// Would every byte of `bytes` be accepted, leaving a valid prefix?
    /// Non-mutating: the caller keeps its committed state.
    #[must_use]
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        let mut probe = *self;
        bytes.iter().all(|&b| probe.step(b))
    }

    /// Commit `bytes`, which must have passed [`Self::accepts`].
    pub fn feed(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| self.step(b))
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

enum Esc {
    Simple,
    Unicode,
}

fn escape(b: u8) -> Option<Esc> {
    match b {
        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => Some(Esc::Simple),
        b'u' => Some(Esc::Unicode),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Token bytes
// ---------------------------------------------------------------------------

/// Every token's decoded bytes, indexed by id — the table that turns a
/// byte-level grammar into a token-level one.
///
/// Built once per loaded model (it is a function of the tokenizer alone) and
/// shared behind an `Arc`, because building it decodes the whole vocabulary
/// one id at a time.
pub struct TokenBytes {
    /// `None` for ids that must never be sampled under a constraint: the
    /// special/control tokens, which decode to nothing and whose literal
    /// spelling (`<|im_end|>`) is not text the grammar should ever see.
    bytes: Vec<Option<Box<[u8]>>>,
}

impl TokenBytes {
    /// Decode every id in the vocabulary.
    ///
    /// Specials are detected by decoding with `skip_special_tokens = true`
    /// and finding nothing left — an API-stable test that does not depend on
    /// how a given tokenizer.json happens to spell its added-token table.
    #[must_use]
    pub fn from_tokenizer(tok: &tokenizers::Tokenizer) -> Self {
        let n = tok.get_vocab_size(true);
        let bytes = (0..n as u32)
            .map(|id| {
                let text = tok.decode(&[id], true).ok()?;
                (!text.is_empty()).then(|| text.into_bytes().into_boxed_slice())
            })
            .collect();
        Self { bytes }
    }

    /// The decoded bytes of `id`, or `None` for a special / unusable token.
    #[must_use]
    pub fn get(&self, id: u32) -> Option<&[u8]> {
        self.bytes.get(id as usize)?.as_deref()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

// ---------------------------------------------------------------------------
// The JSON constraint
// ---------------------------------------------------------------------------

/// [`Json`] lifted to token ids through a [`TokenBytes`] table.
pub struct JsonConstraint<T: std::ops::Deref<Target = TokenBytes> + Send + Sync> {
    vocab: T,
    state: Json,
}

impl<T: std::ops::Deref<Target = TokenBytes> + Send + Sync> JsonConstraint<T> {
    #[must_use]
    pub fn new(vocab: T) -> Self {
        Self {
            vocab,
            state: Json::new(),
        }
    }
}

impl<T: std::ops::Deref<Target = TokenBytes> + Send + Sync> Constraint for JsonConstraint<T> {
    // No memo. A test is a register copy plus a replay of the token's bytes
    // (a handful, and it stops at the first bad one), so caching per id
    // would cost more in hashing than the automaton costs outright — and a
    // cache would have to be invalidated on every accepted token anyway.
    fn allows(&self, id: u32) -> bool {
        self.vocab
            .get(id)
            .is_some_and(|bytes| self.state.accepts(bytes))
    }

    fn accept(&mut self, id: u32) {
        let bytes = self
            .vocab
            .get(id)
            .expect("accept called with a token the constraint rejected");
        let ok = self.state.feed(bytes);
        debug_assert!(ok, "accept called with a token the constraint rejected");
    }

    fn is_complete(&self) -> bool {
        self.state.is_complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(s: &str) -> Json {
        let mut j = Json::new();
        assert!(j.feed(s.as_bytes()), "should accept prefix {s:?}");
        j
    }

    fn rejects(s: &str) -> bool {
        let mut j = Json::new();
        !j.feed(s.as_bytes())
    }

    #[test]
    fn accepts_a_flat_object() {
        let j = feed(r#"{"calories": 412, "protein_g": 31.5}"#);
        assert!(j.is_complete());
    }

    #[test]
    fn accepts_nesting_and_arrays() {
        let j = feed(r#"{"items":[{"n":"egg","kcal":78},{"n":"toast","kcal":120}],"ok":true}"#);
        assert!(j.is_complete());
    }

    #[test]
    fn incomplete_until_the_last_brace() {
        for prefix in [
            r#"{"#,
            r#"{"a""#,
            r#"{"a":"#,
            r#"{"a":1"#,
            r#"{"a":1,"b":2"#,
        ] {
            assert!(
                !feed(prefix).is_complete(),
                "{prefix:?} must not be complete"
            );
        }
        assert!(feed(r#"{"a":1,"b":2}"#).is_complete());
    }

    #[test]
    fn rejects_the_classic_json_mistakes() {
        assert!(rejects(r#"{"a":1,}"#), "trailing comma in object");
        assert!(rejects(r#"[1,2,]"#), "trailing comma in array");
        assert!(rejects(r#"{a:1}"#), "unquoted key");
        assert!(rejects(r#"{'a':1}"#), "single-quoted key");
        assert!(rejects(r#"{"a":01}"#), "leading zero");
        assert!(rejects(r#"{"a":+1}"#), "leading plus");
        assert!(rejects(r#"{"a":.5}"#), "bare fraction");
        assert!(rejects(r#"{"a":1.}"#), "trailing dot");
        assert!(rejects(r#"{"a":tru}"#), "truncated literal");
        assert!(rejects(r#"{"a":True}"#), "capitalised literal");
        assert!(rejects(r#"{"a":1]"#), "bracket closes a brace");
        assert!(rejects("[1,2}"), "brace closes a bracket");
    }

    #[test]
    fn rejects_a_bare_top_level_scalar() {
        // Legal JSON, deliberately excluded: no decidable end. See `Json`.
        assert!(rejects("42"));
        assert!(rejects(r#""hello""#));
        assert!(rejects("true"));
    }

    #[test]
    fn nothing_but_whitespace_follows_a_finished_document() {
        let mut j = feed("{}");
        assert!(j.is_complete());
        assert!(j.feed(b"  \n"), "trailing whitespace is fine");
        assert!(j.is_complete());
        assert!(!j.step(b'{'), "a second document is not");
    }

    #[test]
    fn strings_handle_escapes_and_utf8() {
        assert!(feed(r#"{"a":"line\nbreak é \" \\ done"}"#).is_complete());
        // Raw UTF-8 bytes arrive one at a time, mid-token, and must pass.
        assert!(feed("{\"a\":\"caf\u{e9} \u{1f957}\"}").is_complete());
        assert!(rejects("{\"a\":\"raw\nnewline\"}"), "unescaped control");
        assert!(rejects(r#"{"a":"\q"}"#), "invalid escape");
        assert!(rejects(r#"{"a":"\u00g0"}"#), "bad hex in \\u");
    }

    #[test]
    fn numbers_cover_the_whole_grammar() {
        assert!(feed(r#"{"a":0,"b":-1,"c":1.5,"d":1e9,"e":-2.5E-3,"f":0.0}"#).is_complete());
    }

    #[test]
    fn accepts_probes_without_mutating() {
        let j = feed(r#"{"a":"#);
        assert!(j.accepts(b"123"));
        assert!(j.accepts(b"\"str\""));
        assert!(!j.accepts(b"}"), "a value is mandatory after the colon");
        // The probes left the committed state alone.
        assert!(j.accepts(b"true"));
        assert!(!j.is_complete());
    }

    #[test]
    fn depth_is_capped_rather_than_overflowing_the_stack_word() {
        let mut j = Json::new();
        for _ in 0..MAX_DEPTH {
            assert!(j.step(b'['));
        }
        assert!(!j.step(b'['), "the {MAX_DEPTH}th nesting must be refused");
    }

    /// A tiny byte-level vocabulary is enough to drive the token layer.
    fn toy_vocab() -> TokenBytes {
        let pieces: Vec<Option<Box<[u8]>>> = [
            Some(&b"{"[..]),
            Some(&b"}"[..]),
            Some(&b"\""[..]),
            Some(&b"kcal"[..]),
            Some(&b":"[..]),
            Some(&b"412"[..]),
            Some(&b" Sure!"[..]),
            None, // a special token
        ]
        .into_iter()
        .map(|p| p.map(<[u8]>::to_vec).map(Vec::into_boxed_slice))
        .collect();
        TokenBytes { bytes: pieces }
    }

    #[test]
    fn token_constraint_walks_a_document() {
        let v = toy_vocab();
        let mut c = JsonConstraint::new(&v);
        // Prose is refused at position 0; only `{` opens a document.
        assert!(!c.allows(6), "prose must not start a JSON document");
        assert!(!c.allows(7), "a special token is never allowed");
        assert!(c.allows(0));
        for id in [0u32, 2, 3, 2, 4, 5] {
            assert!(c.allows(id), "id {id} should be legal here");
            c.accept(id);
            assert!(!c.is_complete());
        }
        assert!(c.allows(1));
        c.accept(1);
        assert!(c.is_complete(), "the closing brace completes the document");
    }
}
