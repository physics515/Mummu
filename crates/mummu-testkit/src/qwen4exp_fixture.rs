//! The RECORDED llama.cpp reference for the qwen4exp (Qwen3.8-Flash-Next)
//! parity gate, and the replay-side verdict.
//!
//! Every other GGUF gate runs llama-server live beside our load. This one
//! cannot: the reference mmaps 111 GB of shards under a ~48 GB container cap
//! and our f32 trunk needs ~20 GB more, on a 124 GB box that also hosts the
//! production stack. So the reference is recorded ONCE into
//! `tests/fixtures/qwen4exp_ud_q4kxl_parity.json` by
//! `tests/parity_qwen4exp_record.rs`, and the gate replays it through
//! [`compare_leg`], which hands the verdict to the SAME
//! `gguf_compare::assert_matches_reference` the live legs use.
//!
//! A binary using this module takes it from `mummu_testkit` (the verdict
//! and `logprobs_at` live in its sibling modules).

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::gguf_compare::{self, TOP_K};
use crate::llama_ref::logprobs_at;

/// Where the committed fixture lives.
pub const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../mummu/tests/fixtures/qwen4exp_ud_q4kxl_parity.json"
);

/// The recorded long-prompt reference: one ~560-token leg whose prefill
/// crosses nine 64-token GDN chunks and whose attention reads ~560 cached
/// positions, which the two short legs never exercise.
pub const LONG_FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../mummu/tests/fixtures/qwen4exp_long_prompt_reference.json"
);

/// Greedy tokens recorded (and compared id for id) on the long leg.
pub const LONG_MAX_TOKENS: usize = 16;

/// The long leg's user message: invented local-history notes plus a
/// question, so the continuation depends on the whole context.
pub const LONG_PROMPT: &str = include_str!("long_prompt.txt");

/// The shard name both sides open; llama.cpp and `GgufFile::open_sharded`
/// each follow the `-of-` naming to the other three.
pub const FIRST_SHARD: &str = "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

/// Bump when the JSON shape changes, so a stale fixture fails to parse
/// loudly instead of replaying half-filled.
pub const FORMAT: u32 = 1;

/// Top-k logprobs recorded per generated position (the gate reads the top
/// [`TOP_K`]; the rest is debugging headroom for rank swaps).
pub const N_PROBS: usize = 10;

/// Max |Δlogprob| over the top-k — the quantized-reference bound from
/// `parity_gguf.rs`.
///
/// (llama.cpp's CPU kernels quantize ACTIVATIONS per dot product on a
/// K-quant file; our path dequantizes weights to f32 once.) The Flash-Next
/// file is the same K-quant/Q8_0 regime, so the same bound applies until a
/// measurement on this model says otherwise.
///
/// That measurement now exists and says the bound (with the strict top-3
/// order) is tighter than llama.cpp's own spread on this model:
/// `parity_qwen4exp::the_gate_is_tighter_than_llama_cpps_own_spread_on_this_model`
/// replays llama.cpp under `--no-repack` / `-fa off` against this fixture and
/// three of those settings fail at least one leg (worst rank-aligned
/// |Δlogprob| 0.94, and top-3 swaps). The value is deliberately unchanged
/// here: re-scoping the qwen4exp verdict is the owner's call.
pub const LOGPROB_ABS_TOLERANCE: f64 = 7.5e-1;

/// The recorded legs as `(name, user prompt)`. The first is the prompt every
/// GGUF gate shares; the second exercises a different first-forward
/// distribution so a single lucky top-3 cannot carry the gate.
pub const LEGS: [(&str, &str); 2] = [
    ("primes", gguf_compare::PROMPT),
    ("moon", "Write one sentence about the moon."),
];

/// The whole recorded reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fixture {
    pub format: u32,
    pub model: ModelInfo,
    pub reference: ReferenceInfo,
    pub legs: Vec<Leg>,
}

/// Which weights the reference ran.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Shard 1's file name; the reference was pointed at exactly this.
    pub first_shard: String,
    /// `/props` `model_ftype` as llama.cpp reports it.
    pub ftype: String,
}

/// Which llama.cpp produced the numbers, and how it was run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceInfo {
    /// Container image (with digest) the server ran from.
    pub image: String,
    /// `/props` `build_info`, e.g. `b10991-930e2fa59`.
    pub build_info: String,
    /// The server's command line, verbatim.
    pub server_args: String,
}

/// One prompt's reference completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Leg {
    pub name: String,
    /// The user message before templating.
    pub prompt: String,
    /// `ChatMl::qwen3()` rendering of `prompt` — what was tokenized.
    pub rendered: String,
    /// Token ids sent to the server, from mummu's GGUF tokenizer.
    pub prompt_ids: Vec<u32>,
    /// The server's own count; equal to `prompt_ids.len()` (no BOS injected).
    pub tokens_evaluated: usize,
    /// The exact `/completion` request body.
    pub request: serde_json::Value,
    /// One entry per generated position; `steps[0].top` is the FIRST-forward
    /// distribution the gate compares.
    pub steps: Vec<Step>,
    /// The generated ids (`tokens` in the response).
    pub greedy_ids: Vec<u32>,
    /// The server's greedy text — what the gate byte-compares against.
    pub content: String,
    /// `tokenizer.decode(greedy_ids, skip_special = true)` with mummu's
    /// tokenizer, recorded so a template/special-token difference between the
    /// server's `content` and our decode is visible in the fixture itself.
    pub greedy_text_mummu: String,
    /// `eos`, `limit`, … as the server reported it.
    pub stop_type: String,
    /// The server's `timings` object (prompt/decode ms and tok/s).
    pub timings: serde_json::Value,
}

/// One generated position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    /// The id the server sampled.
    pub id: u32,
    pub token: String,
    pub logprob: f64,
    /// Best-first top-[`N_PROBS`] pre-sampling natural-log probabilities.
    pub top: Vec<TopEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopEntry {
    pub id: u32,
    pub token: String,
    pub logprob: f64,
}

impl Fixture {
    /// Read and parse the committed fixture, refusing a stale format.
    ///
    /// # Panics
    ///
    /// As [`Self::load_from`]: the fixture at [`FIXTURE_PATH`] cannot be
    /// read, is not the expected JSON, or carries a `format` other than
    /// [`FORMAT`].
    #[must_use]
    pub fn load() -> Self {
        Self::load_from(FIXTURE_PATH)
    }

    /// [`Self::load`] for another recorded fixture of the same shape
    /// (e.g. [`LONG_FIXTURE_PATH`]).
    ///
    /// # Panics
    ///
    /// When `path` cannot be read, when its contents do not parse as a
    /// [`Fixture`], or when its `format` is not [`FORMAT`] (a stale
    /// recording must be re-recorded, not replayed half-filled).
    #[must_use]
    pub fn load_from(path: &str) -> Self {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let f: Self = serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"));
        assert_eq!(
            f.format, FORMAT,
            "fixture format {} but this replay reads {FORMAT} — re-record it",
            f.format
        );
        f
    }

    /// The leg called `name`.
    ///
    /// # Panics
    ///
    /// When the fixture has no leg of that name.
    #[must_use]
    pub fn leg(&self, name: &str) -> &Leg {
        self.legs
            .iter()
            .find(|l| l.name == name)
            .unwrap_or_else(|| panic!("fixture has no leg named {name:?}"))
    }
}

impl Leg {
    /// The reference's first-forward top-[`TOP_K`], best first.
    ///
    /// # Panics
    ///
    /// When the leg recorded no generated positions at all.
    #[must_use]
    pub fn first_forward_top(&self) -> Vec<(u32, f64)> {
        let first = self.steps.first().expect("fixture leg has no steps");
        first
            .top
            .iter()
            .take(TOP_K)
            .map(|e| (e.id, e.logprob))
            .collect()
    }
}

/// Render `prompt` as a single user turn with `ChatMl::qwen3()` and tokenize
/// it without re-adding specials.
///
/// Exactly `tests/parity_qwen35.rs`'s path, shared by the recorder and the
/// replay so the ids cannot drift apart.
///
/// # Panics
///
/// When the tokenizer fails to encode the rendered turn, or when it yields
/// fewer than 8 ids (a template that rendered nothing).
pub fn render_prompt_ids(tok: &Tokenizer, prompt: &str) -> (String, Vec<u32>) {
    let rendered = mummu::chat::ChatMl::qwen3().render(&[mummu::chat::Turn::user(prompt)]);
    let ids = tok
        .encode(rendered.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();
    assert!(ids.len() >= 8, "rendered prompt suspiciously short");
    (rendered, ids)
}

/// The gate for one recorded leg.
///
/// `logits` is OUR first forward over `leg.prompt_ids` (the full vocab row
/// for the last prompt token), `greedy_ids` OUR generated ids (prompt
/// excluded, as `greedy_generate` returns them, `gguf_compare::MAX_TOKENS`
/// requested). Applies the `gguf_compare` policy verbatim — top-3 strict
/// order, top-5 overlap >= 4, rank-aligned max |Δlogprob| <=
/// [`LOGPROB_ABS_TOLERANCE`], decoded greedy text byte-equal over the common
/// trimmed prefix (>= 8 bytes) — and panics on divergence.
///
/// # Panics
///
/// When `ChatMl::qwen3()` no longer renders `leg.prompt` to the recorded
/// text or the tokenizer no longer yields the recorded ids (the replay has
/// no server to catch a drifted prompt), when `logits` is shorter than the
/// reference's vocab, when `greedy_ids` fail to decode, and on every
/// divergence `gguf_compare::assert_matches_reference` rejects.
pub fn compare_leg(leg: &Leg, logits: &[f32], greedy_ids: &[u32], tok: &Tokenizer) {
    // A tokenizer or template change would silently make "our" forward run on
    // different ids than the reference saw; the replay has no server to
    // catch that, so re-derive the ids and insist.
    let (rendered, ids) = render_prompt_ids(tok, &leg.prompt);
    assert_eq!(
        rendered, leg.rendered,
        "[{}] ChatMl::qwen3() renders differently from the recording",
        leg.name
    );
    assert_eq!(
        ids, leg.prompt_ids,
        "[{}] our tokenizer no longer produces the recorded prompt ids",
        leg.name
    );
    let ref_top = leg.first_forward_top();
    assert!(
        ref_top.iter().all(|&(id, _)| (id as usize) < logits.len()),
        "[{}] logits row ({}) is shorter than the reference vocab",
        leg.name,
        logits.len()
    );

    // Extra diagnostic only (not asserted): the policy's logprob bound is
    // rank-aligned, so also report the SAME-id difference over the
    // reference's top-k — the number to read when ranks 4-5 swap.
    let ref_ids: Vec<u32> = ref_top.iter().map(|&(id, _)| id).collect();
    let same_id = logprobs_at(logits, &ref_ids)
        .iter()
        .zip(ref_top.iter())
        .map(|(a, &(_, b))| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let tag = format!("qwen4exp/{}", leg.name);
    eprintln!("[parity/gguf/{tag}] same-id max |Δlogprob| over ref top-{TOP_K}: {same_id:e}");

    let ours = tok.decode(greedy_ids, true).expect("decode");
    gguf_compare::assert_matches_reference(
        &tag,
        logits,
        greedy_ids.len(),
        &ours,
        &ref_top,
        &leg.content,
        LOGPROB_ABS_TOLERANCE,
    );
}
