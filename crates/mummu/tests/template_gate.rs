//! Template-vs-renderer BYTE gate (the `hf-chat-template` evaluation, P3).
//!
//! Mummu's prompt wrapping is hardcoded, byte-verified `chat` renderers; the
//! checkpoint ships the *authoritative* Jinja `chat_template` we import but do
//! not render. This gate renders the IMPORTED template through
//! `hf-chat-template` (byte-identical to `transformers.apply_chat_template`)
//! and compares against our `ChatMl::qwen2()` renderer on the same
//! conversation — turning "template-vs-renderer consistency" from a marker
//! check into a byte comparison. Ignored by default; run with:
//!
//! ```text
//! MUMMU_QWEN3_DIR=path/to/qwen3-0.6b \
//!   cargo test -p mummu --test template_gate -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use hf_chat_template::{ChatTemplate, Message, RenderInput};
use mummu::chat::{ChatMl, ToolSpec, Turn};
use mummu::tok_config::TokenizerConfig;

fn dir() -> Option<PathBuf> {
    std::env::var_os("MUMMU_QWEN3_DIR")
        .map(PathBuf::from)
        .filter(|d| d.join("tokenizer_config.json").is_file())
}

/// Load the checkpoint's imported chat template and compile it.
fn imported_template(dir: &Path) -> ChatTemplate {
    let cfg = TokenizerConfig::from_dir(dir).expect("tokenizer_config.json parses");
    let template = cfg
        .chat_template
        .as_deref()
        .expect("Qwen3 ships a chat template");
    assert!(!template.is_empty(), "template is non-empty");
    ChatTemplate::from_str(template).expect("the imported template compiles")
}

/// Point out the first byte where two renders diverge (for a readable failure).
fn first_diff(a: &str, b: &str) -> Option<usize> {
    a.bytes()
        .zip(b.bytes())
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

fn diff_context(label: &str, ours: &str, reference: &str) -> String {
    match first_diff(ours, reference) {
        None => format!("{label}: byte-identical ({} B)", ours.len()),
        Some(pos) => {
            let lo = pos.saturating_sub(60);
            format!(
                "{label}: DIVERGES at byte {pos}\n  ours     …{:?}\n  reference…{:?}",
                &ours[lo..(pos + 60).min(ours.len())],
                &reference[lo..(pos + 60).min(reference.len())],
            )
        }
    }
}

/// The plain conversation leg: system + user + generation prompt must render
/// BYTE-IDENTICALLY through the checkpoint's own template and our renderer.
/// This is the exact shape the parity gates commit.
#[test]
#[ignore = "needs the local Qwen3 checkpoint dir (MUMMU_QWEN3_DIR)"]
fn qwen3_plain_render_byte_matches_the_imported_template() {
    let dir = dir().expect("set MUMMU_QWEN3_DIR to a Qwen3 checkpoint dir");
    let template = imported_template(&dir);

    let reference = template
        .render(&RenderInput {
            messages: vec![
                Message::system("You are a helpful assistant."),
                Message::user("List the first five prime numbers."),
            ],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");
    let ours = ChatMl::qwen2().render(&[
        Turn::system("You are a helpful assistant."),
        Turn::user("List the first five prime numbers."),
    ]);

    println!("{}", diff_context("plain", &ours, &reference));
    assert_eq!(ours, reference, "plain ChatML render must byte-match");
}

/// Multi-turn history (user → assistant → user): the template renders a
/// history assistant turn plainly (no think-block resurrection), same as ours.
#[test]
#[ignore = "needs the local Qwen3 checkpoint dir (MUMMU_QWEN3_DIR)"]
fn qwen3_multi_turn_render_byte_matches_the_imported_template() {
    let dir = dir().expect("set MUMMU_QWEN3_DIR to a Qwen3 checkpoint dir");
    let template = imported_template(&dir);

    let reference = template
        .render(&RenderInput {
            messages: vec![
                Message::system("You are a helpful assistant."),
                Message::user("Name a prime."),
                Message::assistant("2 is prime."),
                Message::user("Another?"),
            ],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");
    let ours = ChatMl::qwen2().render(&[
        Turn::system("You are a helpful assistant."),
        Turn::user("Name a prime."),
        Turn::assistant("2 is prime."),
        Turn::user("Another?"),
    ]);

    println!("{}", diff_context("multi-turn", &ours, &reference));
    assert_eq!(ours, reference, "multi-turn ChatML render must byte-match");
}

/// The tools leg: the full Hermes `# Tools` system block — tool JSON included
/// — must render byte-identically. This holds because `chat` serializes
/// prompt JSON with Python `json.dumps` separators (`python_json`), exactly
/// what the template's `tojson` produces.
#[test]
#[ignore = "needs the local Qwen3 checkpoint dir (MUMMU_QWEN3_DIR)"]
fn qwen3_tools_render_vs_imported_template() {
    let dir = dir().expect("set MUMMU_QWEN3_DIR to a Qwen3 checkpoint dir");
    let template = imported_template(&dir);

    let tool_json = serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    });
    let reference = template
        .render(&RenderInput {
            messages: vec![
                Message::system("You are a helpful assistant."),
                Message::user("What's the weather in Paris?"),
            ],
            tools: vec![tool_json],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");

    let spec = ToolSpec {
        name: "get_weather".into(),
        description: "Get the current weather in a city.".into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }),
    };
    let ours = ChatMl::qwen2().render_with_tools(
        &[spec],
        &[
            Turn::system("You are a helpful assistant."),
            Turn::user("What's the weather in Paris?"),
        ],
    );

    println!("{}", diff_context("tools", &ours, &reference));
    assert_eq!(ours, reference, "Hermes tools render must byte-match");
}

/// Function-calling HISTORY: an assistant `<tool_call>` turn plus its tool
/// response must re-render byte-identically through both paths — the whole
/// multi-step-tool loop an agent replays every round.
#[test]
#[ignore = "needs the local Qwen3 checkpoint dir (MUMMU_QWEN3_DIR)"]
fn qwen3_tool_history_render_byte_matches_the_imported_template() {
    let dir = dir().expect("set MUMMU_QWEN3_DIR to a Qwen3 checkpoint dir");
    let template = imported_template(&dir);

    let call_json = serde_json::json!({
        "name": "get_weather",
        "arguments": {"city": "Paris"}
    });
    let mut assistant_call = Message::assistant("");
    assistant_call.content = None;
    assistant_call.tool_calls = vec![call_json];
    let reference = template
        .render(&RenderInput {
            messages: vec![
                Message::system("You are a helpful assistant."),
                Message::user("What's the weather in Paris?"),
                assistant_call,
                Message::new("tool", "{\"temp_c\": 21}"),
            ],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");

    let calls = [mummu::chat::ToolCall {
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "Paris"}),
    }];
    let ours = ChatMl::qwen2().render(&[
        Turn::system("You are a helpful assistant."),
        Turn::user("What's the weather in Paris?"),
        Turn::assistant_tool_calls(&calls),
        Turn::tool_response("{\"temp_c\": 21}"),
    ]);

    println!("{}", diff_context("tool-history", &ours, &reference));
    assert_eq!(ours, reference, "FC history render must byte-match");
}

// ---------------------------------------------------------------------------
// Coverage beyond Qwen3 (2026-07-24): Qwen2.5 + LFM2.5 legs, and the known
// family divergences PINNED to their exact deltas so any other drift fails.
// Extra env keys: MUMMU_QWEN2_DIR, MUMMU_LFM2_DIR (LFM's legs also exercise
// the standalone `chat_template.jinja` import fallback — the checkpoint dir
// carries the template only as that file).
// ---------------------------------------------------------------------------

fn dir_from(var: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os(var)?);
    dir.is_dir().then_some(dir)
}

fn weather_spec() -> ToolSpec {
    ToolSpec {
        name: "get_weather".into(),
        description: "Get the current weather in a city.".into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }),
    }
}

/// The Hermes wire JSON transformers hands the template as one `tools` entry.
fn hermes_tool_json() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    })
}

/// LFM tools are BARE tool JSON (no Hermes wrapper) — the template runs
/// `tool | tojson` on whatever is passed; the model card shows bare.
fn lfm_tool_json() -> serde_json::Value {
    serde_json::json!({
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    })
}

/// LFM's template opens with `{{- bos_token -}}` — pass it in the context,
/// exactly as transformers does from the tokenizer's special tokens.
fn lfm_input(messages: Vec<Message>, tools: Vec<serde_json::Value>) -> RenderInput {
    let mut extra = serde_json::Map::new();
    extra.insert(
        "bos_token".into(),
        serde_json::Value::String("<|startoftext|>".into()),
    );
    RenderInput {
        messages,
        tools,
        add_generation_prompt: true,
        extra,
        ..RenderInput::default()
    }
}

const SYSTEM: &str = "You are a helpful assistant.";
const USER: &str = "List the first five prime numbers.";

// ---- Qwen2.5 ---------------------------------------------------------------

/// Qwen2.5's plain + tools + FC-history legs, byte-identical like Qwen3's.
#[test]
#[ignore = "needs the local Qwen2.5 checkpoint dir (MUMMU_QWEN2_DIR)"]
fn qwen2_renders_byte_match_the_imported_template() {
    let dir = dir_from("MUMMU_QWEN2_DIR").expect("set MUMMU_QWEN2_DIR");
    let template = imported_template(&dir);

    // Plain — the exact shape the Qwen2 parity gate commits.
    let reference = template
        .render(&RenderInput {
            messages: vec![Message::system(SYSTEM), Message::user(USER)],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");
    let ours = ChatMl::qwen2().render(&[Turn::system(SYSTEM), Turn::user(USER)]);
    println!("{}", diff_context("qwen2 plain", &ours, &reference));
    assert_eq!(ours, reference, "Qwen2.5 plain render must byte-match");

    // Tools.
    let reference = template
        .render(&RenderInput {
            messages: vec![Message::system(SYSTEM), Message::user(USER)],
            tools: vec![hermes_tool_json()],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");
    let ours = ChatMl::qwen2()
        .render_with_tools(&[weather_spec()], &[Turn::system(SYSTEM), Turn::user(USER)]);
    println!("{}", diff_context("qwen2 tools", &ours, &reference));
    assert_eq!(ours, reference, "Qwen2.5 tools render must byte-match");

    // FC history: a <tool_call> turn + two tool responses (merged user turn).
    let mut assistant_call = Message::assistant("");
    assistant_call.content = None;
    assistant_call.tool_calls =
        vec![serde_json::json!({"name": "get_weather", "arguments": {"city": "Paris"}})];
    let reference = template
        .render(&RenderInput {
            messages: vec![
                Message::system(SYSTEM),
                Message::user("What's the weather in Paris?"),
                assistant_call,
                Message::new("tool", "{\"temp_c\": 21}"),
                Message::new("tool", "{\"temp_c\": 24}"),
            ],
            tools: vec![hermes_tool_json()],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");
    let calls = [mummu::chat::ToolCall {
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "Paris"}),
    }];
    let ours = ChatMl::qwen2().render_with_tools(
        &[weather_spec()],
        &[
            Turn::system(SYSTEM),
            Turn::user("What's the weather in Paris?"),
            Turn::assistant_tool_calls(&calls),
            Turn::tool_response("{\"temp_c\": 21}"),
            Turn::tool_response("{\"temp_c\": 24}"),
        ],
    );
    println!("{}", diff_context("qwen2 fc-history", &ours, &reference));
    assert_eq!(ours, reference, "Qwen2.5 FC history render must byte-match");
}

/// Documented divergence, pinned exactly: without a system turn Qwen2.5's
/// template injects its branding preamble where `qwen2()` injects a neutral
/// one (a deliberate neutrality choice — `qwen3()` matches ITS template's
/// no-preamble exactly). The delta must be EXACTLY the preamble swap.
#[test]
#[ignore = "needs the local Qwen2.5 checkpoint dir (MUMMU_QWEN2_DIR)"]
fn qwen2_no_system_defaults_diverge_only_by_the_documented_preamble() {
    let dir = dir_from("MUMMU_QWEN2_DIR").expect("set MUMMU_QWEN2_DIR");
    let template = imported_template(&dir);
    const QWEN_PREAMBLE: &str =
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    // With tools: both sides synthesize a system turn; preambles differ.
    let reference = template
        .render(&RenderInput {
            messages: vec![Message::user(USER)],
            tools: vec![hermes_tool_json()],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");
    let ours = ChatMl::qwen2().render_with_tools(&[weather_spec()], &[Turn::user(USER)]);
    let ours_with_qwen_preamble = ours.replacen(SYSTEM, QWEN_PREAMBLE, 1);
    assert_ne!(
        ours_with_qwen_preamble, ours,
        "our preamble must be present"
    );
    assert_eq!(
        ours_with_qwen_preamble, reference,
        "no-system tools render must diverge ONLY by the default preamble"
    );

    // Without tools: the template injects a whole default system turn; our
    // render() injects nothing (explicit turns are the caller's contract).
    let reference = template
        .render(&RenderInput {
            messages: vec![Message::user(USER)],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");
    let ours = ChatMl::qwen2().render(&[Turn::user(USER)]);
    assert_eq!(
        format!("<|im_start|>system\n{QWEN_PREAMBLE}<|im_end|>\n{ours}"),
        reference,
        "no-system plain render must diverge ONLY by the injected default turn"
    );
}

// ---- Qwen3 behavioral deltas (ChatMl::qwen3) --------------------------------

/// `ChatMl::qwen3()` carries the template's two behavioral deltas exactly —
/// what used to be pinned divergences of the shared `qwen2()` renderer are
/// now byte-equal: (a) with tools and no system turn the template injects NO
/// preamble; (b) `<think>` reasoning is stripped from assistant turns
/// at/before the last user query; (c) an assistant turn AFTER the last user
/// query (mid tool loop) keeps its reasoning in the template's normalized
/// `<think>\n…\n</think>\n\n` re-emission. `qwen2()` still re-renders
/// history verbatim (its own template has no think path — pinned below).
#[test]
#[ignore = "needs the local Qwen3 checkpoint dir (MUMMU_QWEN3_DIR)"]
fn qwen3_history_and_no_system_renders_byte_match_the_imported_template() {
    let dir = dir().expect("set MUMMU_QWEN3_DIR to a Qwen3 checkpoint dir");
    let template = imported_template(&dir);

    // (a) Tools without a system turn: no preamble on either side.
    let reference = template
        .render(&RenderInput {
            messages: vec![Message::user(USER)],
            tools: vec![hermes_tool_json()],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");
    let ours = ChatMl::qwen3().render_with_tools(&[weather_spec()], &[Turn::user(USER)]);
    println!(
        "{}",
        diff_context("qwen3 tools no-system", &ours, &reference)
    );
    assert_eq!(
        ours, reference,
        "Qwen3 no-system tools render must byte-match"
    );

    // (b) Think-strip: reasoning at/before the last user query disappears.
    let think_turn = "<think>2 then 3.</think>The first primes are 2 and 3.";
    let reference = template
        .render(&RenderInput {
            messages: vec![
                Message::system(SYSTEM),
                Message::user(USER),
                Message::assistant(think_turn),
                Message::user("And the next two?"),
            ],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");
    let ours = ChatMl::qwen3().render(&[
        Turn::system(SYSTEM),
        Turn::user(USER),
        Turn::assistant(think_turn),
        Turn::user("And the next two?"),
    ]);
    println!("{}", diff_context("qwen3 think-strip", &ours, &reference));
    assert!(
        !ours.contains("<think>"),
        "history reasoning must be stripped"
    );
    assert_eq!(ours, reference, "Qwen3 think-strip render must byte-match");

    // The shared qwen2() renderer keeps re-rendering verbatim (pinned).
    let verbatim = ChatMl::qwen2().render(&[
        Turn::system(SYSTEM),
        Turn::user(USER),
        Turn::assistant(think_turn),
        Turn::user("And the next two?"),
    ]);
    assert_eq!(
        verbatim.replacen("<think>2 then 3.</think>", "", 1),
        reference,
        "qwen2()'s divergence stays exactly the think block"
    );

    // (c) Reasoning + tool call AFTER the last user query: the template
    // re-emits the think block normalized, then the structural tool_calls;
    // our pre-rendered turn must produce the same bytes.
    let calls = [mummu::chat::ToolCall {
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "Paris"}),
    }];
    let mut assistant_call = Message::assistant("<think>\nneed the live temp\n</think>\n\n");
    assistant_call.tool_calls =
        vec![serde_json::json!({"name": "get_weather", "arguments": {"city": "Paris"}})];
    let reference = template
        .render(&RenderInput {
            messages: vec![
                Message::system(SYSTEM),
                Message::user("What's the weather in Paris?"),
                assistant_call,
                Message::new("tool", "{\"temp_c\": 21}"),
            ],
            add_generation_prompt: true,
            ..RenderInput::default()
        })
        .expect("reference render succeeds");
    let call_turn = Turn::assistant_tool_calls(&calls);
    let ours = ChatMl::qwen3().render(&[
        Turn::system(SYSTEM),
        Turn::user("What's the weather in Paris?"),
        Turn::assistant(format!(
            "<think>\nneed the live temp\n</think>\n\n{}",
            call_turn.content
        )),
        Turn::tool_response("{\"temp_c\": 21}"),
    ]);
    println!("{}", diff_context("qwen3 think+call", &ours, &reference));
    assert!(
        ours.contains("<think>"),
        "reasoning after the last user query is kept"
    );
    assert_eq!(ours, reference, "Qwen3 think+call render must byte-match");
}

// ---- LFM2.5 -----------------------------------------------------------------

/// LFM2.5 plain legs (±system) — these also prove the standalone
/// `chat_template.jinja` import fallback end-to-end on a real checkpoint:
/// the dir ships NO `chat_template` JSON key, only the sibling file.
#[test]
#[ignore = "needs the local LFM2.5 checkpoint dir (MUMMU_LFM2_DIR) with chat_template.jinja"]
fn lfm2_plain_renders_byte_match_the_imported_template() {
    let dir = dir_from("MUMMU_LFM2_DIR").expect("set MUMMU_LFM2_DIR");
    let template = imported_template(&dir);

    let reference = template
        .render(&lfm_input(vec![Message::user(USER)], vec![]))
        .expect("reference render succeeds");
    let ours = ChatMl::lfm2().render(&[Turn::user(USER)]);
    println!("{}", diff_context("lfm plain", &ours, &reference));
    assert_eq!(ours, reference, "LFM no-system render must byte-match");

    let reference = template
        .render(&lfm_input(
            vec![Message::system(SYSTEM), Message::user(USER)],
            vec![],
        ))
        .expect("reference render succeeds");
    let ours = ChatMl::lfm2().render(&[Turn::system(SYSTEM), Turn::user(USER)]);
    println!("{}", diff_context("lfm system", &ours, &reference));
    assert_eq!(ours, reference, "LFM system render must byte-match");
}

/// LFM2.5 tools legs (±system, bare tool JSON): with no system turn BOTH
/// sides inject nothing — LFM has no default preamble, so unlike the Qwen
/// families this case is byte-equal, not a pinned divergence.
#[test]
#[ignore = "needs the local LFM2.5 checkpoint dir (MUMMU_LFM2_DIR) with chat_template.jinja"]
fn lfm2_tools_renders_byte_match_the_imported_template() {
    let dir = dir_from("MUMMU_LFM2_DIR").expect("set MUMMU_LFM2_DIR");
    let template = imported_template(&dir);

    let reference = template
        .render(&lfm_input(
            vec![Message::system(SYSTEM), Message::user(USER)],
            vec![lfm_tool_json()],
        ))
        .expect("reference render succeeds");
    let ours = ChatMl::lfm2()
        .render_with_tools(&[weather_spec()], &[Turn::system(SYSTEM), Turn::user(USER)]);
    println!("{}", diff_context("lfm tools+system", &ours, &reference));
    assert_eq!(ours, reference, "LFM tools+system render must byte-match");

    let reference = template
        .render(&lfm_input(vec![Message::user(USER)], vec![lfm_tool_json()]))
        .expect("reference render succeeds");
    let ours = ChatMl::lfm2().render_with_tools(&[weather_spec()], &[Turn::user(USER)]);
    println!("{}", diff_context("lfm tools no-system", &ours, &reference));
    assert_eq!(
        ours, reference,
        "LFM tools no-system render must byte-match"
    );
}

/// LFM2.5 history semantics: past assistant turns lose their `</think>`
/// reasoning on BOTH sides (keep_past_thinking=false), the LAST assistant
/// turn keeps it; pythonic call turns + real `tool` role turns round-trip.
#[test]
#[ignore = "needs the local LFM2.5 checkpoint dir (MUMMU_LFM2_DIR) with chat_template.jinja"]
fn lfm2_history_renders_byte_match_the_imported_template() {
    let dir = dir_from("MUMMU_LFM2_DIR").expect("set MUMMU_LFM2_DIR");
    let template = imported_template(&dir);

    let past = "<think>2, 3.</think>\n\nThe first two primes are 2 and 3.";
    let last = "<think>5, 7 next.</think>\n\n5 and 7.";
    let turns = [
        Turn::user(USER),
        Turn::assistant(past),
        Turn::user("And the next two?"),
        Turn::assistant(last),
        Turn::user("Thanks — one more?"),
    ];
    let messages = vec![
        Message::user(USER),
        Message::assistant(past),
        Message::user("And the next two?"),
        Message::assistant(last),
        Message::user("Thanks — one more?"),
    ];
    let reference = template
        .render(&lfm_input(messages, vec![]))
        .expect("reference render succeeds");
    let ours = ChatMl::lfm2().render(&turns);
    println!("{}", diff_context("lfm think-strip", &ours, &reference));
    assert_eq!(ours, reference, "LFM think-stripping must byte-match");

    // Pythonic call turn + a real `tool` role turn: the LFM template renders
    // assistant content verbatim, so the reference sees the emitted text.
    let calls = [mummu::chat::ToolCall {
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "Paris"}),
    }];
    let call_turn = Turn::assistant_tool_calls_lfm(&calls);
    let reference = template
        .render(&lfm_input(
            vec![
                Message::user("Weather in Paris?"),
                Message::assistant(call_turn.content.clone()),
                Message::new("tool", "{\"temp_c\": 21}"),
            ],
            vec![],
        ))
        .expect("reference render succeeds");
    let ours = ChatMl::lfm2().render(&[
        Turn::user("Weather in Paris?"),
        call_turn,
        Turn::tool_response("{\"temp_c\": 21}"),
    ]);
    println!("{}", diff_context("lfm pythonic+tool", &ours, &reference));
    assert_eq!(ours, reference, "LFM pythonic/tool turns must byte-match");
}
