//! The template byte gate: a checkpoint's OWN imported `chat_template`,
//! rendered exactly as Python `transformers.apply_chat_template` renders it
//! (via `tools/template-probe`, hf-chat-template = minijinja + a transformers
//! compatibility layer), must reproduce our hardcoded `chat` renderers
//! byte-for-byte on the parity-committed prompt shapes — and where the two
//! deliberately diverge (family default preambles, Qwen3 think-stripping),
//! the divergence is pinned down to the exact expected delta so any OTHER
//! drift still fails loudly.
//!
//! The template travels through the real import path: `TokenizerConfig::
//! from_dir` (JSON key, or the standalone `chat_template.jinja` fallback for
//! LFM2.5). Ignored by default; run with
//!
//! ```text
//! cargo build --release --manifest-path tools/template-probe/Cargo.toml
//! MUMMU_TEMPLATE_PROBE=<target>/release/template-probe.exe \
//! MUMMU_QWEN2_DIR=... MUMMU_QWEN3_DIR=... MUMMU_LFM2_DIR=... \
//! cargo test -p mummu --test template_gate -- --ignored --nocapture
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use mummu::chat::{ChatMl, ToolCall, ToolSpec, Turn, py_json};
use mummu::tok_config::TokenizerConfig;

fn env_dir(var: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os(var)?);
    dir.is_dir().then_some(dir)
}

fn probe_exe() -> PathBuf {
    let Some(path) = std::env::var_os("MUMMU_TEMPLATE_PROBE") else {
        panic!(
            "set MUMMU_TEMPLATE_PROBE to the built tools/template-probe binary \
             (cargo build --release --manifest-path tools/template-probe/Cargo.toml)"
        );
    };
    let path = PathBuf::from(path);
    assert!(
        path.is_file(),
        "MUMMU_TEMPLATE_PROBE is not a file: {path:?}"
    );
    path
}

/// Write the checkpoint's imported template to a temp file and render
/// `input_json` through the reference engine. The template string comes out
/// of the SAME importer the loaders use, so the gate covers import + render.
fn reference_render(dir: &std::path::Path, input_json: &str) -> String {
    assert!(!input_json.is_empty(), "reference_render: empty input");
    let cfg = TokenizerConfig::from_dir(dir).expect("tokenizer_config imports");
    let template = cfg
        .chat_template
        .expect("checkpoint has a chat template (JSON key or chat_template.jinja)");
    // Unique per call — tests run as parallel threads in ONE process, so a
    // pid-only name would let them delete each other's file mid-render.
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!(
        "mummu-template-gate-{}-{seq}.jinja",
        std::process::id()
    ));
    std::fs::write(&tmp, &template).expect("temp template writes");

    let mut child = Command::new(probe_exe())
        .arg(&tmp)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("template-probe spawns");
    child
        .stdin
        .take()
        .expect("probe stdin")
        .write_all(input_json.as_bytes())
        .expect("probe stdin writes");
    let out = child.wait_with_output().expect("probe runs");
    let _ = std::fs::remove_file(&tmp);
    assert!(
        out.status.success(),
        "template-probe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("probe output is utf-8")
}

/// JSON string literal (escaping) for hand-assembled render-input text.
fn js(s: &str) -> String {
    py_json(&s)
}

fn text_msg(role: &str, content: &str) -> String {
    format!("{{\"role\": {}, \"content\": {}}}", js(role), js(content))
}

/// A render-input built as TEXT so key order inside tool payloads survives
/// into the probe's order-preserving parse (a serde_json::Value round-trip
/// on our side would re-sort object keys and break byte-stability).
fn input_json(messages: &[String], tools: &[String], extra: &str) -> String {
    assert!(!messages.is_empty(), "input_json: no messages");
    let mut out = format!("{{\"messages\": [{}]", messages.join(", "));
    if !tools.is_empty() {
        out.push_str(&format!(", \"tools\": [{}]", tools.join(", ")));
    }
    out.push_str(", \"add_generation_prompt\": true");
    if !extra.is_empty() {
        out.push_str(", ");
        out.push_str(extra);
    }
    out.push('}');
    out
}

fn weather_tool() -> ToolSpec {
    ToolSpec {
        name: "get_weather".into(),
        description: "Get the current weather for a city.".into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string", "description": "City name" } },
            "required": ["city"]
        }),
    }
}

/// The Hermes wire shape (mirrors the private struct behind
/// `render_with_tools`): what transformers' `get_json_schema` hands the
/// template as one entry of `tools`.
#[derive(serde::Serialize)]
struct Wire<'a> {
    r#type: &'static str,
    function: &'a ToolSpec,
}

fn hermes_wire(tool: &ToolSpec) -> String {
    py_json(&Wire {
        r#type: "function",
        function: tool,
    })
}

const SYSTEM: &str = "You are a helpful assistant.";
const USER: &str = "List the first five prime numbers.";

// ---- Qwen2.5 ---------------------------------------------------------------

#[test]
#[ignore = "needs MUMMU_TEMPLATE_PROBE + a cached Qwen2.5 checkpoint (MUMMU_QWEN2_DIR)"]
fn qwen2_plain_render_matches_the_imported_template() {
    let Some(dir) = env_dir("MUMMU_QWEN2_DIR") else {
        panic!("set MUMMU_QWEN2_DIR to a dir with tokenizer_config.json");
    };
    let ours = ChatMl::qwen2().render(&[Turn::system(SYSTEM), Turn::user(USER)]);
    let reference = reference_render(
        &dir,
        &input_json(
            &[text_msg("system", SYSTEM), text_msg("user", USER)],
            &[],
            "",
        ),
    );
    assert_eq!(ours, reference, "plain ChatML render diverged");
}

#[test]
#[ignore = "needs MUMMU_TEMPLATE_PROBE + a cached Qwen2.5 checkpoint (MUMMU_QWEN2_DIR)"]
fn qwen2_tools_render_matches_the_imported_template() {
    let Some(dir) = env_dir("MUMMU_QWEN2_DIR") else {
        panic!("set MUMMU_QWEN2_DIR to a dir with tokenizer_config.json");
    };
    let tool = weather_tool();
    let ours = ChatMl::qwen2().render_with_tools(
        std::slice::from_ref(&tool),
        &[Turn::system(SYSTEM), Turn::user(USER)],
    );
    let reference = reference_render(
        &dir,
        &input_json(
            &[text_msg("system", SYSTEM), text_msg("user", USER)],
            &[hermes_wire(&tool)],
            "",
        ),
    );
    assert_eq!(ours, reference, "Hermes tools render diverged");
}

#[test]
#[ignore = "needs MUMMU_TEMPLATE_PROBE + a cached Qwen2.5 checkpoint (MUMMU_QWEN2_DIR)"]
fn qwen2_tool_call_history_matches_the_imported_template() {
    let Some(dir) = env_dir("MUMMU_QWEN2_DIR") else {
        panic!("set MUMMU_QWEN2_DIR to a dir with tokenizer_config.json");
    };
    let tool = weather_tool();
    let calls = [ToolCall {
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "Paris"}),
    }];
    let ours = ChatMl::qwen2().render_with_tools(
        std::slice::from_ref(&tool),
        &[
            Turn::system(SYSTEM),
            Turn::user("Weather in Paris?"),
            Turn::assistant_tool_calls(&calls),
            Turn::tool_response("{\"temp_c\": 21}"),
            Turn::tool_response("{\"temp_c\": 24}"),
        ],
    );
    // The reference sees the call structurally (message.tool_calls), exactly
    // as transformers is fed; ours re-renders the emitted <tool_call> text.
    let call_msg = format!(
        "{{\"role\": \"assistant\", \"tool_calls\": [{}]}}",
        py_json(&calls[0])
    );
    let reference = reference_render(
        &dir,
        &input_json(
            &[
                text_msg("system", SYSTEM),
                text_msg("user", "Weather in Paris?"),
                call_msg,
                text_msg("tool", "{\"temp_c\": 21}"),
                text_msg("tool", "{\"temp_c\": 24}"),
            ],
            &[hermes_wire(&tool)],
            "",
        ),
    );
    assert_eq!(ours, reference, "tool-call history render diverged");
}

/// Documented divergence: without a system turn, Qwen2.5's template injects
/// its own branding preamble where we inject a neutral one (one renderer
/// serves Qwen2 AND Qwen3, and Qwen3's template injects nothing — no single
/// default can match both). The delta must be EXACTLY that preamble swap.
#[test]
#[ignore = "needs MUMMU_TEMPLATE_PROBE + a cached Qwen2.5 checkpoint (MUMMU_QWEN2_DIR)"]
fn qwen2_no_system_defaults_diverge_only_by_the_documented_preamble() {
    let Some(dir) = env_dir("MUMMU_QWEN2_DIR") else {
        panic!("set MUMMU_QWEN2_DIR to a dir with tokenizer_config.json");
    };
    const QWEN_PREAMBLE: &str =
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    let tool = weather_tool();

    // With tools: both sides synthesize a system turn; preambles differ.
    let ours = ChatMl::qwen2().render_with_tools(std::slice::from_ref(&tool), &[Turn::user(USER)]);
    let reference = reference_render(
        &dir,
        &input_json(&[text_msg("user", USER)], &[hermes_wire(&tool)], ""),
    );
    let ours_with_qwen_preamble = ours.replacen(SYSTEM, QWEN_PREAMBLE, 1);
    assert_ne!(ours_with_qwen_preamble, ours, "preamble must be present");
    assert_eq!(
        ours_with_qwen_preamble, reference,
        "no-system tools render must diverge ONLY by the default preamble"
    );

    // Without tools: the template injects a whole default system turn; our
    // render() injects nothing (explicit turns are the caller's contract).
    let ours = ChatMl::qwen2().render(&[Turn::user(USER)]);
    let reference = reference_render(&dir, &input_json(&[text_msg("user", USER)], &[], ""));
    assert_eq!(
        format!("<|im_start|>system\n{QWEN_PREAMBLE}<|im_end|>\n{ours}"),
        reference,
        "no-system plain render must diverge ONLY by the injected default turn"
    );
}

// ---- Qwen3 ------------------------------------------------------------------

#[test]
#[ignore = "needs MUMMU_TEMPLATE_PROBE + a cached Qwen3 checkpoint (MUMMU_QWEN3_DIR)"]
fn qwen3_plain_and_tools_renders_match_the_imported_template() {
    let Some(dir) = env_dir("MUMMU_QWEN3_DIR") else {
        panic!("set MUMMU_QWEN3_DIR to a dir with tokenizer_config.json");
    };
    let ours = ChatMl::qwen2().render(&[Turn::system(SYSTEM), Turn::user(USER)]);
    let reference = reference_render(
        &dir,
        &input_json(
            &[text_msg("system", SYSTEM), text_msg("user", USER)],
            &[],
            "",
        ),
    );
    assert_eq!(ours, reference, "Qwen3 plain render diverged");

    let tool = weather_tool();
    let ours = ChatMl::qwen2().render_with_tools(
        std::slice::from_ref(&tool),
        &[Turn::system(SYSTEM), Turn::user(USER)],
    );
    let reference = reference_render(
        &dir,
        &input_json(
            &[text_msg("system", SYSTEM), text_msg("user", USER)],
            &[hermes_wire(&tool)],
            "",
        ),
    );
    assert_eq!(ours, reference, "Qwen3 tools render diverged");
}

/// Documented divergences vs Qwen3's template: (a) with tools and no system
/// turn it injects NO preamble (ours injects the neutral one); (b) it strips
/// `<think>` reasoning from assistant turns at or before the last user query
/// (ours passes history through verbatim — a `ChatMl::qwen3()` with
/// think-stripping is a ROADMAP item). Pin both deltas exactly.
#[test]
#[ignore = "needs MUMMU_TEMPLATE_PROBE + a cached Qwen3 checkpoint (MUMMU_QWEN3_DIR)"]
fn qwen3_divergences_are_exactly_the_documented_ones() {
    let Some(dir) = env_dir("MUMMU_QWEN3_DIR") else {
        panic!("set MUMMU_QWEN3_DIR to a dir with tokenizer_config.json");
    };
    let tool = weather_tool();
    let ours = ChatMl::qwen2().render_with_tools(std::slice::from_ref(&tool), &[Turn::user(USER)]);
    let reference = reference_render(
        &dir,
        &input_json(&[text_msg("user", USER)], &[hermes_wire(&tool)], ""),
    );
    assert_eq!(
        ours.replacen(&format!("{SYSTEM}\n\n"), "", 1),
        reference,
        "Qwen3 no-system tools render must diverge ONLY by our neutral preamble"
    );

    let think_turn = "<think>2 then 3.</think>The first primes are 2 and 3.";
    let ours = ChatMl::qwen2().render(&[
        Turn::system(SYSTEM),
        Turn::user(USER),
        Turn::assistant(think_turn),
        Turn::user("And the next two?"),
    ]);
    let reference = reference_render(
        &dir,
        &input_json(
            &[
                text_msg("system", SYSTEM),
                text_msg("user", USER),
                text_msg("assistant", think_turn),
                text_msg("user", "And the next two?"),
            ],
            &[],
            "",
        ),
    );
    assert!(ours.contains("<think>"), "ours re-renders history verbatim");
    assert!(
        !reference.contains("<think>"),
        "Qwen3's template strips history reasoning"
    );
    assert_eq!(
        ours.replacen("<think>2 then 3.</think>", "", 1),
        reference,
        "the think block must be the ONLY delta"
    );
}

// ---- LFM2.5 -----------------------------------------------------------------

const LFM_BOS: &str = "\"bos_token\": \"<|startoftext|>\"";

#[test]
#[ignore = "needs MUMMU_TEMPLATE_PROBE + a cached LFM2.5 checkpoint (MUMMU_LFM2_DIR)"]
fn lfm2_plain_renders_match_the_imported_template() {
    let Some(dir) = env_dir("MUMMU_LFM2_DIR") else {
        panic!("set MUMMU_LFM2_DIR to a dir with chat_template.jinja");
    };
    let ours = ChatMl::lfm2().render(&[Turn::user(USER)]);
    let reference = reference_render(&dir, &input_json(&[text_msg("user", USER)], &[], LFM_BOS));
    assert_eq!(ours, reference, "LFM no-system render diverged");

    let ours = ChatMl::lfm2().render(&[Turn::system(SYSTEM), Turn::user(USER)]);
    let reference = reference_render(
        &dir,
        &input_json(
            &[text_msg("system", SYSTEM), text_msg("user", USER)],
            &[],
            LFM_BOS,
        ),
    );
    assert_eq!(ours, reference, "LFM system render diverged");
}

#[test]
#[ignore = "needs MUMMU_TEMPLATE_PROBE + a cached LFM2.5 checkpoint (MUMMU_LFM2_DIR)"]
fn lfm2_tools_renders_match_the_imported_template() {
    let Some(dir) = env_dir("MUMMU_LFM2_DIR") else {
        panic!("set MUMMU_LFM2_DIR to a dir with chat_template.jinja");
    };
    let tool = weather_tool();
    // LFM tools are BARE tool JSON (no Hermes wrapper) — the template runs
    // `tool | tojson` on whatever is passed; the model card shows bare.
    let bare = py_json(&tool);

    // With an explicit system turn…
    let ours = ChatMl::lfm2().render_with_tools(
        std::slice::from_ref(&tool),
        &[Turn::system(SYSTEM), Turn::user(USER)],
    );
    let reference = reference_render(
        &dir,
        &input_json(
            &[text_msg("system", SYSTEM), text_msg("user", USER)],
            std::slice::from_ref(&bare),
            LFM_BOS,
        ),
    );
    assert_eq!(ours, reference, "LFM tools+system render diverged");

    // …and without one BOTH sides inject nothing (byte-equal, no delta).
    let ours = ChatMl::lfm2().render_with_tools(std::slice::from_ref(&tool), &[Turn::user(USER)]);
    let reference = reference_render(
        &dir,
        &input_json(&[text_msg("user", USER)], &[bare], LFM_BOS),
    );
    assert_eq!(ours, reference, "LFM tools no-system render diverged");
}

#[test]
#[ignore = "needs MUMMU_TEMPLATE_PROBE + a cached LFM2.5 checkpoint (MUMMU_LFM2_DIR)"]
fn lfm2_history_think_stripping_and_tool_turns_match_the_imported_template() {
    let Some(dir) = env_dir("MUMMU_LFM2_DIR") else {
        panic!("set MUMMU_LFM2_DIR to a dir with chat_template.jinja");
    };
    // Past assistant turns lose their reasoning on both sides; the LAST
    // assistant turn keeps it (keep_past_thinking=false semantics).
    let past = "<think>2, 3.</think>\n\nThe first two primes are 2 and 3.";
    let last = "<think>5, 7 next.</think>\n\n5 and 7.";
    let ours = ChatMl::lfm2().render(&[
        Turn::user(USER),
        Turn::assistant(past),
        Turn::user("And the next two?"),
        Turn::assistant(last),
        Turn::user("Thanks — one more?"),
    ]);
    let reference = reference_render(
        &dir,
        &input_json(
            &[
                text_msg("user", USER),
                text_msg("assistant", past),
                text_msg("user", "And the next two?"),
                text_msg("assistant", last),
                text_msg("user", "Thanks — one more?"),
            ],
            &[],
            LFM_BOS,
        ),
    );
    assert_eq!(ours, reference, "LFM think-stripping semantics diverged");

    // A Pythonic call turn + real `tool` role turns.
    let calls = [ToolCall {
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "Paris"}),
    }];
    let call_turn = Turn::assistant_tool_calls_lfm(&calls);
    let ours = ChatMl::lfm2().render(&[
        Turn::user("Weather in Paris?"),
        call_turn.clone(),
        Turn::tool_response("{\"temp_c\": 21}"),
    ]);
    let reference = reference_render(
        &dir,
        &input_json(
            &[
                text_msg("user", "Weather in Paris?"),
                text_msg("assistant", &call_turn.content),
                text_msg("tool", "{\"temp_c\": 21}"),
            ],
            &[],
            LFM_BOS,
        ),
    );
    assert_eq!(ours, reference, "LFM pythonic call/tool turns diverged");
}
