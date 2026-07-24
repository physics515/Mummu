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
