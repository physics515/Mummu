//! REAL-CHECKPOINT proof for the general fallback renderer (`crate::template`,
//! feature `jinja-template`): rendering a checkpoint's OWN imported chat
//! template through Mummu's public API must byte-match the from-scratch family
//! renderer that the template byte gate already pins.
//!
//! `tests/template_gate.rs` proves the same equality by driving
//! `hf-chat-template` directly from the test; this proves the *module* wires
//! it correctly — special tokens into the context, tools in the Hermes wire
//! shape, assistant tool calls passed structurally rather than as pre-wrapped
//! markers, and the generation prompt on. A checkpoint outside the zoo has no
//! family renderer to compare against, so this is the only place the fallback
//! path's fidelity can be measured at all. Ignored by default; run with:
//!
//! ```text
//! MUMMU_QWEN3_DIR=path/to/qwen3-0.6b \
//!   cargo test -p mummu --features jinja-template --test imported_render -- --ignored --nocapture
//! ```
#![cfg(feature = "jinja-template")]

use std::path::PathBuf;

use mummu::chat::{ChatMl, ToolCall, ToolSpec, Turn};
use mummu::template::{ImportedTemplate, Renderer};

fn qwen3_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN3_DIR")?);
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

/// Report where two renders diverge, for a readable failure.
fn diff(label: &str, ours: &str, reference: &str) -> String {
    match ours
        .bytes()
        .zip(reference.bytes())
        .position(|(a, b)| a != b)
        .or_else(|| (ours.len() != reference.len()).then(|| ours.len().min(reference.len())))
    {
        None => format!("{label}: byte-identical ({} B)", ours.len()),
        Some(at) => {
            let lo = at.saturating_sub(60);
            format!(
                "{label}: DIVERGES at byte {at}\n  imported…{:?}\n  family  …{:?}",
                &ours[lo..(at + 60).min(ours.len())],
                &reference[lo..(at + 60).min(reference.len())],
            )
        }
    }
}

/// Plain conversation, tools, and a full function-calling history: the
/// imported template rendered through `ImportedTemplate` must equal
/// `ChatMl::qwen3()` byte for byte on all three.
#[test]
#[ignore = "needs the local Qwen3 checkpoint dir (MUMMU_QWEN3_DIR)"]
fn imported_qwen3_template_byte_matches_the_family_renderer() {
    let dir = qwen3_dir().expect("set MUMMU_QWEN3_DIR to a Qwen3 checkpoint dir");
    let imported = ImportedTemplate::from_dir(&dir).expect("the checkpoint's template compiles");
    let family = ChatMl::qwen3();

    let plain = [
        Turn::system("You are a helpful assistant."),
        Turn::user("List the first five prime numbers."),
    ];
    let ours = imported.render(&plain).expect("renders");
    let reference = family.render(&plain);
    println!("{}", diff("plain", &ours, &reference));
    assert_eq!(ours, reference, "plain render must byte-match");

    let tools = [weather_spec()];
    let turns = [
        Turn::system("You are a helpful assistant."),
        Turn::user("What's the weather in Paris?"),
    ];
    let ours = imported.render_with_tools(&tools, &turns).expect("renders");
    let reference = family.render_with_tools(&tools, &turns);
    println!("{}", diff("tools", &ours, &reference));
    assert_eq!(ours, reference, "tools render must byte-match");

    let calls = [ToolCall {
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "Paris"}),
    }];
    let history = [
        Turn::system("You are a helpful assistant."),
        Turn::user("What's the weather in Paris?"),
        Turn::assistant_tool_calls(&calls),
        Turn::tool_response("{\"temp_c\": 21}"),
    ];
    let ours = imported.render(&history).expect("renders");
    let reference = family.render(&history);
    println!("{}", diff("fc-history", &ours, &reference));
    assert_eq!(
        ours, reference,
        "FC history must byte-match — the structural tool_calls path"
    );
}

/// LFM2.5 is the other half of the wiring proof: its checkpoint ships NO
/// `chat_template` JSON key (only a standalone `chat_template.jinja`), and its
/// template opens with `{{- bos_token -}}` — so this leg exercises both the
/// file fallback in `from_dir` AND the special tokens reaching the render
/// context from the parsed config. Get either wrong and the prompt loses its
/// BOS silently.
#[test]
#[ignore = "needs the local LFM2.5 checkpoint dir (MUMMU_LFM2_DIR) with chat_template.jinja"]
fn imported_lfm2_template_byte_matches_the_family_renderer() {
    let Some(dir) = std::env::var_os("MUMMU_LFM2_DIR")
        .map(PathBuf::from)
        .filter(|d| d.is_dir())
    else {
        panic!("set MUMMU_LFM2_DIR to an LFM2.5 checkpoint dir");
    };
    let imported = ImportedTemplate::from_dir(&dir).expect("standalone chat_template.jinja loads");
    let family = ChatMl::lfm2();

    let turns = [
        Turn::system("You are a helpful assistant."),
        Turn::user("List the first five prime numbers."),
    ];
    let ours = imported.render(&turns).expect("renders");
    let reference = family.render(&turns);
    println!("{}", diff("lfm plain", &ours, &reference));
    assert!(
        ours.starts_with("<|startoftext|>"),
        "the config's bos_token reached the template: {:?}",
        &ours[..ours.len().min(40)]
    );
    assert_eq!(ours, reference, "LFM plain render must byte-match");

    // LFM's template wants BARE tool JSON, not the transformers wrapper —
    // the shape `render_with_tools_json` exists for.
    let bare = serde_json::json!({
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    });
    let ours = imported
        .render_with_tools_json(&[bare], &turns)
        .expect("renders");
    let reference = family.render_with_tools(&[weather_spec()], &turns);
    println!("{}", diff("lfm tools", &ours, &reference));
    assert_eq!(ours, reference, "LFM tools render must byte-match");
}

/// The selection rule as a value: `Renderer` with no family renderer falls
/// back to the checkpoint's own template and renders the same bytes.
#[test]
#[ignore = "needs the local Qwen3 checkpoint dir (MUMMU_QWEN3_DIR)"]
fn renderer_falls_back_to_the_checkpoint_template_when_no_family_renderer() {
    let dir = qwen3_dir().expect("set MUMMU_QWEN3_DIR to a Qwen3 checkpoint dir");
    let fallback = Renderer::for_checkpoint(None, &dir).expect("falls back to the template");
    assert!(matches!(fallback, Renderer::Imported(_)));

    let turns = [Turn::user("Say hi.")];
    let ours = fallback.render(&turns).expect("renders");
    let reference = ChatMl::qwen3().render(&turns);
    println!("{}", diff("fallback", &ours, &reference));
    assert_eq!(ours, reference);
}
