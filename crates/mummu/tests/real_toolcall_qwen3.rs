//! Real-model tool-use proof for the **Qwen3 dense arch** (the architecture the
//! Qwen3.5-4B/9B function-calling tier rides): import the checkpoint's
//! `tokenizer_config.json`, let its embedded template pick the tool-call
//! convention (Hermes), render a tools prompt with that style, decode on the
//! GPU, and parse the `<tool_call>` the model emits. Ties the tokenizer-config
//! import to the end-to-end FC path. Ignored by default; run with:
//!
//! ```text
//! MUMMU_QWEN3_DIR=path/to/qwen3-0.6b \
//!   cargo test -p mummu --test real_toolcall_qwen3 -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::backend::Gpu;
use mummu::chat::{ChatMl, ToolSpec, Turn, parse_tool_calls};
use mummu::models::CausalLm;
use mummu::models::qwen3;
use mummu::tok_config::{TokenizerConfig, ToolCallConvention};
use tokenizers::Tokenizer;

fn qwen3_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN3_DIR")?);
    dir.join("model.safetensors").is_file().then_some(dir)
}

#[test]
#[ignore = "needs the local Qwen3 checkpoint dir (MUMMU_QWEN3_DIR) + the reference GPU"]
fn qwen3_emits_a_parseable_hermes_tool_call() {
    let Some(dir) = qwen3_dir() else {
        panic!("set MUMMU_QWEN3_DIR to a Qwen3 checkpoint dir");
    };

    // The imported template drives the render-style choice: Qwen3 is Hermes.
    let cfg = TokenizerConfig::from_dir(&dir).expect("tokenizer_config.json parses");
    assert_eq!(
        cfg.tool_call_convention(),
        Some(ToolCallConvention::Hermes),
        "Qwen3's template selects the Hermes tool-call style"
    );

    let tools = [ToolSpec {
        name: "get_weather".into(),
        description: "Get the current weather for a city.".into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        }),
    }];
    // Hermes renderer (Qwen2.5/Qwen3 share it), chosen per the detection above.
    let raw = ChatMl::qwen2().render_with_tools(
        &tools,
        &[Turn::user(
            "What is the weather in Paris right now? Use the tool.",
        )],
    );

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let prompt = tok
        .encode(raw.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();

    let device = burn::tensor::Device::<Gpu>::default();
    let loaded = qwen3::load_from_dir::<Gpu>(&dir, &device).expect("weights load checked");
    // Generous budget: Qwen3 may emit a <think>…</think> block before the call.
    let ids = loaded
        .greedy_generate(&prompt, 512, &device)
        .expect("greedy decode");
    let text = tok.decode(&ids, false).expect("decode");
    eprintln!("[real_toolcall_qwen3] model emitted: {text:?}");

    let (calls, prose) = parse_tool_calls(&text).expect("emitted tool call parses");
    assert!(!calls.is_empty(), "expected a tool call, prose: {prose:?}");
    assert_eq!(calls[0].name, "get_weather", "calls: {calls:?}");
    assert_eq!(
        calls[0].arguments["city"].as_str(),
        Some("Paris"),
        "arguments: {:?}",
        calls[0].arguments
    );
    eprintln!(
        "[real_toolcall_qwen3] parsed: {} with {}",
        calls[0].name, calls[0].arguments
    );
}
