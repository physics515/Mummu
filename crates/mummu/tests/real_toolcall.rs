//! Real-model tool-use proof: render a Hermes-style tools prompt, let
//! Qwen2.5-1.5B decode on the GPU, and parse the `<tool_call>` block it
//! actually emits. Ignored by default; run with
//!
//! ```text
//! MUMMU_QWEN2_DIR=path/to/qwen2.5-1.5b cargo test -p mummu --release --test real_toolcall -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::chat::{ChatMl, ToolSpec, Turn, parse_tool_calls};
use mummu::models::CausalLm;
use mummu::models::qwen2;
use tokenizers::Tokenizer;

fn qwen2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_QWEN2_DIR")?);
    dir.is_dir().then_some(dir)
}

#[tokio::test]
#[ignore = "needs multi-GB local weights (MUMMU_QWEN2_DIR) + the reference GPU"]
async fn qwen2_emits_a_parseable_tool_call() {
    let Some(dir) = qwen2_dir() else {
        panic!("set MUMMU_QWEN2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
    };

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
    let raw = ChatMl::qwen2().render_with_tools(
        &tools,
        &[Turn::user("What is the weather in Paris right now?")],
    );

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let prompt = tok
        .encode(raw.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();

    let device = mummu::backend::gpu_device();
    let loaded = qwen2::load_from_dir(&dir, &device).expect("weights load checked");
    let ids = loaded
        .greedy_generate(&prompt, 64, &device)
        .await
        .expect("greedy decode");
    let text = tok.decode(&ids, false).expect("decode");
    eprintln!("[real_toolcall] model emitted: {text:?}");

    let (calls, prose) = parse_tool_calls(&text).expect("emitted tool call parses");
    assert_eq!(
        calls.len(),
        1,
        "expected exactly one call, prose: {prose:?}"
    );
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        calls[0].arguments["city"].as_str(),
        Some("Paris"),
        "arguments: {:?}",
        calls[0].arguments
    );
    eprintln!(
        "[real_toolcall] parsed: {} with {}",
        calls[0].name, calls[0].arguments
    );
}
