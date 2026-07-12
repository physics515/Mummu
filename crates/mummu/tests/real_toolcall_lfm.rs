//! Real-model tool-use proof for the LFM convention: render a `List of
//! tools:` prompt, let LFM2.5-1.2B decode on the GPU, and parse the Pythonic
//! `<|tool_call_start|>` block it actually emits. Ignored by default; run with
//!
//! ```text
//! MUMMU_LFM2_DIR=path/to/lfm2.5-1.2b cargo test -p mummu --release --test real_toolcall_lfm -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mummu::backend::Gpu;
use mummu::chat::{ChatMl, ToolSpec, Turn, parse_tool_calls_lfm};
use mummu::models::CausalLm;
use mummu::models::lfm2;
use tokenizers::Tokenizer;

fn lfm2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_LFM2_DIR")?);
    dir.is_dir().then_some(dir)
}

#[test]
#[ignore = "needs multi-GB local weights (MUMMU_LFM2_DIR) + the reference GPU"]
fn lfm2_emits_a_parseable_pythonic_tool_call() {
    let Some(dir) = lfm2_dir() else {
        panic!("set MUMMU_LFM2_DIR to a dir with config.json/tokenizer.json/model.safetensors");
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
    let raw = ChatMl::lfm2().render_with_tools(
        &tools,
        &[Turn::user("What is the weather in Paris right now?")],
    );

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let prompt = tok
        .encode(raw.as_str(), false)
        .expect("encodes")
        .get_ids()
        .to_vec();

    let device = burn::tensor::Device::<Gpu>::default();
    let loaded = lfm2::load_from_dir::<Gpu>(&dir, &device).expect("weights load checked");
    let ids = loaded
        .greedy_generate(&prompt, 128, &device)
        .expect("greedy decode");
    // Special tokens stay in: the <|tool_call_start|> markers ARE the format.
    let text = tok.decode(&ids, false).expect("decode");
    eprintln!("[real_toolcall_lfm] model emitted: {text:?}");

    let (calls, prose) = parse_tool_calls_lfm(&text).expect("emitted tool call parses");
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
        "[real_toolcall_lfm] parsed: {} with {}",
        calls[0].name, calls[0].arguments
    );
}
