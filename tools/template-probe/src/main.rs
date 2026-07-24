//! Render a HuggingFace `chat_template` the way Python
//! `transformers.apply_chat_template` does, for mummu's template byte gate
//! (`crates/mummu/tests/template_gate.rs`).
//!
//! Usage: `template-probe <template-file> < render-input.json`
//!
//! - `argv[1]`: path to a file holding the raw Jinja template text.
//! - stdin: one `RenderInput` JSON object — `messages`, optional `tools`,
//!   `add_generation_prompt`, plus any extra context keys a template reads
//!   (e.g. `bos_token`). Key order in the JSON text is preserved into
//!   `| tojson` (hf-chat-template builds serde_json with `preserve_order`).
//! - stdout: the rendered prompt, raw bytes, no added trailing newline.
//! - Any failure: message on stderr, exit code 1.

use std::io::{Read, Write};

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        die("usage: template-probe <template-file> < render-input.json");
    };
    let template = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => die(&format!("read {path}: {e}")),
    };
    if template.trim().is_empty() {
        die(&format!("{path}: template file is empty"));
    }

    let mut input_json = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut input_json) {
        die(&format!("read stdin: {e}"));
    }
    let input: hf_chat_template::RenderInput = match serde_json::from_str(&input_json) {
        Ok(i) => i,
        Err(e) => die(&format!("render-input json: {e}")),
    };
    if input.messages.is_empty() {
        die("render-input has no messages");
    }

    let tmpl = match hf_chat_template::ChatTemplate::from_str(&template) {
        Ok(t) => t,
        Err(e) => die(&format!("template compile: {e}")),
    };
    match tmpl.render(&input) {
        Ok(out) => {
            if std::io::stdout().write_all(out.as_bytes()).is_err() {
                die("write stdout");
            }
        }
        Err(e) => die(&format!("render: {e}")),
    }
}

fn die(msg: &str) -> ! {
    eprintln!("template-probe: {msg}");
    std::process::exit(1)
}
