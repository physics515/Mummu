//! Shared llama.cpp reference harness for parity tests.
//!
//! Spawns a local `llama-server` (llama.cpp) on a GGUF checkpoint and queries
//! its RAW `/completion` endpoint — never the chat-template stack (see the P7
//! roadmap notes: prompts are rendered by our own byte-verified templates and
//! sent as **token-id arrays**, so neither side's tokenizer nor any server-side
//! BOS insertion can skew the comparison).
//!
//! The server binary is user-supplied (`MUMMU_LLAMA_SERVER`); Ollama installs
//! ship one at `<ollama>/lib/ollama/llama-server.exe`. The reference runs on
//! CPU (`-ngl 0`): a *different* compute stack from the wgpu path under test,
//! which is exactly what a parity reference should be.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// How long we give the server to load the model and report healthy.
const HEALTH_TRIES: u32 = 240;
const HEALTH_INTERVAL: Duration = Duration::from_millis(500);

/// Largest response body we will read from the server (a completion with
/// top-logprobs is a few KiB; 16 MiB is a runaway guard, not a target).
const MAX_RESPONSE_BYTES: u64 = 16 << 20;

/// Path to the llama.cpp server binary, from `MUMMU_LLAMA_SERVER`.
pub fn server_exe() -> Option<PathBuf> {
    let exe = PathBuf::from(std::env::var_os("MUMMU_LLAMA_SERVER")?);
    exe.is_file().then_some(exe)
}

/// A running llama-server bound to one GGUF file; killed on drop so a failing
/// test never leaks a 2 GB process.
pub struct LlamaServer {
    child: Child,
    port: u16,
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl LlamaServer {
    /// Spawn on `port` serving `gguf`, CPU-only, and wait until `/health`
    /// answers ok. Errors are strings: these run inside `#[ignore]`d tests
    /// where the caller panics with context.
    pub fn start(exe: &std::path::Path, gguf: &std::path::Path, port: u16) -> Result<Self, String> {
        assert!(port >= 1024, "pick an unprivileged port");
        assert!(gguf.is_file(), "reference gguf must exist: {gguf:?}");
        let child = Command::new(exe)
            .args(["-m"])
            .arg(gguf)
            .args(["--port", &port.to_string(), "-ngl", "0", "--no-webui"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn {exe:?}: {e}"))?;
        let mut server = Self { child, port };
        for _ in 0..HEALTH_TRIES {
            if let Some(status) = server.child.try_wait().map_err(|e| e.to_string())? {
                return Err(format!("llama-server exited during startup: {status}"));
            }
            if server.healthy() {
                return Ok(server);
            }
            std::thread::sleep(HEALTH_INTERVAL);
        }
        Err(format!(
            "llama-server not healthy after {HEALTH_TRIES} x {HEALTH_INTERVAL:?}"
        ))
    }

    fn healthy(&self) -> bool {
        let url = format!("http://127.0.0.1:{}/health", self.port);
        matches!(ureq::get(&url).call(), Ok(resp) if resp.status() == 200)
    }

    /// Raw greedy completion from exact `prompt_ids`: temperature 0,
    /// `n_predict` tokens, top-`n_probs` pre-sampling logprobs per position.
    /// Asserts the server evaluated exactly the ids we sent (no BOS injection).
    pub fn greedy_completion(
        &self,
        prompt_ids: &[u32],
        n_predict: usize,
        n_probs: usize,
    ) -> Result<Completion, String> {
        assert!(!prompt_ids.is_empty(), "empty prompt");
        assert!((1..=256).contains(&n_predict), "n_predict out of range");
        let body = serde_json::json!({
            "prompt": prompt_ids,
            "temperature": 0.0,
            "n_predict": n_predict,
            "n_probs": n_probs,
            "cache_prompt": false,
        });
        let url = format!("http://127.0.0.1:{}/completion", self.port);
        let payload = body.to_string();
        let mut resp = ureq::post(&url)
            .header("Content-Type", "application/json")
            .send(payload.as_str())
            .map_err(|e| format!("POST /completion: {e}"))?;
        let mut text = String::new();
        resp.body_mut()
            .as_reader()
            .take(MAX_RESPONSE_BYTES)
            .read_to_string(&mut text)
            .map_err(|e| format!("read /completion body: {e}"))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("parse /completion JSON: {e}"))?;

        let evaluated = v["tokens_evaluated"].as_u64().unwrap_or(0) as usize;
        if evaluated != prompt_ids.len() {
            return Err(format!(
                "server evaluated {evaluated} tokens for a {}-id prompt — \
                 token-id passthrough is broken (BOS injection?)",
                prompt_ids.len()
            ));
        }
        let content = v["content"]
            .as_str()
            .ok_or_else(|| format!("no content in: {text}"))?
            .to_string();
        let mut steps = Vec::new();
        if let Some(positions) = v["completion_probabilities"].as_array() {
            for pos in positions {
                let top = pos["top_logprobs"]
                    .as_array()
                    .ok_or_else(|| "completion_probabilities without top_logprobs".to_string())?
                    .iter()
                    .map(|e| {
                        let id = e["id"].as_u64().ok_or("top_logprobs entry without id")? as u32;
                        let logprob = e["logprob"].as_f64().ok_or("entry without logprob")?;
                        Ok((id, logprob))
                    })
                    .collect::<Result<Vec<_>, &str>>()
                    .map_err(str::to_string)?;
                steps.push(top);
            }
        }
        Ok(Completion { content, steps })
    }
}

/// One raw completion: the greedy text plus, per generated position, the
/// top-k `(token id, natural-log probability)` pairs, best first.
pub struct Completion {
    pub content: String,
    pub steps: Vec<Vec<(u32, f64)>>,
}

/// Natural-log softmax probabilities of `logits` at `ids`, computed in f64
/// (the reference reports logprobs, not raw logits — same transform here).
pub fn logprobs_at(logits: &[f32], ids: &[u32]) -> Vec<f64> {
    assert!(!logits.is_empty(), "empty logits");
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(max.is_finite(), "logits must be finite");
    let log_sum: f64 = logits
        .iter()
        .map(|&l| f64::from(l - max).exp())
        .sum::<f64>()
        .ln();
    ids.iter()
        .map(|&id| f64::from(logits[id as usize] - max) - log_sum)
        .collect()
}
