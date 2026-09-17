//! Shared llama.cpp reference harness for parity tests.
//!
//! Spawns a local `llama-server` (llama.cpp) on a GGUF checkpoint — or attaches
//! to one already running — and queries its RAW `/completion` endpoint, never
//! the chat-template stack (see the P7 roadmap notes: prompts are rendered by
//! our own byte-verified templates and sent as **token-id arrays**, so neither
//! side's tokenizer nor any server-side BOS insertion can skew the comparison).
//!
//! The server binary is user-supplied (`MUMMU_LLAMA_SERVER`); Ollama installs
//! ship one at `<ollama>/lib/ollama/llama-server.exe`. The reference runs on
//! CPU (`-ngl 0`): a *different* compute stack from the wgpu path under test,
//! which is exactly what a parity reference should be.
//!
//! Every parity binary includes this module but uses a different slice of it
//! (spawn vs attach, parsed vs raw responses), hence the per-item
//! `allow(dead_code)`s.

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
#[allow(dead_code)] // the attach-only recorder never spawns
pub fn server_exe() -> Option<PathBuf> {
    let exe = PathBuf::from(std::env::var_os("MUMMU_LLAMA_SERVER")?);
    exe.is_file().then_some(exe)
}

/// A llama-server to query. When we spawned it, it is killed on drop so a
/// failing test never leaks a 2 GB process; when we merely attached, it
/// belongs to someone else (a container too big to start per test) and is
/// left running.
pub struct LlamaServer {
    child: Option<Child>,
    /// `http://host:port`, no trailing slash.
    base_url: String,
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl LlamaServer {
    /// Spawn on `port` serving `gguf`, CPU-only, and wait until `/health`
    /// answers ok. Errors are strings: these run inside `#[ignore]`d tests
    /// where the caller panics with context.
    #[allow(dead_code)] // the attach-only recorder never spawns
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
        let mut child = child;
        let server_url = format!("http://127.0.0.1:{port}");
        for _ in 0..HEALTH_TRIES {
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                return Err(format!("llama-server exited during startup: {status}"));
            }
            if healthy(&server_url) {
                return Ok(Self {
                    child: Some(child),
                    base_url: server_url,
                });
            }
            std::thread::sleep(HEALTH_INTERVAL);
        }
        let _ = child.kill();
        let _ = child.wait();
        Err(format!(
            "llama-server not healthy after {HEALTH_TRIES} x {HEALTH_INTERVAL:?}"
        ))
    }

    /// Attach to a llama-server someone else runs at `base_url` (e.g.
    /// `http://127.0.0.1:18650`) and wait, bounded, until `/health` answers
    /// ok. The Flash-Next reference is a 111 GB mmap in its own
    /// memory-capped container that takes minutes to load, so spawning it
    /// per test is not an option — and nothing is killed on drop.
    #[allow(dead_code)] // only the recorder attaches
    pub fn attach(base_url: &str) -> Result<Self, String> {
        let base_url = base_url.trim_end_matches('/').to_string();
        for _ in 0..HEALTH_TRIES {
            if healthy(&base_url) {
                return Ok(Self {
                    child: None,
                    base_url,
                });
            }
            std::thread::sleep(HEALTH_INTERVAL);
        }
        Err(format!(
            "llama-server at {base_url} not healthy after {HEALTH_TRIES} x {HEALTH_INTERVAL:?}"
        ))
    }

    /// GET a JSON endpoint — e.g. `/props`, whose `build_info` names the
    /// build a fixture was recorded against.
    #[allow(dead_code)] // only the recorder reads server metadata
    pub fn get_json(&self, path: &str) -> Result<serde_json::Value, String> {
        let url = format!("{}{path}", self.base_url);
        let mut resp = ureq::get(&url)
            .call()
            .map_err(|e| format!("GET {path}: {e}"))?;
        read_json(&mut resp, path)
    }

    /// POST an arbitrary `/completion` body and return the parsed JSON. The
    /// recorder keeps fields the live legs never read (`timings`, `tokens`,
    /// `stop_type`); [`Self::greedy_completion`] is this transport with the
    /// gates' fixed body.
    pub fn raw_completion(&self, body: &serde_json::Value) -> Result<serde_json::Value, String> {
        let url = format!("{}/completion", self.base_url);
        let payload = body.to_string();
        let mut resp = ureq::post(&url)
            .header("Content-Type", "application/json")
            .send(payload.as_str())
            .map_err(|e| format!("POST /completion: {e}"))?;
        read_json(&mut resp, "/completion")
    }

    /// Raw greedy completion from exact `prompt_ids`: temperature 0,
    /// `n_predict` tokens, top-`n_probs` pre-sampling logprobs per position.
    /// Asserts the server evaluated exactly the ids we sent (no BOS injection).
    #[allow(dead_code)] // the recorder sends its own body through raw_completion
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
        let v = self.raw_completion(&body)?;
        parse_completion(&v, prompt_ids.len())
    }
}

fn healthy(base_url: &str) -> bool {
    let url = format!("{base_url}/health");
    matches!(ureq::get(&url).call(), Ok(resp) if resp.status() == 200)
}

fn read_json(
    resp: &mut ureq::http::Response<ureq::Body>,
    what: &str,
) -> Result<serde_json::Value, String> {
    let mut text = String::new();
    resp.body_mut()
        .as_reader()
        .take(MAX_RESPONSE_BYTES)
        .read_to_string(&mut text)
        .map_err(|e| format!("read {what} body: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {what} JSON: {e}"))
}

/// Parse a raw `/completion` response into a [`Completion`], failing when the
/// server did not evaluate exactly `prompt_len` tokens (BOS injection).
///
/// Field names checked against a live llama.cpp b10991 (930e2fa59) response
/// on 2026-09-16: `tokens_evaluated`, `content`, and
/// `completion_probabilities[]`, each position carrying the sampled
/// `id`/`logprob` plus a best-first `top_logprobs[]` of
/// `{id, token, bytes, logprob}`. The logprobs are natural-log PRE-sampling
/// values (`post_sampling_probs` defaults off), so a `top_k: 1` sampler does
/// not truncate the list.
pub fn parse_completion(v: &serde_json::Value, prompt_len: usize) -> Result<Completion, String> {
    let evaluated = v["tokens_evaluated"].as_u64().unwrap_or(0) as usize;
    if evaluated != prompt_len {
        return Err(format!(
            "server evaluated {evaluated} tokens for a {prompt_len}-id prompt — \
             token-id passthrough is broken (BOS injection?)"
        ));
    }
    let content = v["content"]
        .as_str()
        .ok_or_else(|| format!("no content in: {v}"))?
        .to_string();
    let mut steps = Vec::new();
    let mut chosen = Vec::new();
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
            let id = pos["id"]
                .as_u64()
                .ok_or("completion_probabilities position without a sampled id")?;
            chosen.push(id as u32);
        }
    }
    Ok(Completion {
        content,
        steps,
        chosen,
    })
}

/// One raw completion: the greedy text plus, per generated position, the
/// top-k `(token id, natural-log probability)` pairs, best first.
#[derive(Debug)]
pub struct Completion {
    // Only `parity_lfm2` and the recorder read the greedy text, so the other
    // parity binaries see this field as dead. It is real reference data, not
    // cruft.
    #[allow(dead_code)]
    pub content: String,
    pub steps: Vec<Vec<(u32, f64)>>,
    /// The id the server actually sampled at each position. Under greedy
    /// sampling this is `steps[i][0].0` unless the top two tie exactly; it is
    /// kept separately so a fixture never has to assume that.
    #[allow(dead_code)]
    pub chosen: Vec<u32>,
}

/// Natural-log softmax probabilities of `logits` at `ids`, computed in f64
/// (the reference reports logprobs, not raw logits — same transform here).
#[allow(dead_code)] // the recorder has no logits of its own
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
