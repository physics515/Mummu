//! The `mummu-serve` binary: read the environment, hand it to the library.
//!
//! Everything that used to live here is now `mummu_serve` the library (see
//! `lib.rs`) so the Tauri desktop shell can run the same routers in-process.
//! This file is only the env-to-argument translation and the process exit
//! code, and it is deliberately the *only* place that reads `MUMMU_ADDR` /
//! `MUMMU_OLLAMA_ADDR` — a library that reached for the environment behind
//! its caller's back would give the app a second, invisible config surface.
//!
//! Configuration (env): `MUMMU_ADDR` (default `0.0.0.0:8095`),
//! `MUMMU_OLLAMA_ADDR` (default `0.0.0.0:11435`; `off` disables),
//! `MUMMU_MODELS_DIR` (default `./models`), `MUMMU_BACKEND`,
//! `MUMMU_FORCE_CPU`.

use std::process::ExitCode;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    let addr = std::env::var("MUMMU_ADDR")
        .unwrap_or_else(|_| mummu_serve::DEFAULT_ADDR.into());
    let shim_addr = std::env::var("MUMMU_OLLAMA_ADDR")
        .unwrap_or_else(|_| mummu_serve::DEFAULT_OLLAMA_ADDR.into());

    let root = mummu_serve::prepare_models_root().expect("models dir must be creatable");
    mummu_serve::log_device_policy(&root);

    match mummu_serve::serve(&addr, Some(&shim_addr)).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[mummu-serve] {e}");
            ExitCode::FAILURE
        }
    }
}
