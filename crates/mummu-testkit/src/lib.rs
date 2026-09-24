//! Shared fixtures for `mummu`'s integration tests.
//!
//! - [`llama_ref`] spawns or attaches to a llama.cpp `llama-server` and
//!   queries its raw `/completion` endpoint with token-id prompts.
//! - [`gguf_compare`] is the one quantized-reference verdict every GGUF
//!   parity leg hands its numbers to.
//! - [`qwen4exp_fixture`] is the recorded reference for the Qwen3.8-Flash-Next
//!   gate, whose live reference does not fit beside our load.
//!
//! A library rather than `#[path]`-included files so that a test binary
//! using one half of a module never sees the other half as dead code.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

pub mod gguf_compare;
pub mod llama_ref;
pub mod qwen4exp_fixture;
