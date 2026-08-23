//! Mummu — a from-scratch [Burn](https://burn.dev) model runner.
//!
//! One library that imports open models (LLM, embeddings, later vision),
//! quantizes them to fit, and runs them natively in Rust on whatever hardware
//! is present — CPU, one GPU, or several. All model code is generic over
//! `B: Backend`; consumers (laurelane, Nanna) pick a device at runtime and
//! keep their domain glue out of this crate.

// Burn's `fusion` feature wraps backends in deeply nested generic types.
#![recursion_limit = "512"]

pub mod backend;
pub mod cache;
pub mod chat;
pub mod decode;
pub mod gguf;
/// Generated IQ-quant codebook tables (see the module header).
mod gguf_iq_grids;
pub mod hub;
pub mod import;
pub mod manage;
pub mod models;
pub mod nn;
pub mod pack;
pub mod partition;
pub mod plan;
pub mod quant;
pub mod registry;
pub mod safetensors;
/// Render a checkpoint's own imported chat template (feature `jinja-template`).
#[cfg(feature = "jinja-template")]
pub mod template;
pub mod tier;
pub mod tok_config;
pub mod workingset;
pub mod tokenizer;
pub mod tune;
