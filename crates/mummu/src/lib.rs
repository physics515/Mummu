//! Mummu — a from-scratch [Burn](https://burn.dev) model runner.
//!
//! One library that imports open models (LLM, embeddings, later vision),
//! quantizes them to fit, and runs them natively in Rust on whatever hardware
//! is present — CPU, one GPU, or several. All model code is generic over
//! `B: Backend`; consumers (laurelane, Nanna) pick a device at runtime and
//! keep their domain glue out of this crate.

// Burn's `fusion` feature wraps backends in deeply nested generic types.
#![recursion_limit = "512"]

pub mod adapt;
/// Wall-clock attribution machinery: exact Shapley values over togglable
/// components, with repeated-measure confidence intervals (SPEC 2).
pub mod attrib;
pub mod attn_config;
pub mod backend;
/// Host CPU kernels: the packed-nibble Q4 GEMV at the DRAM roofline
/// (AVX-512 VNNI) and its calibration machinery (SPEC 1).
pub mod flex;
/// Synchronous-dataflow model of the decode step: priced DAG, the maximum
/// cycle ratio, and the T* period floor (SPEC 2).
pub mod sdf;
pub mod cache;
pub mod chat;
pub mod decode;
pub mod gguf;
/// Generated IQ-quant codebook tables (see the module header).
mod gguf_iq_grids;
pub mod hub;
pub mod import;
pub mod manage;
/// Scheduler B — per-tensor precision placement (crate `mummu-mix`).
pub use mummu_mix as mix;
pub mod models;
pub mod nn;
pub mod pack;
pub mod partition;
pub mod plan;
pub mod prof;
pub mod quant;
pub mod registry;
pub mod safetensors;
/// Scheduler A — dividing work across devices (crate `mummu-schedule`).
pub use mummu_schedule as schedule;
/// Render a checkpoint's own imported chat template (feature `jinja-template`).
#[cfg(feature = "jinja-template")]
pub mod template;
pub mod tier;
pub mod tok_config;
pub mod vram;
pub mod workingset;
pub mod tokenizer;
pub mod tune;
