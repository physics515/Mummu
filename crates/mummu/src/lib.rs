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
pub mod nn;
