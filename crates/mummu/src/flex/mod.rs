//! Host CPU kernels.
//!
//! The `mummu::flex::kernels` module holds the packed-nibble Q4 GEMV that
//! reads 0.5625 B/param off DRAM instead of the 1.125 B/param the i8 slab
//! pays, evaluated with AVX-512 VNNI integer dot products (scalar fallback
//! everywhere else). See the module docs in [`kernels`].

pub mod gdn;
pub mod head;
pub mod insitu;
pub mod kernels;
pub mod registry;
