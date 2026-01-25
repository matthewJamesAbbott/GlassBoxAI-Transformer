// Kani Verification Test Suite for GlassBox Transformer
// CISA Secure-by-Design Hardening Proofs
//
// This module provides formal verification harnesses using the Kani Rust Verifier
// to prove security properties required for CISA compliance.

#[cfg(kani)]
pub mod bounds;

#[cfg(kani)]
pub mod arithmetic;

#[cfg(kani)]
pub mod memory;

#[cfg(kani)]
pub mod panics;

#[cfg(kani)]
pub mod enums;

#[cfg(kani)]
pub mod floats;

#[cfg(kani)]
pub mod tokenizer;

#[cfg(kani)]
pub mod quant;

#[cfg(kani)]
pub mod model;
