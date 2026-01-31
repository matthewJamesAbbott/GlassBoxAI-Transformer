# Kani Verification Test Suite

## CISA Secure-by-Design Hardening Proofs

This directory contains **124 formal verification harnesses** using the [Kani Rust Verifier](https://model-checking.github.io/kani/) to prove security properties required for CISA compliance.

**Verification Status**: All harnesses pass verification successfully.

## Requirements Covered

| # | Requirement | Module(s) | Status |
|---|-------------|-----------|--------|
| 1 | Strict Bound Checks | `bounds.rs`, `quant.rs`, `model.rs` | ✅ |
| 2 | Pointer Validity Proofs | `memory.rs` | ✅ |
| 3 | No-Panic Guarantee | `panics.rs` | ✅ |
| 4 | Integer Overflow Prevention | `arithmetic.rs`, `model.rs` | ✅ |
| 5 | Division-by-Zero Exclusion | `arithmetic.rs` | ✅ |
| 6 | Global State Consistency | N/A (no shared mutable state) | ⚪ |
| 7 | Deadlock-Free Logic | N/A (no locks in verified code) | ⚪ |
| 8 | Input Sanitization Bounds | `tokenizer.rs` | ✅ |
| 9 | Result Coverage Audit | `panics.rs`, `enums.rs` | ✅ |
| 10 | Memory Leak/Leakage Proofs | `memory.rs` | ✅ |
| 11 | Constant-Time Execution | N/A (no cryptographic secrets) | ⚪ |
| 12 | State Machine Integrity | `enums.rs`, `model.rs` | ✅ |
| 13 | Enum Exhaustion | `enums.rs` | ✅ |
| 14 | Floating-Point Sanity | `floats.rs` | ✅ |
| 15 | Resource Limit Compliance | `memory.rs`, `model.rs` | ✅ |

## Running Verification

### Prerequisites

Install Kani:
```bash
cargo install --locked kani-verifier
kani setup
```

### Run All Proofs

```bash
cd facaded_rust_cuda
kani --workspace
```

### Run Specific Module

```bash
# Bounds verification
kani --harness verify_q8_0_dequant_bounds

# Arithmetic safety
kani --harness verify_block_count_no_overflow

# Panic freedom
kani --harness verify_ggml_dtype_from_no_panic

# Floating-point safety
kani --harness verify_fp16_to_fp32_safety
```

### Run With Increased Unwind Limit

For proofs with loops:
```bash
kani --harness verify_encode_terminates --unwind 50
```

## Module Overview

### `bounds.rs`
Proves that all array/slice/vector indexing operations are mathematically incapable of out-of-bounds access under any symbolic input.

Key proofs:
- `verify_get_scale_min_k4_bounds` - Scale extraction indexing
- `verify_q8_0_dequant_bounds` - Dequantization output bounds
- `verify_tokenizer_decode_bounds` - Vocabulary access safety

### `arithmetic.rs`
Proves that all arithmetic operations are safe from wrapping, overflowing, underflowing, and division-by-zero.

Key proofs:
- `verify_block_count_no_overflow` - Block calculation safety
- `verify_scale_arithmetic_no_overflow` - Bitwise operation safety
- `verify_qk_k_division_safe` - Constant divisor validation

### `memory.rs`
Verifies memory allocation bounds, struct layouts, and resource budget compliance.

Key proofs:
- `verify_bytemuck_alignment` - Block struct alignment
- `verify_block_struct_sizes` - Struct size invariants
- `verify_tensor_security_budget` - Allocation limits

### `panics.rs`
Proves that functions cannot trigger `panic!`, `unwrap()`, or `expect()` failures.

Key proofs:
- `verify_ggml_dtype_from_no_panic` - Enum conversion safety
- `verify_result_handling_pattern` - Error handling patterns
- `verify_checked_arithmetic` - Safe arithmetic patterns

### `enums.rs`
Verifies exhaustive enum matching and state machine integrity.

Key proofs:
- `verify_ggml_dtype_exhaustive` - All variants handled
- `verify_tensor_data_exhaustive` - TensorData matching
- `verify_quantized_dtype_gate` - Type validation gates

### `floats.rs`
Proves that floating-point operations handle NaN/Infinity safely.

Key proofs:
- `verify_fp16_bit_pattern_properties` - FP16 bit layout validation
- `verify_bf16_bit_pattern_properties` - BF16 conversion safety
- `verify_softmax_denominator` - Division-by-zero prevention
- `verify_rms_scale_calculation` - RMS normalization safety

**Note**: Tests avoid calling the `half` crate directly as it contains inline assembly not supported by Kani.

### `tokenizer.rs`
Verifies tokenizer bounds checking and input sanitization.

Key proofs:
- `verify_decode_bounds` - Vocabulary access bounds
- `verify_encode_terminates` - BPE loop termination
- `verify_bpe_search_bounded` - Search operation limits

### `quant.rs`
Verifies quantization/dequantization arithmetic safety.

Key proofs:
- `verify_q4k_scale_extraction` - Q4_K bit operations
- `verify_q6k_bit_reconstruction` - Q6_K 6-bit unpacking
- `verify_bytes_calculation` - Memory size computation

### `model.rs`
Verifies model loading, dimension calculations, and access patterns.

Key proofs:
- `verify_layer_access_bounds` - Layer indexing safety
- `verify_attention_head_bounds` - GQA head mapping
- `verify_cache_size_calculation` - KV cache sizing

### `trainer.rs`
Verifies training infrastructure safety including backpropagation, Adam optimizer, gradient clipping, and activation caching.

Key proofs:
- `verify_learning_rate_bounds` - Training hyperparameter validation
- `verify_adam_beta_bounds` - Adam optimizer β1/β2 constraints
- `verify_gradient_clip_scaling` - Gradient clipping scale factor
- `verify_adam_weight_update_no_div_zero` - Division-by-zero prevention with ε
- `verify_adam_epsilon_safety` - Adam ε prevents zero denominator
- `verify_cross_entropy_log_safe` - Log computation with clamping
- `verify_softmax_denom_positive` - Softmax denominator safety
- `verify_activation_cache_layer_bounds` - Cache indexing bounds
- `verify_silu_backward_safe` - SiLU activation backward pass
- `verify_attention_weight_dims` - Weight dimension calculations
- `verify_ffn_weight_dims` - FFN weight size safety
- `verify_total_params_calculation` - Parameter count overflow prevention

## Security Budget Constants

The verification uses these security budgets (defined in proofs):

| Resource | Limit | Rationale |
|----------|-------|-----------|
| Max embedding table | 8 GB | Prevents memory exhaustion |
| Max layer weight | 1 GB | Per-layer budget |
| Max KV cache | 2 GB | Cache memory limit |
| Max vocab size | 200,000 | Token space limit |
| Max sequence length | 4,096 | Context window limit |
| Max layers | 128 | Model depth limit |

## Interpretation of Results

- **VERIFICATION SUCCESSFUL**: Property proven for all possible inputs
- **VERIFICATION FAILED**: Counterexample found - fix required
- **UNWINDING ASSERTION**: Increase `--unwind` value
- **OUT OF MEMORY**: Reduce problem complexity

## Contributing

When adding new functionality:

1. Create corresponding Kani proofs
2. Cover all CISA requirements applicable to the new code
3. Use symbolic inputs (`kani::any()`) for exhaustive checking
4. Document assumptions with `kani::assume()`
5. Add assertions with `kani::assert!()`
