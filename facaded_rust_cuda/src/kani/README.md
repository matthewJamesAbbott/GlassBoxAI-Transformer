# Kani Verification Test Suite

## CISA Secure-by-Design Hardening Proofs

This directory contains **196 formal verification harnesses** using the [Kani Rust Verifier](https://model-checking.github.io/kani/) to prove security properties required for CISA/NSA compliance.

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

### `lora.rs`
Verifies LoRA (Low-Rank Adaptation) parameter-efficient fine-tuning safety including adapter arithmetic, memory bounds, serialization, and **validation functions** that enforce CISA requirements at runtime.

**Configuration Validation Proofs (NEW):**
- `verify_validate_rejects_zero_rank` - Division-by-zero prevention (CISA #5)
- `verify_validate_rejects_excessive_rank` - Resource limit enforcement (CISA #15)
- `verify_validate_rejects_invalid_alpha` - Floating-point sanity (CISA #14)
- `verify_validate_rejects_invalid_dropout` - Division-by-zero prevention (CISA #5)
- `verify_valid_config_passes` - No-panic for valid configs (CISA #3)

**Safe Scaling Proofs (NEW):**
- `verify_try_scaling_rank_zero` - try_scaling() returns None for rank=0
- `verify_try_scaling_finite_result` - Scaling is finite for valid inputs

**Constructor Validation Proofs (NEW):**
- `verify_try_new_rejects_zero_heads` - Division-by-zero prevention (CISA #5)
- `verify_try_new_rejects_indivisible_dim` - Dimension validation
- `verify_head_dim_calculation_safe` - Safe division (CISA #4)
- `verify_kv_dim_no_overflow` - Integer overflow prevention (CISA #4)

**Memory Budget Proofs (NEW):**
- `verify_adapter_memory_checked` - Checked arithmetic in memory calc
- `verify_safe_add_params_no_overflow` - Safe parameter accumulation
- `verify_memory_budget_enforcement` - 1GB budget enforced (CISA #15)
- `verify_layer_budget_accumulation` - Layer accumulation safety

**File Parsing Security Proofs (NEW):**
- `verify_load_rejects_negative_rank` - Input validation (CISA #1)
- `verify_load_rejects_excessive_file_rank` - DoS prevention (CISA #15)
- `verify_load_rejects_excessive_name_len` - Name length limit (CISA #15)
- `verify_load_version_bounds` - Version validation (CISA #1)
- `verify_load_dimension_bounds` - Dimension DoS prevention (CISA #15)

**Cleanup & State Proofs (NEW):**
- `verify_cleanup_safe_state` - State reset verification (CISA #10)
- `verify_cleanup_idempotent` - Idempotency guarantee (CISA #10)
- `verify_adam_timestep_increment_safe` - Timestep overflow prevention
- `verify_dropout_seed_valid` - Seed validity

**Floating-Point Safety Proofs (NEW):**
- `verify_inverted_dropout_scale_safe` - Dropout scale bounds
- `verify_a_init_uniform_bounds` - Initialization bounds
- `verify_full_forward_chain_finite` - Forward pass finiteness
- `verify_gradient_clipping_safe` - Gradient clipping safety
- `verify_backward_pass_safe` - Backward pass finiteness

**Original Proofs:**
- `verify_lora_rank_bounds` - LoRA rank configuration validation
- `verify_lora_alpha_bounds` - LoRA alpha scaling factor safety
- `verify_lora_scaling_factor` - alpha/rank division safety
- `verify_lora_dropout_bounds` - Dropout rate in [0, 1)
- `verify_lora_a_matrix_size` - A matrix (rank × in_dim) sizing
- `verify_lora_b_matrix_size` - B matrix (out_dim × rank) sizing
- `verify_lora_adapter_total_params` - Per-adapter parameter count
- `verify_lora_layer_total_params` - Per-layer LoRA budget
- `verify_lora_forward_a_safe` - Forward A @ input computation
- `verify_lora_forward_b_safe` - Forward scaling * B @ temp
- `verify_lora_dropout_scaling` - Inverted dropout scaling
- `verify_lora_backward_b_safe` - Gradient w.r.t. B computation
- `verify_lora_backward_temp_safe` - Gradient w.r.t. temp
- `verify_lora_backward_a_safe` - Gradient w.r.t. A computation
- `verify_lora_adam_update_safe` - LoRA-specific Adam optimizer
- `verify_lora_merge_element_safe` - Merge into base weights
- `verify_lora_ba_product_safe` - B @ A matrix product
- `verify_lora_file_header_bounds` - Serialization header validation
- `verify_lora_flags_exhaustive` - Adapter enable flags
- `verify_lora_name_length_bounds` - Name field length
- `verify_lora_memory_budget` - Total LoRA memory footprint
- `verify_lora_temp_buffer_size` - Temp buffer allocation
- `verify_lora_layers_parsing` - Layer configuration parsing
- `verify_freeze_base_independence` - Base freeze flag
- `verify_lora_a_init_bounds` - A matrix initialization
- `verify_lora_b_init_zero` - B matrix zero initialization
- `verify_layer_lora_access_bounds` - Layer index bounds
- `verify_adapter_element_access` - Adapter element indexing

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
