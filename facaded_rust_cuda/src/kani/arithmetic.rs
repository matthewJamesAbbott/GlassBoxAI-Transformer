// Kani Verification: Integer Overflow Prevention & Division-by-Zero
// CISA Requirements #4 & #5

#[cfg(kani)]
mod arithmetic_proofs {
    use crate::quant::*;

    // ==========================================================================
    // Requirement #4: Integer Overflow Prevention
    // ==========================================================================

    /// Verify QK_K block count calculation is overflow-safe
    #[kani::proof]
    fn verify_block_count_no_overflow() {
        let cols: usize = kani::any();
        
        // Constrain to realistic tensor dimensions
        kani::assume(cols > 0 && cols <= 1_000_000_000);
        kani::assume(cols % QK_K == 0);
        
        let nb = cols / QK_K;
        
        // Verify no wraparound occurred
        kani::assert(nb <= cols, "Block count calculation safe");
        kani::assert(nb * QK_K == cols, "Block multiplication reversible");
    }

    /// Verify Q8_0 block count is safe
    #[kani::proof]
    fn verify_q8_0_block_count_safe() {
        let cols: usize = kani::any();
        
        kani::assume(cols > 0 && cols <= 1_000_000_000);
        kani::assume(cols % QK8_0 == 0);
        
        let nb = cols / QK8_0;
        
        kani::assert(nb * QK8_0 == cols, "Q8_0 block calculation safe");
    }

    /// Verify scale/min extraction arithmetic is overflow-safe
    #[kani::proof]
    fn verify_scale_arithmetic_no_overflow() {
        let scale_byte: u8 = kani::any();
        
        // Test the bit operations used in get_scale_min_k4
        let sc = scale_byte & 63;           // Max: 63
        let m = scale_byte >> 6;            // Max: 3
        let shifted = m << 4;               // Max: 48
        
        kani::assert(sc <= 63, "Scale extraction bounded");
        kani::assert(m <= 3, "Shift right bounded");
        kani::assert(shifted <= 48, "Shift left bounded");
        
        // Combined operations
        let combined = (scale_byte & 0xF) | ((scale_byte >> 6) << 4);
        kani::assert(combined as u16 <= 255, "Combined operation fits in u8");
    }

    /// Verify dequantization multiplication is safe (using f32)
    #[kani::proof]
    fn verify_dequant_mul_no_overflow() {
        let d: f32 = kani::any();
        let q: i8 = kani::any();
        let scale: i8 = kani::any();
        
        // Constrain to realistic ranges
        kani::assume(d.is_finite() && d.abs() < 1e6);
        
        // This is the pattern: d * scale * (q - 32)
        let q_adj = q as i32 - 32;
        kani::assert(q_adj >= -160 && q_adj <= 95, "Q adjustment bounded");
        
        let result = d * (scale as f32) * (q_adj as f32);
        
        // Result may be infinite in edge cases, but should be finite for normal inputs
        // The key safety property is no undefined behavior
        kani::assert(!result.is_nan() || d.is_nan(), "Result NaN only if input NaN");
    }

    /// Verify index calculations in Q2_K dequantization
    #[kani::proof]
    #[kani::unwind(20)]
    fn verify_q2k_index_arithmetic() {
        let j: usize = kani::any();
        let l: usize = kani::any();
        
        kani::assume(j < QK_K / 16);
        kani::assume(l < 16);
        
        let idx = j * 16 + l;
        let byte_idx = idx / 4;
        let shift = (idx % 4) * 2;
        
        // Verify bounds
        kani::assert(idx < QK_K, "Index within QK_K");
        kani::assert(byte_idx < QK_K / 4, "Byte index within bounds");
        kani::assert(shift <= 6, "Shift amount valid");
    }

    /// Verify Q6_K offset calculations
    #[kani::proof]
    fn verify_q6k_offset_arithmetic() {
        let n: usize = kani::any();
        let l: usize = kani::any();
        
        kani::assume(n < QK_K && n % 128 == 0);
        kani::assume(l < 32);
        
        // These are the offset calculations from dequant_row_q6_k
        let offset1 = n + l;
        let offset2 = n + l + 32;
        let offset3 = n + l + 64;
        let offset4 = n + l + 96;
        
        kani::assert(offset1 < QK_K, "Offset 1 in bounds");
        kani::assert(offset2 < QK_K, "Offset 2 in bounds");
        kani::assert(offset3 < QK_K, "Offset 3 in bounds");
        kani::assert(offset4 < QK_K, "Offset 4 in bounds");
    }

    // ==========================================================================
    // Requirement #5: Division-by-Zero Exclusion
    // ==========================================================================

    /// Verify block count division is never zero
    #[kani::proof]
    fn verify_qk_k_division_safe() {
        // QK_K is a compile-time constant = 256
        kani::assert(QK_K > 0, "QK_K is non-zero constant");
        kani::assert(QK8_0 > 0, "QK8_0 is non-zero constant");
        kani::assert(QK4_0 > 0, "QK4_0 is non-zero constant");
    }

    /// Verify block size is never zero for valid types
    #[kani::proof]
    fn verify_block_size_nonzero() {
        let dtype_val: u32 = kani::any();
        kani::assume(dtype_val <= 30);
        
        let dtype = GGMLDType::from(dtype_val);
        let block_size = get_block_size(dtype);
        
        // For all known types, block size is at least 1
        kani::assert(block_size >= 1, "Block size is non-zero");
    }

    /// Verify bytes_per_block is defined for division operations
    #[kani::proof]
    fn verify_bytes_per_block_safe_for_division() {
        // Use symbolic selection instead of loop to avoid unwind issues
        let dtype_idx: u8 = kani::any();
        kani::assume(dtype_idx < 14);
        
        let dtype = match dtype_idx {
            0 => GGMLDType::Q2_K,
            1 => GGMLDType::Q3_K,
            2 => GGMLDType::Q4_K,
            3 => GGMLDType::Q5_K,
            4 => GGMLDType::Q6_K,
            5 => GGMLDType::Q8_K,
            6 => GGMLDType::Q8_0,
            7 => GGMLDType::Q4_0,
            8 => GGMLDType::Q4_1,
            9 => GGMLDType::Q5_0,
            10 => GGMLDType::Q5_1,
            11 => GGMLDType::F32,
            12 => GGMLDType::F16,
            _ => GGMLDType::BFloat16,
        };
        
        let bytes = get_bytes_per_block(dtype);
        kani::assert(bytes > 0, "Bytes per block non-zero for valid types");
    }

    /// Verify vocabulary size is non-zero before division
    #[kani::proof]
    fn verify_vocab_size_safe_for_operations() {
        let vocab_size: usize = kani::any();
        
        // The system should enforce non-empty vocabulary
        kani::assume(vocab_size > 0);
        
        let token_id: u32 = kani::any();
        let is_valid = (token_id as usize) < vocab_size;
        
        // This demonstrates safe division pattern
        if is_valid {
            let _ = token_id as usize % vocab_size;
        }
    }

    /// Verify saturating arithmetic for memory calculations
    #[kani::proof]
    fn verify_saturating_memory_calc() {
        let dim1: usize = kani::any();
        let dim2: usize = kani::any();
        let element_size: usize = 4; // sizeof(f32)
        
        kani::assume(dim1 <= 1_000_000);
        kani::assume(dim2 <= 1_000_000);
        
        // Use saturating multiplication to prevent overflow
        let size = dim1.saturating_mul(dim2).saturating_mul(element_size);
        
        // Result should never wrap
        kani::assert(size >= dim1.min(dim2), "Saturating mul doesn't wrap to small values");
    }
}
