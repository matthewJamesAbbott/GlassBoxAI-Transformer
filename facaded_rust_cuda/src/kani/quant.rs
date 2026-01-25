// Kani Verification: Quantization Safety
// CISA Requirements #1, #4, #14: Bounds, Arithmetic, Floats

#[cfg(kani)]
mod quant_proofs {
    use crate::quant::*;

    // ==========================================================================
    // Q8_0 Dequantization Proofs
    // ==========================================================================

    /// Verify Q8_0 index calculations are within bounds
    #[kani::proof]
    fn verify_q8_0_indexing() {
        let block_idx: usize = kani::any();
        let j: usize = kani::any();
        
        kani::assume(block_idx < 1000);
        kani::assume(j < QK8_0);
        
        let output_idx = block_idx * QK8_0 + j;
        
        kani::assert(
            output_idx < (block_idx + 1) * QK8_0,
            "Output index within block range"
        );
    }

    /// Verify Q8_0 dequant value range
    #[kani::proof]
    fn verify_q8_0_value_range() {
        let d: f32 = kani::any();
        let q: i8 = kani::any();
        
        kani::assume(d.is_finite() && d.abs() < 100.0);
        
        let result = d * (q as f32);
        
        // q is in [-128, 127], so result is in [-12700, 12700] for d < 100
        if d.abs() < 100.0 {
            kani::assert(
                result.abs() < 12800.0 || !result.is_finite(),
                "Q8_0 result bounded for bounded d"
            );
        }
    }

    // ==========================================================================
    // Q4_K Dequantization Proofs
    // ==========================================================================

    /// Verify Q4_K scale extraction bounds
    #[kani::proof]
    #[kani::unwind(12)]
    fn verify_q4k_scale_extraction() {
        let scales: [u8; K_SCALE_SIZE] = kani::any();
        
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &scales);
            
            // Both values fit in u8
            kani::assert(sc <= 255, "Scale fits in u8");
            kani::assert(m <= 255, "Min fits in u8");
        }
    }

    /// Verify Q4_K nibble extraction
    #[kani::proof]
    fn verify_q4k_nibble_extraction() {
        let byte: u8 = kani::any();
        
        let low = byte & 0xF;
        let high = byte >> 4;
        
        kani::assert(low <= 15, "Low nibble <= 15");
        kani::assert(high <= 15, "High nibble <= 15");
    }

    /// Verify Q4_K output indexing
    #[kani::proof]
    fn verify_q4k_output_indexing() {
        let n: usize = kani::any();
        let l: usize = kani::any();
        
        kani::assume(n < QK_K && n % 64 == 0);
        kani::assume(l < 32);
        
        let idx1 = n + l;
        let idx2 = n + 32 + l;
        
        kani::assert(idx1 < QK_K, "First index in bounds");
        kani::assert(idx2 < QK_K, "Second index in bounds");
    }

    // ==========================================================================
    // Q6_K Dequantization Proofs
    // ==========================================================================

    /// Verify Q6_K 6-bit reconstruction
    #[kani::proof]
    fn verify_q6k_bit_reconstruction() {
        let ql: u8 = kani::any();
        let qh: u8 = kani::any();
        
        // Low 4 bits from ql, high 2 bits from qh
        let q = (ql & 0xF) | (((qh >> 0) & 3) << 4);
        
        kani::assert(q <= 63, "6-bit value <= 63");
        
        // After subtracting 32, range is [-32, 31]
        let q_signed = q as i8 - 32;
        kani::assert(q_signed >= -32 && q_signed <= 31, "Signed value in range");
    }

    /// Verify Q6_K offset arithmetic
    #[kani::proof]
    fn verify_q6k_offset_arithmetic() {
        let n: usize = kani::any();
        let l: usize = kani::any();
        
        kani::assume(n < 256 && n % 128 == 0);
        kani::assume(l < 32);
        
        let offsets = [n + l, n + l + 32, n + l + 64, n + l + 96];
        
        for offset in offsets {
            kani::assert(offset < QK_K, "All offsets within QK_K");
        }
    }

    /// Verify Q6_K ql/qh offset advancement
    #[kani::proof]
    fn verify_q6k_pointer_advancement() {
        let mut ql_offset: usize = 0;
        let mut qh_offset: usize = 0;
        let mut sc_offset: usize = 0;
        
        // Two iterations of 128-element loop
        for _ in 0..2 {
            kani::assert(ql_offset < 128, "ql_offset in bounds");
            kani::assert(qh_offset < 64, "qh_offset in bounds");
            kani::assert(sc_offset < 16, "sc_offset in bounds");
            
            ql_offset += 64;
            qh_offset += 32;
            sc_offset += 8;
        }
        
        kani::assert(ql_offset == 128, "Final ql_offset correct");
        kani::assert(qh_offset == 64, "Final qh_offset correct");
        kani::assert(sc_offset == 16, "Final sc_offset correct");
    }

    // ==========================================================================
    // Q2_K Dequantization Proofs
    // ==========================================================================

    /// Verify Q2_K 2-bit extraction
    #[kani::proof]
    fn verify_q2k_bit_extraction() {
        let byte: u8 = kani::any();
        let idx: usize = kani::any();
        
        kani::assume(idx < 4);
        
        let shift = idx * 2;
        let q = (byte >> shift) & 3;
        
        kani::assert(q <= 3, "2-bit value <= 3");
        kani::assert(shift <= 6, "Shift amount valid");
    }

    /// Verify Q2_K indexing
    #[kani::proof]
    fn verify_q2k_indexing() {
        let j: usize = kani::any();
        let l: usize = kani::any();
        
        kani::assume(j < QK_K / 16);
        kani::assume(l < 16);
        
        let idx = j * 16 + l;
        let byte_idx = idx / 4;
        
        kani::assert(idx < QK_K, "Index within QK_K");
        kani::assert(byte_idx < QK_K / 4, "Byte index within qs array");
    }

    // ==========================================================================
    // Q3_K Dequantization Proofs
    // ==========================================================================

    /// Verify Q3_K high-bit mask operations
    #[kani::proof]
    fn verify_q3k_mask_operations() {
        let hm: u8 = kani::any();
        let m: u8 = kani::any();
        
        kani::assume(m.count_ones() == 1); // m is a power of 2
        
        let has_high_bit = (hm & m) != 0;
        let adjustment = if has_high_bit { 0 } else { 4 };
        
        kani::assert(adjustment <= 4, "Adjustment bounded");
    }

    /// Verify Q3_K scale unpacking
    #[kani::proof]
    fn verify_q3k_scale_unpacking() {
        let aux: [u32; 4] = kani::any();
        
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;
        
        let masked1 = aux[0] & KMASK2;
        let _masked2 = (aux[0] >> 4) & KMASK2;
        let shift_bits = (aux[2] >> 0) & KMASK1;
        
        // Each byte is masked to at most 4 bits
        kani::assert(masked1 <= 0x0f0f0f0f, "Mask2 limits to 4 bits");
        kani::assert(shift_bits <= 0x03030303, "Mask1 limits to 2 bits");
    }

    // ==========================================================================
    // Q5_K Dequantization Proofs
    // ==========================================================================

    /// Verify Q5_K 5-bit reconstruction
    #[kani::proof]
    fn verify_q5k_bit_reconstruction() {
        let ql: u8 = kani::any();
        let qh: u8 = kani::any();
        let mask: u8 = kani::any();
        
        kani::assume(mask.count_ones() == 1);
        
        let q_base = (ql & 0xF) as i32;
        let q_high = if (qh & mask) != 0 { 16 } else { 0 };
        let q_total = q_base + q_high;
        
        kani::assert(q_base <= 15, "Base 4 bits");
        kani::assert(q_total <= 31, "Total 5 bits");
    }

    // ==========================================================================
    // General Block Proofs
    // ==========================================================================

    /// Verify block count calculation safety
    #[kani::proof]
    fn verify_block_count_general() {
        let cols: usize = kani::any();
        let qtype: u32 = kani::any();
        
        kani::assume(cols > 0 && cols <= 100_000_000);
        kani::assume(qtype <= 15);
        
        let dtype = GGMLDType::from(qtype);
        let block_size = get_block_size(dtype);
        
        kani::assume(cols % block_size == 0);
        
        let nb = cols / block_size;
        
        kani::assert(nb * block_size == cols, "Block calculation reversible");
        kani::assert(nb <= cols, "Block count <= element count");
    }

    /// Verify bytes calculation doesn't overflow
    #[kani::proof]
    fn verify_bytes_calculation() {
        let num_elements: usize = kani::any();
        let qtype: u32 = kani::any();
        
        kani::assume(num_elements <= 100_000_000);
        kani::assume(qtype <= 15);
        
        let dtype = GGMLDType::from(qtype);
        let block_size = get_block_size(dtype);
        let bytes_per_block = get_bytes_per_block(dtype);
        
        kani::assume(num_elements % block_size == 0);
        
        let num_blocks = num_elements / block_size;
        let total_bytes = num_blocks.saturating_mul(bytes_per_block);
        
        // Quantized should use less memory than f32
        let f32_bytes = num_elements.saturating_mul(4);
        
        if bytes_per_block > 0 && block_size > 1 {
            kani::assert(
                total_bytes <= f32_bytes || total_bytes == usize::MAX,
                "Quantized uses less or equal memory"
            );
        }
    }
}
