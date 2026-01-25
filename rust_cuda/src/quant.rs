use bytemuck::{Pod, Zeroable};

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const QK8_0: usize = 32;
pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum GGMLDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BFloat16 = 30,
    Unknown = -1,
}

impl From<u32> for GGMLDType {
    fn from(v: u32) -> Self {
        match v {
            0 => GGMLDType::F32,
            1 => GGMLDType::F16,
            2 => GGMLDType::Q4_0,
            3 => GGMLDType::Q4_1,
            6 => GGMLDType::Q5_0,
            7 => GGMLDType::Q5_1,
            8 => GGMLDType::Q8_0,
            9 => GGMLDType::Q8_1,
            10 => GGMLDType::Q2_K,
            11 => GGMLDType::Q3_K,
            12 => GGMLDType::Q4_K,
            13 => GGMLDType::Q5_K,
            14 => GGMLDType::Q6_K,
            15 => GGMLDType::Q8_K,
            30 => GGMLDType::BFloat16,
            _ => GGMLDType::Unknown,
        }
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ2K {
    pub scales: [u8; QK_K / 16],
    pub qs: [u8; QK_K / 4],
    pub d: u16,
    pub dmin: u16,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ3K {
    pub hmask: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 4],
    pub scales: [u8; 12],
    pub d: u16,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ4K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qs: [u8; QK_K / 2],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ5K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qh: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 2],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: u16,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: u16,
    pub qs: [i8; QK8_0],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: u16,
    pub qs: [u8; QK4_0 / 2],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub d: u16,
    pub m: u16,
    pub qs: [u8; QK4_1 / 2],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub d: u16,
    pub qh: [u8; 4],
    pub qs: [u8; QK5_0 / 2],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub d: u16,
    pub m: u16,
    pub qh: [u8; 4],
    pub qs: [u8; QK5_1 / 2],
}

pub fn fp16_to_fp32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

pub fn bf16_to_fp32(bf: u16) -> f32 {
    half::bf16::from_bits(bf).to_f32()
}

#[inline]
pub fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let sc = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (sc, m)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

pub fn dequant_row_q2_k(blocks: &[BlockQ2K], output: &mut [f32], cols: usize) {
    let nb = cols / QK_K;
    for i in 0..nb {
        let d = fp16_to_fp32(blocks[i].d);
        let dmin = fp16_to_fp32(blocks[i].dmin);
        let qs = &blocks[i].qs;
        let sc = &blocks[i].scales;

        for j in 0..(QK_K / 16) {
            let scale = d * (sc[j] & 0xF) as f32;
            let min = dmin * (sc[j] >> 4) as f32;

            for l in 0..16 {
                let idx = j * 16 + l;
                let byte_idx = idx / 4;
                let shift = (idx % 4) * 2;
                let q = (qs[byte_idx] >> shift) & 3;
                output[i * QK_K + idx] = scale * q as f32 - min;
            }
        }
    }
}

pub fn dequant_row_q3_k(blocks: &[BlockQ3K], output: &mut [f32], cols: usize) {
    let nb = cols / QK_K;
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    for i in 0..nb {
        let d_all = fp16_to_fp32(blocks[i].d);
        let q = &blocks[i].qs;
        let hm = &blocks[i].hmask;

        let mut aux = [0u32; 4];
        let scales_bytes: [u8; 12] = blocks[i].scales;
        aux[0] = u32::from_le_bytes([scales_bytes[0], scales_bytes[1], scales_bytes[2], scales_bytes[3]]);
        aux[1] = u32::from_le_bytes([scales_bytes[4], scales_bytes[5], scales_bytes[6], scales_bytes[7]]);
        aux[2] = u32::from_le_bytes([scales_bytes[8], scales_bytes[9], scales_bytes[10], scales_bytes[11]]);

        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

        let scales: [i8; 16] = bytemuck::cast(aux);

        let mut m: u8 = 1;
        let mut is = 0usize;
        let mut out_idx = i * QK_K;
        let mut q_offset = 0usize;

        for _ in 0..(QK_K / 128) {
            let mut shift = 0;
            for _ in 0..4 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let qval = ((q[q_offset + l] >> shift) & 3) as i32
                        - if (hm[l] & m) != 0 { 0 } else { 4 };
                    output[out_idx] = dl * qval as f32;
                    out_idx += 1;
                }
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let qval = ((q[q_offset + l + 16] >> shift) & 3) as i32
                        - if (hm[l + 16] & m) != 0 { 0 } else { 4 };
                    output[out_idx] = dl * qval as f32;
                    out_idx += 1;
                }
                shift += 2;
                m <<= 1;
            }
            q_offset += 32;
        }
    }
}

pub fn dequant_row_q4_k(blocks: &[BlockQ4K], output: &mut [f32], cols: usize) {
    let nb = cols / QK_K;

    for i in 0..nb {
        let q = &blocks[i].qs;
        let d = fp16_to_fp32(blocks[i].d);
        let dmin = fp16_to_fp32(blocks[i].dmin);
        let y = &mut output[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut q_offset = 0usize;

        for n in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, &blocks[i].scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, &blocks[i].scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            for l in 0..32 {
                y[n + l] = d1 * (q[q_offset + l] & 0xF) as f32 - min1;
            }
            for l in 0..32 {
                y[n + 32 + l] = d2 * (q[q_offset + l] >> 4) as f32 - min2;
            }

            q_offset += 32;
            is += 2;
        }
    }
}

pub fn dequant_row_q5_k(blocks: &[BlockQ5K], output: &mut [f32], cols: usize) {
    let nb = cols / QK_K;

    for i in 0..nb {
        let ql = &blocks[i].qs;
        let qh = &blocks[i].qh;
        let d = fp16_to_fp32(blocks[i].d);
        let dmin = fp16_to_fp32(blocks[i].dmin);
        let y = &mut output[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut ql_offset = 0usize;

        for n in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, &blocks[i].scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, &blocks[i].scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            for l in 0..32 {
                let q_base = (ql[ql_offset + l] & 0xF) as i32;
                let q_high = if (qh[l] & u1) != 0 { 16 } else { 0 };
                y[n + l] = d1 * (q_base + q_high) as f32 - min1;
            }

            for l in 0..32 {
                let q_base = (ql[ql_offset + l] >> 4) as i32;
                let q_high = if (qh[l] & u2) != 0 { 16 } else { 0 };
                y[n + 32 + l] = d2 * (q_base + q_high) as f32 - min2;
            }

            ql_offset += 32;
            u1 <<= 2;
            u2 <<= 2;
            is += 2;
        }
    }
}

pub fn dequant_row_q6_k(blocks: &[BlockQ6K], output: &mut [f32], cols: usize) {
    let nb = cols / QK_K;

    for i in 0..nb {
        let d = fp16_to_fp32(blocks[i].d);
        let ql = &blocks[i].ql;
        let qh = &blocks[i].qh;
        let sc = &blocks[i].scales;
        let y = &mut output[i * QK_K..(i + 1) * QK_K];

        let mut ql_offset = 0usize;
        let mut qh_offset = 0usize;
        let mut sc_offset = 0usize;

        for n in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is = l / 16;

                let q1 = ((ql[ql_offset + l] & 0xF) | (((qh[qh_offset + l] >> 0) & 3) << 4)) as i8;
                let q2 = ((ql[ql_offset + l + 32] & 0xF) | (((qh[qh_offset + l] >> 2) & 3) << 4)) as i8;
                let q3 = ((ql[ql_offset + l] >> 4) | (((qh[qh_offset + l] >> 4) & 3) << 4)) as i8;
                let q4 = ((ql[ql_offset + l + 32] >> 4) | (((qh[qh_offset + l] >> 6) & 3) << 4)) as i8;

                y[n + l] = d * sc[sc_offset + is] as f32 * (q1 as i32 - 32) as f32;
                y[n + l + 32] = d * sc[sc_offset + is + 2] as f32 * (q2 as i32 - 32) as f32;
                y[n + l + 64] = d * sc[sc_offset + is + 4] as f32 * (q3 as i32 - 32) as f32;
                y[n + l + 96] = d * sc[sc_offset + is + 6] as f32 * (q4 as i32 - 32) as f32;
            }
            ql_offset += 64;
            qh_offset += 32;
            sc_offset += 8;
        }
    }
}

pub fn dequant_row_q8_0(blocks: &[BlockQ8_0], output: &mut [f32], cols: usize) {
    let nb = cols / QK8_0;
    for i in 0..nb {
        let d = fp16_to_fp32(blocks[i].d);
        for j in 0..QK8_0 {
            output[i * QK8_0 + j] = d * blocks[i].qs[j] as f32;
        }
    }
}

pub fn dequant_row_q8_k(blocks: &[BlockQ8K], output: &mut [f32], cols: usize) {
    let nb = cols / QK_K;
    for i in 0..nb {
        let d = blocks[i].d;
        for j in 0..QK_K {
            output[i * QK_K + j] = d * blocks[i].qs[j] as f32;
        }
    }
}

pub fn dequant_row_q4_0(blocks: &[BlockQ4_0], output: &mut [f32], cols: usize) {
    let nb = cols / QK4_0;
    for i in 0..nb {
        let d = fp16_to_fp32(blocks[i].d);
        for j in 0..(QK4_0 / 2) {
            let v = blocks[i].qs[j];
            output[i * QK4_0 + j] = d * ((v & 0x0F) as i32 - 8) as f32;
            output[i * QK4_0 + j + QK4_0 / 2] = d * ((v >> 4) as i32 - 8) as f32;
        }
    }
}

pub fn dequant_row_q4_1(blocks: &[BlockQ4_1], output: &mut [f32], cols: usize) {
    let nb = cols / QK4_1;
    for i in 0..nb {
        let d = fp16_to_fp32(blocks[i].d);
        let m = fp16_to_fp32(blocks[i].m);
        for j in 0..(QK4_1 / 2) {
            let v = blocks[i].qs[j];
            output[i * QK4_1 + j] = d * (v & 0x0F) as f32 + m;
            output[i * QK4_1 + j + QK4_1 / 2] = d * (v >> 4) as f32 + m;
        }
    }
}

pub fn dequant_row_q5_0(blocks: &[BlockQ5_0], output: &mut [f32], cols: usize) {
    let nb = cols / QK5_0;
    for i in 0..nb {
        let d = fp16_to_fp32(blocks[i].d);
        let qh = u32::from_le_bytes(blocks[i].qh);
        for j in 0..(QK5_0 / 2) {
            let v = blocks[i].qs[j];
            let xh_0 = ((qh >> j) & 1) << 4;
            let xh_1 = ((qh >> (j + 16)) & 1) << 4;
            output[i * QK5_0 + j] = d * ((v & 0x0F) as u32 + xh_0) as i32 as f32 - d * 16.0;
            output[i * QK5_0 + j + QK5_0 / 2] = d * ((v >> 4) as u32 + xh_1) as i32 as f32 - d * 16.0;
        }
    }
}

pub fn dequant_row_q5_1(blocks: &[BlockQ5_1], output: &mut [f32], cols: usize) {
    let nb = cols / QK5_1;
    for i in 0..nb {
        let d = fp16_to_fp32(blocks[i].d);
        let m = fp16_to_fp32(blocks[i].m);
        let qh = u32::from_le_bytes(blocks[i].qh);
        for j in 0..(QK5_1 / 2) {
            let v = blocks[i].qs[j];
            let xh_0 = ((qh >> j) & 1) << 4;
            let xh_1 = ((qh >> (j + 16)) & 1) << 4;
            output[i * QK5_1 + j] = d * ((v & 0x0F) as u32 + xh_0) as f32 + m;
            output[i * QK5_1 + j + QK5_1 / 2] = d * ((v >> 4) as u32 + xh_1) as f32 + m;
        }
    }
}

pub fn get_bytes_per_block(qtype: GGMLDType) -> usize {
    match qtype {
        GGMLDType::Q2_K => std::mem::size_of::<BlockQ2K>(),
        GGMLDType::Q3_K => std::mem::size_of::<BlockQ3K>(),
        GGMLDType::Q4_K => std::mem::size_of::<BlockQ4K>(),
        GGMLDType::Q5_K => std::mem::size_of::<BlockQ5K>(),
        GGMLDType::Q6_K => std::mem::size_of::<BlockQ6K>(),
        GGMLDType::Q8_K => std::mem::size_of::<BlockQ8K>(),
        GGMLDType::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
        GGMLDType::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
        GGMLDType::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
        GGMLDType::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
        GGMLDType::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
        GGMLDType::F32 => 4,
        GGMLDType::F16 => 2,
        GGMLDType::BFloat16 => 2,
        _ => 0,
    }
}

pub fn get_block_size(qtype: GGMLDType) -> usize {
    match qtype {
        GGMLDType::Q2_K
        | GGMLDType::Q3_K
        | GGMLDType::Q4_K
        | GGMLDType::Q5_K
        | GGMLDType::Q6_K
        | GGMLDType::Q8_K => QK_K,
        GGMLDType::Q4_0 | GGMLDType::Q4_1 | GGMLDType::Q5_0 | GGMLDType::Q5_1 | GGMLDType::Q8_0 => {
            32
        }
        _ => 1,
    }
}

pub fn get_dtype_name(qtype: GGMLDType) -> &'static str {
    match qtype {
        GGMLDType::F32 => "F32",
        GGMLDType::F16 => "F16",
        GGMLDType::Q2_K => "Q2_K",
        GGMLDType::Q3_K => "Q3_K",
        GGMLDType::Q4_K => "Q4_K",
        GGMLDType::Q5_K => "Q5_K",
        GGMLDType::Q6_K => "Q6_K",
        GGMLDType::Q8_K => "Q8_K",
        GGMLDType::Q8_0 => "Q8_0",
        GGMLDType::Q4_0 => "Q4_0",
        GGMLDType::Q4_1 => "Q4_1",
        GGMLDType::Q5_0 => "Q5_0",
        GGMLDType::Q5_1 => "Q5_1",
        GGMLDType::BFloat16 => "BF16",
        _ => "UNKNOWN",
    }
}
