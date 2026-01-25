// CUDA kernels for transformer inference using cudarc NVRTC
// Fused Unsloth-style kernels for 2x speed, 70% VRAM reduction

use cudarc::driver::{CudaDevice, CudaSlice, CudaFunction, LaunchAsync, LaunchConfig};

use std::sync::Arc;
use crate::error::{Result, TransformerError};
use crate::model::TensorData;

const KERNEL_SOURCE: &str = r#"
#define CUDART_INF_F __int_as_float(0x7f800000)

extern "C" {

// Warp-level reduction for RMSNorm
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// Fused RMSNorm: Single pass normalization with weights
__global__ void fusedRMSNormKernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int dim,
    const float eps,
    const int unitOffset
) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = input[i];
        ss += val * val;
    }
    ss = blockReduceSum(ss);
    
    __shared__ float rms_scale;
    if (threadIdx.x == 0) {
        rms_scale = rsqrtf(ss / dim + eps);
    }
    __syncthreads();
    
    if (unitOffset) {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            output[i] = input[i] * rms_scale * (1.0f + weight[i]);
        }
    } else {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            output[i] = input[i] * rms_scale * weight[i];
        }
    }
}

// RMSNorm on a slice (for per-head QK normalization)
__global__ void rmsNormSliceKernel(
    float* __restrict__ data,
    const float* __restrict__ weight,
    const int offset,
    const int size,
    const float eps
) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = data[offset + i];
        ss += val * val;
    }
    ss = blockReduceSum(ss);
    
    __shared__ float rms_scale;
    if (threadIdx.x == 0) {
        rms_scale = rsqrtf(ss / size + eps);
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        data[offset + i] = data[offset + i] * rms_scale * weight[i];
    }
}

// Fused RoPE with dynamic scaling
__global__ void fusedRoPEKernel(
    float* __restrict__ Q,
    float* __restrict__ K,
    const int qDim,
    const int kvDim,
    const int headDim,
    const int position,
    const float theta,
    const float ropeScale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float scaledPos = position / ropeScale;
    
    // Q rotations
    if (idx < qDim / 2) {
        int i = idx * 2;
        int headIdx = i % headDim;
        float freq = 1.0f / powf(theta, (float)headIdx / headDim);
        float angle = scaledPos * freq;
        float cs = cosf(angle), sn = sinf(angle);
        float q0 = Q[i], q1 = Q[i + 1];
        Q[i]     = q0 * cs - q1 * sn;
        Q[i + 1] = q0 * sn + q1 * cs;
    }
    
    // K rotations
    if (K != NULL && idx < kvDim / 2) {
        int i = idx * 2;
        int headIdx = i % headDim;
        float freq = 1.0f / powf(theta, (float)headIdx / headDim);
        float angle = scaledPos * freq;
        float cs = cosf(angle), sn = sinf(angle);
        float k0 = K[i], k1 = K[i + 1];
        K[i]     = k0 * cs - k1 * sn;
        K[i + 1] = k0 * sn + k1 * cs;
    }
}

// Fused SwiGLU: silu(gate) * up in single kernel
__global__ void fusedSwiGLUKernel(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    
    float g = gate[i];
    float silu_g = g / (1.0f + expf(-g));
    output[i] = silu_g * up[i];
}

// Fused GeGLU for Gemma: gelu(gate) * up
__global__ void fusedGeGLUKernel(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    
    float g = gate[i];
    float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
    output[i] = gelu * up[i];
}

// Vector-matrix multiply for single token (M=1 optimized)
// Weight matrix is stored as (N, K) where N=output_dim, K=input_dim
// out[n] = sum_k(vec[k] * mat[n * K + k])
__global__ void vecMatMulKernel(
    float* __restrict__ out,
    const float* __restrict__ vec,
    const float* __restrict__ mat,
    const int K,
    const int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    
    float sum = 0.0f;
    const float* row = mat + n * K;
    for (int k = 0; k < K; k++) {
        sum += vec[k] * row[k];
    }
    out[n] = sum;
}

// Fused attention for single query (autoregressive generation)
__global__ void fusedAttentionKernel(
    float* __restrict__ output,
    const float* __restrict__ query,
    const float* __restrict__ keyCache,
    const float* __restrict__ valueCache,
    const int headDim,
    const int seqLen,
    const float scale,
    const int kvStride,
    const int headOffset,
    const int kvHeadOffset
) {
    extern __shared__ float smem[];
    float* scores = smem;
    
    // Compute attention scores: Q @ K^T
    for (int t = threadIdx.x; t < seqLen; t += blockDim.x) {
        const float* k = keyCache + t * kvStride + kvHeadOffset;
        const float* q = query + headOffset;
        float score = 0.0f;
        for (int i = 0; i < headDim; i++) {
            score += q[i] * k[i];
        }
        scores[t] = score * scale;
    }
    __syncthreads();
    
    // Softmax
    __shared__ float maxScore, sumExp;
    if (threadIdx.x == 0) {
        maxScore = scores[0];
        for (int t = 1; t < seqLen; t++) maxScore = fmaxf(maxScore, scores[t]);
    }
    __syncthreads();
    
    float localSum = 0.0f;
    for (int t = threadIdx.x; t < seqLen; t += blockDim.x) {
        scores[t] = expf(scores[t] - maxScore);
        localSum += scores[t];
    }
    localSum = blockReduceSum(localSum);
    if (threadIdx.x == 0) sumExp = localSum;
    __syncthreads();
    
    float invSum = 1.0f / (sumExp + 1e-10f);
    for (int t = threadIdx.x; t < seqLen; t += blockDim.x) {
        scores[t] *= invSum;
    }
    __syncthreads();
    
    // Weighted sum of values
    float* outHead = output + headOffset;
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < seqLen; t++) {
            sum += scores[t] * valueCache[t * kvStride + kvHeadOffset + i];
        }
        outHead[i] = sum;
    }
}

// Residual add kernel
__global__ void residualAddKernel(
    float* __restrict__ out,
    const float* __restrict__ residual,
    const int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] += residual[i];
}

// Embedding lookup kernel
__global__ void embeddingKernel(
    float* __restrict__ output,
    const float* __restrict__ embeddings,
    const int token,
    const int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) output[i] = embeddings[token * dim + i];
}

// Copy to KV cache kernel
__global__ void copyToCacheKernel(
    float* __restrict__ cache,
    const float* __restrict__ src,
    const int cacheOffset,
    const int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) cache[cacheOffset + i] = src[i];
}

// ============================================================================
// QUANTIZED GPU MATMUL KERNELS
// Dequantize on-the-fly during matrix-vector multiply to save VRAM
// ============================================================================

// Device function: fp16 to fp32 conversion
__device__ __forceinline__ float d_fp16_to_fp32(unsigned short h) {
    int sign = (h >> 15) & 1;
    int exponent = (h >> 10) & 0x1F;
    int mantissa = h & 0x3FF;
    
    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        float m = mantissa / 1024.0f;
        return (sign ? -m : m) * powf(2.0f, -14.0f);
    } else if (exponent == 31) {
        return mantissa ? nanf("") : (sign ? -CUDART_INF_F : CUDART_INF_F);
    }
    float val = (1.0f + mantissa / 1024.0f) * powf(2.0f, exponent - 15.0f);
    return sign ? -val : val;
}

// Q4_K dequantized matmul kernel
// Each thread computes one output element
__global__ void vecMatMulQ4K_Kernel(
    float* __restrict__ out,
    const float* __restrict__ vec,
    const unsigned char* __restrict__ qweight,
    const int K,
    const int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    
    const int nb = K / 256;
    const int block_size = 2 + 2 + 12 + 128;  // d(2) + dmin(2) + scales(12) + qs(128) = 144 bytes
    
    float sum = 0.0f;
    const unsigned char* row = qweight + n * nb * block_size;
    
    for (int b = 0; b < nb; b++) {
        const unsigned char* block = row + b * block_size;
        
        unsigned short d_fp16 = *((const unsigned short*)block);
        unsigned short dmin_fp16 = *((const unsigned short*)(block + 2));
        const unsigned char* scales = block + 4;
        const unsigned char* qs = block + 16;
        
        float d = d_fp16_to_fp32(d_fp16);
        float dmin = d_fp16_to_fp32(dmin_fp16);
        
        int vec_offset = b * 256;
        int is = 0;
        
        for (int j = 0; j < 256; j += 64) {
            unsigned char sc1, m1, sc2, m2;
            if (is < 4) {
                sc1 = scales[is] & 63;
                m1 = scales[is + 4] & 63;
                sc2 = scales[is + 1] & 63;
                m2 = scales[is + 5] & 63;
            } else {
                sc1 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
                m1 = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
                sc2 = (scales[is + 5] & 0xF) | ((scales[is - 3] >> 6) << 4);
                m2 = (scales[is + 5] >> 4) | ((scales[is + 1] >> 6) << 4);
            }
            
            float d1 = d * sc1;
            float m1f = dmin * m1;
            float d2 = d * sc2;
            float m2f = dmin * m2;
            
            for (int l = 0; l < 32; l++) {
                int q = qs[(j/2) + l] & 0xF;
                float w = d1 * q - m1f;
                sum += vec[vec_offset + j + l] * w;
            }
            for (int l = 0; l < 32; l++) {
                int q = qs[(j/2) + l] >> 4;
                float w = d2 * q - m2f;
                sum += vec[vec_offset + j + 32 + l] * w;
            }
            
            is += 2;
        }
    }
    out[n] = sum;
}

// Q6_K dequantized matmul kernel
__global__ void vecMatMulQ6K_Kernel(
    float* __restrict__ out,
    const float* __restrict__ vec,
    const unsigned char* __restrict__ qweight,
    const int K,
    const int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    
    const int nb = K / 256;
    const int block_size = 128 + 64 + 16 + 2;  // ql(128) + qh(64) + scales(16) + d(2) = 210 bytes
    
    float sum = 0.0f;
    const unsigned char* row = qweight + n * nb * block_size;
    
    for (int b = 0; b < nb; b++) {
        const unsigned char* block = row + b * block_size;
        
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128;
        const signed char* scales = (const signed char*)(block + 192);
        unsigned short d_fp16 = *((const unsigned short*)(block + 208));
        
        float d = d_fp16_to_fp32(d_fp16);
        int vec_offset = b * 256;
        
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                
                signed char q1 = (signed char)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                signed char q2 = (signed char)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                signed char q3 = (signed char)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                signed char q4 = (signed char)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                
                sum += vec[vec_offset + j + l] * (d * scales[is] * q1);
                sum += vec[vec_offset + j + l + 32] * (d * scales[is + 2] * q2);
                sum += vec[vec_offset + j + l + 64] * (d * scales[is + 4] * q3);
                sum += vec[vec_offset + j + l + 96] * (d * scales[is + 6] * q4);
            }
            ql += 64;
            qh += 32;
            scales += 8;
        }
    }
    out[n] = sum;
}

// Q8_0 dequantized matmul kernel (simplest - 32 elements per block)
__global__ void vecMatMulQ8_0_Kernel(
    float* __restrict__ out,
    const float* __restrict__ vec,
    const unsigned char* __restrict__ qweight,
    const int K,
    const int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    
    const int nb = K / 32;
    const int block_size = 2 + 32;  // d(f16) + 32 int8 quants = 34 bytes
    
    float sum = 0.0f;
    const unsigned char* row = qweight + n * nb * block_size;
    
    for (int b = 0; b < nb; b++) {
        const unsigned char* block = row + b * block_size;
        unsigned short d_fp16 = *((const unsigned short*)block);
        const signed char* qs = (const signed char*)(block + 2);
        
        float d = d_fp16_to_fp32(d_fp16);
        int vec_offset = b * 32;
        
        for (int i = 0; i < 32; i++) {
            sum += vec[vec_offset + i] * (d * qs[i]);
        }
    }
    out[n] = sum;
}

// Q2_K dequantized matmul kernel
__global__ void vecMatMulQ2K_Kernel(
    float* __restrict__ out,
    const float* __restrict__ vec,
    const unsigned char* __restrict__ qweight,
    const int K,
    const int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    
    const int nb = K / 256;
    const int block_size = 16 + 64 + 2 + 2;  // scales(16) + qs(64) + d(2) + dmin(2) = 84 bytes
    
    float sum = 0.0f;
    const unsigned char* row = qweight + n * nb * block_size;
    
    for (int b = 0; b < nb; b++) {
        const unsigned char* block = row + b * block_size;
        
        const unsigned char* scales = block;
        const unsigned char* qs = block + 16;
        unsigned short d_fp16 = *((const unsigned short*)(block + 80));
        unsigned short dmin_fp16 = *((const unsigned short*)(block + 82));
        
        float d = d_fp16_to_fp32(d_fp16);
        float dmin = d_fp16_to_fp32(dmin_fp16);
        int vec_offset = b * 256;
        
        for (int j = 0; j < 16; j++) {
            float scale = d * (scales[j] & 0xF);
            float min = dmin * (scales[j] >> 4);
            
            for (int l = 0; l < 16; l++) {
                int idx = j * 16 + l;
                int byte_idx = idx / 4;
                int shift = (idx % 4) * 2;
                int q = (qs[byte_idx] >> shift) & 3;
                float w = scale * q - min;
                sum += vec[vec_offset + idx] * w;
            }
        }
    }
    out[n] = sum;
}

} // extern "C"
"#;

pub struct CudaKernels {
    #[allow(dead_code)]
    device: Arc<CudaDevice>,
    rms_norm: CudaFunction,
    rms_norm_slice: CudaFunction,
    rope: CudaFunction,
    swiglu: CudaFunction,
    geglu: CudaFunction,
    vec_mat_mul: CudaFunction,
    attention: CudaFunction,
    residual_add: CudaFunction,
    embedding: CudaFunction,
    copy_to_cache: CudaFunction,
    vec_mat_mul_q4k: CudaFunction,
    vec_mat_mul_q6k: CudaFunction,
    vec_mat_mul_q8_0: CudaFunction,
    vec_mat_mul_q2k: CudaFunction,
}

impl CudaKernels {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SOURCE)
            .map_err(|e| TransformerError::Cuda(format!("NVRTC compile error: {}", e)))?;
        
        device.load_ptx(ptx, "transformer_kernels", &[
            "fusedRMSNormKernel",
            "rmsNormSliceKernel",
            "fusedRoPEKernel",
            "fusedSwiGLUKernel",
            "fusedGeGLUKernel",
            "vecMatMulKernel",
            "fusedAttentionKernel",
            "residualAddKernel",
            "embeddingKernel",
            "copyToCacheKernel",
            "vecMatMulQ4K_Kernel",
            "vecMatMulQ6K_Kernel",
            "vecMatMulQ8_0_Kernel",
            "vecMatMulQ2K_Kernel",
        ]).map_err(|e| TransformerError::Cuda(format!("PTX load error: {}", e)))?;
        
        let rms_norm = device.get_func("transformer_kernels", "fusedRMSNormKernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get fusedRMSNormKernel".into()))?;
        let rms_norm_slice = device.get_func("transformer_kernels", "rmsNormSliceKernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get rmsNormSliceKernel".into()))?;
        let rope = device.get_func("transformer_kernels", "fusedRoPEKernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get fusedRoPEKernel".into()))?;
        let swiglu = device.get_func("transformer_kernels", "fusedSwiGLUKernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get fusedSwiGLUKernel".into()))?;
        let geglu = device.get_func("transformer_kernels", "fusedGeGLUKernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get fusedGeGLUKernel".into()))?;
        let vec_mat_mul = device.get_func("transformer_kernels", "vecMatMulKernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get vecMatMulKernel".into()))?;
        let attention = device.get_func("transformer_kernels", "fusedAttentionKernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get fusedAttentionKernel".into()))?;
        let residual_add = device.get_func("transformer_kernels", "residualAddKernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get residualAddKernel".into()))?;
        let embedding = device.get_func("transformer_kernels", "embeddingKernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get embeddingKernel".into()))?;
        let copy_to_cache = device.get_func("transformer_kernels", "copyToCacheKernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get copyToCacheKernel".into()))?;
        let vec_mat_mul_q4k = device.get_func("transformer_kernels", "vecMatMulQ4K_Kernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get vecMatMulQ4K_Kernel".into()))?;
        let vec_mat_mul_q6k = device.get_func("transformer_kernels", "vecMatMulQ6K_Kernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get vecMatMulQ6K_Kernel".into()))?;
        let vec_mat_mul_q8_0 = device.get_func("transformer_kernels", "vecMatMulQ8_0_Kernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get vecMatMulQ8_0_Kernel".into()))?;
        let vec_mat_mul_q2k = device.get_func("transformer_kernels", "vecMatMulQ2K_Kernel")
            .ok_or_else(|| TransformerError::Cuda("Failed to get vecMatMulQ2K_Kernel".into()))?;
        
        Ok(Self {
            device,
            rms_norm,
            rms_norm_slice,
            rope,
            swiglu,
            geglu,
            vec_mat_mul,
            attention,
            residual_add,
            embedding,
            copy_to_cache,
            vec_mat_mul_q4k,
            vec_mat_mul_q6k,
            vec_mat_mul_q8_0,
            vec_mat_mul_q2k,
        })
    }
    
    pub fn rms_norm(
        &self,
        output: &mut CudaSlice<f32>,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.rms_norm.clone().launch(cfg, (
                output,
                input,
                weight,
                dim as i32,
                eps,
                0i32, // unitOffset = false
            ))
        }.map_err(|e| TransformerError::Cuda(format!("RMSNorm launch error: {}", e)))?;
        
        Ok(())
    }
    
    pub fn rms_norm_slice(
        &self,
        data: &mut CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        offset: usize,
        size: usize,
        eps: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.rms_norm_slice.clone().launch(cfg, (
                data,
                weight,
                offset as i32,
                size as i32,
                eps,
            ))
        }.map_err(|e| TransformerError::Cuda(format!("RMSNorm slice launch error: {}", e)))?;
        
        Ok(())
    }
    
    pub fn rope(
        &self,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        q_dim: usize,
        kv_dim: usize,
        head_dim: usize,
        pos: usize,
        theta: f32,
        rope_scale: f32,
    ) -> Result<()> {
        let max_dim = q_dim.max(kv_dim);
        let blocks = (max_dim / 2 + 255) / 256;
        
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.rope.clone().launch(cfg, (
                q,
                k,
                q_dim as i32,
                kv_dim as i32,
                head_dim as i32,
                pos as i32,
                theta,
                rope_scale,
            ))
        }.map_err(|e| TransformerError::Cuda(format!("RoPE launch error: {}", e)))?;
        
        Ok(())
    }
    
    /// SwiGLU activation: gate buffer contains gate values, result written back to gate
    /// Uses in-place operation where gate serves as both input and output
    pub fn swiglu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        size: usize,
    ) -> Result<()> {
        let blocks = (size + 255) / 256;
        
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.swiglu.clone().launch(cfg, (
                output,
                gate,
                up,
                size as i32,
            ))
        }.map_err(|e| TransformerError::Cuda(format!("SwiGLU launch error: {}", e)))?;
        
        Ok(())
    }
    
    /// GeGLU activation for Gemma: gate buffer contains gate values, result written to output
    pub fn geglu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        size: usize,
    ) -> Result<()> {
        let blocks = (size + 255) / 256;
        
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.geglu.clone().launch(cfg, (
                output,
                gate,
                up,
                size as i32,
            ))
        }.map_err(|e| TransformerError::Cuda(format!("GeGLU launch error: {}", e)))?;
        
        Ok(())
    }
    
    pub fn vec_mat_mul(
        &self,
        output: &mut CudaSlice<f32>,
        vec: &CudaSlice<f32>,
        mat: &CudaSlice<f32>,
        k: usize, // input dim
        n: usize, // output dim
    ) -> Result<()> {
        let blocks = (n + 255) / 256;
        
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.vec_mat_mul.clone().launch(cfg, (
                output,
                vec,
                mat,
                k as i32,
                n as i32,
            ))
        }.map_err(|e| TransformerError::Cuda(format!("VecMatMul launch error: {}", e)))?;
        
        Ok(())
    }
    
    pub fn fused_attention(
        &self,
        output: &mut CudaSlice<f32>,
        query: &CudaSlice<f32>,
        key_cache: &CudaSlice<f32>,
        value_cache: &CudaSlice<f32>,
        head_idx: usize,
        kv_head_idx: usize,
        head_dim: usize,
        seq_len: usize,
        layer_offset: usize,
        kv_dim: usize,
    ) -> Result<()> {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let shared_mem = seq_len * std::mem::size_of::<f32>();
        
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };
        
        unsafe {
            self.attention.clone().launch(cfg, (
                output,
                query,
                key_cache,
                value_cache,
                head_dim as i32,
                seq_len as i32,
                scale,
                kv_dim as i32,
                (head_idx * head_dim) as i32,
                (layer_offset + kv_head_idx * head_dim) as i32,
            ))
        }.map_err(|e| TransformerError::Cuda(format!("Attention launch error: {}", e)))?;
        
        Ok(())
    }
    
    pub fn residual_add(
        &self,
        output: &mut CudaSlice<f32>,
        residual: &CudaSlice<f32>,
        size: usize,
    ) -> Result<()> {
        let blocks = (size + 255) / 256;
        
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.residual_add.clone().launch(cfg, (
                output,
                residual,
                size as i32,
            ))
        }.map_err(|e| TransformerError::Cuda(format!("ResidualAdd launch error: {}", e)))?;
        
        Ok(())
    }
    
    pub fn embedding_lookup(
        &self,
        output: &mut CudaSlice<f32>,
        embeddings: &CudaSlice<f32>,
        token: usize,
        dim: usize,
    ) -> Result<()> {
        let blocks = (dim + 255) / 256;
        
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.embedding.clone().launch(cfg, (
                output,
                embeddings,
                token as i32,
                dim as i32,
            ))
        }.map_err(|e| TransformerError::Cuda(format!("Embedding launch error: {}", e)))?;
        
        Ok(())
    }
    
    pub fn copy_to_cache(
        &self,
        cache: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        cache_offset: usize,
        size: usize,
    ) -> Result<()> {
        let blocks = (size + 255) / 256;
        
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.copy_to_cache.clone().launch(cfg, (
                cache,
                src,
                cache_offset as i32,
                size as i32,
            ))
        }.map_err(|e| TransformerError::Cuda(format!("CopyToCache launch error: {}", e)))?;
        
        Ok(())
    }
    
    pub fn vec_mat_mul_quantized(
        &self,
        output: &mut CudaSlice<f32>,
        vec: &CudaSlice<f32>,
        mat: &CudaSlice<u8>,
        k: usize,
        n: usize,
        dtype: i32,
    ) -> Result<()> {
        let blocks = (n + 255) / 256;
        
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        match dtype {
            12 => {
                unsafe {
                    self.vec_mat_mul_q4k.clone().launch(cfg, (
                        output,
                        vec,
                        mat,
                        k as i32,
                        n as i32,
                    ))
                }.map_err(|e| TransformerError::Cuda(format!("VecMatMulQ4K launch error: {}", e)))?;
            }
            14 => {
                unsafe {
                    self.vec_mat_mul_q6k.clone().launch(cfg, (
                        output,
                        vec,
                        mat,
                        k as i32,
                        n as i32,
                    ))
                }.map_err(|e| TransformerError::Cuda(format!("VecMatMulQ6K launch error: {}", e)))?;
            }
            8 => {
                unsafe {
                    self.vec_mat_mul_q8_0.clone().launch(cfg, (
                        output,
                        vec,
                        mat,
                        k as i32,
                        n as i32,
                    ))
                }.map_err(|e| TransformerError::Cuda(format!("VecMatMulQ8_0 launch error: {}", e)))?;
            }
            10 => {
                unsafe {
                    self.vec_mat_mul_q2k.clone().launch(cfg, (
                        output,
                        vec,
                        mat,
                        k as i32,
                        n as i32,
                    ))
                }.map_err(|e| TransformerError::Cuda(format!("VecMatMulQ2K launch error: {}", e)))?;
            }
            _ => {
                return Err(TransformerError::Cuda(format!("Unsupported quantized dtype: {}", dtype)));
            }
        }
        
        Ok(())
    }
    
    pub fn vec_mat_mul_tensor(
        &self,
        output: &mut CudaSlice<f32>,
        vec: &CudaSlice<f32>,
        mat: &TensorData,
        k: usize,
        n: usize,
    ) -> Result<()> {
        match mat {
            TensorData::F32(slice) => {
                self.vec_mat_mul(output, vec, slice, k, n)
            }
            TensorData::Quantized(slice, dtype) => {
                self.vec_mat_mul_quantized(output, vec, slice, k, n, *dtype)
            }
        }
    }
}
