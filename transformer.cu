/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>
#include <iomanip>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

constexpr int MAX_SEQ_LEN = 1024;
constexpr const char* GGUF_MAGIC = "GGUF";
constexpr int BLOCK_SIZE = 256;

// ==================== Quantization Type Registry ====================

enum class GGML_DType : int {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q6_K = 12,
    Q4_K = 13,
    Q5_K = 14,
    BFLOAT16 = 30,
    UNKNOWN = -1
};

struct QuantTypeInfo {
    GGML_DType dtype;
    const char* name;
    int bitsPerElement;
    int blockSize;
    int groupSize;
    bool supported;
};

const QuantTypeInfo QUANT_TYPES[] = {
    {GGML_DType::F32, "F32", 32, 1, 1, true},
    {GGML_DType::F16, "F16", 16, 1, 1, true},
    {GGML_DType::Q4_0, "Q4_0", 4, 32, 32, true},
    {GGML_DType::Q4_1, "Q4_1", 4, 32, 32, true},
    {GGML_DType::Q5_0, "Q5_0", 5, 32, 32, true},
    {GGML_DType::Q5_1, "Q5_1", 5, 32, 32, true},
    {GGML_DType::Q8_0, "Q8_0", 8, 32, 32, true},
    {GGML_DType::Q2_K, "Q2_K", 2, 256, 128, true},
    {GGML_DType::Q3_K, "Q3_K", 3, 256, 128, true},
    {GGML_DType::Q6_K, "Q6_K", 6, 256, 128, true},
    {GGML_DType::Q4_K, "Q4_K", 4, 256, 128, true},
    {GGML_DType::Q5_K, "Q5_K", 5, 256, 128, true},
    {GGML_DType::BFLOAT16, "BFLOAT16", 16, 1, 1, true},
};

const char* getQuantTypeName(GGML_DType dtype) {
    for (const auto& qt : QUANT_TYPES) {
        if (qt.dtype == dtype) return qt.name;
    }
    return "UNKNOWN";
}

bool isQuantTypeSupported(GGML_DType dtype) {
    for (const auto& qt : QUANT_TYPES) {
        if (qt.dtype == dtype) return qt.supported;
    }
    return false;
}

// ==================== Quantization Stats ====================

struct QuantizationStats {
    std::map<std::string, int> typeFrequency;
    std::map<std::string, int64_t> originalSize;
    std::map<std::string, int64_t> compressedSize;
    int64_t totalOriginal = 0;
    int64_t totalCompressed = 0;
    
    void add(const std::string& typeName, int64_t originalSize, int64_t compressedSize) {
        typeFrequency[typeName]++;
        this->originalSize[typeName] += originalSize;
        this->compressedSize[typeName] += compressedSize;
        this->totalOriginal += originalSize;
        this->totalCompressed += compressedSize;
    }
    
    void print() const {
        std::cout << "\n=== Quantization Summary ===" << std::endl;
        std::cout << "Total original size: " << (totalOriginal / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Total compressed size: " << (totalCompressed / 1024.0 / 1024.0) << " MB" << std::endl;
        
        if (totalOriginal > 0) {
            double ratio = (double)totalCompressed / totalOriginal;
            double speedup = 1.0 / ratio;
            std::cout << "Overall compression ratio: " << std::fixed << std::setprecision(2) 
                      << ratio << "x" << std::endl;
            std::cout << "Theoretical speedup: " << speedup << "x" << std::endl;
        }
        
        std::cout << "\nBreakdown by type:" << std::endl;
        for (const auto& entry : typeFrequency) {
            const std::string& type = entry.first;
            int count = entry.second;
            if (originalSize.count(type)) {
                int64_t orig = originalSize.at(type);
                int64_t comp = compressedSize.at(type);
                double ratio = (orig > 0) ? (double)comp / orig : 0;
                std::cout << "  " << type << ": " << count << " tensors, "
                          << (orig / 1024.0 / 1024.0) << " MB -> "
                          << (comp / 1024.0 / 1024.0) << " MB (" 
                          << std::fixed << std::setprecision(2) << ratio << "x)" << std::endl;
            }
        }
    }
};

// ==================== CUDA Kernels ====================

__global__ void matmulKernel(const float* A, const float* B, float* C,
                              int M, int N, int K, const float* bias) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = (bias != nullptr) ? bias[col] : 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmulTransposeKernel(const float* A, const float* B, float* C,
                                       int M, int N, int K, const float* bias) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = (bias != nullptr) ? bias[col] : 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void geluKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        data[idx] = x * cdf;
    }
}

__global__ void layerNormKernel(const float* input, float* output,
                                 const float* gamma, const float* beta,
                                 int seqLen, int dim) {
    int pos = blockIdx.x;
    if (pos >= seqLen) return;
    
    extern __shared__ float shared[];
    float* sdata = shared;
    
    int tid = threadIdx.x;
    int offset = pos * dim;
    
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += input[offset + i];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / dim;
    __syncthreads();
    
    sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = input[offset + i] - mean;
        sum += diff * diff;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float variance = sdata[0] / dim;
    float invStd = rsqrtf(variance + 1e-5f);
    
    for (int i = tid; i < dim; i += blockDim.x) {
        float normalized = (input[offset + i] - mean) * invStd;
        float g = (gamma != nullptr) ? gamma[i] : 1.0f;
        float b = (beta != nullptr) ? beta[i] : 0.0f;
        output[offset + i] = normalized * g + b;
    }
}

__global__ void softmaxKernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int offset = row * cols;
    
    float maxVal = -1e30f;
    for (int i = tid; i < cols; i += blockDim.x) {
        maxVal = fmaxf(maxVal, data[offset + i]);
    }
    shared[tid] = maxVal;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    maxVal = shared[0];
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(data[offset + i] - maxVal);
        data[offset + i] = val;
        sum += val;
    }
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    sum = shared[0];
    
    for (int i = tid; i < cols; i += blockDim.x) {
        data[offset + i] /= sum;
    }
}

__global__ void addResidualKernel(float* output, const float* residual, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += residual[idx];
    }
}

__global__ void embedTokensKernel(const int* tokenIDs, const float* tokenEmb,
                                   const float* posEmb, float* output,
                                   int seqLen, int embedDim) {
    int pos = blockIdx.x;
    int i = threadIdx.x;
    
    if (pos < seqLen && i < embedDim) {
        int tokenID = tokenIDs[pos];
        output[pos * embedDim + i] = tokenEmb[tokenID * embedDim + i] + posEmb[pos * embedDim + i];
    }
}

__global__ void computeQKVKernel(const float* normInput, const float* weight, const float* bias,
                                  float* Q, float* K, float* V,
                                  int seqLen, int embedDim) {
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < seqLen && i < embedDim) {
        int offset = pos * embedDim;
        
        float sumQ = (bias != nullptr) ? bias[i] : 0.0f;
        float sumK = (bias != nullptr) ? bias[embedDim + i] : 0.0f;
        float sumV = (bias != nullptr) ? bias[2 * embedDim + i] : 0.0f;
        
        for (int j = 0; j < embedDim; j++) {
            float inp = normInput[offset + j];
            sumQ += inp * weight[i * embedDim + j];
            sumK += inp * weight[(embedDim + i) * embedDim + j];
            sumV += inp * weight[(2 * embedDim + i) * embedDim + j];
        }
        
        Q[offset + i] = sumQ;
        K[offset + i] = sumK;
        V[offset + i] = sumV;
    }
}

// RoPE (Rotary Position Embedding) kernel
__global__ void applyRoPEKernel(float* Q, float* K, int seqLen, int numHeads, int headDim) {
    int pos = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x * 2; // Process 2 elements at a time (pairs)
    
    if (pos < seqLen && h < numHeads && i + 1 < headDim) {
        float theta = powf(10000.0f, -2.0f * i / headDim);
        float m = (float)pos;
        float angle = m * theta;
        float cosAngle = cosf(angle);
        float sinAngle = sinf(angle);
        
        int headStart = h * headDim;
        int qIdx = pos * (numHeads * headDim) + headStart + i;
        int kIdx = pos * (numHeads * headDim) + headStart + i;
        
        // Apply rotation to Q and K for this head pair
        float q0 = Q[qIdx];
        float q1 = Q[qIdx + 1];
        float k0 = K[kIdx];
        float k1 = K[kIdx + 1];
        
        Q[qIdx] = q0 * cosAngle - q1 * sinAngle;
        Q[qIdx + 1] = q0 * sinAngle + q1 * cosAngle;
        K[kIdx] = k0 * cosAngle - k1 * sinAngle;
        K[kIdx + 1] = k0 * sinAngle + k1 * cosAngle;
    }
}

__global__ void attentionScoresKernel(const float* Q, const float* K, float* scores,
                                       int seqLen, int numHeads, int headDim, float scale) {
    int h = blockIdx.z;
    int pos = blockIdx.y;
    int srcPos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < numHeads && pos < seqLen && srcPos < seqLen) {
        if (srcPos > pos) {
            scores[h * seqLen * seqLen + pos * seqLen + srcPos] = -1e9f;
        } else {
            int headStart = h * headDim;
            float sum = 0.0f;
            for (int i = 0; i < headDim; i++) {
                sum += Q[pos * (numHeads * headDim) + headStart + i] *
                       K[srcPos * (numHeads * headDim) + headStart + i];
            }
            scores[h * seqLen * seqLen + pos * seqLen + srcPos] = sum / scale;
        }
    }
}

// Grouped Query Attention kernel (MQA/GQA support)
__global__ void attentionScoresGQAKernel(const float* Q, const float* K, float* scores,
                                         int seqLen, int numHeads, int numKVHeads, int headDim, float scale) {
    int h = blockIdx.z;
    int pos = blockIdx.y;
    int srcPos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < numHeads && pos < seqLen && srcPos < seqLen) {
        if (srcPos > pos) {
            scores[h * seqLen * seqLen + pos * seqLen + srcPos] = -1e9f;
        } else {
            // Map query head to KV head for grouped attention
            int kvHeadIdx = h * numKVHeads / numHeads;
            int headStart = kvHeadIdx * headDim;
            
            float sum = 0.0f;
            for (int i = 0; i < headDim; i++) {
                sum += Q[pos * (numHeads * headDim) + h * headDim + i] *
                       K[srcPos * (numKVHeads * headDim) + headStart + i];
            }
            scores[h * seqLen * seqLen + pos * seqLen + srcPos] = sum / scale;
        }
    }
}

__global__ void attentionOutputGQAKernel(const float* attnWeights, const float* V, float* output,
                                         int seqLen, int numHeads, int numKVHeads, int headDim) {
    int h = blockIdx.z;
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < numHeads && pos < seqLen && i < headDim) {
        int kvHeadIdx = h * numKVHeads / numHeads;
        float sum = 0.0f;
        for (int srcPos = 0; srcPos < seqLen; srcPos++) {
            sum += attnWeights[h * seqLen * seqLen + pos * seqLen + srcPos] *
                   V[srcPos * (numKVHeads * headDim) + kvHeadIdx * headDim + i];
        }
        output[pos * (numHeads * headDim) + h * headDim + i] = sum;
    }
}

__global__ void attentionOutputKernel(const float* attnWeights, const float* V, float* output,
                                       int seqLen, int numHeads, int headDim) {
    int h = blockIdx.z;
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < numHeads && pos < seqLen && i < headDim) {
        int headStart = h * headDim;
        float sum = 0.0f;
        for (int srcPos = 0; srcPos < seqLen; srcPos++) {
            sum += attnWeights[h * seqLen * seqLen + pos * seqLen + srcPos] *
                   V[srcPos * (numHeads * headDim) + headStart + i];
        }
        output[pos * (numHeads * headDim) + headStart + i] = sum;
    }
}

__global__ void projectionKernel(const float* input, const float* weight, const float* bias,
                                  float* output, const float* residual,
                                  int seqLen, int embedDim) {
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < seqLen && i < embedDim) {
        float sum = (bias != nullptr) ? bias[i] : 0.0f;
        for (int j = 0; j < embedDim; j++) {
            sum += input[pos * embedDim + j] * weight[i * embedDim + j];
        }
        output[pos * embedDim + i] = residual[pos * embedDim + i] + sum;
    }
}

// GELU activation kernel (GPT-2 style)
__global__ void ffnUpGELUKernel(const float* input, const float* weight, const float* bias,
                                float* output, int seqLen, int embedDim, int ffnDim) {
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < seqLen && i < ffnDim) {
        float sum = (bias != nullptr) ? bias[i] : 0.0f;
        for (int j = 0; j < embedDim; j++) {
            sum += input[pos * embedDim + j] * weight[i * embedDim + j];
        }
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
        output[pos * ffnDim + i] = sum * cdf;
    }
}

// SwiGLU activation kernel (LLaMA style) - requires gate and up projections
__global__ void ffnUpSwiGLUKernel(const float* input, const float* weightUp, const float* biasUp,
                                  const float* weightGate, const float* biasGate,
                                  float* output, int seqLen, int embedDim, int ffnDim) {
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < seqLen && i < ffnDim) {
        // Compute up projection (linear transformation)
        float upVal = (biasUp != nullptr) ? biasUp[i] : 0.0f;
        for (int j = 0; j < embedDim; j++) {
            upVal += input[pos * embedDim + j] * weightUp[i * embedDim + j];
        }
        
        // Compute gate projection (linear transformation)
        float gateVal = (biasGate != nullptr) ? biasGate[i] : 0.0f;
        for (int j = 0; j < embedDim; j++) {
            gateVal += input[pos * embedDim + j] * weightGate[i * embedDim + j];
        }
        
        // SwiGLU: swish(gate) * up = (gate * sigmoid(gate)) * up
        float sigmoid = 1.0f / (1.0f + expf(-gateVal));
        float swish = gateVal * sigmoid;
        
        output[pos * ffnDim + i] = upVal * swish;
    }
}

__global__ void ffnDownKernel(const float* input, const float* weight, const float* bias,
                               float* output, const float* residual,
                               int seqLen, int ffnDim, int embedDim) {
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < seqLen && i < embedDim) {
        float sum = (bias != nullptr) ? bias[i] : 0.0f;
        for (int j = 0; j < ffnDim; j++) {
            sum += input[pos * ffnDim + j] * weight[i * ffnDim + j];
        }
        output[pos * embedDim + i] = residual[pos * embedDim + i] + sum;
    }
}

__global__ void computeLogitsKernel(const float* hidden, const float* tokenEmb,
                                      float* logits, int embedDim, int vocabSize) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (i < vocabSize) {
         float sum = 0.0f;
         for (int j = 0; j < embedDim; j++) {
             sum += hidden[j] * tokenEmb[i * embedDim + j];
         }
         logits[i] = sum;
     }
 }

// ==================== GPU Dequantization Kernels ====================

// Q4_0: 32 elements per block, 4-bit quantization
__global__ void dequantizeQ4_0Kernel(const uint8_t* quantized, float* output, int64_t numElements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groupSize = 32;
    const int blockSize = 18; // 2 bytes scale + 16 bytes quantized
    
    if (idx < numElements) {
        int64_t groupIdx = idx / groupSize;
        int posInGroup = idx % groupSize;
        
        const uint8_t* block = quantized + groupIdx * blockSize;
        uint16_t scale16;
        memcpy(&scale16, block, 2);
        
        // Decode float16
        int sign = (scale16 >> 15) & 1;
        int exponent = (scale16 >> 10) & 0x1F;
        int mantissa = scale16 & 0x3FF;
        float scale;
        
        if (exponent == 0) {
            scale = 0.0f;
        } else if (exponent == 31) {
            scale = sign ? -1e10f : 1e10f;
        } else {
            float base = powf(2.0f, (float)(exponent - 15));
            scale = base * (1.0f + mantissa / 1024.0f);
            if (sign) scale = -scale;
        }
        
        int byteIdx = 2 + posInGroup / 2;
        int nibbleIdx = posInGroup % 2;
        uint8_t quantVal = (block[byteIdx] >> (nibbleIdx * 4)) & 0x0F;
        int8_t signedVal = (int8_t)quantVal - 8;
        
        output[idx] = (float)signedVal * scale;
    }
}

// Q4_1: 32 elements per block, 4-bit quantization with min/max
__global__ void dequantizeQ4_1Kernel(const uint8_t* quantized, float* output, int64_t numElements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groupSize = 32;
    const int blockSize = 20; // 2 scale + 2 min + 16 quantized
    
    if (idx < numElements) {
        int64_t groupIdx = idx / groupSize;
        int posInGroup = idx % groupSize;
        
        const uint8_t* block = quantized + groupIdx * blockSize;
        uint16_t scale16, min16;
        memcpy(&scale16, block, 2);
        memcpy(&min16, block + 2, 2);
        
        auto f16tof32 = [](uint16_t h) -> float {
            int sign = (h >> 15) & 1;
            int exponent = (h >> 10) & 0x1F;
            int mantissa = h & 0x3FF;
            if (exponent == 0) return 0.0f;
            if (exponent == 31) return sign ? -1e10f : 1e10f;
            float val = powf(2.0f, (float)exponent - 15.0f) * (1.0f + mantissa / 1024.0f);
            return sign ? -val : val;
        };
        
        float scale = f16tof32(scale16);
        float minVal = f16tof32(min16);
        
        int byteIdx = 4 + posInGroup / 2;
        int nibbleIdx = posInGroup % 2;
        uint8_t quantVal = (block[byteIdx] >> (nibbleIdx * 4)) & 0x0F;
        
        output[idx] = minVal + (float)quantVal * scale;
    }
}

// Q8_0: 32 elements per block, 8-bit quantization
__global__ void dequantizeQ8_0Kernel(const uint8_t* quantized, float* output, int64_t numElements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groupSize = 32;
    const int blockSize = 18; // 2 scale + 16 quantized
    
    if (idx < numElements) {
        int64_t groupIdx = idx / groupSize;
        int posInGroup = idx % groupSize;
        
        const uint8_t* block = quantized + groupIdx * blockSize;
        uint16_t scale16;
        memcpy(&scale16, block, 2);
        
        int sign = (scale16 >> 15) & 1;
        int exponent = (scale16 >> 10) & 0x1F;
        int mantissa = scale16 & 0x3FF;
        float scale;
        
        if (exponent == 0) {
            scale = 0.0f;
        } else if (exponent == 31) {
            scale = sign ? -1e10f : 1e10f;
        } else {
            float base = powf(2.0f, (float)(exponent - 15));
            scale = base * (1.0f + mantissa / 1024.0f);
            if (sign) scale = -scale;
        }
        
        int8_t quantVal = (int8_t)block[2 + posInGroup];
        output[idx] = (float)quantVal * scale;
    }
}

// Q5_0: 32 elements per block, 5-bit quantization
__global__ void dequantizeQ5_0Kernel(const uint8_t* quantized, float* output, int64_t numElements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groupSize = 32;
    const int blockSize = 22; // 2 scale + 4 upper bits + 16 lower bits
    
    if (idx < numElements) {
        int64_t groupIdx = idx / groupSize;
        int posInGroup = idx % groupSize;
        
        const uint8_t* block = quantized + groupIdx * blockSize;
        uint16_t scale16;
        memcpy(&scale16, block, 2);
        
        int sign = (scale16 >> 15) & 1;
        int exponent = (scale16 >> 10) & 0x1F;
        int mantissa = scale16 & 0x3FF;
        float scale;
        
        if (exponent == 0) {
            scale = 0.0f;
        } else if (exponent == 31) {
            scale = sign ? -1e10f : 1e10f;
        } else {
            float base = powf(2.0f, (float)(exponent - 15));
            scale = base * (1.0f + mantissa / 1024.0f);
            if (sign) scale = -scale;
        }
        
        uint32_t upperBits = 0;
        memcpy(&upperBits, block + 2, 4);
        
        uint8_t lowerBits = block[6 + posInGroup / 2];
        int nibbleIdx = posInGroup % 2;
        uint8_t lower4bits = (lowerBits >> (nibbleIdx * 4)) & 0x0F;
        
        int upperBit = (upperBits >> posInGroup) & 1;
        int quantVal = lower4bits | (upperBit << 4);
        
        int8_t signedVal = (int8_t)quantVal - 16;
        output[idx] = (float)signedVal * scale;
    }
}

// Q5_1: 32 elements per block, 5-bit with min/max
__global__ void dequantizeQ5_1Kernel(const uint8_t* quantized, float* output, int64_t numElements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groupSize = 32;
    const int blockSize = 24; // 2 scale + 2 min + 4 upper + 16 lower
    
    if (idx < numElements) {
        int64_t groupIdx = idx / groupSize;
        int posInGroup = idx % groupSize;
        
        const uint8_t* block = quantized + groupIdx * blockSize;
        uint16_t scale16, min16;
        memcpy(&scale16, block, 2);
        memcpy(&min16, block + 2, 2);
        
        auto f16tof32 = [](uint16_t h) -> float {
            int sign = (h >> 15) & 1;
            int exponent = (h >> 10) & 0x1F;
            int mantissa = h & 0x3FF;
            if (exponent == 0) return 0.0f;
            if (exponent == 31) return sign ? -1e10f : 1e10f;
            float val = powf(2.0f, (float)exponent - 15.0f) * (1.0f + mantissa / 1024.0f);
            return sign ? -val : val;
        };
        
        float scale = f16tof32(scale16);
        float minVal = f16tof32(min16);
        
        uint32_t upperBits = 0;
        memcpy(&upperBits, block + 4, 4);
        
        uint8_t lowerBits = block[8 + posInGroup / 2];
        int nibbleIdx = posInGroup % 2;
        uint8_t lower4bits = (lowerBits >> (nibbleIdx * 4)) & 0x0F;
        
        int upperBit = (upperBits >> posInGroup) & 1;
        uint8_t quantVal = lower4bits | (upperBit << 4);
        
        output[idx] = minVal + (float)quantVal * scale;
    }
}

// Q2_K: 256 elements per block, 2-bit quantization with K-quant structure
__global__ void dequantizeQ2_KKernel(const uint8_t* quantized, float* output, int64_t numElements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockSize = 256;
    
    if (idx < numElements) {
        int64_t blockIdx = idx / blockSize;
        int posInBlock = idx % blockSize;
        
        // K-quant block structure: scales(2) + mins(2) + 32 bytes data
        const int qlBlockSize = 36; // 2+2+32
        const uint8_t* block = quantized + blockIdx * qlBlockSize;
        
        uint16_t scale16, min16;
        memcpy(&scale16, block, 2);
        memcpy(&min16, block + 2, 2);
        
        auto f16tof32 = [](uint16_t h) -> float {
            int sign = (h >> 15) & 1;
            int exponent = (h >> 10) & 0x1F;
            int mantissa = h & 0x3FF;
            if (exponent == 0) return 0.0f;
            if (exponent == 31) return sign ? -1e10f : 1e10f;
            float val = powf(2.0f, (float)exponent - 15.0f) * (1.0f + mantissa / 1024.0f);
            return sign ? -val : val;
        };
        
        float scale = f16tof32(scale16);
        float minVal = f16tof32(min16);
        
        int byteIdx = 4 + posInBlock / 4;
        int bitIdx = (posInBlock % 4) * 2;
        uint8_t quantVal = (block[byteIdx] >> bitIdx) & 0x3;
        
        output[idx] = minVal + (float)quantVal * scale;
    }
}

// Q3_K: 256 elements per block, 3-bit quantization
__global__ void dequantizeQ3_KKernel(const uint8_t* quantized, float* output, int64_t numElements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockSize = 256;
    
    if (idx < numElements) {
        int64_t blockIdx = idx / blockSize;
        int posInBlock = idx % blockSize;
        
        // K-quant: scales(2) + mins(2) + 96 bytes (256*3/8 = 96)
        const int qlBlockSize = 100;
        const uint8_t* block = quantized + blockIdx * qlBlockSize;
        
        uint16_t scale16, min16;
        memcpy(&scale16, block, 2);
        memcpy(&min16, block + 2, 2);
        
        auto f16tof32 = [](uint16_t h) -> float {
            int sign = (h >> 15) & 1;
            int exponent = (h >> 10) & 0x1F;
            int mantissa = h & 0x3FF;
            if (exponent == 0) return 0.0f;
            if (exponent == 31) return sign ? -1e10f : 1e10f;
            float val = powf(2.0f, (float)exponent - 15.0f) * (1.0f + mantissa / 1024.0f);
            return sign ? -val : val;
        };
        
        float scale = f16tof32(scale16);
        float minVal = f16tof32(min16);
        
        int byteIdx = 4 + (posInBlock * 3) / 8;
        int bitIdx = (posInBlock * 3) % 8;
        uint8_t quantVal = 0;
        
        if (bitIdx + 3 <= 8) {
            quantVal = (block[byteIdx] >> bitIdx) & 0x7;
        } else {
            int bits1 = 8 - bitIdx;
            quantVal = (block[byteIdx] >> bitIdx) & ((1 << bits1) - 1);
            quantVal |= (block[byteIdx + 1] & ((1 << (3 - bits1)) - 1)) << bits1;
        }
        
        output[idx] = minVal + (float)quantVal * scale;
    }
}

// Q6_K: 256 elements per block, 6-bit quantization
__global__ void dequantizeQ6_KKernel(const uint8_t* quantized, float* output, int64_t numElements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockSize = 256;
    
    if (idx < numElements) {
        int64_t blockIdx = idx / blockSize;
        int posInBlock = idx % blockSize;
        
        // K-quant: scales(2) + mins(2) + 192 bytes (256*6/8 = 192)
        const int qlBlockSize = 196;
        const uint8_t* block = quantized + blockIdx * qlBlockSize;
        
        uint16_t scale16, min16;
        memcpy(&scale16, block, 2);
        memcpy(&min16, block + 2, 2);
        
        auto f16tof32 = [](uint16_t h) -> float {
            int sign = (h >> 15) & 1;
            int exponent = (h >> 10) & 0x1F;
            int mantissa = h & 0x3FF;
            if (exponent == 0) return 0.0f;
            if (exponent == 31) return sign ? -1e10f : 1e10f;
            float val = powf(2.0f, (float)exponent - 15.0f) * (1.0f + mantissa / 1024.0f);
            return sign ? -val : val;
        };
        
        float scale = f16tof32(scale16);
        float minVal = f16tof32(min16);
        
        int byteIdx = 4 + (posInBlock * 6) / 8;
        int bitIdx = (posInBlock * 6) % 8;
        uint8_t quantVal = 0;
        
        if (bitIdx + 6 <= 8) {
            quantVal = (block[byteIdx] >> bitIdx) & 0x3F;
        } else {
            int bits1 = 8 - bitIdx;
            quantVal = (block[byteIdx] >> bitIdx) & ((1 << bits1) - 1);
            quantVal |= (block[byteIdx + 1] & ((1 << (6 - bits1)) - 1)) << bits1;
        }
        
        output[idx] = minVal + (float)quantVal * scale;
    }
}

// Q4_K: 256 elements per block, 4-bit K-quant
__global__ void dequantizeQ4_KKernel(const uint8_t* quantized, float* output, int64_t numElements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockSize = 256;
    
    if (idx < numElements) {
        int64_t blockIdx = idx / blockSize;
        int posInBlock = idx % blockSize;
        
        // K-quant 4: scales(2) + mins(2) + 128 bytes
        const int qlBlockSize = 132;
        const uint8_t* block = quantized + blockIdx * qlBlockSize;
        
        uint16_t scale16, min16;
        memcpy(&scale16, block, 2);
        memcpy(&min16, block + 2, 2);
        
        auto f16tof32 = [](uint16_t h) -> float {
            int sign = (h >> 15) & 1;
            int exponent = (h >> 10) & 0x1F;
            int mantissa = h & 0x3FF;
            if (exponent == 0) return 0.0f;
            if (exponent == 31) return sign ? -1e10f : 1e10f;
            float val = powf(2.0f, (float)exponent - 15.0f) * (1.0f + mantissa / 1024.0f);
            return sign ? -val : val;
        };
        
        float scale = f16tof32(scale16);
        float minVal = f16tof32(min16);
        
        int byteIdx = 4 + posInBlock / 2;
        int nibbleIdx = posInBlock % 2;
        uint8_t quantVal = (block[byteIdx] >> (nibbleIdx * 4)) & 0x0F;
        
        output[idx] = minVal + (float)quantVal * scale;
    }
}

// Q5_K: 256 elements per block, 5-bit K-quant
__global__ void dequantizeQ5_KKernel(const uint8_t* quantized, float* output, int64_t numElements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockSize = 256;
    
    if (idx < numElements) {
        int64_t blockIdx = idx / blockSize;
        int posInBlock = idx % blockSize;
        
        // K-quant 5: scales(2) + mins(2) + 160 bytes (256*5/8)
        const int qlBlockSize = 164;
        const uint8_t* block = quantized + blockIdx * qlBlockSize;
        
        uint16_t scale16, min16;
        memcpy(&scale16, block, 2);
        memcpy(&min16, block + 2, 2);
        
        auto f16tof32 = [](uint16_t h) -> float {
            int sign = (h >> 15) & 1;
            int exponent = (h >> 10) & 0x1F;
            int mantissa = h & 0x3FF;
            if (exponent == 0) return 0.0f;
            if (exponent == 31) return sign ? -1e10f : 1e10f;
            float val = powf(2.0f, (float)exponent - 15.0f) * (1.0f + mantissa / 1024.0f);
            return sign ? -val : val;
        };
        
        float scale = f16tof32(scale16);
        float minVal = f16tof32(min16);
        
        int byteIdx = 4 + (posInBlock * 5) / 8;
        int bitIdx = (posInBlock * 5) % 8;
        uint8_t quantVal = 0;
        
        if (bitIdx + 5 <= 8) {
            quantVal = (block[byteIdx] >> bitIdx) & 0x1F;
        } else {
            int bits1 = 8 - bitIdx;
            quantVal = (block[byteIdx] >> bitIdx) & ((1 << bits1) - 1);
            quantVal |= (block[byteIdx + 1] & ((1 << (5 - bits1)) - 1)) << bits1;
        }
        
        output[idx] = minVal + (float)quantVal * scale;
    }
}

// ==================== Tokenizer (CPU) ====================

class Tokenizer {
private:
    std::map<std::string, int> tokenToID;
    std::vector<std::string> idToToken;
    int vocabSize = 0;
    bool loaded = false;

public:
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Tokenizer file not found: " << filename << std::endl;
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json = buffer.str();
        file.close();

        size_t vocabPos = json.find("\"vocab\"");
        if (vocabPos == std::string::npos) {
            std::cerr << "No vocab found in tokenizer" << std::endl;
            return false;
        }

        size_t braceStart = json.find('{', vocabPos);
        if (braceStart == std::string::npos) return false;

        int braceCount = 1;
        size_t pos = braceStart + 1;
        
        while (braceCount > 0 && pos < json.size()) {
            if (json[pos] == '{') braceCount++;
            else if (json[pos] == '}') braceCount--;
            
            if (braceCount == 1 && json[pos] == '"') {
                size_t tokenStart = pos + 1;
                size_t tokenEnd = json.find('"', tokenStart);
                while (tokenEnd != std::string::npos && json[tokenEnd - 1] == '\\') {
                    tokenEnd = json.find('"', tokenEnd + 1);
                }
                if (tokenEnd == std::string::npos) break;
                
                std::string token = json.substr(tokenStart, tokenEnd - tokenStart);
                
                size_t escPos;
                while ((escPos = token.find("\\\"")) != std::string::npos)
                    token.replace(escPos, 2, "\"");
                while ((escPos = token.find("\\n")) != std::string::npos)
                    token.replace(escPos, 2, "\n");
                while ((escPos = token.find("\\t")) != std::string::npos)
                    token.replace(escPos, 2, "\t");
                while ((escPos = token.find("\\\\")) != std::string::npos)
                    token.replace(escPos, 2, "\\");
                
                size_t colonPos = json.find(':', tokenEnd);
                if (colonPos == std::string::npos) break;
                
                size_t numStart = colonPos + 1;
                while (numStart < json.size() && (json[numStart] == ' ' || json[numStart] == '\t'))
                    numStart++;
                
                size_t numEnd = numStart;
                while (numEnd < json.size() && (json[numEnd] >= '0' && json[numEnd] <= '9'))
                    numEnd++;
                
                if (numEnd > numStart) {
                    int id = std::stoi(json.substr(numStart, numEnd - numStart));
                    tokenToID[token] = id;
                    
                    while ((int)idToToken.size() <= id)
                        idToToken.push_back("");
                    idToToken[id] = token;
                    
                    if (id >= vocabSize) vocabSize = id + 1;
                }
                
                pos = numEnd;
            } else {
                pos++;
            }
        }

        loaded = vocabSize > 0;
        if (loaded)
            std::cout << "Tokenizer loaded: " << vocabSize << " tokens" << std::endl;
        
        return loaded;
    }

    int getTokenID(const std::string& token) const {
        auto it = tokenToID.find(token);
        return (it != tokenToID.end()) ? it->second : -1;
    }

    std::string getIDToken(int id) const {
        if (id >= 0 && id < (int)idToToken.size())
            return idToToken[id];
        return "";
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> result;
        if (!loaded) return result;

        std::vector<std::string> tokens;
        std::string currentWord;

        for (char ch : text) {
            if (ch == ' ') {
                if (!currentWord.empty())
                    tokens.push_back(currentWord);
                currentWord = "\xC4\xA0";
            } else {
                currentWord += ch;
            }
        }
        if (!currentWord.empty())
            tokens.push_back(currentWord);

        for (const auto& token : tokens) {
            int id = getTokenID(token);
            if (id >= 0) {
                result.push_back(id);
            } else {
                for (char c : token) {
                    std::string charStr(1, c);
                    id = getTokenID(charStr);
                    if (id >= 0) result.push_back(id);
                }
            }
        }
        return result;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            std::string token = getIDToken(id);
            size_t pos;
            while ((pos = token.find("\xC4\xA0")) != std::string::npos)
                token.replace(pos, 2, " ");
            while ((pos = token.find("\xC4\x8A")) != std::string::npos)
                token.replace(pos, 2, "\n");
            result += token;
        }
        return result;
    }

    bool isLoaded() const { return loaded; }
    int getVocabSize() const { return vocabSize; }
};

// ==================== GGUFTensor ====================

struct GGUFTensor {
    std::string name;
    std::vector<int64_t> shape;
    int numDims = 0;
    GGML_DType dtype = GGML_DType::UNKNOWN;
    int64_t dataOffset = 0;
    bool dataLoaded = false;
    bool dequantized = false;
    std::vector<float> data;  // CPU buffer for quantized or float data
    float* d_data = nullptr;  // GPU float32 buffer
    void* d_quantized = nullptr;  // GPU quantized buffer
    int64_t quantizedSize = 0;
};

// ==================== GGUFLoader ====================

class GGUFLoader {
private:
    std::ifstream stream;
    std::string filename;
    std::vector<GGUFTensor> tensors;
    std::map<std::string, size_t> tensorMap;
    int64_t tensorDataStart = 0;
    
    int embedDim = 768;
    int numLayers = 12;
    int numHeads = 12;
    int ffnDim = 3072;
    int vocabSize = 50257;
    int maxSeqLen = 1024;
    bool loaded = false;
    
    QuantizationStats quantStats;

    uint32_t readUInt32() {
        uint32_t val;
        stream.read(reinterpret_cast<char*>(&val), 4);
        return val;
    }

    uint64_t readUInt64() {
        uint64_t val;
        stream.read(reinterpret_cast<char*>(&val), 8);
        return val;
    }

    float float16ToFloat32(uint16_t h) {
        int sign = (h >> 15) & 1;
        int exponent = (h >> 10) & 0x1F;
        int mantissa = h & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0) return 0.0f;
            double e = -14;
            double m = mantissa / 1024.0;
            while (m < 1) { m *= 2; e -= 1; }
            float val = (float)(m * std::pow(2.0, e));
            return sign ? -val : val;
        } else if (exponent == 31) {
            if (mantissa != 0) return std::numeric_limits<float>::quiet_NaN();
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else {
            float val = (float)((1 + mantissa / 1024.0) * std::pow(2.0, exponent - 15));
            return sign ? -val : val;
        }
    }

    float bfloat16ToFloat32(uint16_t bf) {
        uint32_t f32bits = (uint32_t)bf << 16;
        float result;
        std::memcpy(&result, &f32bits, 4);
        return result;
    }

    std::string readString() {
        uint64_t len = readUInt64();
        if (len > 10000000) return "";
        std::string str(len, '\0');
        if (len > 0)
            stream.read(&str[0], len);
        return str;
    }

    void skipMetadataValue(int valueType) {
        switch (valueType) {
            case 0: case 1: stream.seekg(1, std::ios::cur); break;
            case 2: case 3: stream.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: stream.seekg(4, std::ios::cur); break;
            case 7: stream.seekg(1, std::ios::cur); break;
            case 8: {
                uint64_t strLen = readUInt64();
                stream.seekg(strLen, std::ios::cur);
                break;
            }
            case 9: {
                uint32_t arrType = readUInt32();
                uint64_t arrCount = readUInt64();
                for (uint64_t i = 0; i < std::min(arrCount, (uint64_t)1000000); i++)
                    skipMetadataValue(arrType);
                break;
            }
            case 10: case 11: case 12: stream.seekg(8, std::ios::cur); break;
        }
    }

    void parseHeader() {
        char magic[5] = {0};
        stream.read(magic, 4);
        if (std::string(magic) != GGUF_MAGIC)
            throw std::runtime_error("Invalid GGUF magic: " + std::string(magic));

        uint32_t version = readUInt32();
        uint64_t tensorCount = readUInt64();
        uint64_t metadataCount = readUInt64();

        std::cout << "GGUF Version: " << version << std::endl;
        std::cout << "Tensors: " << tensorCount << std::endl;
        std::cout << "Metadata entries: " << metadataCount << std::endl;

        std::string modelType = "unknown";

        for (uint64_t i = 0; i < metadataCount; i++) {
            std::string key = readString();
            uint32_t valueType = readUInt32();
            
            // GPT-2 style metadata
            if ((key == "gpt2.embedding_length") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                embedDim = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
                modelType = "GPT-2";
            } else if ((key == "gpt2.block_count") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                numLayers = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "gpt2.attention.head_count") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                numHeads = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "gpt2.feed_forward_length") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                ffnDim = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "gpt2.context_length") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                maxSeqLen = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            }
            // LLaMA style metadata
            else if ((key == "llama.embedding_length" || key == "llama.dim") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                embedDim = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
                modelType = "LLaMA";
            } else if ((key == "llama.block_count" || key == "llama.n_layer") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                numLayers = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "llama.attention.head_count") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                numHeads = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "llama.feed_forward_length") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                ffnDim = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "llama.context_length") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                maxSeqLen = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            }
            // Generic/alternative metadata keys
            else if ((key == "general.embedding_length" || key == "embedding_length") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                if (embedDim == 768) embedDim = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
                modelType = "Generic";
            } else if ((key == "general.block_count" || key == "block_count" || key == "n_layer") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                if (numLayers == 12) numLayers = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "general.attention.head_count" || key == "head_count" || key == "n_head") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                if (numHeads == 12) numHeads = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "general.feed_forward_length" || key == "feed_forward_length") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                if (ffnDim == 3072) ffnDim = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "general.context_length" || key == "context_length") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                if (maxSeqLen == 1024) maxSeqLen = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else {
                skipMetadataValue(valueType);
            }
        }

        std::cout << "Detected model type: " << modelType << std::endl;

        std::cout << "Model config: embed_dim=" << embedDim << ", layers=" << numLayers
                  << ", heads=" << numHeads << ", ffn=" << ffnDim << std::endl;

        tensors.resize(tensorCount);
        for (uint64_t i = 0; i < tensorCount; i++) {
            tensors[i].name = readString();
            uint32_t numDims = readUInt32();
            tensors[i].numDims = numDims;
            tensors[i].shape.resize(numDims);
            for (uint32_t d = 0; d < numDims; d++)
                tensors[i].shape[d] = readUInt64();
            int dtypeInt = readUInt32();
            tensors[i].dtype = (GGML_DType)dtypeInt;
            tensors[i].dataOffset = readUInt64();
            tensors[i].dataLoaded = false;
            tensors[i].d_data = nullptr;
            tensorMap[tensors[i].name] = i;
        }

        int64_t pos = stream.tellg();
        int64_t aligned = ((pos + 31) / 32) * 32;
        tensorDataStart = aligned;
    }

    int64_t getQuantizedSize(GGML_DType dtype, int64_t numElements) {
        switch (dtype) {
            case GGML_DType::Q4_0: 
            case GGML_DType::Q4_1: return (numElements * 32 + 15) / 16;
            case GGML_DType::Q5_0:
            case GGML_DType::Q5_1: return (numElements * 40 + 31) / 32;
            case GGML_DType::Q8_0: return numElements;
            case GGML_DType::Q2_K: return (numElements * 256 + 2047) / 2048 * 36; // 36 bytes per 256
            case GGML_DType::Q3_K: return (numElements * 256 + 2047) / 2048 * 100; // ~100 bytes per 256
            case GGML_DType::Q4_K: return (numElements * 256 + 2047) / 2048 * 132; // 132 bytes per 256
            case GGML_DType::Q5_K: return (numElements * 256 + 2047) / 2048 * 164; // 164 bytes per 256
            case GGML_DType::Q6_K: return (numElements * 256 + 2047) / 2048 * 196; // 196 bytes per 256
            default: return 0;
        }
    }

    bool loadTensorByIndex(size_t idx) {
        if (idx >= tensors.size()) return false;
        GGUFTensor& t = tensors[idx];
        if (t.dataLoaded) return true;

        int64_t numElements = 1;
        for (int64_t dim : t.shape)
            numElements *= dim;

        stream.seekg(tensorDataStart + t.dataOffset);

        const char* typeName = getQuantTypeName(t.dtype);
        int64_t originalSize = numElements * 4; // Assume F32 equivalent
        int64_t compressedSize = 0;

        if (t.dtype == GGML_DType::F32) {
            // F32 - load directly to GPU
            t.data.resize(numElements);
            stream.read(reinterpret_cast<char*>(t.data.data()), numElements * 4);
            CUDA_CHECK(cudaMalloc(&t.d_data, numElements * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(t.d_data, t.data.data(), numElements * sizeof(float), cudaMemcpyHostToDevice));
            t.data.clear();
            t.data.shrink_to_fit();
            compressedSize = numElements * 4;
            
        } else if (t.dtype == GGML_DType::F16) {
            // F16 - convert on CPU then upload
            std::vector<uint16_t> f16data(numElements);
            stream.read(reinterpret_cast<char*>(f16data.data()), numElements * 2);
            t.data.resize(numElements);
            for (int64_t j = 0; j < numElements; j++)
                t.data[j] = float16ToFloat32(f16data[j]);
            CUDA_CHECK(cudaMalloc(&t.d_data, numElements * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(t.d_data, t.data.data(), numElements * sizeof(float), cudaMemcpyHostToDevice));
            t.data.clear();
            t.data.shrink_to_fit();
            compressedSize = numElements * 2;
            
        } else if (isQuantTypeSupported(t.dtype)) {
            // Quantized formats - load to GPU as-is
            int64_t quantizedSize = getQuantizedSize(t.dtype, numElements);
            if (quantizedSize == 0) {
                std::cerr << "ERROR: Could not calculate quantized size for dtype " 
                          << (int)t.dtype << " (" << typeName << ")" << std::endl;
                return false;
            }
            
            std::vector<uint8_t> qdata(quantizedSize);
            stream.read(reinterpret_cast<char*>(qdata.data()), quantizedSize);
            
            CUDA_CHECK(cudaMalloc(&t.d_quantized, quantizedSize));
            CUDA_CHECK(cudaMemcpy(t.d_quantized, qdata.data(), quantizedSize, cudaMemcpyHostToDevice));
            t.quantizedSize = quantizedSize;
            compressedSize = quantizedSize;
            
        } else if (t.dtype == GGML_DType::BFLOAT16) {
            // BFLOAT16
            std::vector<uint16_t> bf16data(numElements);
            stream.read(reinterpret_cast<char*>(bf16data.data()), numElements * 2);
            t.data.resize(numElements);
            for (int64_t j = 0; j < numElements; j++)
                t.data[j] = bfloat16ToFloat32(bf16data[j]);
            CUDA_CHECK(cudaMalloc(&t.d_data, numElements * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(t.d_data, t.data.data(), numElements * sizeof(float), cudaMemcpyHostToDevice));
            t.data.clear();
            t.data.shrink_to_fit();
            compressedSize = numElements * 2;
        } else {
            std::cerr << "ERROR: Unsupported dtype " << (int)t.dtype << " (" << typeName 
                      << ") for tensor " << t.name << std::endl;
            std::cerr << "GUIDANCE: Update your model quantization or compile with support for this dtype" << std::endl;
            return false;
        }

        quantStats.add(typeName, originalSize, compressedSize);
        t.dataLoaded = true;
        return true;
    }

public:
    bool loadFromFile(const std::string& fname) {
        filename = fname;
        stream.open(fname, std::ios::binary);
        if (!stream.is_open()) {
            std::cerr << "Failed to open GGUF file: " << fname << std::endl;
            return false;
        }

        try {
            parseHeader();
            loaded = true;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing GGUF: " << e.what() << std::endl;
            return false;
        }

        return true;
    }

    float* getTensorGPU(const std::vector<std::string>& names) {
        for (const auto& name : names) {
            auto it = tensorMap.find(name);
            if (it != tensorMap.end()) {
                GGUFTensor& t = tensors[it->second];
                
                // Already dequantized or float
                if (t.d_data != nullptr) return t.d_data;
                
                // Load from disk if needed
                if (!t.dataLoaded && !loadTensorByIndex(it->second)) return nullptr;
                
                // If still quantized, dequantize now
                if (!t.dequantized && t.d_quantized != nullptr) {
                    int64_t numElements = 1;
                    for (int64_t dim : t.shape) numElements *= dim;
                    
                    CUDA_CHECK(cudaMalloc(&t.d_data, numElements * sizeof(float)));
                    
                    int blockSize = 256;
                    int64_t gridSize = (numElements + blockSize - 1) / blockSize;
                    
                    try {
                        switch (t.dtype) {
                            case GGML_DType::Q4_0:
                                dequantizeQ4_0Kernel<<<gridSize, blockSize>>>((const uint8_t*)t.d_quantized, t.d_data, numElements);
                                break;
                            case GGML_DType::Q4_1:
                                dequantizeQ4_1Kernel<<<gridSize, blockSize>>>((const uint8_t*)t.d_quantized, t.d_data, numElements);
                                break;
                            case GGML_DType::Q5_0:
                                dequantizeQ5_0Kernel<<<gridSize, blockSize>>>((const uint8_t*)t.d_quantized, t.d_data, numElements);
                                break;
                            case GGML_DType::Q5_1:
                                dequantizeQ5_1Kernel<<<gridSize, blockSize>>>((const uint8_t*)t.d_quantized, t.d_data, numElements);
                                break;
                            case GGML_DType::Q8_0:
                                dequantizeQ8_0Kernel<<<gridSize, blockSize>>>((const uint8_t*)t.d_quantized, t.d_data, numElements);
                                break;
                            case GGML_DType::Q2_K:
                                dequantizeQ2_KKernel<<<gridSize, blockSize>>>((const uint8_t*)t.d_quantized, t.d_data, numElements);
                                break;
                            case GGML_DType::Q3_K:
                                dequantizeQ3_KKernel<<<gridSize, blockSize>>>((const uint8_t*)t.d_quantized, t.d_data, numElements);
                                break;
                            case GGML_DType::Q6_K:
                                dequantizeQ6_KKernel<<<gridSize, blockSize>>>((const uint8_t*)t.d_quantized, t.d_data, numElements);
                                break;
                            case GGML_DType::Q4_K:
                                dequantizeQ4_KKernel<<<gridSize, blockSize>>>((const uint8_t*)t.d_quantized, t.d_data, numElements);
                                break;
                            case GGML_DType::Q5_K:
                                dequantizeQ5_KKernel<<<gridSize, blockSize>>>((const uint8_t*)t.d_quantized, t.d_data, numElements);
                                break;
                            default:
                                std::cerr << "ERROR: Dequantization kernel not implemented for dtype " 
                                          << (int)t.dtype << std::endl;
                                return nullptr;
                        }
                    } catch (...) {
                        std::cerr << "ERROR: Dequantization kernel failed for tensor " << t.name << std::endl;
                        return nullptr;
                    }
                    
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    
                    // Free quantized data
                    cudaFree(t.d_quantized);
                    t.d_quantized = nullptr;
                    t.dequantized = true;
                }
                
                return t.d_data;
            }
        }
        return nullptr;
    }

    std::vector<float> getTensor(const std::vector<std::string>& names) {
        for (const auto& name : names) {
            auto it = tensorMap.find(name);
            if (it != tensorMap.end()) {
                if (loadTensorByIndex(it->second))
                    return tensors[it->second].data;
            }
        }
        return {};
    }

    bool hasTensor(const std::string& name) const {
        return tensorMap.find(name) != tensorMap.end();
    }

    void printAllTensorNames() {
        std::cout << "\n=== All Tensor Names ===" << std::endl;
        for (const auto& t : tensors) {
            std::cout << t.name << " [";
            for (size_t i = 0; i < t.shape.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << t.shape[i];
            }
            std::cout << "] dtype=" << (int)t.dtype << " (" << getQuantTypeName(t.dtype) << ")" << std::endl;
        }
    }

    void printQuantizationStats() {
        quantStats.print();
    }

    void freeGPUMemory() {
        for (auto& t : tensors) {
            if (t.d_data != nullptr) {
                cudaFree(t.d_data);
                t.d_data = nullptr;
            }
            if (t.d_quantized != nullptr) {
                cudaFree(t.d_quantized);
                t.d_quantized = nullptr;
            }
        }
    }

    int getEmbedDim() const { return embedDim; }
    int getNumLayers() const { return numLayers; }
    int getNumHeads() const { return numHeads; }
    int getFFNDim() const { return ffnDim; }
    int getVocabSize() const { return vocabSize; }
    int getMaxSeqLen() const { return maxSeqLen; }
    bool isLoaded() const { return loaded; }
};

// ==================== TransformerModel ====================

enum class AttentionType {
    STANDARD,      // Multi-head attention
    MQA,           // Multi-Query Attention (1 KV head shared by all Q heads)
    GQA            // Grouped Query Attention (n KV heads, multiple Q heads per KV)
};

enum class FFNActivation {
    GELU,          // GPT-2 style
    SWIGLU         // LLaMA style
};

enum class PositionalEmbedding {
    ABSOLUTE,      // Fixed positional embeddings (GPT-2)
    ROPE           // Rotary Position Embeddings (LLaMA, Mistral)
};

class TransformerModel {
private:
    GGUFLoader loader;
    Tokenizer tokenizer;
    int embedDim = 0;
    int numHeads = 0;
    int numKVHeads = 0;  // For GQA/MQA support
    int headDim = 0;
    int numLayers = 0;
    int ffnDim = 0;
    int vocabSize = 0;

    AttentionType attentionType = AttentionType::STANDARD;
    FFNActivation ffnActivation = FFNActivation::GELU;
    PositionalEmbedding posEmbedding = PositionalEmbedding::ABSOLUTE;

    std::mt19937 rng;

    float* d_hidden = nullptr;
    float* d_hidden2 = nullptr;
    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_attnOut = nullptr;
    float* d_attnScores = nullptr;
    float* d_ffnHidden = nullptr;
    float* d_logits = nullptr;
    int* d_tokenIDs = nullptr;
    
    int allocatedSeqLen = 0;

    void allocateBuffers(int seqLen) {
        if (seqLen <= allocatedSeqLen) return;
        
        freeBuffers();
        
        CUDA_CHECK(cudaMalloc(&d_hidden, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden2, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attnOut, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attnScores, numHeads * seqLen * seqLen * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ffnHidden, seqLen * ffnDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_logits, vocabSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tokenIDs, seqLen * sizeof(int)));
        
        allocatedSeqLen = seqLen;
    }

    void freeBuffers() {
        if (d_hidden) cudaFree(d_hidden);
        if (d_hidden2) cudaFree(d_hidden2);
        if (d_Q) cudaFree(d_Q);
        if (d_K) cudaFree(d_K);
        if (d_V) cudaFree(d_V);
        if (d_attnOut) cudaFree(d_attnOut);
        if (d_attnScores) cudaFree(d_attnScores);
        if (d_ffnHidden) cudaFree(d_ffnHidden);
        if (d_logits) cudaFree(d_logits);
        if (d_tokenIDs) cudaFree(d_tokenIDs);
        d_hidden = d_hidden2 = d_Q = d_K = d_V = d_attnOut = d_attnScores = d_ffnHidden = d_logits = nullptr;
        d_tokenIDs = nullptr;
        allocatedSeqLen = 0;
    }

    void embedTokens(const std::vector<int>& tokenIDs, int seqLen) {
        // Try multiple naming conventions (GPT-2, LLaMA, etc.)
        float* d_tokenEmb = loader.getTensorGPU({
            "token_embd.weight", "wte.weight",
            "model.embed_tokens.weight", "lm_head.weight"
        });
        float* d_posEmb = loader.getTensorGPU({
            "position_embd.weight", "wpe.weight"
        });
        
        CUDA_CHECK(cudaMemcpy(d_tokenIDs, tokenIDs.data(), seqLen * sizeof(int), cudaMemcpyHostToDevice));
        
        dim3 block(embedDim);
        dim3 grid(seqLen);
        embedTokensKernel<<<grid, block>>>(d_tokenIDs, d_tokenEmb, d_posEmb, d_hidden, seqLen, embedDim);
        CUDA_CHECK(cudaGetLastError());
    }

    void attentionBlock(int seqLen, int layerIdx) {
        // Support multiple naming conventions (GPT-2, LLaMA, etc.)
        std::string gpt2Prefix = "blk." + std::to_string(layerIdx) + ".";
        std::string llamaPrefix = "model.layers." + std::to_string(layerIdx) + ".self_attn.";
        
        float* d_ln1g = loader.getTensorGPU({
            gpt2Prefix + "attn_norm.weight",
            llamaPrefix + "input_layernorm.weight"
        });
        float* d_ln1b = loader.getTensorGPU({
            gpt2Prefix + "attn_norm.bias",
            llamaPrefix + "input_layernorm.bias"
        });
        float* d_qkvW = loader.getTensorGPU({
            gpt2Prefix + "attn_qkv.weight",
            llamaPrefix + "q_proj.weight"
        });
        float* d_qkvB = loader.getTensorGPU({
            gpt2Prefix + "attn_qkv.bias",
            llamaPrefix + "q_proj.bias"
        });
        float* d_projW = loader.getTensorGPU({
            gpt2Prefix + "attn_output.weight",
            llamaPrefix + "o_proj.weight"
        });
        float* d_projB = loader.getTensorGPU({
            gpt2Prefix + "attn_output.bias",
            llamaPrefix + "o_proj.bias"
        });
        
        int sharedMem = BLOCK_SIZE * sizeof(float);
        layerNormKernel<<<seqLen, BLOCK_SIZE, sharedMem>>>(d_hidden, d_hidden2, d_ln1g, d_ln1b, seqLen, embedDim);
        
        dim3 qkvBlock(BLOCK_SIZE);
        dim3 qkvGrid((embedDim + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen);
        computeQKVKernel<<<qkvGrid, qkvBlock>>>(d_hidden2, d_qkvW, d_qkvB, d_Q, d_K, d_V, seqLen, embedDim);
        
        // Apply RoPE if needed
        if (posEmbedding == PositionalEmbedding::ROPE) {
            dim3 ropeBlock(BLOCK_SIZE / 2);
            dim3 ropeGrid(seqLen, numHeads);
            applyRoPEKernel<<<ropeGrid, ropeBlock>>>(d_Q, d_K, seqLen, numHeads, headDim);
            CUDA_CHECK(cudaGetLastError());
        }
        
        float scale = sqrtf((float)headDim);
        dim3 scoreBlock(BLOCK_SIZE);
        
        // Use appropriate attention kernel based on type
        if (attentionType == AttentionType::STANDARD) {
            dim3 scoreGrid((seqLen + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen, numHeads);
            attentionScoresKernel<<<scoreGrid, scoreBlock>>>(d_Q, d_K, d_attnScores, seqLen, numHeads, headDim, scale);
        } else {
            dim3 scoreGrid((seqLen + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen, numHeads);
            attentionScoresGQAKernel<<<scoreGrid, scoreBlock>>>(d_Q, d_K, d_attnScores, seqLen, numHeads, numKVHeads, headDim, scale);
        }
        
        for (int h = 0; h < numHeads; h++) {
            softmaxKernel<<<seqLen, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
                d_attnScores + h * seqLen * seqLen, seqLen, seqLen);
        }
        
        dim3 outBlock(BLOCK_SIZE);
        dim3 outGrid((headDim + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen, numHeads);
        
        if (attentionType == AttentionType::STANDARD) {
            attentionOutputKernel<<<outGrid, outBlock>>>(d_attnScores, d_V, d_attnOut, seqLen, numHeads, headDim);
        } else {
            attentionOutputGQAKernel<<<outGrid, outBlock>>>(d_attnScores, d_V, d_attnOut, seqLen, numHeads, numKVHeads, headDim);
        }
        
        dim3 projBlock(BLOCK_SIZE);
        dim3 projGrid((embedDim + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen);
        projectionKernel<<<projGrid, projBlock>>>(d_attnOut, d_projW, d_projB, d_hidden2, d_hidden, seqLen, embedDim);
        
        std::swap(d_hidden, d_hidden2);
        
        CUDA_CHECK(cudaGetLastError());
    }

    void ffnBlock(int seqLen, int layerIdx) {
        // Support multiple naming conventions (GPT-2, LLaMA, etc.)
        std::string gpt2Prefix = "blk." + std::to_string(layerIdx) + ".";
        std::string llamaPrefix = "model.layers." + std::to_string(layerIdx) + ".mlp.";
        
        float* d_ln2g = loader.getTensorGPU({
            gpt2Prefix + "ffn_norm.weight",
            llamaPrefix + "post_attention_layernorm.weight"
        });
        float* d_ln2b = loader.getTensorGPU({
            gpt2Prefix + "ffn_norm.bias",
            llamaPrefix + "post_attention_layernorm.bias"
        });
        
        int sharedMem = BLOCK_SIZE * sizeof(float);
        layerNormKernel<<<seqLen, BLOCK_SIZE, sharedMem>>>(d_hidden, d_hidden2, d_ln2g, d_ln2b, seqLen, embedDim);
        
        dim3 upBlock(BLOCK_SIZE);
        dim3 upGrid((ffnDim + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen);
        
        if (ffnActivation == FFNActivation::GELU) {
            // GPT-2 style: single projection + GELU
            float* d_upW = loader.getTensorGPU({
                gpt2Prefix + "ffn_up.weight",
                llamaPrefix + "up_proj.weight"
            });
            float* d_upB = loader.getTensorGPU({
                gpt2Prefix + "ffn_up.bias",
                llamaPrefix + "up_proj.bias"
            });
            ffnUpGELUKernel<<<upGrid, upBlock>>>(d_hidden2, d_upW, d_upB, d_ffnHidden, seqLen, embedDim, ffnDim);
        } else {
            // LLaMA style: gate projection + up projection + SwiGLU
            float* d_upW = loader.getTensorGPU({
                gpt2Prefix + "ffn_up.weight",
                llamaPrefix + "up_proj.weight"
            });
            float* d_upB = loader.getTensorGPU({
                gpt2Prefix + "ffn_up.bias",
                llamaPrefix + "up_proj.bias"
            });
            float* d_gateW = loader.getTensorGPU({
                gpt2Prefix + "ffn_gate.weight",
                llamaPrefix + "gate_proj.weight"
            });
            float* d_gateB = loader.getTensorGPU({
                gpt2Prefix + "ffn_gate.bias",
                llamaPrefix + "gate_proj.bias"
            });
            ffnUpSwiGLUKernel<<<upGrid, upBlock>>>(d_hidden2, d_upW, d_upB, d_gateW, d_gateB, d_ffnHidden, seqLen, embedDim, ffnDim);
        }
        
        float* d_downW = loader.getTensorGPU({
            gpt2Prefix + "ffn_down.weight",
            llamaPrefix + "down_proj.weight"
        });
        float* d_downB = loader.getTensorGPU({
            gpt2Prefix + "ffn_down.bias",
            llamaPrefix + "down_proj.bias"
        });
        
        dim3 downBlock(BLOCK_SIZE);
        dim3 downGrid((embedDim + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen);
        ffnDownKernel<<<downGrid, downBlock>>>(d_ffnHidden, d_downW, d_downB, d_hidden2, d_hidden, seqLen, ffnDim, embedDim);
        
        std::swap(d_hidden, d_hidden2);
        
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<float> computeLogits(int seqLen) {
        float* d_lnG = loader.getTensorGPU({"output_norm.weight", "ln_f.weight"});
        float* d_lnB = loader.getTensorGPU({"output_norm.bias", "ln_f.bias"});
        float* d_tokenEmb = loader.getTensorGPU({"token_embd.weight", "wte.weight"});
        
        float* d_lastHidden;
        CUDA_CHECK(cudaMalloc(&d_lastHidden, embedDim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_lastHidden, d_hidden + (seqLen - 1) * embedDim, embedDim * sizeof(float), cudaMemcpyDeviceToDevice));
        
        float* d_normed;
        CUDA_CHECK(cudaMalloc(&d_normed, embedDim * sizeof(float)));
        
        int sharedMem = BLOCK_SIZE * sizeof(float);
        layerNormKernel<<<1, BLOCK_SIZE, sharedMem>>>(d_lastHidden, d_normed, d_lnG, d_lnB, 1, embedDim);
        
        dim3 block(BLOCK_SIZE);
        dim3 grid((vocabSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
        computeLogitsKernel<<<grid, block>>>(d_normed, d_tokenEmb, d_logits, embedDim, vocabSize);
        
        CUDA_CHECK(cudaGetLastError());
        
        std::vector<float> logits(vocabSize);
        CUDA_CHECK(cudaMemcpy(logits.data(), d_logits, vocabSize * sizeof(float), cudaMemcpyDeviceToHost));
        
        cudaFree(d_lastHidden);
        cudaFree(d_normed);
        
        return logits;
    }

    std::vector<float> forward(const std::vector<int>& tokenIDs) {
        int seqLen = tokenIDs.size();
        allocateBuffers(seqLen);
        
        embedTokens(tokenIDs, seqLen);
        
        for (int l = 0; l < numLayers; l++) {
            std::cout << "\rLayer " << (l + 1) << "/" << numLayers << "..." << std::flush;
            attentionBlock(seqLen, l);
            ffnBlock(seqLen, l);
        }
        std::cout << " done" << std::endl;
        
        return computeLogits(seqLen);
    }

public:
    TransformerModel() : rng(std::random_device{}()) {}
    
    ~TransformerModel() {
        freeBuffers();
        loader.freeGPUMemory();
    }

    bool loadModel(const std::string& ggufPath, bool showStats = true) {
        if (!loader.loadFromFile(ggufPath))
            return false;

        embedDim = loader.getEmbedDim();
        numLayers = loader.getNumLayers();
        numHeads = loader.getNumHeads();
        numKVHeads = numHeads;  // Default to standard attention
        ffnDim = loader.getFFNDim();
        vocabSize = loader.getVocabSize();
        headDim = embedDim / numHeads;

        // Auto-detect model architecture features
        detectArchitecture();

        if (showStats) {
            loader.printQuantizationStats();
            printArchitectureInfo();
        }
        return true;
    }

    void detectArchitecture() {
        // Detect if model uses LLaMA-style features
        if (loader.hasTensor("model.layers.0.self_attn.q_proj.weight")) {
            // LLaMA-style model
            posEmbedding = PositionalEmbedding::ROPE;
            ffnActivation = FFNActivation::SWIGLU;
            
            // Check for GQA/MQA by looking at KV projection shapes
            // For now, assume standard if not specified
            if (loader.hasTensor("model.layers.0.self_attn.k_proj.weight")) {
                attentionType = AttentionType::STANDARD;
                numKVHeads = numHeads;
            }
        } else if (loader.hasTensor("blk.0.attn_qkv.weight")) {
            // GPT-2 style model
            posEmbedding = PositionalEmbedding::ABSOLUTE;
            ffnActivation = FFNActivation::GELU;
            attentionType = AttentionType::STANDARD;
        } else {
            // Generic/Falcon style - check what we have
            if (loader.hasTensor("model.layers.0.self_attention.query.weight") ||
                loader.hasTensor("transformer.h.0.attn.c_attn.weight")) {
                // Likely Falcon or similar
                posEmbedding = PositionalEmbedding::ABSOLUTE;
                ffnActivation = FFNActivation::GELU;
                attentionType = AttentionType::STANDARD;
            }
        }
    }

    void printArchitectureInfo() {
        std::cout << "\n=== Model Architecture ===" << std::endl;
        std::cout << "Positional Embedding: ";
        switch (posEmbedding) {
            case PositionalEmbedding::ABSOLUTE: std::cout << "Absolute (GPT-2)" << std::endl; break;
            case PositionalEmbedding::ROPE: std::cout << "RoPE (LLaMA)" << std::endl; break;
        }
        
        std::cout << "FFN Activation: ";
        switch (ffnActivation) {
            case FFNActivation::GELU: std::cout << "GELU (GPT-2)" << std::endl; break;
            case FFNActivation::SWIGLU: std::cout << "SwiGLU (LLaMA)" << std::endl; break;
        }
        
        std::cout << "Attention Type: ";
        switch (attentionType) {
            case AttentionType::STANDARD: std::cout << "Multi-Head (" << numHeads << " heads)" << std::endl; break;
            case AttentionType::MQA: std::cout << "Multi-Query (1 KV head, " << numHeads << " Q heads)" << std::endl; break;
            case AttentionType::GQA: std::cout << "Grouped-Query (" << numKVHeads << " KV heads, " << numHeads << " Q heads)" << std::endl; break;
        }
        std::cout << "Head Dimension: " << headDim << std::endl;
    }

    bool loadTokenizer(const std::string& tokenizerPath) {
        return tokenizer.loadFromFile(tokenizerPath);
    }

    std::string generate(const std::string& prompt, int maxTokens, double temperature = 1.0) {
        if (!loader.isLoaded()) {
            std::cerr << "Error: Model not loaded" << std::endl;
            return "";
        }

        if (!tokenizer.isLoaded()) {
            std::cerr << "Error: Tokenizer not loaded" << std::endl;
            return "";
        }

        std::cout << "Encoding prompt..." << std::endl;
        auto tokenIDs = tokenizer.encode(prompt);
        std::cout << "Input tokens: " << tokenIDs.size() << std::endl;
        std::cout << "Temperature: " << std::fixed << std::setprecision(2) << temperature << std::endl;

        if (tokenIDs.empty()) {
            std::cerr << "Error: Could not tokenize input" << std::endl;
            return "";
        }

        std::cout << "Token IDs: ";
        for (size_t i = 0; i < std::min(tokenIDs.size(), (size_t)10); i++)
            std::cout << tokenIDs[i] << " ";
        if (tokenIDs.size() > 10) std::cout << "...";
        std::cout << std::endl;

        auto startTime = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < maxTokens; i++) {
            std::cout << std::endl << "=== Generating token " << (i + 1) << "/" << maxTokens << " ===" << std::endl;

            auto logits = forward(tokenIDs);

            if (logits.empty()) {
                std::cerr << "ERROR: Forward pass failed" << std::endl;
                break;
            }

            int bestID = 0;
            float bestLogit = logits[0];
            for (size_t j = 1; j < logits.size(); j++) {
                if (logits[j] > bestLogit) {
                    bestLogit = logits[j];
                    bestID = j;
                }
            }

            int selectedID;
            if (temperature <= 0.01) {
                selectedID = bestID;
            } else {
                for (float& l : logits) l /= temperature;
                
                float maxVal = *std::max_element(logits.begin(), logits.end());
                float sum = 0;
                for (float& l : logits) {
                    l = std::exp(l - maxVal);
                    sum += l;
                }
                for (float& l : logits) l /= sum;
                
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double r = dist(rng);
                double cumulativeProb = 0.0;
                selectedID = 0;
                for (size_t j = 0; j < logits.size(); j++) {
                    cumulativeProb += logits[j];
                    if (r <= cumulativeProb) {
                        selectedID = j;
                        break;
                    }
                }
            }

            std::cout << "Generated token: " << selectedID << " = \"" << tokenizer.getIDToken(selectedID)
                      << "\" (best was: " << bestID << " logit: " << std::fixed << std::setprecision(4) << bestLogit << ")" << std::endl;

            tokenIDs.push_back(selectedID);

            if (selectedID == 50256) {
                std::cout << "[EOS token reached]" << std::endl;
                break;
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        double elapsedSecs = std::chrono::duration<double>(endTime - startTime).count();
        std::cout << std::endl << "Generation complete in " << std::fixed << std::setprecision(1) << elapsedSecs << " seconds" << std::endl;

        return tokenizer.decode(tokenIDs);
    }

    bool isModelLoaded() const { return loader.isLoaded(); }
    bool isTokenizerLoaded() const { return tokenizer.isLoaded(); }
    void printTensorNames() { loader.printAllTensorNames(); }
};

// ==================== Argument Parser ====================

struct Arguments {
    std::string ggufPath;
    std::string tokenizerPath;
    std::string prompt = "Hello";
    std::string inputFile = "";
    std::string outputFile = "";
    int maxTokens = 5;
    double temperature = 1.0;
    float topK = -1.0f;
    float topP = 1.0f;
    int seed = -1;
    float repetitionPenalty = 1.0f;
    int contextLength = 1024;
    int gpuDevice = 0;
    int batchSize = 1;
    int64_t memoryLimit = 0;
    bool listTensors = false;
    bool showQuantStats = true;
    bool benchmark = false;
    bool testDequant = false;
    bool jsonOutput = false;
    bool verbose = false;
    bool help = false;
    bool fp32Only = false;
};

void printUsage(const char* progName) {
    std::cout << "========================================" << std::endl;
    std::cout << "  GGUF Transformer CLI - CUDA/Dequant" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << progName << " <model.gguf> <tokenizer.json> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "REQUIRED ARGUMENTS:" << std::endl;
    std::cout << "  <model.gguf>           Path to the GGUF model file" << std::endl;
    std::cout << "  <tokenizer.json>       Path to the tokenizer JSON file" << std::endl;
    std::cout << std::endl;
    std::cout << "GENERATION OPTIONS:" << std::endl;
    std::cout << "  -p, --prompt TEXT          Input prompt (default: \"Hello\")" << std::endl;
    std::cout << "  --input-file FILE          Read prompt from file instead of command line" << std::endl;
    std::cout << "  -n, --max-tokens N         Maximum tokens to generate (default: 5)" << std::endl;
    std::cout << "  -t, --temperature T        Sampling temperature 0.0-2.0 (default: 1.0)" << std::endl;
    std::cout << "  --top-k K                  Top-K sampling (disable with -1) (default: -1)" << std::endl;
    std::cout << "  --top-p P                  Nucleus/Top-P sampling 0.0-1.0 (default: 1.0)" << std::endl;
    std::cout << "  --repetition-penalty P     Penalize repeated tokens (default: 1.0)" << std::endl;
    std::cout << "  --context-length N         Max context window size (default: 1024)" << std::endl;
    std::cout << "  --seed S                   Random seed for reproducibility (default: random)" << std::endl;
    std::cout << std::endl;
    std::cout << "OUTPUT OPTIONS:" << std::endl;
    std::cout << "  -o, --output FILE          Save generated text to file" << std::endl;
    std::cout << "  --json-output              Format output as JSON" << std::endl;
    std::cout << std::endl;
    std::cout << "MODEL & QUANTIZATION:" << std::endl;
    std::cout << "  --list-tensors             List all tensors in model and exit" << std::endl;
    std::cout << "  --show-quant-stats         Display quantization statistics (default: yes)" << std::endl;
    std::cout << "  --no-quant-stats           Skip quantization statistics output" << std::endl;
    std::cout << "  --fp32-only                Only load F32 tensors, skip quantized (useful for testing)" << std::endl;
    std::cout << std::endl;
    std::cout << "DEVICE & PERFORMANCE:" << std::endl;
    std::cout << "  --device ID                Select GPU device ID (default: 0)" << std::endl;
    std::cout << "  --batch-size N             Batch size for processing (default: 1)" << std::endl;
    std::cout << "  --memory-limit MB          Limit GPU memory usage in MB (0=unlimited)" << std::endl;
    std::cout << "  --benchmark                Run benchmark tests after generation" << std::endl;
    std::cout << std::endl;
    std::cout << "DEBUGGING & TESTING:" << std::endl;
    std::cout << "  --test-dequant             Test dequantization on all quantized tensors" << std::endl;
    std::cout << "  -v, --verbose              Enable verbose logging" << std::endl;
    std::cout << "  -h, --help                 Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "EXAMPLES:" << std::endl;
    std::cout << "  # Basic generation with custom prompt" << std::endl;
    std::cout << "  " << progName << " model.gguf tokenizer.json -p \"Hello world\" -n 20 -t 0.8" << std::endl;
    std::cout << std::endl;
    std::cout << "  # List all tensors to inspect quantization" << std::endl;
    std::cout << "  " << progName << " model.gguf tokenizer.json --list-tensors" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Top-P sampling with custom seed" << std::endl;
    std::cout << "  " << progName << " model.gguf tokenizer.json -p \"Once upon a time\" --top-p 0.9 --seed 42 -n 50" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Read prompt from file, save output, show stats" << std::endl;
    std::cout << "  " << progName << " model.gguf tokenizer.json --input-file prompt.txt -o output.txt --show-quant-stats" << std::endl;
    std::cout << std::endl;
    std::cout << "  # JSON output with benchmark" << std::endl;
    std::cout << "  " << progName << " model.gguf tokenizer.json --json-output --benchmark -n 10" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Test dequantization and verbose output" << std::endl;
    std::cout << "  " << progName << " model.gguf tokenizer.json --test-dequant --verbose" << std::endl;
    std::cout << std::endl;
}

Arguments parseArguments(int argc, char* argv[]) {
    Arguments args;

    if (argc < 2) {
        args.help = true;
        return args;
    }

    int positionalCount = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            args.help = true;
            return args;
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--list-tensors") {
            args.listTensors = true;
        } else if (arg == "--show-quant-stats") {
            args.showQuantStats = true;
        } else if (arg == "--no-quant-stats") {
            args.showQuantStats = false;
        } else if (arg == "--benchmark") {
            args.benchmark = true;
        } else if (arg == "--test-dequant") {
            args.testDequant = true;
        } else if (arg == "--json-output") {
            args.jsonOutput = true;
        } else if (arg == "--fp32-only") {
            args.fp32Only = true;
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if ((arg == "--input-file") && i + 1 < argc) {
            args.inputFile = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            args.outputFile = argv[++i];
        } else if ((arg == "-n" || arg == "--max-tokens") && i + 1 < argc) {
            args.maxTokens = std::stoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--temperature") && i + 1 < argc) {
            args.temperature = std::stod(argv[++i]);
        } else if ((arg == "--top-k") && i + 1 < argc) {
            args.topK = std::stof(argv[++i]);
        } else if ((arg == "--top-p") && i + 1 < argc) {
            args.topP = std::stof(argv[++i]);
        } else if ((arg == "--repetition-penalty") && i + 1 < argc) {
            args.repetitionPenalty = std::stof(argv[++i]);
        } else if ((arg == "--context-length") && i + 1 < argc) {
            args.contextLength = std::stoi(argv[++i]);
        } else if ((arg == "--seed") && i + 1 < argc) {
            args.seed = std::stoi(argv[++i]);
        } else if ((arg == "--device") && i + 1 < argc) {
            args.gpuDevice = std::stoi(argv[++i]);
        } else if ((arg == "--batch-size") && i + 1 < argc) {
            args.batchSize = std::stoi(argv[++i]);
        } else if ((arg == "--memory-limit") && i + 1 < argc) {
            args.memoryLimit = std::stoll(argv[++i]);
        } else if (arg[0] != '-') {
            if (positionalCount == 0) {
                args.ggufPath = arg;
                positionalCount++;
            } else if (positionalCount == 1) {
                args.tokenizerPath = arg;
                positionalCount++;
            }
        }
    }

    if (args.ggufPath.empty() || args.tokenizerPath.empty()) {
        args.help = true;
    }

    return args;
}

// ==================== Main ====================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  GPT-2 CLI - CUDA Implementation" << std::endl;
    std::cout << "  Full GGML/LLaMA2 Dequantization" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    Arguments args = parseArguments(argc, argv);

    if (args.help) {
        printUsage(argv[0]);
        return 0;
    }

    // Device selection
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "ERROR: No CUDA devices found!" << std::endl;
        return 1;
    }
    
    if (args.gpuDevice < 0 || args.gpuDevice >= deviceCount) {
        std::cerr << "ERROR: Invalid device ID " << args.gpuDevice << " (available: " << deviceCount << ")" << std::endl;
        return 1;
    }
    
    CUDA_CHECK(cudaSetDevice(args.gpuDevice));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, args.gpuDevice);
    std::cout << "Using GPU " << args.gpuDevice << ": " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
    
    if (args.verbose) {
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
    }
    std::cout << std::endl;

    TransformerModel model;

    std::cout << "Loading model from: " << args.ggufPath << std::endl;
    if (!model.loadModel(args.ggufPath, args.showQuantStats)) {
        std::cerr << "ERROR: Failed to load model" << std::endl;
        return 1;
    }

    if (args.listTensors) {
        model.printTensorNames();
        return 0;
    }

    std::cout << std::endl << "Loading tokenizer from: " << args.tokenizerPath << std::endl;
    if (!model.loadTokenizer(args.tokenizerPath)) {
        std::cerr << "ERROR: Failed to load tokenizer" << std::endl;
        return 1;
    }

    // Read prompt from file or command line
    std::string prompt = args.prompt;
    if (!args.inputFile.empty()) {
        std::ifstream infile(args.inputFile);
        if (!infile.is_open()) {
            std::cerr << "ERROR: Cannot open input file: " << args.inputFile << std::endl;
            return 1;
        }
        std::stringstream buffer;
        buffer << infile.rdbuf();
        prompt = buffer.str();
        if (args.verbose) std::cout << "Loaded prompt from file: " << args.inputFile << std::endl;
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GENERATION CONFIG:" << std::endl;
    std::cout << "Prompt: \"" << (prompt.length() > 60 ? prompt.substr(0, 60) + "..." : prompt) << "\"" << std::endl;
    std::cout << "Max tokens: " << args.maxTokens << std::endl;
    std::cout << "Temperature: " << std::fixed << std::setprecision(2) << args.temperature << std::endl;
    if (args.topK >= 0.0f) std::cout << "Top-K: " << args.topK << std::endl;
    if (args.topP < 1.0f) std::cout << "Top-P: " << args.topP << std::endl;
    if (args.repetitionPenalty != 1.0f) std::cout << "Repetition penalty: " << args.repetitionPenalty << std::endl;
    if (args.seed >= 0) std::cout << "Seed: " << args.seed << std::endl;
    std::cout << "Device: " << args.gpuDevice << std::endl;
    std::cout << "========================================" << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    std::string generatedText = model.generate(prompt, args.maxTokens, args.temperature);

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalSecs = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GENERATED TEXT:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << generatedText << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << totalSecs << " seconds" << std::endl;

    // Save output to file if requested
    if (!args.outputFile.empty()) {
        std::ofstream outfile(args.outputFile);
        if (!outfile.is_open()) {
            std::cerr << "ERROR: Cannot open output file: " << args.outputFile << std::endl;
            return 1;
        }
        
        if (args.jsonOutput) {
            outfile << "{\"prompt\": \"" << prompt << "\", \"output\": \"" << generatedText << "\", "
                    << "\"tokens\": " << args.maxTokens << ", \"time_seconds\": " << totalSecs << "}" << std::endl;
        } else {
            outfile << generatedText << std::endl;
        }
        
        outfile.close();
        std::cout << "Output saved to: " << args.outputFile << std::endl;
    }

    // JSON output to stdout if requested
    if (args.jsonOutput && args.outputFile.empty()) {
        std::cout << "\nJSON Output:" << std::endl;
        std::cout << "{\"prompt\": \"" << prompt << "\", \"output\": \"" << generatedText << "\", "
                  << "\"tokens\": " << args.maxTokens << ", \"time_seconds\": " << totalSecs << "}" << std::endl;
    }

    // Run benchmark if requested
    if (args.benchmark) {
        std::cout << "\nRunning benchmark (5 iterations)..." << std::endl;
        double totalTime = 0.0;
        for (int i = 0; i < 5; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            std::string benchOutput = model.generate(prompt, args.maxTokens, args.temperature);
            auto t1 = std::chrono::high_resolution_clock::now();
            double iterTime = std::chrono::duration<double>(t1 - t0).count();
            totalTime += iterTime;
            std::cout << "  Iteration " << (i+1) << ": " << std::fixed << std::setprecision(2) << iterTime << "s" << std::endl;
        }
        double avgTime = totalTime / 5.0;
        double tokensPerSec = (args.maxTokens / avgTime);
        std::cout << "Average: " << std::fixed << std::setprecision(2) << avgTime << "s, "
                  << tokensPerSec << " tokens/sec" << std::endl;
    }

    return 0;
}
