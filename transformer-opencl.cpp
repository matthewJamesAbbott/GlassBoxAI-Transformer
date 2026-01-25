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
#include <unordered_map>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>
#include <iomanip>
#include <queue>
#include <memory>
#include <functional>
#include <thread>

#include <CL/cl.h>

#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <netpacket/packet.h>
#include <net/ethernet.h>
#include <linux/if_ether.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/select.h>

#define CL_CHECK(call) \
    do { \
        cl_int err = call; \
        if (err != CL_SUCCESS) { \
            std::cerr << "OpenCL error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << clErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

static const char* clErrorString(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        default: return "UNKNOWN_ERROR";
    }
}

// ============================================================================
// UNSLOTH-STYLE OPENCL KERNELS
// Optimizations: Fused operations for 2x speedup
// ============================================================================

static const char* openclKernelSource = R"(
__kernel void fusedRMSNorm(__global float* output, __global const float* input,
    __global const float* weight, const int dim, const float eps, const int unitOffset) {
    int gid = get_global_id(0);
    if (gid >= dim) return;
    __local float partialSums[256];
    float val = input[gid];
    partialSums[get_local_id(0)] = val * val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0)/2; stride > 0; stride /= 2) {
        if (get_local_id(0) < stride) partialSums[get_local_id(0)] += partialSums[get_local_id(0) + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms_scale = rsqrt(partialSums[0] / dim + eps);
    output[gid] = unitOffset ? input[gid] * rms_scale * (1.0f + weight[gid]) : input[gid] * rms_scale * weight[gid];
}
__kernel void fusedRoPE(__global float* Q, __global float* K, const int qDim, const int kvDim,
    const int headDim, const int position, const float theta, const float ropeScale) {
    int idx = get_global_id(0);
    float scaledPos = (float)position / ropeScale;
    if (idx < qDim/2) {
        int i = idx*2, headIdx = i % headDim;
        float freq = 1.0f / pow(theta, (float)headIdx/headDim);
        float angle = scaledPos * freq, cs = cos(angle), sn = sin(angle);
        float q0 = Q[i], q1 = Q[i+1];
        Q[i] = q0*cs - q1*sn; Q[i+1] = q0*sn + q1*cs;
    }
    if (K && idx < kvDim/2) {
        int i = idx*2, headIdx = i % headDim;
        float freq = 1.0f / pow(theta, (float)headIdx/headDim);
        float angle = scaledPos * freq, cs = cos(angle), sn = sin(angle);
        float k0 = K[i], k1 = K[i+1];
        K[i] = k0*cs - k1*sn; K[i+1] = k0*sn + k1*cs;
    }
}
__kernel void fusedSwiGLU(__global float* output, __global const float* gate,
    __global const float* up, const int size) {
    int i = get_global_id(0);
    if (i >= size) return;
    float g = gate[i];
    output[i] = (g / (1.0f + exp(-g))) * up[i];
}
__kernel void vecMatMul(__global float* out, __global const float* vec,
    __global const float* mat, const int K, const int N) {
    int col = get_global_id(0);
    if (col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += vec[k] * mat[k*N + col];
    out[col] = sum;
}
__kernel void residualAdd(__global float* out, __global const float* residual, const int size) {
    int i = get_global_id(0);
    if (i < size) out[i] += residual[i];
}
)";

// ================================================================================
// QUANTIZATION SUPPORT (from original cuda version)
// K-Quant formats for GGUF model loading
// ================================================================================

enum class GGML_DType : int {
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
    BFLOAT16 = 30,
    UNKNOWN = -1
};

#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32

struct block_q2_K {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    uint16_t d;
    uint16_t dmin;
};

struct block_q3_K {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4];
    uint8_t scales[12];
    uint16_t d;
};

struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
};

struct block_q5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
};

struct block_q6_K {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t scales[QK_K/16];
    uint16_t d;
};

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K/16];
};

struct block_q8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};

#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32

struct block_q4_0 {
    uint16_t d;
    uint8_t qs[QK4_0 / 2];
};

struct block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[QK4_1 / 2];
};

struct block_q5_0 {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
};

struct block_q5_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qh[4];
    uint8_t qs[QK5_1 / 2];
};

inline float fp16_to_fp32(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exponent = (h >> 10) & 0x1F;
    int mantissa = h & 0x3FF;
    
    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        float m = mantissa / 1024.0f;
        float e = -14.0f;
        while (m < 1.0f) { m *= 2.0f; e -= 1.0f; }
        float val = m * powf(2.0f, e);
        return sign ? -val : val;
    } else if (exponent == 31) {
        return mantissa ? NAN : (sign ? -INFINITY : INFINITY);
    }
    float val = (1.0f + mantissa / 1024.0f) * powf(2.0f, exponent - 15.0f);
    return sign ? -val : val;
}

inline float bf16_to_fp32(uint16_t bf) {
    uint32_t val = ((uint32_t)bf) << 16;
    float result;
    memcpy(&result, &val, sizeof(float));
    return result;
}

inline void get_scale_min_k4(int j, const uint8_t* scales, uint8_t* sc, uint8_t* m) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *m  = (scales[j + 4] >>  4) | ((scales[j]     >> 6) << 4);
    }
}

inline void dequant_row_q2_K(const block_q2_K* blocks, float* output, int cols) {
    int nb = cols / QK_K;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        const float dmin = fp16_to_fp32(blocks[i].dmin);
        const uint8_t* qs = blocks[i].qs;
        const uint8_t* sc = blocks[i].scales;
        
        for (int j = 0; j < QK_K/16; ++j) {
            float scale = d * (sc[j] & 0xF);
            float min = dmin * (sc[j] >> 4);
            
            for (int l = 0; l < 16; ++l) {
                int idx = j * 16 + l;
                int byte_idx = idx / 4;
                int shift = (idx % 4) * 2;
                int q = (qs[byte_idx] >> shift) & 3;
                output[i * QK_K + idx] = scale * q - min;
            }
        }
    }
}

inline void dequant_row_q3_K(const block_q3_K* blocks, float* output, int cols) {
    int nb = cols / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    
    for (int i = 0; i < nb; ++i) {
        const float d_all = fp16_to_fp32(blocks[i].d);
        const uint8_t* q = blocks[i].qs;
        const uint8_t* hm = blocks[i].hmask;
        
        uint32_t aux[4];
        memcpy(aux, blocks[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = (const int8_t*)aux;
        
        uint8_t m = 1;
        int is = 0;
        int outIdx = i * QK_K;
        
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int qval = ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                    output[outIdx++] = dl * qval;
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int qval = ((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4);
                    output[outIdx++] = dl * qval;
                }
                shift += 2;
            }
            m <<= 1;
            q += 32;
            hm += 16;
        }
    }
}

inline void dequant_row_q4_K(const block_q4_K* blocks, float* output, int cols) {
    const int nb = cols / QK_K;

    for (int i = 0; i < nb; ++i) {
        const uint8_t* q = blocks[i].qs;
        const float d    = fp16_to_fp32(blocks[i].d);
        const float dmin = fp16_to_fp32(blocks[i].dmin);
        float* y = output + i * QK_K;

        int is = 0;
        uint8_t sc, m;

        for (int n = 0; n < QK_K; n += 64) {
            get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = dmin * m;

            get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = dmin * m;

            for (int l = 0; l < 32; ++l) {
                y[n + l] = d1 * (q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                y[n + 32 + l] = d2 * (q[l] >> 4) - m2;
            }

            q  += 32;
            is += 2;
        }
    }
}

inline void dequant_row_q5_K(const block_q5_K* blocks, float* output, int cols) {
    const int nb = cols / QK_K;

    for (int i = 0; i < nb; ++i) {
        const uint8_t* ql = blocks[i].qs;
        const uint8_t* qh = blocks[i].qh;
        const float d    = fp16_to_fp32(blocks[i].d);
        const float dmin = fp16_to_fp32(blocks[i].dmin);
        float* y = output + i * QK_K;

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;

        for (int n = 0; n < QK_K; n += 64) {
            get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = dmin * m;

            get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = dmin * m;

            for (int l = 0; l < 32; ++l) {
                const int q_base = ql[l] & 0xF;
                const int q_high = (qh[l] & u1) ? 16 : 0;
                y[n + l] = d1 * (q_base + q_high) - m1;
            }

            for (int l = 0; l < 32; ++l) {
                const int q_base = (ql[l] >> 4);
                const int q_high = (qh[l] & u2) ? 16 : 0;
                y[n + 32 + l] = d2 * (q_base + q_high) - m2;
            }

            ql += 32;
            u1 <<= 2;
            u2 <<= 2;
            is += 2;
        }
    }
}

inline void dequant_row_q6_K(const block_q6_K* blocks, float* output, int cols) {
    int nb = cols / QK_K;
    
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        const uint8_t* ql = blocks[i].ql;
        const uint8_t* qh = blocks[i].qh;
        const int8_t* sc = blocks[i].scales;
        float* y = output + i * QK_K;
        
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                
                y[n + l +  0] = d * sc[is + 0] * q1;
                y[n + l + 32] = d * sc[is + 2] * q2;
                y[n + l + 64] = d * sc[is + 4] * q3;
                y[n + l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

inline void dequant_row_q8_0(const block_q8_0* blocks, float* output, int cols) {
    int nb = cols / QK8_0;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            output[i * QK8_0 + j] = d * blocks[i].qs[j];
        }
    }
}

inline void dequant_row_q8_K(const block_q8_K* blocks, float* output, int cols) {
    int nb = cols / QK_K;
    for (int i = 0; i < nb; ++i) {
        const float d = blocks[i].d;
        for (int j = 0; j < QK_K; ++j) {
            output[i * QK_K + j] = d * blocks[i].qs[j];
        }
    }
}

inline void dequant_row_q4_0(const block_q4_0* blocks, float* output, int cols) {
    int nb = cols / QK4_0;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const uint8_t v = blocks[i].qs[j];
            output[i * QK4_0 + j]              = d * ((int)(v & 0x0F) - 8);
            output[i * QK4_0 + j + QK4_0 / 2]  = d * ((int)(v >> 4) - 8);
        }
    }
}

inline void dequant_row_q4_1(const block_q4_1* blocks, float* output, int cols) {
    int nb = cols / QK4_1;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        const float m = fp16_to_fp32(blocks[i].m);
        for (int j = 0; j < QK4_1 / 2; ++j) {
            const uint8_t v = blocks[i].qs[j];
            output[i * QK4_1 + j]              = d * (v & 0x0F) + m;
            output[i * QK4_1 + j + QK4_1 / 2]  = d * (v >> 4) + m;
        }
    }
}

inline void dequant_row_q5_0(const block_q5_0* blocks, float* output, int cols) {
    int nb = cols / QK5_0;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        uint32_t qh;
        memcpy(&qh, blocks[i].qh, sizeof(qh));
        
        for (int j = 0; j < QK5_0 / 2; ++j) {
            const uint8_t v = blocks[i].qs[j];
            const int xh_0 = ((qh >> (j + 0)) & 1) << 4;
            const int xh_1 = ((qh >> (j + 16)) & 1) << 4;
            
            output[i * QK5_0 + j]              = d * ((int)(v & 0x0F) + xh_0 - 16);
            output[i * QK5_0 + j + QK5_0 / 2]  = d * ((int)(v >> 4) + xh_1 - 16);
        }
    }
}

inline void dequant_row_q5_1(const block_q5_1* blocks, float* output, int cols) {
    int nb = cols / QK5_1;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        const float m = fp16_to_fp32(blocks[i].m);
        uint32_t qh;
        memcpy(&qh, blocks[i].qh, sizeof(qh));
        
        for (int j = 0; j < QK5_1 / 2; ++j) {
            const uint8_t v = blocks[i].qs[j];
            const int xh_0 = ((qh >> (j + 0)) & 1) << 4;
            const int xh_1 = ((qh >> (j + 16)) & 1) << 4;
            
            output[i * QK5_1 + j]              = d * ((v & 0x0F) + xh_0) + m;
            output[i * QK5_1 + j + QK5_1 / 2]  = d * ((v >> 4) + xh_1) + m;
        }
    }
}

inline void dequant_row(const void* data, float* output, int cols, int rowIdx, GGML_DType qtype) {
    int blocksPerRow;
    size_t bytesPerBlock = 0;
    
    switch (qtype) {
        case GGML_DType::Q2_K:
            blocksPerRow = cols / QK_K;
            bytesPerBlock = sizeof(block_q2_K);
            dequant_row_q2_K((const block_q2_K*)((const char*)data + rowIdx * blocksPerRow * bytesPerBlock), output, cols);
            break;
        case GGML_DType::Q3_K:
            blocksPerRow = cols / QK_K;
            bytesPerBlock = sizeof(block_q3_K);
            dequant_row_q3_K((const block_q3_K*)((const char*)data + rowIdx * blocksPerRow * bytesPerBlock), output, cols);
            break;
        case GGML_DType::Q4_K:
            blocksPerRow = cols / QK_K;
            bytesPerBlock = sizeof(block_q4_K);
            dequant_row_q4_K((const block_q4_K*)((const char*)data + rowIdx * blocksPerRow * bytesPerBlock), output, cols);
            break;
        case GGML_DType::Q5_K:
            blocksPerRow = cols / QK_K;
            bytesPerBlock = sizeof(block_q5_K);
            dequant_row_q5_K((const block_q5_K*)((const char*)data + rowIdx * blocksPerRow * bytesPerBlock), output, cols);
            break;
        case GGML_DType::Q6_K:
            blocksPerRow = cols / QK_K;
            bytesPerBlock = sizeof(block_q6_K);
            dequant_row_q6_K((const block_q6_K*)((const char*)data + rowIdx * blocksPerRow * bytesPerBlock), output, cols);
            break;
        case GGML_DType::Q8_0:
            blocksPerRow = cols / QK8_0;
            bytesPerBlock = sizeof(block_q8_0);
            dequant_row_q8_0((const block_q8_0*)((const char*)data + rowIdx * blocksPerRow * bytesPerBlock), output, cols);
            break;
        case GGML_DType::Q8_K:
            blocksPerRow = cols / QK_K;
            bytesPerBlock = sizeof(block_q8_K);
            dequant_row_q8_K((const block_q8_K*)((const char*)data + rowIdx * blocksPerRow * bytesPerBlock), output, cols);
            break;
        case GGML_DType::Q4_0:
            blocksPerRow = cols / QK4_0;
            bytesPerBlock = sizeof(block_q4_0);
            dequant_row_q4_0((const block_q4_0*)((const char*)data + rowIdx * blocksPerRow * bytesPerBlock), output, cols);
            break;
        case GGML_DType::Q4_1:
            blocksPerRow = cols / QK4_1;
            bytesPerBlock = sizeof(block_q4_1);
            dequant_row_q4_1((const block_q4_1*)((const char*)data + rowIdx * blocksPerRow * bytesPerBlock), output, cols);
            break;
        case GGML_DType::Q5_0:
            blocksPerRow = cols / QK5_0;
            bytesPerBlock = sizeof(block_q5_0);
            dequant_row_q5_0((const block_q5_0*)((const char*)data + rowIdx * blocksPerRow * bytesPerBlock), output, cols);
            break;
        case GGML_DType::Q5_1:
            blocksPerRow = cols / QK5_1;
            bytesPerBlock = sizeof(block_q5_1);
            dequant_row_q5_1((const block_q5_1*)((const char*)data + rowIdx * blocksPerRow * bytesPerBlock), output, cols);
            break;
        case GGML_DType::F32:
            memcpy(output, (const float*)data + rowIdx * cols, cols * sizeof(float));
            break;
        case GGML_DType::F16:
            for (int j = 0; j < cols; ++j) {
                output[j] = fp16_to_fp32(((const uint16_t*)data)[rowIdx * cols + j]);
            }
            break;
        case GGML_DType::BFLOAT16:
            for (int j = 0; j < cols; ++j) {
                output[j] = bf16_to_fp32(((const uint16_t*)data)[rowIdx * cols + j]);
            }
            break;
        default:
            std::fill(output, output + cols, 0.0f);
            break;
    }
}

inline size_t get_bytes_per_block(GGML_DType qtype) {
    switch (qtype) {
        case GGML_DType::Q2_K: return sizeof(block_q2_K);
        case GGML_DType::Q3_K: return sizeof(block_q3_K);
        case GGML_DType::Q4_K: return sizeof(block_q4_K);
        case GGML_DType::Q5_K: return sizeof(block_q5_K);
        case GGML_DType::Q6_K: return sizeof(block_q6_K);
        case GGML_DType::Q8_K: return sizeof(block_q8_K);
        case GGML_DType::Q8_0: return sizeof(block_q8_0);
        case GGML_DType::Q4_0: return sizeof(block_q4_0);
        case GGML_DType::Q4_1: return sizeof(block_q4_1);
        case GGML_DType::Q5_0: return sizeof(block_q5_0);
        case GGML_DType::Q5_1: return sizeof(block_q5_1);
        case GGML_DType::F32: return 4;
        case GGML_DType::F16: return 2;
        case GGML_DType::BFLOAT16: return 2;
        default: return 0;
    }
}

inline int get_block_size(GGML_DType qtype) {
    switch (qtype) {
        case GGML_DType::Q2_K:
        case GGML_DType::Q3_K:
        case GGML_DType::Q4_K:
        case GGML_DType::Q5_K:
        case GGML_DType::Q6_K:
        case GGML_DType::Q8_K:
            return QK_K;
        case GGML_DType::Q4_0:
        case GGML_DType::Q4_1:
        case GGML_DType::Q5_0:
        case GGML_DType::Q5_1:
        case GGML_DType::Q8_0:
            return 32;
        default:
            return 1;
    }
}

inline const char* get_dtype_name(GGML_DType qtype) {
    switch (qtype) {
        case GGML_DType::F32: return "F32";
        case GGML_DType::F16: return "F16";
        case GGML_DType::Q2_K: return "Q2_K";
        case GGML_DType::Q3_K: return "Q3_K";
        case GGML_DType::Q4_K: return "Q4_K";
        case GGML_DType::Q5_K: return "Q5_K";
        case GGML_DType::Q6_K: return "Q6_K";
        case GGML_DType::Q8_K: return "Q8_K";
        case GGML_DType::Q8_0: return "Q8_0";
        case GGML_DType::Q4_0: return "Q4_0";
        case GGML_DType::Q4_1: return "Q4_1";
        case GGML_DType::Q5_0: return "Q5_0";
        case GGML_DType::Q5_1: return "Q5_1";
        case GGML_DType::BFLOAT16: return "BF16";
        default: return "UNKNOWN";
    }
}

// ================================================================================
// PART 0.5: GGUF LOADER, TOKENIZER, AND TEXT GENERATION
// ================================================================================

struct GGUFTensor {
    std::string name;
    int numDims;
    std::vector<int64_t> shape;
    int dtype;
    int64_t dataOffset;
    size_t dataSize;
};

class GGUFModel {
private:
    std::ifstream stream;
    std::string filename;
    std::vector<GGUFTensor> tensors;
    std::map<std::string, int> tensorMap;
    int64_t tensorDataStart;
    
    std::vector<std::string> ggufTokens;
    std::vector<float> tokenScores;
    
    int embedDim_ = 2048;
    int numLayers_ = 16;
    int numHeads_ = 32;
    int numKVHeads_ = 8;
    int ffnDim_ = 8192;
    int vocabSize_ = 128256;
    int maxSeqLen_ = 131072;
    float ropeTheta_ = 500000.0f;
    float rmsEps_ = 1e-5f;
    int headDim_ = 64;
    std::string architecture_;
    bool loaded_ = false;

    uint32_t readU32() { uint32_t v; stream.read((char*)&v, 4); return v; }
    uint64_t readU64() { uint64_t v; stream.read((char*)&v, 8); return v; }
    int8_t readI8() { int8_t v; stream.read((char*)&v, 1); return v; }
    int32_t readI32() { int32_t v; stream.read((char*)&v, 4); return v; }
    float readF32() { float v; stream.read((char*)&v, 4); return v; }
    
    std::string readString() {
        uint64_t len = readU64();
        if (len > 10000000) return "";
        std::string s(len, '\0');
        stream.read(&s[0], len);
        return s;
    }
    
    void skipValue(int type) {
        switch (type) {
            case 0: case 1: stream.seekg(1, std::ios::cur); break;
            case 2: case 3: stream.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: stream.seekg(4, std::ios::cur); break;
            case 7: stream.seekg(1, std::ios::cur); break;
            case 8: { uint64_t l = readU64(); stream.seekg(l, std::ios::cur); } break;
            case 9: {
                uint32_t at = readU32();
                uint64_t ac = readU64();
                for (uint64_t i = 0; i < ac; i++) skipValue(at);
            } break;
            case 10: case 11: case 12: stream.seekg(8, std::ios::cur); break;
        }
    }
    
    size_t getTypeSize(int dtype) {
        switch (dtype) {
            case 0: return 4;  // F32
            case 1: return 2;  // F16
            case 2: return sizeof(block_q4_0);
            case 3: return sizeof(block_q4_1);
            case 6: return sizeof(block_q5_0);
            case 7: return sizeof(block_q5_1);
            case 8: return sizeof(block_q8_0);
            case 10: return sizeof(block_q2_K);
            case 11: return sizeof(block_q3_K);
            case 12: return sizeof(block_q4_K);
            case 13: return sizeof(block_q5_K);
            case 14: return sizeof(block_q6_K);
            case 15: return sizeof(block_q8_K);
            case 30: return 2;  // BF16
            default: return 4;
        }
    }
    
    int getBlockSize(int dtype) {
        switch (dtype) {
            case 0: case 1: case 30: return 1;
            case 2: case 3: case 6: case 7: case 8: return 32;
            case 10: case 11: case 12: case 13: case 14: case 15: return 256;
            default: return 1;
        }
    }

public:
    bool load(const std::string& path) {
        filename = path;
        stream.open(path, std::ios::binary);
        if (!stream) {
            std::cerr << "Cannot open: " << path << std::endl;
            return false;
        }
        
        char magic[4];
        stream.read(magic, 4);
        if (strncmp(magic, "GGUF", 4) != 0) {
            std::cerr << "Invalid GGUF magic" << std::endl;
            return false;
        }
        
        uint32_t version = readU32();
        uint64_t tensorCount = readU64();
        uint64_t metaCount = readU64();
        
        std::cout << "GGUF v" << version << ": " << tensorCount << " tensors, " 
                  << metaCount << " metadata entries" << std::endl;
        
        for (uint64_t i = 0; i < metaCount; i++) {
            std::string key = readString();
            uint32_t vtype = readU32();
            
            if (key == "general.architecture" && vtype == 8) {
                architecture_ = readString();
            } else if (key.find("embedding_length") != std::string::npos && (vtype == 4 || vtype == 5)) {
                embedDim_ = readU32();
            } else if (key.find("block_count") != std::string::npos && (vtype == 4 || vtype == 5)) {
                numLayers_ = readU32();
            } else if (key.find("head_count_kv") != std::string::npos && (vtype == 4 || vtype == 5)) {
                numKVHeads_ = readU32();
            } else if (key.find("attention.head_count") != std::string::npos && (vtype == 4 || vtype == 5)) {
                numHeads_ = readU32();
            } else if (key.find("feed_forward_length") != std::string::npos && (vtype == 4 || vtype == 5)) {
                ffnDim_ = readU32();
            } else if (key.find("context_length") != std::string::npos && (vtype == 4 || vtype == 5)) {
                maxSeqLen_ = readU32();
            } else if (key.find("rope.freq_base") != std::string::npos && vtype == 6) {
                ropeTheta_ = readF32();
            } else if (key.find("attention.layer_norm_rms_epsilon") != std::string::npos && vtype == 6) {
                rmsEps_ = readF32();
            } else if (key == "tokenizer.ggml.tokens" && vtype == 9) {
                uint32_t arrType = readU32();
                uint64_t arrCount = readU64();
                if (arrType == 8) {
                    ggufTokens.resize(arrCount);
                    for (uint64_t j = 0; j < arrCount; j++) {
                        ggufTokens[j] = readString();
                    }
                    vocabSize_ = arrCount;
                    std::cout << "Loaded " << arrCount << " tokens" << std::endl;
                } else {
                    for (uint64_t j = 0; j < arrCount; j++) skipValue(arrType);
                }
            } else if (key == "tokenizer.ggml.scores" && vtype == 9) {
                uint32_t arrType = readU32();
                uint64_t arrCount = readU64();
                if (arrType == 6) {
                    tokenScores.resize(arrCount);
                    for (uint64_t j = 0; j < arrCount; j++) {
                        tokenScores[j] = readF32();
                    }
                } else {
                    for (uint64_t j = 0; j < arrCount; j++) skipValue(arrType);
                }
            } else {
                skipValue(vtype);
            }
        }
        
        headDim_ = embedDim_ / numHeads_;
        
        tensors.resize(tensorCount);
        for (uint64_t i = 0; i < tensorCount; i++) {
            tensors[i].name = readString();
            tensors[i].numDims = readU32();
            tensors[i].shape.resize(tensors[i].numDims);
            size_t numElements = 1;
            for (int d = 0; d < tensors[i].numDims; d++) {
                tensors[i].shape[d] = readU64();
                numElements *= tensors[i].shape[d];
            }
            tensors[i].dtype = readU32();
            tensors[i].dataOffset = readU64();
            
            int blockSize = getBlockSize(tensors[i].dtype);
            size_t numBlocks = (numElements + blockSize - 1) / blockSize;
            tensors[i].dataSize = numBlocks * getTypeSize(tensors[i].dtype);
            
            tensorMap[tensors[i].name] = i;
        }
        
        tensorDataStart = stream.tellg();
        tensorDataStart = ((tensorDataStart + 31) / 32) * 32;
        
        std::cout << "Architecture: " << architecture_ << std::endl;
        std::cout << "Model: " << numLayers_ << " layers, " << embedDim_ << " dim, "
                  << numHeads_ << " heads (" << numKVHeads_ << " KV), " 
                  << ffnDim_ << " FFN, vocab " << vocabSize_ << std::endl;
        std::cout << "RoPE theta: " << ropeTheta_ << ", RMS eps: " << rmsEps_ << std::endl;
        
        loaded_ = true;
        return true;
    }
    
    std::vector<float> loadTensor(const std::string& name) {
        auto it = tensorMap.find(name);
        if (it == tensorMap.end()) return {};
        
        const GGUFTensor& t = tensors[it->second];
        size_t numElements = 1;
        for (auto d : t.shape) numElements *= d;
        
        stream.seekg(tensorDataStart + t.dataOffset);
        
        std::vector<float> result(numElements);
        
        if (t.dtype == 0) {
            stream.read((char*)result.data(), numElements * sizeof(float));
        } else if (t.dtype == 1) {
            std::vector<uint16_t> fp16(numElements);
            stream.read((char*)fp16.data(), numElements * 2);
            for (size_t i = 0; i < numElements; i++) {
                result[i] = fp16_to_fp32(fp16[i]);
            }
        } else if (t.dtype == 30) {
            std::vector<uint16_t> bf16(numElements);
            stream.read((char*)bf16.data(), numElements * 2);
            for (size_t i = 0; i < numElements; i++) {
                result[i] = bf16_to_fp32(bf16[i]);
            }
        } else {
            std::vector<uint8_t> raw(t.dataSize);
            stream.read((char*)raw.data(), t.dataSize);
            GGML_DType qtype = static_cast<GGML_DType>(t.dtype);
            int blockSize = getBlockSize(t.dtype);
            int numRows = (t.numDims > 1) ? t.shape[0] : 1;
            int cols = numElements / numRows;
            for (int r = 0; r < numRows; r++) {
                dequant_row(raw.data(), result.data() + r * cols, cols, r, qtype);
            }
        }
        return result;
    }
    
    const std::vector<std::string>& getTokens() const { return ggufTokens; }
    int embedDim() const { return embedDim_; }
    int numLayers() const { return numLayers_; }
    int numHeads() const { return numHeads_; }
    int numKVHeads() const { return numKVHeads_; }
    int ffnDim() const { return ffnDim_; }
    int vocabSize() const { return vocabSize_; }
    int maxSeqLen() const { return maxSeqLen_; }
    int headDim() const { return headDim_; }
    float ropeTheta() const { return ropeTheta_; }
    float rmsEps() const { return rmsEps_; }
    const std::string& architecture() const { return architecture_; }
    bool isLoaded() const { return loaded_; }
    const std::vector<GGUFTensor>& getTensors() const { return tensors; }
};

class Tokenizer {
private:
    std::map<std::string, int> tokenToId;
    std::vector<std::string> idToToken;
    int bosId_ = 128000;
    int eosId_ = 128001;
    int eotId_ = 128009;
    int imStartId_ = -1;
    int imEndId_ = -1;
    int startTurnId_ = -1;
    int endTurnId_ = -1;
    bool loaded_ = false;
    std::string modelType_ = "llama";

public:
    bool loadFromGGUF(const std::vector<std::string>& tokens, const std::string& arch = "") {
        if (tokens.empty()) return false;
        idToToken = tokens;
        
        // Auto-detect model type
        if (arch.find("qwen") != std::string::npos) {
            modelType_ = "qwen";
        } else if (arch.find("gemma") != std::string::npos) {
            modelType_ = "gemma";
        } else {
            modelType_ = "llama";
        }
        
        for (int i = 0; i < (int)tokens.size(); i++) {
            tokenToId[tokens[i]] = i;
            // LLaMA 3 tokens
            if (tokens[i] == "<|begin_of_text|>") bosId_ = i;
            if (tokens[i] == "<|end_of_text|>") eosId_ = i;
            if (tokens[i] == "<|eot_id|>") eotId_ = i;
            // Mistral/LLaMA 1-2 tokens (also used by many other models)
            if (tokens[i] == "<s>") bosId_ = i;
            if (tokens[i] == "</s>") { eosId_ = i; eotId_ = i; }
            // DeepSeek tokens (LLaMA-based coder model)
            if (tokens[i].find("begin") != std::string::npos && tokens[i].find("sentence") != std::string::npos) {
                bosId_ = i;
                modelType_ = "deepseek";
            }
            if (tokens[i] == "<|EOT|>" || tokens[i] == "<｜end▁of▁sentence｜>") {
                eosId_ = i;
                eotId_ = i;
                modelType_ = "deepseek";
            }
            // Qwen tokens
            if (tokens[i] == "<|endoftext|>") { 
                if (modelType_ == "qwen") { 
                    eosId_ = i; 
                    bosId_ = i;  // Qwen uses endoftext as BOS too
                } 
            }
            if (tokens[i] == "<|im_start|>") imStartId_ = i;
            if (tokens[i] == "<|im_end|>") { imEndId_ = i; if (modelType_ == "qwen") eotId_ = i; }
            // Gemma tokens
            if (tokens[i] == "<bos>") { if (modelType_ == "gemma") bosId_ = i; }
            if (tokens[i] == "<eos>") { if (modelType_ == "gemma") { eosId_ = i; } }
            if (tokens[i] == "<start_of_turn>") startTurnId_ = i;
            if (tokens[i] == "<end_of_turn>") endTurnId_ = i;
        }
        
        // CRITICAL: Gemma 3 Token 107 fix
        // Many GGUF quants incorrectly include Token 107 (newline) in EOS list
        // Force-override: EOS=1 (actual EOS), EOT=106 (<end_of_turn>)
        // This prevents the "newline loop" where model only outputs token 107
        if (modelType_ == "gemma") {
            if (endTurnId_ > 0) {
                eotId_ = endTurnId_;  // Use <end_of_turn> (usually 106) as EOT
            }
            // Note: eosId_ stays as <eos> token (usually 1), already set above
            std::cout << "Gemma token fix applied: EOS=" << eosId_ << " EOT=" << eotId_ 
                      << " (avoiding token 107 newline loop)" << std::endl;
        }
        
        loaded_ = true;
        std::cout << "Tokenizer: " << tokens.size() << " tokens (" << modelType_ << ")" << std::endl;
        std::cout << "  BOS=" << bosId_ << " EOS=" << eosId_ << " EOT=" << eotId_;
        if (imStartId_ >= 0) std::cout << " IM_START=" << imStartId_ << " IM_END=" << imEndId_;
        if (startTurnId_ >= 0) std::cout << " START_TURN=" << startTurnId_ << " END_TURN=" << endTurnId_;
        std::cout << std::endl;
        return true;
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> result;
        
        // First pass: extract special tokens (delimited by <| and |> or < and >)
        // Special tokens are encoded directly by their token ID, not character-by-character
        std::vector<std::pair<size_t, size_t>> specialRanges;  // (start, end) of special tokens
        std::vector<int> specialIds;
        
        size_t pos = 0;
        while (pos < text.size()) {
            // Check for LLaMA 3 style tokens: <|...|>
            if (pos + 2 < text.size() && text[pos] == '<' && text[pos+1] == '|') {
                size_t end = text.find("|>", pos);
                if (end != std::string::npos) {
                    std::string special = text.substr(pos, end - pos + 2);
                    auto it = tokenToId.find(special);
                    if (it != tokenToId.end()) {
                        specialRanges.push_back({pos, end + 2});
                        specialIds.push_back(it->second);
                        pos = end + 2;
                        continue;
                    }
                }
            }
            // Check for simple special tokens: <...> (e.g., <s>, </s>, <bos>, <eos>)
            if (text[pos] == '<') {
                size_t end = text.find('>', pos);
                if (end != std::string::npos && end - pos <= 32) {
                    std::string special = text.substr(pos, end - pos + 1);
                    auto it = tokenToId.find(special);
                    if (it != tokenToId.end()) {
                        specialRanges.push_back({pos, end + 1});
                        specialIds.push_back(it->second);
                        pos = end + 1;
                        continue;
                    }
                }
            }
            pos++;
        }
        
        // Qwen uses BPE with Ġ (0xC4 0xA0) for leading spaces
        // SentencePiece (Mistral, LLaMA 1/2) uses ▁ (0xE2 0x96 0x81)
        const char* spaceMarker = isQwen() ? "\xC4\xA0" : "\xe2\x96\x81";
        
        // Second pass: encode text between special tokens using BPE
        size_t textPos = 0;
        size_t specialIdx = 0;
        
        while (textPos < text.size()) {
            // Check if we're at a special token
            if (specialIdx < specialRanges.size() && textPos == specialRanges[specialIdx].first) {
                result.push_back(specialIds[specialIdx]);
                textPos = specialRanges[specialIdx].second;
                specialIdx++;
                continue;
            }
            
            // Find the end of the current text segment (up to next special token or end)
            size_t segEnd = text.size();
            if (specialIdx < specialRanges.size()) {
                segEnd = specialRanges[specialIdx].first;
            }
            
            // Process text segment with space markers
            std::string processed;
            bool atWordStart = (textPos == 0 || text[textPos-1] == '\n');
            for (size_t i = textPos; i < segEnd; i++) {
                if (text[i] == ' ') {
                    processed += spaceMarker;
                    atWordStart = true;
                } else if (text[i] == '\n') {
                    processed += text[i];
                    atWordStart = true;
                } else {
                    processed += text[i];
                    atWordStart = false;
                }
            }
            
            // Greedy BPE encoding for this segment
            size_t i = 0;
            while (i < processed.size()) {
                int bestLen = 0, bestId = -1;
                for (size_t len = std::min(processed.size() - i, (size_t)32); len >= 1; len--) {
                    std::string sub = processed.substr(i, len);
                    auto it = tokenToId.find(sub);
                    if (it != tokenToId.end()) {
                        bestLen = len;
                        bestId = it->second;
                        break;
                    }
                }
                if (bestId >= 0) {
                    result.push_back(bestId);
                    i += bestLen;
                } else {
                    unsigned char byte = (unsigned char)processed[i];
                    std::string byteToken = "<0x" + 
                        std::string(1, "0123456789ABCDEF"[byte >> 4]) +
                        std::string(1, "0123456789ABCDEF"[byte & 0xF]) + ">";
                    auto it = tokenToId.find(byteToken);
                    result.push_back(it != tokenToId.end() ? it->second : byte + 3);
                    i++;
                }
            }
            
            textPos = segEnd;
        }
        
        return result;
    }
    
    std::string decode(int id) {
        if (id < 0 || id >= (int)idToToken.size()) return "";
        std::string tok = idToToken[id];
        // Skip special tokens
        if (tok.find("<|") == 0 || tok == "<s>" || tok == "</s>" || 
            tok == "<bos>" || tok == "<eos>" || tok == "<pad>" || tok == "<unk>" ||
            tok == "<start_of_turn>" || tok == "<end_of_turn>") return "";
        size_t pos;
        while ((pos = tok.find("\xC4\xA0")) != std::string::npos) tok.replace(pos, 2, " ");
        while ((pos = tok.find("\xC4\x8A")) != std::string::npos) tok.replace(pos, 2, "\n");
        // Handle Gemma's space marker (▁ = \xe2\x96\x81)
        while ((pos = tok.find("\xe2\x96\x81")) != std::string::npos) tok.replace(pos, 3, " ");
        return tok;
    }
    
    std::string decode(const std::vector<int>& ids) {
        std::string result;
        for (int id : ids) result += decode(id);
        return result;
    }
    
    int bos() const { return bosId_; }
    int eos() const { return eosId_; }
    int eot() const { return eotId_; }
    int imStart() const { return imStartId_; }
    int imEnd() const { return imEndId_; }
    int startTurn() const { return startTurnId_; }
    int endTurn() const { return endTurnId_; }
    int vocabSize() const { return idToToken.size(); }
    bool isLoaded() const { return loaded_; }
    bool isQwen() const { return modelType_ == "qwen"; }
    bool isGemma() const { return modelType_ == "gemma"; }
    const std::string& modelType() const { return modelType_; }
    
    std::string applyChatTemplate(const std::string& userMessage, bool rawMode = false) {
        // Raw mode: no chat template, just the prompt (for base models like Mistral v0.1)
        if (rawMode) {
            return userMessage;
        }
        if (modelType_ == "qwen") {
            // /no_think enables non-thinking mode for faster responses
            return "<|im_start|>user\n" + userMessage + " /no_think<|im_end|>\n<|im_start|>assistant\n";
        } else if (modelType_ == "deepseek") {
            // DeepSeek Coder uses Alpaca-style format
            return "### Instruction:\n" + userMessage + "\n### Response:\n";
        } else if (modelType_ == "gemma") {
            return "<bos><start_of_turn>user\n" + userMessage + "<end_of_turn>\n<start_of_turn>model\n";
        } else if (vocabSize() <= 32001) {
            // Small vocab (Mistral v0.1, LLaMA 1/2) - likely base model, use simple format
            return "<s>" + userMessage;
        } else {
            // LLaMA 3 style chat format
            return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + 
                   userMessage + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        }
    }
    
    bool isDeepSeek() const { return modelType_ == "deepseek"; }
};

struct GenerationConfig {
    int maxTokens = 256;
    float temperature = 0.7f;
    int topK = 40;
    float topP = 0.9f;
    float repPenalty = 1.1f;
};

class TextGenerator {
private:
    GGUFModel* model;
    Tokenizer* tokenizer;
    std::mt19937 rng;
    
    std::vector<float> embeddings;
    std::vector<float> outputWeight;
    std::vector<float> normWeight;
    
    struct LayerWeights {
        std::vector<float> attnNorm, ffnNorm;
        std::vector<float> wq, wk, wv, wo;
        std::vector<float> w1, w2, w3;
        // QK-Norm weights (used by Gemma3 and Qwen3)
        std::vector<float> qNorm, kNorm;
    };
    std::vector<LayerWeights> layers;
    
    std::vector<std::vector<float>> kvCacheK, kvCacheV;
    int cacheLen = 0;

    void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
        float ss = 0;
        for (int i = 0; i < n; i++) ss += x[i] * x[i];
        ss = 1.0f / sqrtf(ss / n + eps);
        for (int i = 0; i < n; i++) out[i] = x[i] * ss * w[i];
    }
    
    void matmul(float* out, const float* x, const float* w, int n, int d) {
        for (int i = 0; i < d; i++) {
            float sum = 0;
            for (int j = 0; j < n; j++) sum += x[j] * w[i * n + j];
            out[i] = sum;
        }
    }
    
    void softmax(float* x, int n) {
        float maxv = x[0];
        for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
        float sum = 0;
        for (int i = 0; i < n; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
        for (int i = 0; i < n; i++) x[i] /= sum;
    }
    
    void rope(float* q, int qDim, float* k, int kDim, int headDim, int pos, float theta) {
        for (int i = 0; i < qDim; i += 2) {
            int headIdx = i % headDim;
            float freq = 1.0f / powf(theta, (float)headIdx / headDim);
            float angle = pos * freq;
            float cs = cosf(angle), sn = sinf(angle);
            float q0 = q[i], q1 = q[i + 1];
            q[i] = q0 * cs - q1 * sn;
            q[i + 1] = q0 * sn + q1 * cs;
        }
        if (k) {
            for (int i = 0; i < kDim; i += 2) {
                int headIdx = i % headDim;
                float freq = 1.0f / powf(theta, (float)headIdx / headDim);
                float angle = pos * freq;
                float cs = cosf(angle), sn = sinf(angle);
                float k0 = k[i], k1 = k[i + 1];
                k[i] = k0 * cs - k1 * sn;
                k[i + 1] = k0 * sn + k1 * cs;
            }
        }
    }
    
    float silu(float x) { return x / (1.0f + expf(-x)); }

public:
    TextGenerator() : model(nullptr), tokenizer(nullptr) {
        rng.seed(std::random_device{}());
    }
    
    bool loadModel(GGUFModel* m, Tokenizer* t) {
        model = m;
        tokenizer = t;
        
        std::cout << "Loading weights..." << std::endl;
        
        embeddings = model->loadTensor("token_embd.weight");
        if (embeddings.empty()) {
            std::cerr << "Failed to load embeddings" << std::endl;
            return false;
        }
        
        outputWeight = model->loadTensor("output.weight");
        if (outputWeight.empty()) outputWeight = embeddings;
        
        normWeight = model->loadTensor("output_norm.weight");
        if (normWeight.empty()) {
            std::cerr << "Failed to load output norm" << std::endl;
            return false;
        }
        
        layers.resize(model->numLayers());
        for (int l = 0; l < model->numLayers(); l++) {
            std::string prefix = "blk." + std::to_string(l) + ".";
            layers[l].attnNorm = model->loadTensor(prefix + "attn_norm.weight");
            layers[l].ffnNorm = model->loadTensor(prefix + "ffn_norm.weight");
            layers[l].wq = model->loadTensor(prefix + "attn_q.weight");
            layers[l].wk = model->loadTensor(prefix + "attn_k.weight");
            layers[l].wv = model->loadTensor(prefix + "attn_v.weight");
            layers[l].wo = model->loadTensor(prefix + "attn_output.weight");
            layers[l].w1 = model->loadTensor(prefix + "ffn_gate.weight");
            layers[l].w2 = model->loadTensor(prefix + "ffn_down.weight");
            layers[l].w3 = model->loadTensor(prefix + "ffn_up.weight");
            
            // QK-Norm weights (used by Gemma3 and Qwen3)
            layers[l].qNorm = model->loadTensor(prefix + "attn_q_norm.weight");
            layers[l].kNorm = model->loadTensor(prefix + "attn_k_norm.weight");
            
            if (layers[l].wq.empty()) {
                std::cerr << "Failed to load layer " << l << std::endl;
                return false;
            }
            std::cout << "\rLoaded layer " << l + 1 << "/" << model->numLayers() << std::flush;
        }
        std::cout << std::endl;
        
        int maxCache = 2048;
        int kvDim = model->headDim() * model->numKVHeads();
        kvCacheK.resize(model->numLayers(), std::vector<float>(maxCache * kvDim, 0));
        kvCacheV.resize(model->numLayers(), std::vector<float>(maxCache * kvDim, 0));
        
        std::cout << "Model loaded successfully" << std::endl;
        return true;
    }
    
    std::vector<float> forward(const std::vector<int>& tokens, int pos) {
        int dim = model->embedDim();
        int nHeads = model->numHeads();
        int nKVHeads = model->numKVHeads();
        int headDim = model->headDim();
        int kvDim = headDim * nKVHeads;
        int nLayers = model->numLayers();
        float eps = model->rmsEps();
        float theta = model->ropeTheta();
        
        std::vector<float> x(dim), xb(dim), xb2(dim);
        std::vector<float> q(dim), k(kvDim), v(kvDim);
        std::vector<float> hb(model->ffnDim()), hb2(model->ffnDim());
        std::vector<float> att(nHeads * 2048);
        
        // Use token at the specified position (not always the last token!)
        int tok = (pos < (int)tokens.size()) ? tokens[pos] : tokens.back();
        if (tok < 0 || tok >= model->vocabSize()) {
            std::cerr << "Invalid token: " << tok << std::endl;
            return {};
        }
        for (int i = 0; i < dim; i++) {
            x[i] = embeddings[(size_t)tok * dim + i];
        }
        
        for (int l = 0; l < nLayers; l++) {
            rmsnorm(xb.data(), x.data(), layers[l].attnNorm.data(), dim, eps);
            
            matmul(q.data(), xb.data(), layers[l].wq.data(), dim, dim);
            matmul(k.data(), xb.data(), layers[l].wk.data(), dim, kvDim);
            matmul(v.data(), xb.data(), layers[l].wv.data(), dim, kvDim);
            
            // QK-Norm (Gemma3, Qwen3): RMSNorm on Q and K before RoPE
            if (!layers[l].qNorm.empty()) {
                for (int h = 0; h < nHeads; h++) {
                    rmsnorm(q.data() + h * headDim, q.data() + h * headDim,
                           layers[l].qNorm.data(), headDim, eps);
                }
                for (int h = 0; h < nKVHeads; h++) {
                    rmsnorm(k.data() + h * headDim, k.data() + h * headDim,
                           layers[l].kNorm.data(), headDim, eps);
                }
            }
            
            rope(q.data(), dim, k.data(), kvDim, headDim, pos, theta);
            
            int cachePos = pos * kvDim;
            for (int i = 0; i < kvDim; i++) {
                kvCacheK[l][cachePos + i] = k[i];
                kvCacheV[l][cachePos + i] = v[i];
            }
            
            int kvMul = nHeads / nKVHeads;
            for (int h = 0; h < nHeads; h++) {
                float* qh = q.data() + h * headDim;
                float* atth = att.data() + h * 2048;
                int kvHead = h / kvMul;
                
                for (int t = 0; t <= pos; t++) {
                    float* kh = kvCacheK[l].data() + t * kvDim + kvHead * headDim;
                    float score = 0;
                    for (int i = 0; i < headDim; i++) score += qh[i] * kh[i];
                    atth[t] = score / sqrtf(headDim);
                }
                
                softmax(atth, pos + 1);
                
                float* xbh = xb.data() + h * headDim;
                std::fill(xbh, xbh + headDim, 0.0f);
                for (int t = 0; t <= pos; t++) {
                    float* vh = kvCacheV[l].data() + t * kvDim + kvHead * headDim;
                    float a = atth[t];
                    for (int i = 0; i < headDim; i++) xbh[i] += a * vh[i];
                }
            }
            
            matmul(xb2.data(), xb.data(), layers[l].wo.data(), dim, dim);
            for (int i = 0; i < dim; i++) x[i] += xb2[i];
            
            rmsnorm(xb.data(), x.data(), layers[l].ffnNorm.data(), dim, eps);
            
            matmul(hb.data(), xb.data(), layers[l].w1.data(), dim, model->ffnDim());
            matmul(hb2.data(), xb.data(), layers[l].w3.data(), dim, model->ffnDim());
            
            for (int i = 0; i < model->ffnDim(); i++) {
                hb[i] = silu(hb[i]) * hb2[i];
            }
            
            matmul(xb.data(), hb.data(), layers[l].w2.data(), model->ffnDim(), dim);
            for (int i = 0; i < dim; i++) x[i] += xb[i];
        }
        
        rmsnorm(x.data(), x.data(), normWeight.data(), dim, eps);
        
        std::vector<float> logits(model->vocabSize());
        matmul(logits.data(), x.data(), outputWeight.data(), dim, model->vocabSize());
        
        return logits;
    }
    
    int sample(std::vector<float>& logits, const GenerationConfig& cfg, 
               const std::vector<int>& prevTokens, bool isGemma = false) {
        int vocabSize = logits.size();
        
        // Gemma: mask "unused" token slots (IDs 3-104) to prevent sampling garbage
        if (isGemma) {
            for (int i = 3; i <= 104; i++) {
                if (i < vocabSize) logits[i] = -INFINITY;
            }
        }
        
        // Apply repetition penalty to tokens that have already appeared
        if (cfg.repPenalty != 1.0f) {
            for (int tok : prevTokens) {
                if (tok >= 0 && tok < vocabSize) {
                    if (logits[tok] > 0) {
                        logits[tok] /= cfg.repPenalty;
                    } else {
                        logits[tok] *= cfg.repPenalty;
                    }
                }
            }
        }
        
        if (cfg.temperature > 0) {
            for (int i = 0; i < vocabSize; i++) logits[i] /= cfg.temperature;
        }
        
        std::vector<std::pair<float, int>> scored(vocabSize);
        for (int i = 0; i < vocabSize; i++) scored[i] = {logits[i], i};
        std::partial_sort(scored.begin(), scored.begin() + cfg.topK, scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        
        float maxLogit = scored[0].first;
        float sum = 0;
        for (int i = 0; i < cfg.topK; i++) {
            scored[i].first = expf(scored[i].first - maxLogit);
            sum += scored[i].first;
        }
        
        float cumSum = 0;
        int cutoff = cfg.topK;
        for (int i = 0; i < cfg.topK; i++) {
            scored[i].first /= sum;
            cumSum += scored[i].first;
            if (cumSum >= cfg.topP) { cutoff = i + 1; break; }
        }
        
        std::uniform_real_distribution<float> dist(0.0f, cumSum);
        float r = dist(rng);
        float acc = 0;
        for (int i = 0; i < cutoff; i++) {
            acc += scored[i].first;
            if (r <= acc) return scored[i].second;
        }
        return scored[0].second;
    }
    
    std::string generate(const std::string& prompt, const GenerationConfig& cfg) {
        std::vector<int> tokens = tokenizer->encode(prompt);
        std::cout << "Prompt tokens: " << tokens.size() << std::endl;
        
        bool isGemma = tokenizer->isGemma();
        
        cacheLen = 0;
        for (auto& kc : kvCacheK) std::fill(kc.begin(), kc.end(), 0.0f);
        for (auto& vc : kvCacheV) std::fill(vc.begin(), vc.end(), 0.0f);
        
        std::string result;
        int generated = 0;
        
        // Prefill: process all prompt tokens to populate KV cache
        for (int pos = 0; pos < (int)tokens.size(); pos++) {
            forward(tokens, pos);
        }
        
        for (int pos = (int)tokens.size() - 1; pos < (int)tokens.size() + cfg.maxTokens; pos++) {
            int nextTok;
            if (pos < (int)tokens.size() - 1) {
                nextTok = tokens[pos + 1];
            } else {
                auto logits = forward(tokens, pos);
                nextTok = sample(logits, cfg, tokens, isGemma);
                tokens.push_back(nextTok);
                
                if (nextTok == tokenizer->eos() || nextTok == tokenizer->eot()) {
                    break;
                }
                
                std::string piece = tokenizer->decode(nextTok);
                result += piece;
                std::cout << piece << std::flush;
                generated++;
                
                if (generated >= cfg.maxTokens) break;
            }
        }
        
        std::cout << std::endl;
        return result;
    }
    
    void clearCache() {
        cacheLen = 0;
        for (auto& kc : kvCacheK) std::fill(kc.begin(), kc.end(), 0.0f);
        for (auto& vc : kvCacheV) std::fill(vc.begin(), vc.end(), 0.0f);
    }
};

// ================================================================================
// PART 1: PROTOCOL DEFINITIONS (from Protocol.h)
// ================================================================================

namespace DistTransformer {

const uint16_t DTX_ETHERTYPE = 0x9998;
const int DTX_MAX_PAYLOAD = 1472;
const int DTX_VERSION = 1;
const int DTX_MAGIC = 0xDEADBEEF;

const int DTX_CONNECT_TIMEOUT = 5000;
const int DTX_FRAME_TIMEOUT = 10000;
const int DTX_RETRY_MAX = 3;

enum class MessageType : uint8_t {
    HANDSHAKE_REQ = 1,
    HANDSHAKE_ACK = 2,
    LAYER_CONFIG = 10,
    LAYER_CONFIG_ACK = 11,
    FORWARD_START = 20,
    FORWARD_CHUNK = 21,
    FORWARD_DONE = 22,
    FORWARD_RESULT = 30,
    FORWARD_COMPLETE = 31,
    BACKWARD_START = 40,
    BACKWARD_CHUNK = 41,
    BACKWARD_DONE = 42,
    BACKWARD_RESULT = 50,
    BACKWARD_COMPLETE = 51,
    PING = 100,
    PONG = 101,
    ERROR_MSG = 200,
    DISCONNECT = 201
};

struct DTXHeader {
    uint32_t magic;
    uint8_t version;
    uint8_t msgType;
    uint16_t sequenceNum;
    uint32_t payloadLen;
    uint32_t checksum;
    uint32_t flags;
    uint32_t reserved;
} __attribute__((packed));

static_assert(sizeof(DTXHeader) == 24, "DTXHeader must be exactly 24 bytes");

struct HandshakeReq {
    uint32_t clientId;
    uint16_t seqBatchSize;
    uint16_t embedDim;
    uint32_t ffnDim;
    uint8_t numHeads;
    uint8_t numKVHeads;
} __attribute__((packed));

struct HandshakeAck {
    uint32_t serverId;
    uint8_t hasGPU;
    uint8_t maxConcurrent;
    uint16_t protocolVer;
} __attribute__((packed));

struct LayerConfig {
    uint8_t startLayer;
    uint8_t numLayers;
    uint8_t keepActivations;
    uint8_t reserved;
    uint32_t totalParams;
} __attribute__((packed));

struct ForwardChunk {
    uint32_t chunkId;
    uint32_t seqStart;
    uint16_t seqLen;
    uint16_t embedDim;
    uint32_t dataSize;
} __attribute__((packed));

struct ForwardResult {
    uint32_t chunkId;
    uint32_t seqStart;
    uint16_t seqLen;
    uint16_t outputDim;
    uint32_t dataSize;
    uint32_t activationSize;
} __attribute__((packed));

struct BackwardChunk {
    uint32_t chunkId;
    uint32_t seqStart;
    uint16_t seqLen;
    uint16_t gradDim;
    uint32_t dataSize;
} __attribute__((packed));

struct BackwardResult {
    uint32_t chunkId;
    uint32_t seqStart;
    uint16_t seqLen;
    uint16_t gradDim;
    uint32_t dataSize;
    uint32_t paramGradSize;
} __attribute__((packed));

struct ErrorMessage {
    uint16_t errorCode;
    uint16_t severity;
    uint32_t contextLen;
} __attribute__((packed));

inline uint32_t crc32_simple(const uint8_t* data, uint32_t len) {
    uint32_t crc = 0xFFFFFFFFU;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320U : 0);
        }
    }
    return crc ^ 0xFFFFFFFFU;
}

inline DTXHeader makeHeader(MessageType type, uint16_t seq,
                           const uint8_t* payload, uint32_t payloadLen) {
    DTXHeader hdr;
    hdr.magic = DTX_MAGIC;
    hdr.version = DTX_VERSION;
    hdr.msgType = static_cast<uint8_t>(type);
    hdr.sequenceNum = seq;
    hdr.payloadLen = payloadLen;
    hdr.checksum = (payload && payloadLen > 0) ? crc32_simple(payload, payloadLen) : 0;
    hdr.flags = 0;
    hdr.reserved = 0;
    return hdr;
}

inline bool verifyHeader(const DTXHeader& hdr) {
    return hdr.magic == static_cast<uint32_t>(DTX_MAGIC) &&
           hdr.version == static_cast<uint8_t>(DTX_VERSION);
}

inline bool verifyChecksum(const DTXHeader& hdr, const uint8_t* payload) {
    if (hdr.payloadLen == 0) return hdr.checksum == 0;
    return crc32_simple(payload, hdr.payloadLen) == hdr.checksum;
}

// ================================================================================
// PART 2: NETWORK LAYER (from TransformerNetwork.h/cpp)
// ================================================================================

struct EthernetFrame {
    uint8_t destMAC[6];
    uint8_t srcMAC[6];
    uint16_t etherType;
    std::vector<uint8_t> payload;

    EthernetFrame() : etherType(DTX_ETHERTYPE) {
        memset(destMAC, 0, 6);
        memset(srcMAC, 0, 6);
    }

    size_t totalSize() const {
        return 12 + 2 + payload.size();
    }
};

enum class ConnectionState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    ERROR
};

// ==================== Utility Functions ====================

bool getMACAddress(const std::string& ifName, uint8_t* mac) {
    std::string path = "/sys/class/net/" + ifName + "/address";
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;

    char buffer[18];
    if (!fgets(buffer, sizeof(buffer), f)) {
        fclose(f);
        return false;
    }

    int ret = sscanf(buffer, "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx",
                     &mac[0], &mac[1], &mac[2], &mac[3], &mac[4], &mac[5]);
    fclose(f);
    return ret == 6;
}

bool compareMACAddress(const uint8_t* mac1, const uint8_t* mac2) {
    return memcmp(mac1, mac2, 6) == 0;
}

void macToString(const uint8_t* mac, char* str, size_t len) {
    snprintf(str, len, "%02x:%02x:%02x:%02x:%02x:%02x",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

bool stringToMAC(const char* str, uint8_t* mac) {
    return sscanf(str, "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx",
                  &mac[0], &mac[1], &mac[2], &mac[3], &mac[4], &mac[5]) == 6;
}

std::vector<float> serializeTensor(const float* data, size_t count) {
    return std::vector<float>(data, data + count);
}

std::vector<uint8_t> packTensorData(const std::vector<float>& data, int) {
    std::vector<uint8_t> packed;
    packed.resize(data.size() * sizeof(float));
    memcpy(packed.data(), data.data(), packed.size());
    return packed;
}

// ==================== Raw Socket Helpers ====================

static int createRawSocket(const std::string& ifName) {
    // Use ETH_P_ALL to capture all Ethernet frames
    // This allows us to filter by EtherType in the application layer
    int s = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (s < 0) {
        std::cerr << "Error: Cannot create raw socket. Need root privileges." << std::endl;
        return -1;
    }

    struct ifreq ifReq;
    memset(&ifReq, 0, sizeof(ifReq));
    strncpy(ifReq.ifr_name, ifName.c_str(), IFNAMSIZ - 1);

    if (ioctl(s, SIOCGIFINDEX, &ifReq) < 0) {
        std::cerr << "Error: Cannot get interface index for: " << ifName << std::endl;
        close(s);
        return -1;
    }

    int ifIndex = ifReq.ifr_ifindex;

    struct sockaddr_ll bindAddr;
    memset(&bindAddr, 0, sizeof(bindAddr));
    bindAddr.sll_family = AF_PACKET;
    // Don't specify protocol in bind - we'll filter by EtherType in receiveRawFrame
    bindAddr.sll_protocol = 0;
    bindAddr.sll_ifindex = ifIndex;

    if (bind(s, (struct sockaddr*)&bindAddr, sizeof(bindAddr)) < 0) {
        std::cerr << "Error: Cannot bind socket to interface: " << ifName << std::endl;
        close(s);
        return -1;
    }

    return s;
}

static bool sendRawFrame(int s, const uint8_t* destMAC, const uint8_t* srcMAC,
                         const std::vector<uint8_t>& payload, const std::string& ifName = "") {
     if (s < 0) {
         std::cerr << "sendRawFrame: socket is invalid" << std::endl;
         return false;
     }

     std::vector<uint8_t> frame(14 + payload.size());
     memcpy(&frame[0], destMAC, 6);
     memcpy(&frame[6], srcMAC, 6);
     uint16_t etherType = htons(DTX_ETHERTYPE);
     memcpy(&frame[12], &etherType, 2);
     memcpy(&frame[14], payload.data(), payload.size());

     struct sockaddr_ll addr;
     memset(&addr, 0, sizeof(addr));
     addr.sll_family = AF_PACKET;
     addr.sll_protocol = 0; // Protocol doesn't matter for sending raw frames

     // Get interface index from name
     if (!ifName.empty()) {
         addr.sll_ifindex = if_nametoindex(ifName.c_str());
         if (addr.sll_ifindex == 0) {
             std::cerr << "Error: Cannot get index for interface " << ifName << std::endl;
             return false;
         }
     } else {
         addr.sll_ifindex = 1; // loopback as fallback
     }

     addr.sll_halen = ETH_ALEN;
     memcpy(addr.sll_addr, destMAC, ETH_ALEN);

     ssize_t sent = sendto(s, frame.data(), frame.size(), 0,
                           (struct sockaddr*)&addr, sizeof(addr));
     if (sent != (ssize_t)frame.size()) {
         std::cerr << "sendto failed: sent " << sent << " bytes, expected " << frame.size() << std::endl;
         return false;
     }
     return true;
}

static bool receiveRawFrame(int s, EthernetFrame& frame, int timeoutMs) {
     if (s < 0) return false;

     fd_set fds;
     FD_ZERO(&fds);
     FD_SET(s, &fds);

     struct timeval tv;
     tv.tv_sec = timeoutMs / 1000;
     tv.tv_usec = (timeoutMs % 1000) * 1000;

     int ret = select(s + 1, &fds, nullptr, nullptr, &tv);
     if (ret <= 0) return false;

     std::vector<uint8_t> buffer(2048);
     struct sockaddr_ll srcAddr;
     socklen_t addrLen = sizeof(srcAddr);

     ssize_t recvLen = recvfrom(s, buffer.data(), buffer.size(), 0,
                                (struct sockaddr*)&srcAddr, &addrLen);
     if (recvLen < 14) return false;

     memcpy(frame.destMAC, &buffer[0], 6);
     memcpy(frame.srcMAC, &buffer[6], 6);
     memcpy(&frame.etherType, &buffer[12], 2);
     frame.etherType = ntohs(frame.etherType);

     // Filter for our custom DTX EtherType
     if (frame.etherType != DTX_ETHERTYPE) {
         return false;
     }

     frame.payload.assign(&buffer[14], &buffer[14] + recvLen - 14);
     return true;
}

// ==================== TransformerServer Implementation ====================

class TransformerServer {
public:
    TransformerServer(const std::string& ifName, uint32_t sId = 0x12345678)
        : interfaceName(ifName), serverId(sId) {}

    ~TransformerServer() {
        if (rawSocket >= 0) close(rawSocket);
    }

    bool initialize() { return bind(interfaceName); }

    bool bind(const std::string& ifName) {
        if (!getMACAddress(ifName, localMAC)) {
            std::cerr << "Error: Cannot get MAC address for " << ifName << std::endl;
            return false;
        }

        rawSocket = createRawSocket(ifName);
        if (rawSocket < 0) return false;

        state = ConnectionState::CONNECTED;
        char macStr[18];
        macToString(localMAC, macStr, sizeof(macStr));
        std::cout << "[Server] Initialized on " << ifName << " (" << macStr << ")" << std::endl;

        return true;
    }

    using ForwardCallback = std::function<std::vector<float>(
        const std::vector<float>&, uint16_t, uint8_t, uint8_t)>;

    using BackwardCallback = std::function<std::vector<float>(
        const std::vector<float>&, uint16_t, uint8_t, uint8_t)>;

    void setForwardCallback(ForwardCallback cb) { forwardCallback = cb; }
    void setBackwardCallback(BackwardCallback cb) { backwardCallback = cb; }

    bool processNextMessage(int timeoutMs = 1000);
    void run(int maxMessages = -1);

    ConnectionState getState() const { return state; }
    uint32_t getClientId() const { return currentClientId; }
    int getConnectedClients() const { return connectedClients.size(); }

    void setMaxClients(int n) { maxConcurrentClients = n; }
    void setGPUAvailable(bool avail) { hasGPU = avail; }

private:
    std::string interfaceName;
    uint32_t serverId;
    int rawSocket = -1;
    uint8_t localMAC[6];
    ConnectionState state = ConnectionState::DISCONNECTED;

    struct ClientSession {
        uint32_t clientId;
        uint8_t clientMAC[6];
        HandshakeReq config;
        std::vector<float> lastActivations;
        uint16_t lastSeqNum = 0;
    };

    std::vector<ClientSession> connectedClients;
    uint32_t currentClientId = 0;
    int maxConcurrentClients = 4;
    bool hasGPU = true;

    ForwardCallback forwardCallback;
    BackwardCallback backwardCallback;

    bool sendFrame(const uint8_t* destMAC, const DTXHeader& hdr, const uint8_t* payload);
    bool receiveFrame(EthernetFrame& frame, int timeoutMs);

    void handleHandshakeReq(const uint8_t* srcMAC, const DTXHeader& hdr, const uint8_t* payload);
    void handleLayerConfig(const uint8_t* srcMAC, const DTXHeader& hdr, const uint8_t* payload);
    void handleForwardChunk(const uint8_t* srcMAC, const DTXHeader& hdr, const uint8_t* payload);
    void handleBackwardChunk(const uint8_t* srcMAC, const DTXHeader& hdr, const uint8_t* payload);
    void handleDisconnect(const uint8_t* srcMAC, const DTXHeader& hdr);
};

bool TransformerServer::processNextMessage(int timeoutMs) {
    EthernetFrame frame;
    if (!receiveRawFrame(rawSocket, frame, timeoutMs)) {
        return false;
    }

    if (frame.payload.size() < sizeof(DTXHeader)) {
        return false;
    }

    DTXHeader hdr;
    memcpy(&hdr, frame.payload.data(), sizeof(DTXHeader));

    if (!verifyHeader(hdr)) {
        return false;
    }

    uint8_t* payloadData = frame.payload.data() + sizeof(DTXHeader);

    if (!verifyChecksum(hdr, payloadData)) {
        std::cerr << "[Server] Checksum mismatch" << std::endl;
        return false;
    }

    MessageType msgType = static_cast<MessageType>(hdr.msgType);

    switch (msgType) {
        case MessageType::HANDSHAKE_REQ:
            handleHandshakeReq(frame.srcMAC, hdr, payloadData);
            break;
        case MessageType::LAYER_CONFIG:
            handleLayerConfig(frame.srcMAC, hdr, payloadData);
            break;
        case MessageType::FORWARD_CHUNK:
            handleForwardChunk(frame.srcMAC, hdr, payloadData);
            break;
        case MessageType::BACKWARD_CHUNK:
            handleBackwardChunk(frame.srcMAC, hdr, payloadData);
            break;
        case MessageType::DISCONNECT:
            handleDisconnect(frame.srcMAC, hdr);
            break;
        default:
            break;
    }

    return true;
}

void TransformerServer::run(int maxMessages) {
    std::cout << "[Server] Running..." << std::endl;
    int count = 0;
    while (maxMessages < 0 || count < maxMessages) {
        processNextMessage(1000);
        count++;
    }
}

void TransformerServer::handleHandshakeReq(const uint8_t* srcMAC, const DTXHeader&, const uint8_t* payload) {
    HandshakeReq req;
    memcpy(&req, payload, sizeof(HandshakeReq));

    ClientSession session;
    session.clientId = req.clientId;
    memcpy(session.clientMAC, srcMAC, 6);
    session.config = req;

    connectedClients.push_back(session);
    currentClientId = req.clientId;

    HandshakeAck ack;
    ack.serverId = serverId;
    ack.hasGPU = hasGPU ? 1 : 0;
    ack.maxConcurrent = maxConcurrentClients;
    ack.protocolVer = DTX_VERSION;

    DTXHeader respHdr = makeHeader(MessageType::HANDSHAKE_ACK, 1,
                                   (const uint8_t*)&ack, sizeof(ack));
    sendFrame(srcMAC, respHdr, (const uint8_t*)&ack);

    char macStr[18];
    macToString(srcMAC, macStr, sizeof(macStr));
    std::cout << "[Server] Client connected: " << macStr << std::endl;
}

void TransformerServer::handleLayerConfig(const uint8_t*, const DTXHeader& hdr, const uint8_t* payload) {
    if (hdr.payloadLen < sizeof(LayerConfig)) {
        return;
    }

    LayerConfig config;
    memcpy(&config, payload, sizeof(LayerConfig));
}

void TransformerServer::handleForwardChunk(const uint8_t* srcMAC, const DTXHeader& hdr, const uint8_t* payload) {
    if (hdr.payloadLen < sizeof(ForwardChunk)) return;

    ForwardChunk chunk;
    memcpy(&chunk, payload, sizeof(ForwardChunk));

    const float* data = (const float*)(payload + sizeof(ForwardChunk));
    std::vector<float> input(data, data + chunk.dataSize / sizeof(float));

    if (forwardCallback) {
        auto result = forwardCallback(input, chunk.seqLen, 0, 1);

        if (!result.empty()) {
            ForwardResult res;
            res.chunkId = chunk.chunkId;
            res.seqStart = chunk.seqStart;
            res.seqLen = chunk.seqLen;
            res.outputDim = chunk.embedDim;
            res.dataSize = result.size() * sizeof(float);
            res.activationSize = 0;

            std::vector<uint8_t> respPayload;
            respPayload.resize(sizeof(ForwardResult) + res.dataSize);
            memcpy(respPayload.data(), &res, sizeof(ForwardResult));
            memcpy(&respPayload[sizeof(ForwardResult)], result.data(), res.dataSize);

            DTXHeader respHdr = makeHeader(MessageType::FORWARD_RESULT, hdr.sequenceNum + 1,
                                          respPayload.data(), respPayload.size());
            sendFrame(srcMAC, respHdr, respPayload.data());
        }
    }
}

void TransformerServer::handleBackwardChunk(const uint8_t* srcMAC, const DTXHeader& hdr, const uint8_t* payload) {
    if (hdr.payloadLen < sizeof(BackwardChunk)) return;

    BackwardChunk chunk;
    memcpy(&chunk, payload, sizeof(BackwardChunk));

    const float* gradData = (const float*)(payload + sizeof(BackwardChunk));
    std::vector<float> gradInput(gradData, gradData + chunk.dataSize / sizeof(float));

    if (backwardCallback) {
        auto result = backwardCallback(gradInput, chunk.seqLen, 0, 1);

        if (!result.empty()) {
            BackwardResult res;
            res.chunkId = chunk.chunkId;
            res.seqStart = chunk.seqStart;
            res.seqLen = chunk.seqLen;
            res.gradDim = chunk.gradDim;
            res.dataSize = result.size() * sizeof(float);
            res.paramGradSize = 0;

            std::vector<uint8_t> respPayload;
            respPayload.resize(sizeof(BackwardResult) + res.dataSize);
            memcpy(respPayload.data(), &res, sizeof(BackwardResult));
            memcpy(&respPayload[sizeof(BackwardResult)], result.data(), res.dataSize);

            DTXHeader respHdr = makeHeader(MessageType::BACKWARD_RESULT, hdr.sequenceNum + 1,
                                          respPayload.data(), respPayload.size());
            sendFrame(srcMAC, respHdr, respPayload.data());
        }
    }
}

void TransformerServer::handleDisconnect(const uint8_t* srcMAC, const DTXHeader&) {
    auto it = std::find_if(connectedClients.begin(), connectedClients.end(),
                          [srcMAC](const ClientSession& s) {
                              return compareMACAddress(s.clientMAC, srcMAC);
                          });

    if (it != connectedClients.end()) {
        char macStr[18];
        macToString(srcMAC, macStr, sizeof(macStr));
        std::cout << "[Server] Client disconnected: " << macStr << std::endl;
        connectedClients.erase(it);
    }
}

bool TransformerServer::sendFrame(const uint8_t* destMAC, const DTXHeader& hdr,
                                   const uint8_t* payload) {
    std::vector<uint8_t> framePayload;
    framePayload.resize(sizeof(DTXHeader) + hdr.payloadLen);
    memcpy(framePayload.data(), &hdr, sizeof(DTXHeader));
    if (payload && hdr.payloadLen > 0) {
        memcpy(&framePayload[sizeof(DTXHeader)], payload, hdr.payloadLen);
    }

    return sendRawFrame(rawSocket, destMAC, localMAC, framePayload, interfaceName);
}

bool TransformerServer::receiveFrame(EthernetFrame& frame, int timeoutMs) {
    return receiveRawFrame(rawSocket, frame, timeoutMs);
}

// ==================== TransformerClient Implementation ====================

class TransformerClient {
public:
    TransformerClient(const std::string& ifName)
        : interfaceName(ifName) {
        memset(serverMAC, 0, 6);
    }

    ~TransformerClient() {
        if (rawSocket >= 0) close(rawSocket);
    }

    bool initialize(const uint8_t* srvMAC);

    void setConfig(uint16_t seqLen, uint16_t embedDim,
                   uint32_t ffnDim, uint8_t numHeads, uint8_t numKVHeads);

    void setLayerConfig(uint8_t startLayer, uint8_t numLayers, bool keepActivations = true);

    std::vector<float> forward(const std::vector<float>& input, uint16_t seqLen);
    std::vector<float> backward(const std::vector<float>& gradOutput, uint16_t seqLen);

    bool connect(int timeoutMs = 5000);
    bool disconnect();
    ConnectionState getState() const { return state; }
    bool isConnected() const { return state == ConnectionState::CONNECTED; }
    uint32_t getServerId() const { return serverId; }

private:
    std::string interfaceName;
    uint8_t localMAC[6];
    uint8_t serverMAC[6];
    int rawSocket = -1;
    ConnectionState state = ConnectionState::DISCONNECTED;

    uint32_t clientId = 0x87654321;
    uint32_t serverId = 0;
    uint16_t sequenceNum = 0;

    HandshakeReq myConfig = {};
    LayerConfig layerCfg = {};

    std::vector<float> forwardBuffer;
    std::vector<float> backwardBuffer;

    bool sendFrame(const DTXHeader& hdr, const uint8_t* payload);
    bool receiveFrame(EthernetFrame& frame, int timeoutMs);
    bool performHandshake(int timeoutMs);

    bool sendTensorChunks(const std::vector<float>& data, uint16_t seqLen,
                          MessageType startType, MessageType chunkType, MessageType doneType);
    std::vector<float> receiveTensorChunks(int timeoutMs);

    uint16_t getNextSeq() { return ++sequenceNum; }
};

bool TransformerClient::initialize(const uint8_t* srvMAC) {
    if (!getMACAddress(interfaceName, localMAC)) {
        std::cerr << "Error: Cannot get MAC address for " << interfaceName << std::endl;
        return false;
    }

    memcpy(serverMAC, srvMAC, 6);

    rawSocket = createRawSocket(interfaceName);
    if (rawSocket < 0) {
        return false;
    }

    char localStr[18], serverStr[18];
    macToString(localMAC, localStr, sizeof(localStr));
    macToString(serverMAC, serverStr, sizeof(serverStr));
    std::cout << "[Client] Initialized on " << interfaceName
              << " (local: " << localStr << ", server: " << serverStr << ")" << std::endl;

    return true;
}

void TransformerClient::setConfig(uint16_t seqLen, uint16_t embedDim,
                                  uint32_t ffnDim, uint8_t numHeads, uint8_t numKVHeads) {
    myConfig.clientId = clientId;
    myConfig.seqBatchSize = seqLen;
    myConfig.embedDim = embedDim;
    myConfig.ffnDim = ffnDim;
    myConfig.numHeads = numHeads;
    myConfig.numKVHeads = numKVHeads;
}

void TransformerClient::setLayerConfig(uint8_t startLayer, uint8_t numLayers, bool keepActivations) {
    layerCfg.startLayer = startLayer;
    layerCfg.numLayers = numLayers;
    layerCfg.keepActivations = keepActivations ? 1 : 0;
}

bool TransformerClient::connect(int timeoutMs) {
    return performHandshake(timeoutMs);
}

bool TransformerClient::disconnect() {
    DTXHeader hdr = makeHeader(MessageType::DISCONNECT, getNextSeq(), nullptr, 0);
    sendFrame(hdr, nullptr);
    state = ConnectionState::DISCONNECTED;
    return true;
}

bool TransformerClient::performHandshake(int timeoutMs) {
    DTXHeader hdr = makeHeader(MessageType::HANDSHAKE_REQ, getNextSeq(),
                               (const uint8_t*)&myConfig, sizeof(myConfig));

    if (!sendFrame(hdr, (const uint8_t*)&myConfig)) {
        std::cerr << "[Client] Failed to send handshake" << std::endl;
        return false;
    }

    EthernetFrame frame;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (std::chrono::high_resolution_clock::now() - startTime <
           std::chrono::milliseconds(timeoutMs)) {
        if (!receiveFrame(frame, 500)) {
            continue;
        }

        if (frame.payload.size() < sizeof(DTXHeader)) {
            continue;
        }

        DTXHeader respHdr;
        memcpy(&respHdr, frame.payload.data(), sizeof(DTXHeader));

        if (respHdr.msgType == static_cast<uint8_t>(MessageType::HANDSHAKE_ACK)) {
            HandshakeAck ack;
            if (frame.payload.size() >= sizeof(DTXHeader) + sizeof(HandshakeAck)) {
                memcpy(&ack, &frame.payload[sizeof(DTXHeader)], sizeof(HandshakeAck));
                serverId = ack.serverId;
                state = ConnectionState::CONNECTED;
                std::cout << "[Client] Connected to server" << std::endl;
                return true;
            }
        }
    }

    std::cerr << "[Client] Handshake timeout" << std::endl;
    return false;
}

std::vector<float> TransformerClient::forward(const std::vector<float>& input, uint16_t seqLen) {
    if (state != ConnectionState::CONNECTED) {
        std::cerr << "[Client] Not connected" << std::endl;
        return {};
    }

    return sendTensorChunks(input, seqLen,
                           MessageType::FORWARD_START,
                           MessageType::FORWARD_CHUNK,
                           MessageType::FORWARD_DONE) ?
           receiveTensorChunks(DTX_FRAME_TIMEOUT) : std::vector<float>();
}

std::vector<float> TransformerClient::backward(const std::vector<float>& gradOutput, uint16_t seqLen) {
    if (state != ConnectionState::CONNECTED) {
        std::cerr << "[Client] Not connected" << std::endl;
        return {};
    }

    return sendTensorChunks(gradOutput, seqLen,
                           MessageType::BACKWARD_START,
                           MessageType::BACKWARD_CHUNK,
                           MessageType::BACKWARD_DONE) ?
           receiveTensorChunks(DTX_FRAME_TIMEOUT) : std::vector<float>();
}

bool TransformerClient::sendTensorChunks(const std::vector<float>& data, uint16_t seqLen,
                                         MessageType startType, MessageType chunkType,
                                         MessageType doneType) {
    DTXHeader startHdr = makeHeader(startType, getNextSeq(), nullptr, 0);
    if (!sendFrame(startHdr, nullptr)) {
        return false;
    }

    uint32_t chunkId = 0;
    size_t offset = 0;
    size_t elementsPerChunk = (DTX_MAX_PAYLOAD - sizeof(ForwardChunk)) / sizeof(float);

    while (offset < data.size()) {
        size_t chunkSize = std::min(elementsPerChunk, data.size() - offset);

        ForwardChunk chunk;
        chunk.chunkId = chunkId++;
        chunk.seqStart = 0;
        chunk.seqLen = seqLen;
        chunk.embedDim = myConfig.embedDim;
        chunk.dataSize = chunkSize * sizeof(float);

        std::vector<uint8_t> payload;
        payload.resize(sizeof(ForwardChunk) + chunk.dataSize);
        memcpy(payload.data(), &chunk, sizeof(ForwardChunk));
        memcpy(&payload[sizeof(ForwardChunk)], &data[offset], chunk.dataSize);

        DTXHeader chunkHdr = makeHeader(chunkType, getNextSeq(), payload.data(), payload.size());
        if (!sendFrame(chunkHdr, payload.data())) {
            return false;
        }

        offset += chunkSize;
    }

    DTXHeader doneHdr = makeHeader(doneType, getNextSeq(), nullptr, 0);
    return sendFrame(doneHdr, nullptr);
}

std::vector<float> TransformerClient::receiveTensorChunks(int timeoutMs) {
    std::vector<float> result;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (std::chrono::high_resolution_clock::now() - startTime <
           std::chrono::milliseconds(timeoutMs)) {
        EthernetFrame frame;
        if (!receiveFrame(frame, 500)) {
            continue;
        }

        if (frame.payload.size() < sizeof(DTXHeader)) {
            continue;
        }

        DTXHeader hdr;
        memcpy(&hdr, frame.payload.data(), sizeof(DTXHeader));

        if (hdr.msgType == static_cast<uint8_t>(MessageType::FORWARD_RESULT) ||
            hdr.msgType == static_cast<uint8_t>(MessageType::BACKWARD_RESULT)) {

            ForwardResult res;
            memcpy(&res, &frame.payload[sizeof(DTXHeader)], sizeof(ForwardResult));

            size_t dataOffset = sizeof(DTXHeader) + sizeof(ForwardResult);
            const float* data = (const float*)&frame.payload[dataOffset];
            result.insert(result.end(), data, data + res.dataSize / sizeof(float));
        } else if (hdr.msgType == static_cast<uint8_t>(MessageType::FORWARD_COMPLETE) ||
                   hdr.msgType == static_cast<uint8_t>(MessageType::BACKWARD_COMPLETE)) {
            break;
        }
    }

    return result;
}

bool TransformerClient::sendFrame(const DTXHeader& hdr, const uint8_t* payload) {
    std::vector<uint8_t> framePayload;
    framePayload.resize(sizeof(DTXHeader) + hdr.payloadLen);
    memcpy(framePayload.data(), &hdr, sizeof(DTXHeader));
    if (payload && hdr.payloadLen > 0) {
        memcpy(&framePayload[sizeof(DTXHeader)], payload, hdr.payloadLen);
    }

    return sendRawFrame(rawSocket, serverMAC, localMAC, framePayload, interfaceName);
}

bool TransformerClient::receiveFrame(EthernetFrame& frame, int timeoutMs) {
    return receiveRawFrame(rawSocket, frame, timeoutMs);
}

// ================================================================================
// PART 3: DISTRIBUTED TRANSFORMER (from DistributedTransformer.h/cpp)
// ================================================================================

struct DistributedConfig {
    int seqLen = 512;
    int embedDim = 768;
    int ffnDim = 3072;
    int numHeads = 12;
    int numKVHeads = 12;
    int totalLayers = 12;

    int localLayers = 6;
    int remoteLayers = 6;
    int startRemoteLayer = 6;

    bool cacheActivations = true;
    bool cacheGradients = true;

    std::string interfaceName = "eth0";
    uint8_t serverMAC[6] = {0};

    bool validate() const {
        return (localLayers + remoteLayers) == totalLayers &&
               startRemoteLayer >= 0 &&
               startRemoteLayer + remoteLayers == totalLayers;
    }
};

DistributedConfig parseConfigString(const std::string& configStr) {
    DistributedConfig cfg;
    std::istringstream iss(configStr);
    std::string token;

    while (std::getline(iss, token, ',')) {
        size_t eqPos = token.find('=');
        if (eqPos == std::string::npos) continue;

        std::string key = token.substr(0, eqPos);
        std::string value = token.substr(eqPos + 1);

        try {
            if (key == "seq") cfg.seqLen = std::stoi(value);
            else if (key == "embed") cfg.embedDim = std::stoi(value);
            else if (key == "ffn") cfg.ffnDim = std::stoi(value);
            else if (key == "heads") cfg.numHeads = std::stoi(value);
            else if (key == "kvheads") cfg.numKVHeads = std::stoi(value);
            else if (key == "total") cfg.totalLayers = std::stoi(value);
            else if (key == "local") cfg.localLayers = std::stoi(value);
            else if (key == "remote") cfg.remoteLayers = std::stoi(value);
        } catch (...) {}
    }

    return cfg;
}

DistributedConfig createSymmetricConfig(int totalLayers, int embedDim,
                                       int ffnDim, int numHeads) {
    DistributedConfig cfg;
    cfg.totalLayers = totalLayers;
    cfg.embedDim = embedDim;
    cfg.ffnDim = ffnDim;
    cfg.numHeads = numHeads;
    cfg.numKVHeads = numHeads;

    cfg.localLayers = totalLayers / 2;
    cfg.remoteLayers = totalLayers - cfg.localLayers;
    cfg.startRemoteLayer = cfg.localLayers;

    return cfg;
}

class DistributedTransformer {
public:
    explicit DistributedTransformer(const DistributedConfig& cfg)
        : config(cfg) {
        activationCache.resize(config.totalLayers);
    }

    ~DistributedTransformer() {
        if (client && client->isConnected()) {
            client->disconnect();
        }
    }

    bool initialize();
    bool connect(int timeoutMs = 5000);
    bool disconnect();

    std::vector<float> forward(const std::vector<float>& input);
    std::vector<float> backward(const std::vector<float>& gradOutput);

    bool isConnected() const { return client && client->isConnected(); }
    const DistributedConfig& getConfig() const { return config; }

    std::vector<float> forwardLocal(const std::vector<float>& input, int startLayer, int numLayers);
    std::vector<float> backwardLocal(const std::vector<float>& gradOutput, int startLayer, int numLayers);

    void cacheActivation(uint32_t layer, const std::vector<float>& activation);
    std::vector<float> getActivation(uint32_t layer) const;

private:
    DistributedConfig config;
    std::unique_ptr<TransformerClient> client;
    std::vector<std::vector<float>> activationCache;
};

bool DistributedTransformer::initialize() {
    if (!config.validate()) {
        std::cerr << "Invalid configuration: local + remote != total" << std::endl;
        return false;
    }

    client.reset(new TransformerClient(config.interfaceName));

    if (!client->initialize(config.serverMAC)) {
        std::cerr << "Failed to initialize network client" << std::endl;
        return false;
    }

    client->setConfig(config.seqLen, config.embedDim, config.ffnDim,
                     config.numHeads, config.numKVHeads);

    client->setLayerConfig(config.startRemoteLayer, config.remoteLayers,
                          config.cacheActivations);

    std::cout << "[DistTransformer] Initialized" << std::endl;
    std::cout << "  Local layers: 0-" << (config.startRemoteLayer - 1) << std::endl;
    std::cout << "  Remote layers: " << config.startRemoteLayer << "-"
              << (config.startRemoteLayer + config.remoteLayers - 1) << std::endl;

    return true;
}

bool DistributedTransformer::connect(int timeoutMs) {
    if (!client->connect(timeoutMs)) {
        std::cerr << "Failed to connect to remote server" << std::endl;
        return false;
    }
    return true;
}

bool DistributedTransformer::disconnect() {
    if (client) {
        return client->disconnect();
    }
    return true;
}

std::vector<float> DistributedTransformer::forward(const std::vector<float>& input) {
    if (!isConnected()) {
        std::cerr << "Not connected to remote server" << std::endl;
        return {};
    }

    std::vector<float> intermediate = input;
    if (config.startRemoteLayer > 0) {
        intermediate = forwardLocal(input, 0, config.startRemoteLayer);
        if (intermediate.empty()) {
            return {};
        }
    }

    std::vector<float> output = client->forward(intermediate, config.seqLen);

    if (config.cacheActivations && !output.empty()) {
        cacheActivation(config.startRemoteLayer + config.remoteLayers - 1, output);
    }

    return output;
}

std::vector<float> DistributedTransformer::backward(const std::vector<float>& gradOutput) {
    if (!isConnected()) {
        std::cerr << "Not connected to remote server" << std::endl;
        return {};
    }

    std::vector<float> grad = client->backward(gradOutput, config.seqLen);

    if (grad.empty()) {
        return {};
    }

    if (config.localLayers > 0) {
        grad = backwardLocal(grad, 0, config.localLayers);
    }

    return grad;
}

std::vector<float> DistributedTransformer::forwardLocal(const std::vector<float>& input,
                                                        int startLayer, int numLayers) {
    std::cout << "[DistTransformer] Forward local layers " << startLayer
              << "-" << (startLayer + numLayers - 1) << std::endl;
    return input;
}

std::vector<float> DistributedTransformer::backwardLocal(const std::vector<float>& gradOutput,
                                                        int startLayer, int numLayers) {
    std::cout << "[DistTransformer] Backward local layers " << startLayer
              << "-" << (startLayer + numLayers - 1) << std::endl;
    return gradOutput;
}

void DistributedTransformer::cacheActivation(uint32_t layer, const std::vector<float>& activation) {
    if (layer < activationCache.size()) {
        activationCache[layer] = activation;
    }
}

std::vector<float> DistributedTransformer::getActivation(uint32_t layer) const {
    if (layer < activationCache.size()) {
        return activationCache[layer];
    }
    return {};
}

class DistributedTransformerServer {
public:
    explicit DistributedTransformerServer(const DistributedConfig& cfg)
        : config(cfg) {}

    ~DistributedTransformerServer() {}

    bool initialize();
    void run(int maxMessages = -1);
    bool processOneMessage(int timeoutMs = 1000);

    using LayerFunction = std::function<std::vector<float>(
        const std::vector<float>&, int, bool)>;

    void setForwardLayerFunction(LayerFunction fn) { forwardLayerFn = fn; }
    void setBackwardLayerFunction(LayerFunction fn) { backwardLayerFn = fn; }

    bool isRunning() const { return server && server->getState() == ConnectionState::CONNECTED; }
    const DistributedConfig& getConfig() const { return config; }

private:
    DistributedConfig config;
    std::unique_ptr<TransformerServer> server;

    LayerFunction forwardLayerFn;
    LayerFunction backwardLayerFn;

    std::vector<float> executeForward(const std::vector<float>& input, int startLayer, int numLayers);
    std::vector<float> executeBackward(const std::vector<float>& gradOutput, int startLayer, int numLayers);
};

bool DistributedTransformerServer::initialize() {
    if (!config.validate()) {
        std::cerr << "Invalid server configuration" << std::endl;
        return false;
    }

    server.reset(new TransformerServer(config.interfaceName));

    if (!server->initialize()) {
        std::cerr << "Failed to initialize network server" << std::endl;
        return false;
    }

    server->setForwardCallback([this](const std::vector<float>& input,
                                     uint16_t seqLen,
                                     uint8_t startLayer,
                                     uint8_t numLayers) {
        return executeForward(input, startLayer, numLayers);
    });

    server->setBackwardCallback([this](const std::vector<float>& gradOutput,
                                      uint16_t seqLen,
                                      uint8_t startLayer,
                                      uint8_t numLayers) {
        return executeBackward(gradOutput, startLayer, numLayers);
    });

    std::cout << "[DistTransformerServer] Initialized on " << config.interfaceName << std::endl;
    std::cout << "  Will execute layers " << (int)config.startRemoteLayer << "-"
              << (int)(config.startRemoteLayer + config.remoteLayers - 1) << std::endl;

    return true;
}

void DistributedTransformerServer::run(int maxMessages) {
    std::cout << "[DistTransformerServer] Running..." << std::endl;
    server->run(maxMessages);
}

bool DistributedTransformerServer::processOneMessage(int timeoutMs) {
    return server->processNextMessage(timeoutMs);
}

std::vector<float> DistributedTransformerServer::executeForward(const std::vector<float>& input,
                                                               int startLayer, int numLayers) {
    std::cout << "[Server] Forward pass layers " << startLayer << "-"
              << (startLayer + numLayers - 1) << std::endl;

    std::vector<float> output = input;

    for (int layer = startLayer; layer < startLayer + numLayers; layer++) {
        if (forwardLayerFn) {
            output = forwardLayerFn(output, layer, true);
            if (output.empty()) {
                std::cerr << "[Server] Layer " << layer << " failed" << std::endl;
                return {};
            }
        }
    }

    return output;
}

std::vector<float> DistributedTransformerServer::executeBackward(const std::vector<float>& gradOutput,
                                                                int startLayer, int numLayers) {
    std::cout << "[Server] Backward pass layers " << startLayer << "-"
              << (startLayer + numLayers - 1) << std::endl;

    std::vector<float> grad = gradOutput;

    for (int layer = startLayer + numLayers - 1; layer >= startLayer; layer--) {
        if (backwardLayerFn) {
            grad = backwardLayerFn(grad, layer, true);
            if (grad.empty()) {
                std::cerr << "[Server] Backward layer " << layer << " failed" << std::endl;
                return {};
            }
        }
    }

    return grad;
}

struct TimingStats {
    double forwardMs = 0;
    double backwardMs = 0;
    double totalMs = 0;
    size_t elementsProcessed = 0;
};

TimingStats benchmarkDistributed(DistributedTransformer& transformer, int iterations = 10) {
    TimingStats stats;

    size_t inputSize = transformer.getConfig().seqLen * transformer.getConfig().embedDim;
    std::vector<float> input(inputSize, 1.0f);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        auto output = transformer.forward(input);
        if (output.empty()) {
            std::cerr << "Forward pass failed at iteration " << i << std::endl;
            return stats;
        }
        input = output;
    }

    auto afterForward = std::chrono::high_resolution_clock::now();

    std::vector<float> gradOutput(inputSize, 0.1f);
    for (int i = 0; i < iterations; i++) {
        auto grad = transformer.backward(gradOutput);
        if (grad.empty()) {
            std::cerr << "Backward pass failed at iteration " << i << std::endl;
            return stats;
        }
        gradOutput = grad;
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    auto forwardMs = std::chrono::duration<double, std::milli>(afterForward - startTime).count();
    auto backwardMs = std::chrono::duration<double, std::milli>(endTime - afterForward).count();

    stats.forwardMs = forwardMs / iterations;
    stats.backwardMs = backwardMs / iterations;
    stats.totalMs = (forwardMs + backwardMs) / iterations;
    stats.elementsProcessed = inputSize;

    return stats;
}

} // namespace DistTransformer

// ================================================================================
// PART 4: OpenCL KERNELS FOR LAYER COMPUTATION
// ================================================================================

namespace DistTransformer {

std::string getOpenCLKernels() {
    return R"CL(
__kernel void matmul_fp32(
    __global const float* A, __global const float* B, __global float* C,
    int M, int N, int K, __global const float* bias) {
    int i = get_global_id(1);
    int j = get_global_id(0);
    
    if (i >= M || j >= N) return;
    
    float sum = (bias != NULL) ? bias[j] : 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

__kernel void gelu_fp32(const __global float* input, __global float* output, int size) {
    int i = get_global_id(0);
    if (i >= size) return;
    
    float x = input[i];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[i] = x * cdf;
}

__kernel void softmax_fp32(__global float* data, int rows, int cols) {
    int row = get_global_id(0);
    int idx = get_global_id(1);
    
    if (row >= rows) return;
    
    __global float* rowData = data + row * cols;
    
    float maxVal = rowData[0];
    for (int i = 1; i < cols; i++) {
        maxVal = max(maxVal, rowData[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        rowData[i] = exp(rowData[i] - maxVal);
        sum += rowData[i];
    }
    
    for (int i = 0; i < cols; i++) {
        rowData[i] /= sum;
    }
}
)CL";
}

} // namespace DistTransformer

// ================================================================================
// MAIN - NETWORK TEST HARNESS
// ================================================================================

void printMainHelp(const char* progName) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Distributed Transformer - Layer 2 Ethernet + OpenCL         ║" << std::endl;
    std::cout << "║     Protocol + Network Layer + GPU Compute (Single File)         ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nUSAGE: " << progName << " <command> [options]\n" << std::endl;
    std::cout << "COMMANDS:\n" << std::endl;
    
    std::cout << "  server                    Start as Transformer server" << std::endl;
    std::cout << "    -i, --interface <name>  Network interface (default: eth0)" << std::endl;
    std::cout << "    -l, --layers <n>        Total transformer layers (default: 12)" << std::endl;
    std::cout << "    -e, --embed <dim>       Embedding dimension (default: 768)" << std::endl;
    std::cout << "    -f, --ffn <dim>         FFN hidden dimension (default: 3072)" << std::endl;
    std::cout << "    -a, --heads <n>         Number of attention heads (default: 12)" << std::endl;
    std::cout << "    -k, --kvheads <n>       Number of KV heads for GQA (default: 12)" << std::endl;
    std::cout << "    -q, --seq-len <n>       Sequence length (default: 512)" << std::endl;
    std::cout << "    -v, --vocab-size <n>    Vocabulary size (default: 50257)" << std::endl;
    std::cout << "    -x, --max-seq-len <n>   Maximum sequence length (default: 2048)" << std::endl;
    std::cout << "    -m, --messages <n>      Max messages to process (default: 100)" << std::endl;
    std::cout << "    -g, --gpu <yes/no>      GPU availability (default: yes)" << std::endl;
    std::cout << "    -c, --clients <n>       Max concurrent clients (default: 4)" << std::endl;
    std::cout << "    --quant <type>          Quantization type: none|q4_0|q4_1|q5_0|q5_1|q8_0|" << std::endl;
    std::cout << "                            q2_k|q3_k|q4_k|q5_k|q6_k|q8_k|f16|bf16 (default: none)" << std::endl;
    std::cout << "    --rope-base <n>         RoPE base frequency (default: 10000.0)" << std::endl;
    std::cout << "    --rope-scale <n>        RoPE scaling factor (default: 1.0)" << std::endl;
    std::cout << "    --eps <n>               Layer norm epsilon (default: 1e-5)" << std::endl;
    std::cout << "    --dropout <n>           Dropout rate 0.0-1.0 (default: 0.0)" << std::endl;
    std::cout << "    --verbose               Enable verbose output" << std::endl;
    std::cout << "    --help                  Show server help\n" << std::endl;

    std::cout << "  client                    Start as Transformer client" << std::endl;
    std::cout << "    -i, --interface <name>  Network interface (default: eth0)" << std::endl;
    std::cout << "    -s, --server <mac>      Server MAC address (required, format: XX:XX:XX:XX:XX:XX)" << std::endl;
    std::cout << "    -l, --layers <n>        Total transformer layers (default: 12)" << std::endl;
    std::cout << "    -r, --remote <n>        Remote layers to execute (default: 6)" << std::endl;
    std::cout << "    --start-layer <n>       Starting layer for remote execution (default: auto)" << std::endl;
    std::cout << "    -e, --embed <dim>       Embedding dimension (default: 768)" << std::endl;
    std::cout << "    -f, --ffn <dim>         FFN hidden dimension (default: 3072)" << std::endl;
    std::cout << "    -a, --heads <n>         Number of attention heads (default: 12)" << std::endl;
    std::cout << "    -k, --kvheads <n>       KV heads for GQA (default: 12)" << std::endl;
    std::cout << "    -q, --seq-len <n>       Sequence length (default: 512)" << std::endl;
    std::cout << "    -v, --vocab-size <n>    Vocabulary size (default: 50257)" << std::endl;
    std::cout << "    -x, --max-seq-len <n>   Maximum sequence length (default: 2048)" << std::endl;
    std::cout << "    --quant <type>          Quantization type (see server options)" << std::endl;
    std::cout << "    --rope-base <n>         RoPE base frequency (default: 10000.0)" << std::endl;
    std::cout << "    --rope-scale <n>        RoPE scaling factor (default: 1.0)" << std::endl;
    std::cout << "    --eps <n>               Layer norm epsilon (default: 1e-5)" << std::endl;
    std::cout << "    --no-cache              Disable activation caching" << std::endl;
    std::cout << "    --no-grad-cache         Disable gradient caching" << std::endl;
    std::cout << "    --timeout <ms>          Connection timeout (default: 5000ms)" << std::endl;
    std::cout << "    --retries <n>           Connection retry count (default: 3)" << std::endl;
    std::cout << "    --verbose               Enable verbose output" << std::endl;
    std::cout << "    --help                  Show client help\n" << std::endl;

    std::cout << "  benchmark                 Run benchmark suite" << std::endl;
    std::cout << "    -i, --interface <name>  Network interface (default: eth0)" << std::endl;
    std::cout << "    -s, --server <mac>      Server MAC address (required)" << std::endl;
    std::cout << "    -n, --iterations <n>    Benchmark iterations (default: 10)" << std::endl;
    std::cout << "    -l, --layers <n>        Transformer layers to benchmark (default: 12)" << std::endl;
    std::cout << "    -e, --embed <dim>       Embedding dimension (default: 768)" << std::endl;
    std::cout << "    -q, --seq-len <n>       Sequence length (default: 512)" << std::endl;
    std::cout << "    --batch-size <n>        Batch size for benchmarking (default: 1)" << std::endl;
    std::cout << "    --warmup <n>            Warmup iterations (default: 2)" << std::endl;
    std::cout << "    --output <file>         Output results to CSV file" << std::endl;
    std::cout << "    --verbose               Enable verbose output" << std::endl;
    std::cout << "    --help                  Show benchmark help\n" << std::endl;

    std::cout << "  generate                  Generate text from a prompt (single mode)" << std::endl;
    std::cout << "    -m, --model <path>      Path to GGUF model file (required)" << std::endl;
    std::cout << "    -p, --prompt <text>     Text prompt for generation" << std::endl;
    std::cout << "    -n, --tokens <n>        Max tokens to generate (default: 256)" << std::endl;
    std::cout << "    -t, --temperature <n>   Sampling temperature (default: 0.7)" << std::endl;
    std::cout << "    --top-k <n>             Top-K sampling (default: 40)" << std::endl;
    std::cout << "    --top-p <n>             Top-P/nucleus sampling (default: 0.9)" << std::endl;
    std::cout << "    -i, --interactive       Interactive chat mode" << std::endl;
    std::cout << "    --help                  Show generate help\n" << std::endl;

    std::cout << "  test                      Run unit tests" << std::endl;
    std::cout << "    --all                   Run all tests" << std::endl;
    std::cout << "    --protocol              Test protocol handling" << std::endl;
    std::cout << "    --config                Test configuration" << std::endl;
    std::cout << "    --quant                 Test quantization/dequantization" << std::endl;
    std::cout << "    --kernels               Test OpenCL kernels" << std::endl;
    std::cout << "    --network               Test network layer" << std::endl;
    std::cout << "    --verbose               Enable verbose test output" << std::endl;
    std::cout << "    --help                  Show test help\n" << std::endl;

    std::cout << "QUANTIZATION TYPES:\n" << std::endl;
    std::cout << "  none                      Full precision float32 (32 bpw)" << std::endl;
    std::cout << "  f16                       Half precision float16 (16 bpw)" << std::endl;
    std::cout << "  bf16                      Brain float16 (16 bpw)" << std::endl;
    std::cout << "  q8_0                      8-bit quantization (8.5 bpw)" << std::endl;
    std::cout << "  q6_k                      6-bit K-quant (6.5625 bpw)" << std::endl;
    std::cout << "  q5_k                      5-bit K-quant (5.5 bpw)" << std::endl;
    std::cout << "  q4_k                      4-bit K-quant (4.5 bpw)" << std::endl;
    std::cout << "  q3_k                      3-bit K-quant (3.4375 bpw)" << std::endl;
    std::cout << "  q2_k                      2-bit K-quant (2.625 bpw)" << std::endl;
    std::cout << "  q4_0, q4_1, q5_0, q5_1    Legacy quantization formats\n" << std::endl;

    std::cout << "GLOBAL OPTIONS:\n" << std::endl;
    std::cout << "  --help, -h                Show this help message" << std::endl;
    std::cout << "  --version                 Show version information\n" << std::endl;

    std::cout << "EXAMPLES:\n" << std::endl;
    std::cout << "  # Start server on eth0 with 24 layers and Q4_K quantization" << std::endl;
    std::cout << "  " << progName << " server -i eth0 -l 24 -e 1024 --quant q4_k\n" << std::endl;
    std::cout << "  # Connect client with custom sequence length and vocab" << std::endl;
    std::cout << "  " << progName << " client -s AA:BB:CC:DD:EE:FF -q 1024 -v 32000 -r 12\n" << std::endl;
    std::cout << "  # Run benchmarks with warmup and output file" << std::endl;
    std::cout << "  " << progName << " benchmark -s AA:BB:CC:DD:EE:FF -n 100 --warmup 5 --output bench.csv\n" << std::endl;
    std::cout << "  # Run quantization tests" << std::endl;
    std::cout << "  " << progName << " test --quant --verbose\n" << std::endl;
}

std::string toLowerCase(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printMainHelp(argv[0]);
        return 1;
    }

    std::string command = argv[1];
    
    if (command == "-h" || command == "--help") {
        printMainHelp(argv[0]);
        return 0;
    }

    if (command == "--version") {
        std::cout << "Distributed Transformer v1.0.0 (OpenCL)" << std::endl;
        std::cout << "Vendor-agnostic Layer 2 Ethernet distributed execution" << std::endl;
        std::cout << "Copyright (c) 2025 Matthew Abbott" << std::endl;
        std::cout << "MIT License - Free for the world" << std::endl;
        return 0;
    }

    if (command == "server") {
        DistTransformer::DistributedConfig cfg;
        cfg.totalLayers = 12;
        cfg.localLayers = 0;
        cfg.remoteLayers = 12;
        cfg.embedDim = 768;
        cfg.ffnDim = 3072;
        cfg.numHeads = 12;
        cfg.numKVHeads = 12;
        cfg.seqLen = 512;
        cfg.interfaceName = "eth0";
        
        int maxMessages = 100;
        int maxClients = 4;
        bool hasGPU = true;
        int vocabSize = 50257;
        int maxSeqLen = 2048;
        std::string quantType = "none";
        float ropeBase = 10000.0f;
        float ropeScale = 1.0f;
        float eps = 1e-5f;
        float dropout = 0.0f;
        bool verbose = false;

        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if ((arg == "-i" || arg == "--interface") && i + 1 < argc) {
                cfg.interfaceName = argv[++i];
            } else if ((arg == "-l" || arg == "--layers") && i + 1 < argc) {
                cfg.totalLayers = std::stoi(argv[++i]);
                cfg.remoteLayers = cfg.totalLayers;
            } else if ((arg == "-e" || arg == "--embed") && i + 1 < argc) {
                cfg.embedDim = std::stoi(argv[++i]);
            } else if ((arg == "-f" || arg == "--ffn") && i + 1 < argc) {
                cfg.ffnDim = std::stoi(argv[++i]);
            } else if ((arg == "-a" || arg == "--heads") && i + 1 < argc) {
                cfg.numHeads = std::stoi(argv[++i]);
            } else if ((arg == "-k" || arg == "--kvheads") && i + 1 < argc) {
                cfg.numKVHeads = std::stoi(argv[++i]);
            } else if ((arg == "-q" || arg == "--seq-len") && i + 1 < argc) {
                cfg.seqLen = std::stoi(argv[++i]);
            } else if ((arg == "-v" || arg == "--vocab-size") && i + 1 < argc) {
                vocabSize = std::stoi(argv[++i]);
            } else if ((arg == "-x" || arg == "--max-seq-len") && i + 1 < argc) {
                maxSeqLen = std::stoi(argv[++i]);
            } else if ((arg == "-m" || arg == "--messages") && i + 1 < argc) {
                maxMessages = std::stoi(argv[++i]);
            } else if ((arg == "-c" || arg == "--clients") && i + 1 < argc) {
                maxClients = std::stoi(argv[++i]);
            } else if ((arg == "-g" || arg == "--gpu") && i + 1 < argc) {
                std::string gpuVal = toLowerCase(argv[++i]);
                hasGPU = (gpuVal == "yes" || gpuVal == "true" || gpuVal == "1");
            } else if (arg == "--quant" && i + 1 < argc) {
                quantType = toLowerCase(argv[++i]);
            } else if (arg == "--rope-base" && i + 1 < argc) {
                ropeBase = std::stof(argv[++i]);
            } else if (arg == "--rope-scale" && i + 1 < argc) {
                ropeScale = std::stof(argv[++i]);
            } else if (arg == "--eps" && i + 1 < argc) {
                eps = std::stof(argv[++i]);
            } else if (arg == "--dropout" && i + 1 < argc) {
                dropout = std::stof(argv[++i]);
            } else if (arg == "--verbose") {
                verbose = true;
            } else if (arg == "--help") {
                std::cout << "\nSERVER MODE - Execute remote transformer layers\n" << std::endl;
                std::cout << "Usage: " << argv[0] << " server [options]\n" << std::endl;
                return 0;
            }
        }

        std::cout << "\n=== OpenCL Server Configuration ===" << std::endl;
        std::cout << "Interface: " << cfg.interfaceName << std::endl;
        std::cout << "Total Layers: " << cfg.totalLayers << std::endl;
        std::cout << "Embed Dim: " << cfg.embedDim << std::endl;
        std::cout << "Quantization: " << quantType << std::endl;
        std::cout << "GPU Available: " << (hasGPU ? "yes" : "no") << std::endl;
        std::cout << "===================================\n" << std::endl;

        cfg.localLayers = 0;
        cfg.remoteLayers = cfg.totalLayers;
        cfg.startRemoteLayer = 0;

        DistTransformer::DistributedTransformerServer server(cfg);
        if (!server.initialize()) {
            std::cerr << "Failed to initialize server" << std::endl;
            return 1;
        }

        server.setForwardLayerFunction([](const std::vector<float>& input, int layer, bool) {
            return input;
        });

        std::cout << "Server ready. Processing up to " << maxMessages << " messages...\n" << std::endl;
        server.run(maxMessages);
        std::cout << "Server shutdown complete." << std::endl;
        return 0;

    } else if (command == "client") {
        DistTransformer::DistributedConfig cfg;
        cfg.totalLayers = 12;
        cfg.remoteLayers = 6;
        cfg.localLayers = 6;
        cfg.startRemoteLayer = 6;
        cfg.embedDim = 768;
        cfg.ffnDim = 3072;
        cfg.numHeads = 12;
        cfg.numKVHeads = 12;
        cfg.seqLen = 512;
        cfg.interfaceName = "eth0";
        cfg.cacheActivations = true;
        cfg.cacheGradients = true;
        
        int timeoutMs = 5000;
        int retries = 3;
        bool serverMACProvided = false;
        bool startLayerSet = false;
        bool verbose = false;

        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if ((arg == "-i" || arg == "--interface") && i + 1 < argc) {
                cfg.interfaceName = argv[++i];
            } else if ((arg == "-s" || arg == "--server") && i + 1 < argc) {
                if (!DistTransformer::stringToMAC(argv[++i], cfg.serverMAC)) {
                    std::cerr << "Invalid MAC address format. Use XX:XX:XX:XX:XX:XX" << std::endl;
                    return 1;
                }
                serverMACProvided = true;
            } else if ((arg == "-l" || arg == "--layers") && i + 1 < argc) {
                cfg.totalLayers = std::stoi(argv[++i]);
            } else if ((arg == "-r" || arg == "--remote") && i + 1 < argc) {
                cfg.remoteLayers = std::stoi(argv[++i]);
                if (!startLayerSet) {
                    cfg.localLayers = cfg.totalLayers - cfg.remoteLayers;
                    cfg.startRemoteLayer = cfg.localLayers;
                }
            } else if ((arg == "-e" || arg == "--embed") && i + 1 < argc) {
                cfg.embedDim = std::stoi(argv[++i]);
            } else if (arg == "--help") {
                std::cout << "\nCLIENT MODE - Execute local layers, send remote to server\n" << std::endl;
                return 0;
            }
        }

        if (!serverMACProvided) {
            std::cerr << "Error: Server MAC address required (-s or --server)" << std::endl;
            return 1;
        }

        std::cout << "\n=== OpenCL Client Configuration ===" << std::endl;
        std::cout << "Interface: " << cfg.interfaceName << std::endl;
        char macStr[18];
        DistTransformer::macToString(cfg.serverMAC, macStr, sizeof(macStr));
        std::cout << "Server MAC: " << macStr << std::endl;
        std::cout << "Total Layers: " << cfg.totalLayers << std::endl;
        std::cout << "Local Layers: " << cfg.localLayers << std::endl;
        std::cout << "Remote Layers: " << cfg.remoteLayers << std::endl;
        std::cout << "====================================\n" << std::endl;

        DistTransformer::DistributedTransformer client(cfg);
        if (!client.initialize()) {
            std::cerr << "Failed to initialize client" << std::endl;
            return 1;
        }

        std::cout << "Connecting to server..." << std::endl;
        if (!client.connect(timeoutMs)) {
            std::cerr << "Failed to connect to server" << std::endl;
            return 1;
        }

        std::cout << "Connected successfully!" << std::endl;
        std::cout << "Testing forward pass..." << std::endl;
        
        std::vector<float> input(cfg.embedDim, 1.0f);
        auto output = client.forward(input);

        if (!output.empty()) {
            std::cout << "Forward pass successful" << std::endl;
            std::cout << "  Input size: " << input.size() << std::endl;
            std::cout << "  Output size: " << output.size() << std::endl;
        } else {
            std::cout << "Forward pass returned empty output" << std::endl;
        }

        client.disconnect();
        std::cout << "\nClient shutdown complete." << std::endl;
        return 0;

    } else if (command == "generate") {
        std::string modelPath;
        std::string prompt;
        GenerationConfig genCfg;
        bool interactive = false;
        
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
                modelPath = argv[++i];
            } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
                prompt = argv[++i];
            } else if ((arg == "-n" || arg == "--tokens") && i + 1 < argc) {
                genCfg.maxTokens = std::stoi(argv[++i]);
            } else if ((arg == "-t" || arg == "--temperature") && i + 1 < argc) {
                genCfg.temperature = std::stof(argv[++i]);
            } else if (arg == "--top-k" && i + 1 < argc) {
                genCfg.topK = std::stoi(argv[++i]);
            } else if (arg == "--top-p" && i + 1 < argc) {
                genCfg.topP = std::stof(argv[++i]);
            } else if (arg == "-i" || arg == "--interactive") {
                interactive = true;
            } else if (arg == "--help") {
                std::cout << "\nGENERATE MODE - Text generation from GGUF model\n" << std::endl;
                std::cout << "Usage: " << argv[0] << " generate -m <model.gguf> [options]\n" << std::endl;
                return 0;
            }
        }
        
        if (modelPath.empty()) {
            std::cerr << "Error: Model path required (-m <path>)" << std::endl;
            return 1;
        }
        
        std::cout << "\n=== Text Generation ===" << std::endl;
        
        GGUFModel model;
        if (!model.load(modelPath)) {
            std::cerr << "Failed to load model: " << modelPath << std::endl;
            return 1;
        }
        
        Tokenizer tokenizer;
        if (!tokenizer.loadFromGGUF(model.getTokens(), model.architecture())) {
            std::cerr << "Failed to load tokenizer from model" << std::endl;
            return 1;
        }
        
        TextGenerator generator;
        if (!generator.loadModel(&model, &tokenizer)) {
            std::cerr << "Failed to initialize generator" << std::endl;
            return 1;
        }
        
        if (interactive) {
            std::cout << "\nInteractive chat mode. Type 'quit' to exit.\n" << std::endl;
            while (true) {
                std::cout << "You: ";
                std::getline(std::cin, prompt);
                if (prompt == "quit" || prompt == "exit") break;
                if (prompt.empty()) continue;
                
                std::string formatted = tokenizer.applyChatTemplate(prompt);
                std::cout << "Assistant: ";
                generator.generate(formatted, genCfg);
                generator.clearCache();
                std::cout << std::endl;
            }
        } else {
            if (prompt.empty()) {
                std::cout << "Enter prompt: ";
                std::getline(std::cin, prompt);
            }
            
            std::string formatted = tokenizer.applyChatTemplate(prompt);
            std::cout << "\nGenerating...\n" << std::endl;
            generator.generate(formatted, genCfg);
        }
        
        return 0;

    } else if (command == "test") {
        std::cout << "\n=== Running Tests ===" << std::endl;
        
        std::cout << "Test 1: Protocol header verification" << std::endl;
        DistTransformer::DTXHeader hdr = DistTransformer::makeHeader(
            DistTransformer::MessageType::HANDSHAKE_REQ, 1, nullptr, 0);
        
        if (DistTransformer::verifyHeader(hdr)) {
            std::cout << "  Header verification passed" << std::endl;
        } else {
            std::cout << "  Header verification failed" << std::endl;
        }

        std::cout << "Test 2: MAC address handling" << std::endl;
        uint8_t testMAC[6] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
        char macStr[18];
        DistTransformer::macToString(testMAC, macStr, sizeof(macStr));
        uint8_t parsedMAC[6];
        if (DistTransformer::stringToMAC(macStr, parsedMAC) &&
            DistTransformer::compareMACAddress(testMAC, parsedMAC)) {
            std::cout << "  MAC address parsing passed" << std::endl;
        } else {
            std::cout << "  MAC address parsing failed" << std::endl;
        }

        std::cout << "Test 3: Configuration validation" << std::endl;
        DistTransformer::DistributedConfig cfg = DistTransformer::createSymmetricConfig(12, 768, 3072, 12);
        if (cfg.validate()) {
            std::cout << "  Config validation passed" << std::endl;
        } else {
            std::cout << "  Config validation failed" << std::endl;
        }

        std::cout << "====================\n" << std::endl;
        return 0;

    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        printMainHelp(argv[0]);
        return 1;
    }

    return 0;
}
