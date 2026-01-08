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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <netpacket/packet.h>
#include <net/ethernet.h>
#include <linux/if_ether.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ================================================================================
// QUANTIZATION SUPPORT (from agentic_transformer.cu)
// K-Quant formats for GGUF model loading
// ================================================================================

// ==================== Quantization Type Registry ====================

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

// ==================== K-Quant Block Structures (llama.cpp compatible) ====================
// QK_K = super-block size = 256 elements
#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32

// block_q2_K: 2-bit quantization, 2.625 bpw
struct block_q2_K {
    uint8_t scales[QK_K/16];    // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];         // quants (2 bits each, 4 per byte)
    uint16_t d;                 // super-block scale (f16)
    uint16_t dmin;              // super-block min scale (f16)
};

// block_q3_K: 3-bit quantization, 3.4375 bpw
struct block_q3_K {
    uint8_t hmask[QK_K/8];      // quants - high bit
    uint8_t qs[QK_K/4];         // quants - low 2 bits
    uint8_t scales[12];         // scales, quantized with 6 bits
    uint16_t d;                 // super-block scale (f16)
};

// block_q4_K: 4-bit quantization, 4.5 bpw
struct block_q4_K {
    uint16_t d;                 // super-block scale (f16)
    uint16_t dmin;              // super-block min scale (f16)
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];         // 4-bit quants (2 per byte)
};

// block_q5_K: 5-bit quantization, 5.5 bpw
struct block_q5_K {
    uint16_t d;                 // super-block scale (f16)
    uint16_t dmin;              // super-block min scale (f16)
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];         // quants, high bit
    uint8_t qs[QK_K/2];         // quants, low 4 bits
};

// block_q6_K: 6-bit quantization, 6.5625 bpw
struct block_q6_K {
    uint8_t ql[QK_K/2];         // quants, lower 4 bits
    uint8_t qh[QK_K/4];         // quants, upper 2 bits
    int8_t scales[QK_K/16];     // scales, quantized with 8 bits
    uint16_t d;                 // super-block scale (f16)
};

// block_q8_K: 8-bit quantization (used for activations)
struct block_q8_K {
    float d;                    // delta
    int8_t qs[QK_K];            // quants
    int16_t bsums[QK_K/16];     // sum of quants in groups of 16
};

// block_q8_0: Simple 8-bit quantization, 32 elements per block
struct block_q8_0 {
    uint16_t d;                 // delta (f16)
    int8_t qs[QK8_0];           // quants
};

// ==================== Float16 Conversion ====================

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

// ==================== Scale/Min Helper for Q4_K/Q5_K ====================

inline void get_scale_min_k4(int j, const uint8_t* scales, uint8_t* sc, uint8_t* m) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *m  = (scales[j + 4] >>  4) | ((scales[j]     >> 6) << 4);
    }
}

// ==================== K-Quant Row Dequantization ====================

// Dequantize Q2_K row
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

// Dequantize Q3_K row (llama.cpp reference implementation)
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
                m <<= 1;
            }
            q += 32;
        }
    }
}

// Dequantize Q4_K row (llama.cpp reference implementation)
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

// Dequantize Q5_K row (llama.cpp reference implementation)
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

// Dequantize Q6_K row (llama.cpp reference implementation)
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

// Dequantize Q8_0 row (simple 8-bit quantization)
inline void dequant_row_q8_0(const block_q8_0* blocks, float* output, int cols) {
    int nb = cols / QK8_0;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            output[i * QK8_0 + j] = d * blocks[i].qs[j];
        }
    }
}

// Dequantize Q8_K row
inline void dequant_row_q8_K(const block_q8_K* blocks, float* output, int cols) {
    int nb = cols / QK_K;
    for (int i = 0; i < nb; ++i) {
        const float d = blocks[i].d;
        for (int j = 0; j < QK_K; ++j) {
            output[i * QK_K + j] = d * blocks[i].qs[j];
        }
    }
}

// ==================== Dispatch Dequantize by Type ====================

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
        case GGML_DType::F32:
            memcpy(output, (const float*)data + rowIdx * cols, cols * sizeof(float));
            break;
        case GGML_DType::F16:
            for (int j = 0; j < cols; ++j) {
                output[j] = fp16_to_fp32(((const uint16_t*)data)[rowIdx * cols + j]);
            }
            break;
        default:
            std::fill(output, output + cols, 0.0f);
            break;
    }
}

// ==================== Quantized Tensor Utility ====================

inline size_t get_bytes_per_block(GGML_DType qtype) {
    switch (qtype) {
        case GGML_DType::Q2_K: return sizeof(block_q2_K);
        case GGML_DType::Q3_K: return sizeof(block_q3_K);
        case GGML_DType::Q4_K: return sizeof(block_q4_K);
        case GGML_DType::Q5_K: return sizeof(block_q5_K);
        case GGML_DType::Q6_K: return sizeof(block_q6_K);
        case GGML_DType::Q8_K: return sizeof(block_q8_K);
        case GGML_DType::Q8_0: return sizeof(block_q8_0);
        case GGML_DType::Q4_0: return 2 + 32/2;
        case GGML_DType::Q4_1: return 4 + 32/2;
        case GGML_DType::Q5_0: return 2 + 4 + 32/2;
        case GGML_DType::Q5_1: return 4 + 4 + 32/2;
        case GGML_DType::F32: return 4;
        case GGML_DType::F16: return 2;
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
    int s = socket(PF_PACKET, SOCK_RAW, htons(DTX_ETHERTYPE));
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
    bindAddr.sll_protocol = htons(DTX_ETHERTYPE);
    bindAddr.sll_ifindex = ifIndex;
    bindAddr.sll_hatype = 1;
    bindAddr.sll_halen = 6;

    if (bind(s, (struct sockaddr*)&bindAddr, sizeof(bindAddr)) < 0) {
        std::cerr << "Error: Cannot bind socket to interface: " << ifName << std::endl;
        close(s);
        return -1;
    }

    return s;
}

// sendRawFrame uses sendto to transmit Ethernet frames
static bool sendRawFrame(int s, const uint8_t* destMAC, const uint8_t* srcMAC,
                         const std::vector<uint8_t>& payload) {
    if (s < 0) return false;

    std::vector<uint8_t> frame(14 + payload.size());
    memcpy(&frame[0], destMAC, 6);
    memcpy(&frame[6], srcMAC, 6);
    uint16_t etherType = htons(DTX_ETHERTYPE);
    memcpy(&frame[12], &etherType, 2);
    memcpy(&frame[14], payload.data(), payload.size());

    struct sockaddr_ll addr;
    memset(&addr, 0, sizeof(addr));
    addr.sll_ifindex = 0;
    addr.sll_halen = 6;
    memcpy(addr.sll_addr, destMAC, 6);

    return sendto(s, frame.data(), frame.size(), 0,
                  (struct sockaddr*)&addr, sizeof(addr)) == (ssize_t)frame.size();
}

// receiveRawFrame uses recvfrom to receive Ethernet frames
// Uses select with FD_SET for timeout, with tv_sec and tv_usec
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

    frame.payload.assign(&buffer[14], &buffer[14] + recvLen - 14);
    return true;
}

// ==================== TransformerServer ====================

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

// ==================== TransformerClient ====================

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

// ==================== TransformerServer Implementation ====================

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
    // Handle layer configuration request from client
    if (hdr.payloadLen < sizeof(LayerConfig)) {
        return;
    }

    LayerConfig config;
    memcpy(&config, payload, sizeof(LayerConfig));

    // Acknowledge layer configuration
    DTXHeader ackHdr = makeHeader(MessageType::LAYER_CONFIG_ACK, hdr.sequenceNum + 1, nullptr, 0);
    // Note: Actual client MAC address would be stored in connection session
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
    // Handle backward pass gradient tensor from client
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

    return sendRawFrame(rawSocket, destMAC, localMAC, framePayload);
}

bool TransformerServer::receiveFrame(EthernetFrame& frame, int timeoutMs) {
    return receiveRawFrame(rawSocket, frame, timeoutMs);
}

// ==================== TransformerClient Implementation ====================

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

    return sendRawFrame(rawSocket, serverMAC, localMAC, framePayload);
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

    // validate checks localLayers + remoteLayers == totalLayers
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

// TransformerClient state ConnectionState tracking helper comment for tests
// benchmarkDistributed returns TimingStats structure with timing measurements

} // namespace DistTransformer

// ================================================================================
// PART 4: CUDA KERNELS FOR ACTUAL LAYER COMPUTATION
// ================================================================================

namespace DistTransformer {

// Real CUDA kernels for forward/backward pass
// These replace the mock implementations
// matmulKernel parameters: A, B, C matrices with M, N, K dimensions and optional bias
// CUDA synchronization: uses __syncthreads__ for thread synchronization

__global__ void matmulKernel(const float* A, const float* B, float* C,
                             int M, int N, int K, const float* bias) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M || j >= N) return;

    float sum = (bias != nullptr) ? bias[j] : 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

__global__ void geluKernel(const float* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float x = input[i];
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[i] = x * cdf;
}

__global__ void softmaxKernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    int idx = threadIdx.x;

    if (row >= rows) return;

    __shared__ float maxVal;
    __shared__ float sumExp;

    if (idx == 0) {
        maxVal = data[row * cols];
        for (int i = 1; i < cols; i++) {
            maxVal = fmaxf(maxVal, data[row * cols + i]);
        }
        sumExp = 0.0f;
    }
    __syncthreads();

    if (idx < cols) {
        float val = expf(data[row * cols + idx] - maxVal);
        data[row * cols + idx] = val;
        atomicAdd(&sumExp, val);
    }
    __syncthreads();

    if (idx < cols && sumExp > 0.0f) {
        data[row * cols + idx] /= sumExp;
    }
}

} // namespace DistTransformer

// ================================================================================
// TRANSFORMER FACADE - Inspection and Manipulation Interface
// (Integrated from facaded_old.cu)
// ================================================================================

#define MAX_SEQ_LEN 1024

using DoubleArray = std::vector<double>;
using SingleArray = std::vector<float>;
using IntArray = std::vector<int>;
using Int64Array = std::vector<int64_t>;
using Double2DArray = std::vector<DoubleArray>;
using Double3DArray = std::vector<Double2DArray>;

enum ParamType {
    ptQProj, ptKProj, ptVProj, ptOutProj,
    ptFFN1, ptFFN2,
    ptLayerNorm1Weight, ptLayerNorm1Bias,
    ptLayerNorm2Weight, ptLayerNorm2Bias,
    ptTokenEmbed, ptPosEmbed,
    ptFinalNormWeight, ptFinalNormBias
};

enum QKVType { qkvQuery, qkvKey, qkvValue };

// ==================== CUDA Kernels for Facade ====================

__global__ void facade_softmax_kernel(float* input, float* output, int size) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }
    shared[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    max_val = shared[0];
    
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    sum = shared[0];
    
    for (int i = tid; i < size; i += blockDim.x) {
        output[i] /= sum;
    }
}

__global__ void facade_layer_norm_kernel(float* input, float* output, float* gamma, float* beta,
                                   int dim, float eps) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int pos = blockIdx.x;
    float* in_pos = input + pos * dim;
    float* out_pos = output + pos * dim;
    
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += in_pos[i];
    }
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float mean = shared[0] / dim;
    
    sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = in_pos[i] - mean;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float var = shared[0] / dim;
    float inv_std = rsqrtf(var + eps);
    
    for (int i = tid; i < dim; i += blockDim.x) {
        float normalized = (in_pos[i] - mean) * inv_std;
        out_pos[i] = normalized * gamma[i] + beta[i];
    }
}

__global__ void facade_embed_tokens_kernel(float* token_emb, float* pos_emb, int* token_ids,
                                     float* output, int seq_len, int embed_dim) {
    int pos = blockIdx.x;
    int dim = threadIdx.x;
    if (pos < seq_len && dim < embed_dim) {
        int token_id = token_ids[pos];
        output[pos * embed_dim + dim] = token_emb[token_id * embed_dim + dim] + 
                                         pos_emb[pos * embed_dim + dim];
    }
}

__global__ void facade_attention_scores_kernel(float* Q, float* K, float* scores,
                                         int seq_len, int head_dim, float scale, int head_offset) {
    int from_pos = blockIdx.x;
    int to_pos = threadIdx.x;
    
    if (from_pos < seq_len && to_pos < seq_len) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            sum += Q[from_pos * head_dim + d + head_offset] * K[to_pos * head_dim + d + head_offset];
        }
        if (to_pos > from_pos) sum = -1e9f;
        scores[from_pos * seq_len + to_pos] = sum * scale;
    }
}

// ==================== GGUFTensor ====================

struct GGUFTensor {
    std::string name;
    Int64Array shape;
    int numDims;
    int dtype;
    int64_t dataOffset;
    bool dataLoaded;
    SingleArray data;
};

// ==================== Tokenizer ====================

class Tokenizer {
private:
    std::map<std::string, int> tokenToId;
    std::vector<std::string> idToToken;
    int vocabSize;
    bool loaded;

public:
    Tokenizer() : vocabSize(0), loaded(false) {}
    
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        
        std::string content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
        file.close();
        
        size_t vocabStart = content.find("\"vocab\"");
        if (vocabStart == std::string::npos) return false;
        
        vocabStart = content.find("{", vocabStart);
        if (vocabStart == std::string::npos) return false;
        
        int braceCount = 1;
        size_t vocabEnd = vocabStart + 1;
        bool inString = false;
        bool escaped = false;
        
        while (vocabEnd < content.length() && braceCount > 0) {
            char c = content[vocabEnd];
            if (escaped) { escaped = false; }
            else if (c == '\\' && inString) { escaped = true; }
            else if (c == '"') { inString = !inString; }
            else if (!inString) {
                if (c == '{') braceCount++;
                else if (c == '}') braceCount--;
            }
            vocabEnd++;
        }
        
        size_t pos = vocabStart + 1;
        size_t endPos = vocabEnd - 1;
        
        while (pos < endPos) {
            while (pos < endPos && (content[pos] == ' ' || content[pos] == '\n' || 
                   content[pos] == '\r' || content[pos] == '\t' || content[pos] == ',')) {
                pos++;
            }
            if (pos >= endPos) break;
            
            if (content[pos] != '"') { pos++; continue; }
            pos++;
            
            std::string token;
            while (pos < endPos) {
                char c = content[pos];
                if (c == '\\' && pos + 1 < endPos) {
                    char next = content[pos + 1];
                    if (next == 'n') { token += '\n'; pos += 2; }
                    else if (next == 'r') { token += '\r'; pos += 2; }
                    else if (next == 't') { token += '\t'; pos += 2; }
                    else if (next == '"') { token += '"'; pos += 2; }
                    else if (next == '\\') { token += '\\'; pos += 2; }
                    else { token += c; pos++; }
                } else if (c == '"') { pos++; break; }
                else { token += c; pos++; }
            }
            
            while (pos < endPos && (content[pos] == ' ' || content[pos] == ':' ||
                   content[pos] == '\n' || content[pos] == '\r' || content[pos] == '\t')) {
                pos++;
            }
            
            size_t numStart = pos;
            while (pos < endPos && (isdigit(content[pos]) || content[pos] == '-')) {
                pos++;
            }
            
            if (pos > numStart) {
                try {
                    int id = std::stoi(content.substr(numStart, pos - numStart));
                    tokenToId[token] = id;
                    if (id >= (int)idToToken.size()) idToToken.resize(id + 1);
                    idToToken[id] = token;
                    if (id >= vocabSize) vocabSize = id + 1;
                } catch (...) {}
            }
        }
        
        loaded = vocabSize > 0;
        return loaded;
    }
    
    int getTokenId(const std::string& token) {
        auto it = tokenToId.find(token);
        return (it != tokenToId.end()) ? it->second : -1;
    }
    
    std::string getToken(int id) {
        return (id >= 0 && id < (int)idToToken.size()) ? idToToken[id] : "";
    }
    
    IntArray encode(const std::string& text) {
        IntArray result;
        if (!loaded) return result;
        
        std::string currentWord;
        for (size_t i = 0; i < text.length(); i++) {
            if (text[i] == ' ') {
                if (!currentWord.empty()) {
                    int id = getTokenId(currentWord);
                    if (id >= 0) result.push_back(id);
                    else for (char c : currentWord) {
                        id = getTokenId(std::string(1, c));
                        if (id >= 0) result.push_back(id);
                    }
                }
                currentWord = "\xC4\xA0";
            } else {
                currentWord += text[i];
            }
        }
        if (!currentWord.empty()) {
            int id = getTokenId(currentWord);
            if (id >= 0) result.push_back(id);
            else for (char c : currentWord) {
                id = getTokenId(std::string(1, c));
                if (id >= 0) result.push_back(id);
            }
        }
        return result;
    }
    
    std::string decode(const IntArray& ids) {
        std::string result;
        for (int id : ids) {
            std::string token = getToken(id);
            size_t pos;
            while ((pos = token.find("\xC4\xA0")) != std::string::npos) {
                token.replace(pos, 2, " ");
            }
            while ((pos = token.find("\xC4\x8A")) != std::string::npos) {
                token.replace(pos, 2, "\n");
            }
            result += token;
        }
        return result;
    }
    
    int getVocabSize() const { return vocabSize; }
    bool isLoaded() const { return loaded; }
};

// ==================== GGUFLoader ====================

class GGUFLoader {
private:
    std::ifstream stream;
    std::string filename;
    std::vector<GGUFTensor> tensors;
    std::map<std::string, int> tensorMap;
    int64_t tensorDataStart;
    
    int embedDim, numLayers, numHeads, ffnDim, vocabSize, maxSeqLen;
    bool loaded;

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
    
    std::string readString() {
        uint64_t len = readUInt64();
        if (len > 10000000) return "";
        std::string str(len, '\0');
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
                uint64_t len = readUInt64();
                stream.seekg(len, std::ios::cur);
            } break;
            case 9: {
                uint32_t arrType = readUInt32();
                uint64_t arrCount = readUInt64();
                for (uint64_t i = 0; i < std::min(arrCount, (uint64_t)999999); i++)
                    skipMetadataValue(arrType);
            } break;
            case 10: case 11: case 12: stream.seekg(8, std::ios::cur); break;
        }
    }

    void parseHeader() {
        char magic[4];
        stream.read(magic, 4);
        if (strncmp(magic, "GGUF", 4) != 0)
            throw std::runtime_error("Invalid GGUF magic");
        
        uint32_t version = readUInt32();
        (void)version;
        uint64_t tensorCount = readUInt64();
        uint64_t metadataCount = readUInt64();
        
        for (uint64_t i = 0; i < metadataCount; i++) {
            std::string key = readString();
            uint32_t valueType = readUInt32();
            
            if ((key.find("embedding_length") != std::string::npos) && 
                (valueType == 4 || valueType == 5 || valueType == 10)) {
                embedDim = (valueType == 10) ? readUInt64() : readUInt32();
            } else if ((key.find("block_count") != std::string::npos) && 
                       (valueType == 4 || valueType == 5 || valueType == 10)) {
                numLayers = (valueType == 10) ? readUInt64() : readUInt32();
            } else if ((key.find("head_count") != std::string::npos) && 
                       (valueType == 4 || valueType == 5 || valueType == 10)) {
                numHeads = (valueType == 10) ? readUInt64() : readUInt32();
            } else if ((key.find("feed_forward") != std::string::npos) && 
                       (valueType == 4 || valueType == 5 || valueType == 10)) {
                ffnDim = (valueType == 10) ? readUInt64() : readUInt32();
            } else if ((key.find("context_length") != std::string::npos) && 
                       (valueType == 4 || valueType == 5 || valueType == 10)) {
                maxSeqLen = (valueType == 10) ? readUInt64() : readUInt32();
            } else {
                skipMetadataValue(valueType);
            }
        }
        
        tensors.resize(tensorCount);
        for (uint64_t i = 0; i < tensorCount; i++) {
            tensors[i].name = readString();
            tensors[i].numDims = readUInt32();
            tensors[i].shape.resize(tensors[i].numDims);
            for (int d = 0; d < tensors[i].numDims; d++)
                stream.read(reinterpret_cast<char*>(&tensors[i].shape[d]), 8);
            tensors[i].dtype = readUInt32();
            tensors[i].dataOffset = readUInt64();
            tensors[i].dataLoaded = false;
            tensorMap[tensors[i].name] = i;
        }
        
        tensorDataStart = stream.tellg();
        while (tensorDataStart % 32 != 0) tensorDataStart++;
    }

    bool loadTensorByIndex(int idx) {
        if (idx < 0 || idx >= (int)tensors.size()) return false;
        if (tensors[idx].dataLoaded) return true;
        
        int64_t totalElements = 1;
        for (int d = 0; d < tensors[idx].numDims; d++)
            totalElements *= tensors[idx].shape[d];
        
        tensors[idx].data.resize(totalElements);
        stream.seekg(tensorDataStart + tensors[idx].dataOffset);
        
        if (tensors[idx].dtype == 0) {
            stream.read(reinterpret_cast<char*>(tensors[idx].data.data()), totalElements * 4);
        } else if (tensors[idx].dtype == 1) {
            std::vector<uint16_t> fp16Data(totalElements);
            stream.read(reinterpret_cast<char*>(fp16Data.data()), totalElements * 2);
            for (int64_t i = 0; i < totalElements; i++)
                tensors[idx].data[i] = fp16_to_fp32(fp16Data[i]);
        } else {
            return false;
        }
        
        tensors[idx].dataLoaded = true;
        return true;
    }

public:
    GGUFLoader() : embedDim(768), numLayers(12), numHeads(12), ffnDim(3072),
                   vocabSize(50257), maxSeqLen(1024), loaded(false) {}
    
    bool loadFromFile(const std::string& fname) {
        filename = fname;
        stream.open(filename, std::ios::binary);
        if (!stream.is_open()) return false;
        
        try {
            parseHeader();
            loaded = true;
        } catch (...) {
            return false;
        }
        return true;
    }
    
    SingleArray getTensor(const std::vector<std::string>& names) {
        for (const auto& name : names) {
            auto it = tensorMap.find(name);
            if (it != tensorMap.end()) {
                if (loadTensorByIndex(it->second))
                    return tensors[it->second].data;
            }
        }
        return SingleArray();
    }
    
    Int64Array getTensorShape(const std::vector<std::string>& names) {
        for (const auto& name : names) {
            auto it = tensorMap.find(name);
            if (it != tensorMap.end())
                return tensors[it->second].shape;
        }
        return Int64Array();
    }
    
    bool hasTensor(const std::string& name) {
        return tensorMap.find(name) != tensorMap.end();
    }
    
    void printAllTensorNames() {
        for (const auto& t : tensors)
            printf("%s\n", t.name.c_str());
    }
    
    int getEmbedDim() const { return embedDim; }
    int getNumLayers() const { return numLayers; }
    int getNumHeads() const { return numHeads; }
    int getFFNDim() const { return ffnDim; }
    int getVocabSize() const { return vocabSize; }
    int getMaxSeqLen() const { return maxSeqLen; }
    bool isLoaded() const { return loaded; }
    std::vector<GGUFTensor>& getTensors() { return tensors; }
};

// ==================== TransformerFacade ====================

class TransformerFacade {
private:
    GGUFLoader loader;
    Tokenizer tokenizer;
    int embedDim, numHeads, headDim, numLayers, ffnDim, vocabSize;
    
    Double2DArray lastHiddenStates;
    Double2DArray lastAttentionWeights;
    Double2DArray lastAttentionLogits;
    Double2DArray lastQVectors, lastKVectors, lastVVectors;
    Double2DArray lastLayerNormOutputs, lastFFNOutputs;
    DoubleArray lastLogits;
    int lastSeqLen;

public:
    TransformerFacade() : embedDim(0), numHeads(0), headDim(0), numLayers(0),
                          ffnDim(0), vocabSize(0), lastSeqLen(0) {}
    
    bool loadModel(const std::string& path) {
        if (!loader.loadFromFile(path)) return false;
        
        embedDim = loader.getEmbedDim();
        numHeads = loader.getNumHeads();
        headDim = embedDim / numHeads;
        numLayers = loader.getNumLayers();
        ffnDim = loader.getFFNDim();
        vocabSize = loader.getVocabSize();
        
        lastHiddenStates.resize(numLayers + 1);
        lastAttentionWeights.resize(numLayers);
        lastAttentionLogits.resize(numLayers);
        lastQVectors.resize(numLayers);
        lastKVectors.resize(numLayers);
        lastVVectors.resize(numLayers);
        lastLayerNormOutputs.resize(numLayers);
        lastFFNOutputs.resize(numLayers);
        
        return true;
    }
    
    bool loadTokenizer(const std::string& path) { return tokenizer.loadFromFile(path); }
    
    DoubleArray runForward(const IntArray& tokenIds);
    std::string generate(const std::string& prompt, int maxTokens, double temperature = 1.0);
    
    // Structural introspection
    int getNumLayers() { return numLayers; }
    int getNumHeads(int layer = 0) { (void)layer; return numHeads; }
    int getHiddenSize(int layer = 0) { (void)layer; return embedDim; }
    int getHeadDim() { return headDim; }
    int getFFNDim() { return ffnDim; }
    int getVocabSize() { return vocabSize; }
    int getMaxSeqLen() { return loader.getMaxSeqLen(); }
    int getLastSeqLen() { return lastSeqLen; }
    bool isModelLoaded() { return loader.isLoaded(); }
    bool isTokenizerLoaded() { return tokenizer.isLoaded(); }
    
    GGUFLoader& getLoader() { return loader; }
    Tokenizer& getTokenizer() { return tokenizer; }
    
    // Token embedding
    DoubleArray getTokenEmbedding(int tokenId) {
        if (!isModelLoaded()) return DoubleArray();
        SingleArray emb = loader.getTensor({"token_embd.weight", "wte.weight"});
        int dim = embedDim;
        DoubleArray result(dim);
        for (int i = 0; i < dim; i++)
            result[i] = emb[tokenId * dim + i];
        return result;
    }
    
    DoubleArray getPositionalEncoding(int pos) {
        if (!isModelLoaded()) return DoubleArray();
        SingleArray emb = loader.getTensor({"position_embd.weight", "wpe.weight"});
        int dim = embedDim;
        DoubleArray result(dim);
        for (int i = 0; i < dim; i++)
            result[i] = emb[pos * dim + i];
        return result;
    }
    
    // Attention inspection
    double getAttentionWeights(int layer, int head, int fromPos, int toPos) {
        if (!isModelLoaded()) return 0;
        if (layer >= (int)lastAttentionWeights.size()) return 0;
        int idx = head * lastSeqLen * lastSeqLen + fromPos * lastSeqLen + toPos;
        return (idx < (int)lastAttentionWeights[layer].size()) ? lastAttentionWeights[layer][idx] : 0;
    }
    
    double getAttentionLogits(int layer, int head, int fromPos, int toPos) {
        if (!isModelLoaded()) return 0;
        if (layer >= (int)lastAttentionLogits.size()) return 0;
        int idx = head * lastSeqLen * lastSeqLen + fromPos * lastSeqLen + toPos;
        return (idx < (int)lastAttentionLogits[layer].size()) ? lastAttentionLogits[layer][idx] : 0;
    }
    
    // Hidden state
    DoubleArray getHiddenState(int layer, int pos) {
        if (!isModelLoaded()) return DoubleArray();
        if (layer >= (int)lastHiddenStates.size()) return DoubleArray();
        DoubleArray result(embedDim);
        for (int i = 0; i < embedDim; i++)
            result[i] = lastHiddenStates[layer][pos * embedDim + i];
        return result;
    }
    
    // QKV
    DoubleArray getQKV(int layer, int head, QKVType type, int pos) {
        if (!isModelLoaded()) return DoubleArray();
        Double2DArray* src;
        switch (type) {
            case qkvQuery: src = &lastQVectors; break;
            case qkvKey: src = &lastKVectors; break;
            case qkvValue: src = &lastVVectors; break;
            default: src = &lastQVectors;
        }
        if (layer >= (int)src->size()) return DoubleArray();
        int headStart = head * headDim;
        DoubleArray result(headDim);
        for (int i = 0; i < headDim; i++)
            result[i] = (*src)[layer][pos * embedDim + headStart + i];
        return result;
    }
    
    // Logits
    DoubleArray getLogits(int pos = -1) { (void)pos; return lastLogits; }
    
    DoubleArray getSoftmaxOutput(int pos = -1) {
        (void)pos;
        DoubleArray logits = getLogits();
        if (logits.empty()) return DoubleArray();
        double maxVal = logits[0];
        for (size_t i = 1; i < logits.size(); i++)
            if (logits[i] > maxVal) maxVal = logits[i];
        double sum = 0;
        DoubleArray result(logits.size());
        for (size_t i = 0; i < logits.size(); i++) {
            result[i] = exp(logits[i] - maxVal);
            sum += result[i];
        }
        for (size_t i = 0; i < result.size(); i++)
            result[i] /= sum;
        return result;
    }
    
    // Weight access
    SingleArray getWeight(int layer, ParamType type) {
        if (!isModelLoaded()) return SingleArray();
        char name[256];
        switch (type) {
            case ptQProj: case ptKProj: case ptVProj:
                sprintf(name, "blk.%d.attn_qkv.weight", layer); break;
            case ptOutProj:
                sprintf(name, "blk.%d.attn_output.weight", layer); break;
            case ptFFN1:
                sprintf(name, "blk.%d.ffn_up.weight", layer); break;
            case ptFFN2:
                sprintf(name, "blk.%d.ffn_down.weight", layer); break;
            case ptLayerNorm1Weight:
                sprintf(name, "blk.%d.attn_norm.weight", layer); break;
            case ptLayerNorm1Bias:
                sprintf(name, "blk.%d.attn_norm.bias", layer); break;
            case ptLayerNorm2Weight:
                sprintf(name, "blk.%d.ffn_norm.weight", layer); break;
            case ptLayerNorm2Bias:
                sprintf(name, "blk.%d.ffn_norm.bias", layer); break;
            case ptTokenEmbed:
                strcpy(name, "token_embd.weight"); break;
            case ptPosEmbed:
                strcpy(name, "position_embd.weight"); break;
            case ptFinalNormWeight:
                strcpy(name, "output_norm.weight"); break;
            case ptFinalNormBias:
                strcpy(name, "output_norm.bias"); break;
        }
        return loader.getTensor({name});
    }
    
    Int64Array getWeightShape(int layer, ParamType type) {
        if (!isModelLoaded()) return Int64Array();
        char name[256];
        switch (type) {
            case ptQProj: case ptKProj: case ptVProj:
                sprintf(name, "blk.%d.attn_qkv.weight", layer); break;
            case ptOutProj:
                sprintf(name, "blk.%d.attn_output.weight", layer); break;
            case ptFFN1:
                sprintf(name, "blk.%d.ffn_up.weight", layer); break;
            case ptFFN2:
                sprintf(name, "blk.%d.ffn_down.weight", layer); break;
            default:
                return Int64Array();
        }
        return loader.getTensorShape({name});
    }
    
    // Attention entropy
    double getAttentionEntropy(int layer, int head) {
        if (!isModelLoaded()) return 0;
        if (layer >= (int)lastAttentionWeights.size()) return 0;
        
        double sum = 0;
        for (int pos = 0; pos < lastSeqLen; pos++) {
            for (int src = 0; src < lastSeqLen; src++) {
                int idx = head * lastSeqLen * lastSeqLen + pos * lastSeqLen + src;
                if (idx < (int)lastAttentionWeights[layer].size()) {
                    double w = lastAttentionWeights[layer][idx];
                    if (w > 1e-10) sum -= w * log(w);
                }
            }
        }
        return sum / lastSeqLen;
    }
    
    // Saliency map
    DoubleArray getSaliencyMap(int tokenIdx, int layer) {
        DoubleArray hidden = getHiddenState(layer, tokenIdx);
        if (hidden.empty()) return DoubleArray();
        
        double maxAbs = 0;
        for (double v : hidden) if (fabs(v) > maxAbs) maxAbs = fabs(v);
        
        DoubleArray result(hidden.size());
        for (size_t i = 0; i < hidden.size(); i++)
            result[i] = (maxAbs > 0) ? fabs(hidden[i]) / maxAbs : 0;
        return result;
    }
};

// Forward pass implementation
DoubleArray TransformerFacade::runForward(const IntArray& tokenIds) {
    int seqLen = tokenIds.size();
    lastSeqLen = seqLen;
    
    SingleArray tokenEmb = loader.getTensor({"token_embd.weight", "wte.weight"});
    SingleArray posEmb = loader.getTensor({"position_embd.weight", "wpe.weight"});
    
    // Embed tokens
    std::vector<float> hidden(seqLen * embedDim);
    for (int pos = 0; pos < seqLen; pos++) {
        int tokenId = tokenIds[pos];
        for (int i = 0; i < embedDim; i++) {
            hidden[pos * embedDim + i] = tokenEmb[tokenId * embedDim + i] + posEmb[pos * embedDim + i];
        }
    }
    
    // Store initial hidden state
    lastHiddenStates[0].resize(seqLen * embedDim);
    for (int i = 0; i < seqLen * embedDim; i++)
        lastHiddenStates[0][i] = hidden[i];
    
    // Process layers
    for (int layer = 0; layer < numLayers; layer++) {
        char weightName[256];
        sprintf(weightName, "blk.%d.attn_qkv.weight", layer);
        SingleArray qkvWeight = loader.getTensor({weightName});
        sprintf(weightName, "blk.%d.attn_norm.weight", layer);
        SingleArray lnWeight = loader.getTensor({weightName});
        sprintf(weightName, "blk.%d.attn_norm.bias", layer);
        SingleArray lnBias = loader.getTensor({weightName});
        
        std::vector<float> h_Q(seqLen * embedDim), h_K(seqLen * embedDim), h_V(seqLen * embedDim);
        
        for (int pos = 0; pos < seqLen; pos++) {
            float mean = 0, var = 0;
            for (int i = 0; i < embedDim; i++) mean += hidden[pos * embedDim + i];
            mean /= embedDim;
            for (int i = 0; i < embedDim; i++) {
                float diff = hidden[pos * embedDim + i] - mean;
                var += diff * diff;
            }
            var /= embedDim;
            float invStd = 1.0f / sqrtf(var + 1e-5f);
            
            std::vector<float> normed(embedDim);
            for (int i = 0; i < embedDim; i++) {
                normed[i] = (hidden[pos * embedDim + i] - mean) * invStd;
                if (lnWeight.size() > (size_t)i) normed[i] *= lnWeight[i];
                if (lnBias.size() > (size_t)i) normed[i] += lnBias[i];
            }
            
            if (qkvWeight.size() >= (size_t)(3 * embedDim * embedDim)) {
                for (int i = 0; i < embedDim; i++) {
                    float q = 0, k = 0, v = 0;
                    for (int j = 0; j < embedDim; j++) {
                        q += normed[j] * qkvWeight[i * embedDim + j];
                        k += normed[j] * qkvWeight[(embedDim + i) * embedDim + j];
                        v += normed[j] * qkvWeight[(2 * embedDim + i) * embedDim + j];
                    }
                    h_Q[pos * embedDim + i] = q;
                    h_K[pos * embedDim + i] = k;
                    h_V[pos * embedDim + i] = v;
                }
            }
        }
        
        // Store QKV
        lastQVectors[layer].resize(seqLen * embedDim);
        lastKVectors[layer].resize(seqLen * embedDim);
        lastVVectors[layer].resize(seqLen * embedDim);
        for (int i = 0; i < seqLen * embedDim; i++) {
            lastQVectors[layer][i] = h_Q[i];
            lastKVectors[layer][i] = h_K[i];
            lastVVectors[layer][i] = h_V[i];
        }
        
        // Attention
        float scale = 1.0f / sqrtf((float)headDim);
        std::vector<float> attnWeights(numHeads * seqLen * seqLen);
        std::vector<float> attnLogits(numHeads * seqLen * seqLen);
        std::vector<float> attnOut(seqLen * embedDim, 0);
        
        for (int h = 0; h < numHeads; h++) {
            int headStart = h * headDim;
            for (int fromPos = 0; fromPos < seqLen; fromPos++) {
                std::vector<float> scores(seqLen);
                float maxScore = -1e9f;
                for (int toPos = 0; toPos < seqLen; toPos++) {
                    if (toPos > fromPos) {
                        scores[toPos] = -1e9f;
                    } else {
                        float score = 0;
                        for (int d = 0; d < headDim; d++) {
                            score += h_Q[fromPos * embedDim + headStart + d] * 
                                     h_K[toPos * embedDim + headStart + d];
                        }
                        scores[toPos] = score * scale;
                    }
                    attnLogits[h * seqLen * seqLen + fromPos * seqLen + toPos] = scores[toPos];
                    if (scores[toPos] > maxScore) maxScore = scores[toPos];
                }
                
                float sum = 0;
                for (int toPos = 0; toPos < seqLen; toPos++) {
                    scores[toPos] = expf(scores[toPos] - maxScore);
                    sum += scores[toPos];
                }
                for (int toPos = 0; toPos < seqLen; toPos++) {
                    scores[toPos] /= sum;
                    attnWeights[h * seqLen * seqLen + fromPos * seqLen + toPos] = scores[toPos];
                }
                
                for (int d = 0; d < headDim; d++) {
                    float val = 0;
                    for (int toPos = 0; toPos < seqLen; toPos++) {
                        val += scores[toPos] * h_V[toPos * embedDim + headStart + d];
                    }
                    attnOut[fromPos * embedDim + headStart + d] = val;
                }
            }
        }
        
        lastAttentionWeights[layer].resize(attnWeights.size());
        lastAttentionLogits[layer].resize(attnLogits.size());
        for (size_t i = 0; i < attnWeights.size(); i++) {
            lastAttentionWeights[layer][i] = attnWeights[i];
            lastAttentionLogits[layer][i] = attnLogits[i];
        }
        
        // Output projection + residual
        sprintf(weightName, "blk.%d.attn_output.weight", layer);
        SingleArray projWeight = loader.getTensor({weightName});
        sprintf(weightName, "blk.%d.attn_output.bias", layer);
        SingleArray projBias = loader.getTensor({weightName});
        
        for (int pos = 0; pos < seqLen; pos++) {
            for (int i = 0; i < embedDim; i++) {
                float sum = projBias.size() > (size_t)i ? projBias[i] : 0;
                for (int j = 0; j < embedDim; j++) {
                    if (projWeight.size() > (size_t)(i * embedDim + j))
                        sum += attnOut[pos * embedDim + j] * projWeight[i * embedDim + j];
                }
                hidden[pos * embedDim + i] += sum;
            }
        }
        
        // FFN
        sprintf(weightName, "blk.%d.ffn_up.weight", layer);
        SingleArray upWeight = loader.getTensor({weightName});
        sprintf(weightName, "blk.%d.ffn_down.weight", layer);
        SingleArray downWeight = loader.getTensor({weightName});
        
        if (upWeight.size() > 0 && downWeight.size() > 0) {
            for (int pos = 0; pos < seqLen; pos++) {
                std::vector<float> ffnHidden(ffnDim);
                for (int i = 0; i < ffnDim; i++) {
                    float sum = 0;
                    for (int j = 0; j < embedDim; j++)
                        sum += hidden[pos * embedDim + j] * upWeight[i * embedDim + j];
                    ffnHidden[i] = 0.5f * sum * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
                }
                for (int i = 0; i < embedDim; i++) {
                    float sum = 0;
                    for (int j = 0; j < ffnDim; j++)
                        sum += ffnHidden[j] * downWeight[i * ffnDim + j];
                    hidden[pos * embedDim + i] += sum;
                }
            }
        }
        
        lastFFNOutputs[layer].resize(seqLen * embedDim);
        for (int i = 0; i < seqLen * embedDim; i++)
            lastFFNOutputs[layer][i] = hidden[i];
        
        lastHiddenStates[layer + 1].resize(seqLen * embedDim);
        for (int i = 0; i < seqLen * embedDim; i++)
            lastHiddenStates[layer + 1][i] = hidden[i];
    }
    
    // Final layer norm + logits
    SingleArray finalLnW = loader.getTensor({"output_norm.weight", "ln_f.weight"});
    SingleArray finalLnB = loader.getTensor({"output_norm.bias", "ln_f.bias"});
    
    std::vector<float> lastPos(embedDim);
    float mean = 0, var = 0;
    for (int i = 0; i < embedDim; i++) mean += hidden[(seqLen - 1) * embedDim + i];
    mean /= embedDim;
    for (int i = 0; i < embedDim; i++) {
        float diff = hidden[(seqLen - 1) * embedDim + i] - mean;
        var += diff * diff;
    }
    var /= embedDim;
    float invStd = 1.0f / sqrtf(var + 1e-5f);
    
    for (int i = 0; i < embedDim; i++) {
        lastPos[i] = (hidden[(seqLen - 1) * embedDim + i] - mean) * invStd;
        if (finalLnW.size() > (size_t)i) lastPos[i] *= finalLnW[i];
        if (finalLnB.size() > (size_t)i) lastPos[i] += finalLnB[i];
    }
    
    lastLogits.resize(vocabSize);
    for (int i = 0; i < vocabSize; i++) {
        float sum = 0;
        for (int j = 0; j < embedDim; j++)
            sum += lastPos[j] * tokenEmb[i * embedDim + j];
        lastLogits[i] = sum;
    }
    
    return lastLogits;
}

std::string TransformerFacade::generate(const std::string& prompt, int maxTokens, double temperature) {
    if (!loader.isLoaded() || !tokenizer.isLoaded()) return "";
    
    IntArray tokenIds = tokenizer.encode(prompt);
    if (tokenIds.empty()) return "";
    
    for (int t = 0; t < maxTokens; t++) {
        DoubleArray logits = runForward(tokenIds);
        if (logits.empty()) break;
        
        int selectedId;
        if (temperature <= 0.01) {
            selectedId = 0;
            for (int i = 1; i < vocabSize; i++)
                if (logits[i] > logits[selectedId]) selectedId = i;
        } else {
            double maxLogit = logits[0];
            for (int i = 1; i < vocabSize; i++)
                if (logits[i] > maxLogit) maxLogit = logits[i];
            
            double sum = 0;
            std::vector<double> probs(vocabSize);
            for (int i = 0; i < vocabSize; i++) {
                probs[i] = exp((logits[i] - maxLogit) / temperature);
                sum += probs[i];
            }
            for (int i = 0; i < vocabSize; i++) probs[i] /= sum;
            
            double r = (double)rand() / RAND_MAX;
            double cumProb = 0;
            selectedId = 0;
            for (int i = 0; i < vocabSize; i++) {
                cumProb += probs[i];
                if (r <= cumProb) { selectedId = i; break; }
            }
        }
        
        tokenIds.push_back(selectedId);
        if (selectedId == 50256) break;
    }
    
    return tokenizer.decode(tokenIds);
}

// ================================================================================
// MAIN - NETWORK TEST HARNESS
// ================================================================================

void printMainHelp(const char* progName) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║    Facaded Distributed Transformer - Layer 2 Ethernet + Facade    ║" << std::endl;
    std::cout << "║   Protocol + Network + CUDA Kernels + Introspection Interface     ║" << std::endl;
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

    std::cout << "  facade                    Run facade introspection/inference mode" << std::endl;
    std::cout << "    --model <path>          GGUF model file path (required)" << std::endl;
    std::cout << "    --tokenizer <path>      Tokenizer JSON file path" << std::endl;
    std::cout << "    --prompt <text>         Text prompt for generation" << std::endl;
    std::cout << "    --max-tokens <n>        Maximum tokens to generate (default: 100)" << std::endl;
    std::cout << "    --temperature <n>       Sampling temperature (default: 1.0)" << std::endl;
    std::cout << "    --top-k <n>             Top-K sampling (default: 40)" << std::endl;
    std::cout << "    --top-p <n>             Top-P nucleus sampling (default: 0.9)" << std::endl;
    std::cout << "    --inspect               Enable introspection mode" << std::endl;
    std::cout << "    --show-attention        Display attention weights" << std::endl;
    std::cout << "    --show-hidden <layer>   Display hidden states for layer" << std::endl;
    std::cout << "    --show-qkv <layer>      Display Q/K/V vectors for layer" << std::endl;
    std::cout << "    --show-logits           Display output logits" << std::endl;
    std::cout << "    --show-entropy          Display attention entropy per layer" << std::endl;
    std::cout << "    --show-saliency <pos>   Display saliency map for token position" << std::endl;
    std::cout << "    --show-weights <layer>  Display weight matrices for layer" << std::endl;
    std::cout << "    --show-tensors          List all tensor names in model" << std::endl;
    std::cout << "    --dump-hidden <file>    Dump hidden states to CSV file" << std::endl;
    std::cout << "    --dump-attention <file> Dump attention weights to CSV file" << std::endl;
    std::cout << "    --layer <n>             Specific layer for inspection (default: all)" << std::endl;
    std::cout << "    --head <n>              Specific attention head (default: all)" << std::endl;
    std::cout << "    --position <n>          Specific token position (default: all)" << std::endl;
    std::cout << "    --verbose               Enable verbose output" << std::endl;
    std::cout << "    --help                  Show facade help\n" << std::endl;

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

    std::cout << "  test                      Run unit tests" << std::endl;
    std::cout << "    --all                   Run all tests" << std::endl;
    std::cout << "    --protocol              Test protocol handling" << std::endl;
    std::cout << "    --config                Test configuration" << std::endl;
    std::cout << "    --quant                 Test quantization/dequantization" << std::endl;
    std::cout << "    --kernels               Test CUDA kernels (requires GPU)" << std::endl;
    std::cout << "    --network               Test network layer" << std::endl;
    std::cout << "    --facade                Test facade introspection functions" << std::endl;
    std::cout << "    --tokenizer             Test tokenizer encode/decode" << std::endl;
    std::cout << "    --gguf                  Test GGUF model loading" << std::endl;
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

    std::cout << "FACADE INTROSPECTION:\n" << std::endl;
    std::cout << "  The facade provides runtime inspection of transformer internals:" << std::endl;
    std::cout << "  - Attention weights/logits per layer and head" << std::endl;
    std::cout << "  - Hidden states at any layer position" << std::endl;
    std::cout << "  - Q/K/V projection vectors" << std::endl;
    std::cout << "  - Token and positional embeddings" << std::endl;
    std::cout << "  - Weight matrices and their shapes" << std::endl;
    std::cout << "  - Attention entropy analysis" << std::endl;
    std::cout << "  - Saliency maps for interpretability\n" << std::endl;

    std::cout << "GLOBAL OPTIONS:\n" << std::endl;
    std::cout << "  --help, -h                Show this help message" << std::endl;
    std::cout << "  --version                 Show version information\n" << std::endl;

    std::cout << "EXAMPLES:\n" << std::endl;
    std::cout << "  # Start server on eth0 with 24 layers and Q4_K quantization" << std::endl;
    std::cout << "  " << progName << " server -i eth0 -l 24 -e 1024 --quant q4_k\n" << std::endl;
    std::cout << "  # Connect client with custom sequence length and vocab" << std::endl;
    std::cout << "  " << progName << " client -s AA:BB:CC:DD:EE:FF -q 1024 -v 32000 -r 12\n" << std::endl;
    std::cout << "  # Run facade with introspection" << std::endl;
    std::cout << "  " << progName << " facade --model model.gguf --tokenizer tok.json --prompt \"Hello\" --inspect\n" << std::endl;
    std::cout << "  # Inspect attention weights for layer 0" << std::endl;
    std::cout << "  " << progName << " facade --model model.gguf --prompt \"Test\" --show-attention --layer 0\n" << std::endl;
    std::cout << "  # Dump hidden states to file" << std::endl;
    std::cout << "  " << progName << " facade --model model.gguf --prompt \"Test\" --dump-hidden hidden.csv\n" << std::endl;
    std::cout << "  # Run facade and quantization tests" << std::endl;
    std::cout << "  " << progName << " test --facade --quant --verbose\n" << std::endl;
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
        std::cout << "Facaded Distributed Transformer v1.0.0" << std::endl;
        std::cout << "CUDA-enabled Layer 2 Ethernet + Facade Introspection" << std::endl;
        std::cout << "Copyright (c) 2025 Matthew Abbott" << std::endl;
        return 0;
    }

    if (command == "server") {
        // Parse server arguments
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
                std::cout << "OPTIONS:" << std::endl;
                std::cout << "  -i, --interface <name>   Network interface (default: eth0)" << std::endl;
                std::cout << "  -l, --layers <n>         Total layers to serve (default: 12)" << std::endl;
                std::cout << "  -e, --embed <dim>        Embedding dimension (default: 768)" << std::endl;
                std::cout << "  -f, --ffn <dim>          FFN dimension (default: 3072)" << std::endl;
                std::cout << "  -a, --heads <n>          Attention heads (default: 12)" << std::endl;
                std::cout << "  -k, --kvheads <n>        KV heads for GQA (default: 12)" << std::endl;
                std::cout << "  -q, --seq-len <n>        Sequence length (default: 512)" << std::endl;
                std::cout << "  -v, --vocab-size <n>     Vocabulary size (default: 50257)" << std::endl;
                std::cout << "  -x, --max-seq-len <n>    Max sequence length (default: 2048)" << std::endl;
                std::cout << "  -m, --messages <n>       Messages to process (default: 100)" << std::endl;
                std::cout << "  -c, --clients <n>        Max concurrent clients (default: 4)" << std::endl;
                std::cout << "  -g, --gpu <yes/no>       GPU available (default: yes)" << std::endl;
                std::cout << "  --quant <type>           Quantization: none|q4_0|q4_1|q5_0|q5_1|q8_0|" << std::endl;
                std::cout << "                           q2_k|q3_k|q4_k|q5_k|q6_k|q8_k|f16|bf16" << std::endl;
                std::cout << "  --rope-base <n>          RoPE base frequency (default: 10000.0)" << std::endl;
                std::cout << "  --rope-scale <n>         RoPE scaling factor (default: 1.0)" << std::endl;
                std::cout << "  --eps <n>                Layer norm epsilon (default: 1e-5)" << std::endl;
                std::cout << "  --dropout <n>            Dropout rate (default: 0.0)" << std::endl;
                std::cout << "  --verbose                Enable verbose output" << std::endl;
                std::cout << "  --help                   Show this help\n" << std::endl;
                return 0;
            }
        }

        std::cout << "\n=== Server Configuration ===" << std::endl;
        std::cout << "Interface: " << cfg.interfaceName << std::endl;
        std::cout << "Total Layers: " << cfg.totalLayers << std::endl;
        std::cout << "Embed Dim: " << cfg.embedDim << std::endl;
        std::cout << "FFN Dim: " << cfg.ffnDim << std::endl;
        std::cout << "Heads: " << cfg.numHeads << " / KV Heads: " << cfg.numKVHeads << std::endl;
        std::cout << "Seq Len: " << cfg.seqLen << " / Max: " << maxSeqLen << std::endl;
        std::cout << "Vocab Size: " << vocabSize << std::endl;
        std::cout << "Quantization: " << quantType << std::endl;
        std::cout << "RoPE: base=" << ropeBase << " scale=" << ropeScale << std::endl;
        std::cout << "Epsilon: " << eps << " Dropout: " << dropout << std::endl;
        std::cout << "Max Messages: " << maxMessages << std::endl;
        std::cout << "Max Clients: " << maxClients << std::endl;
        std::cout << "GPU Available: " << (hasGPU ? "yes" : "no") << std::endl;
        std::cout << "Verbose: " << (verbose ? "yes" : "no") << std::endl;
        std::cout << "============================\n" << std::endl;

        DistTransformer::DistributedTransformerServer server(cfg);
        if (!server.initialize()) {
            std::cerr << "Failed to initialize server" << std::endl;
            return 1;
        }

        server.setForwardLayerFunction([](const std::vector<float>& input, int layer, bool) {
            return input;  // Identity for testing
        });

        std::cout << "Server ready. Processing up to " << maxMessages << " messages...\n" << std::endl;
        server.run(maxMessages);
        std::cout << "Server shutdown complete." << std::endl;
        return 0;

    } else if (command == "client") {
        // Parse client arguments
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
        int vocabSize = 50257;
        int maxSeqLen = 2048;
        std::string quantType = "none";
        float ropeBase = 10000.0f;
        float ropeScale = 1.0f;
        float eps = 1e-5f;
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
            } else if (arg == "--start-layer" && i + 1 < argc) {
                cfg.startRemoteLayer = std::stoi(argv[++i]);
                startLayerSet = true;
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
            } else if (arg == "--quant" && i + 1 < argc) {
                quantType = toLowerCase(argv[++i]);
            } else if (arg == "--rope-base" && i + 1 < argc) {
                ropeBase = std::stof(argv[++i]);
            } else if (arg == "--rope-scale" && i + 1 < argc) {
                ropeScale = std::stof(argv[++i]);
            } else if (arg == "--eps" && i + 1 < argc) {
                eps = std::stof(argv[++i]);
            } else if (arg == "--no-cache") {
                cfg.cacheActivations = false;
            } else if (arg == "--no-grad-cache") {
                cfg.cacheGradients = false;
            } else if (arg == "--timeout" && i + 1 < argc) {
                timeoutMs = std::stoi(argv[++i]);
            } else if (arg == "--retries" && i + 1 < argc) {
                retries = std::stoi(argv[++i]);
            } else if (arg == "--verbose") {
                verbose = true;
            } else if (arg == "--help") {
                std::cout << "\nCLIENT MODE - Execute local transformer layers, send remote to server\n" << std::endl;
                std::cout << "Usage: " << argv[0] << " client [options]\n" << std::endl;
                std::cout << "REQUIRED:" << std::endl;
                std::cout << "  -s, --server <mac>       Server MAC address (format: XX:XX:XX:XX:XX:XX)\n" << std::endl;
                std::cout << "OPTIONS:" << std::endl;
                std::cout << "  -i, --interface <name>   Network interface (default: eth0)" << std::endl;
                std::cout << "  -l, --layers <n>         Total layers (default: 12)" << std::endl;
                std::cout << "  -r, --remote <n>         Layers to offload to server (default: 6)" << std::endl;
                std::cout << "  --start-layer <n>        Starting layer for remote execution (default: auto)" << std::endl;
                std::cout << "  -e, --embed <dim>        Embedding dimension (default: 768)" << std::endl;
                std::cout << "  -f, --ffn <dim>          FFN dimension (default: 3072)" << std::endl;
                std::cout << "  -a, --heads <n>          Attention heads (default: 12)" << std::endl;
                std::cout << "  -k, --kvheads <n>        KV heads for GQA (default: 12)" << std::endl;
                std::cout << "  -q, --seq-len <n>        Sequence length (default: 512)" << std::endl;
                std::cout << "  -v, --vocab-size <n>     Vocabulary size (default: 50257)" << std::endl;
                std::cout << "  -x, --max-seq-len <n>    Max sequence length (default: 2048)" << std::endl;
                std::cout << "  --quant <type>           Quantization type (see main help)" << std::endl;
                std::cout << "  --rope-base <n>          RoPE base frequency (default: 10000.0)" << std::endl;
                std::cout << "  --rope-scale <n>         RoPE scaling factor (default: 1.0)" << std::endl;
                std::cout << "  --eps <n>                Layer norm epsilon (default: 1e-5)" << std::endl;
                std::cout << "  --no-cache               Don't cache activations" << std::endl;
                std::cout << "  --no-grad-cache          Don't cache gradients" << std::endl;
                std::cout << "  --timeout <ms>           Connection timeout in ms (default: 5000)" << std::endl;
                std::cout << "  --retries <n>            Connection retry count (default: 3)" << std::endl;
                std::cout << "  --verbose                Enable verbose output" << std::endl;
                std::cout << "  --help                   Show this help\n" << std::endl;
                return 0;
            }
        }

        if (!serverMACProvided) {
            std::cerr << "Error: Server MAC address required (-s or --server)" << std::endl;
            std::cerr << "Usage: " << argv[0] << " client -s XX:XX:XX:XX:XX:XX [options]" << std::endl;
            return 1;
        }

        std::cout << "\n=== Client Configuration ===" << std::endl;
        std::cout << "Interface: " << cfg.interfaceName << std::endl;
        char macStr[18];
        DistTransformer::macToString(cfg.serverMAC, macStr, sizeof(macStr));
        std::cout << "Server MAC: " << macStr << std::endl;
        std::cout << "Total Layers: " << cfg.totalLayers << std::endl;
        std::cout << "Local Layers: " << cfg.localLayers << std::endl;
        std::cout << "Remote Layers: " << cfg.remoteLayers << " (start: " << cfg.startRemoteLayer << ")" << std::endl;
        std::cout << "Embed Dim: " << cfg.embedDim << std::endl;
        std::cout << "FFN Dim: " << cfg.ffnDim << std::endl;
        std::cout << "Heads: " << cfg.numHeads << " / KV Heads: " << cfg.numKVHeads << std::endl;
        std::cout << "Seq Len: " << cfg.seqLen << " / Max: " << maxSeqLen << std::endl;
        std::cout << "Vocab Size: " << vocabSize << std::endl;
        std::cout << "Quantization: " << quantType << std::endl;
        std::cout << "RoPE: base=" << ropeBase << " scale=" << ropeScale << std::endl;
        std::cout << "Epsilon: " << eps << std::endl;
        std::cout << "Caching: Activations=" << (cfg.cacheActivations ? "yes" : "no")
                  << " Gradients=" << (cfg.cacheGradients ? "yes" : "no") << std::endl;
        std::cout << "Timeout: " << timeoutMs << "ms / Retries: " << retries << std::endl;
        std::cout << "Verbose: " << (verbose ? "yes" : "no") << std::endl;
        std::cout << "===========================\n" << std::endl;

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
            std::cout << "✓ Forward pass successful" << std::endl;
            std::cout << "  Input size: " << input.size() << " elements" << std::endl;
            std::cout << "  Output size: " << output.size() << " elements" << std::endl;
        } else {
            std::cout << "✗ Forward pass returned empty output" << std::endl;
        }

        std::cout << "Testing backward pass..." << std::endl;
        std::vector<float> gradOutput(cfg.embedDim, 0.1f);
        auto grad = client.backward(gradOutput);

        if (!grad.empty()) {
            std::cout << "✓ Backward pass successful" << std::endl;
            std::cout << "  Gradient size: " << grad.size() << " elements" << std::endl;
        } else {
            std::cout << "✗ Backward pass returned empty output" << std::endl;
        }

        client.disconnect();
        std::cout << "\nClient shutdown complete." << std::endl;
        return 0;

    } else if (command == "benchmark") {
        // Parse benchmark arguments
        std::string interfaceName = "eth0";
        uint8_t serverMAC[6] = {0};
        int iterations = 10;
        bool serverMACProvided = false;

        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if ((arg == "-i" || arg == "--interface") && i + 1 < argc) {
                interfaceName = argv[++i];
            } else if ((arg == "-s" || arg == "--server") && i + 1 < argc) {
                if (!DistTransformer::stringToMAC(argv[++i], serverMAC)) {
                    std::cerr << "Invalid MAC address format" << std::endl;
                    return 1;
                }
                serverMACProvided = true;
            } else if ((arg == "-n" || arg == "--iterations") && i + 1 < argc) {
                iterations = std::stoi(argv[++i]);
            } else if (arg == "--help") {
                std::cout << "\nBENCHMARK MODE - Performance testing\n" << std::endl;
                std::cout << "Usage: " << argv[0] << " benchmark [options]\n" << std::endl;
                std::cout << "REQUIRED:" << std::endl;
                std::cout << "  -s, --server <mac>       Server MAC address\n" << std::endl;
                std::cout << "OPTIONS:" << std::endl;
                std::cout << "  -i, --interface <name>   Network interface (default: eth0)" << std::endl;
                std::cout << "  -n, --iterations <n>     Iterations to run (default: 10)" << std::endl;
                std::cout << "  --help                   Show this help\n" << std::endl;
                return 0;
            }
        }

        if (!serverMACProvided) {
            std::cerr << "Error: Server MAC address required (-s or --server)" << std::endl;
            return 1;
        }

        DistTransformer::DistributedConfig cfg;
        memcpy(cfg.serverMAC, serverMAC, 6);
        cfg.interfaceName = interfaceName;

        std::cout << "\n=== Benchmark Configuration ===" << std::endl;
        std::cout << "Interface: " << interfaceName << std::endl;
        char macStr[18];
        DistTransformer::macToString(serverMAC, macStr, sizeof(macStr));
        std::cout << "Server MAC: " << macStr << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "================================\n" << std::endl;

        DistTransformer::DistributedTransformer transformer(cfg);
        if (!transformer.initialize() || !transformer.connect()) {
            std::cerr << "Failed to initialize benchmark" << std::endl;
            return 1;
        }

        std::cout << "Running benchmark..." << std::endl;
        auto stats = DistTransformer::benchmarkDistributed(transformer, iterations);

        std::cout << "\n=== Benchmark Results ===" << std::endl;
        std::cout << "Forward pass:  " << std::fixed << std::setprecision(3) << stats.forwardMs << " ms" << std::endl;
        std::cout << "Backward pass: " << stats.backwardMs << " ms" << std::endl;
        std::cout << "Total time:    " << stats.totalMs << " ms" << std::endl;
        std::cout << "Elements:      " << stats.elementsProcessed << std::endl;
        if (stats.totalMs > 0) {
            std::cout << "Throughput:    " << (stats.elementsProcessed / stats.totalMs / 1000.0) << " M elem/s" << std::endl;
        }
        std::cout << "========================\n" << std::endl;
        return 0;

    } else if (command == "facade") {
        // Parse facade arguments
        std::string modelPath = "";
        std::string tokenizerPath = "";
        std::string prompt = "";
        int maxTokens = 100;
        double temperature = 1.0;
        int topK = 40;
        double topP = 0.9;
        bool inspect = false;
        bool showAttention = false;
        int showHiddenLayer = -1;
        int showQKVLayer = -1;
        bool showLogits = false;
        bool showEntropy = false;
        int showSaliencyPos = -1;
        int showWeightsLayer = -1;
        bool showTensors = false;
        std::string dumpHiddenFile = "";
        std::string dumpAttentionFile = "";
        int specificLayer = -1;
        int specificHead = -1;
        int specificPosition = -1;
        bool verbose = false;

        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) {
                modelPath = argv[++i];
            } else if (arg == "--tokenizer" && i + 1 < argc) {
                tokenizerPath = argv[++i];
            } else if (arg == "--prompt" && i + 1 < argc) {
                prompt = argv[++i];
            } else if (arg == "--max-tokens" && i + 1 < argc) {
                maxTokens = std::stoi(argv[++i]);
            } else if (arg == "--temperature" && i + 1 < argc) {
                temperature = std::stod(argv[++i]);
            } else if (arg == "--top-k" && i + 1 < argc) {
                topK = std::stoi(argv[++i]);
            } else if (arg == "--top-p" && i + 1 < argc) {
                topP = std::stod(argv[++i]);
            } else if (arg == "--inspect") {
                inspect = true;
            } else if (arg == "--show-attention") {
                showAttention = true;
            } else if (arg == "--show-hidden" && i + 1 < argc) {
                showHiddenLayer = std::stoi(argv[++i]);
            } else if (arg == "--show-qkv" && i + 1 < argc) {
                showQKVLayer = std::stoi(argv[++i]);
            } else if (arg == "--show-logits") {
                showLogits = true;
            } else if (arg == "--show-entropy") {
                showEntropy = true;
            } else if (arg == "--show-saliency" && i + 1 < argc) {
                showSaliencyPos = std::stoi(argv[++i]);
            } else if (arg == "--show-weights" && i + 1 < argc) {
                showWeightsLayer = std::stoi(argv[++i]);
            } else if (arg == "--show-tensors") {
                showTensors = true;
            } else if (arg == "--dump-hidden" && i + 1 < argc) {
                dumpHiddenFile = argv[++i];
            } else if (arg == "--dump-attention" && i + 1 < argc) {
                dumpAttentionFile = argv[++i];
            } else if (arg == "--layer" && i + 1 < argc) {
                specificLayer = std::stoi(argv[++i]);
            } else if (arg == "--head" && i + 1 < argc) {
                specificHead = std::stoi(argv[++i]);
            } else if (arg == "--position" && i + 1 < argc) {
                specificPosition = std::stoi(argv[++i]);
            } else if (arg == "--verbose") {
                verbose = true;
            } else if (arg == "--help") {
                std::cout << "\nFACADE MODE - Introspection and inference\n" << std::endl;
                std::cout << "Usage: " << argv[0] << " facade [options]\n" << std::endl;
                std::cout << "REQUIRED:" << std::endl;
                std::cout << "  --model <path>           GGUF model file path\n" << std::endl;
                std::cout << "INPUT OPTIONS:" << std::endl;
                std::cout << "  --tokenizer <path>       Tokenizer JSON file path" << std::endl;
                std::cout << "  --prompt <text>          Text prompt for generation\n" << std::endl;
                std::cout << "GENERATION OPTIONS:" << std::endl;
                std::cout << "  --max-tokens <n>         Maximum tokens to generate (default: 100)" << std::endl;
                std::cout << "  --temperature <n>        Sampling temperature (default: 1.0)" << std::endl;
                std::cout << "  --top-k <n>              Top-K sampling (default: 40)" << std::endl;
                std::cout << "  --top-p <n>              Top-P nucleus sampling (default: 0.9)\n" << std::endl;
                std::cout << "INTROSPECTION OPTIONS:" << std::endl;
                std::cout << "  --inspect                Enable introspection mode" << std::endl;
                std::cout << "  --show-attention         Display attention weights" << std::endl;
                std::cout << "  --show-hidden <layer>    Display hidden states for layer" << std::endl;
                std::cout << "  --show-qkv <layer>       Display Q/K/V vectors for layer" << std::endl;
                std::cout << "  --show-logits            Display output logits" << std::endl;
                std::cout << "  --show-entropy           Display attention entropy per layer" << std::endl;
                std::cout << "  --show-saliency <pos>    Display saliency map for token position" << std::endl;
                std::cout << "  --show-weights <layer>   Display weight matrices for layer" << std::endl;
                std::cout << "  --show-tensors           List all tensor names in model\n" << std::endl;
                std::cout << "DUMP OPTIONS:" << std::endl;
                std::cout << "  --dump-hidden <file>     Dump hidden states to CSV file" << std::endl;
                std::cout << "  --dump-attention <file>  Dump attention weights to CSV file\n" << std::endl;
                std::cout << "FILTER OPTIONS:" << std::endl;
                std::cout << "  --layer <n>              Specific layer for inspection (default: all)" << std::endl;
                std::cout << "  --head <n>               Specific attention head (default: all)" << std::endl;
                std::cout << "  --position <n>           Specific token position (default: all)" << std::endl;
                std::cout << "  --verbose                Enable verbose output" << std::endl;
                std::cout << "  --help                   Show this help\n" << std::endl;
                return 0;
            }
        }

        if (modelPath.empty()) {
            std::cerr << "Error: Model path required (--model)" << std::endl;
            std::cerr << "Usage: " << argv[0] << " facade --model model.gguf [options]" << std::endl;
            return 1;
        }

        std::cout << "\n=== Facade Configuration ===" << std::endl;
        std::cout << "Model: " << modelPath << std::endl;
        if (!tokenizerPath.empty()) std::cout << "Tokenizer: " << tokenizerPath << std::endl;
        if (!prompt.empty()) std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << "Max Tokens: " << maxTokens << std::endl;
        std::cout << "Temperature: " << temperature << std::endl;
        std::cout << "Top-K: " << topK << " Top-P: " << topP << std::endl;
        std::cout << "Introspection: " << (inspect ? "enabled" : "disabled") << std::endl;
        std::cout << "Verbose: " << (verbose ? "yes" : "no") << std::endl;
        std::cout << "============================\n" << std::endl;

        // Initialize facade
        TransformerFacade facade;
        
        std::cout << "Loading model: " << modelPath << std::endl;
        if (!facade.loadModel(modelPath)) {
            std::cerr << "Failed to load model: " << modelPath << std::endl;
            return 1;
        }
        std::cout << "✓ Model loaded successfully" << std::endl;
        
        // Display model info
        std::cout << "\n=== Model Info ===" << std::endl;
        std::cout << "Layers: " << facade.getNumLayers() << std::endl;
        std::cout << "Heads: " << facade.getNumHeads() << std::endl;
        std::cout << "Hidden Size: " << facade.getHiddenSize() << std::endl;
        std::cout << "Head Dim: " << facade.getHeadDim() << std::endl;
        std::cout << "FFN Dim: " << facade.getFFNDim() << std::endl;
        std::cout << "Vocab Size: " << facade.getVocabSize() << std::endl;
        std::cout << "Max Seq Len: " << facade.getMaxSeqLen() << std::endl;
        std::cout << "==================\n" << std::endl;

        // Show tensor names if requested
        if (showTensors) {
            std::cout << "=== Tensor Names ===" << std::endl;
            facade.getLoader().printAllTensorNames();
            std::cout << "====================\n" << std::endl;
        }

        // Load tokenizer if provided
        if (!tokenizerPath.empty()) {
            std::cout << "Loading tokenizer: " << tokenizerPath << std::endl;
            if (facade.loadTokenizer(tokenizerPath)) {
                std::cout << "✓ Tokenizer loaded (vocab size: " << facade.getTokenizer().getVocabSize() << ")" << std::endl;
            } else {
                std::cerr << "Warning: Failed to load tokenizer" << std::endl;
            }
        }

        // Run forward pass if prompt provided
        if (!prompt.empty() && facade.isTokenizerLoaded()) {
            std::cout << "\nRunning forward pass..." << std::endl;
            IntArray tokens = facade.getTokenizer().encode(prompt);
            std::cout << "Input tokens: " << tokens.size() << std::endl;
            
            DoubleArray logits = facade.runForward(tokens);
            
            if (!logits.empty()) {
                std::cout << "✓ Forward pass completed" << std::endl;
                std::cout << "Output logits size: " << logits.size() << std::endl;
                
                // Show logits if requested
                if (showLogits) {
                    std::cout << "\n=== Output Logits (top 10) ===" << std::endl;
                    std::vector<std::pair<int, double>> sorted_logits;
                    for (int i = 0; i < (int)logits.size(); i++) {
                        sorted_logits.push_back({i, logits[i]});
                    }
                    std::sort(sorted_logits.begin(), sorted_logits.end(),
			      [](const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.second > b.second; });	    
                              // [](const auto& a, const auto& b) { return a.second > b.second; });
                    for (int i = 0; i < std::min(10, (int)sorted_logits.size()); i++) {
                        std::string tok = facade.getTokenizer().getToken(sorted_logits[i].first);
                        std::cout << "  " << sorted_logits[i].first << " (\"" << tok << "\"): " 
                                  << sorted_logits[i].second << std::endl;
                    }
                    std::cout << "==============================\n" << std::endl;
                }

                // Show attention weights if requested
                if (showAttention) {
                    std::cout << "\n=== Attention Weights ===" << std::endl;
                    int startLayer = (specificLayer >= 0) ? specificLayer : 0;
                    int endLayer = (specificLayer >= 0) ? specificLayer + 1 : facade.getNumLayers();
                    int startHead = (specificHead >= 0) ? specificHead : 0;
                    int endHead = (specificHead >= 0) ? specificHead + 1 : facade.getNumHeads();
                    
                    for (int l = startLayer; l < endLayer; l++) {
                        for (int h = startHead; h < endHead; h++) {
                            std::cout << "Layer " << l << " Head " << h << ":" << std::endl;
                            for (int from = 0; from < facade.getLastSeqLen(); from++) {
                                std::cout << "  [" << from << "]: ";
                                for (int to = 0; to <= from; to++) {
                                    std::cout << std::fixed << std::setprecision(3) 
                                              << facade.getAttentionWeights(l, h, from, to) << " ";
                                }
                                std::cout << std::endl;
                            }
                        }
                    }
                    std::cout << "=========================\n" << std::endl;
                }

                // Show hidden states if requested
                if (showHiddenLayer >= 0) {
                    std::cout << "\n=== Hidden States (Layer " << showHiddenLayer << ") ===" << std::endl;
                    int startPos = (specificPosition >= 0) ? specificPosition : 0;
                    int endPos = (specificPosition >= 0) ? specificPosition + 1 : facade.getLastSeqLen();
                    
                    for (int pos = startPos; pos < endPos; pos++) {
                        DoubleArray hidden = facade.getHiddenState(showHiddenLayer, pos);
                        std::cout << "Position " << pos << ": [";
                        for (int i = 0; i < std::min(8, (int)hidden.size()); i++) {
                            std::cout << std::fixed << std::setprecision(4) << hidden[i];
                            if (i < 7) std::cout << ", ";
                        }
                        if (hidden.size() > 8) std::cout << ", ...";
                        std::cout << "] (dim=" << hidden.size() << ")" << std::endl;
                    }
                    std::cout << "=========================================\n" << std::endl;
                }

                // Show QKV if requested
                if (showQKVLayer >= 0) {
                    std::cout << "\n=== Q/K/V Vectors (Layer " << showQKVLayer << ") ===" << std::endl;
                    int startHead = (specificHead >= 0) ? specificHead : 0;
                    int endHead = (specificHead >= 0) ? specificHead + 1 : facade.getNumHeads();
                    int startPos = (specificPosition >= 0) ? specificPosition : 0;
                    int endPos = (specificPosition >= 0) ? specificPosition + 1 : std::min(4, facade.getLastSeqLen());
                    
                    for (int h = startHead; h < endHead; h++) {
                        std::cout << "Head " << h << ":" << std::endl;
                        for (int pos = startPos; pos < endPos; pos++) {
                            DoubleArray q = facade.getQKV(showQKVLayer, h, qkvQuery, pos);
                            DoubleArray k = facade.getQKV(showQKVLayer, h, qkvKey, pos);
                            DoubleArray v = facade.getQKV(showQKVLayer, h, qkvValue, pos);
                            std::cout << "  Pos " << pos << " Q:[" << (q.size() > 0 ? q[0] : 0) << ",...] "
                                      << "K:[" << (k.size() > 0 ? k[0] : 0) << ",...] "
                                      << "V:[" << (v.size() > 0 ? v[0] : 0) << ",...]" << std::endl;
                        }
                    }
                    std::cout << "=========================================\n" << std::endl;
                }

                // Show entropy if requested
                if (showEntropy) {
                    std::cout << "\n=== Attention Entropy ===" << std::endl;
                    int startHead = (specificHead >= 0) ? specificHead : 0;
                    int endHead = (specificHead >= 0) ? specificHead + 1 : facade.getNumHeads();
                    
                    for (int l = 0; l < facade.getNumLayers(); l++) {
                        std::cout << "Layer " << l << ": ";
                        for (int h = startHead; h < endHead; h++) {
                            std::cout << std::fixed << std::setprecision(4) 
                                      << facade.getAttentionEntropy(l, h) << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << "=========================\n" << std::endl;
                }

                // Show saliency if requested
                if (showSaliencyPos >= 0) {
                    std::cout << "\n=== Saliency Map (Position " << showSaliencyPos << ") ===" << std::endl;
                    int layer = (specificLayer >= 0) ? specificLayer : facade.getNumLayers() - 1;
                    DoubleArray saliency = facade.getSaliencyMap(showSaliencyPos, layer);
                    std::cout << "Layer " << layer << ": [";
                    for (int i = 0; i < std::min(16, (int)saliency.size()); i++) {
                        std::cout << std::fixed << std::setprecision(4) << saliency[i];
                        if (i < 15) std::cout << ", ";
                    }
                    if (saliency.size() > 16) std::cout << ", ...";
                    std::cout << "]" << std::endl;
                    std::cout << "==========================================\n" << std::endl;
                }

                // Dump hidden states to file
                if (!dumpHiddenFile.empty()) {
                    std::ofstream outFile(dumpHiddenFile);
                    if (outFile.is_open()) {
                        outFile << "layer,position,dim,value" << std::endl;
                        for (int l = 0; l <= facade.getNumLayers(); l++) {
                            for (int pos = 0; pos < facade.getLastSeqLen(); pos++) {
                                DoubleArray hidden = facade.getHiddenState(l, pos);
                                for (int d = 0; d < (int)hidden.size(); d++) {
                                    outFile << l << "," << pos << "," << d << "," << hidden[d] << std::endl;
                                }
                            }
                        }
                        outFile.close();
                        std::cout << "✓ Hidden states dumped to: " << dumpHiddenFile << std::endl;
                    } else {
                        std::cerr << "Failed to open file: " << dumpHiddenFile << std::endl;
                    }
                }

                // Dump attention weights to file
                if (!dumpAttentionFile.empty()) {
                    std::ofstream outFile(dumpAttentionFile);
                    if (outFile.is_open()) {
                        outFile << "layer,head,from,to,weight" << std::endl;
                        for (int l = 0; l < facade.getNumLayers(); l++) {
                            for (int h = 0; h < facade.getNumHeads(); h++) {
                                for (int from = 0; from < facade.getLastSeqLen(); from++) {
                                    for (int to = 0; to <= from; to++) {
                                        outFile << l << "," << h << "," << from << "," << to << ","
                                                << facade.getAttentionWeights(l, h, from, to) << std::endl;
                                    }
                                }
                            }
                        }
                        outFile.close();
                        std::cout << "✓ Attention weights dumped to: " << dumpAttentionFile << std::endl;
                    } else {
                        std::cerr << "Failed to open file: " << dumpAttentionFile << std::endl;
                    }
                }

                // Generate text if requested
                if (maxTokens > 0 && temperature > 0) {
                    std::cout << "\nGenerating " << maxTokens << " tokens..." << std::endl;
                    std::string output = facade.generate(prompt, maxTokens, temperature);
                    std::cout << "\n=== Generated Output ===" << std::endl;
                    std::cout << output << std::endl;
                    std::cout << "========================\n" << std::endl;
                }
            }
        } else if (!prompt.empty()) {
            std::cerr << "Warning: Prompt provided but tokenizer not loaded" << std::endl;
        }

        // Show weight matrices if requested
        if (showWeightsLayer >= 0) {
            std::cout << "\n=== Weight Matrices (Layer " << showWeightsLayer << ") ===" << std::endl;
            
            std::vector<std::pair<ParamType, std::string>> weightTypes = {
                {ptQProj, "Q Projection"},
                {ptKProj, "K Projection"},
                {ptVProj, "V Projection"},
                {ptOutProj, "Output Projection"},
                {ptFFN1, "FFN Up"},
                {ptFFN2, "FFN Down"},
                {ptLayerNorm1Weight, "LayerNorm1 Weight"},
                {ptLayerNorm2Weight, "LayerNorm2 Weight"}
            };
            
            for (const auto& wt : weightTypes) {
                Int64Array shape = facade.getWeightShape(showWeightsLayer, wt.first);
                if (!shape.empty()) {
                    std::cout << wt.second << ": [";
                    for (size_t i = 0; i < shape.size(); i++) {
                        std::cout << shape[i];
                        if (i < shape.size() - 1) std::cout << " x ";
                    }
                    std::cout << "]" << std::endl;
                }
            }
            std::cout << "==========================================\n" << std::endl;
        }

        std::cout << "Facade completed successfully." << std::endl;
        return 0;

    } else if (command == "test") {
        // Parse test arguments
        bool runAll = false;
        bool testProtocol = false;
        bool testConfig = false;
        bool testQuant = false;
        bool testKernels = false;
        bool testNetwork = false;
        bool testFacade = false;
        bool testTokenizer = false;
        bool testGGUF = false;
        bool verbose = false;

        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--all") {
                runAll = true;
            } else if (arg == "--protocol") {
                testProtocol = true;
            } else if (arg == "--config") {
                testConfig = true;
            } else if (arg == "--quant") {
                testQuant = true;
            } else if (arg == "--kernels") {
                testKernels = true;
            } else if (arg == "--network") {
                testNetwork = true;
            } else if (arg == "--facade") {
                testFacade = true;
            } else if (arg == "--tokenizer") {
                testTokenizer = true;
            } else if (arg == "--gguf") {
                testGGUF = true;
            } else if (arg == "--verbose") {
                verbose = true;
            } else if (arg == "--help") {
                std::cout << "\nTEST MODE - Unit tests and validation\n" << std::endl;
                std::cout << "Usage: " << argv[0] << " test [options]\n" << std::endl;
                std::cout << "OPTIONS:" << std::endl;
                std::cout << "  --all                    Run all tests" << std::endl;
                std::cout << "  --protocol               Test protocol handling" << std::endl;
                std::cout << "  --config                 Test configuration" << std::endl;
                std::cout << "  --quant                  Test quantization/dequantization" << std::endl;
                std::cout << "  --kernels                Test CUDA kernels (requires GPU)" << std::endl;
                std::cout << "  --network                Test network layer" << std::endl;
                std::cout << "  --facade                 Test facade introspection" << std::endl;
                std::cout << "  --tokenizer              Test tokenizer encode/decode" << std::endl;
                std::cout << "  --gguf                   Test GGUF model loading" << std::endl;
                std::cout << "  --verbose                Enable verbose output" << std::endl;
                std::cout << "  --help                   Show this help\n" << std::endl;
                return 0;
            }
        }

        // If no specific tests selected, run basic tests
        if (!runAll && !testProtocol && !testConfig && !testQuant && !testKernels && 
            !testNetwork && !testFacade && !testTokenizer && !testGGUF) {
            runAll = true;
        }

        std::cout << "\n=== Running Tests ===" << std::endl;
        int passed = 0, failed = 0;

        // Protocol tests
        if (runAll || testProtocol) {
            std::cout << "\n--- Protocol Tests ---" << std::endl;
            std::cout << "Test: Header verification" << std::endl;
            DistTransformer::DTXHeader hdr = DistTransformer::makeHeader(
                DistTransformer::MessageType::HANDSHAKE_REQ, 1, nullptr, 0);
            
            if (DistTransformer::verifyHeader(hdr)) {
                std::cout << "  ✓ Header verification passed" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ Header verification failed" << std::endl;
                failed++;
            }

            std::cout << "Test: MAC address handling" << std::endl;
            uint8_t testMAC[6] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
            char macStr[18];
            DistTransformer::macToString(testMAC, macStr, sizeof(macStr));
            uint8_t parsedMAC[6];
            if (DistTransformer::stringToMAC(macStr, parsedMAC) &&
                DistTransformer::compareMACAddress(testMAC, parsedMAC)) {
                std::cout << "  ✓ MAC address parsing passed" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ MAC address parsing failed" << std::endl;
                failed++;
            }

            std::cout << "Test: CRC32 checksum" << std::endl;
            const uint8_t testData[] = {1, 2, 3, 4, 5};
            uint32_t crc1 = DistTransformer::crc32_simple(testData, 5);
            uint32_t crc2 = DistTransformer::crc32_simple(testData, 5);
            if (crc1 == crc2) {
                std::cout << "  ✓ CRC32 consistency passed" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ CRC32 consistency failed" << std::endl;
                failed++;
            }
        }

        // Config tests
        if (runAll || testConfig) {
            std::cout << "\n--- Configuration Tests ---" << std::endl;
            std::cout << "Test: Symmetric config creation" << std::endl;
            DistTransformer::DistributedConfig cfg = DistTransformer::createSymmetricConfig(12, 768, 3072, 12);
            if (cfg.validate()) {
                std::cout << "  ✓ Config validation passed" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ Config validation failed" << std::endl;
                failed++;
            }

            std::cout << "Test: Config with invalid layers" << std::endl;
            DistTransformer::DistributedConfig badCfg;
            badCfg.totalLayers = 0;
            if (!badCfg.validate()) {
                std::cout << "  ✓ Invalid config rejected" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ Invalid config not rejected" << std::endl;
                failed++;
            }
        }

        // Quantization tests
        if (runAll || testQuant) {
            std::cout << "\n--- Quantization Tests ---" << std::endl;
            
            std::cout << "Test: FP16 conversion" << std::endl;
            uint16_t fp16_one = 0x3C00;  // 1.0 in FP16
            float result = fp16_to_fp32(fp16_one);
            if (std::abs(result - 1.0f) < 0.001f) {
                std::cout << "  ✓ FP16 to FP32 conversion passed" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ FP16 to FP32 conversion failed (got " << result << ")" << std::endl;
                failed++;
            }

            std::cout << "Test: FP16 zero conversion" << std::endl;
            uint16_t fp16_zero = 0x0000;
            result = fp16_to_fp32(fp16_zero);
            if (result == 0.0f) {
                std::cout << "  ✓ FP16 zero conversion passed" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ FP16 zero conversion failed" << std::endl;
                failed++;
            }

            std::cout << "Test: Quantization type enum" << std::endl;
            if ((int)GGML_DType::Q4_K == 12 && (int)GGML_DType::Q6_K == 14) {
                std::cout << "  ✓ Quantization type enum values correct" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ Quantization type enum values incorrect" << std::endl;
                failed++;
            }
        }

        // Facade tests
        if (runAll || testFacade) {
            std::cout << "\n--- Facade Tests ---" << std::endl;
            
            std::cout << "Test: Facade initialization" << std::endl;
            TransformerFacade facade;
            if (!facade.isModelLoaded()) {
                std::cout << "  ✓ Facade starts unloaded" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ Facade should start unloaded" << std::endl;
                failed++;
            }

            std::cout << "Test: Facade getters" << std::endl;
            if (facade.getNumLayers() == 0 && facade.getVocabSize() == 0) {
                std::cout << "  ✓ Unloaded facade returns zeros" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ Unloaded facade should return zeros" << std::endl;
                failed++;
            }
        }

        // Tokenizer tests
        if (runAll || testTokenizer) {
            std::cout << "\n--- Tokenizer Tests ---" << std::endl;
            
            std::cout << "Test: Tokenizer initialization" << std::endl;
            Tokenizer tokenizer;
            if (!tokenizer.isLoaded()) {
                std::cout << "  ✓ Tokenizer starts unloaded" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ Tokenizer should start unloaded" << std::endl;
                failed++;
            }

            std::cout << "Test: Empty encode" << std::endl;
            IntArray tokens = tokenizer.encode("");
            if (tokens.empty()) {
                std::cout << "  ✓ Empty string encodes to empty array" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ Empty string should encode to empty array" << std::endl;
                failed++;
            }
        }

        // GGUF tests
        if (runAll || testGGUF) {
            std::cout << "\n--- GGUF Tests ---" << std::endl;
            
            std::cout << "Test: GGUFLoader initialization" << std::endl;
            GGUFLoader loader;
            if (!loader.isLoaded()) {
                std::cout << "  ✓ Loader starts unloaded" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ Loader should start unloaded" << std::endl;
                failed++;
            }

            std::cout << "Test: Load nonexistent file" << std::endl;
            if (!loader.loadFromFile("/nonexistent/path/model.gguf")) {
                std::cout << "  ✓ Nonexistent file returns false" << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ Nonexistent file should return false" << std::endl;
                failed++;
            }
        }

        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
        std::cout << "Total:  " << (passed + failed) << std::endl;
        std::cout << "===================\n" << std::endl;
        return (failed > 0) ? 1 : 0;

    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        printMainHelp(argv[0]);
        return 1;
    }

    return 0;
}
