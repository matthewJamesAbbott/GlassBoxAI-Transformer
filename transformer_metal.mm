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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// macOS networking
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <net/bpf.h>
#include <net/if_dl.h>
#include <net/ethernet.h>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/select.h>

#define MTL_CHECK(obj, msg) \
    do { \
        if (!(obj)) { \
            std::cerr << "Metal error: " << msg << std::endl; \
            exit(1); \
        } \
    } while(0)

// ================================================================================
// QUANTIZATION SUPPORT
// ================================================================================

enum class GGML_DType : int {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12,
    Q5_K = 13, Q6_K = 14, Q8_K = 15, BFLOAT16 = 30, UNKNOWN = -1
};

#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32

struct block_q2_K { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; uint16_t d; uint16_t dmin; };
struct block_q3_K { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; uint16_t d; };
struct block_q4_K { uint16_t d; uint16_t dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; };
struct block_q5_K { uint16_t d; uint16_t dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; };
struct block_q6_K { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; uint16_t d; };
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };
struct block_q8_0 { uint16_t d; int8_t qs[QK8_0]; };

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

inline void get_scale_min_k4(int j, const uint8_t* scales, uint8_t* sc, uint8_t* m) {
    if (j < 4) { *sc = scales[j] & 63; *m = scales[j + 4] & 63; }
    else { *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4); *m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4); }
}

inline void dequant_row_q4_K(const block_q4_K* blocks, float* output, int cols) {
    const int nb = cols / QK_K;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* q = blocks[i].qs;
        const float d = fp16_to_fp32(blocks[i].d);
        const float dmin = fp16_to_fp32(blocks[i].dmin);
        float* y = output + i * QK_K;
        int is = 0;
        uint8_t sc, m;
        for (int n = 0; n < QK_K; n += 64) {
            get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) y[n + l] = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) y[n + 32 + l] = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
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
                const int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[n + l] = d * sc[is] * q1;
                y[n + l + 32] = d * sc[is + 2] * q2;
                y[n + l + 64] = d * sc[is + 4] * q3;
                y[n + l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
}

inline void dequant_row_q8_0(const block_q8_0* blocks, float* output, int cols) {
    int nb = cols / QK8_0;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < QK8_0; ++j) output[i * QK8_0 + j] = d * blocks[i].qs[j];
    }
}

// ================================================================================
// METAL COMPUTE ENGINE
// ================================================================================

class MetalCompute {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;
    
    MetalCompute() : device(nil), commandQueue(nil), library(nil) {}
    
    ~MetalCompute() {
        pipelines.clear();
        library = nil;
        commandQueue = nil;
        device = nil;
    }
    
    bool initialize(const std::string& metalLibPath = "") {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
        }
        if (!device) {
            std::cerr << "Metal error: No Metal device found (MTLCreateSystemDefaultDevice returned nil)" << std::endl;
            std::cerr << "This is unexpected on Apple Silicon. Try running with: sudo ./transformer_metal test" << std::endl;
            return false;
        }
        
        commandQueue = [device newCommandQueue];
        MTL_CHECK(commandQueue, "Failed to create command queue");
        
        NSError* error = nil;
        if (!metalLibPath.empty()) {
            NSString* path = [NSString stringWithUTF8String:metalLibPath.c_str()];
            NSURL* url = [NSURL fileURLWithPath:path];
            library = [device newLibraryWithURL:url error:&error];
        } else {
            NSString* path = [[NSBundle mainBundle] pathForResource:@"transformer_kernels" ofType:@"metallib"];
            if (path) {
                library = [device newLibraryWithFile:path error:&error];
            } else {
                library = [device newDefaultLibrary];
            }
        }
        
        if (!library) {
            std::cerr << "Warning: Could not load Metal library, will compile from source" << std::endl;
            return loadFromSource();
        }
        
        return loadPipelines();
    }
    
    bool loadFromSource() {
        NSError* error = nil;
        
        // Try multiple paths to find the Metal source file
        NSArray* searchPaths = @[
            @"transformer_kernels.metal",
            [[NSBundle mainBundle].bundlePath stringByAppendingPathComponent:@"transformer_kernels.metal"],
            [[[NSBundle mainBundle].executablePath stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"transformer_kernels.metal"],
            [NSString stringWithFormat:@"%s/transformer_kernels.metal", getenv("HOME") ?: "."]
        ];
        
        NSString* source = nil;
        for (NSString* path in searchPaths) {
            source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
            if (source) {
                std::cout << "[Metal] Loaded source from: " << [path UTF8String] << std::endl;
                break;
            }
        }
        
        if (!source) {
            std::cerr << "[Metal] Could not load Metal source file from any search path" << std::endl;
            return false;
        }
        library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            std::cerr << "[Metal] Failed to compile Metal source: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        return loadPipelines();
    }
    
    bool loadPipelines() {
        std::vector<std::string> kernelNames = {
            "matmul_fp32", "matmul_tiled_fp32", "gelu_fp32", "silu_fp32", "relu_fp32",
            "softmax_fp32", "softmax_parallel_fp32", "layernorm_fp32", "rmsnorm_fp32",
            "rope_fp32", "add_fp32", "mul_fp32", "scale_fp32", "apply_causal_mask_fp32",
            "embedding_lookup_fp32", "copy_fp32"
        };
        
        NSError* error = nil;
        for (const auto& name : kernelNames) {
            NSString* nsName = [NSString stringWithUTF8String:name.c_str()];
            id<MTLFunction> func = [library newFunctionWithName:nsName];
            if (!func) {
                std::cerr << "[Metal] Failed to find function: " << name << std::endl;
                continue;
            }
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&error];
            if (!pipeline) {
                std::cerr << "[Metal] Failed to create pipeline for " << name << ": "
                          << (error ? [[error localizedDescription] UTF8String] : "unknown error") << std::endl;
                continue;
            }
            pipelines[name] = pipeline;
        }
        
        std::cout << "[Metal] Loaded " << pipelines.size() << " compute pipelines" << std::endl;
        return !pipelines.empty();
    }
    
    id<MTLBuffer> createBuffer(size_t size) {
        return [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    }
    
    id<MTLBuffer> createBuffer(const void* data, size_t size) {
        return [device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    }
    
    void matmul(id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                id<MTLBuffer> bias, int M, int N, int K, bool useTiled = true) {
        std::string kernelName = useTiled ? "matmul_tiled_fp32" : "matmul_fp32";
        auto it = pipelines.find(kernelName);
        if (it == pipelines.end()) return;
        
        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:it->second];
        [encoder setBuffer:A offset:0 atIndex:0];
        [encoder setBuffer:B offset:0 atIndex:1];
        [encoder setBuffer:C offset:0 atIndex:2];
        [encoder setBuffer:bias offset:0 atIndex:3];
        [encoder setBytes:&M length:sizeof(int) atIndex:4];
        [encoder setBytes:&N length:sizeof(int) atIndex:5];
        [encoder setBytes:&K length:sizeof(int) atIndex:6];
        int hasBias = (bias != nil) ? 1 : 0;
        [encoder setBytes:&hasBias length:sizeof(int) atIndex:7];
        
        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
    
    void gelu(id<MTLBuffer> input, id<MTLBuffer> output, int size) {
        runElementwise("gelu_fp32", input, output, size);
    }
    
    void softmax(id<MTLBuffer> data, int rows, int cols) {
        auto it = pipelines.find("softmax_fp32");
        if (it == pipelines.end()) return;
        
        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:it->second];
        [encoder setBuffer:data offset:0 atIndex:0];
        [encoder setBytes:&rows length:sizeof(int) atIndex:1];
        [encoder setBytes:&cols length:sizeof(int) atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(rows, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(std::min(rows, 256), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
    
private:
    void runElementwise(const std::string& kernel, id<MTLBuffer> input, id<MTLBuffer> output, int size) {
        auto it = pipelines.find(kernel);
        if (it == pipelines.end()) return;
        
        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:it->second];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBytes:&size length:sizeof(int) atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        NSUInteger threadGroupWidth = it->second.maxTotalThreadsPerThreadgroup;
        MTLSize threadGroupSize = MTLSizeMake(std::min((NSUInteger)size, threadGroupWidth), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
};

// ================================================================================
// PROTOCOL DEFINITIONS
// ================================================================================

namespace DistTransformer {

const uint16_t DTX_ETHERTYPE = 0x9998;
const int DTX_MAX_PAYLOAD = 1472;
const int DTX_VERSION = 1;
const int DTX_MAGIC = 0xDEADBEEF;
const int DTX_CONNECT_TIMEOUT = 5000;
const int DTX_FRAME_TIMEOUT = 10000;

enum class MessageType : uint8_t {
    HANDSHAKE_REQ = 1, HANDSHAKE_ACK = 2, LAYER_CONFIG = 10, LAYER_CONFIG_ACK = 11,
    FORWARD_START = 20, FORWARD_CHUNK = 21, FORWARD_DONE = 22,
    FORWARD_RESULT = 30, FORWARD_COMPLETE = 31,
    BACKWARD_START = 40, BACKWARD_CHUNK = 41, BACKWARD_DONE = 42,
    BACKWARD_RESULT = 50, BACKWARD_COMPLETE = 51,
    PING = 100, PONG = 101, ERROR_MSG = 200, DISCONNECT = 201
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

inline uint32_t crc32_simple(const uint8_t* data, uint32_t len) {
    uint32_t crc = 0xFFFFFFFFU;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320U : 0);
    }
    return crc ^ 0xFFFFFFFFU;
}

inline DTXHeader makeHeader(MessageType type, uint16_t seq, const uint8_t* payload, uint32_t payloadLen) {
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
    return hdr.magic == DTX_MAGIC && hdr.version == DTX_VERSION;
}

inline bool verifyChecksum(const DTXHeader& hdr, const uint8_t* payload) {
    if (hdr.payloadLen == 0) return hdr.checksum == 0;
    return crc32_simple(payload, hdr.payloadLen) == hdr.checksum;
}

// ================================================================================
// macOS NETWORK LAYER (BPF-based raw Ethernet)
// ================================================================================

struct EthernetFrame {
    uint8_t destMAC[6];
    uint8_t srcMAC[6];
    uint16_t etherType;
    std::vector<uint8_t> payload;
    EthernetFrame() : etherType(DTX_ETHERTYPE) { memset(destMAC, 0, 6); memset(srcMAC, 0, 6); }
};

enum class ConnectionState { DISCONNECTED, CONNECTING, CONNECTED, ERROR };

bool getMACAddress(const std::string& ifName, uint8_t* mac) {
    struct ifaddrs* ifap;
    if (getifaddrs(&ifap) != 0) return false;
    
    bool found = false;
    for (struct ifaddrs* ifa = ifap; ifa; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_LINK && ifName == ifa->ifa_name) {
            struct sockaddr_dl* sdl = (struct sockaddr_dl*)ifa->ifa_addr;
            if (sdl->sdl_alen == 6) {
                memcpy(mac, LLADDR(sdl), 6);
                found = true;
                break;
            }
        }
    }
    freeifaddrs(ifap);
    return found;
}

bool compareMACAddress(const uint8_t* mac1, const uint8_t* mac2) { return memcmp(mac1, mac2, 6) == 0; }

void macToString(const uint8_t* mac, char* str, size_t len) {
    snprintf(str, len, "%02x:%02x:%02x:%02x:%02x:%02x", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

bool stringToMAC(const char* str, uint8_t* mac) {
    return sscanf(str, "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx", &mac[0], &mac[1], &mac[2], &mac[3], &mac[4], &mac[5]) == 6;
}

static int openBPF(const std::string& ifName) {
    char bpfDev[32];
    int bpfFd = -1;
    
    for (int i = 0; i < 256; i++) {
        snprintf(bpfDev, sizeof(bpfDev), "/dev/bpf%d", i);
        bpfFd = open(bpfDev, O_RDWR);
        if (bpfFd >= 0) break;
    }
    
    if (bpfFd < 0) {
        std::cerr << "Error: Cannot open BPF device. Need root privileges." << std::endl;
        return -1;
    }
    
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, ifName.c_str(), IFNAMSIZ - 1);
    
    if (ioctl(bpfFd, BIOCSETIF, &ifr) < 0) {
        std::cerr << "Error: Cannot bind BPF to interface: " << ifName << std::endl;
        close(bpfFd);
        return -1;
    }
    
    int enable = 1;
    ioctl(bpfFd, BIOCIMMEDIATE, &enable);
    ioctl(bpfFd, BIOCSHDRCMPLT, &enable);
    
    unsigned int bufLen = 0;
    ioctl(bpfFd, BIOCGBLEN, &bufLen);
    
    return bpfFd;
}

static bool sendBPFFrame(int bpfFd, const uint8_t* destMAC, const uint8_t* srcMAC,
                         const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> frame(14 + payload.size());
    memcpy(&frame[0], destMAC, 6);
    memcpy(&frame[6], srcMAC, 6);
    uint16_t etherType = htons(DTX_ETHERTYPE);
    memcpy(&frame[12], &etherType, 2);
    memcpy(&frame[14], payload.data(), payload.size());
    
    ssize_t written = write(bpfFd, frame.data(), frame.size());
    return written == (ssize_t)frame.size();
}

static bool receiveBPFFrame(int bpfFd, EthernetFrame& frame, int timeoutMs, std::vector<uint8_t>& bpfBuffer) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(bpfFd, &fds);
    
    struct timeval tv;
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    
    int ret = select(bpfFd + 1, &fds, nullptr, nullptr, &tv);
    if (ret <= 0) return false;
    
    if (bpfBuffer.empty()) {
        unsigned int bufLen = 0;
        ioctl(bpfFd, BIOCGBLEN, &bufLen);
        bpfBuffer.resize(bufLen);
    }
    
    ssize_t readLen = read(bpfFd, bpfBuffer.data(), bpfBuffer.size());
    if (readLen <= 0) return false;
    
    struct bpf_hdr* bh = (struct bpf_hdr*)bpfBuffer.data();
    uint8_t* pkt = bpfBuffer.data() + bh->bh_hdrlen;
    uint32_t pktLen = bh->bh_caplen;
    
    if (pktLen < 14) return false;
    
    memcpy(frame.destMAC, pkt, 6);
    memcpy(frame.srcMAC, pkt + 6, 6);
    memcpy(&frame.etherType, pkt + 12, 2);
    frame.etherType = ntohs(frame.etherType);
    
    if (frame.etherType != DTX_ETHERTYPE) return false;
    
    frame.payload.assign(pkt + 14, pkt + pktLen);
    return true;
}

// ================================================================================
// TRANSFORMER SERVER
// ================================================================================

class TransformerServer {
public:
    TransformerServer(const std::string& ifName, uint32_t sId = 0x12345678)
        : interfaceName(ifName), serverId(sId), bpfFd(-1) {}
    
    ~TransformerServer() { if (bpfFd >= 0) close(bpfFd); }
    
    bool initialize() {
        if (!getMACAddress(interfaceName, localMAC)) {
            std::cerr << "Error: Cannot get MAC address for " << interfaceName << std::endl;
            return false;
        }
        bpfFd = openBPF(interfaceName);
        if (bpfFd < 0) return false;
        state = ConnectionState::CONNECTED;
        char macStr[18];
        macToString(localMAC, macStr, sizeof(macStr));
        std::cout << "[Server] Initialized on " << interfaceName << " (" << macStr << ")" << std::endl;
        return true;
    }
    
    using ForwardCallback = std::function<std::vector<float>(const std::vector<float>&, uint16_t, uint8_t, uint8_t)>;
    using BackwardCallback = std::function<std::vector<float>(const std::vector<float>&, uint16_t, uint8_t, uint8_t)>;
    
    void setForwardCallback(ForwardCallback cb) { forwardCallback = cb; }
    void setBackwardCallback(BackwardCallback cb) { backwardCallback = cb; }
    
    bool processNextMessage(int timeoutMs = 1000);
    void run(int maxMessages = -1);
    ConnectionState getState() const { return state; }
    
private:
    std::string interfaceName;
    uint32_t serverId;
    int bpfFd;
    uint8_t localMAC[6];
    ConnectionState state = ConnectionState::DISCONNECTED;
    std::vector<uint8_t> bpfBuffer;
    
    struct ClientSession {
        uint32_t clientId;
        uint8_t clientMAC[6];
        HandshakeReq config;
    };
    std::vector<ClientSession> connectedClients;
    ForwardCallback forwardCallback;
    BackwardCallback backwardCallback;
    
    bool sendFrame(const uint8_t* destMAC, const DTXHeader& hdr, const uint8_t* payload);
    void handleHandshakeReq(const uint8_t* srcMAC, const DTXHeader& hdr, const uint8_t* payload);
    void handleForwardChunk(const uint8_t* srcMAC, const DTXHeader& hdr, const uint8_t* payload);
    void handleBackwardChunk(const uint8_t* srcMAC, const DTXHeader& hdr, const uint8_t* payload);
    void handleDisconnect(const uint8_t* srcMAC);
};

bool TransformerServer::processNextMessage(int timeoutMs) {
    EthernetFrame frame;
    if (!receiveBPFFrame(bpfFd, frame, timeoutMs, bpfBuffer)) return false;
    if (frame.payload.size() < sizeof(DTXHeader)) return false;
    
    DTXHeader hdr;
    memcpy(&hdr, frame.payload.data(), sizeof(DTXHeader));
    if (!verifyHeader(hdr)) return false;
    
    uint8_t* payloadData = frame.payload.data() + sizeof(DTXHeader);
    if (!verifyChecksum(hdr, payloadData)) return false;
    
    switch (static_cast<MessageType>(hdr.msgType)) {
        case MessageType::HANDSHAKE_REQ: handleHandshakeReq(frame.srcMAC, hdr, payloadData); break;
        case MessageType::FORWARD_CHUNK: handleForwardChunk(frame.srcMAC, hdr, payloadData); break;
        case MessageType::BACKWARD_CHUNK: handleBackwardChunk(frame.srcMAC, hdr, payloadData); break;
        case MessageType::DISCONNECT: handleDisconnect(frame.srcMAC); break;
        default: break;
    }
    return true;
}

void TransformerServer::run(int maxMessages) {
    std::cout << "[Server] Running..." << std::endl;
    int count = 0;
    while (maxMessages < 0 || count < maxMessages) { processNextMessage(1000); count++; }
}

void TransformerServer::handleHandshakeReq(const uint8_t* srcMAC, const DTXHeader&, const uint8_t* payload) {
    HandshakeReq req;
    memcpy(&req, payload, sizeof(HandshakeReq));
    
    ClientSession session;
    session.clientId = req.clientId;
    memcpy(session.clientMAC, srcMAC, 6);
    session.config = req;
    connectedClients.push_back(session);
    
    HandshakeAck ack = {serverId, 1, 4, DTX_VERSION};
    DTXHeader respHdr = makeHeader(MessageType::HANDSHAKE_ACK, 1, (const uint8_t*)&ack, sizeof(ack));
    sendFrame(srcMAC, respHdr, (const uint8_t*)&ack);
    
    char macStr[18];
    macToString(srcMAC, macStr, sizeof(macStr));
    std::cout << "[Server] Client connected: " << macStr << std::endl;
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
            ForwardResult res = {chunk.chunkId, chunk.seqStart, chunk.seqLen, chunk.embedDim,
                                (uint32_t)(result.size() * sizeof(float)), 0};
            std::vector<uint8_t> respPayload(sizeof(ForwardResult) + res.dataSize);
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
            BackwardResult res = {chunk.chunkId, chunk.seqStart, chunk.seqLen, chunk.gradDim,
                                 (uint32_t)(result.size() * sizeof(float)), 0};
            std::vector<uint8_t> respPayload(sizeof(BackwardResult) + res.dataSize);
            memcpy(respPayload.data(), &res, sizeof(BackwardResult));
            memcpy(&respPayload[sizeof(BackwardResult)], result.data(), res.dataSize);
            DTXHeader respHdr = makeHeader(MessageType::BACKWARD_RESULT, hdr.sequenceNum + 1,
                                          respPayload.data(), respPayload.size());
            sendFrame(srcMAC, respHdr, respPayload.data());
        }
    }
}

void TransformerServer::handleDisconnect(const uint8_t* srcMAC) {
    auto it = std::find_if(connectedClients.begin(), connectedClients.end(),
                          [srcMAC](const ClientSession& s) { return compareMACAddress(s.clientMAC, srcMAC); });
    if (it != connectedClients.end()) {
        char macStr[18];
        macToString(srcMAC, macStr, sizeof(macStr));
        std::cout << "[Server] Client disconnected: " << macStr << std::endl;
        connectedClients.erase(it);
    }
}

bool TransformerServer::sendFrame(const uint8_t* destMAC, const DTXHeader& hdr, const uint8_t* payload) {
    std::vector<uint8_t> framePayload(sizeof(DTXHeader) + hdr.payloadLen);
    memcpy(framePayload.data(), &hdr, sizeof(DTXHeader));
    if (payload && hdr.payloadLen > 0) memcpy(&framePayload[sizeof(DTXHeader)], payload, hdr.payloadLen);
    return sendBPFFrame(bpfFd, destMAC, localMAC, framePayload);
}

// ================================================================================
// TRANSFORMER CLIENT
// ================================================================================

class TransformerClient {
public:
    TransformerClient(const std::string& ifName) : interfaceName(ifName), bpfFd(-1) { memset(serverMAC, 0, 6); }
    ~TransformerClient() { if (bpfFd >= 0) close(bpfFd); }
    
    bool initialize(const uint8_t* srvMAC);
    void setConfig(uint16_t seqLen, uint16_t embedDim, uint32_t ffnDim, uint8_t numHeads, uint8_t numKVHeads);
    void setLayerConfig(uint8_t startLayer, uint8_t numLayers, bool keepActivations = true);
    std::vector<float> forward(const std::vector<float>& input, uint16_t seqLen);
    std::vector<float> backward(const std::vector<float>& gradOutput, uint16_t seqLen);
    bool connect(int timeoutMs = 5000);
    bool disconnect();
    bool isConnected() const { return state == ConnectionState::CONNECTED; }
    
private:
    std::string interfaceName;
    uint8_t localMAC[6];
    uint8_t serverMAC[6];
    int bpfFd;
    ConnectionState state = ConnectionState::DISCONNECTED;
    std::vector<uint8_t> bpfBuffer;
    uint32_t clientId = 0x87654321;
    uint32_t serverId = 0;
    uint16_t sequenceNum = 0;
    HandshakeReq myConfig = {};
    LayerConfig layerCfg = {};
    
    bool sendFrame(const DTXHeader& hdr, const uint8_t* payload);
    bool receiveFrame(EthernetFrame& frame, int timeoutMs);
    uint16_t getNextSeq() { return ++sequenceNum; }
};

bool TransformerClient::initialize(const uint8_t* srvMAC) {
    if (!getMACAddress(interfaceName, localMAC)) return false;
    memcpy(serverMAC, srvMAC, 6);
    bpfFd = openBPF(interfaceName);
    if (bpfFd < 0) return false;
    
    char localStr[18], serverStr[18];
    macToString(localMAC, localStr, sizeof(localStr));
    macToString(serverMAC, serverStr, sizeof(serverStr));
    std::cout << "[Client] Initialized on " << interfaceName << " (local: " << localStr << ", server: " << serverStr << ")" << std::endl;
    return true;
}

void TransformerClient::setConfig(uint16_t seqLen, uint16_t embedDim, uint32_t ffnDim, uint8_t numHeads, uint8_t numKVHeads) {
    myConfig = {clientId, seqLen, embedDim, ffnDim, numHeads, numKVHeads};
}

void TransformerClient::setLayerConfig(uint8_t startLayer, uint8_t numLayers, bool keepActivations) {
    layerCfg = {startLayer, numLayers, (uint8_t)(keepActivations ? 1 : 0), 0, 0};
}

bool TransformerClient::connect(int timeoutMs) {
    DTXHeader hdr = makeHeader(MessageType::HANDSHAKE_REQ, getNextSeq(), (const uint8_t*)&myConfig, sizeof(myConfig));
    if (!sendFrame(hdr, (const uint8_t*)&myConfig)) return false;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    while (std::chrono::high_resolution_clock::now() - startTime < std::chrono::milliseconds(timeoutMs)) {
        EthernetFrame frame;
        if (!receiveFrame(frame, 500)) continue;
        if (frame.payload.size() < sizeof(DTXHeader)) continue;
        
        DTXHeader respHdr;
        memcpy(&respHdr, frame.payload.data(), sizeof(DTXHeader));
        if (respHdr.msgType == static_cast<uint8_t>(MessageType::HANDSHAKE_ACK)) {
            if (frame.payload.size() >= sizeof(DTXHeader) + sizeof(HandshakeAck)) {
                HandshakeAck ack;
                memcpy(&ack, &frame.payload[sizeof(DTXHeader)], sizeof(HandshakeAck));
                serverId = ack.serverId;
                state = ConnectionState::CONNECTED;
                std::cout << "[Client] Connected to server" << std::endl;
                return true;
            }
        }
    }
    return false;
}

bool TransformerClient::disconnect() {
    DTXHeader hdr = makeHeader(MessageType::DISCONNECT, getNextSeq(), nullptr, 0);
    sendFrame(hdr, nullptr);
    state = ConnectionState::DISCONNECTED;
    return true;
}

std::vector<float> TransformerClient::forward(const std::vector<float>& input, uint16_t seqLen) {
    if (state != ConnectionState::CONNECTED) return {};
    
    DTXHeader startHdr = makeHeader(MessageType::FORWARD_START, getNextSeq(), nullptr, 0);
    sendFrame(startHdr, nullptr);
    
    size_t elementsPerChunk = (DTX_MAX_PAYLOAD - sizeof(ForwardChunk)) / sizeof(float);
    uint32_t chunkId = 0;
    for (size_t offset = 0; offset < input.size(); offset += elementsPerChunk) {
        size_t chunkSize = std::min(elementsPerChunk, input.size() - offset);
        ForwardChunk chunk = {chunkId++, 0, seqLen, myConfig.embedDim, (uint32_t)(chunkSize * sizeof(float))};
        std::vector<uint8_t> payload(sizeof(ForwardChunk) + chunk.dataSize);
        memcpy(payload.data(), &chunk, sizeof(ForwardChunk));
        memcpy(&payload[sizeof(ForwardChunk)], &input[offset], chunk.dataSize);
        DTXHeader chunkHdr = makeHeader(MessageType::FORWARD_CHUNK, getNextSeq(), payload.data(), payload.size());
        sendFrame(chunkHdr, payload.data());
    }
    
    DTXHeader doneHdr = makeHeader(MessageType::FORWARD_DONE, getNextSeq(), nullptr, 0);
    sendFrame(doneHdr, nullptr);
    
    std::vector<float> result;
    auto startTime = std::chrono::high_resolution_clock::now();
    while (std::chrono::high_resolution_clock::now() - startTime < std::chrono::milliseconds(DTX_FRAME_TIMEOUT)) {
        EthernetFrame frame;
        if (!receiveFrame(frame, 500)) continue;
        if (frame.payload.size() < sizeof(DTXHeader)) continue;
        
        DTXHeader hdr;
        memcpy(&hdr, frame.payload.data(), sizeof(DTXHeader));
        if (hdr.msgType == static_cast<uint8_t>(MessageType::FORWARD_RESULT)) {
            ForwardResult res;
            memcpy(&res, &frame.payload[sizeof(DTXHeader)], sizeof(ForwardResult));
            const float* data = (const float*)&frame.payload[sizeof(DTXHeader) + sizeof(ForwardResult)];
            result.insert(result.end(), data, data + res.dataSize / sizeof(float));
        } else if (hdr.msgType == static_cast<uint8_t>(MessageType::FORWARD_COMPLETE)) {
            break;
        }
    }
    return result;
}

std::vector<float> TransformerClient::backward(const std::vector<float>& gradOutput, uint16_t seqLen) {
    if (state != ConnectionState::CONNECTED) return {};
    // Similar to forward, omitted for brevity - same pattern
    return gradOutput;
}

bool TransformerClient::sendFrame(const DTXHeader& hdr, const uint8_t* payload) {
    std::vector<uint8_t> framePayload(sizeof(DTXHeader) + hdr.payloadLen);
    memcpy(framePayload.data(), &hdr, sizeof(DTXHeader));
    if (payload && hdr.payloadLen > 0) memcpy(&framePayload[sizeof(DTXHeader)], payload, hdr.payloadLen);
    return sendBPFFrame(bpfFd, serverMAC, localMAC, framePayload);
}

bool TransformerClient::receiveFrame(EthernetFrame& frame, int timeoutMs) {
    return receiveBPFFrame(bpfFd, frame, timeoutMs, bpfBuffer);
}

// ================================================================================
// DISTRIBUTED TRANSFORMER
// ================================================================================

struct DistributedConfig {
    int seqLen = 512, embedDim = 768, ffnDim = 3072, numHeads = 12, numKVHeads = 12, totalLayers = 12;
    int localLayers = 6, remoteLayers = 6, startRemoteLayer = 6;
    bool cacheActivations = true, cacheGradients = true;
    std::string interfaceName = "en0";
    uint8_t serverMAC[6] = {0};
    bool validate() const {
        return (localLayers + remoteLayers) == totalLayers && startRemoteLayer >= 0 && startRemoteLayer + remoteLayers == totalLayers;
    }
};

class DistributedTransformer {
public:
    explicit DistributedTransformer(const DistributedConfig& cfg) : config(cfg) { activationCache.resize(config.totalLayers); }
    ~DistributedTransformer() { if (client && client->isConnected()) client->disconnect(); }
    
    bool initialize();
    bool connect(int timeoutMs = 5000);
    bool disconnect();
    std::vector<float> forward(const std::vector<float>& input);
    std::vector<float> backward(const std::vector<float>& gradOutput);
    bool isConnected() const { return client && client->isConnected(); }
    const DistributedConfig& getConfig() const { return config; }
    
private:
    DistributedConfig config;
    std::unique_ptr<TransformerClient> client;
    std::vector<std::vector<float>> activationCache;
};

bool DistributedTransformer::initialize() {
    if (!config.validate()) { std::cerr << "Invalid configuration" << std::endl; return false; }
    client.reset(new TransformerClient(config.interfaceName));
    if (!client->initialize(config.serverMAC)) return false;
    client->setConfig(config.seqLen, config.embedDim, config.ffnDim, config.numHeads, config.numKVHeads);
    client->setLayerConfig(config.startRemoteLayer, config.remoteLayers, config.cacheActivations);
    return true;
}

bool DistributedTransformer::connect(int timeoutMs) { return client->connect(timeoutMs); }
bool DistributedTransformer::disconnect() { return client ? client->disconnect() : true; }

std::vector<float> DistributedTransformer::forward(const std::vector<float>& input) {
    if (!isConnected()) return {};
    return client->forward(input, config.seqLen);
}

std::vector<float> DistributedTransformer::backward(const std::vector<float>& gradOutput) {
    if (!isConnected()) return {};
    return client->backward(gradOutput, config.seqLen);
}

class DistributedTransformerServer {
public:
    explicit DistributedTransformerServer(const DistributedConfig& cfg) : config(cfg) {}
    
    bool initialize();
    void run(int maxMessages = -1);
    
    using LayerFunction = std::function<std::vector<float>(const std::vector<float>&, int, bool)>;
    void setForwardLayerFunction(LayerFunction fn) { forwardLayerFn = fn; }
    
private:
    DistributedConfig config;
    std::unique_ptr<TransformerServer> server;
    LayerFunction forwardLayerFn;
};

bool DistributedTransformerServer::initialize() {
    if (!config.validate()) return false;
    server.reset(new TransformerServer(config.interfaceName));
    if (!server->initialize()) return false;
    
    server->setForwardCallback([this](const std::vector<float>& input, uint16_t, uint8_t, uint8_t) {
        if (forwardLayerFn) return forwardLayerFn(input, 0, true);
        return input;
    });
    return true;
}

void DistributedTransformerServer::run(int maxMessages) { server->run(maxMessages); }

} // namespace DistTransformer

// ================================================================================
// MAIN
// ================================================================================

void printHelp(const char* progName) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Distributed Transformer - Layer 2 Ethernet + Metal            ║" << std::endl;
    std::cout << "║     macOS Native GPU Compute (Metal) + BPF Networking             ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nUSAGE: " << progName << " <command> [options]\n" << std::endl;
    std::cout << "COMMANDS:\n" << std::endl;
    std::cout << "  server -i <interface>     Start as Transformer server" << std::endl;
    std::cout << "  client -i <if> -s <mac>   Start as Transformer client" << std::endl;
    std::cout << "  test                      Run unit tests" << std::endl;
    std::cout << "\nNOTE: Requires root privileges for BPF access.\n" << std::endl;
}

int main(int argc, char* argv[]) {
    @autoreleasepool {
        if (argc < 2) { printHelp(argv[0]); return 1; }
        
        std::string command = argv[1];
        
        if (command == "--version") {
            std::cout << "Distributed Transformer v1.0.0 (Metal)" << std::endl;
            std::cout << "macOS Layer 2 Ethernet + Metal GPU" << std::endl;
            std::cout << "Copyright (c) 2025 Matthew Abbott - MIT License" << std::endl;
            return 0;
        }
        
        if (command == "server") {
            DistTransformer::DistributedConfig cfg;
            cfg.totalLayers = 12; cfg.localLayers = 0; cfg.remoteLayers = 12; cfg.startRemoteLayer = 0;
            cfg.interfaceName = "en0";
            int maxMessages = 100;
            
            for (int i = 2; i < argc; i++) {
                std::string arg = argv[i];
                if ((arg == "-i" || arg == "--interface") && i + 1 < argc) cfg.interfaceName = argv[++i];
                else if ((arg == "-l" || arg == "--layers") && i + 1 < argc) {
                    cfg.totalLayers = std::stoi(argv[++i]); cfg.remoteLayers = cfg.totalLayers;
                }
                else if ((arg == "-m" || arg == "--messages") && i + 1 < argc) maxMessages = std::stoi(argv[++i]);
            }
            
            std::cout << "\n=== Metal Server Configuration ===" << std::endl;
            std::cout << "Interface: " << cfg.interfaceName << std::endl;
            std::cout << "Total Layers: " << cfg.totalLayers << std::endl;
            
            MetalCompute metal;
            if (!metal.initialize()) {
                std::cerr << "Warning: Metal not available, using CPU fallback" << std::endl;
            }
            
            DistTransformer::DistributedTransformerServer server(cfg);
            if (!server.initialize()) { std::cerr << "Failed to initialize server" << std::endl; return 1; }
            
            server.setForwardLayerFunction([](const std::vector<float>& input, int, bool) { return input; });
            
            std::cout << "Server ready. Processing messages...\n" << std::endl;
            server.run(maxMessages);
            return 0;
            
        } else if (command == "client") {
            DistTransformer::DistributedConfig cfg;
            cfg.interfaceName = "en0";
            bool serverMACProvided = false;
            
            for (int i = 2; i < argc; i++) {
                std::string arg = argv[i];
                if ((arg == "-i" || arg == "--interface") && i + 1 < argc) cfg.interfaceName = argv[++i];
                else if ((arg == "-s" || arg == "--server") && i + 1 < argc) {
                    if (!DistTransformer::stringToMAC(argv[++i], cfg.serverMAC)) {
                        std::cerr << "Invalid MAC format" << std::endl; return 1;
                    }
                    serverMACProvided = true;
                }
            }
            
            if (!serverMACProvided) { std::cerr << "Server MAC required (-s)" << std::endl; return 1; }
            
            DistTransformer::DistributedTransformer client(cfg);
            if (!client.initialize()) return 1;
            if (!client.connect(5000)) { std::cerr << "Connection failed" << std::endl; return 1; }
            
            std::vector<float> input(cfg.embedDim, 1.0f);
            auto output = client.forward(input);
            std::cout << "Forward: in=" << input.size() << " out=" << output.size() << std::endl;
            
            client.disconnect();
            return 0;
            
        } else if (command == "test") {
            std::cout << "\n=== Running Tests ===" << std::endl;
            
            std::cout << "Test 1: Metal initialization" << std::endl;
            MetalCompute metal;
            if (metal.initialize()) std::cout << "  Metal OK" << std::endl;
            else std::cout << "  Metal failed (may need .metallib)" << std::endl;
            
            std::cout << "Test 2: Protocol verification" << std::endl;
            auto hdr = DistTransformer::makeHeader(DistTransformer::MessageType::HANDSHAKE_REQ, 1, nullptr, 0);
            std::cout << "  Header: " << (DistTransformer::verifyHeader(hdr) ? "OK" : "FAIL") << std::endl;
            
            std::cout << "Test 3: MAC parsing" << std::endl;
            uint8_t mac[6];
            std::cout << "  Parse: " << (DistTransformer::stringToMAC("AA:BB:CC:DD:EE:FF", mac) ? "OK" : "FAIL") << std::endl;
            
            std::cout << "====================\n" << std::endl;
            return 0;
        }
        
        printHelp(argv[0]);
        return 1;
    }
}
