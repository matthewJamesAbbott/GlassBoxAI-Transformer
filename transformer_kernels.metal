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

#include <metal_stdlib>
using namespace metal;

// Matrix multiplication: C = A * B + bias
// A: [M x K], B: [K x N], C: [M x N], bias: [N] (optional)
kernel void matmul_fp32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    constant int& hasBias [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.y;
    int j = gid.x;
    
    if (i >= M || j >= N) return;
    
    float sum = (hasBias != 0) ? bias[j] : 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

// Tiled matrix multiplication for better performance
// Uses threadgroup shared memory
kernel void matmul_tiled_fp32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    constant int& hasBias [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const int TILE_SIZE = 16;
    
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = tgid.y * TILE_SIZE + tid.y;
    int col = tgid.x * TILE_SIZE + tid.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + tid.x;
        int bRow = t * TILE_SIZE + tid.y;
        
        if (row < M && aCol < K) {
            tileA[tid.y][tid.x] = A[row * K + aCol];
        } else {
            tileA[tid.y][tid.x] = 0.0f;
        }
        
        if (bRow < K && col < N) {
            tileB[tid.y][tid.x] = B[bRow * N + col];
        } else {
            tileB[tid.y][tid.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        if (hasBias != 0) {
            sum += bias[col];
        }
        C[row * N + col] = sum;
    }
}

// GELU activation function
// gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_fp32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    
    float x = input[gid];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[gid] = x * cdf;
}

// SiLU/Swish activation: x * sigmoid(x)
kernel void silu_fp32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    
    float x = input[gid];
    output[gid] = x / (1.0f + exp(-x));
}

// ReLU activation
kernel void relu_fp32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    output[gid] = max(0.0f, input[gid]);
}

// Softmax - single row processing
// This kernel processes one row per threadgroup
kernel void softmax_fp32(
    device float* data [[buffer(0)]],
    constant int& rows [[buffer(1)]],
    constant int& cols [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(rows)) return;
    
    device float* rowData = data + gid * cols;
    
    // Find max for numerical stability
    float maxVal = rowData[0];
    for (int i = 1; i < cols; i++) {
        maxVal = max(maxVal, rowData[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        rowData[i] = exp(rowData[i] - maxVal);
        sum += rowData[i];
    }
    
    // Normalize
    float invSum = 1.0f / sum;
    for (int i = 0; i < cols; i++) {
        rowData[i] *= invSum;
    }
}

// Parallel softmax using threadgroup for larger sequences
// One threadgroup per row, 1D dispatch
kernel void softmax_parallel_fp32(
    device float* data [[buffer(0)]],
    constant int& rows [[buffer(1)]],
    constant int& cols [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    int row = gid;
    if (row >= rows) return;
    
    device float* rowData = data + row * cols;
    
    threadgroup float sharedMax[256];
    threadgroup float sharedSum[256];
    
    // Find local max
    float localMax = -INFINITY;
    for (uint i = tid; i < uint(cols); i += tgSize) {
        localMax = max(localMax, rowData[i]);
    }
    sharedMax[tid] = localMax;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to find global max
    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float maxVal = sharedMax[0];
    
    // Compute local exp sum
    float localSum = 0.0f;
    for (uint i = tid; i < uint(cols); i += tgSize) {
        float expVal = exp(rowData[i] - maxVal);
        rowData[i] = expVal;
        localSum += expVal;
    }
    sharedSum[tid] = localSum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to find global sum
    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float invSum = 1.0f / sharedSum[0];
    
    // Normalize
    for (uint i = tid; i < uint(cols); i += tgSize) {
        rowData[i] *= invSum;
    }
}

// Layer normalization
kernel void layernorm_fp32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant int& batchSize [[buffer(4)]],
    constant int& hiddenSize [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(batchSize)) return;
    
    device const float* row = input + gid * hiddenSize;
    device float* outRow = output + gid * hiddenSize;
    
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < hiddenSize; i++) {
        mean += row[i];
    }
    mean /= float(hiddenSize);
    
    // Compute variance
    float variance = 0.0f;
    for (int i = 0; i < hiddenSize; i++) {
        float diff = row[i] - mean;
        variance += diff * diff;
    }
    variance /= float(hiddenSize);
    
    // Normalize
    float invStd = rsqrt(variance + eps);
    for (int i = 0; i < hiddenSize; i++) {
        outRow[i] = (row[i] - mean) * invStd * gamma[i] + beta[i];
    }
}

// RMS normalization (used in LLaMA-style models)
kernel void rmsnorm_fp32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant int& batchSize [[buffer(3)]],
    constant int& hiddenSize [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(batchSize)) return;
    
    device const float* row = input + gid * hiddenSize;
    device float* outRow = output + gid * hiddenSize;
    
    // Compute sum of squares
    float sumSq = 0.0f;
    for (int i = 0; i < hiddenSize; i++) {
        sumSq += row[i] * row[i];
    }
    
    float rms = rsqrt(sumSq / float(hiddenSize) + eps);
    
    for (int i = 0; i < hiddenSize; i++) {
        outRow[i] = row[i] * rms * weight[i];
    }
}

// RoPE (Rotary Position Embedding)
kernel void rope_fp32(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant int& seqLen [[buffer(2)]],
    constant int& numHeads [[buffer(3)]],
    constant int& headDim [[buffer(4)]],
    constant float& ropeBase [[buffer(5)]],
    constant int& posOffset [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    int pos = gid.x;
    int head = gid.y;
    int pair = gid.z;
    
    if (pos >= seqLen || head >= numHeads || pair >= headDim / 2) return;
    
    float freq = 1.0f / pow(ropeBase, float(2 * pair) / float(headDim));
    float angle = float(pos + posOffset) * freq;
    float cosA = cos(angle);
    float sinA = sin(angle);
    
    int idx0 = pos * numHeads * headDim + head * headDim + 2 * pair;
    int idx1 = idx0 + 1;
    
    // Apply to Q
    float q0 = q[idx0];
    float q1 = q[idx1];
    q[idx0] = q0 * cosA - q1 * sinA;
    q[idx1] = q0 * sinA + q1 * cosA;
    
    // Apply to K
    float k0 = k[idx0];
    float k1 = k[idx1];
    k[idx0] = k0 * cosA - k1 * sinA;
    k[idx1] = k0 * sinA + k1 * cosA;
}

// Element-wise add
kernel void add_fp32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    c[gid] = a[gid] + b[gid];
}

// Element-wise multiply
kernel void mul_fp32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    c[gid] = a[gid] * b[gid];
}

// Scale tensor by constant
kernel void scale_fp32(
    device float* data [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    data[gid] *= scale;
}

// Causal attention mask
kernel void apply_causal_mask_fp32(
    device float* attnScores [[buffer(0)]],
    constant int& seqLen [[buffer(1)]],
    constant int& numHeads [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    int q = gid.x;
    int k = gid.y;
    int h = gid.z;
    
    if (q >= seqLen || k >= seqLen || h >= numHeads) return;
    
    if (k > q) {
        int idx = h * seqLen * seqLen + q * seqLen + k;
        attnScores[idx] = -INFINITY;
    }
}

// Embedding lookup
kernel void embedding_lookup_fp32(
    device const float* embeddings [[buffer(0)]],
    device const int* tokenIds [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& seqLen [[buffer(3)]],
    constant int& embedDim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int pos = gid.x;
    int dim = gid.y;
    
    if (pos >= seqLen || dim >= embedDim) return;
    
    int tokenId = tokenIds[pos];
    output[pos * embedDim + dim] = embeddings[tokenId * embedDim + dim];
}

// Copy kernel
kernel void copy_fp32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    dst[gid] = src[gid];
}
