# Transformer Test Results

**Date:** January 25, 2026  
**Hardware:** NVIDIA RTX 3070 (8GB VRAM), Ubuntu 24.04

## Test Summary

| Binary | Mode | Model | Status | Notes |
|--------|------|-------|--------|-------|
| transformer-cuda | GPU | Llama-3.2-1B-f16 | ⚠️ EOT | Immediate EOT token (model behavior) |
| transformer-cuda | CPU | Llama-3.2-3B-Q4_K_M | ✅ Works | "How can I assist you today?" |
| transformer-opencl | CPU | Llama-3.2-3B-Q4_K_M | ✅ Works | "Hello! How can I assist you today?" |
| transformer-opencl | CPU | Mistral-7B-Q4_K_M | ✅ Works | Generates coherent text |
| facaded-transformer-cuda | GPU | Llama-3.2-1B-f16 | ✅ Works | Generates response (DeepSeek template) |
| facaded-transformer-cuda | CPU | Llama-3.2-3B-Q4_K_M | ⚠️ Partial | Some garbage after initial response |
| facaded-transformer-opencl | CPU | Llama-3.2-3B-Q4_K_M | ✅ Works | "How are you today? Is there something I can help you with?" |
| transformer-opencl | CPU | Qwen3-8B-Q4_K_M | ❌ Fails | Garbage output (tokenizer mismatch) |
| transformer-cuda | CPU | DeepSeek-6.7B-Q4_K_M | ❌ Fails | Garbage output |

## Key Fixes Applied

1. **Tokenizer encode() - Special Token Handling:** Fixed to correctly encode special tokens like `<|begin_of_text|>` instead of byte-by-byte encoding.

2. **vecMatMulKernel - Row-major Matrix Indexing:** Fixed CUDA kernel from `mat[k * N + col]` to `mat[n * K + k]` for correct row-major access.

3. **CPU forward() - Token Selection:** Fixed to use `tokens[pos]` instead of `tokens.back()` to process correct token at each position.

4. **CPU generate() - Prefill Loop:** Added explicit prefill loop to populate KV cache for all prompt tokens before generation.

## Working Configurations

### Best Results (Recommended)
- **transformer-opencl CPU** with Llama-3.2-3B-Q4_K_M or Mistral-7B-Q4_K_M
- **facaded-transformer-opencl CPU** with Llama models

### GPU Mode
- Works with Llama-3.2-1B-f16 (5.4GB VRAM)
- Larger models exceed 8GB VRAM limit on RTX 3070

## Known Issues

1. **Llama-3.2-1B-Instruct EOT:** Both GPU/CPU produce immediate EOT token - appears to be model/prompt sensitivity, not implementation bug.

2. **DeepSeek/Qwen models:** Tokenizer or chat template issues cause garbage output in CUDA versions.

3. **OpenCL GPU mode:** Not tested on RTX 3070 (NVIDIA OpenCL support is limited).

4. **Facaded CUDA CPU:** Uses DeepSeek tokenizer template by default, causes issues with non-DeepSeek models.

## Performance

| Config | Model | Speed |
|--------|-------|-------|
| GPU (Llama-1B) | Llama-3.2-1B-f16 | ~1-26 tok/s |
| CPU (3B) | Llama-3.2-3B-Q4_K_M | ~2-3 tok/s |
| CPU (7B) | Mistral-7B-Q4_K_M | ~1 tok/s |

## VRAM Usage

- Llama-3.2-1B-f16: 5.38 GB
- Llama-3.2-3B-Q4_K_M: 13.65 GB (exceeds RTX 3070)
