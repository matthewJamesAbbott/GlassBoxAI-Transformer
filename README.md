# GlassBoxAI-Transformer

**Author:** Matthew Abbott (2025)

A modern, fully-transparent CUDA implementation of Transformer language models—targeting *maximum hackability and educational clarity*. This project provides:

- **transformer.cu:** Direct, efficient CUDA implementation of a GPT-2/LLM Transformer, including GGUF model loading and custom tokenizer.
- **facaded_transformer.cu:** A detailed C++17/CUDA inspection and manipulation facade, with tools for deep model analysis and full introspection of all tensors, weights, and internal states.

Both are designed for **GPU-first research, tinkering, and instruction**—not "black box" deployment, but as stepping stones for developing and understanding transformer language models.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [transformer.cu](#transformercu)
  - [Design](#design)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [Public Methods](#public-methods)
- [facaded_transformer.cu (Transformer Facade)](#facaded_transformercu-transformer-facade)
  - [Design](#design-1)
  - [Usage](#usage-1)
  - [Arguments](#arguments-1)
  - [Public Methods](#public-methods-1)
- [Data Structures & Format](#data-structures--format)
- [Overview & Notes](#overview--notes)

---

## Features

- **Direct GGUF Model Loading:** No external deps—loads GGUF weights/tensors "by hand".
- **CUDA-Accelerated Transformer Inference:** All matrix, normalization, QKV, attention, softmax, and FFN biological in CUDA.
- **Custom Tokenizer Support:** Loads vocab from GPT-2 compatible tokenizers.
- **GPU-efficient memory management, sequence batching, and buffer allocation.**
- **Temperature sampling, max-logit, and token generation in CLI.**
- **All tensors and attention available for user inspection.**
- **Facade includes: Stepwise/debug-inspect attention, FFN, residuals, and all embeddings.**
- **No icons or logos—just source and docs.**
- **License:** MIT

---

## Requirements

- NVIDIA GPU with CUDA (compute 6.0+ highly recommended)
- CUDA toolkit (tested with CUDA 11-12+)
- C++14 (main) or C++17 (facade) for optional features
- [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) model weights (convert from HuggingFace etc. with suitable scripts)
- **No external deep learning library required**

---

## transformer.cu

### Design

Implements a full Transformer LLM from first principles for GPU in a single file:

- Loads GGUF file, locates all relevant layers and tensors
- Loads GPT-2 compatible vocab/tokenizer
- Allocates and manages all working tensors on the GPU
- Embedding, multi-head attention, residuals, layer norm, FFN (GELU), logits all in custom GPU kernels
- CLI for model inference ("generation") from a prompt

### Usage

```bash
nvcc -O3 -std=c++14 transformer.cu -o transformer_cuda
```

**Example (CLI):**
```
./transformer_cuda --model mymodel.gguf --tokenizer tokenizer.json --prompt "The quick brown fox" --max-tokens 64 --temperature 1.2
```
Features:
- Can auto-detect GGUF config (embed size, n_layers, n_heads, ff_dim, vocab, etc.)
- Works on single or batch prompts

### Arguments

#### Model/CLI Parameters

- `--model`: Path to `*.gguf` transformer checkpoint
- `--tokenizer`: Path to tokenizer JSON (GPT-2 vocab format)
- `--prompt`: Text to be tokenized and completed/generated
- `--max-tokens`: Maximum tokens to generate
- `--temperature`: Sampling temperature for logits (float, default 1.0)
- More CLI options for batch/generation/inference are available in code comments.

#### Model I/O

- Model weights are expected to be in GGUF format (float32, float16, or bfloat16 supported).
- Tokenizer should match model vocabulary.

### Public Methods

#### Class: `TransformerModel`

- `bool loadModel(const std::string& path)`
- `bool loadTokenizer(const std::string& path)`
- `std::string generate(const std::string& prompt, int maxTokens, double temperature = 1.0)`
  - Does both tokenization, forward/inference, and next-token sampling.
- `std::vector<float> forward(const std::vector<int>& tokenIDs)`
  - Runs forward pass and returns logits of last token for sampling

#### GGUF/Tokenizer Access

- GPU/CPU tensor accessor for all parameters (see `GGUFLoader` in code)
- Custom Tokenizer: `encode(text)` and `decode(token_ids)` methods, with space-aware and fallback logic

---

## facaded_transformer.cu (Transformer Facade)

### Design

A C++/CUDA introspection tool for the transformer. The facade class provides:

- **Detailed per-layer access**: You can extract activations, attention logits/weights, all Q/K/V vectors, per-head results, layer normalizations, etc.
- **Inspection, visualization, and manipulation ready**: All tensors are exported as CPU-side arrays for analysis/plotting.
- **Designed for advanced curriculum, tinkering, and scientific reporting.**
- **Pythonic API and maximal transparency!** No hidden states, all weights and inference steps are hackable.

### Usage

```bash
nvcc -O3 -std=c++17 facaded_transformer.cu -o facadedtransformer_cuda -lcublas
```
(Tested with CUDA 11+, cublas required for some internal matmul)

**Typical pattern:**
- Load GGUF, tokenizer
- Use `forward(tokenIds)` for a forward pass of your prompt
- Access all intermediate states or weights for every step/layer

### Arguments

#### Facade Constructor & Main API

- `bool loadModel(path)`
- `bool loadTokenizer(path)`
- `DoubleArray forward(const IntArray& tokenIds)`
- `std::string generate(prompt, maxTokens, temperature)`
- **Introspection functions:**
  - Per-layer: Q/K/V matrices, attention logits/weights, hidden states/residuals
  - Layer normalization, FFN, and GELU info
  - Final logits, all output layers
- `GGUFLoader` and `Tokenizer` available for low-level hacking
- Model loader provides `getTensor("name")`, `printAllTensorNames()`, and configuration getters for:
  - Embedding size, n_layers, n_heads, FFN size, vocab size, max context

### Public Methods

#### Class: `TransformerModel` (Facade)

- `bool loadModel(const std::string& path)`
- `bool loadTokenizer(const std::string& path)`
- `DoubleArray forward(const IntArray& tokenIds)`
- `std::string generate(const std::string& prompt, int maxTokens, double temperature = 1.0)`
- Introspection:
  - `getLastHiddenStates()`
  - `getLastAttentionWeights()`, `getLastAttentionLogits()`
  - `getLastQVectors()`, `getLastKVectors()`, `getLastVVectors()`
  - `getLastLayerNormOutputs()`, `getLastFFNOutputs()`
  - `getLastResidualInputs()`, `getLastResidualOutputs()`
  - `getLastLogits()`
- Model/weight access via `GGUFLoader`: fully documented in-code

---

## Data Structures & Format

- **GGUFTensor / GGUFLoader**: Loads, parses, and allows direct access to all tensors.
- **Tokenizer**: Parses and encodes/decodes tokens from the JSON vocab format (GPT-2 compatible).
- **CPU/GPU buffer design**: Designed so ALL major tensors and layers can be pulled to CPU at any time for debugging or display.
- **Attention representations**: All head-level and intermediate states accessible after each forward pass.

---

## Overview & Notes

- **No icons, branding, or product logos.** Docs and code only.
- All source is *deliberately hackable*: you can change, inspect, or checkpoint any parameter/state.
- For sequence/token input/output, see class comments and Python-like `encode`/`decode` helpers.
- No third-party deep learning frameworks needed or used—*completely self-contained* CUDA/C++.

---

## License

MIT License, Copyright © 2025 Matthew Abbott
