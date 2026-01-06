# GlassBoxAI-Transformer

**Author:** Matthew Abbott (2025)

A modern, fully-transparent CUDA implementation of Transformer language models—targeting *maximum hackability and educational clarity*.  
This project provides modular GPU-accelerated transformer inference, agentic inspection, and interactive CLI/GUI, with direct GGUF model support.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [agentic_transformer.cu (Agentic Universal Transformer)](#agentic_transformercu-agentic-universal-transformer)
  - [Design](#design-agentic)
  - [Quantization Support](#quantization-support)
  - [Device Offloading](#device-offloading)
  - [Usage & API](#usage-api-agentic)
- [transformer_gui.py (Interactive CLI GUI)](#transformer_guipy-interactive-cli-gui)
  - [Features](#features-gui)
  - [Usage](#usage-gui)
- [transformer.cu (Standard Transformer)](#transformercu-standard-transformer)
  - [Design](#design)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [Public Methods](#public-methods)
- [facaded_transformer.cu (Transformer Facade)](#facaded_transformercu-transformer-facade)
  - [Design](#facade-design)
  - [Usage](#facade-usage)
  - [Arguments](#facade-arguments)
  - [Public Methods](#facade-methods)
- [Data Structures & Format](#data-structures--format)
- [Overview & Notes](#overview--notes)
- [License](#license)

---

## Features

- **Direct GGUF Model Loading:** No external deps—loads GGUF weights/tensors "by hand".
- **CUDA-Accelerated Transformer Inference:** All matrix, normalization, QKV, attention, softmax, and FFN logic in CUDA.
- **Universal Quantized Model Support:** Full support for all K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K), plus legacy formats and extensible design for future types (K_M, K_L, K_S).
- **Agentic Inspection & Manipulation:** Exposes all activations, weights, inputs/outputs by layer and token for advanced debugging, visualization, or agentic control.
- **Interactive CLI and GUI:** Chat, batch inference, benchmarking, quant stats, and file management via both CLI and GUI.
- **Custom Tokenizer Support:** Loads vocab from GPT-2 compatible tokenizers.
- **GPU-efficient memory management, sequence batching, and buffer allocation.**
- **Temperature sampling, max-logit, and token generation in CLI.**
- **Stepwise/debug-inspect attention, FFN, residuals, and all embeddings.**
- **No icons or logos—just source and docs.**
- **License:** MIT

---

## Requirements

- NVIDIA GPU with CUDA (compute 6.0+ highly recommended)
- CUDA toolkit (tested with CUDA 11-12+)
- C++14 (main) or C++17 (facade/agentic) for optional features
- Python 3.6+ for GUI frontend
- [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) model weights (convert from HuggingFace etc. as needed)
- **No external deep learning library required**

---

## agentic_transformer.cu (Agentic Universal Transformer)

### Design

A transparent, introspectable GGUF transformer core with agentic hooks, designed for universal quantized inference and fine-grained layer/device inspection.

- **Full Quantization Registry:** Implements all llama.cpp-style K-quants (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`) with accurate block layout and dequantization, plus legacy quant formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`).
- **Device Offloading:** Per-layer configurable GPU/CPU device assignment lets you offload layers for memory/concurrency benchmarking.
- **Layer, Token & Activation Inspection:** Exposes all internal states—hidden states, attention weights/logits, QKV, residuals, FFN output, normalization—at every step.
- **Open Kernel Design:** All math (RoPE, matmul, GELU, RMSNorm, etc.) is visible for modification and debugging.
- **Compact Design:** Complete transformer, quants, and inspection all in a single CUDA file under 1MB.
- **MIT License.**

#### Quantization Support

- Complete and extensible support for K-quants (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`).
- Legacy/compat formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`) for older GGUF models.
- Modular dequantization dispatch—ready to add new quant types (like K_M, K_L, K_S) as they appear; match tensor metadata and add block logic.

#### Device Offloading

- Flexible `LayerDeviceConfig`: Assign GPU or CPU to each transformer layer for profiling or hybrid runs.
- Unified dispatch ensures correct routing of all matmul, normalization, and quant kernels per device.

#### Usage & API

Build:

```bash
nvcc -O3 -std=c++14 agentic_transformer.cu -o agentic_transformer
```

Run:

```bash
./agentic_transformer --model mymodel.gguf --tokenizer tokenizer.json --prompt "Hello world" --max-tokens 32
```

Inspect/override per-layer device (see code comments):

```bash
./agentic_transformer --model model.gguf --cpu-layers 0,1,2   # Offload first 3 layers to CPU
```

Access all activations via API or connect to the Python GUI for inspection.

---

## transformer_gui.py (Interactive CLI GUI)

### Features

- Terminal-based GUI for running and controlling CUDA transformer inference.
- Supports all model loading, generation, file ops, benchmarking, quant/stat commands.
- Color-coded status and help.
- Conversation history, log saving.
- Compilation management.
- Model inspection: list tensors, quant stats, benchmarking.

### Usage

```bash
python3 transformer_gui.py
```

Workflow:
- Compile CUDA (`compile`)
- Load model (`load <model.gguf> [tokenizer.json]`)
- Chat (`chat <prompt>`)
- Inspect (`info`, `tensors`, `quant`, `benchmark`)
- Manage files, settings, logs.  
Type `help` for all commands.

---

## transformer.cu (Standard Transformer)

### Design

Implements a full Transformer LLM from first principles for GPU in a single file:

- Loads GGUF file, locates all layers and tensors.
- Loads GPT-2 vocab/tokenizer.
- Allocates/manages GPU tensors.
- Embedding, attention, residuals, layer norm, FFN (GELU), logits all in custom kernels.
- CLI for model inference ("generation").

### Usage

```bash
nvcc -O3 -std=c++14 transformer.cu -o transformer_cuda
```

Example:

```bash
./transformer_cuda --model mymodel.gguf --tokenizer tokenizer.json --prompt "The quick brown fox" --max-tokens 64 --temperature 1.2
```

### Arguments

- `--model`      Path to GGUF transformer checkpoint
- `--tokenizer`  Path to tokenizer JSON (GPT-2 vocab format)
- `--prompt`     Text to be tokenized/generate
- `--max-tokens` Maximum tokens to generate
- `--temperature` Sampling temperature (default 1.0)
- More options for batch/generation in code comments.

### Public Methods

**Class: `TransformerModel`**
- `bool loadModel(const std::string& path)`
- `bool loadTokenizer(const std::string& path)`
- `std::string generate(const std::string& prompt, int maxTokens, double temperature = 1.0)`
  - Tokenizes, runs forward/inference, samples next token(s)
- `std::vector<float> forward(const std::vector<int>& tokenIDs)`
  - Forward pass, returns logits

#### GGUF/Tokenizer Access

- GPU/CPU tensor accessor for all parameters (`GGUFLoader`)
- Tokenizer: `encode(text)` and `decode(token_ids)`

---

## facaded_transformer.cu (Transformer Facade)

### Design

A C++/CUDA introspection tool for the transformer. The facade gives:

- **Detailed per-layer access**: all activations, attention logits/weights, Q/K/V vectors, per-head output, layer norms, residuals, etc.
- **Inspection, visualization, manipulation ready**: Export tensors as CPU-side arrays for plotting, analysis.
- **Designed for research, teaching, and tinkering.**
- **Pythonic API-style maximal transparency.**

### Usage

```bash
nvcc -O3 -std=c++17 facaded_transformer.cu -o facadedtransformer_cuda -lcublas
```

Pattern:
- Load GGUF, tokenizer.
- Use `forward(tokenIds)` for prompt inference.
- Access all states/weights per step/layer.

### Arguments

- `bool loadModel(path)`
- `bool loadTokenizer(path)`
- `DoubleArray forward(IntArray)`
- `std::string generate(prompt, maxTokens, temperature)`
- Introspection functions:  
  - Per-layer Q/K/V, attention weights/logits, hidden/residuals, layer norm, FFN.
  - Final logits, output layers.
- `GGUFLoader` and `Tokenizer` for low-level hacking
- Model loader: `getTensor("name")`, `printAllTensorNames()`, config getters.

### Public Methods

**Class: `TransformerModel` (Facade)**
- `bool loadModel(const std::string& path)`
- `bool loadTokenizer(const std::string& path)`
- `DoubleArray forward(const IntArray& tokenIds)`
- `std::string generate(const std::string& prompt, int maxTokens, double temperature = 1.0)`
- Introspection:
  - `getLastHiddenStates()`, `getLastAttentionWeights()`, `getLastQVectors()`, `getLastKVectors()`, `getLastLayerNormOutputs()`, `getLastFFNOutputs()`, `getLastLogits()`, etc.

---

## Data Structures & Format

- **GGUFTensor / GGUFLoader:** Loads, parses, and gives direct access to all tensors.
- **Tokenizer:** GPT-2 compatible JSON vocab, encode/decode.
- **CPU/GPU buffer design:** Can extract CPU tensors at any time for debugging/display.
- **Attention:** All head-level/intermediate states accessible after forward pass.

---

## Overview & Notes

- **No icons, branding, or product logos.** Docs and source only.
- All code is *deliberately hackable*: change, inspect, checkpoint any parameter/state.
- **No third-party ML frameworks**—entirely self-contained CUDA/C++.

---

## License

MIT License © 2025 Matthew Abbott

See LICENSE or header in source files for full terms.
