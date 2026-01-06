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

- **Direct GGUF Model Loading:**  Loads GGUF weights/tensors "by hand" for full transparency.
- **CUDA-Accelerated Inference:**  Matrix ops, quantization, attention, softmax and FFN all implemented in device code.
- **Universal Quantized Model Support:**  Full support for all K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K), plus legacy formats, with modular code to extend to future quantization types (K_M, K_L, K_S).
- **Agentic Inspection & Manipulation:**  Exposes all activations, weights, inputs/outputs by layer and token for advanced debugging, visualization, or agentic control.
- **Interactive CLI and GUI:**  Chat, batch inference, benchmarking, quant stats, and file management.
- **Custom Tokenizer Support:**  Loads GPT-2 compatible vocabularies.
- **MIT License.**  All source is open for modification and commercial/academic/educational use.

---

## Requirements

- NVIDIA GPU with CUDA (compute 6.0+ recommended)
- CUDA toolkit (tested with CUDA 11-12+)
- C++14 (main) or C++17 for advanced facade features
- Python 3.6+ for GUI frontend
- GGUF model weights (float32, float16, bfloat16, or quantized)
- **No external deep learning library required**

---

## agentic_transformer.cu (Agentic Universal Transformer)

### Design

A transparent, introspectable GGUF transformer core with agentic hooks, designed for universal quantized inference and fine-grained layer/device inspection.

- **Full Quantization Registry:**  Implements all llama.cpp-style K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K) with accurate block layout and dequantization, plus legacy quant formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0).
- **Device Offloading:**  Per-layer configurable GPU/CPU device assignment so you can selectively offload layers for memory/concurrency benchmarking.
- **Layer, Token & Activation Inspection:**  Exposes all internal states—hidden states, attention weights/logits, QKV, residuals, FFN output, normalization—at every step.
- **Open Kernel Design:**  All math (RoPE, matmul, GELU, RMSNorm, etc.) is visible for modification and debugging.
- **MIT License**: Fully open for commercial/research/educational extension.

#### Quantization Support

- Complete and extensible support for K-quants (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`).
- Legacy/compat formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`) for older GGUF models.
- Modular dequantization dispatch—ready to add new quant types (like K_M, K_L, K_S) as they appear; just match the tensor metadata and plug in your block logic.

#### Device Offloading

- Flexible `LayerDeviceConfig`: Assign GPU or CPU to each transformer layer, for performance profiling or hybrid runs.
- Unified dispatch ensures correct routing of all matmul, normalization, and quant kernels per device.

#### Usage & API

Build:

```bash
nvcc -O3 -std=c++14 agentic_transformer.cu -o agentic_transformer
```

Basic command-line interface:

```bash
./agentic_transformer --model mymodel.gguf --tokenizer tokenizer.json --prompt "Hello world" --max-tokens 32
```

Inspect/override per-layer device (example in code comments):

```bash
./agentic_transformer --model model.gguf --cpu-layers 0,1,2    # Offload first 3 layers to CPU
```

Access all activations using the exposed API, or connect to the Python CLI for advanced inspection.

---

## transformer_gui.py (Interactive CLI GUI)

### Features

- Terminal-based GUI for running and controlling CUDA transformer inference.
- Supports all common model loading, generation, file management, benchmarking, and quant/stat commands.
- Color-coded status, help, and error messages.
- Conversation history and log saving for chat sessions.
- Direct compilation/management of CUDA source files.
- Model inspection: tensor listing, quantization statistics, benchmarking.

### Usage

Run:

```bash
python3 transformer_gui.py
```

Typical workflow:
- Compile CUDA source (`compile`)
- Load model (`load <model.gguf> [tokenizer.json]`)
- Chat (`chat <prompt>`)
- Inspect model info (`info`, `tensors`, `quant`, `benchmark`)
- Manage files, settings, and logs interactively

See `help` in the GUI for all commands.

---

## transformer.cu (Standard Transformer)

*(Retained for ease of development, basic testing, and clarity—see original README sections for details.)*

---

## facaded_transformer.cu (Transformer Facade)

*(Full detailed inspection and scientific visualization of all model states during inference—see original README sections for details.)*

---

## Data Structures & Format

- **GGUFTensor / GGUFLoader:** Direct raw access to all GGUF weight tensors and configuration.
- **Custom Tokenizer:** Loads and encodes/decodes vocab from JSON (GPT-2 format), or direct mapping.
- All buffer designs support CPU-side extraction for debugging.
- All quantization blocks and dequant logics commented and reference-accurate.

---

## Overview & Notes

- No icons, branding, or product logos.
- All code is *deliberately hackable*—adapt for teaching, debugging, agentic interfaces, or plugin development.
- Completely independent CUDA/C++—no third-party ML frameworks.
- Designed for direct integration into larger agentic AI stacks, local RAG pipelines, or custom GGUF model workflows.

---

## License

MIT License Copyright © 2025 Matthew Abbott

> See LICENSE / header in each source file.
