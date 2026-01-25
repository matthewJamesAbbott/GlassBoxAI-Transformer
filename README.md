# GlassBoxAI-Transformer

## **Large Language Model Inference Suite**

### *GPU-Accelerated Transformer Implementation with Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/opencl/)
[![Metal](https://img.shields.io/badge/Metal-macOS-silver.svg)](https://developer.apple.com/metal/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Kani](https://img.shields.io/badge/Kani-99%20Proofs-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-Transformer is a comprehensive, production-ready Large Language Model (LLM) inference implementation suite featuring:

- **Multiple GPU backends**: CUDA, OpenCL, and Metal (in development) acceleration
- **Multiple language implementations**: C++ and Rust
- **GGUF model format support**: Load quantized models from llama.cpp ecosystem
- **Quantization support**: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 formats with GPU-accelerated dequantization
- **Facade pattern architecture**: Clean API separation with deep introspection capabilities
- **Formal verification**: 99 Kani-verified proof harnesses for memory safety guarantees
- **CISA/NSA Secure by Design compliance**: Built following government cybersecurity standards

This project demonstrates enterprise-grade software engineering practices including comprehensive testing, formal verification, cross-platform compatibility, and security-first development.

---

## **Table of Contents**

1. [Features](#features)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Prerequisites](#prerequisites)
5. [Installation & Compilation](#installation--compilation)
6. [CLI Reference](#cli-reference)
   - [Standard Transformer Commands](#standard-transformer-commands)
   - [Facade Transformer Commands](#facade-transformer-commands)
7. [Testing](#testing)
8. [Formal Verification with Kani](#formal-verification-with-kani)
9. [CISA/NSA Compliance](#cisansa-compliance)
10. [License](#license)
11. [Author](#author)

---

## **Features**

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **GGUF Model Loading** | Native support for llama.cpp GGUF format |
| **Quantized Inference** | GPU-accelerated Q2_K through Q8_0 dequantization |
| **Multi-Head Attention** | Grouped Query Attention (GQA) support |
| **RoPE Embeddings** | Rotary Position Embeddings with scaling |
| **KV Cache** | Efficient key-value caching for autoregressive generation |
| **BPE Tokenization** | Byte-Pair Encoding with chat template support |
| **Sampling Methods** | Temperature, Top-K, Top-P (nucleus) sampling |
| **Streaming Output** | Token-by-token generation output |

### GPU Acceleration

| Backend | Implementation | Performance | Status |
|---------|---------------|-------------|--------|
| **CUDA** | Native CUDA kernels with fused operations | Optimal for NVIDIA GPUs | ✅ Stable |
| **OpenCL** | Cross-platform GPU kernels | AMD, Intel, NVIDIA support | ✅ Stable |
| **Metal** | Apple GPU acceleration | M1/M2/M3 Mac support | 🚧 In Development |

### Quantization Formats

| Format | Bits/Weight | Description | Status |
|--------|-------------|-------------|--------|
| **Q8_0** | 8.5 | Simple 8-bit quantization | ✅ Full support |
| **Q6_K** | 6.6 | 6-bit K-quant | ✅ Full support |
| **Q5_K** | 5.5 | 5-bit K-quant | ✅ Full support |
| **Q4_K** | 4.5 | 4-bit K-quant (recommended) | ✅ Full support |
| **Q3_K** | 3.4 | 3-bit K-quant | ✅ Full support |
| **Q2_K** | 2.6 | 2-bit K-quant | ⚠️ Experimental |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership model |
| **Formal Verification** | 99 Kani proof harnesses |
| **Bounds Checking** | Verified array access |
| **Input Validation** | CLI argument validation |
| **CISA Compliance** | 12 of 15 requirements verified |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GlassBoxAI-Transformer                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  C++ CUDA    │  │ C++ OpenCL   │  │  C++ Metal   │  │   Rust CUDA     │  │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤  ├─────────────────┤  │
│  │ transformer  │  │ transformer- │  │ transformer_ │  │ rust_cuda/      │  │
│  │   .cu        │  │ opencl.cpp   │  │ metal.mm     │  │                 │  │
│  │ facaded_     │  │ facaded-     │  │ transformer_ │  │ facaded_rust_   │  │
│  │ transformer  │  │ transformer- │  │ kernels      │  │ cuda/           │  │
│  │   .cu        │  │ opencl.cpp   │  │ .metal       │  │  ├─ kani/       │  │
│  │              │  │              │  │              │  │  │  (99 proofs) │  │
│  │              │  │              │  │ 🚧 In Dev    │  │  └─ facade.rs   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        Shared Features                                  ││
│  │  • GGUF model format parsing with full metadata support                 ││
│  │  • K-quant dequantization (Q2_K through Q8_0)                           ││
│  │  • BPE tokenization with chat templates (Llama, ChatML, etc.)           ││
│  │  • Grouped Query Attention with RoPE                                    ││
│  │  • Consistent CLI interface across all implementations                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
GlassBoxAI-Transformer/
│
├── transformer.cu                  # C++ CUDA Transformer implementation
├── facaded_transformer.cu          # C++ CUDA Transformer with Facade pattern
├── transformer-opencl.cpp          # C++ OpenCL Transformer implementation
├── facaded-transformer-opencl.cpp  # C++ OpenCL Transformer with Facade pattern
│
├── transformer_metal.mm            # C++ Metal Transformer (🚧 In Development)
├── transformer_kernels.metal       # Metal shader kernels (🚧 In Development)
│
├── rust_cuda/                      # Rust CUDA Transformer implementation
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                 # CLI entry point
│       ├── generator.rs            # Text generation logic
│       ├── model.rs                # Transformer model implementation
│       ├── gguf.rs                 # GGUF file parser
│       ├── tokenizer.rs            # BPE tokenizer
│       ├── quant.rs                # Dequantization routines
│       ├── kernels.rs              # CUDA kernel definitions
│       └── error.rs                # Error types
│
├── facaded_rust_cuda/              # Rust CUDA Transformer with Facade pattern
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                 # CLI entry point
│       ├── facade.rs               # Introspection facade API
│       ├── model.rs                # Transformer model
│       ├── gguf.rs                 # GGUF file parser
│       ├── tokenizer.rs            # BPE tokenizer
│       ├── quant.rs                # Dequantization routines
│       ├── kernels.rs              # CUDA kernel definitions
│       ├── error.rs                # Error types
│       └── kani/                   # Formal verification proofs
│           ├── mod.rs              # Module index
│           ├── bounds.rs           # Bounds checking proofs (8)
│           ├── arithmetic.rs       # Arithmetic safety proofs (11)
│           ├── memory.rs           # Memory safety proofs (9)
│           ├── panics.rs           # No-panic proofs (12)
│           ├── enums.rs            # Enum exhaustion proofs (8)
│           ├── floats.rs           # Float safety proofs (11)
│           ├── tokenizer.rs        # Tokenizer proofs (12)
│           ├── quant.rs            # Quantization proofs (15)
│           ├── model.rs            # Model proofs (13)
│           └── README.md           # Verification documentation
│
├── models/                         # Model storage directory
│   └── *.gguf                      # GGUF model files
│
├── transformer_tests_cuda.sh       # CUDA test suite
├── transformer_tests_opencl.sh     # OpenCL test suite
│
├── license.md                      # MIT License
└── README.md                       # This file
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **GCC/G++** | 11+ | C++ compilation |
| **CUDA Toolkit** | 12.0+ | CUDA compilation |
| **Rust** | 1.75+ | Rust compilation |

### Optional

| Dependency | Version | Purpose |
|------------|---------|---------|
| **OpenCL SDK** | 3.0 | OpenCL compilation |
| **Xcode** | 15+ | Metal compilation (macOS) |
| **Kani** | 0.67+ | Formal verification |

---

## **Installation & Compilation**

### **C++ CUDA Implementation**

```bash
# Standard Transformer
nvcc -O2 -arch=sm_86 -o transformer-cuda transformer.cu

# Facade Transformer
nvcc -O2 -arch=sm_86 -o facaded-transformer-cuda facaded_transformer.cu
```

**Note**: Adjust `-arch=sm_XX` to match your GPU architecture (e.g., `sm_75` for Turing, `sm_80` for Ampere, `sm_86` for RTX 3000 series).

### **C++ OpenCL Implementation**

```bash
# Standard Transformer
g++ -O2 -std=c++17 -o transformer-opencl transformer-opencl.cpp -lOpenCL

# Facade Transformer
g++ -O2 -std=c++17 -o facaded-transformer-opencl facaded-transformer-opencl.cpp -lOpenCL
```

### **C++ Metal Implementation** (🚧 In Development)

```bash
# Requires macOS with Xcode command line tools
clang++ -O2 -std=c++17 -framework Metal -framework Foundation \
    -o transformer-metal transformer_metal.mm
```

**Note**: The Metal implementation is currently in active development. Basic functionality works but some features may be incomplete.

### **Rust CUDA Implementation**

```bash
# Standard Transformer
cd rust_cuda
cargo build --release

# Facade Transformer
cd facaded_rust_cuda
cargo build --release
```

### **Build All**

```bash
# Build all CUDA implementations
nvcc -O2 -arch=sm_86 -o transformer-cuda transformer.cu
nvcc -O2 -arch=sm_86 -o facaded-transformer-cuda facaded_transformer.cu
(cd rust_cuda && cargo build --release)
(cd facaded_rust_cuda && cargo build --release)

# Build all OpenCL implementations
g++ -O2 -std=c++17 -o transformer-opencl transformer-opencl.cpp -lOpenCL
g++ -O2 -std=c++17 -o facaded-transformer-opencl facaded-transformer-opencl.cpp -lOpenCL
```

---

## **CLI Reference**

### **Standard Transformer Commands**

The standard transformer implementations provide core LLM inference functionality.

#### Usage

```
transformer-cuda <command> [options]
transformer-opencl <command> [options]
rust_cuda/target/release/glassbox-transformer <command> [options]
```

#### Commands

| Command | Description |
|---------|-------------|
| `generate` | Generate text from a prompt |
| `info` | Display model information |
| `test` | Run built-in tests |
| `help` | Show help information |

#### Generate Options

| Option | Description |
|--------|-------------|
| `-m, --model <path>` | Path to GGUF model file (required) |
| `-p, --prompt <text>` | Text prompt for generation |
| `-n, --tokens <n>` | Max tokens to generate (default: 256) |
| `-t, --temperature <n>` | Sampling temperature (default: 0.7) |
| `--top-k <n>` | Top-K sampling (default: 40) |
| `--top-p <n>` | Top-P/nucleus sampling (default: 0.9) |
| `-i, --interactive` | Interactive chat mode |
| `-g, --gpu` | Use GPU-accelerated inference |

#### Examples

```bash
# Generate text from a prompt
./transformer-cuda generate -m models/tinyllama.gguf -p "Hello, world" -n 100

# Interactive chat mode with GPU acceleration
./transformer-cuda generate -m models/llama-7b.Q4_K_M.gguf -i -g

# Show model information
./transformer-cuda info -m models/tinyllama.gguf

# Run tests
./transformer-cuda test --all
```

---

### **Facade Transformer Commands**

The facade implementations add deep introspection capabilities for analyzing model internals.

#### Additional Commands

| Command | Description |
|---------|-------------|
| `analyze` | Analyze model internals for a prompt |
| `inspect` | Interactive inspection mode |
| `introspect` | Access hidden states and attention |

#### Introspection Options

| Option | Description |
|--------|-------------|
| `--show-hidden` | Show hidden state statistics |
| `--show-entropy` | Show attention entropy |
| `--show-qkv` | Show Q/K/V vectors |
| `--show-logits` | Show top-k logits |
| `--show-saliency` | Show saliency map |
| `--layer <n>` | Layer to inspect (default: last) |
| `--head <n>` | Attention head to inspect (default: 0) |

#### Facade API Methods

The facade provides programmatic access to internal states:

| Method | Description |
|--------|-------------|
| `getHiddenState(layer, pos)` | Get hidden state vector |
| `getAttentionScores(layer, head)` | Get attention weights |
| `getQKV(layer, pos, type)` | Get Q/K/V vectors |
| `getLogits()` | Get output logits |
| `getLayerNormStats(layer)` | Get normalization statistics |
| `getTokenProbabilities(topK)` | Get token probabilities |

#### Facade Examples

```bash
# Analyze model internals
./facaded-transformer-cuda analyze -m models/llama.gguf -p "What is AI?" --show-qkv

# Generate with hidden state display
./facaded-transformer-cuda generate -m models/llama.gguf -p "Hello" --show-hidden

# Interactive inspection
./facaded-transformer-cuda inspect -m models/llama.gguf
```

---

## **Testing**

### Running All Tests

```bash
# Run CUDA tests
./transformer_tests_cuda.sh

# Run OpenCL tests
./transformer_tests_opencl.sh

# Run Rust tests
cd rust_cuda && cargo test
cd facaded_rust_cuda && cargo test
```

### Test Categories

Each test suite covers:

| Category | Tests |
|----------|-------|
| **Help & Usage** | Command-line interface verification |
| **Model Loading** | GGUF parsing and validation |
| **Quantization** | Dequantization accuracy |
| **Tokenization** | BPE encoding/decoding |
| **Generation** | End-to-end inference |
| **Introspection** | Facade API functionality |
| **Error Handling** | Invalid input handling |

### Test Output Example

```
=========================================
Transformer CUDA Comprehensive Test Suite
=========================================

Group: Quantization Tests
Test: FP16 conversion
  ✓ FP16 to FP32 conversion passed
Test: FP16 zero conversion
  ✓ FP16 zero conversion passed
Test: Quantization type enum
  ✓ Quantization type enum values correct

Group: Facade Tests
Test: Facade initialization
  ✓ Facade starts unloaded
Test: Facade getters
  ✓ Unloaded facade returns zeros

=== Test Results ===
Passed: 15
Failed: 0
Total:  15
====================
```

---

## **Formal Verification with Kani**

### Overview

The Rust Facade implementation includes **99 Kani formal verification proof harnesses** that mathematically prove the absence of certain classes of bugs. This goes beyond traditional testing to provide **mathematical guarantees** about code correctness.

### Verification Categories

The test suite covers 12 of 15 CISA security verification requirements:

| # | Requirement | Module(s) | Status |
|---|-------------|-----------|--------|
| 1 | Strict Bound Checks | `bounds.rs`, `quant.rs`, `model.rs` | ✅ |
| 2 | Pointer Validity Proofs | `memory.rs` | ✅ |
| 3 | No-Panic Guarantee | `panics.rs` | ✅ |
| 4 | Integer Overflow Prevention | `arithmetic.rs`, `model.rs` | ✅ |
| 5 | Division-by-Zero Exclusion | `arithmetic.rs` | ✅ |
| 6 | Global State Consistency | N/A (no shared mutable state) | ⚪ |
| 7 | Deadlock-Free Logic | N/A (no locks in verified code) | ⚪ |
| 8 | Input Sanitization Bounds | `tokenizer.rs` | ✅ |
| 9 | Result Coverage Audit | `panics.rs`, `enums.rs` | ✅ |
| 10 | Memory Leak/Leakage Proofs | `memory.rs` | ✅ |
| 11 | Constant-Time Execution | N/A (no cryptographic secrets) | ⚪ |
| 12 | State Machine Integrity | `enums.rs`, `model.rs` | ✅ |
| 13 | Enum Exhaustion | `enums.rs` | ✅ |
| 14 | Floating-Point Sanity | `floats.rs` | ✅ |
| 15 | Resource Limit Compliance | `memory.rs`, `model.rs` | ✅ |

### Module Proof Counts

| Module | Harnesses | Purpose |
|--------|-----------|---------|
| `bounds.rs` | 8 | Array/slice bounds checking |
| `arithmetic.rs` | 11 | Overflow/division-by-zero prevention |
| `memory.rs` | 9 | Memory safety & resource limits |
| `panics.rs` | 12 | No-panic guarantees |
| `enums.rs` | 8 | Exhaustive enum matching |
| `floats.rs` | 11 | Floating-point safety |
| `tokenizer.rs` | 12 | Input sanitization |
| `quant.rs` | 15 | Quantization arithmetic |
| `model.rs` | 13 | Model loading safety |
| **Total** | **99** | |

### Key Kani Proofs

#### Bounds Checking Proofs
- `verify_get_scale_min_k4_bounds` ✓
- `verify_q8_0_dequant_bounds` ✓
- `verify_tokenizer_decode_bounds` ✓

#### Quantization Safety Proofs
- `verify_q4k_scale_extraction` ✓
- `verify_q6k_bit_reconstruction` ✓
- `verify_bytes_calculation` ✓

#### Arithmetic Safety Proofs
- `verify_block_count_no_overflow` ✓
- `verify_scale_arithmetic_no_overflow` ✓
- `verify_qk_k_division_safe` ✓

#### Memory Safety Proofs
- `verify_bytemuck_alignment` ✓
- `verify_block_struct_sizes` ✓
- `verify_tensor_security_budget` ✓

### Running Kani Verification

```bash
# Run all proofs
cd facaded_rust_cuda
cargo kani

# Run specific proof
cargo kani --harness verify_q8_0_dequant_bounds

# Run proofs for a specific module
cargo kani --harness "verify_q*"
```

### Why Formal Verification Matters

Traditional testing can only verify specific test cases. Formal verification with Kani:

- **Exhaustively checks all possible inputs** within defined bounds
- **Mathematically proves** absence of panics, buffer overflows, and undefined behavior
- **Catches edge cases** that random testing might miss
- **Provides cryptographic-level assurance** for safety-critical code

---

## **CISA/NSA Compliance**

### Secure by Design

This project follows **CISA (Cybersecurity and Infrastructure Security Agency)** and **NSA (National Security Agency)** Secure by Design principles:

| Principle | Implementation |
|-----------|---------------|
| **Memory Safety** | Rust ownership model eliminates buffer overflows, use-after-free, and data races |
| **Formal Verification** | 99 Kani proofs mathematically verify absence of critical bugs |
| **Input Validation** | All CLI inputs validated before processing |
| **Defense in Depth** | Multiple layers of safety (language, compiler, runtime checks) |
| **Secure Defaults** | Safe default configurations throughout |
| **Transparency** | Open source with full code visibility |

### Compliance Checklist

- [x] **Memory-safe language** (Rust implementation)
- [x] **Static analysis** (Rust compiler + Clippy)
- [x] **Formal verification** (99 Kani proof harnesses)
- [x] **Comprehensive testing** (Unit tests + integration tests)
- [x] **Bounds checking** (Verified array access)
- [x] **Input validation** (CLI argument parsing)
- [x] **No unsafe code in critical paths** (Where possible)
- [x] **Documentation** (Inline docs + README)
- [x] **Version control** (Git)
- [x] **License clarity** (MIT License)

### Attestation

This codebase has been developed following secure software development lifecycle (SSDLC) practices and demonstrates:

- **99 formal verification proofs passed** (Kani proofs across 9 modules)
- **Zero warnings** compilation across all implementations
- **Consistent API** across all language/backend combinations
- **Production-ready** code quality

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## **Author**

**Matthew Abbott**  
Email: mattbachg@gmail.com

---

*Built with precision. Verified with rigor. Secured by design.*
