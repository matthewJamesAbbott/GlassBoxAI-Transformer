NOTE: cuda version is rock stable I am working on the opencl version. its passing tests but is timming out on inference thanx for your patience.


# GlassBoxAI-Transformer

**A distributed, introspectable, cross-platform transformer for everyone.**

> This project provides an open, well-documented, and thoroughly tested implementation of an agentic transformer model with full Layer 2 Ethernet support, multi-backend GPU/CPU execution, quantization, GGUF model format, and detailed introspection/facade tools for transparency and practical use.

---

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Features](#features)
- [Command-Line Interface (Help Reference)](#command-line-interface-help-reference)
- [Facade Introspection](#facade-introspection)
- [Supported Platforms](#supported-platforms)
- [Quantization Types](#quantization-types)
- [Usage Examples](#usage-examples)
- [Testing & Safety](#testing--safety)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

GlassBoxAI-Transformer is a fully open, distributed transformer framework designed for research, creative coding, plugin/app integration, and general LLM deployment.  
It uses **Layer 2 Ethernet** for high-performance local-area orchestration, supports **full CPU and GPU fallback/offloading**, and implements a **facade interface** that lets users inspect, intervene, and understand exactly what the model is doing at every layer.

### Why GlassBox?

- **Transparent:** Every key function, kernel, and protocol is open for inspection and modification.
- **Distributed:** Delegate work across machines, devices, and network interfaces—no cloud or vendor lock-in.
- **Cross-platform:** CUDA, Metal, OpenCL, and CPU, all supported (per build); extensible to other backends.
- **Introspectable:** See activations, QKV projections, attention, logits, and more in real time.
- **Reproducible:** 372 comprehensive tests, all passing—verify yourself.
- **Quantized:** Supports a wide array of quant formats for fast, efficient inference.

---

## Architecture

### Core Components

- **transformer.cu:** CUDA-based distributed transformer implementation (baseline reference).
- **facaded_transformer.cu:** Enhanced implementation with full facade agentic interface and introspection controls.
- **Layer 2 networking:** Low-latency protocol, custom framing, and raw socket handling enables direct, secure, and efficient networked inference.
- **Facade:** Runtime inspection, layer/accounting, transparent forward/backward passes.
- **GGUF/Tokenizer:** Modern file formats, flexible, fast and open model support.
- **Quantization:** Multiple quant types for memory/speed optimization.
- **Test Suite:** Comprehensive, cross-cutting, open verification.

### How It Works

1. **Model Loading:**  
   Load GGUF model files and tokenizer data.
2. **CLI/Server/Client Mode:**  
   Run as a server (waiting for requests), client (connecting for inference), or facade (interactive/inspection mode).
3. **Distributed Offloading:**  
   CPU and/or GPU computation is performed locally or offloaded across Layer 2 network participants.
4. **Quantized and Flexible:**  
   Configure quantization, params, runtime modes, and precision to suit your hardware and needs.
5. **Introspection & Facade:**  
   At any prompt or step, inspect internal states: attention, activations, embeddings, QKV, entropy, weights—export, visualize, or analyze.
6. **Safe and reproducible:**  
   Every pathway is covered by automated tests—builds are predictable, safe for plugin development, and ready for production or experimentation.

---

## Features

- **Agentic facade pattern:**  
  Full control and introspection over every layer and operation.
- **Layer 2 Ethernet protocol:**  
  Fast, direct, low-overhead networking for distributed and local operation.
- **Multi-platform, multi-backend:**  
  CUDA, Metal, OpenCL, and CPU fallback/offload (cross-platform binaries, future integrations).
- **Quantization support:**  
  Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, fp16, bf16, legacy GGML types.
- **Comprehensive test suite:**  
  317+ pass-checked tests for protocol, networking, quantization, kernel logic, CLI UX, facade, and more.
- **Transparent CLI/GUI:**  
  Detailed help and option management, explicit controls for research and day-to-day work.
- **Open, well-commented source:**  
  Easy to read, study, modify, and adapt for your own needs.

---

## Command-Line Interface (Help Reference)

### **Server Mode**
```
./facaded_transformer server [options]
  -i, --interface <name>       Network interface (default: eth0)
  -l, --layers <n>             Total transformer layers (default: 12)
  -e, --embed <dim>            Embedding dimension (default: 768)
  -f, --ffn <dim>              FFN hidden dimension (default: 3072)
  -a, --heads <n>              Number of attention heads (default: 12)
  -k, --kvheads <n>            Number of KV heads for GQA (default: 12)
  -q, --seq-len <n>            Sequence length (default: 512)
  -v, --vocab-size <n>         Vocabulary size (default: 50257)
  -x, --max-seq-len <n>        Maximum sequence length (default: 2048)
  -m, --messages <n>           Max messages to process (default: 100)
  -g, --gpu <yes/no>           GPU availability (default: yes)
  -c, --clients <n>            Max concurrent clients (default: 4)
  --quant <type>               Quantization type: none|q4_0|q4_1|q5_0|q5_1|q8_0|
                               q2_k|q3_k|q4_k|q5_k|q6_k|q8_k|f16|bf16 (default: none)
  --rope-base <n>              RoPE base frequency (default: 10000.0)
  --rope-scale <n>             RoPE scaling factor (default: 1.0)
  --eps <n>                    Layer norm epsilon (default: 1e-5)
  --dropout <n>                Dropout rate 0.0-1.0 (default: 0.0)
  --verbose                    Enable verbose output
  --help                       Show help
```

### **Client Mode**
```
./facaded_transformer client [options]
  -i, --interface <name>       Network interface (default: eth0)
  -s, --server <mac>           Server MAC address (required)
  -l, --layers <n>             Total transformer layers (default: 12)
  -r, --remote <n>             Remote layers to execute (default: 6)
  --start-layer <n>            Starting layer for remote execution (default: auto)
  -e, --embed <dim>            Embedding dimension (default: 768)
  -f, --ffn <dim>              FFN hidden dimension (default: 3072)
  -a, --heads <n>              Number of attention heads (default: 12)
  -k, --kvheads <n>            KV heads for GQA (default: 12)
  -q, --seq-len <n>            Sequence length (default: 512)
  -v, --vocab-size <n>         Vocabulary size (default: 50257)
  -x, --max-seq-len <n>        Maximum sequence length (default: 2048)
  --quant <type>               Quantization type (see server options)
  --rope-base <n>              RoPE base frequency (default: 10000.0)
  --rope-scale <n>             RoPE scaling factor (default: 1.0)
  --eps <n>                    Layer norm epsilon (default: 1e-5)
  --no-cache                   Disable activation caching
  --no-grad-cache              Disable gradient caching
  --timeout <ms>               Connection timeout (default: 5000ms)
  --retries <n>                Connection retry count (default: 3)
  --verbose                    Enable verbose output
  --help                       Show help
```

### **Facade Mode**
```
./facaded_transformer facade [options]
  --model <path>               GGUF model file path (required)
  --tokenizer <path>           Tokenizer JSON file path
  --prompt <text>              Text prompt for generation
  --max-tokens <n>             Maximum tokens to generate (default: 100)
  --temperature <n>            Sampling temperature (default: 1.0)
  --top-k <n>                  Top-K sampling (default: 40)
  --top-p <n>                  Top-P nucleus sampling (default: 0.9)
  --inspect                    Enable introspection mode
  --show-attention             Display attention weights
  --show-hidden <layer>        Display hidden states for layer
  --show-qkv <layer>           Display Q/K/V vectors for layer
  --show-logits                Display output logits
  --show-entropy               Display attention entropy per layer
  --show-saliency <pos>        Display saliency map for token position
  --show-weights <layer>       Display weight matrices for layer
  --show-tensors               List all tensor names in model
  --dump-hidden <file>         Dump hidden states to CSV file
  --dump-attention <file>      Dump attention weights to CSV file
  --layer <n>                  Specific layer for inspection (default: all)
  --head <n>                   Specific attention head (default: all)
  --position <n>               Specific token position (default: all)
  --verbose                    Enable verbose
  --help                       Show help
```

### **Benchmark Mode**
```
./facaded_transformer benchmark [options]
  -i, --interface <name>       Network interface (default: eth0)
  -s, --server <mac>           Server MAC address (required)
  -n, --iterations <n>         Benchmark iterations (default: 10)
  -l, --layers <n>             Layers to benchmark (default: 12)
  -e, --embed <dim>            Embedding dimension (default: 768)
  -q, --seq-len <n>            Sequence length (default: 512)
  --batch-size <n>             Batch size (default: 1)
  --warmup <n>                 Warmup iterations (default: 2)
  --output <file>              Save results to CSV file
  --verbose                    Enable verbose output
  --help                       Show help
```

### **Test Suite Mode**
```
./facaded_transformer test [options]
  --all                        Run all tests
  --protocol                   Test protocol handling
  --config                     Test configuration
  --quant                      Test quantization/dequantization
  --kernels                    Test CUDA kernels (requires GPU)
  --network                    Test network layer
  --facade                     Test facade introspection functions
  --tokenizer                  Test tokenizer encode/decode
  --gguf                       Test GGUF model loading
  --verbose                    Verbose test output
  --help                       Show help
```

---

## Facade Introspection

The facade interface enables deep insight into model operation:
- **Attention weights and entropy**—see where the model looks.
- **Layer states**—inspect activations, hidden states, and Q/K/V vectors.
- **Embeddings**—inspect positional and token embeddings.
- **Weights and tensors**—list, view, and export all model tensors.
- **Saliency maps**—explainability for tokens and layers.
- **Dump/Export**—CSV output for scientific analysis or plugin integration.

Example:
```
./facaded_transformer facade --model model.gguf --prompt "Explain quantum tunneling" --show-attention
```

---

## Supported Platforms

- **GPU:** CUDA (NVIDIA), Metal (Apple Silicon), OpenCL (Intel/AMD/Apple), Hybrid pipelines (planned/coming soon)
- **CPU fallback:** Full inference and layer offloading supported.
- **Distributed networking:** Layer 2 Ethernet/LAN—works on Linux, macOS, and supported Windows builds.

All main features match across platforms; ports are developed to maintain identical capability and option parity.

---

## Quantization Types

- **none:** float32, 32 bpw
- **f16:** float16, 16 bpw
- **bf16:** brain float16, 16 bpw
- **q8_0:** 8-bit, 8.5 bpw
- **q6_k, q5_k, q4_k, q3_k, q2_k:** advanced GGML quant types (2.625 bpw and up)
- **q4_0, q4_1, q5_0, q5_1:** legacy quant formats

Smaller quantizations mean faster inference and lower memory footprint, at (potential) cost of minor accuracy.

---

## Usage Examples

- **Start server on eth0 with 24 layers and Q4_K quantization:**
  ```
  ./facaded_transformer server -i eth0 -l 24 -e 1024 --quant q4_k
  ```
- **Connect client with custom sequence length and vocab:**
  ```
  ./facaded_transformer client -s AA:BB:CC:DD:EE:FF -q 1024 -v 32000 -r 12
  ```
- **Run facade with introspection:**
  ```
  ./facaded_transformer facade --model model.gguf --tokenizer tok.json --prompt "Hello" --inspect
  ```
- **Inspect attention weights for layer 0:**
  ```
  ./facaded_transformer facade --model model.gguf --prompt "Test" --show-attention --layer 0
  ```
- **Dump hidden states to file:**
  ```
  ./facaded_transformer facade --model model.gguf --prompt "Test" --dump-hidden hidden.csv
  ```
- **Run facade and quantization tests:**
  ```
  ./facaded_transformer test --facade --quant --verbose
  ```

---

## Testing & Safety

This project maintains one of the most exhaustive open test suites for transformers and distributed LLMs:

- **317+ Tests, 100% Passing (as of latest release).**
  - Protocol, networking, edge cases, code quality, kernel correctness, quantization, GGUF/tokenizer/file handling, CLI coverage, and introspection.
- **Code checks for style, safety, and reproducibility.**
- **Easy to validate—run the suite yourself (`./COMPREHENSIVE_TEST_SUITE.sh`).**
- **Bit-exactness and gold-standard output validation planned for future releases.**

---

## License

Open source (MIT). See [LICENSE](LICENSE).

---

## Acknowledgments & History

This project was designed and implemented by [Matthew Abbott](https://github.com/matthewJamesAbbott) and community, inspired by the need for a robust, distributed, transparent transformer for everyone—not just for the audio or scene community, but all users, domains, and hardware.

---

## Contact / Further Info

Issues, improvements, and discussion are welcome via GitHub issues and Pull Requests.

---
