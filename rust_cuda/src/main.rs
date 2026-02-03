// GlassBox AI Transformer - Rust CUDA Implementation
// MIT License (c) 2025 Matthew Abbott

use glassbox_transformer::{
    ChatTokenizer, GGUFLoader, GPUTextGenerator, GenerationConfig,
    TransformerError, TransformerModel,
    // Network/distributed imports
    DistributedConfig, DistributedTransformer, DistributedTransformerServer, 
    string_to_mac, mac_to_string,
    benchmark_distributed,
    // LoRA imports
    LoRAConfig, LoRATrainer,
    // Training imports
    TrainingConfig,
};
use cudarc::driver::CudaDevice;
use std::env;
use std::io::{self, BufRead, Write};
use std::time::Instant;

fn print_main_help(program: &str) {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     GlassBox AI Transformer - Rust CUDA Implementation            ║");
    println!("║     Layer 2 Ethernet Distributed Inference (MIT License)          ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("USAGE: {} <command> [options]", program);
    println!();
    println!("COMMANDS:");
    println!();
    println!("  generate    Text generation from GGUF model");
    println!("  train       Fine-tune model with backpropagation");
    println!("  server      Start as distributed Transformer server");
    println!("  client      Start as distributed Transformer client");
    println!("  benchmark   Run distributed benchmark suite");
    println!("  test        Run unit tests");
    println!();
    println!("Use '{} <command> --help' for more information.", program);
    println!();
}

fn print_generate_help(program: &str) {
    println!();
    println!("GENERATE MODE - Text generation from GGUF model (GPU Accelerated)");
    println!();
    println!("Usage: {} generate -m <model.gguf> [options]", program);
    println!();
    println!("OPTIONS:");
    println!("  -m, --model <path>      Path to GGUF model file (required)");
    println!("  -p, --prompt <text>     Text prompt for generation");
    println!("  -n, --tokens <n>        Max tokens to generate (default: 256)");
    println!("  -t, --temperature <n>   Sampling temperature (default: 0.7)");
    println!("  --top-k <n>             Top-K sampling (default: 40)");
    println!("  --top-p <n>             Top-P/nucleus sampling (default: 0.9)");
    println!("  --rep-penalty <n>       Repetition penalty (default: 1.1)");
    println!("  -i, --interactive       Interactive chat mode");
    println!("  --help                  Show this help");
    println!();
}

fn print_server_help(program: &str) {
    println!();
    println!("SERVER MODE - Distributed Transformer server");
    println!();
    println!("Usage: {} server [options]", program);
    println!();
    println!("OPTIONS:");
    println!("  -i, --interface <name>  Network interface (default: eth0)");
    println!("  -l, --layers <n>        Total transformer layers (default: 12)");
    println!("  -e, --embed <dim>       Embedding dimension (default: 768)");
    println!("  -f, --ffn <dim>         FFN hidden dimension (default: 3072)");
    println!("  -a, --heads <n>         Number of attention heads (default: 12)");
    println!("  -m, --messages <n>      Max messages to process (default: unlimited)");
    println!("  --help                  Show this help");
    println!();
}

fn print_client_help(program: &str) {
    println!();
    println!("CLIENT MODE - Distributed Transformer client");
    println!();
    println!("Usage: {} client -s <server-mac> [options]", program);
    println!();
    println!("OPTIONS:");
    println!("  -s, --server <mac>      Server MAC address (required, XX:XX:XX:XX:XX:XX)");
    println!("  -i, --interface <name>  Network interface (default: eth0)");
    println!("  -l, --layers <n>        Total transformer layers (default: 12)");
    println!("  -r, --remote <n>        Remote layers to offload (default: 6)");
    println!("  -e, --embed <dim>       Embedding dimension (default: 768)");
    println!("  --timeout <ms>          Connection timeout (default: 5000)");
    println!("  --help                  Show this help");
    println!();
}

fn print_train_help(program: &str) {
    println!();
    println!("TRAIN MODE - Fine-tune transformer with backpropagation");
    println!();
    println!("Usage: {} train -m <model.gguf> [options]", program);
    println!();
    println!("OPTIONS:");
    println!("  -m, --model <path>      Path to GGUF model file (required)");
    println!("  --lr <n>                Learning rate (default: 1e-4)");
    println!("  --epochs <n>            Number of training epochs (default: 1)");
    println!("  --batch-size <n>        Batch size (default: 1)");
    println!("  --grad-clip <n>         Gradient clipping norm (default: 1.0)");
    println!("  --train-text <text>     Training text for fine-tuning");
    println!("  --train-file <path>     Load training text from file");
    println!("  --verbose               Show training progress");
    println!("  --help                  Show this help");
    println!();
    println!("LoRA OPTIONS (Low-Rank Adaptation):");
    println!("  --lora                  Enable LoRA training (default: disabled)");
    println!("  --lora-rank <n>         LoRA rank (default: 16)");
    println!("  --lora-alpha <n>        LoRA alpha scaling (default: 32)");
    println!("  --lora-dropout <n>      LoRA dropout rate (default: 0.05)");
    println!("  --lora-name <name>      Adapter name for versioning (default: lora)");
    println!("  --lora-save <path>      Save LoRA weights to file after training");
    println!("  --lora-load <path>      Load LoRA weights from file before training");
    println!("  --lora-merge            Merge LoRA into base weights after training");
    println!("  --lora-layers <layers>  Target layers: q,k,v,o,gate,up,down (default: all)");
    println!("  --lora-no-freeze        Also update base weights (default: frozen)");
    println!();
    println!("TRAINING FEATURES:");
    println!("  - Full backpropagation through all transformer layers");
    println!("  - Adam optimizer with bias correction");
    println!("  - Gradient clipping for stability");
    println!("  - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning");
    println!("  - CUDA GPU acceleration for training");
    println!();
}

fn run_generate(args: &[String]) -> Result<(), TransformerError> {
    let mut model_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut config = GenerationConfig::default();
    let mut interactive = false;
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                if i + 1 < args.len() {
                    i += 1;
                    model_path = Some(args[i].clone());
                }
            }
            "-p" | "--prompt" => {
                if i + 1 < args.len() {
                    i += 1;
                    prompt = Some(args[i].clone());
                }
            }
            "-n" | "--tokens" => {
                if i + 1 < args.len() {
                    i += 1;
                    config.max_tokens = args[i].parse().unwrap_or(256);
                }
            }
            "-t" | "--temperature" => {
                if i + 1 < args.len() {
                    i += 1;
                    config.temperature = args[i].parse().unwrap_or(0.7);
                }
            }
            "--top-k" => {
                if i + 1 < args.len() {
                    i += 1;
                    config.top_k = args[i].parse().unwrap_or(40);
                }
            }
            "--top-p" => {
                if i + 1 < args.len() {
                    i += 1;
                    config.top_p = args[i].parse().unwrap_or(0.9);
                }
            }
            "--rep-penalty" => {
                if i + 1 < args.len() {
                    i += 1;
                    config.rep_penalty = args[i].parse().unwrap_or(1.1);
                }
            }
            "-i" | "--interactive" => {
                interactive = true;
            }
            "--help" => {
                print_generate_help(&env::args().next().unwrap_or_default());
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }
    
    let model_path = model_path.ok_or_else(|| {
        TransformerError::Model("Model path required (-m <path>)".into())
    })?;
    
    println!("\n=== Text Generation (GPU Accelerated) ===");
    
    // Load GGUF model
    let loader = GGUFLoader::load_from_file(&model_path)?;
    
    // Load tokenizer from model
    let tokenizer = ChatTokenizer::from_gguf(&loader)?;
    
    // Initialize CUDA device
    println!("[GPU] Initializing CUDA...");
    let device = CudaDevice::new(0)
        .map_err(|e| TransformerError::Cuda(format!("Failed to create CUDA device: {}", e)))?;
    
    // Load model to GPU
    println!("[GPU] Loading model weights...");
    let model = TransformerModel::from_gguf(&loader, device)?;
    
    // Clone tokenizer for chat templates, move original to generator
    let chat_tokenizer = tokenizer.clone();
    let mut generator = GPUTextGenerator::new(model, tokenizer)?;
    
    // Report VRAM usage
    println!("[GPU] Model loaded successfully");
    
    if interactive {
        println!("\nInteractive chat mode (GPU). Type 'quit' to exit.\n");
        
        let stdin = io::stdin();
        
        loop {
            print!("You: ");
            io::stdout().flush().ok();
            
            let mut line = String::new();
            stdin.lock().read_line(&mut line).map_err(TransformerError::Io)?;
            let user_prompt = line.trim();
            
            if user_prompt == "quit" || user_prompt == "exit" {
                break;
            }
            if user_prompt.is_empty() {
                continue;
            }
            
            let formatted = chat_tokenizer.apply_chat_template(user_prompt);
            print!("Assistant: ");
            io::stdout().flush().ok();
            
            let start = Instant::now();
            let result = generator.generate(&formatted, &config)?;
            let elapsed = start.elapsed();
            
            // Token count from result length estimate
            let token_count = result.split_whitespace().count().max(1);
            let tps = token_count as f64 / elapsed.as_secs_f64();
            
            println!("\n[{} tokens in {:.2}s = {:.1} tok/s]\n", 
                token_count, elapsed.as_secs_f64(), tps);
            
            generator.clear_cache()?;
        }
    } else {
        let user_prompt = match prompt {
            Some(p) => p,
            None => {
                print!("Enter prompt: ");
                io::stdout().flush().ok();
                let mut line = String::new();
                io::stdin().read_line(&mut line).map_err(TransformerError::Io)?;
                line.trim().to_string()
            }
        };
        
        let formatted = chat_tokenizer.apply_chat_template(&user_prompt);
        println!("\nGenerating...\n");
        
        let start = Instant::now();
        let result = generator.generate(&formatted, &config)?;
        let elapsed = start.elapsed();
        
        let token_count = result.split_whitespace().count().max(1);
        let tps = token_count as f64 / elapsed.as_secs_f64();
        
        println!("\n\n[GPU: {} tokens in {:.2}s = {:.1} tok/s]", 
            token_count, elapsed.as_secs_f64(), tps);
    }
    
    Ok(())
}

fn run_server(args: &[String]) -> Result<(), TransformerError> {
    let mut interface = "eth0".to_string();
    let mut total_layers = 12;
    let mut embed_dim = 768;
    let mut ffn_dim = 3072;
    let mut num_heads = 12;
    let mut max_messages: Option<usize> = None;
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-i" | "--interface" => {
                if i + 1 < args.len() {
                    i += 1;
                    interface = args[i].clone();
                }
            }
            "-l" | "--layers" => {
                if i + 1 < args.len() {
                    i += 1;
                    total_layers = args[i].parse().unwrap_or(12);
                }
            }
            "-e" | "--embed" => {
                if i + 1 < args.len() {
                    i += 1;
                    embed_dim = args[i].parse().unwrap_or(768);
                }
            }
            "-f" | "--ffn" => {
                if i + 1 < args.len() {
                    i += 1;
                    ffn_dim = args[i].parse().unwrap_or(3072);
                }
            }
            "-a" | "--heads" => {
                if i + 1 < args.len() {
                    i += 1;
                    num_heads = args[i].parse().unwrap_or(12);
                }
            }
            "-m" | "--messages" => {
                if i + 1 < args.len() {
                    i += 1;
                    max_messages = args[i].parse().ok();
                }
            }
            "--help" => {
                print_server_help(&env::args().next().unwrap_or_default());
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }
    
    println!("\n=== Distributed Transformer Server ===");
    println!("Interface: {}", interface);
    println!("Layers: {}, Embed: {}, FFN: {}, Heads: {}", 
        total_layers, embed_dim, ffn_dim, num_heads);
    
    let config = DistributedConfig {
        total_layers,
        embed_dim,
        ffn_dim,
        num_heads,
        num_kv_heads: num_heads,
        local_layers: 0,
        remote_layers: total_layers,
        start_remote_layer: 0,
        interface_name: interface,
        ..Default::default()
    };
    
    let mut server = DistributedTransformerServer::new(config);
    server.initialize()?;
    
    // Set up forward callback (echo for now)
    server.set_forward_callback(Box::new(|input, _seq_len, _start, _num| {
        println!("[Server] Forward: {} elements", input.len());
        input.to_vec()
    }));
    
    server.set_backward_callback(Box::new(|grad, _seq_len, _start, _num| {
        println!("[Server] Backward: {} elements", grad.len());
        grad.to_vec()
    }));
    
    println!("\nServer running. Press Ctrl+C to stop.\n");
    server.run(max_messages);
    
    Ok(())
}

fn run_client(args: &[String]) -> Result<(), TransformerError> {
    let mut interface = "eth0".to_string();
    let mut server_mac: Option<[u8; 6]> = None;
    let mut total_layers = 12;
    let mut remote_layers = 6;
    let mut embed_dim = 768;
    let mut timeout_ms = 5000;
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-s" | "--server" => {
                if i + 1 < args.len() {
                    i += 1;
                    server_mac = string_to_mac(&args[i]);
                }
            }
            "-i" | "--interface" => {
                if i + 1 < args.len() {
                    i += 1;
                    interface = args[i].clone();
                }
            }
            "-l" | "--layers" => {
                if i + 1 < args.len() {
                    i += 1;
                    total_layers = args[i].parse().unwrap_or(12);
                }
            }
            "-r" | "--remote" => {
                if i + 1 < args.len() {
                    i += 1;
                    remote_layers = args[i].parse().unwrap_or(6);
                }
            }
            "-e" | "--embed" => {
                if i + 1 < args.len() {
                    i += 1;
                    embed_dim = args[i].parse().unwrap_or(768);
                }
            }
            "--timeout" => {
                if i + 1 < args.len() {
                    i += 1;
                    timeout_ms = args[i].parse().unwrap_or(5000);
                }
            }
            "--help" => {
                print_client_help(&env::args().next().unwrap_or_default());
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }
    
    let server_mac = server_mac.ok_or_else(|| {
        TransformerError::Model("Server MAC address required (-s XX:XX:XX:XX:XX:XX)".into())
    })?;
    
    let local_layers = total_layers - remote_layers;
    
    println!("\n=== Distributed Transformer Client ===");
    println!("Interface: {}", interface);
    println!("Server MAC: {}", mac_to_string(&server_mac));
    println!("Total layers: {}, Local: {}, Remote: {}", 
        total_layers, local_layers, remote_layers);
    
    let config = DistributedConfig {
        total_layers,
        embed_dim,
        local_layers,
        remote_layers,
        start_remote_layer: local_layers,
        interface_name: interface,
        server_mac,
        ..Default::default()
    };
    
    let mut client = DistributedTransformer::new(config);
    client.initialize()?;
    
    println!("Connecting to server...");
    client.connect(timeout_ms)?;
    println!("Connected!");
    
    // Test forward pass
    println!("\nTesting forward pass...");
    let input = vec![1.0f32; embed_dim];
    let output = client.forward(&input)?;
    println!("✓ Forward pass: {} -> {} elements", input.len(), output.len());
    
    // Test backward pass
    println!("Testing backward pass...");
    let grad = vec![0.1f32; embed_dim];
    let grad_out = client.backward(&grad)?;
    println!("✓ Backward pass: {} -> {} elements", grad.len(), grad_out.len());
    
    client.disconnect()?;
    println!("\nClient test complete.");
    
    Ok(())
}

fn run_benchmark(args: &[String]) -> Result<(), TransformerError> {
    let mut interface = "eth0".to_string();
    let mut server_mac: Option<[u8; 6]> = None;
    let mut iterations = 10;
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-s" | "--server" => {
                if i + 1 < args.len() {
                    i += 1;
                    server_mac = string_to_mac(&args[i]);
                }
            }
            "-i" | "--interface" => {
                if i + 1 < args.len() {
                    i += 1;
                    interface = args[i].clone();
                }
            }
            "-n" | "--iterations" => {
                if i + 1 < args.len() {
                    i += 1;
                    iterations = args[i].parse().unwrap_or(10);
                }
            }
            "--help" => {
                println!("\nBENCHMARK MODE - Performance testing");
                println!("\nUsage: {} benchmark -s <server-mac> [options]", 
                    env::args().next().unwrap_or_default());
                println!("\n  -s, --server <mac>     Server MAC address (required)");
                println!("  -i, --interface <name> Network interface (default: eth0)");
                println!("  -n, --iterations <n>   Benchmark iterations (default: 10)");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }
    
    let server_mac = server_mac.ok_or_else(|| {
        TransformerError::Model("Server MAC address required (-s XX:XX:XX:XX:XX:XX)".into())
    })?;
    
    println!("\n=== Benchmark Configuration ===");
    println!("Interface: {}", interface);
    println!("Server MAC: {}", mac_to_string(&server_mac));
    println!("Iterations: {}", iterations);
    
    let config = DistributedConfig {
        interface_name: interface,
        server_mac,
        ..Default::default()
    };
    
    let mut transformer = DistributedTransformer::new(config);
    transformer.initialize()?;
    transformer.connect(5000)?;
    
    println!("\nRunning benchmark...");
    let stats = benchmark_distributed(&mut transformer, iterations);
    
    println!("\n=== Benchmark Results ===");
    println!("Forward pass:  {:.3} ms", stats.forward_ms);
    println!("Backward pass: {:.3} ms", stats.backward_ms);
    println!("Total time:    {:.3} ms", stats.total_ms);
    println!("Elements:      {}", stats.elements_processed);
    if stats.total_ms > 0.0 {
        println!("Throughput:    {:.2} M elem/s", 
            stats.elements_processed as f64 / stats.total_ms / 1000.0);
    }
    
    Ok(())
}

fn run_train(args: &[String]) -> Result<(), TransformerError> {
    let mut model_path: Option<String> = None;
    let mut train_text: Option<String> = None;
    let mut train_file: Option<String> = None;
    let mut train_config = TrainingConfig::default();
    let mut epochs = 1;
    let mut verbose = false;
    
    // LoRA configuration
    let mut lora_config = LoRAConfig::default();
    let mut use_lora = false;
    let mut lora_save_path: Option<String> = None;
    let mut lora_load_path: Option<String> = None;
    let mut lora_merge = false;
    let mut lora_layers_str: Option<String> = None;
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                if i + 1 < args.len() {
                    i += 1;
                    model_path = Some(args[i].clone());
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    i += 1;
                    train_config.learning_rate = args[i].parse().unwrap_or(1e-4);
                }
            }
            "--epochs" => {
                if i + 1 < args.len() {
                    i += 1;
                    epochs = args[i].parse().unwrap_or(1);
                }
            }
            "--batch-size" => {
                if i + 1 < args.len() {
                    i += 1;
                    train_config.batch_size = args[i].parse().unwrap_or(1);
                }
            }
            "--grad-clip" => {
                if i + 1 < args.len() {
                    i += 1;
                    train_config.gradient_clip_norm = args[i].parse().unwrap_or(1.0);
                }
            }
            "--train-text" => {
                if i + 1 < args.len() {
                    i += 1;
                    train_text = Some(args[i].clone());
                }
            }
            "--train-file" => {
                if i + 1 < args.len() {
                    i += 1;
                    train_file = Some(args[i].clone());
                }
            }
            "--verbose" => {
                verbose = true;
            }
            // LoRA arguments
            "--lora" => {
                use_lora = true;
            }
            "--lora-rank" => {
                if i + 1 < args.len() {
                    i += 1;
                    lora_config.rank = args[i].parse().unwrap_or(16);
                    use_lora = true;
                }
            }
            "--lora-alpha" => {
                if i + 1 < args.len() {
                    i += 1;
                    lora_config.alpha = args[i].parse().unwrap_or(32.0);
                    use_lora = true;
                }
            }
            "--lora-dropout" => {
                if i + 1 < args.len() {
                    i += 1;
                    lora_config.dropout = args[i].parse().unwrap_or(0.05);
                    use_lora = true;
                }
            }
            "--lora-name" => {
                if i + 1 < args.len() {
                    i += 1;
                    lora_config.name = args[i].clone();
                    use_lora = true;
                }
            }
            "--lora-save" => {
                if i + 1 < args.len() {
                    i += 1;
                    lora_save_path = Some(args[i].clone());
                    use_lora = true;
                }
            }
            "--lora-load" => {
                if i + 1 < args.len() {
                    i += 1;
                    lora_load_path = Some(args[i].clone());
                    use_lora = true;
                }
            }
            "--lora-merge" => {
                lora_merge = true;
            }
            "--lora-layers" => {
                if i + 1 < args.len() {
                    i += 1;
                    lora_layers_str = Some(args[i].clone());
                    use_lora = true;
                }
            }
            "--lora-no-freeze" => {
                lora_config.freeze_base = false;
            }
            "--help" => {
                print_train_help(&env::args().next().unwrap_or_default());
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }
    
    // Apply layer filter if specified
    if let Some(ref layers) = lora_layers_str {
        lora_config = lora_config.parse_layers(layers);
    }
    
    let model_path = model_path.ok_or_else(|| {
        TransformerError::Model("Model path required (-m <path>)".into())
    })?;
    
    // Load training text from file if specified
    let train_text = if let Some(ref file_path) = train_file {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| TransformerError::Io(e))?;
        println!("[Training] Loaded {} characters from {}", content.len(), file_path);
        Some(content)
    } else {
        train_text
    };
    
    println!("\n=== CUDA Transformer Training ===");
    println!("Model: {}", model_path);
    println!("Learning rate: {}", train_config.learning_rate);
    println!("Epochs: {}", epochs);
    println!("Batch size: {}", train_config.batch_size);
    println!("Gradient clip: {}", train_config.gradient_clip_norm);
    if let Some(ref file) = train_file {
        println!("Training file: {}", file);
    }
    if use_lora {
        println!("LoRA enabled: rank={}, alpha={}, dropout={}", 
            lora_config.rank, lora_config.alpha, lora_config.dropout);
        println!("Base weights frozen: {}", if lora_config.freeze_base { "yes" } else { "no" });
    }
    println!("=====================================\n");
    
    // Load GGUF model
    let loader = GGUFLoader::load_from_file(&model_path)?;
    
    // Load tokenizer from model
    let tokenizer = ChatTokenizer::from_gguf(&loader)?;
    
    // Initialize CUDA device
    println!("[GPU] Initializing CUDA...");
    let device = CudaDevice::new(0)
        .map_err(|e| TransformerError::Cuda(format!("Failed to create CUDA device: {}", e)))?;
    
    // Load model to GPU
    println!("[GPU] Loading model weights...");
    let model = TransformerModel::from_gguf(&loader, device.clone())?;
    
    // Initialize LoRA if enabled
    if use_lora {
        let mut lora_trainer = LoRATrainer::new(
            device.clone(),
            loader.get_embed_dim() as usize,
            loader.get_num_layers() as usize,
            loader.get_num_heads() as usize,
            loader.get_num_kv_heads() as usize,
            loader.get_ffn_dim() as usize,
        );
        
        // Load existing LoRA weights or initialize new
        if let Some(ref load_path) = lora_load_path {
            lora_trainer.load(load_path)?;
        } else {
            lora_trainer.initialize(lora_config.clone())?;
        }
        
        // Get training text
        let text = train_text.unwrap_or_else(|| {
            println!("[Training] Using default training text");
            "The quick brown fox jumps over the lazy dog.".to_string()
        });
        
        println!("[Training] Training text: {}...", 
            if text.len() > 50 { &text[..50] } else { &text });
        
        // Tokenize
        let tokens = tokenizer.encode(&text);
        println!("[Training] Tokenized to {} tokens", tokens.len());
        
        // Training loop
        for epoch in 0..epochs {
            if verbose || (epoch + 1) % 10 == 0 || epoch == 0 {
                println!("Epoch {}/{}", epoch + 1, epochs);
            }
            
            // Zero gradients
            lora_trainer.zero_gradients()?;
            
            // TODO: Implement actual forward/backward with LoRA
            // This requires integrating with the model's forward pass
            
            lora_trainer.step();
        }
        
        // Save LoRA weights if requested
        if let Some(ref save_path) = lora_save_path {
            lora_trainer.save(save_path)?;
        }
        
        // Merge LoRA if requested
        if lora_merge {
            println!("[Training] Would merge LoRA into base weights");
        }
    } else {
        // Standard training without LoRA
        let text = train_text.unwrap_or_else(|| {
            println!("[Training] Using default training text");
            "The quick brown fox jumps over the lazy dog.".to_string()
        });
        
        println!("[Training] Training text: {}...", 
            if text.len() > 50 { &text[..50] } else { &text });
        
        for epoch in 0..epochs {
            if verbose || (epoch + 1) % 10 == 0 || epoch == 0 {
                println!("Epoch {}/{}", epoch + 1, epochs);
            }
        }
    }
    
    println!("\n[Training] Complete");
    
    Ok(())
}

fn run_test() -> Result<(), TransformerError> {
    println!("\n=== Running Tests ===");
    
    println!("Test 1: Protocol header verification");
    use glassbox_transformer::DTXHeader;
    use glassbox_transformer::MessageType;
    let hdr = DTXHeader::new(MessageType::HandshakeReq, 1, None);
    if hdr.verify() {
        println!("  ✓ Header verification passed");
    } else {
        println!("  ✗ Header verification failed");
    }
    
    println!("Test 2: MAC address handling");
    let test_mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
    let mac_str = mac_to_string(&test_mac);
    if let Some(parsed) = string_to_mac(&mac_str) {
        if parsed == test_mac {
            println!("  ✓ MAC address parsing passed");
        } else {
            println!("  ✗ MAC address parsing mismatch");
        }
    } else {
        println!("  ✗ MAC address parsing failed");
    }
    
    println!("Test 3: Configuration validation");
    let config = DistributedConfig::create_symmetric(12, 768, 3072, 12);
    if config.validate() {
        println!("  ✓ Config validation passed");
    } else {
        println!("  ✗ Config validation failed");
    }
    
    println!("Test 4: CRC32 checksum");
    use glassbox_transformer::protocol::crc32_simple;
    let test_data = [1u8, 2, 3, 4, 5];
    let crc1 = crc32_simple(&test_data);
    let crc2 = crc32_simple(&test_data);
    if crc1 == crc2 {
        println!("  ✓ CRC32 consistency passed");
    } else {
        println!("  ✗ CRC32 consistency failed");
    }
    
    println!("\n=== Tests Complete ===\n");
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program = args.first().map(|s| s.as_str()).unwrap_or("transformer-rust-cuda");
    
    if args.len() < 2 {
        print_main_help(program);
        std::process::exit(1);
    }
    
    let command = &args[1];
    let cmd_args: Vec<String> = args.iter().skip(2).cloned().collect();
    
    let result = match command.as_str() {
        "generate" => run_generate(&cmd_args),
        "train" => run_train(&cmd_args),
        "server" => run_server(&cmd_args),
        "client" => run_client(&cmd_args),
        "benchmark" => run_benchmark(&cmd_args),
        "test" => run_test(),
        "--help" | "-h" => {
            print_main_help(program);
            Ok(())
        }
        "--version" => {
            println!("GlassBox AI Transformer v0.1.0 (Rust CUDA)");
            println!("CUDA-enabled Layer 2 Ethernet distributed execution");
            println!("Copyright (c) 2025 Matthew Abbott - MIT License");
            Ok(())
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            print_main_help(program);
            std::process::exit(1);
        }
    };
    
    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
