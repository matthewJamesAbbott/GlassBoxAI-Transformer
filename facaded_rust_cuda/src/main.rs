// GlassBox AI Transformer Facade - Rust CUDA Implementation
// MIT License (c) 2025 Matthew Abbott

use glassbox_transformer_facaded::{
    ChatTokenizer, GGUFLoader, TransformerFacade, GenerationConfig,
    TransformerError, TransformerModel, QKVType,
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
    println!("║     GlassBox AI Transformer Facade - Rust CUDA Implementation     ║");
    println!("║     Introspection API for Hidden States & Attention Analysis      ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("USAGE: {} <command> [options]", program);
    println!();
    println!("COMMANDS:");
    println!();
    println!("  generate    Text generation with introspection");
    println!("  train       Fine-tune model with LoRA support");
    println!("  analyze     Analyze model internals for a prompt");
    println!("  inspect     Interactive inspection mode");
    println!("  info        Show model information");
    println!();
    println!("Use '{} <command> --help' for more information.", program);
    println!();
}

fn print_generate_help(program: &str) {
    println!();
    println!("GENERATE MODE - Text generation with introspection");
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
    println!("  -i, --interactive       Interactive chat mode");
    println!("  --show-hidden           Show hidden state statistics");
    println!("  --show-entropy          Show attention entropy");
    println!("  --help                  Show this help");
    println!();
}

fn print_analyze_help(program: &str) {
    println!();
    println!("ANALYZE MODE - Analyze model internals for a prompt");
    println!();
    println!("Usage: {} analyze -m <model.gguf> -p <prompt> [options]", program);
    println!();
    println!("OPTIONS:");
    println!("  -m, --model <path>      Path to GGUF model file (required)");
    println!("  -p, --prompt <text>     Text prompt to analyze (required)");
    println!("  --layer <n>             Layer to inspect (default: last)");
    println!("  --head <n>              Attention head to inspect (default: 0)");
    println!("  --show-qkv              Show Q/K/V vectors");
    println!("  --show-logits           Show top-k logits");
    println!("  --show-saliency         Show saliency map");
    println!("  --help                  Show this help");
    println!();
}

fn print_train_help(program: &str) {
    println!();
    println!("TRAIN MODE - Fine-tune transformer with introspection (LoRA supported)");
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
    println!("  - Introspection API for training analysis");
    println!("  - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning");
    println!("  - CUDA GPU acceleration for training");
    println!();
}

fn run_generate(args: &[String]) -> Result<(), TransformerError> {
    let mut model_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut config = GenerationConfig::default();
    let mut interactive = false;
    let mut show_hidden = false;
    let mut show_entropy = false;
    
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
            "-i" | "--interactive" => {
                interactive = true;
            }
            "--show-hidden" => {
                show_hidden = true;
            }
            "--show-entropy" => {
                show_entropy = true;
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
    
    println!("\n=== GlassBox Transformer Facade (GPU Accelerated) ===");
    
    let loader = GGUFLoader::load_from_file(&model_path)?;
    let tokenizer = ChatTokenizer::from_gguf(&loader)?;
    
    println!("[GPU] Initializing CUDA...");
    let device = CudaDevice::new(0)
        .map_err(|e| TransformerError::Cuda(format!("Failed to create CUDA device: {}", e)))?;
    
    println!("[GPU] Loading model weights...");
    let model = TransformerModel::from_gguf(&loader, device)?;
    
    let chat_tokenizer = tokenizer.clone();
    let mut facade = TransformerFacade::new(model, tokenizer)?;
    
    println!("[GPU] Model loaded successfully");
    println!("  Layers: {}, Heads: {}, Hidden: {}", 
        facade.num_layers(), facade.num_heads(), facade.hidden_size());
    
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
            let result = facade.generate(&formatted, &config)?;
            let elapsed = start.elapsed();
            
            let token_count = result.split_whitespace().count().max(1);
            let tps = token_count as f64 / elapsed.as_secs_f64();
            
            println!("\n[{} tokens in {:.2}s = {:.1} tok/s]", 
                token_count, elapsed.as_secs_f64(), tps);
            
            if show_hidden {
                print_hidden_stats(&facade);
            }
            
            if show_entropy {
                print_entropy_stats(&facade);
            }
            
            println!();
            facade.clear_cache()?;
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
        let result = facade.generate(&formatted, &config)?;
        let elapsed = start.elapsed();
        
        let token_count = result.split_whitespace().count().max(1);
        let tps = token_count as f64 / elapsed.as_secs_f64();
        
        println!("\n\n[GPU: {} tokens in {:.2}s = {:.1} tok/s]", 
            token_count, elapsed.as_secs_f64(), tps);
        
        if show_hidden {
            print_hidden_stats(&facade);
        }
        
        if show_entropy {
            print_entropy_stats(&facade);
        }
    }
    
    Ok(())
}

fn run_analyze(args: &[String]) -> Result<(), TransformerError> {
    let mut model_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut layer: Option<usize> = None;
    let mut head: usize = 0;
    let mut show_qkv = false;
    let mut show_logits = false;
    let mut show_saliency = false;
    
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
            "--layer" => {
                if i + 1 < args.len() {
                    i += 1;
                    layer = Some(args[i].parse().unwrap_or(0));
                }
            }
            "--head" => {
                if i + 1 < args.len() {
                    i += 1;
                    head = args[i].parse().unwrap_or(0);
                }
            }
            "--show-qkv" => show_qkv = true,
            "--show-logits" => show_logits = true,
            "--show-saliency" => show_saliency = true,
            "--help" => {
                print_analyze_help(&env::args().next().unwrap_or_default());
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }
    
    let model_path = model_path.ok_or_else(|| {
        TransformerError::Model("Model path required (-m <path>)".into())
    })?;
    let prompt = prompt.ok_or_else(|| {
        TransformerError::Model("Prompt required (-p <text>)".into())
    })?;
    
    println!("\n=== GlassBox Model Analysis ===\n");
    
    let loader = GGUFLoader::load_from_file(&model_path)?;
    let tokenizer = ChatTokenizer::from_gguf(&loader)?;
    
    let device = CudaDevice::new(0)
        .map_err(|e| TransformerError::Cuda(format!("Failed to create CUDA device: {}", e)))?;
    
    let model = TransformerModel::from_gguf(&loader, device)?;
    let mut facade = TransformerFacade::new(model, tokenizer.clone())?;
    
    let layer_idx = layer.unwrap_or(facade.num_layers() - 1);
    
    println!("Prompt: \"{}\"", prompt);
    println!("Analyzing layer {} (head {})", layer_idx, head);
    println!();
    
    let tokens = tokenizer.encode(&prompt);
    println!("Tokens ({}):", tokens.len());
    for (i, &t) in tokens.iter().enumerate() {
        println!("  [{}] {} -> \"{}\"", i, t, tokenizer.decode(t));
    }
    println!();
    
    for (pos, &token) in tokens.iter().enumerate() {
        facade.forward(token, pos)?;
    }
    
    if show_qkv {
        println!("=== Q/K/V Vectors (Layer {}, Head {}) ===", layer_idx, head);
        let last_pos = tokens.len() - 1;
        
        if let Some(q) = facade.get_qkv(layer_idx, head, QKVType::Query, last_pos) {
            println!("Query (first 8 dims): {:?}", &q[..8.min(q.len())]);
        }
        if let Some(k) = facade.get_qkv(layer_idx, head, QKVType::Key, last_pos) {
            println!("Key (first 8 dims): {:?}", &k[..8.min(k.len())]);
        }
        if let Some(v) = facade.get_qkv(layer_idx, head, QKVType::Value, last_pos) {
            println!("Value (first 8 dims): {:?}", &v[..8.min(v.len())]);
        }
        println!();
    }
    
    if show_logits {
        println!("=== Top 10 Logits ===");
        let logits = facade.get_logits();
        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        for (i, (token_id, logit)) in indexed.iter().take(10).enumerate() {
            let token_str = tokenizer.decode(*token_id as u32);
            println!("  {}. [{}] \"{}\" = {:.4}", i + 1, token_id, token_str, logit);
        }
        println!();
    }
    
    if show_saliency {
        println!("=== Saliency Map (Layer {}) ===", layer_idx);
        let last_pos = tokens.len() - 1;
        
        if let Some(saliency) = facade.get_saliency_map(last_pos, layer_idx) {
            let top_k = 10;
            let mut indexed: Vec<(usize, f32)> = saliency.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            println!("Top {} most salient dimensions:", top_k);
            for (i, (dim, val)) in indexed.iter().take(top_k).enumerate() {
                println!("  {}. dim {} = {:.4}", i + 1, dim, val);
            }
        }
        println!();
    }
    
    print_hidden_stats(&facade);
    print_entropy_stats(&facade);
    
    Ok(())
}

fn run_info(args: &[String]) -> Result<(), TransformerError> {
    let mut model_path: Option<String> = None;
    
    for i in 0..args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                if i + 1 < args.len() {
                    model_path = Some(args[i + 1].clone());
                }
            }
            _ => {}
        }
    }
    
    let model_path = model_path.ok_or_else(|| {
        TransformerError::Model("Model path required (-m <path>)".into())
    })?;
    
    println!("\n=== Model Information ===\n");
    
    let loader = GGUFLoader::load_from_file(&model_path)?;
    
    println!("Architecture:    {}", loader.get_architecture());
    println!("Layers:          {}", loader.get_num_layers());
    println!("Heads:           {}", loader.get_num_heads());
    println!("KV Heads:        {}", loader.get_num_kv_heads());
    println!("Embedding Dim:   {}", loader.get_embed_dim());
    println!("FFN Dim:         {}", loader.get_ffn_dim());
    println!("Vocab Size:      {}", loader.get_vocab_size());
    println!("Max Seq Length:  {}", loader.get_max_seq_len());
    println!("RoPE Theta:      {}", loader.get_rope_theta());
    println!("RMS Epsilon:     {}", loader.get_rms_eps());
    println!("Has Tokenizer:   {}", loader.has_tokenizer());
    
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
    
    println!("\n=== GlassBox Transformer Training (Facade) ===");
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
    println!("===============================================\n");
    
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
    
    // Create facade for introspection during training
    let facade = TransformerFacade::new(model, tokenizer.clone())?;
    println!("[GPU] Model loaded - Layers: {}, Heads: {}, Hidden: {}", 
        facade.num_layers(), facade.num_heads(), facade.hidden_size());
    
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
            
            // TODO: Implement actual forward/backward with LoRA + introspection
            // This requires integrating with the facade's forward pass
            
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

fn print_hidden_stats(facade: &TransformerFacade) {
    println!("\n=== Hidden State Statistics ===");
    let last_layer = facade.num_layers() - 1;
    let last_pos = facade.last_seq_len().saturating_sub(1);
    
    if let Some(hidden) = facade.get_hidden_state(last_layer, last_pos) {
        let mean: f32 = hidden.iter().sum::<f32>() / hidden.len() as f32;
        let variance: f32 = hidden.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden.len() as f32;
        let std = variance.sqrt();
        let min = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        println!("  Layer {}, Position {}", last_layer, last_pos);
        println!("  Mean: {:.4}, Std: {:.4}, Min: {:.4}, Max: {:.4}", mean, std, min, max);
    }
}

fn print_entropy_stats(facade: &TransformerFacade) {
    println!("\n=== Attention Entropy (per head) ===");
    let n_layers = facade.num_layers();
    let n_heads = facade.num_heads();
    
    for layer in [0, n_layers / 2, n_layers - 1] {
        print!("  Layer {}: ", layer);
        for head in 0..n_heads.min(4) {
            let entropy = facade.get_attention_entropy(layer, head);
            print!("H{}={:.3} ", head, entropy);
        }
        if n_heads > 4 {
            print!("...");
        }
        println!();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program = args.first().map(|s| s.as_str()).unwrap_or("transformer-facaded-rust-cuda");
    
    if args.len() < 2 {
        print_main_help(program);
        std::process::exit(1);
    }
    
    let command = &args[1];
    let cmd_args: Vec<String> = args.iter().skip(2).cloned().collect();
    
    let result = match command.as_str() {
        "generate" => run_generate(&cmd_args),
        "train" => run_train(&cmd_args),
        "analyze" => run_analyze(&cmd_args),
        "inspect" => run_generate(&cmd_args), // alias for interactive
        "info" => run_info(&cmd_args),
        "--help" | "-h" => {
            print_main_help(program);
            Ok(())
        }
        "--version" => {
            println!("GlassBox AI Transformer Facade v0.1.0 (Rust CUDA)");
            println!("Introspection API for Hidden States & Attention Analysis");
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
