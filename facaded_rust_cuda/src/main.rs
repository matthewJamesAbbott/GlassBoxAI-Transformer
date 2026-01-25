// GlassBox AI Transformer Facade - Rust CUDA Implementation
// MIT License (c) 2025 Matthew Abbott

use glassbox_transformer_facaded::{
    ChatTokenizer, GGUFLoader, TransformerFacade, GenerationConfig,
    TransformerError, TransformerModel, QKVType,
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
