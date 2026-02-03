// LoRA (Low-Rank Adaptation) for Rust CUDA Transformer
// MIT License (c) 2025 Matthew Abbott
//
// Implements parameter-efficient fine-tuning via low-rank matrix decomposition.
// W' = W + B @ A * scaling, where A: (rank x in_dim), B: (out_dim x rank)

use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;
use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use rand::Rng;
use rand::distributions::Uniform;

use crate::error::{Result, TransformerError};

/// LoRA configuration
#[derive(Clone, Debug)]
pub struct LoRAConfig {
    /// Low-rank dimension (r)
    pub rank: usize,
    /// Scaling factor, effective scale = alpha/rank
    pub alpha: f32,
    /// Dropout between A and B matrices
    pub dropout: f32,
    /// Apply LoRA to attention Q projection
    pub enable_q: bool,
    /// Apply LoRA to attention K projection
    pub enable_k: bool,
    /// Apply LoRA to attention V projection
    pub enable_v: bool,
    /// Apply LoRA to attention output projection
    pub enable_o: bool,
    /// Apply LoRA to FFN gate (w1)
    pub enable_gate: bool,
    /// Apply LoRA to FFN up (w3)
    pub enable_up: bool,
    /// Apply LoRA to FFN down (w2)
    pub enable_down: bool,
    /// Freeze base weights, only train LoRA
    pub freeze_base: bool,
    /// Adapter name for versioning
    pub name: String,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            enable_q: true,
            enable_k: true,
            enable_v: true,
            enable_o: true,
            enable_gate: true,
            enable_up: true,
            enable_down: true,
            freeze_base: true,
            name: "lora".to_string(),
        }
    }
}

impl LoRAConfig {
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
    
    /// Parse target layers from comma-separated string
    pub fn parse_layers(mut self, layers_str: &str) -> Self {
        self.enable_q = false;
        self.enable_k = false;
        self.enable_v = false;
        self.enable_o = false;
        self.enable_gate = false;
        self.enable_up = false;
        self.enable_down = false;
        
        for layer in layers_str.split(',') {
            match layer.trim().to_lowercase().as_str() {
                "q" => self.enable_q = true,
                "k" => self.enable_k = true,
                "v" => self.enable_v = true,
                "o" => self.enable_o = true,
                "gate" => self.enable_gate = true,
                "up" => self.enable_up = true,
                "down" => self.enable_down = true,
                _ => {}
            }
        }
        self
    }
}

/// LoRA adapter weights for a single projection
/// W' = W + B @ A * scaling, where A: (rank x in_dim), B: (out_dim x rank)
pub struct LoRAAdapter {
    /// A matrix on GPU: (rank x in_dim)
    pub a: Option<CudaSlice<f32>>,
    /// B matrix on GPU: (out_dim x rank)
    pub b: Option<CudaSlice<f32>>,
    /// Gradient for A
    pub d_a: Option<CudaSlice<f32>>,
    /// Gradient for B
    pub d_b: Option<CudaSlice<f32>>,
    /// Adam first moment for A
    pub m_a: Option<CudaSlice<f32>>,
    /// Adam second moment for A
    pub v_a: Option<CudaSlice<f32>>,
    /// Adam first moment for B
    pub m_b: Option<CudaSlice<f32>>,
    /// Adam second moment for B
    pub v_b: Option<CudaSlice<f32>>,
    /// Input dimension
    pub in_dim: usize,
    /// Output dimension
    pub out_dim: usize,
    /// Rank
    pub rank: usize,
    /// Whether this adapter is enabled
    pub enabled: bool,
}

impl Default for LoRAAdapter {
    fn default() -> Self {
        Self {
            a: None,
            b: None,
            d_a: None,
            d_b: None,
            m_a: None,
            v_a: None,
            m_b: None,
            v_b: None,
            in_dim: 0,
            out_dim: 0,
            rank: 0,
            enabled: false,
        }
    }
}

/// Per-layer LoRA adapters for all projections
#[derive(Default)]
pub struct LayerLoRA {
    /// Attention Q: (dim -> q_dim)
    pub q: LoRAAdapter,
    /// Attention K: (dim -> kv_dim)
    pub k: LoRAAdapter,
    /// Attention V: (dim -> kv_dim)
    pub v: LoRAAdapter,
    /// Attention O: (q_dim -> dim)
    pub o: LoRAAdapter,
    /// FFN gate/w1: (dim -> ffn_dim)
    pub gate: LoRAAdapter,
    /// FFN up/w3: (dim -> ffn_dim)
    pub up: LoRAAdapter,
    /// FFN down/w2: (ffn_dim -> dim)
    pub down: LoRAAdapter,
}

/// LoRA trainer for GPU-accelerated low-rank adaptation
pub struct LoRATrainer {
    device: Arc<CudaDevice>,
    config: LoRAConfig,
    
    // Model dimensions
    dim: usize,
    n_layers: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,
    
    // Per-layer LoRA adapters
    layer_lora: Vec<LayerLoRA>,
    
    // Temp buffers for forward/backward
    lora_temp: Option<CudaSlice<f32>>,
    lora_d_temp: Option<CudaSlice<f32>>,
    
    // State
    initialized: bool,
    adam_timestep: i32,
    dropout_seed: u64,
}

impl LoRATrainer {
    pub fn new(
        device: Arc<CudaDevice>,
        dim: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        ffn_dim: usize,
    ) -> Self {
        let head_dim = dim / n_heads;
        let q_dim = dim;
        let kv_dim = head_dim * n_kv_heads;
        
        Self {
            device,
            config: LoRAConfig::default(),
            dim,
            n_layers,
            q_dim,
            kv_dim,
            ffn_dim,
            layer_lora: Vec::new(),
            lora_temp: None,
            lora_d_temp: None,
            initialized: false,
            adam_timestep: 0,
            dropout_seed: 42,
        }
    }
    
    /// Initialize LoRA adapters with given configuration
    pub fn initialize(&mut self, config: LoRAConfig) -> Result<()> {
        self.cleanup();
        self.config = config.clone();
        
        println!("[LoRA] Initializing adapters...");
        println!("[LoRA] Config: rank={}, alpha={}, dropout={}, scaling={:.4}",
            config.rank, config.alpha, config.dropout, config.scaling());
        
        // Allocate temp buffers
        self.lora_temp = Some(self.device.alloc_zeros::<f32>(config.rank)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc lora_temp: {}", e)))?);
        self.lora_d_temp = Some(self.device.alloc_zeros::<f32>(config.rank)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc lora_d_temp: {}", e)))?);
        
        // Initialize per-layer adapters
        self.layer_lora = Vec::with_capacity(self.n_layers);
        let mut rng = rand::thread_rng();
        let mut total_params: usize = 0;
        
        for _ in 0..self.n_layers {
            let mut layer = LayerLoRA::default();
            
            if config.enable_q {
                self.init_adapter(&mut layer.q, self.dim, self.q_dim, config.rank, &mut rng)?;
                total_params += config.rank * self.dim + self.q_dim * config.rank;
            }
            if config.enable_k {
                self.init_adapter(&mut layer.k, self.dim, self.kv_dim, config.rank, &mut rng)?;
                total_params += config.rank * self.dim + self.kv_dim * config.rank;
            }
            if config.enable_v {
                self.init_adapter(&mut layer.v, self.dim, self.kv_dim, config.rank, &mut rng)?;
                total_params += config.rank * self.dim + self.kv_dim * config.rank;
            }
            if config.enable_o {
                self.init_adapter(&mut layer.o, self.q_dim, self.dim, config.rank, &mut rng)?;
                total_params += config.rank * self.q_dim + self.dim * config.rank;
            }
            if config.enable_gate {
                self.init_adapter(&mut layer.gate, self.dim, self.ffn_dim, config.rank, &mut rng)?;
                total_params += config.rank * self.dim + self.ffn_dim * config.rank;
            }
            if config.enable_up {
                self.init_adapter(&mut layer.up, self.dim, self.ffn_dim, config.rank, &mut rng)?;
                total_params += config.rank * self.dim + self.ffn_dim * config.rank;
            }
            if config.enable_down {
                self.init_adapter(&mut layer.down, self.ffn_dim, self.dim, config.rank, &mut rng)?;
                total_params += config.rank * self.ffn_dim + self.dim * config.rank;
            }
            
            self.layer_lora.push(layer);
        }
        
        self.initialized = true;
        self.dropout_seed = rng.gen();
        
        println!("[LoRA] Initialized {} trainable parameters ({:.2} MB)",
            total_params, (total_params * 4) as f64 / 1024.0 / 1024.0);
        println!("[LoRA] Base model frozen: {}", if config.freeze_base { "yes" } else { "no" });
        
        Ok(())
    }
    
    fn init_adapter(
        &self,
        adapter: &mut LoRAAdapter,
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        rng: &mut impl Rng,
    ) -> Result<()> {
        adapter.in_dim = in_dim;
        adapter.out_dim = out_dim;
        adapter.rank = rank;
        adapter.enabled = true;
        
        let a_size = rank * in_dim;
        let b_size = out_dim * rank;
        
        // Initialize A with small random values (Kaiming-style)
        let dist = Uniform::new(-0.01f32, 0.01f32);
        let a_data: Vec<f32> = (0..a_size).map(|_| rng.sample(dist)).collect();
        adapter.a = Some(self.device.htod_sync_copy(&a_data)
            .map_err(|e| TransformerError::Cuda(format!("Failed to copy A: {}", e)))?);
        
        // Initialize B to zeros (so initial delta = 0)
        adapter.b = Some(self.device.alloc_zeros::<f32>(b_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc B: {}", e)))?);
        
        // Allocate gradients
        adapter.d_a = Some(self.device.alloc_zeros::<f32>(a_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc dA: {}", e)))?);
        adapter.d_b = Some(self.device.alloc_zeros::<f32>(b_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc dB: {}", e)))?);
        
        // Allocate Adam state
        adapter.m_a = Some(self.device.alloc_zeros::<f32>(a_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc mA: {}", e)))?);
        adapter.v_a = Some(self.device.alloc_zeros::<f32>(a_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc vA: {}", e)))?);
        adapter.m_b = Some(self.device.alloc_zeros::<f32>(b_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc mB: {}", e)))?);
        adapter.v_b = Some(self.device.alloc_zeros::<f32>(b_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc vB: {}", e)))?);
        
        Ok(())
    }
    
    /// Zero all LoRA gradients
    pub fn zero_gradients(&mut self) -> Result<()> {
        for layer in &mut self.layer_lora {
            Self::zero_adapter_gradients_static(&self.device, &mut layer.q)?;
            Self::zero_adapter_gradients_static(&self.device, &mut layer.k)?;
            Self::zero_adapter_gradients_static(&self.device, &mut layer.v)?;
            Self::zero_adapter_gradients_static(&self.device, &mut layer.o)?;
            Self::zero_adapter_gradients_static(&self.device, &mut layer.gate)?;
            Self::zero_adapter_gradients_static(&self.device, &mut layer.up)?;
            Self::zero_adapter_gradients_static(&self.device, &mut layer.down)?;
        }
        Ok(())
    }
    
    fn zero_adapter_gradients_static(device: &Arc<CudaDevice>, adapter: &mut LoRAAdapter) -> Result<()> {
        if !adapter.enabled { return Ok(()); }
        
        let a_size = adapter.rank * adapter.in_dim;
        let b_size = adapter.out_dim * adapter.rank;
        
        if let Some(ref mut d_a) = adapter.d_a {
            let zeros = vec![0.0f32; a_size];
            device.htod_sync_copy_into(&zeros, d_a)
                .map_err(|e| TransformerError::Cuda(format!("Failed to zero dA: {}", e)))?;
        }
        if let Some(ref mut d_b) = adapter.d_b {
            let zeros = vec![0.0f32; b_size];
            device.htod_sync_copy_into(&zeros, d_b)
                .map_err(|e| TransformerError::Cuda(format!("Failed to zero dB: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Save LoRA weights to file
    pub fn save(&self, path: &str) -> Result<()> {
        if !self.initialized {
            return Err(TransformerError::Model("[LoRA] Not initialized".into()));
        }
        
        let file = File::create(path)
            .map_err(|e| TransformerError::Io(e))?;
        let mut writer = BufWriter::new(file);
        
        // Magic and version
        writer.write_all(b"LORA").map_err(TransformerError::Io)?;
        let version: i32 = 1;
        writer.write_all(&version.to_le_bytes()).map_err(TransformerError::Io)?;
        
        // Config
        writer.write_all(&(self.config.rank as i32).to_le_bytes()).map_err(TransformerError::Io)?;
        writer.write_all(&self.config.alpha.to_le_bytes()).map_err(TransformerError::Io)?;
        writer.write_all(&self.config.dropout.to_le_bytes()).map_err(TransformerError::Io)?;
        
        // Dimensions
        writer.write_all(&(self.n_layers as i32).to_le_bytes()).map_err(TransformerError::Io)?;
        writer.write_all(&(self.dim as i32).to_le_bytes()).map_err(TransformerError::Io)?;
        writer.write_all(&(self.q_dim as i32).to_le_bytes()).map_err(TransformerError::Io)?;
        writer.write_all(&(self.kv_dim as i32).to_le_bytes()).map_err(TransformerError::Io)?;
        writer.write_all(&(self.ffn_dim as i32).to_le_bytes()).map_err(TransformerError::Io)?;
        
        // Flags
        let mut flags: u8 = 0;
        if self.config.enable_q { flags |= 0x01; }
        if self.config.enable_k { flags |= 0x02; }
        if self.config.enable_v { flags |= 0x04; }
        if self.config.enable_o { flags |= 0x08; }
        if self.config.enable_gate { flags |= 0x10; }
        if self.config.enable_up { flags |= 0x20; }
        if self.config.enable_down { flags |= 0x40; }
        writer.write_all(&[flags]).map_err(TransformerError::Io)?;
        
        // Name
        let name_bytes = self.config.name.as_bytes();
        writer.write_all(&(name_bytes.len() as u64).to_le_bytes()).map_err(TransformerError::Io)?;
        writer.write_all(name_bytes).map_err(TransformerError::Io)?;
        
        // Layer weights
        for layer in &self.layer_lora {
            self.save_adapter(&mut writer, &layer.q)?;
            self.save_adapter(&mut writer, &layer.k)?;
            self.save_adapter(&mut writer, &layer.v)?;
            self.save_adapter(&mut writer, &layer.o)?;
            self.save_adapter(&mut writer, &layer.gate)?;
            self.save_adapter(&mut writer, &layer.up)?;
            self.save_adapter(&mut writer, &layer.down)?;
        }
        
        writer.flush().map_err(TransformerError::Io)?;
        println!("[LoRA] Saved to: {}", path);
        Ok(())
    }
    
    fn save_adapter<W: Write>(&self, writer: &mut W, adapter: &LoRAAdapter) -> Result<()> {
        if !adapter.enabled { return Ok(()); }
        
        let a_size = adapter.rank * adapter.in_dim;
        let b_size = adapter.out_dim * adapter.rank;
        
        if let Some(ref a) = adapter.a {
            let host_a = self.device.dtoh_sync_copy(a)
                .map_err(|e| TransformerError::Cuda(format!("Failed to copy A: {}", e)))?;
            for val in &host_a {
                writer.write_all(&val.to_le_bytes()).map_err(TransformerError::Io)?;
            }
        }
        if let Some(ref b) = adapter.b {
            let host_b = self.device.dtoh_sync_copy(b)
                .map_err(|e| TransformerError::Cuda(format!("Failed to copy B: {}", e)))?;
            for val in &host_b {
                writer.write_all(&val.to_le_bytes()).map_err(TransformerError::Io)?;
            }
        }
        
        Ok(())
    }
    
    /// Load LoRA weights from file
    pub fn load(&mut self, path: &str) -> Result<()> {
        let file = File::open(path)
            .map_err(|e| TransformerError::Io(e))?;
        let mut reader = BufReader::new(file);
        
        // Magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(TransformerError::Io)?;
        if &magic != b"LORA" {
            return Err(TransformerError::Model("[LoRA] Invalid file format".into()));
        }
        
        // Version
        let mut buf_i32 = [0u8; 4];
        reader.read_exact(&mut buf_i32).map_err(TransformerError::Io)?;
        let _version = i32::from_le_bytes(buf_i32);
        
        // Config
        reader.read_exact(&mut buf_i32).map_err(TransformerError::Io)?;
        let rank = i32::from_le_bytes(buf_i32) as usize;
        
        let mut buf_f32 = [0u8; 4];
        reader.read_exact(&mut buf_f32).map_err(TransformerError::Io)?;
        let alpha = f32::from_le_bytes(buf_f32);
        
        reader.read_exact(&mut buf_f32).map_err(TransformerError::Io)?;
        let dropout = f32::from_le_bytes(buf_f32);
        
        // Dimensions
        reader.read_exact(&mut buf_i32).map_err(TransformerError::Io)?;
        let saved_layers = i32::from_le_bytes(buf_i32) as usize;
        
        reader.read_exact(&mut buf_i32).map_err(TransformerError::Io)?;
        let saved_dim = i32::from_le_bytes(buf_i32) as usize;
        
        reader.read_exact(&mut buf_i32).map_err(TransformerError::Io)?;
        let saved_q_dim = i32::from_le_bytes(buf_i32) as usize;
        
        reader.read_exact(&mut buf_i32).map_err(TransformerError::Io)?;
        let saved_kv_dim = i32::from_le_bytes(buf_i32) as usize;
        
        reader.read_exact(&mut buf_i32).map_err(TransformerError::Io)?;
        let saved_ffn_dim = i32::from_le_bytes(buf_i32) as usize;
        
        // Validate dimensions
        if saved_layers != self.n_layers || saved_dim != self.dim ||
           saved_q_dim != self.q_dim || saved_kv_dim != self.kv_dim ||
           saved_ffn_dim != self.ffn_dim {
            return Err(TransformerError::Model("[LoRA] Model dimensions mismatch".into()));
        }
        
        // Flags
        let mut flags = [0u8; 1];
        reader.read_exact(&mut flags).map_err(TransformerError::Io)?;
        let flags = flags[0];
        
        let config = LoRAConfig {
            rank,
            alpha,
            dropout,
            enable_q: (flags & 0x01) != 0,
            enable_k: (flags & 0x02) != 0,
            enable_v: (flags & 0x04) != 0,
            enable_o: (flags & 0x08) != 0,
            enable_gate: (flags & 0x10) != 0,
            enable_up: (flags & 0x20) != 0,
            enable_down: (flags & 0x40) != 0,
            freeze_base: true,
            name: String::new(),
        };
        
        // Name
        let mut buf_u64 = [0u8; 8];
        reader.read_exact(&mut buf_u64).map_err(TransformerError::Io)?;
        let name_len = u64::from_le_bytes(buf_u64) as usize;
        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes).map_err(TransformerError::Io)?;
        let name = String::from_utf8(name_bytes).unwrap_or_default();
        
        // Initialize with loaded config
        self.initialize(LoRAConfig { name: name.clone(), ..config })?;
        
        // Load weights
        for layer in &mut self.layer_lora {
            Self::load_adapter_static(&self.device, &mut reader, &mut layer.q)?;
            Self::load_adapter_static(&self.device, &mut reader, &mut layer.k)?;
            Self::load_adapter_static(&self.device, &mut reader, &mut layer.v)?;
            Self::load_adapter_static(&self.device, &mut reader, &mut layer.o)?;
            Self::load_adapter_static(&self.device, &mut reader, &mut layer.gate)?;
            Self::load_adapter_static(&self.device, &mut reader, &mut layer.up)?;
            Self::load_adapter_static(&self.device, &mut reader, &mut layer.down)?;
        }
        
        println!("[LoRA] Loaded from: {} (name: {})", path, name);
        Ok(())
    }
    
    fn load_adapter_static<R: Read>(device: &Arc<CudaDevice>, reader: &mut R, adapter: &mut LoRAAdapter) -> Result<()> {
        if !adapter.enabled { return Ok(()); }
        
        let a_size = adapter.rank * adapter.in_dim;
        let b_size = adapter.out_dim * adapter.rank;
        
        // Load A
        let mut host_a = vec![0.0f32; a_size];
        for val in &mut host_a {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf).map_err(TransformerError::Io)?;
            *val = f32::from_le_bytes(buf);
        }
        if let Some(ref mut a) = adapter.a {
            device.htod_sync_copy_into(&host_a, a)
                .map_err(|e| TransformerError::Cuda(format!("Failed to copy A: {}", e)))?;
        }
        
        // Load B
        let mut host_b = vec![0.0f32; b_size];
        for val in &mut host_b {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf).map_err(TransformerError::Io)?;
            *val = f32::from_le_bytes(buf);
        }
        if let Some(ref mut b) = adapter.b {
            device.htod_sync_copy_into(&host_b, b)
                .map_err(|e| TransformerError::Cuda(format!("Failed to copy B: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Cleanup all LoRA resources
    pub fn cleanup(&mut self) {
        self.layer_lora.clear();
        self.lora_temp = None;
        self.lora_d_temp = None;
        self.initialized = false;
    }
    
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    pub fn config(&self) -> &LoRAConfig {
        &self.config
    }
    
    pub fn layer_lora(&self) -> &[LayerLoRA] {
        &self.layer_lora
    }
    
    pub fn layer_lora_mut(&mut self) -> &mut [LayerLoRA] {
        &mut self.layer_lora
    }
    
    /// Get scaling factor
    pub fn scaling(&self) -> f32 {
        self.config.scaling()
    }
    
    /// Increment Adam timestep
    pub fn step(&mut self) {
        self.adam_timestep += 1;
    }
    
    pub fn adam_timestep(&self) -> i32 {
        self.adam_timestep
    }
}

impl Drop for LoRATrainer {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// =============================================================================
// Unit Tests (run with `cargo test`)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoRAConfig::default();
        assert_eq!(config.rank, 16);
        assert_eq!(config.alpha, 32.0);
        assert_eq!(config.dropout, 0.05);
        assert!(config.enable_q);
        assert!(config.enable_k);
        assert!(config.enable_v);
        assert!(config.enable_o);
        assert!(config.enable_gate);
        assert!(config.enable_up);
        assert!(config.enable_down);
        assert!(config.freeze_base);
        assert_eq!(config.name, "lora");
    }

    #[test]
    fn test_lora_config_scaling() {
        let config = LoRAConfig {
            rank: 16,
            alpha: 32.0,
            ..Default::default()
        };
        assert_eq!(config.scaling(), 2.0);

        let config2 = LoRAConfig {
            rank: 8,
            alpha: 16.0,
            ..Default::default()
        };
        assert_eq!(config2.scaling(), 2.0);

        let config3 = LoRAConfig {
            rank: 32,
            alpha: 32.0,
            ..Default::default()
        };
        assert_eq!(config3.scaling(), 1.0);
    }

    #[test]
    fn test_lora_config_parse_layers_q_only() {
        let config = LoRAConfig::default().parse_layers("q");
        assert!(config.enable_q);
        assert!(!config.enable_k);
        assert!(!config.enable_v);
        assert!(!config.enable_o);
        assert!(!config.enable_gate);
        assert!(!config.enable_up);
        assert!(!config.enable_down);
    }

    #[test]
    fn test_lora_config_parse_layers_attention_only() {
        let config = LoRAConfig::default().parse_layers("q,k,v,o");
        assert!(config.enable_q);
        assert!(config.enable_k);
        assert!(config.enable_v);
        assert!(config.enable_o);
        assert!(!config.enable_gate);
        assert!(!config.enable_up);
        assert!(!config.enable_down);
    }

    #[test]
    fn test_lora_config_parse_layers_ffn_only() {
        let config = LoRAConfig::default().parse_layers("gate,up,down");
        assert!(!config.enable_q);
        assert!(!config.enable_k);
        assert!(!config.enable_v);
        assert!(!config.enable_o);
        assert!(config.enable_gate);
        assert!(config.enable_up);
        assert!(config.enable_down);
    }

    #[test]
    fn test_lora_config_parse_layers_mixed() {
        let config = LoRAConfig::default().parse_layers("q,v,gate,down");
        assert!(config.enable_q);
        assert!(!config.enable_k);
        assert!(config.enable_v);
        assert!(!config.enable_o);
        assert!(config.enable_gate);
        assert!(!config.enable_up);
        assert!(config.enable_down);
    }

    #[test]
    fn test_lora_config_parse_layers_empty() {
        let config = LoRAConfig::default().parse_layers("");
        assert!(!config.enable_q);
        assert!(!config.enable_k);
        assert!(!config.enable_v);
        assert!(!config.enable_o);
        assert!(!config.enable_gate);
        assert!(!config.enable_up);
        assert!(!config.enable_down);
    }

    #[test]
    fn test_lora_adapter_default() {
        let adapter = LoRAAdapter::default();
        assert!(adapter.a.is_none());
        assert!(adapter.b.is_none());
        assert_eq!(adapter.in_dim, 0);
        assert_eq!(adapter.out_dim, 0);
        assert_eq!(adapter.rank, 0);
        assert!(!adapter.enabled);
    }

    #[test]
    fn test_layer_lora_default() {
        let layer = LayerLoRA::default();
        assert!(!layer.q.enabled);
        assert!(!layer.k.enabled);
        assert!(!layer.v.enabled);
        assert!(!layer.o.enabled);
        assert!(!layer.gate.enabled);
        assert!(!layer.up.enabled);
        assert!(!layer.down.enabled);
    }

    #[test]
    fn test_lora_scaling_values() {
        let cases = [
            (8, 8.0, 1.0),
            (16, 32.0, 2.0),
            (4, 16.0, 4.0),
            (64, 32.0, 0.5),
            (1, 1.0, 1.0),
        ];

        for (rank, alpha, expected) in cases {
            let config = LoRAConfig {
                rank,
                alpha,
                ..Default::default()
            };
            let scaling = config.scaling();
            assert!(
                (scaling - expected).abs() < 1e-6,
                "rank={}, alpha={}: expected {}, got {}",
                rank, alpha, expected, scaling
            );
        }
    }

    #[test]
    fn test_lora_parameter_efficiency() {
        let rank: usize = 16;
        let dim: usize = 4096;
        let ffn_dim: usize = 11008;

        let full_attn_weight = dim * dim;
        let lora_attn_params = 2 * rank * dim;
        assert!(lora_attn_params < full_attn_weight / 100);

        let full_ffn_weight = dim * ffn_dim;
        let lora_ffn_params = rank * (dim + ffn_dim);
        assert!(lora_ffn_params < full_ffn_weight / 100);
    }

    #[test]
    fn test_lora_flags_encoding() {
        let config = LoRAConfig {
            enable_q: true,
            enable_k: false,
            enable_v: true,
            enable_o: false,
            enable_gate: true,
            enable_up: false,
            enable_down: true,
            ..Default::default()
        };

        let mut flags: u8 = 0;
        if config.enable_q { flags |= 0x01; }
        if config.enable_k { flags |= 0x02; }
        if config.enable_v { flags |= 0x04; }
        if config.enable_o { flags |= 0x08; }
        if config.enable_gate { flags |= 0x10; }
        if config.enable_up { flags |= 0x20; }
        if config.enable_down { flags |= 0x40; }

        assert_eq!(flags, 0x55);
    }

    #[test]
    fn test_lora_adapter_sizes() {
        let configs = [
            (16, 4096, 4096),
            (16, 4096, 11008),
            (16, 11008, 4096),
            (8, 2048, 5632),
            (32, 8192, 28672),
        ];

        for (rank, in_dim, out_dim) in configs {
            let a_size = rank * in_dim;
            let b_size = out_dim * rank;
            let total = a_size + b_size;
            let full = in_dim * out_dim;
            
            assert!(a_size > 0);
            assert!(b_size > 0);
            assert!(total < full, "LoRA should be smaller than full");
        }
    }
}
