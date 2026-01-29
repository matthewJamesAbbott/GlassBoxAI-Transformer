// GPU Trainer for backpropagation training using CUDA
// Implements training with Adam optimizer following facaded_transformer.cu's GPUTrainer

use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;
use crate::error::{Result, TransformerError};
use crate::kernels::CudaKernels;

/// Training configuration
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub adam_eps: f32,
    pub gradient_clip_norm: f32,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub use_gradient_checkpointing: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            adam_eps: 1e-8,
            gradient_clip_norm: 1.0,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            use_gradient_checkpointing: false,
        }
    }
}

/// Per-layer gradients stored on GPU
pub struct LayerGradients {
    pub d_attn_norm: CudaSlice<f32>,
    pub d_ffn_norm: CudaSlice<f32>,
    pub d_wq: CudaSlice<f32>,
    pub d_wk: CudaSlice<f32>,
    pub d_wv: CudaSlice<f32>,
    pub d_wo: CudaSlice<f32>,
    pub d_w1: CudaSlice<f32>,
    pub d_w2: CudaSlice<f32>,
    pub d_w3: CudaSlice<f32>,
}

/// Adam optimizer state (m and v) for a layer
pub struct LayerAdamState {
    pub m_wq: CudaSlice<f32>,
    pub v_wq: CudaSlice<f32>,
    pub m_wk: CudaSlice<f32>,
    pub v_wk: CudaSlice<f32>,
    pub m_wv: CudaSlice<f32>,
    pub v_wv: CudaSlice<f32>,
    pub m_wo: CudaSlice<f32>,
    pub v_wo: CudaSlice<f32>,
    pub m_w1: CudaSlice<f32>,
    pub v_w1: CudaSlice<f32>,
    pub m_w2: CudaSlice<f32>,
    pub v_w2: CudaSlice<f32>,
    pub m_w3: CudaSlice<f32>,
    pub v_w3: CudaSlice<f32>,
    pub m_attn_norm: CudaSlice<f32>,
    pub v_attn_norm: CudaSlice<f32>,
    pub m_ffn_norm: CudaSlice<f32>,
    pub v_ffn_norm: CudaSlice<f32>,
}

/// Cached activations for backward pass
pub struct ForwardActivations {
    pub pre_attn_norm: CudaSlice<f32>,
    pub post_attn_norm: CudaSlice<f32>,
    pub q: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub attn_output: CudaSlice<f32>,
    pub post_attn_residual: CudaSlice<f32>,
    pub pre_ffn_norm: CudaSlice<f32>,
    pub post_ffn_norm: CudaSlice<f32>,
    pub gate: CudaSlice<f32>,
    pub up: CudaSlice<f32>,
    pub ffn_hidden: CudaSlice<f32>,
}

/// GPU-based trainer for transformer models
pub struct GpuTrainer {
    device: Arc<CudaDevice>,
    kernels: Arc<CudaKernels>,
    config: TrainingConfig,
    
    // Model dimensions
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ffn_dim: usize,
    vocab_size: usize,
    max_seq_len: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    eps: f32,
    theta: f32,
    rope_scale: f32,
    
    // Adam timestep
    adam_timestep: i32,
    
    // Gradient storage
    layer_gradients: Vec<LayerGradients>,
    layer_adam_state: Vec<LayerAdamState>,
    forward_cache: Vec<ForwardActivations>,
    
    // Global gradients
    d_embeddings: Option<CudaSlice<f32>>,
    d_output_weight: Option<CudaSlice<f32>>,
    d_norm_weight: Option<CudaSlice<f32>>,
    
    // Global Adam state
    m_embeddings: Option<CudaSlice<f32>>,
    v_embeddings: Option<CudaSlice<f32>>,
    m_output_weight: Option<CudaSlice<f32>>,
    v_output_weight: Option<CudaSlice<f32>>,
    m_norm_weight: Option<CudaSlice<f32>>,
    v_norm_weight: Option<CudaSlice<f32>>,
    
    // Working buffers
    d_hidden: CudaSlice<f32>,
    d_xb: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    d_k: CudaSlice<f32>,
    d_v: CudaSlice<f32>,
    d_attn_out: CudaSlice<f32>,
    d_hb: CudaSlice<f32>,
    d_hb2: CudaSlice<f32>,
    d_logits: CudaSlice<f32>,
    
    // Gradient working buffers
    d_d_hidden: CudaSlice<f32>,
    d_d_xb: CudaSlice<f32>,
    d_d_q: CudaSlice<f32>,
    d_d_k: CudaSlice<f32>,
    d_d_v: CudaSlice<f32>,
    d_d_attn_out: CudaSlice<f32>,
    d_d_hb: CudaSlice<f32>,
    d_d_hb2: CudaSlice<f32>,
    d_d_logits: CudaSlice<f32>,
    
    // Training specific
    d_targets: CudaSlice<i32>,
    d_loss: CudaSlice<f32>,
    d_grad_norm: CudaSlice<f32>,
    
    initialized: bool,
}

impl GpuTrainer {
    pub fn new(
        device: Arc<CudaDevice>,
        kernels: Arc<CudaKernels>,
        config: TrainingConfig,
        dim: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        ffn_dim: usize,
        vocab_size: usize,
        max_seq_len: usize,
        eps: f32,
        theta: f32,
        rope_scale: f32,
    ) -> Result<Self> {
        let head_dim = dim / n_heads;
        let q_dim = dim;
        let kv_dim = (dim / n_heads) * n_kv_heads;
        
        // Allocate working buffers
        let d_hidden = device.alloc_zeros::<f32>(dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_hidden: {}", e)))?;
        let d_xb = device.alloc_zeros::<f32>(dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_xb: {}", e)))?;
        let d_q = device.alloc_zeros::<f32>(q_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_q: {}", e)))?;
        let d_k = device.alloc_zeros::<f32>(kv_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_k: {}", e)))?;
        let d_v = device.alloc_zeros::<f32>(kv_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_v: {}", e)))?;
        let d_attn_out = device.alloc_zeros::<f32>(q_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_attn_out: {}", e)))?;
        let d_hb = device.alloc_zeros::<f32>(ffn_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_hb: {}", e)))?;
        let d_hb2 = device.alloc_zeros::<f32>(ffn_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_hb2: {}", e)))?;
        let d_logits = device.alloc_zeros::<f32>(vocab_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_logits: {}", e)))?;
        
        // Allocate gradient working buffers
        let d_d_hidden = device.alloc_zeros::<f32>(dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_d_hidden: {}", e)))?;
        let d_d_xb = device.alloc_zeros::<f32>(dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_d_xb: {}", e)))?;
        let d_d_q = device.alloc_zeros::<f32>(q_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_d_q: {}", e)))?;
        let d_d_k = device.alloc_zeros::<f32>(kv_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_d_k: {}", e)))?;
        let d_d_v = device.alloc_zeros::<f32>(kv_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_d_v: {}", e)))?;
        let d_d_attn_out = device.alloc_zeros::<f32>(q_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_d_attn_out: {}", e)))?;
        let d_d_hb = device.alloc_zeros::<f32>(ffn_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_d_hb: {}", e)))?;
        let d_d_hb2 = device.alloc_zeros::<f32>(ffn_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_d_hb2: {}", e)))?;
        let d_d_logits = device.alloc_zeros::<f32>(vocab_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_d_logits: {}", e)))?;
        
        // Training buffers
        let d_targets = device.alloc_zeros::<i32>(config.batch_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_targets: {}", e)))?;
        let d_loss = device.alloc_zeros::<f32>(1)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_loss: {}", e)))?;
        let d_grad_norm = device.alloc_zeros::<f32>(1)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_grad_norm: {}", e)))?;
        
        Ok(Self {
            device,
            kernels,
            config,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            ffn_dim,
            vocab_size,
            max_seq_len,
            head_dim,
            q_dim,
            kv_dim,
            eps,
            theta,
            rope_scale,
            adam_timestep: 0,
            layer_gradients: Vec::new(),
            layer_adam_state: Vec::new(),
            forward_cache: Vec::new(),
            d_embeddings: None,
            d_output_weight: None,
            d_norm_weight: None,
            m_embeddings: None,
            v_embeddings: None,
            m_output_weight: None,
            v_output_weight: None,
            m_norm_weight: None,
            v_norm_weight: None,
            d_hidden,
            d_xb,
            d_q,
            d_k,
            d_v,
            d_attn_out,
            d_hb,
            d_hb2,
            d_logits,
            d_d_hidden,
            d_d_xb,
            d_d_q,
            d_d_k,
            d_d_v,
            d_d_attn_out,
            d_d_hb,
            d_d_hb2,
            d_d_logits,
            d_targets,
            d_loss,
            d_grad_norm,
            initialized: false,
        })
    }
    
    /// Initialize layer gradients and Adam state
    pub fn initialize(&mut self) -> Result<()> {
        // Allocate per-layer gradients
        for _ in 0..self.n_layers {
            let grads = self.allocate_layer_gradients()?;
            self.layer_gradients.push(grads);
            
            let adam = self.allocate_layer_adam_state()?;
            self.layer_adam_state.push(adam);
            
            let cache = self.allocate_forward_cache()?;
            self.forward_cache.push(cache);
        }
        
        // Allocate global gradients
        let emb_size = self.vocab_size * self.dim;
        self.d_embeddings = Some(self.device.alloc_zeros::<f32>(emb_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_embeddings: {}", e)))?);
        self.d_output_weight = Some(self.device.alloc_zeros::<f32>(emb_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_output_weight: {}", e)))?);
        self.d_norm_weight = Some(self.device.alloc_zeros::<f32>(self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_norm_weight: {}", e)))?);
        
        // Allocate global Adam state
        self.m_embeddings = Some(self.device.alloc_zeros::<f32>(emb_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc m_embeddings: {}", e)))?);
        self.v_embeddings = Some(self.device.alloc_zeros::<f32>(emb_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc v_embeddings: {}", e)))?);
        self.m_output_weight = Some(self.device.alloc_zeros::<f32>(emb_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc m_output_weight: {}", e)))?);
        self.v_output_weight = Some(self.device.alloc_zeros::<f32>(emb_size)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc v_output_weight: {}", e)))?);
        self.m_norm_weight = Some(self.device.alloc_zeros::<f32>(self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc m_norm_weight: {}", e)))?);
        self.v_norm_weight = Some(self.device.alloc_zeros::<f32>(self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc v_norm_weight: {}", e)))?);
        
        self.initialized = true;
        Ok(())
    }
    
    fn allocate_layer_gradients(&self) -> Result<LayerGradients> {
        let d_attn_norm = self.device.alloc_zeros::<f32>(self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_attn_norm: {}", e)))?;
        let d_ffn_norm = self.device.alloc_zeros::<f32>(self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_ffn_norm: {}", e)))?;
        let d_wq = self.device.alloc_zeros::<f32>(self.q_dim * self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_wq: {}", e)))?;
        let d_wk = self.device.alloc_zeros::<f32>(self.kv_dim * self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_wk: {}", e)))?;
        let d_wv = self.device.alloc_zeros::<f32>(self.kv_dim * self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_wv: {}", e)))?;
        let d_wo = self.device.alloc_zeros::<f32>(self.dim * self.q_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_wo: {}", e)))?;
        let d_w1 = self.device.alloc_zeros::<f32>(self.ffn_dim * self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_w1: {}", e)))?;
        let d_w2 = self.device.alloc_zeros::<f32>(self.dim * self.ffn_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_w2: {}", e)))?;
        let d_w3 = self.device.alloc_zeros::<f32>(self.ffn_dim * self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc d_w3: {}", e)))?;
        
        Ok(LayerGradients {
            d_attn_norm,
            d_ffn_norm,
            d_wq,
            d_wk,
            d_wv,
            d_wo,
            d_w1,
            d_w2,
            d_w3,
        })
    }
    
    fn allocate_layer_adam_state(&self) -> Result<LayerAdamState> {
        let alloc_pair = |size: usize| -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
            let m = self.device.alloc_zeros::<f32>(size)
                .map_err(|e| TransformerError::Cuda(format!("Failed to alloc Adam m: {}", e)))?;
            let v = self.device.alloc_zeros::<f32>(size)
                .map_err(|e| TransformerError::Cuda(format!("Failed to alloc Adam v: {}", e)))?;
            Ok((m, v))
        };
        
        let (m_wq, v_wq) = alloc_pair(self.q_dim * self.dim)?;
        let (m_wk, v_wk) = alloc_pair(self.kv_dim * self.dim)?;
        let (m_wv, v_wv) = alloc_pair(self.kv_dim * self.dim)?;
        let (m_wo, v_wo) = alloc_pair(self.dim * self.q_dim)?;
        let (m_w1, v_w1) = alloc_pair(self.ffn_dim * self.dim)?;
        let (m_w2, v_w2) = alloc_pair(self.dim * self.ffn_dim)?;
        let (m_w3, v_w3) = alloc_pair(self.ffn_dim * self.dim)?;
        let (m_attn_norm, v_attn_norm) = alloc_pair(self.dim)?;
        let (m_ffn_norm, v_ffn_norm) = alloc_pair(self.dim)?;
        
        Ok(LayerAdamState {
            m_wq,
            v_wq,
            m_wk,
            v_wk,
            m_wv,
            v_wv,
            m_wo,
            v_wo,
            m_w1,
            v_w1,
            m_w2,
            v_w2,
            m_w3,
            v_w3,
            m_attn_norm,
            v_attn_norm,
            m_ffn_norm,
            v_ffn_norm,
        })
    }
    
    fn allocate_forward_cache(&self) -> Result<ForwardActivations> {
        let pre_attn_norm = self.device.alloc_zeros::<f32>(self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc pre_attn_norm: {}", e)))?;
        let post_attn_norm = self.device.alloc_zeros::<f32>(self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc post_attn_norm: {}", e)))?;
        let q = self.device.alloc_zeros::<f32>(self.q_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc q: {}", e)))?;
        let k = self.device.alloc_zeros::<f32>(self.kv_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc k: {}", e)))?;
        let v = self.device.alloc_zeros::<f32>(self.kv_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc v: {}", e)))?;
        let attn_output = self.device.alloc_zeros::<f32>(self.q_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc attn_output: {}", e)))?;
        let post_attn_residual = self.device.alloc_zeros::<f32>(self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc post_attn_residual: {}", e)))?;
        let pre_ffn_norm = self.device.alloc_zeros::<f32>(self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc pre_ffn_norm: {}", e)))?;
        let post_ffn_norm = self.device.alloc_zeros::<f32>(self.dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc post_ffn_norm: {}", e)))?;
        let gate = self.device.alloc_zeros::<f32>(self.ffn_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc gate: {}", e)))?;
        let up = self.device.alloc_zeros::<f32>(self.ffn_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc up: {}", e)))?;
        let ffn_hidden = self.device.alloc_zeros::<f32>(self.ffn_dim)
            .map_err(|e| TransformerError::Cuda(format!("Failed to alloc ffn_hidden: {}", e)))?;
        
        Ok(ForwardActivations {
            pre_attn_norm,
            post_attn_norm,
            q,
            k,
            v,
            attn_output,
            post_attn_residual,
            pre_ffn_norm,
            post_ffn_norm,
            gate,
            up,
            ffn_hidden,
        })
    }
    
    /// Zero all gradients
    pub fn zero_gradients(&mut self) -> Result<()> {
        for grads in &mut self.layer_gradients {
            self.kernels.zero_gradients(&mut grads.d_attn_norm, self.dim)?;
            self.kernels.zero_gradients(&mut grads.d_ffn_norm, self.dim)?;
            self.kernels.zero_gradients(&mut grads.d_wq, self.q_dim * self.dim)?;
            self.kernels.zero_gradients(&mut grads.d_wk, self.kv_dim * self.dim)?;
            self.kernels.zero_gradients(&mut grads.d_wv, self.kv_dim * self.dim)?;
            self.kernels.zero_gradients(&mut grads.d_wo, self.dim * self.q_dim)?;
            self.kernels.zero_gradients(&mut grads.d_w1, self.ffn_dim * self.dim)?;
            self.kernels.zero_gradients(&mut grads.d_w2, self.dim * self.ffn_dim)?;
            self.kernels.zero_gradients(&mut grads.d_w3, self.ffn_dim * self.dim)?;
        }
        
        let emb_size = self.vocab_size * self.dim;
        if let Some(ref mut d_emb) = self.d_embeddings {
            self.kernels.zero_gradients(d_emb, emb_size)?;
        }
        if let Some(ref mut d_out) = self.d_output_weight {
            self.kernels.zero_gradients(d_out, emb_size)?;
        }
        if let Some(ref mut d_norm) = self.d_norm_weight {
            self.kernels.zero_gradients(d_norm, self.dim)?;
        }
        
        Ok(())
    }
    
    /// Compute loss for given logits and targets
    pub fn compute_loss(&mut self, targets: &[i32]) -> Result<f32> {
        let batch_size = targets.len();
        
        self.device.htod_copy_into(targets.to_vec(), &mut self.d_targets)
            .map_err(|e| TransformerError::Cuda(format!("Failed to copy targets: {}", e)))?;
        
        self.kernels.compute_loss(
            &mut self.d_loss,
            &self.d_logits,
            &self.d_targets,
            self.vocab_size,
            batch_size,
        )?;
        
        let loss = self.device.dtoh_sync_copy(&self.d_loss)
            .map_err(|e| TransformerError::Cuda(format!("Failed to copy loss: {}", e)))?;
        
        Ok(loss[0])
    }
    
    /// Perform backward pass
    pub fn backward(
        &mut self,
        layer_weights: &[LayerWeightsRef],
        output_weight: &CudaSlice<f32>,
        norm_weight: &CudaSlice<f32>,
        targets: &[i32],
    ) -> Result<()> {
        let batch_size = targets.len();
        
        self.device.htod_copy_into(targets.to_vec(), &mut self.d_targets)
            .map_err(|e| TransformerError::Cuda(format!("Failed to copy targets: {}", e)))?;
        
        // Cross-entropy backward
        self.kernels.cross_entropy_backward(
            &mut self.d_d_logits,
            &self.d_logits,
            &self.d_targets,
            self.vocab_size,
            batch_size,
        )?;
        
        // Output weight backward
        self.kernels.vec_mat_mul_backward_input(
            &mut self.d_d_hidden,
            &self.d_d_logits,
            output_weight,
            self.dim,
            self.vocab_size,
        )?;
        
        if let Some(ref mut d_out) = self.d_output_weight {
            self.kernels.vec_mat_mul_backward_weight(
                d_out,
                &self.d_hidden,
                &self.d_d_logits,
                self.dim,
                self.vocab_size,
            )?;
        }
        
        // Layer-by-layer backward
        for l in (0..self.n_layers).rev() {
            let weights = &layer_weights[l];
            let grads = &mut self.layer_gradients[l];
            let cache = &self.forward_cache[l];
            
            // FFN norm backward
            self.kernels.rms_norm_backward(
                &mut self.d_d_xb,
                &mut grads.d_ffn_norm,
                &self.d_d_hidden,
                &cache.pre_ffn_norm,
                weights.ffn_norm,
                self.dim,
                self.eps,
            )?;
            
            // W2 backward
            self.kernels.vec_mat_mul_backward_input(
                &mut self.d_d_hb,
                &self.d_d_xb,
                weights.w2,
                self.ffn_dim,
                self.dim,
            )?;
            self.kernels.vec_mat_mul_backward_weight(
                &mut grads.d_w2,
                &cache.ffn_hidden,
                &self.d_d_xb,
                self.ffn_dim,
                self.dim,
            )?;
            
            // SwiGLU backward - need to use d_d_attn_out as temp buffer
            // because Rust borrow checker doesn't allow same slice as both input and output
            // First copy d_d_hb to temp, then use temp as input
            self.device.dtod_copy(&self.d_d_hb, &mut self.d_d_attn_out)
                .map_err(|e| TransformerError::Cuda(format!("Failed to copy for SwiGLU backward: {}", e)))?;
            self.kernels.swiglu_backward(
                &mut self.d_d_hb2,
                &mut self.d_d_hb,
                &self.d_d_attn_out,
                &cache.gate,
                &cache.up,
                self.ffn_dim,
            )?;
            
            // W1 and W3 backward
            self.kernels.vec_mat_mul_backward_weight(
                &mut grads.d_w1,
                &cache.post_ffn_norm,
                &self.d_d_hb2,
                self.dim,
                self.ffn_dim,
            )?;
            self.kernels.vec_mat_mul_backward_weight(
                &mut grads.d_w3,
                &cache.post_ffn_norm,
                &self.d_d_hb,
                self.dim,
                self.ffn_dim,
            )?;
            
            // Residual backward
            self.kernels.residual_backward(
                &mut self.d_d_hidden,
                &self.d_d_xb,
                self.dim,
            )?;
            
            // Attention norm backward
            self.kernels.rms_norm_backward(
                &mut self.d_d_xb,
                &mut grads.d_attn_norm,
                &self.d_d_hidden,
                &cache.pre_attn_norm,
                weights.attn_norm,
                self.dim,
                self.eps,
            )?;
            
            // Output projection backward
            self.kernels.vec_mat_mul_backward_input(
                &mut self.d_d_attn_out,
                &self.d_d_xb,
                weights.wo,
                self.q_dim,
                self.dim,
            )?;
            self.kernels.vec_mat_mul_backward_weight(
                &mut grads.d_wo,
                &cache.attn_output,
                &self.d_d_xb,
                self.q_dim,
                self.dim,
            )?;
            
            // Q, K, V backward
            self.kernels.vec_mat_mul_backward_input(
                &mut self.d_d_q,
                &self.d_d_attn_out,
                weights.wq,
                self.dim,
                self.q_dim,
            )?;
            self.kernels.vec_mat_mul_backward_weight(
                &mut grads.d_wq,
                &cache.post_attn_norm,
                &self.d_d_q,
                self.dim,
                self.q_dim,
            )?;
            self.kernels.vec_mat_mul_backward_weight(
                &mut grads.d_wk,
                &cache.post_attn_norm,
                &self.d_d_k,
                self.dim,
                self.kv_dim,
            )?;
            self.kernels.vec_mat_mul_backward_weight(
                &mut grads.d_wv,
                &cache.post_attn_norm,
                &self.d_d_v,
                self.dim,
                self.kv_dim,
            )?;
            
            // Residual backward
            self.kernels.residual_backward(
                &mut self.d_d_hidden,
                &self.d_d_xb,
                self.dim,
            )?;
        }
        
        Ok(())
    }
    
    /// Perform optimizer step
    pub fn optimizer_step(
        &mut self,
        layer_weights: &mut [LayerWeightsMut],
        embeddings: &mut CudaSlice<f32>,
        output_weight: &mut CudaSlice<f32>,
        norm_weight: &mut CudaSlice<f32>,
    ) -> Result<()> {
        self.adam_timestep += 1;
        let t = self.adam_timestep;
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.adam_eps;
        
        // Update layer weights
        for l in 0..self.n_layers {
            let weights = &mut layer_weights[l];
            let grads = &self.layer_gradients[l];
            let adam = &mut self.layer_adam_state[l];
            
            self.kernels.adam_optimizer(
                weights.wq, &grads.d_wq,
                &mut adam.m_wq, &mut adam.v_wq,
                self.q_dim * self.dim, lr, beta1, beta2, eps, t,
            )?;
            self.kernels.adam_optimizer(
                weights.wk, &grads.d_wk,
                &mut adam.m_wk, &mut adam.v_wk,
                self.kv_dim * self.dim, lr, beta1, beta2, eps, t,
            )?;
            self.kernels.adam_optimizer(
                weights.wv, &grads.d_wv,
                &mut adam.m_wv, &mut adam.v_wv,
                self.kv_dim * self.dim, lr, beta1, beta2, eps, t,
            )?;
            self.kernels.adam_optimizer(
                weights.wo, &grads.d_wo,
                &mut adam.m_wo, &mut adam.v_wo,
                self.dim * self.q_dim, lr, beta1, beta2, eps, t,
            )?;
            self.kernels.adam_optimizer(
                weights.w1, &grads.d_w1,
                &mut adam.m_w1, &mut adam.v_w1,
                self.ffn_dim * self.dim, lr, beta1, beta2, eps, t,
            )?;
            self.kernels.adam_optimizer(
                weights.w2, &grads.d_w2,
                &mut adam.m_w2, &mut adam.v_w2,
                self.dim * self.ffn_dim, lr, beta1, beta2, eps, t,
            )?;
            self.kernels.adam_optimizer(
                weights.w3, &grads.d_w3,
                &mut adam.m_w3, &mut adam.v_w3,
                self.ffn_dim * self.dim, lr, beta1, beta2, eps, t,
            )?;
            self.kernels.adam_optimizer(
                weights.attn_norm, &grads.d_attn_norm,
                &mut adam.m_attn_norm, &mut adam.v_attn_norm,
                self.dim, lr, beta1, beta2, eps, t,
            )?;
            self.kernels.adam_optimizer(
                weights.ffn_norm, &grads.d_ffn_norm,
                &mut adam.m_ffn_norm, &mut adam.v_ffn_norm,
                self.dim, lr, beta1, beta2, eps, t,
            )?;
        }
        
        // Update embeddings
        let emb_size = self.vocab_size * self.dim;
        if let (Some(ref d_emb), Some(ref mut m_emb), Some(ref mut v_emb)) = 
            (&self.d_embeddings, &mut self.m_embeddings, &mut self.v_embeddings) 
        {
            self.kernels.adam_optimizer(
                embeddings, d_emb, m_emb, v_emb,
                emb_size, lr, beta1, beta2, eps, t,
            )?;
        }
        
        // Update output weight
        if let (Some(ref d_out), Some(ref mut m_out), Some(ref mut v_out)) = 
            (&self.d_output_weight, &mut self.m_output_weight, &mut self.v_output_weight) 
        {
            self.kernels.adam_optimizer(
                output_weight, d_out, m_out, v_out,
                emb_size, lr, beta1, beta2, eps, t,
            )?;
        }
        
        // Update norm weight
        if let (Some(ref d_norm), Some(ref mut m_norm), Some(ref mut v_norm)) = 
            (&self.d_norm_weight, &mut self.m_norm_weight, &mut self.v_norm_weight) 
        {
            self.kernels.adam_optimizer(
                norm_weight, d_norm, m_norm, v_norm,
                self.dim, lr, beta1, beta2, eps, t,
            )?;
        }
        
        Ok(())
    }
    
    /// Perform a complete training step
    pub fn train_step(
        &mut self,
        layer_weights_ref: &[LayerWeightsRef],
        layer_weights_mut: &mut [LayerWeightsMut],
        embeddings: &mut CudaSlice<f32>,
        output_weight: &CudaSlice<f32>,
        output_weight_mut: &mut CudaSlice<f32>,
        norm_weight: &CudaSlice<f32>,
        norm_weight_mut: &mut CudaSlice<f32>,
        targets: &[i32],
    ) -> Result<f32> {
        self.zero_gradients()?;
        
        let loss = self.compute_loss(targets)?;
        
        self.backward(layer_weights_ref, output_weight, norm_weight, targets)?;
        
        self.optimizer_step(layer_weights_mut, embeddings, output_weight_mut, norm_weight_mut)?;
        
        Ok(loss)
    }
    
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }
    
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

/// Reference to layer weights for reading
pub struct LayerWeightsRef<'a> {
    pub attn_norm: &'a CudaSlice<f32>,
    pub ffn_norm: &'a CudaSlice<f32>,
    pub wq: &'a CudaSlice<f32>,
    pub wk: &'a CudaSlice<f32>,
    pub wv: &'a CudaSlice<f32>,
    pub wo: &'a CudaSlice<f32>,
    pub w1: &'a CudaSlice<f32>,
    pub w2: &'a CudaSlice<f32>,
    pub w3: &'a CudaSlice<f32>,
}

/// Mutable reference to layer weights for updating
pub struct LayerWeightsMut<'a> {
    pub attn_norm: &'a mut CudaSlice<f32>,
    pub ffn_norm: &'a mut CudaSlice<f32>,
    pub wq: &'a mut CudaSlice<f32>,
    pub wk: &'a mut CudaSlice<f32>,
    pub wv: &'a mut CudaSlice<f32>,
    pub wo: &'a mut CudaSlice<f32>,
    pub w1: &'a mut CudaSlice<f32>,
    pub w2: &'a mut CudaSlice<f32>,
    pub w3: &'a mut CudaSlice<f32>,
}
