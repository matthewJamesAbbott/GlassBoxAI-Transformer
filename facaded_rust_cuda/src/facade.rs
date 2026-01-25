// GlassBox AI Transformer Facade - Introspection API
// Exposes hidden states, attention weights, QKV vectors for analysis

use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use std::sync::Arc;
use rand::prelude::*;
use rand::rngs::StdRng;
use crate::error::{Result, TransformerError};
use crate::model::TransformerModel;
use crate::tokenizer::ChatTokenizer;
use crate::kernels::CudaKernels;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QKVType {
    Query,
    Key,
    Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamType {
    QProj,
    KProj,
    VProj,
    OutProj,
    FFN1,
    FFN2,
    LayerNorm1Weight,
    LayerNorm1Bias,
    LayerNorm2Weight,
    LayerNorm2Bias,
    TokenEmbed,
    PosEmbed,
    FinalNormWeight,
    FinalNormBias,
}

#[derive(Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub rep_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            rep_penalty: 1.1,
        }
    }
}

pub struct LayerIntrospection {
    pub hidden_states: Vec<f32>,
    pub q_vectors: Vec<f32>,
    pub k_vectors: Vec<f32>,
    pub v_vectors: Vec<f32>,
    pub attention_weights: Vec<f32>,
    pub attention_logits: Vec<f32>,
    pub ffn_output: Vec<f32>,
    pub layer_norm_output: Vec<f32>,
}

impl Default for LayerIntrospection {
    fn default() -> Self {
        Self {
            hidden_states: Vec::new(),
            q_vectors: Vec::new(),
            k_vectors: Vec::new(),
            v_vectors: Vec::new(),
            attention_weights: Vec::new(),
            attention_logits: Vec::new(),
            ffn_output: Vec::new(),
            layer_norm_output: Vec::new(),
        }
    }
}

pub struct TransformerFacade {
    model: TransformerModel,
    tokenizer: ChatTokenizer,
    kernels: CudaKernels,
    #[allow(dead_code)]
    stream: CudaStream,
    
    d_hidden: CudaSlice<f32>,
    d_xb: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    d_k: CudaSlice<f32>,
    d_v: CudaSlice<f32>,
    d_attn_out: CudaSlice<f32>,
    d_hb: CudaSlice<f32>,
    d_hb2: CudaSlice<f32>,
    d_ffn_act: CudaSlice<f32>,
    d_logits: CudaSlice<f32>,
    
    d_kv_cache_k: CudaSlice<f32>,
    d_kv_cache_v: CudaSlice<f32>,
    
    h_logits: Vec<f32>,
    current_pos: usize,
    last_seq_len: usize,
    
    layer_introspection: Vec<LayerIntrospection>,
    last_logits: Vec<f32>,
    
    rng: StdRng,
}

impl TransformerFacade {
    pub fn new(model: TransformerModel, tokenizer: ChatTokenizer) -> Result<Self> {
        let device = model.device.clone();
        let stream = device.fork_default_stream()
            .map_err(|e| TransformerError::Cuda(format!("Failed to create stream: {}", e)))?;
        let kernels = CudaKernels::new(device.clone())?;
        
        let dim = model.dim;
        let q_dim = model.q_dim;
        let kv_dim = model.kv_dim;
        let ffn_dim = model.ffn_dim;
        let vocab_size = model.vocab_size;
        let n_layers = model.n_layers;
        let max_seq_len = model.max_seq_len.min(4096);
        
        let d_hidden = alloc_zeros(&device, dim)?;
        let d_xb = alloc_zeros(&device, dim)?;
        let d_q = alloc_zeros(&device, q_dim)?;
        let d_k = alloc_zeros(&device, kv_dim)?;
        let d_v = alloc_zeros(&device, kv_dim)?;
        let d_attn_out = alloc_zeros(&device, dim)?;
        let d_hb = alloc_zeros(&device, ffn_dim)?;
        let d_hb2 = alloc_zeros(&device, ffn_dim)?;
        let d_ffn_act = alloc_zeros(&device, ffn_dim)?;
        let d_logits = alloc_zeros(&device, vocab_size)?;
        
        let kv_cache_size = n_layers * max_seq_len * kv_dim;
        let d_kv_cache_k = alloc_zeros(&device, kv_cache_size)?;
        let d_kv_cache_v = alloc_zeros(&device, kv_cache_size)?;
        
        let h_logits = vec![0.0f32; vocab_size];
        
        let layer_introspection = (0..n_layers).map(|_| LayerIntrospection::default()).collect();
        
        let rng = StdRng::from_entropy();
        
        Ok(Self {
            model,
            tokenizer,
            kernels,
            stream,
            d_hidden,
            d_xb,
            d_q,
            d_k,
            d_v,
            d_attn_out,
            d_hb,
            d_hb2,
            d_ffn_act,
            d_logits,
            d_kv_cache_k,
            d_kv_cache_v,
            h_logits,
            current_pos: 0,
            last_seq_len: 0,
            layer_introspection,
            last_logits: Vec::new(),
            rng,
        })
    }

    pub fn forward(&mut self, token: u32, pos: usize) -> Result<()> {
        let dim = self.model.dim;
        let n_heads = self.model.n_heads;
        let n_kv_heads = self.model.n_kv_heads;
        let head_dim = self.model.head_dim;
        let q_dim = self.model.q_dim;
        let kv_dim = self.model.kv_dim;
        let ffn_dim = self.model.ffn_dim;
        let eps = self.model.eps;
        let theta = self.model.theta;
        let rope_scale = self.model.rope_scale;
        let max_seq_len = self.model.max_seq_len.min(4096);
        
        self.kernels.embedding_lookup(
            &mut self.d_hidden,
            &self.model.embeddings,
            token as usize,
            dim,
        )?;
        
        for l in 0..self.model.n_layers {
            let layer = &self.model.layers[l];
            
            self.kernels.rms_norm(
                &mut self.d_xb,
                &self.d_hidden,
                &layer.attn_norm,
                dim,
                eps,
            )?;
            
            self.layer_introspection[l].layer_norm_output = self.copy_from_gpu(&self.d_xb, dim)?;
            
            self.kernels.vec_mat_mul_tensor(&mut self.d_q, &self.d_xb, &layer.wq, dim, q_dim)?;
            self.kernels.vec_mat_mul_tensor(&mut self.d_k, &self.d_xb, &layer.wk, dim, kv_dim)?;
            self.kernels.vec_mat_mul_tensor(&mut self.d_v, &self.d_xb, &layer.wv, dim, kv_dim)?;
            
            if let (Some(ref q_norm), Some(ref k_norm)) = (&layer.q_norm, &layer.k_norm) {
                for h in 0..n_heads {
                    self.kernels.rms_norm_slice(
                        &mut self.d_q,
                        q_norm,
                        h * head_dim,
                        head_dim,
                        eps,
                    )?;
                }
                for h in 0..n_kv_heads {
                    self.kernels.rms_norm_slice(
                        &mut self.d_k,
                        k_norm,
                        h * head_dim,
                        head_dim,
                        eps,
                    )?;
                }
            }
            
            self.layer_introspection[l].q_vectors = self.copy_from_gpu(&self.d_q, q_dim)?;
            self.layer_introspection[l].k_vectors = self.copy_from_gpu(&self.d_k, kv_dim)?;
            self.layer_introspection[l].v_vectors = self.copy_from_gpu(&self.d_v, kv_dim)?;
            
            self.kernels.rope(
                &mut self.d_q,
                &mut self.d_k,
                q_dim,
                kv_dim,
                head_dim,
                pos,
                theta,
                rope_scale,
            )?;
            
            let layer_offset = l * max_seq_len * kv_dim;
            let cache_pos = pos * kv_dim;
            self.kernels.copy_to_cache(
                &mut self.d_kv_cache_k,
                &self.d_k,
                layer_offset + cache_pos,
                kv_dim,
            )?;
            self.kernels.copy_to_cache(
                &mut self.d_kv_cache_v,
                &self.d_v,
                layer_offset + cache_pos,
                kv_dim,
            )?;
            
            let kv_mul = n_heads / n_kv_heads;
            for h in 0..n_heads {
                let kv_head = h / kv_mul;
                self.kernels.fused_attention(
                    &mut self.d_attn_out,
                    &self.d_q,
                    &self.d_kv_cache_k,
                    &self.d_kv_cache_v,
                    h,
                    kv_head,
                    head_dim,
                    pos + 1,
                    layer_offset,
                    kv_dim,
                )?;
            }
            
            self.kernels.vec_mat_mul_tensor(&mut self.d_xb, &self.d_attn_out, &layer.wo, dim, dim)?;
            self.kernels.residual_add(&mut self.d_hidden, &self.d_xb, dim)?;
            
            self.layer_introspection[l].hidden_states = self.copy_from_gpu(&self.d_hidden, dim)?;
            
            self.kernels.rms_norm(
                &mut self.d_xb,
                &self.d_hidden,
                &layer.ffn_norm,
                dim,
                eps,
            )?;
            
            self.kernels.vec_mat_mul_tensor(&mut self.d_hb, &self.d_xb, &layer.w1, dim, ffn_dim)?;
            self.kernels.vec_mat_mul_tensor(&mut self.d_hb2, &self.d_xb, &layer.w3, dim, ffn_dim)?;
            
            if self.model.is_gemma {
                self.kernels.geglu(&self.d_hb, &self.d_hb2, &mut self.d_ffn_act, ffn_dim)?;
            } else {
                self.kernels.swiglu(&self.d_hb, &self.d_hb2, &mut self.d_ffn_act, ffn_dim)?;
            }
            
            self.layer_introspection[l].ffn_output = self.copy_from_gpu(&self.d_ffn_act, ffn_dim)?;
            
            self.kernels.vec_mat_mul_tensor(&mut self.d_xb, &self.d_ffn_act, &layer.w2, ffn_dim, dim)?;
            self.kernels.residual_add(&mut self.d_hidden, &self.d_xb, dim)?;
        }
        
        self.kernels.rms_norm(
            &mut self.d_xb,
            &self.d_hidden,
            &self.model.norm_weight,
            dim,
            eps,
        )?;
        
        self.kernels.vec_mat_mul(
            &mut self.d_logits,
            &self.d_xb,
            &self.model.output_weight,
            dim,
            self.model.vocab_size,
        )?;
        
        self.current_pos = pos;
        self.last_seq_len = pos + 1;
        Ok(())
    }
    
    fn copy_from_gpu(&self, slice: &CudaSlice<f32>, size: usize) -> Result<Vec<f32>> {
        let mut result = vec![0.0f32; size];
        self.model.device.dtoh_sync_copy_into(slice, &mut result)
            .map_err(|e| TransformerError::Cuda(format!("GPU copy failed: {}", e)))?;
        Ok(result)
    }

    pub fn sample(&mut self, config: &GenerationConfig) -> Result<u32> {
        self.model.device.dtoh_sync_copy_into(&self.d_logits, &mut self.h_logits)
            .map_err(|e| TransformerError::Cuda(format!("Failed to copy logits: {}", e)))?;
        
        self.last_logits = self.h_logits.clone();
        
        if config.temperature > 0.0 {
            for logit in &mut self.h_logits {
                *logit /= config.temperature;
            }
        }
        
        let mut scored: Vec<(f32, u32)> = self.h_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect();
        
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(config.top_k);
        
        let max_logit = scored[0].0;
        let mut sum = 0.0f32;
        for (logit, _) in &mut scored {
            *logit = (*logit - max_logit).exp();
            sum += *logit;
        }
        
        let mut cumsum = 0.0f32;
        let mut cutoff = scored.len();
        for (i, (prob, _)) in scored.iter_mut().enumerate() {
            *prob /= sum;
            cumsum += *prob;
            if cumsum >= config.top_p {
                cutoff = i + 1;
                break;
            }
        }
        scored.truncate(cutoff);
        
        let renorm_sum: f32 = scored.iter().map(|(p, _)| p).sum();
        let r: f32 = self.rng.gen::<f32>() * renorm_sum;
        
        let mut cumulative = 0.0f32;
        for (prob, token) in &scored {
            cumulative += prob;
            if cumulative >= r {
                return Ok(*token);
            }
        }
        
        Ok(scored.last().map(|(_, t)| *t).unwrap_or(0))
    }

    pub fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        let tokens = self.tokenizer.encode(prompt);
        
        self.clear_cache()?;
        
        let mut all_tokens = tokens.clone();
        
        for (i, &token) in tokens.iter().enumerate() {
            self.forward(token, i)?;
        }
        
        let mut pos = tokens.len();
        let mut output_tokens = Vec::new();
        
        for _ in 0..config.max_tokens {
            let next_token = self.sample(config)?;
            
            if self.tokenizer.is_eos(next_token) {
                break;
            }
            
            output_tokens.push(next_token);
            all_tokens.push(next_token);
            
            let piece = self.tokenizer.decode(next_token);
            print!("{}", piece);
            std::io::Write::flush(&mut std::io::stdout()).ok();
            
            self.forward(next_token, pos)?;
            pos += 1;
            
            if pos >= self.model.max_seq_len.min(4096) - 1 {
                break;
            }
        }
        
        Ok(self.tokenizer.decode_tokens(&output_tokens))
    }

    pub fn clear_cache(&mut self) -> Result<()> {
        let n_layers = self.model.n_layers;
        let max_seq_len = self.model.max_seq_len.min(4096);
        let kv_dim = self.model.kv_dim;
        let cache_size = n_layers * max_seq_len * kv_dim;
        
        let zeros = vec![0.0f32; cache_size];
        self.model.device.htod_sync_copy_into(&zeros, &mut self.d_kv_cache_k)
            .map_err(|e| TransformerError::Cuda(format!("Failed to clear K cache: {}", e)))?;
        self.model.device.htod_sync_copy_into(&zeros, &mut self.d_kv_cache_v)
            .map_err(|e| TransformerError::Cuda(format!("Failed to clear V cache: {}", e)))?;
        
        self.current_pos = 0;
        self.last_seq_len = 0;
        
        for intro in &mut self.layer_introspection {
            *intro = LayerIntrospection::default();
        }
        
        Ok(())
    }
    
    // ==================== Introspection API ====================
    
    pub fn num_layers(&self) -> usize {
        self.model.n_layers
    }
    
    pub fn num_heads(&self) -> usize {
        self.model.n_heads
    }
    
    pub fn hidden_size(&self) -> usize {
        self.model.dim
    }
    
    pub fn head_dim(&self) -> usize {
        self.model.head_dim
    }
    
    pub fn ffn_dim(&self) -> usize {
        self.model.ffn_dim
    }
    
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size
    }
    
    pub fn max_seq_len(&self) -> usize {
        self.model.max_seq_len
    }
    
    pub fn last_seq_len(&self) -> usize {
        self.last_seq_len
    }
    
    pub fn get_hidden_state(&self, layer: usize, pos: usize) -> Option<Vec<f32>> {
        if layer >= self.layer_introspection.len() {
            return None;
        }
        let dim = self.model.dim;
        let start = pos * dim;
        let intro = &self.layer_introspection[layer];
        if start + dim <= intro.hidden_states.len() {
            Some(intro.hidden_states[start..start + dim].to_vec())
        } else if intro.hidden_states.len() >= dim {
            Some(intro.hidden_states[..dim].to_vec())
        } else {
            None
        }
    }
    
    pub fn get_qkv(&self, layer: usize, head: usize, qkv_type: QKVType, pos: usize) -> Option<Vec<f32>> {
        if layer >= self.layer_introspection.len() {
            return None;
        }
        let head_dim = self.model.head_dim;
        let intro = &self.layer_introspection[layer];
        
        let vec = match qkv_type {
            QKVType::Query => &intro.q_vectors,
            QKVType::Key => &intro.k_vectors,
            QKVType::Value => &intro.v_vectors,
        };
        
        let embed_dim = if qkv_type == QKVType::Query { self.model.q_dim } else { self.model.kv_dim };
        let head_start = head * head_dim;
        let start = pos * embed_dim + head_start;
        
        if start + head_dim <= vec.len() {
            Some(vec[start..start + head_dim].to_vec())
        } else if head_start + head_dim <= vec.len() {
            Some(vec[head_start..head_start + head_dim].to_vec())
        } else {
            None
        }
    }
    
    pub fn get_logits(&self) -> &[f32] {
        &self.last_logits
    }
    
    pub fn get_softmax_output(&self) -> Vec<f32> {
        let logits = &self.last_logits;
        if logits.is_empty() {
            return Vec::new();
        }
        
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut result: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = result.iter().sum();
        for v in &mut result {
            *v /= sum;
        }
        result
    }
    
    pub fn get_token_embedding(&self, token_id: usize) -> Result<Vec<f32>> {
        self.model.get_embedding(token_id)
    }
    
    pub fn get_attention_entropy(&self, layer: usize, head: usize) -> f64 {
        if layer >= self.layer_introspection.len() {
            return 0.0;
        }
        let intro = &self.layer_introspection[layer];
        let seq_len = self.last_seq_len;
        if seq_len == 0 || intro.attention_weights.is_empty() {
            return 0.0;
        }
        
        let mut sum = 0.0f64;
        for pos in 0..seq_len {
            for src in 0..seq_len {
                let idx = head * seq_len * seq_len + pos * seq_len + src;
                if idx < intro.attention_weights.len() {
                    let w = intro.attention_weights[idx] as f64;
                    if w > 1e-10 {
                        sum -= w * w.ln();
                    }
                }
            }
        }
        sum / seq_len as f64
    }
    
    pub fn get_saliency_map(&self, token_idx: usize, layer: usize) -> Option<Vec<f32>> {
        let hidden = self.get_hidden_state(layer, token_idx)?;
        let max_abs = hidden.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        
        if max_abs > 0.0 {
            Some(hidden.iter().map(|x| x.abs() / max_abs).collect())
        } else {
            Some(vec![0.0; hidden.len()])
        }
    }
    
    pub fn tokenizer(&self) -> &ChatTokenizer {
        &self.tokenizer
    }
}

fn alloc_zeros(device: &Arc<CudaDevice>, size: usize) -> Result<CudaSlice<f32>> {
    let zeros = vec![0.0f32; size];
    device.htod_sync_copy(&zeros)
        .map_err(|e| TransformerError::Cuda(format!("Failed to allocate buffer: {}", e)))
}
