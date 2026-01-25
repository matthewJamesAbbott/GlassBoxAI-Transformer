use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use std::sync::Arc;
use rand::prelude::*;
use rand::rngs::StdRng;
use crate::error::{Result, TransformerError};
use crate::model::TransformerModel;
use crate::tokenizer::ChatTokenizer;
use crate::kernels::CudaKernels;

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

pub struct GPUTextGenerator {
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
    d_ffn_act: CudaSlice<f32>,  // FFN activation output buffer
    d_logits: CudaSlice<f32>,
    
    d_kv_cache_k: CudaSlice<f32>,
    d_kv_cache_v: CudaSlice<f32>,
    
    h_logits: Vec<f32>,
    current_pos: usize,
    
    rng: StdRng,
}

impl GPUTextGenerator {
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
            
            self.kernels.rms_norm(
                &mut self.d_xb,
                &self.d_hidden,
                &layer.ffn_norm,
                dim,
                eps,
            )?;
            
            self.kernels.vec_mat_mul_tensor(&mut self.d_hb, &self.d_xb, &layer.w1, dim, ffn_dim)?;
            self.kernels.vec_mat_mul_tensor(&mut self.d_hb2, &self.d_xb, &layer.w3, dim, ffn_dim)?;
            
            // Apply activation (SwiGLU or GeGLU) - output goes to d_ffn_act
            if self.model.is_gemma {
                self.kernels.geglu(&self.d_hb, &self.d_hb2, &mut self.d_ffn_act, ffn_dim)?;
            } else {
                self.kernels.swiglu(&self.d_hb, &self.d_hb2, &mut self.d_ffn_act, ffn_dim)?;
            }
            
            // Down projection from activation output
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
        Ok(())
    }

    pub fn sample(&mut self, config: &GenerationConfig) -> Result<u32> {
        self.model.device.dtoh_sync_copy_into(&self.d_logits, &mut self.h_logits)
            .map_err(|e| TransformerError::Cuda(format!("Failed to copy logits: {}", e)))?;
        
        let _vocab_size = self.h_logits.len();
        
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
            
            // Print token as it's generated (streaming)
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
        Ok(())
    }
}

fn alloc_zeros(device: &Arc<CudaDevice>, size: usize) -> Result<CudaSlice<f32>> {
    let zeros = vec![0.0f32; size];
    device.htod_sync_copy(&zeros)
        .map_err(|e| TransformerError::Cuda(format!("Failed to allocate buffer: {}", e)))
}
