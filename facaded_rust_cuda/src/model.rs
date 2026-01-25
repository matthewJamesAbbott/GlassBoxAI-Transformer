use cudarc::driver::{CudaDevice, CudaSlice, result};
use std::sync::Arc;
use crate::error::{Result, TransformerError};
use crate::gguf::GGUFLoader;

pub enum TensorData {
    F32(CudaSlice<f32>),
    Quantized(CudaSlice<u8>, i32),
}

impl TensorData {
    pub fn as_f32(&self) -> Option<&CudaSlice<f32>> {
        match self {
            TensorData::F32(s) => Some(s),
            _ => None,
        }
    }
    
    pub fn as_quantized(&self) -> Option<(&CudaSlice<u8>, i32)> {
        match self {
            TensorData::Quantized(s, dtype) => Some((s, *dtype)),
            _ => None,
        }
    }
    
    pub fn is_quantized(&self) -> bool {
        matches!(self, TensorData::Quantized(_, _))
    }
    
    pub fn dtype(&self) -> i32 {
        match self {
            TensorData::F32(_) => 0,
            TensorData::Quantized(_, dtype) => *dtype,
        }
    }
}

pub struct LayerWeights {
    pub attn_norm: CudaSlice<f32>,
    pub ffn_norm: CudaSlice<f32>,
    pub wq: TensorData,
    pub wk: TensorData,
    pub wv: TensorData,
    pub wo: TensorData,
    pub w1: TensorData,
    pub w2: TensorData,
    pub w3: TensorData,
    pub q_norm: Option<CudaSlice<f32>>,
    pub k_norm: Option<CudaSlice<f32>>,
}

pub struct TransformerModel {
    pub device: Arc<CudaDevice>,
    pub embeddings: CudaSlice<f32>,
    pub output_weight: CudaSlice<f32>,
    pub norm_weight: CudaSlice<f32>,
    pub layers: Vec<LayerWeights>,
    
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub eps: f32,
    pub theta: f32,
    pub rope_scale: f32,
    pub is_gemma: bool,
}

impl TransformerModel {
    pub fn from_gguf(loader: &GGUFLoader, device: Arc<CudaDevice>) -> Result<Self> {
        let dim = loader.get_embed_dim() as usize;
        let n_layers = loader.get_num_layers() as usize;
        let n_heads = loader.get_num_heads() as usize;
        let n_kv_heads = loader.get_num_kv_heads() as usize;
        let ffn_dim = loader.get_ffn_dim() as usize;
        let vocab_size = loader.get_vocab_size() as usize;
        let max_seq_len_raw = loader.get_max_seq_len() as usize;
        let max_seq_len = max_seq_len_raw.min(4096);
        let head_dim = loader.get_head_dim() as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let eps = loader.get_rms_eps();
        let theta = loader.get_rope_theta();
        let rope_scale = loader.get_rope_scale();
        let arch = loader.get_architecture().to_lowercase();
        let is_gemma = arch.contains("gemma");

        let (free_mem, total_mem) = result::mem_get_info()
            .map_err(|e| TransformerError::Cuda(format!("Failed to get memory info: {:?}", e)))?;
        
        println!("[GPU] Available: {} MB / {} MB", free_mem / (1024*1024), total_mem / (1024*1024));

        let embeddings = Self::load_f32_tensor_to_gpu(loader, &device, "token_embd.weight")?;
        
        let output_weight = Self::try_load_f32_tensor_to_gpu(loader, &device, "output.weight")
            .or_else(|_| Self::try_load_f32_tensor_to_gpu(loader, &device, "token_embd.weight"))?;
        
        let norm_weight = Self::load_f32_tensor_to_gpu(loader, &device, "output_norm.weight")?;
        
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let attn_norm = Self::load_f32_tensor_to_gpu(
                loader, &device, &format!("blk.{}.attn_norm.weight", l)
            )?;
            let ffn_norm = Self::load_f32_tensor_to_gpu(
                loader, &device, &format!("blk.{}.ffn_norm.weight", l)
            )?;
            
            let wq = Self::load_weight_tensor(loader, &device, &format!("blk.{}.attn_q.weight", l))?;
            let wk = Self::load_weight_tensor(loader, &device, &format!("blk.{}.attn_k.weight", l))?;
            let wv = Self::load_weight_tensor(loader, &device, &format!("blk.{}.attn_v.weight", l))?;
            let wo = Self::load_weight_tensor(loader, &device, &format!("blk.{}.attn_output.weight", l))?;
            let w1 = Self::load_weight_tensor(loader, &device, &format!("blk.{}.ffn_gate.weight", l))?;
            let w2 = Self::load_weight_tensor(loader, &device, &format!("blk.{}.ffn_down.weight", l))?;
            let w3 = Self::load_weight_tensor(loader, &device, &format!("blk.{}.ffn_up.weight", l))?;
            
            let q_norm = Self::try_load_f32_tensor_to_gpu(
                loader, &device, &format!("blk.{}.attn_q_norm.weight", l)
            ).ok();
            let k_norm = Self::try_load_f32_tensor_to_gpu(
                loader, &device, &format!("blk.{}.attn_k_norm.weight", l)
            ).ok();
            
            if l == 0 {
                println!("[GPU] Layer {} wq dtype: {}", l, wq.dtype());
            }
            
            layers.push(LayerWeights {
                attn_norm,
                ffn_norm,
                wq,
                wk,
                wv,
                wo,
                w1,
                w2,
                w3,
                q_norm,
                k_norm,
            });
        }

        Ok(Self {
            device,
            embeddings,
            output_weight,
            norm_weight,
            layers,
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
            is_gemma,
        })
    }

    fn load_weight_tensor(
        loader: &GGUFLoader,
        device: &Arc<CudaDevice>,
        name: &str,
    ) -> Result<TensorData> {
        let dtype = loader.get_tensor_dtype(name).unwrap_or(0);
        
        if Self::is_quantized_dtype(dtype) {
            let raw_data = loader.load_tensor_raw(name)?;
            let gpu_slice = device.htod_sync_copy(&raw_data)
                .map_err(|e| TransformerError::Cuda(format!("Failed to upload quantized {}: {}", name, e)))?;
            Ok(TensorData::Quantized(gpu_slice, dtype))
        } else {
            let data = loader.load_tensor_data(name)?;
            let gpu_slice = device.htod_sync_copy(&data)
                .map_err(|e| TransformerError::Cuda(format!("Failed to upload {}: {}", name, e)))?;
            Ok(TensorData::F32(gpu_slice))
        }
    }
    
    fn is_quantized_dtype(dtype: i32) -> bool {
        matches!(dtype, 8 | 10 | 12 | 14)
    }
    
    fn load_f32_tensor_to_gpu(
        loader: &GGUFLoader,
        device: &Arc<CudaDevice>,
        name: &str,
    ) -> Result<CudaSlice<f32>> {
        let data = loader.load_tensor_data(name)?;
        device.htod_sync_copy(&data)
            .map_err(|e| TransformerError::Cuda(format!("Failed to upload {}: {}", name, e)))
    }

    fn try_load_f32_tensor_to_gpu(
        loader: &GGUFLoader,
        device: &Arc<CudaDevice>,
        name: &str,
    ) -> Result<CudaSlice<f32>> {
        Self::load_f32_tensor_to_gpu(loader, device, name)
    }
    
    pub fn get_embedding(&self, token_id: usize) -> Result<Vec<f32>> {
        let start = token_id * self.dim;
        let mut result = vec![0.0f32; self.dim];
        
        let all_embeddings = self.device.dtoh_sync_copy(&self.embeddings)
            .map_err(|e| TransformerError::Cuda(format!("Failed to copy embeddings: {}", e)))?;
        
        if start + self.dim <= all_embeddings.len() {
            result.copy_from_slice(&all_embeddings[start..start + self.dim]);
        }
        Ok(result)
    }
}
