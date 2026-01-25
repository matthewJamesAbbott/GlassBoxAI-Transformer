use crate::error::{Result, TransformerError};
use crate::quant::*;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;


#[derive(Debug, Clone)]
pub struct GGUFTensor {
    pub name: String,
    pub num_dims: u32,
    pub shape: Vec<i64>,
    pub dtype: u32,
    pub data_offset: u64,
    pub raw_data: Vec<u8>,
}

pub struct GGUFLoader {
    mmap: Mmap,
    tensor_map: HashMap<String, usize>,
    tensors: Vec<GGUFTensor>,
    tensor_data_start: u64,

    embed_dim: i32,
    num_layers: i32,
    num_heads: i32,
    ffn_dim: i32,
    vocab_size: i32,
    max_seq_len: i32,
    num_kv_heads: i32,
    key_length: i32,
    sliding_window: i32,
    rope_theta: f32,
    rope_scale: f32,
    rms_eps: f32,
    query_pre_attn_scalar: f32,
    architecture: String,

    gguf_tokens: Vec<String>,
    gguf_merges: Vec<String>,
}

impl GGUFLoader {
    pub fn new() -> Self {
        Self {
            mmap: unsafe { Mmap::map(&File::open("/dev/null").unwrap()).unwrap() },
            tensor_map: HashMap::new(),
            tensors: Vec::new(),
            tensor_data_start: 0,
            embed_dim: 2048,
            num_layers: 16,
            num_heads: 32,
            ffn_dim: 8192,
            vocab_size: 128256,
            max_seq_len: 131072,
            num_kv_heads: 8,
            key_length: 0,
            sliding_window: 0,
            rope_theta: 500000.0,
            rope_scale: 1.0,
            rms_eps: 1e-5,
            query_pre_attn_scalar: 0.0,
            architecture: String::new(),
            gguf_tokens: Vec::new(),
            gguf_merges: Vec::new(),
        }
    }

    pub fn load_from_file(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut loader = Self {
            mmap,
            tensor_map: HashMap::new(),
            tensors: Vec::new(),
            tensor_data_start: 0,
            embed_dim: 2048,
            num_layers: 16,
            num_heads: 32,
            ffn_dim: 8192,
            vocab_size: 128256,
            max_seq_len: 131072,
            num_kv_heads: 8,
            key_length: 0,
            sliding_window: 0,
            rope_theta: 500000.0,
            rope_scale: 1.0,
            rms_eps: 1e-5,
            query_pre_attn_scalar: 0.0,
            architecture: String::new(),
            gguf_tokens: Vec::new(),
            gguf_merges: Vec::new(),
        };

        loader.parse_header()?;

        println!("Architecture: {}", loader.architecture);
        println!(
            "Model: {} layers, {} dim, {} heads ({} KV), {} FFN, vocab {}",
            loader.num_layers,
            loader.embed_dim,
            loader.num_heads,
            loader.num_kv_heads,
            loader.ffn_dim,
            loader.vocab_size
        );
        println!(
            "RoPE theta: {}, RMS eps: {}",
            loader.rope_theta, loader.rms_eps
        );

        Ok(loader)
    }

    fn read_u32(&self, offset: &mut usize) -> u32 {
        let val = u32::from_le_bytes(self.mmap[*offset..*offset + 4].try_into().unwrap());
        *offset += 4;
        val
    }

    fn read_u64(&self, offset: &mut usize) -> u64 {
        let val = u64::from_le_bytes(self.mmap[*offset..*offset + 8].try_into().unwrap());
        *offset += 8;
        val
    }

    fn read_i64(&self, offset: &mut usize) -> i64 {
        let val = i64::from_le_bytes(self.mmap[*offset..*offset + 8].try_into().unwrap());
        *offset += 8;
        val
    }

    fn read_f32(&self, offset: &mut usize) -> f32 {
        let val = f32::from_le_bytes(self.mmap[*offset..*offset + 4].try_into().unwrap());
        *offset += 4;
        val
    }

    fn read_string(&self, offset: &mut usize) -> String {
        let len = self.read_u64(offset) as usize;
        if len > 10_000_000 {
            return String::new();
        }
        let s = String::from_utf8_lossy(&self.mmap[*offset..*offset + len]).to_string();
        *offset += len;
        s
    }

    fn skip_metadata_value(&self, offset: &mut usize, value_type: u32) {
        match value_type {
            0 | 1 => *offset += 1,
            2 | 3 => *offset += 2,
            4 | 5 | 6 => *offset += 4,
            7 => *offset += 1,
            8 => {
                let len = self.read_u64(offset) as usize;
                *offset += len;
            }
            9 => {
                let arr_type = self.read_u32(offset);
                let arr_count = self.read_u64(offset) as usize;
                for _ in 0..arr_count.min(999999) {
                    self.skip_metadata_value(offset, arr_type);
                }
            }
            10 | 11 | 12 => *offset += 8,
            _ => {}
        }
    }

    fn is_text_model_key(key: &str) -> bool {
        !key.contains("vision.")
    }

    fn parse_header(&mut self) -> Result<()> {
        let mut offset = 0usize;

        let magic = &self.mmap[offset..offset + 4];
        if magic != b"GGUF" {
            return Err(TransformerError::GGUFParse("Invalid GGUF magic".into()));
        }
        offset += 4;

        let _version = self.read_u32(&mut offset);
        let tensor_count = self.read_u64(&mut offset);
        let metadata_count = self.read_u64(&mut offset);

        for _ in 0..metadata_count {
            let key = self.read_string(&mut offset);
            let value_type = self.read_u32(&mut offset);

            if key == "general.architecture" && value_type == 8 {
                self.architecture = self.read_string(&mut offset);
            } else if key.contains("embedding_length")
                && Self::is_text_model_key(&key)
                && matches!(value_type, 4 | 5 | 10)
            {
                self.embed_dim = if value_type == 10 {
                    self.read_u64(&mut offset) as i32
                } else {
                    self.read_u32(&mut offset) as i32
                };
            } else if key.contains("block_count")
                && Self::is_text_model_key(&key)
                && matches!(value_type, 4 | 5 | 10)
            {
                self.num_layers = if value_type == 10 {
                    self.read_u64(&mut offset) as i32
                } else {
                    self.read_u32(&mut offset) as i32
                };
            } else if key.contains("head_count_kv")
                && Self::is_text_model_key(&key)
                && matches!(value_type, 4 | 5 | 10)
            {
                self.num_kv_heads = if value_type == 10 {
                    self.read_u64(&mut offset) as i32
                } else {
                    self.read_u32(&mut offset) as i32
                };
            } else if key.contains("attention.head_count")
                && Self::is_text_model_key(&key)
                && matches!(value_type, 4 | 5 | 10)
            {
                self.num_heads = if value_type == 10 {
                    self.read_u64(&mut offset) as i32
                } else {
                    self.read_u32(&mut offset) as i32
                };
            } else if key.contains("feed_forward")
                && Self::is_text_model_key(&key)
                && matches!(value_type, 4 | 5 | 10)
            {
                self.ffn_dim = if value_type == 10 {
                    self.read_u64(&mut offset) as i32
                } else {
                    self.read_u32(&mut offset) as i32
                };
            } else if key.contains("context_length")
                && Self::is_text_model_key(&key)
                && matches!(value_type, 4 | 5 | 10)
            {
                self.max_seq_len = if value_type == 10 {
                    self.read_u64(&mut offset) as i32
                } else {
                    self.read_u32(&mut offset) as i32
                };
            } else if key.contains("rope.freq_base") && value_type == 6 {
                self.rope_theta = self.read_f32(&mut offset);
            } else if key.contains("layer_norm_rms_epsilon") && value_type == 6 {
                self.rms_eps = self.read_f32(&mut offset);
            } else if key.contains("attention.key_length")
                && Self::is_text_model_key(&key)
                && matches!(value_type, 4 | 5 | 10)
            {
                self.key_length = if value_type == 10 {
                    self.read_u64(&mut offset) as i32
                } else {
                    self.read_u32(&mut offset) as i32
                };
            } else if key.contains("attention.sliding_window")
                && Self::is_text_model_key(&key)
                && matches!(value_type, 4 | 5 | 10)
            {
                self.sliding_window = if value_type == 10 {
                    self.read_u64(&mut offset) as i32
                } else {
                    self.read_u32(&mut offset) as i32
                };
            } else if key.contains("rope.scaling.factor") && value_type == 6 {
                self.rope_scale = self.read_f32(&mut offset);
            } else if key.contains("query_pre_attn_scalar") && value_type == 6 {
                self.query_pre_attn_scalar = self.read_f32(&mut offset);
                println!("Query pre-attn scalar: {}", self.query_pre_attn_scalar);
            } else if key.contains("attn_logit_softcapping") && value_type == 6 {
                let val = self.read_f32(&mut offset);
                println!("Attn logit softcap (ignored): {}", val);
            } else if key == "tokenizer.ggml.tokens" && value_type == 9 {
                let arr_type = self.read_u32(&mut offset);
                let arr_count = self.read_u64(&mut offset) as usize;
                if arr_type == 8 {
                    self.gguf_tokens.reserve(arr_count);
                    for _ in 0..arr_count {
                        self.gguf_tokens.push(self.read_string(&mut offset));
                    }
                    self.vocab_size = arr_count as i32;
                    println!("Loaded {} tokens from GGUF", arr_count);
                } else {
                    for _ in 0..arr_count {
                        self.skip_metadata_value(&mut offset, arr_type);
                    }
                }
            } else if key == "tokenizer.ggml.merges" && value_type == 9 {
                let arr_type = self.read_u32(&mut offset);
                let arr_count = self.read_u64(&mut offset) as usize;
                if arr_type == 8 {
                    self.gguf_merges.reserve(arr_count);
                    for _ in 0..arr_count {
                        self.gguf_merges.push(self.read_string(&mut offset));
                    }
                    println!("Loaded {} merges from GGUF", arr_count);
                } else {
                    for _ in 0..arr_count {
                        self.skip_metadata_value(&mut offset, arr_type);
                    }
                }
            } else {
                self.skip_metadata_value(&mut offset, value_type);
            }
        }

        self.tensors.reserve(tensor_count as usize);
        for i in 0..tensor_count as usize {
            let name = self.read_string(&mut offset);
            let num_dims = self.read_u32(&mut offset);
            let mut shape = Vec::with_capacity(num_dims as usize);
            for _ in 0..num_dims {
                shape.push(self.read_i64(&mut offset));
            }
            let dtype = self.read_u32(&mut offset);
            let data_offset = self.read_u64(&mut offset);

            self.tensor_map.insert(name.clone(), i);
            self.tensors.push(GGUFTensor {
                name,
                num_dims,
                shape,
                dtype,
                data_offset,
                raw_data: Vec::new(),
            });
        }

        self.tensor_data_start = ((offset as u64 + 31) / 32) * 32;
        Ok(())
    }

    pub fn load_tensor_data(&self, name: &str) -> Result<Vec<f32>> {
        let idx = self
            .tensor_map
            .get(name)
            .ok_or_else(|| TransformerError::GGUFParse(format!("Tensor not found: {}", name)))?;

        let t = &self.tensors[*idx];
        let num_elements: usize = t.shape.iter().map(|&d| d as usize).product();
        let dtype = GGMLDType::from(t.dtype);

        let data_start = (self.tensor_data_start + t.data_offset) as usize;
        let mut result = vec![0.0f32; num_elements];

        match dtype {
            GGMLDType::F32 => {
                let data = &self.mmap[data_start..data_start + num_elements * 4];
                for (i, chunk) in data.chunks_exact(4).enumerate() {
                    result[i] = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            GGMLDType::F16 => {
                let data = &self.mmap[data_start..data_start + num_elements * 2];
                for (i, chunk) in data.chunks_exact(2).enumerate() {
                    result[i] = fp16_to_fp32(u16::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            GGMLDType::BFloat16 => {
                let data = &self.mmap[data_start..data_start + num_elements * 2];
                for (i, chunk) in data.chunks_exact(2).enumerate() {
                    result[i] = bf16_to_fp32(u16::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            _ => {
                let block_size = get_block_size(dtype);
                let bytes_per_block = get_bytes_per_block(dtype);

                if t.num_dims == 2 {
                    let cols = t.shape[0] as usize;
                    let rows = t.shape[1] as usize;
                    let blocks_per_row = cols / block_size;
                    let row_bytes = blocks_per_row * bytes_per_block;

                    for row in 0..rows {
                        let row_start = data_start + row * row_bytes;
                        let row_data = &self.mmap[row_start..row_start + row_bytes];
                        let out_slice = &mut result[row * cols..(row + 1) * cols];
                        self.dequant_row(row_data, out_slice, cols, dtype)?;
                    }
                } else {
                    let num_blocks = (num_elements + block_size - 1) / block_size;
                    let raw_bytes = num_blocks * bytes_per_block;
                    let raw_data = &self.mmap[data_start..data_start + raw_bytes];
                    self.dequant_row(raw_data, &mut result, num_elements, dtype)?;
                }
            }
        }

        Ok(result)
    }

    fn dequant_row(
        &self,
        data: &[u8],
        output: &mut [f32],
        cols: usize,
        dtype: GGMLDType,
    ) -> Result<()> {
        match dtype {
            GGMLDType::Q2_K => {
                let blocks: &[BlockQ2K] = bytemuck::cast_slice(data);
                dequant_row_q2_k(blocks, output, cols);
            }
            GGMLDType::Q3_K => {
                let blocks: &[BlockQ3K] = bytemuck::cast_slice(data);
                dequant_row_q3_k(blocks, output, cols);
            }
            GGMLDType::Q4_K => {
                let blocks: &[BlockQ4K] = bytemuck::cast_slice(data);
                dequant_row_q4_k(blocks, output, cols);
            }
            GGMLDType::Q5_K => {
                let blocks: &[BlockQ5K] = bytemuck::cast_slice(data);
                dequant_row_q5_k(blocks, output, cols);
            }
            GGMLDType::Q6_K => {
                let blocks: &[BlockQ6K] = bytemuck::cast_slice(data);
                dequant_row_q6_k(blocks, output, cols);
            }
            GGMLDType::Q8_0 => {
                let blocks: &[BlockQ8_0] = bytemuck::cast_slice(data);
                dequant_row_q8_0(blocks, output, cols);
            }
            GGMLDType::Q8_K => {
                let blocks: &[BlockQ8K] = bytemuck::cast_slice(data);
                dequant_row_q8_k(blocks, output, cols);
            }
            GGMLDType::Q4_0 => {
                let blocks: &[BlockQ4_0] = bytemuck::cast_slice(data);
                dequant_row_q4_0(blocks, output, cols);
            }
            GGMLDType::Q4_1 => {
                let blocks: &[BlockQ4_1] = bytemuck::cast_slice(data);
                dequant_row_q4_1(blocks, output, cols);
            }
            GGMLDType::Q5_0 => {
                let blocks: &[BlockQ5_0] = bytemuck::cast_slice(data);
                dequant_row_q5_0(blocks, output, cols);
            }
            GGMLDType::Q5_1 => {
                let blocks: &[BlockQ5_1] = bytemuck::cast_slice(data);
                dequant_row_q5_1(blocks, output, cols);
            }
            _ => {
                output.fill(0.0);
            }
        }
        Ok(())
    }

    pub fn get_embed_dim(&self) -> i32 {
        self.embed_dim
    }
    pub fn get_num_layers(&self) -> i32 {
        self.num_layers
    }
    pub fn get_num_heads(&self) -> i32 {
        self.num_heads
    }
    pub fn get_ffn_dim(&self) -> i32 {
        self.ffn_dim
    }
    pub fn get_vocab_size(&self) -> i32 {
        self.vocab_size
    }
    pub fn get_max_seq_len(&self) -> i32 {
        self.max_seq_len
    }
    pub fn get_head_dim(&self) -> i32 {
        if self.key_length > 0 {
            self.key_length
        } else {
            self.embed_dim / self.num_heads
        }
    }
    pub fn get_num_kv_heads(&self) -> i32 {
        self.num_kv_heads
    }
    pub fn get_sliding_window(&self) -> i32 {
        if self.sliding_window > 0 {
            self.sliding_window
        } else {
            1024
        }
    }
    pub fn get_rope_theta(&self) -> f32 {
        self.rope_theta
    }
    pub fn get_rope_scale(&self) -> f32 {
        self.rope_scale
    }
    pub fn get_rms_eps(&self) -> f32 {
        self.rms_eps
    }
    pub fn get_query_pre_attn_scalar(&self) -> f32 {
        self.query_pre_attn_scalar
    }
    pub fn get_architecture(&self) -> &str {
        &self.architecture
    }

    pub fn has_tokenizer(&self) -> bool {
        !self.gguf_tokens.is_empty()
    }
    pub fn get_tokens(&self) -> &[String] {
        &self.gguf_tokens
    }
    pub fn get_merges(&self) -> &[String] {
        &self.gguf_merges
    }

    pub fn get_tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<(&GGUFTensor, GGMLDType)> {
        self.tensor_map.get(name).map(|&idx| {
            let t = &self.tensors[idx];
            (t, GGMLDType::from(t.dtype))
        })
    }
    
    pub fn load_tensor_raw(&self, name: &str) -> Result<Vec<u8>> {
        let idx = self
            .tensor_map
            .get(name)
            .ok_or_else(|| TransformerError::GGUFParse(format!("Tensor not found: {}", name)))?;

        let t = &self.tensors[*idx];
        let num_elements: usize = t.shape.iter().map(|&d| d as usize).product();
        let dtype = GGMLDType::from(t.dtype);

        let block_size = get_block_size(dtype);
        let bytes_per_block = get_bytes_per_block(dtype);
        let num_blocks = (num_elements + block_size - 1) / block_size;
        let raw_bytes = num_blocks * bytes_per_block;

        let data_start = (self.tensor_data_start + t.data_offset) as usize;
        Ok(self.mmap[data_start..data_start + raw_bytes].to_vec())
    }
    
    pub fn get_tensor_dtype(&self, name: &str) -> Option<i32> {
        self.tensor_map.get(name).map(|&idx| {
            self.tensors[idx].dtype as i32
        })
    }
    
    pub fn get_tensor_shape(&self, name: &str) -> Option<Vec<i64>> {
        self.tensor_map.get(name).map(|&idx| {
            self.tensors[idx].shape.clone()
        })
    }
}

impl Default for GGUFLoader {
    fn default() -> Self {
        Self::new()
    }
}
