use crate::error::{Result, TransformerError};
use crate::gguf::GGUFLoader;
use std::collections::HashMap;

#[derive(Clone)]
pub struct ChatTokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges: Vec<(String, String)>,
    architecture: String,
    bos_id: u32,
    eos_id: u32,
    eot_id: u32,
}

impl ChatTokenizer {
    pub fn from_gguf(loader: &GGUFLoader) -> Result<Self> {
        if !loader.has_tokenizer() {
            return Err(TransformerError::Tokenizer(
                "No tokenizer data in GGUF file".into(),
            ));
        }

        let tokens = loader.get_tokens();
        let merges_raw = loader.get_merges();
        let architecture = loader.get_architecture().to_string();

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        let merges: Vec<(String, String)> = merges_raw
            .iter()
            .filter_map(|m| {
                let parts: Vec<&str> = m.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        let bos_id = token_to_id.get("<s>").copied().unwrap_or(1);
        let eos_id = token_to_id.get("</s>").copied().unwrap_or(2);
        let eot_id = token_to_id
            .get("<|eot_id|>")
            .or_else(|| token_to_id.get("<|end_of_turn|>"))
            .or_else(|| token_to_id.get("<|im_end|>"))
            .copied()
            .unwrap_or(eos_id);

        println!(
            "Tokenizer: {} vocab, {} merges, arch={}",
            tokens.len(),
            merges.len(),
            architecture
        );

        Ok(Self {
            vocab: tokens.to_vec(),
            token_to_id,
            merges,
            architecture,
            bos_id,
            eos_id,
            eot_id,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        if self.architecture.contains("llama") || self.architecture.contains("gemma") {
            tokens.push(self.bos_id);
        }

        let mut chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        loop {
            let mut best_merge: Option<(usize, &(String, String))> = None;
            let mut best_priority = usize::MAX;

            for i in 0..chars.len().saturating_sub(1) {
                let pair = (chars[i].clone(), chars[i + 1].clone());
                for (priority, merge) in self.merges.iter().enumerate() {
                    if merge.0 == pair.0 && merge.1 == pair.1 {
                        if priority < best_priority {
                            best_priority = priority;
                            best_merge = Some((i, merge));
                        }
                        break;
                    }
                }
            }

            match best_merge {
                Some((i, merge)) => {
                    let merged = format!("{}{}", merge.0, merge.1);
                    chars[i] = merged;
                    chars.remove(i + 1);
                }
                None => break,
            }
        }

        for piece in chars {
            if let Some(&id) = self.token_to_id.get(&piece) {
                tokens.push(id);
            } else {
                for c in piece.chars() {
                    let s = c.to_string();
                    if let Some(&id) = self.token_to_id.get(&s) {
                        tokens.push(id);
                    }
                }
            }
        }

        tokens
    }

    pub fn decode(&self, token_id: u32) -> String {
        if (token_id as usize) < self.vocab.len() {
            let mut s = self.vocab[token_id as usize].clone();
            s = s.replace("▁", " ");
            s = s.replace("Ġ", " ");
            s
        } else {
            String::new()
        }
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|&t| self.decode(t)).collect()
    }

    pub fn apply_chat_template(&self, user_message: &str) -> String {
        if self.architecture.contains("llama") {
            if self.token_to_id.contains_key("<|start_header_id|>") {
                format!(
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    user_message
                )
            } else {
                format!(
                    "<s>[INST] {} [/INST]",
                    user_message
                )
            }
        } else if self.architecture.contains("gemma") {
            format!(
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                user_message
            )
        } else if self.architecture.contains("qwen") {
            format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                user_message
            )
        } else if self.architecture.contains("phi") {
            format!(
                "<|user|>\n{}<|end|>\n<|assistant|>\n",
                user_message
            )
        } else {
            format!("User: {}\nAssistant:", user_message)
        }
    }

    pub fn bos(&self) -> u32 {
        self.bos_id
    }

    pub fn eos(&self) -> u32 {
        self.eos_id
    }

    pub fn eot(&self) -> u32 {
        self.eot_id
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    pub fn is_eos(&self, token_id: u32) -> bool {
        token_id == self.eos_id || token_id == self.eot_id
    }
}
