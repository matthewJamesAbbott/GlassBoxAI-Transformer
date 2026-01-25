use thiserror::Error;
use std::io;

#[derive(Error, Debug)]
pub enum TransformerError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("GGUF error: {0}")]
    Gguf(String),
    
    #[error("GGUF parse error: {0}")]
    GGUFParse(String),
    
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    
    #[error("CUDA error: {0}")]
    Cuda(String),
    
    #[error("Facade error: {0}")]
    Facade(String),
}

pub type Result<T> = std::result::Result<T, TransformerError>;
