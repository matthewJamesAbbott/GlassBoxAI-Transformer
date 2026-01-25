use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransformerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("GGUF parse error: {0}")]
    GGUFParse(String),
    #[error("CUDA error: {0}")]
    Cuda(String),
    #[error("Model error: {0}")]
    Model(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
}

pub type Result<T> = std::result::Result<T, TransformerError>;
