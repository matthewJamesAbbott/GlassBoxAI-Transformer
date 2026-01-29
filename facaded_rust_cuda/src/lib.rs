pub mod error;
pub mod quant;
pub mod gguf;
pub mod tokenizer;
pub mod model;
pub mod kernels;
pub mod facade;
pub mod trainer;

#[cfg(kani)]
pub mod kani;

pub use error::TransformerError;
pub use gguf::GGUFLoader;
pub use tokenizer::ChatTokenizer;
pub use model::TransformerModel;
pub use facade::{
    TransformerFacade, GenerationConfig, LayerIntrospection,
    QKVType, ParamType,
};
pub use trainer::{
    GpuTrainer, TrainingConfig, LayerGradients, LayerAdamState,
    ForwardActivations, LayerWeightsRef, LayerWeightsMut,
};
