pub mod error;
pub mod quant;
pub mod gguf;
pub mod tokenizer;
pub mod model;
pub mod kernels;
pub mod generator;
pub mod protocol;
pub mod network;
pub mod trainer;

pub use error::TransformerError;
pub use gguf::GGUFLoader;
pub use tokenizer::ChatTokenizer;
pub use model::TransformerModel;
pub use generator::{GPUTextGenerator, GenerationConfig};
pub use protocol::{
    DTXHeader, MessageType, ConnectionState, DistributedConfig,
    EthernetFrame, RawSocket, mac_to_string, string_to_mac,
};
pub use network::{
    TransformerServer, TransformerClient,
    DistributedTransformer, DistributedTransformerServer,
    benchmark_distributed,
};
pub use trainer::{
    GpuTrainer, TrainingConfig, LayerGradients, LayerAdamState,
    ForwardActivations, LayerWeightsRef, LayerWeightsMut,
};
