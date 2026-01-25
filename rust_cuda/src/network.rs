// Network layer for distributed transformer inference
// TransformerServer and TransformerClient using Layer 2 Ethernet

use std::io;
use std::collections::HashMap;
use crate::protocol::*;
use crate::error::{Result, TransformerError};

// Client session tracking
pub struct ClientSession {
    pub client_id: u32,
    pub client_mac: [u8; 6],
    pub config: HandshakeReq,
    pub last_activations: Vec<f32>,
    pub last_seq_num: u16,
}

// Forward callback type
pub type ForwardCallback = Box<dyn Fn(&[f32], u16, u8, u8) -> Vec<f32> + Send + Sync>;
pub type BackwardCallback = Box<dyn Fn(&[f32], u16, u8, u8) -> Vec<f32> + Send + Sync>;

// TransformerServer - Receives and processes tensor operations over Layer 2 Ethernet
pub struct TransformerServer {
    socket: RawSocket,
    #[allow(dead_code)]
    interface_name: String,
    server_id: u32,
    #[allow(dead_code)]
    local_mac: [u8; 6],
    state: ConnectionState,
    connected_clients: HashMap<u32, ClientSession>,
    max_concurrent_clients: usize,
    has_gpu: bool,
    forward_callback: Option<ForwardCallback>,
    backward_callback: Option<BackwardCallback>,
}

impl TransformerServer {
    pub fn new(interface: &str, server_id: u32) -> io::Result<Self> {
        let socket = RawSocket::new(interface)?;
        let local_mac = *socket.local_mac();
        
        println!("[Server] Initialized on {} ({})", interface, mac_to_string(&local_mac));
        
        Ok(Self {
            socket,
            interface_name: interface.to_string(),
            server_id,
            local_mac,
            state: ConnectionState::Connected,
            connected_clients: HashMap::new(),
            max_concurrent_clients: 4,
            has_gpu: true,
            forward_callback: None,
            backward_callback: None,
        })
    }
    
    pub fn set_forward_callback(&mut self, cb: ForwardCallback) {
        self.forward_callback = Some(cb);
    }
    
    pub fn set_backward_callback(&mut self, cb: BackwardCallback) {
        self.backward_callback = Some(cb);
    }
    
    pub fn set_gpu_available(&mut self, available: bool) {
        self.has_gpu = available;
    }
    
    pub fn set_max_clients(&mut self, max: usize) {
        self.max_concurrent_clients = max;
    }
    
    pub fn state(&self) -> ConnectionState {
        self.state
    }
    
    pub fn connected_clients(&self) -> usize {
        self.connected_clients.len()
    }
    
    pub fn process_next_message(&mut self, timeout_ms: u32) -> io::Result<bool> {
        let frame = match self.socket.receive_frame(timeout_ms)? {
            Some(f) => f,
            None => return Ok(false),
        };
        
        if frame.payload.len() < std::mem::size_of::<DTXHeader>() {
            return Ok(false);
        }
        
        let hdr: DTXHeader = *bytemuck::from_bytes(&frame.payload[..std::mem::size_of::<DTXHeader>()]);
        
        if !hdr.verify() {
            return Ok(false);
        }
        
        let payload_start = std::mem::size_of::<DTXHeader>();
        let payload = &frame.payload[payload_start..];
        
        if !hdr.verify_checksum(payload) {
            eprintln!("[Server] Checksum mismatch");
            return Ok(false);
        }
        
        match hdr.message_type() {
            MessageType::HandshakeReq => self.handle_handshake(&frame.src_mac, &hdr, payload),
            MessageType::LayerConfig => self.handle_layer_config(&frame.src_mac, &hdr, payload),
            MessageType::ForwardChunk => self.handle_forward_chunk(&frame.src_mac, &hdr, payload),
            MessageType::BackwardChunk => self.handle_backward_chunk(&frame.src_mac, &hdr, payload),
            MessageType::Ping => self.handle_ping(&frame.src_mac, &hdr),
            MessageType::Disconnect => self.handle_disconnect(&frame.src_mac),
            _ => {}
        }
        
        Ok(true)
    }
    
    pub fn run(&mut self, max_messages: Option<usize>) {
        println!("[Server] Running...");
        let mut count = 0;
        loop {
            if let Some(max) = max_messages {
                if count >= max {
                    break;
                }
            }
            let _ = self.process_next_message(1000);
            count += 1;
        }
    }
    
    fn send_response(&self, dest_mac: &[u8; 6], hdr: &DTXHeader, payload: &[u8]) -> io::Result<()> {
        let mut frame_payload = Vec::with_capacity(std::mem::size_of::<DTXHeader>() + payload.len());
        frame_payload.extend_from_slice(bytemuck::bytes_of(hdr));
        frame_payload.extend_from_slice(payload);
        self.socket.send_frame(dest_mac, &frame_payload)?;
        Ok(())
    }
    
    fn handle_handshake(&mut self, src_mac: &[u8; 6], _hdr: &DTXHeader, payload: &[u8]) {
        if payload.len() < std::mem::size_of::<HandshakeReq>() {
            return;
        }
        
        let req: HandshakeReq = *bytemuck::from_bytes(&payload[..std::mem::size_of::<HandshakeReq>()]);
        
        let session = ClientSession {
            client_id: req.client_id,
            client_mac: *src_mac,
            config: req,
            last_activations: Vec::new(),
            last_seq_num: 0,
        };
        
        self.connected_clients.insert(req.client_id, session);
        
        let ack = HandshakeAck {
            server_id: self.server_id,
            has_gpu: if self.has_gpu { 1 } else { 0 },
            max_concurrent: self.max_concurrent_clients as u8,
            protocol_ver: DTX_VERSION as u16,
        };
        
        let ack_bytes = bytemuck::bytes_of(&ack);
        let resp_hdr = DTXHeader::new(MessageType::HandshakeAck, 1, Some(ack_bytes));
        
        if let Err(e) = self.send_response(src_mac, &resp_hdr, ack_bytes) {
            eprintln!("[Server] Failed to send handshake ack: {}", e);
        } else {
            println!("[Server] Client connected: {}", mac_to_string(src_mac));
        }
    }
    
    fn handle_layer_config(&mut self, src_mac: &[u8; 6], hdr: &DTXHeader, payload: &[u8]) {
        if payload.len() < std::mem::size_of::<LayerConfig>() {
            return;
        }
        
        let _config: LayerConfig = *bytemuck::from_bytes(&payload[..std::mem::size_of::<LayerConfig>()]);
        
        // Send acknowledgement
        let resp_hdr = DTXHeader::new(MessageType::LayerConfigAck, hdr.sequence_num + 1, None);
        let _ = self.send_response(src_mac, &resp_hdr, &[]);
    }
    
    fn handle_forward_chunk(&mut self, src_mac: &[u8; 6], hdr: &DTXHeader, payload: &[u8]) {
        if payload.len() < std::mem::size_of::<ForwardChunk>() {
            return;
        }
        
        let chunk: ForwardChunk = *bytemuck::from_bytes(&payload[..std::mem::size_of::<ForwardChunk>()]);
        let data_offset = std::mem::size_of::<ForwardChunk>();
        let tensor_data = deserialize_tensor(&payload[data_offset..]);
        
        if let Some(ref callback) = self.forward_callback {
            let result = callback(&tensor_data, chunk.seq_len, 0, 1);
            
            if !result.is_empty() {
                let result_bytes = serialize_tensor(&result);
                
                let res = ForwardResult {
                    chunk_id: chunk.chunk_id,
                    seq_start: chunk.seq_start,
                    seq_len: chunk.seq_len,
                    output_dim: chunk.embed_dim,
                    data_size: result_bytes.len() as u32,
                    activation_size: 0,
                };
                
                let mut resp_payload = Vec::with_capacity(std::mem::size_of::<ForwardResult>() + result_bytes.len());
                resp_payload.extend_from_slice(bytemuck::bytes_of(&res));
                resp_payload.extend_from_slice(&result_bytes);
                
                let resp_hdr = DTXHeader::new(MessageType::ForwardResult, hdr.sequence_num + 1, Some(&resp_payload));
                let _ = self.send_response(src_mac, &resp_hdr, &resp_payload);
            }
        }
    }
    
    fn handle_backward_chunk(&mut self, src_mac: &[u8; 6], hdr: &DTXHeader, payload: &[u8]) {
        if payload.len() < std::mem::size_of::<BackwardChunk>() {
            return;
        }
        
        let chunk: BackwardChunk = *bytemuck::from_bytes(&payload[..std::mem::size_of::<BackwardChunk>()]);
        let data_offset = std::mem::size_of::<BackwardChunk>();
        let grad_data = deserialize_tensor(&payload[data_offset..]);
        
        if let Some(ref callback) = self.backward_callback {
            let result = callback(&grad_data, chunk.seq_len, 0, 1);
            
            if !result.is_empty() {
                let result_bytes = serialize_tensor(&result);
                
                let res = BackwardResult {
                    chunk_id: chunk.chunk_id,
                    seq_start: chunk.seq_start,
                    seq_len: chunk.seq_len,
                    grad_dim: chunk.grad_dim,
                    data_size: result_bytes.len() as u32,
                    param_grad_size: 0,
                };
                
                let mut resp_payload = Vec::with_capacity(std::mem::size_of::<BackwardResult>() + result_bytes.len());
                resp_payload.extend_from_slice(bytemuck::bytes_of(&res));
                resp_payload.extend_from_slice(&result_bytes);
                
                let resp_hdr = DTXHeader::new(MessageType::BackwardResult, hdr.sequence_num + 1, Some(&resp_payload));
                let _ = self.send_response(src_mac, &resp_hdr, &resp_payload);
            }
        }
    }
    
    fn handle_ping(&self, src_mac: &[u8; 6], hdr: &DTXHeader) {
        let resp_hdr = DTXHeader::new(MessageType::Pong, hdr.sequence_num, None);
        let _ = self.send_response(src_mac, &resp_hdr, &[]);
    }
    
    fn handle_disconnect(&mut self, src_mac: &[u8; 6]) {
        let client_id = self.connected_clients.iter()
            .find(|(_, s)| compare_mac(&s.client_mac, src_mac))
            .map(|(id, _)| *id);
        
        if let Some(id) = client_id {
            self.connected_clients.remove(&id);
            println!("[Server] Client disconnected: {}", mac_to_string(src_mac));
        }
    }
}

// TransformerClient - Connects to server and sends tensor operations
pub struct TransformerClient {
    socket: RawSocket,
    #[allow(dead_code)]
    interface_name: String,
    #[allow(dead_code)]
    local_mac: [u8; 6],
    server_mac: [u8; 6],
    state: ConnectionState,
    client_id: u32,
    server_id: u32,
    sequence_num: u16,
    config: HandshakeReq,
    layer_config: LayerConfig,
}

impl TransformerClient {
    pub fn new(interface: &str, server_mac: [u8; 6]) -> io::Result<Self> {
        let socket = RawSocket::new(interface)?;
        let local_mac = *socket.local_mac();
        
        println!("[Client] Initialized on {} (local: {}, server: {})",
            interface, mac_to_string(&local_mac), mac_to_string(&server_mac));
        
        Ok(Self {
            socket,
            interface_name: interface.to_string(),
            local_mac,
            server_mac,
            state: ConnectionState::Disconnected,
            client_id: 0x87654321,
            server_id: 0,
            sequence_num: 0,
            config: HandshakeReq {
                client_id: 0x87654321,
                seq_batch_size: 512,
                embed_dim: 768,
                ffn_dim: 3072,
                num_heads: 12,
                num_kv_heads: 12,
            },
            layer_config: LayerConfig {
                start_layer: 0,
                num_layers: 0,
                keep_activations: 1,
                reserved: 0,
                total_params: 0,
            },
        })
    }
    
    pub fn set_config(&mut self, seq_len: u16, embed_dim: u16, ffn_dim: u32, num_heads: u8, num_kv_heads: u8) {
        self.config = HandshakeReq {
            client_id: self.client_id,
            seq_batch_size: seq_len,
            embed_dim,
            ffn_dim,
            num_heads,
            num_kv_heads,
        };
    }
    
    pub fn set_layer_config(&mut self, start_layer: u8, num_layers: u8, keep_activations: bool) {
        self.layer_config = LayerConfig {
            start_layer,
            num_layers,
            keep_activations: if keep_activations { 1 } else { 0 },
            reserved: 0,
            total_params: 0,
        };
    }
    
    pub fn state(&self) -> ConnectionState {
        self.state
    }
    
    pub fn is_connected(&self) -> bool {
        self.state == ConnectionState::Connected
    }
    
    pub fn server_id(&self) -> u32 {
        self.server_id
    }
    
    fn next_seq(&mut self) -> u16 {
        self.sequence_num = self.sequence_num.wrapping_add(1);
        self.sequence_num
    }
    
    fn send_message(&mut self, msg_type: MessageType, payload: Option<&[u8]>) -> io::Result<()> {
        let seq = self.next_seq();
        let hdr = DTXHeader::new(msg_type, seq, payload);
        
        let mut frame_payload = Vec::with_capacity(std::mem::size_of::<DTXHeader>() + payload.map(|p| p.len()).unwrap_or(0));
        frame_payload.extend_from_slice(bytemuck::bytes_of(&hdr));
        if let Some(p) = payload {
            frame_payload.extend_from_slice(p);
        }
        
        self.socket.send_frame(&self.server_mac, &frame_payload)?;
        Ok(())
    }
    
    fn receive_response(&mut self, timeout_ms: u32) -> io::Result<Option<(DTXHeader, Vec<u8>)>> {
        let start = std::time::Instant::now();
        
        while start.elapsed().as_millis() < timeout_ms as u128 {
            if let Some(frame) = self.socket.receive_frame(500)? {
                if !compare_mac(&frame.src_mac, &self.server_mac) {
                    continue;
                }
                
                if frame.payload.len() < std::mem::size_of::<DTXHeader>() {
                    continue;
                }
                
                let hdr: DTXHeader = *bytemuck::from_bytes(&frame.payload[..std::mem::size_of::<DTXHeader>()]);
                
                if !hdr.verify() {
                    continue;
                }
                
                let payload = frame.payload[std::mem::size_of::<DTXHeader>()..].to_vec();
                return Ok(Some((hdr, payload)));
            }
        }
        
        Ok(None)
    }
    
    pub fn connect(&mut self, timeout_ms: u32) -> io::Result<bool> {
        self.state = ConnectionState::Connecting;
        
        // Clone config to avoid borrow conflict
        let config_copy = self.config;
        let config_bytes = bytemuck::bytes_of(&config_copy);
        self.send_message(MessageType::HandshakeReq, Some(config_bytes))?;
        
        if let Some((hdr, payload)) = self.receive_response(timeout_ms)? {
            if hdr.message_type() == MessageType::HandshakeAck {
                if payload.len() >= std::mem::size_of::<HandshakeAck>() {
                    let ack: HandshakeAck = *bytemuck::from_bytes(&payload[..std::mem::size_of::<HandshakeAck>()]);
                    self.server_id = ack.server_id;
                    self.state = ConnectionState::Connected;
                    println!("[Client] Connected to server");
                    return Ok(true);
                }
            }
        }
        
        self.state = ConnectionState::Error;
        eprintln!("[Client] Handshake timeout");
        Ok(false)
    }
    
    pub fn disconnect(&mut self) -> io::Result<()> {
        self.send_message(MessageType::Disconnect, None)?;
        self.state = ConnectionState::Disconnected;
        Ok(())
    }
    
    pub fn forward(&mut self, input: &[f32], seq_len: u16) -> io::Result<Vec<f32>> {
        if self.state != ConnectionState::Connected {
            return Err(io::Error::new(io::ErrorKind::NotConnected, "Not connected"));
        }
        
        self.send_tensor_chunks(input, seq_len, 
            MessageType::ForwardStart, MessageType::ForwardChunk, MessageType::ForwardDone)?;
        
        self.receive_tensor_chunks(DTX_FRAME_TIMEOUT)
    }
    
    pub fn backward(&mut self, grad_output: &[f32], seq_len: u16) -> io::Result<Vec<f32>> {
        if self.state != ConnectionState::Connected {
            return Err(io::Error::new(io::ErrorKind::NotConnected, "Not connected"));
        }
        
        self.send_tensor_chunks(grad_output, seq_len,
            MessageType::BackwardStart, MessageType::BackwardChunk, MessageType::BackwardDone)?;
        
        self.receive_tensor_chunks(DTX_FRAME_TIMEOUT)
    }
    
    fn send_tensor_chunks(&mut self, data: &[f32], seq_len: u16,
        start_type: MessageType, chunk_type: MessageType, done_type: MessageType) -> io::Result<()> {
        
        self.send_message(start_type, None)?;
        
        let elements_per_chunk = (DTX_MAX_PAYLOAD - std::mem::size_of::<ForwardChunk>()) / 4;
        let mut chunk_id = 0u32;
        let mut offset = 0;
        
        while offset < data.len() {
            let chunk_size = (data.len() - offset).min(elements_per_chunk);
            let chunk_data = &data[offset..offset + chunk_size];
            let tensor_bytes = serialize_tensor(chunk_data);
            
            let chunk = ForwardChunk {
                chunk_id,
                seq_start: 0,
                seq_len,
                embed_dim: self.config.embed_dim,
                data_size: tensor_bytes.len() as u32,
            };
            
            let mut payload = Vec::with_capacity(std::mem::size_of::<ForwardChunk>() + tensor_bytes.len());
            payload.extend_from_slice(bytemuck::bytes_of(&chunk));
            payload.extend_from_slice(&tensor_bytes);
            
            self.send_message(chunk_type, Some(&payload))?;
            
            chunk_id += 1;
            offset += chunk_size;
        }
        
        self.send_message(done_type, None)?;
        Ok(())
    }
    
    fn receive_tensor_chunks(&mut self, timeout_ms: u32) -> io::Result<Vec<f32>> {
        let mut result = Vec::new();
        let start = std::time::Instant::now();
        
        while start.elapsed().as_millis() < timeout_ms as u128 {
            if let Some((hdr, payload)) = self.receive_response(500)? {
                match hdr.message_type() {
                    MessageType::ForwardResult | MessageType::BackwardResult => {
                        let data_offset = std::mem::size_of::<ForwardResult>();
                        if payload.len() > data_offset {
                            let tensor_data = deserialize_tensor(&payload[data_offset..]);
                            result.extend(tensor_data);
                        }
                    }
                    MessageType::ForwardComplete | MessageType::BackwardComplete => {
                        break;
                    }
                    _ => {}
                }
            }
        }
        
        Ok(result)
    }
}

// DistributedTransformer - High-level interface for distributed inference
pub struct DistributedTransformer {
    config: DistributedConfig,
    client: Option<TransformerClient>,
    activation_cache: Vec<Vec<f32>>,
}

impl DistributedTransformer {
    pub fn new(config: DistributedConfig) -> Self {
        let activation_cache = vec![Vec::new(); config.total_layers];
        
        Self {
            config,
            client: None,
            activation_cache,
        }
    }
    
    pub fn initialize(&mut self) -> Result<()> {
        if !self.config.validate() {
            return Err(TransformerError::Model("Invalid config: local + remote != total".into()));
        }
        
        let client = TransformerClient::new(&self.config.interface_name, self.config.server_mac)
            .map_err(|e| TransformerError::Model(format!("Failed to create client: {}", e)))?;
        
        self.client = Some(client);
        
        println!("[DistTransformer] Initialized");
        println!("  Local layers: 0-{}", self.config.start_remote_layer.saturating_sub(1));
        println!("  Remote layers: {}-{}", 
            self.config.start_remote_layer,
            self.config.start_remote_layer + self.config.remote_layers - 1);
        
        Ok(())
    }
    
    pub fn connect(&mut self, timeout_ms: u32) -> Result<()> {
        if let Some(ref mut client) = self.client {
            client.connect(timeout_ms)
                .map_err(|e| TransformerError::Model(format!("Connect failed: {}", e)))?;
            Ok(())
        } else {
            Err(TransformerError::Model("Client not initialized".into()))
        }
    }
    
    pub fn disconnect(&mut self) -> Result<()> {
        if let Some(ref mut client) = self.client {
            client.disconnect()
                .map_err(|e| TransformerError::Model(format!("Disconnect failed: {}", e)))?;
        }
        Ok(())
    }
    
    pub fn is_connected(&self) -> bool {
        self.client.as_ref().map(|c| c.is_connected()).unwrap_or(false)
    }
    
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if !self.is_connected() {
            return Err(TransformerError::Model("Not connected to remote server".into()));
        }
        
        // Process local layers first (if any)
        let mut intermediate = input.to_vec();
        if self.config.start_remote_layer > 0 {
            intermediate = self.forward_local(&intermediate, 0, self.config.start_remote_layer)?;
        }
        
        // Send to remote for processing
        let client = self.client.as_mut().unwrap();
        let output = client.forward(&intermediate, self.config.seq_len as u16)
            .map_err(|e| TransformerError::Model(format!("Forward failed: {}", e)))?;
        
        // Cache activations if enabled
        if self.config.cache_activations && !output.is_empty() {
            let cache_layer = self.config.start_remote_layer + self.config.remote_layers - 1;
            if cache_layer < self.activation_cache.len() {
                self.activation_cache[cache_layer] = output.clone();
            }
        }
        
        Ok(output)
    }
    
    pub fn backward(&mut self, grad_output: &[f32]) -> Result<Vec<f32>> {
        if !self.is_connected() {
            return Err(TransformerError::Model("Not connected to remote server".into()));
        }
        
        let client = self.client.as_mut().unwrap();
        let mut grad = client.backward(grad_output, self.config.seq_len as u16)
            .map_err(|e| TransformerError::Model(format!("Backward failed: {}", e)))?;
        
        // Process local layers in reverse
        if self.config.local_layers > 0 {
            grad = self.backward_local(&grad, 0, self.config.local_layers)?;
        }
        
        Ok(grad)
    }
    
    fn forward_local(&self, input: &[f32], start_layer: usize, num_layers: usize) -> Result<Vec<f32>> {
        println!("[DistTransformer] Forward local layers {}-{}", 
            start_layer, start_layer + num_layers - 1);
        // Local processing would be done here
        Ok(input.to_vec())
    }
    
    fn backward_local(&self, grad_output: &[f32], start_layer: usize, num_layers: usize) -> Result<Vec<f32>> {
        println!("[DistTransformer] Backward local layers {}-{}", 
            start_layer, start_layer + num_layers - 1);
        // Local processing would be done here
        Ok(grad_output.to_vec())
    }
    
    pub fn cache_activation(&mut self, layer: usize, activation: Vec<f32>) {
        if layer < self.activation_cache.len() {
            self.activation_cache[layer] = activation;
        }
    }
    
    pub fn get_activation(&self, layer: usize) -> Option<&[f32]> {
        self.activation_cache.get(layer).map(|v| v.as_slice())
    }
    
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }
}

// DistributedTransformerServer - High-level server wrapper
pub struct DistributedTransformerServer {
    config: DistributedConfig,
    server: Option<TransformerServer>,
}

impl DistributedTransformerServer {
    pub fn new(config: DistributedConfig) -> Self {
        Self {
            config,
            server: None,
        }
    }
    
    pub fn initialize(&mut self) -> Result<()> {
        if !self.config.validate() {
            return Err(TransformerError::Model("Invalid server configuration".into()));
        }
        
        let server = TransformerServer::new(&self.config.interface_name, 0x12345678)
            .map_err(|e| TransformerError::Model(format!("Failed to create server: {}", e)))?;
        
        self.server = Some(server);
        
        println!("[DistTransformerServer] Initialized on {}", self.config.interface_name);
        println!("  Will execute layers {}-{}", 
            self.config.start_remote_layer,
            self.config.start_remote_layer + self.config.remote_layers - 1);
        
        Ok(())
    }
    
    pub fn set_forward_callback(&mut self, cb: ForwardCallback) {
        if let Some(ref mut server) = self.server {
            server.set_forward_callback(cb);
        }
    }
    
    pub fn set_backward_callback(&mut self, cb: BackwardCallback) {
        if let Some(ref mut server) = self.server {
            server.set_backward_callback(cb);
        }
    }
    
    pub fn run(&mut self, max_messages: Option<usize>) {
        if let Some(ref mut server) = self.server {
            server.run(max_messages);
        }
    }
    
    pub fn process_one_message(&mut self, timeout_ms: u32) -> bool {
        if let Some(ref mut server) = self.server {
            server.process_next_message(timeout_ms).unwrap_or(false)
        } else {
            false
        }
    }
    
    pub fn is_running(&self) -> bool {
        self.server.as_ref()
            .map(|s| s.state() == ConnectionState::Connected)
            .unwrap_or(false)
    }
    
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }
}

// Benchmark function
pub fn benchmark_distributed(transformer: &mut DistributedTransformer, iterations: usize) -> TimingStats {
    let mut stats = TimingStats::default();
    
    let input_size = transformer.config().seq_len * transformer.config().embed_dim;
    let mut input = vec![1.0f32; input_size];
    
    let start_time = std::time::Instant::now();
    
    for i in 0..iterations {
        match transformer.forward(&input) {
            Ok(output) => {
                if output.is_empty() {
                    eprintln!("Forward pass failed at iteration {}", i);
                    return stats;
                }
                input = output;
            }
            Err(e) => {
                eprintln!("Forward pass error at iteration {}: {}", i, e);
                return stats;
            }
        }
    }
    
    let after_forward = std::time::Instant::now();
    
    let mut grad_output = vec![0.1f32; input_size];
    for i in 0..iterations {
        match transformer.backward(&grad_output) {
            Ok(grad) => {
                if grad.is_empty() {
                    eprintln!("Backward pass failed at iteration {}", i);
                    return stats;
                }
                grad_output = grad;
            }
            Err(e) => {
                eprintln!("Backward pass error at iteration {}: {}", i, e);
                return stats;
            }
        }
    }
    
    let end_time = std::time::Instant::now();
    
    let forward_ms = after_forward.duration_since(start_time).as_secs_f64() * 1000.0;
    let backward_ms = end_time.duration_since(after_forward).as_secs_f64() * 1000.0;
    
    stats.forward_ms = forward_ms / iterations as f64;
    stats.backward_ms = backward_ms / iterations as f64;
    stats.total_ms = (forward_ms + backward_ms) / iterations as f64;
    stats.elements_processed = input_size;
    
    stats
}
