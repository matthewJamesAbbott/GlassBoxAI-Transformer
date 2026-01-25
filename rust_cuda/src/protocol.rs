// Layer 2 Ethernet Protocol for Distributed Transformer Inference
// DTX Protocol - Raw Ethernet frames for low-latency tensor transfer

use std::io;
use bytemuck::{Pod, Zeroable};

// Protocol constants
pub const DTX_ETHERTYPE: u16 = 0x9998;
pub const DTX_MAX_PAYLOAD: usize = 1472;
pub const DTX_VERSION: u8 = 1;
pub const DTX_MAGIC: u32 = 0xDEADBEEF;

pub const DTX_CONNECT_TIMEOUT: u32 = 5000;
pub const DTX_FRAME_TIMEOUT: u32 = 10000;
pub const DTX_RETRY_MAX: u32 = 3;

// Message types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    HandshakeReq = 1,
    HandshakeAck = 2,
    LayerConfig = 10,
    LayerConfigAck = 11,
    ForwardStart = 20,
    ForwardChunk = 21,
    ForwardDone = 22,
    ForwardResult = 30,
    ForwardComplete = 31,
    BackwardStart = 40,
    BackwardChunk = 41,
    BackwardDone = 42,
    BackwardResult = 50,
    BackwardComplete = 51,
    Ping = 100,
    Pong = 101,
    ErrorMsg = 200,
    Disconnect = 201,
}

impl From<u8> for MessageType {
    fn from(v: u8) -> Self {
        match v {
            1 => MessageType::HandshakeReq,
            2 => MessageType::HandshakeAck,
            10 => MessageType::LayerConfig,
            11 => MessageType::LayerConfigAck,
            20 => MessageType::ForwardStart,
            21 => MessageType::ForwardChunk,
            22 => MessageType::ForwardDone,
            30 => MessageType::ForwardResult,
            31 => MessageType::ForwardComplete,
            40 => MessageType::BackwardStart,
            41 => MessageType::BackwardChunk,
            42 => MessageType::BackwardDone,
            50 => MessageType::BackwardResult,
            51 => MessageType::BackwardComplete,
            100 => MessageType::Ping,
            101 => MessageType::Pong,
            200 => MessageType::ErrorMsg,
            201 => MessageType::Disconnect,
            _ => MessageType::ErrorMsg,
        }
    }
}

// Protocol header - 24 bytes, packed
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, packed)]
pub struct DTXHeader {
    pub magic: u32,
    pub version: u8,
    pub msg_type: u8,
    pub sequence_num: u16,
    pub payload_len: u32,
    pub checksum: u32,
    pub flags: u32,
    pub reserved: u32,
}

impl DTXHeader {
    pub fn new(msg_type: MessageType, seq: u16, payload: Option<&[u8]>) -> Self {
        let payload_len = payload.map(|p| p.len() as u32).unwrap_or(0);
        let checksum = payload.map(|p| crc32_simple(p)).unwrap_or(0);
        
        Self {
            magic: DTX_MAGIC,
            version: DTX_VERSION,
            msg_type: msg_type as u8,
            sequence_num: seq,
            payload_len,
            checksum,
            flags: 0,
            reserved: 0,
        }
    }
    
    pub fn verify(&self) -> bool {
        self.magic == DTX_MAGIC && self.version == DTX_VERSION
    }
    
    pub fn verify_checksum(&self, payload: &[u8]) -> bool {
        if self.payload_len == 0 {
            return self.checksum == 0;
        }
        crc32_simple(payload) == self.checksum
    }
    
    pub fn message_type(&self) -> MessageType {
        MessageType::from(self.msg_type)
    }
}

// Handshake request
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, packed)]
pub struct HandshakeReq {
    pub client_id: u32,
    pub seq_batch_size: u16,
    pub embed_dim: u16,
    pub ffn_dim: u32,
    pub num_heads: u8,
    pub num_kv_heads: u8,
}

// Handshake acknowledgement
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, packed)]
pub struct HandshakeAck {
    pub server_id: u32,
    pub has_gpu: u8,
    pub max_concurrent: u8,
    pub protocol_ver: u16,
}

// Layer configuration
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, packed)]
pub struct LayerConfig {
    pub start_layer: u8,
    pub num_layers: u8,
    pub keep_activations: u8,
    pub reserved: u8,
    pub total_params: u32,
}

// Forward chunk header
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, packed)]
pub struct ForwardChunk {
    pub chunk_id: u32,
    pub seq_start: u32,
    pub seq_len: u16,
    pub embed_dim: u16,
    pub data_size: u32,
}

// Forward result header
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, packed)]
pub struct ForwardResult {
    pub chunk_id: u32,
    pub seq_start: u32,
    pub seq_len: u16,
    pub output_dim: u16,
    pub data_size: u32,
    pub activation_size: u32,
}

// Backward chunk header
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, packed)]
pub struct BackwardChunk {
    pub chunk_id: u32,
    pub seq_start: u32,
    pub seq_len: u16,
    pub grad_dim: u16,
    pub data_size: u32,
}

// Backward result header
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, packed)]
pub struct BackwardResult {
    pub chunk_id: u32,
    pub seq_start: u32,
    pub seq_len: u16,
    pub grad_dim: u16,
    pub data_size: u32,
    pub param_grad_size: u32,
}

// Error message header
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, packed)]
pub struct ErrorMessage {
    pub error_code: u16,
    pub severity: u16,
    pub context_len: u32,
}

// CRC32 checksum (simple implementation)
pub fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB88320u32
            } else {
                crc >> 1
            };
        }
    }
    crc ^ 0xFFFFFFFFu32
}

// Ethernet frame structure
#[derive(Clone)]
pub struct EthernetFrame {
    pub dest_mac: [u8; 6],
    pub src_mac: [u8; 6],
    pub ether_type: u16,
    pub payload: Vec<u8>,
}

impl EthernetFrame {
    pub fn new() -> Self {
        Self {
            dest_mac: [0; 6],
            src_mac: [0; 6],
            ether_type: DTX_ETHERTYPE,
            payload: Vec::new(),
        }
    }
    
    pub fn total_size(&self) -> usize {
        14 + self.payload.len()
    }
    
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.total_size());
        bytes.extend_from_slice(&self.dest_mac);
        bytes.extend_from_slice(&self.src_mac);
        bytes.extend_from_slice(&self.ether_type.to_be_bytes());
        bytes.extend_from_slice(&self.payload);
        bytes
    }
    
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 14 {
            return None;
        }
        
        let mut frame = Self::new();
        frame.dest_mac.copy_from_slice(&data[0..6]);
        frame.src_mac.copy_from_slice(&data[6..12]);
        frame.ether_type = u16::from_be_bytes([data[12], data[13]]);
        frame.payload = data[14..].to_vec();
        Some(frame)
    }
}

impl Default for EthernetFrame {
    fn default() -> Self {
        Self::new()
    }
}

// Connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Error,
}

// MAC address utilities
pub fn mac_to_string(mac: &[u8; 6]) -> String {
    format!(
        "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]
    )
}

pub fn string_to_mac(s: &str) -> Option<[u8; 6]> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 6 {
        return None;
    }
    
    let mut mac = [0u8; 6];
    for (i, part) in parts.iter().enumerate() {
        mac[i] = u8::from_str_radix(part, 16).ok()?;
    }
    Some(mac)
}

pub fn compare_mac(mac1: &[u8; 6], mac2: &[u8; 6]) -> bool {
    mac1 == mac2
}

// Tensor serialization utilities
pub fn serialize_tensor(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

pub fn deserialize_tensor(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

// Distributed configuration
#[derive(Clone)]
pub struct DistributedConfig {
    pub seq_len: usize,
    pub embed_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub total_layers: usize,
    pub local_layers: usize,
    pub remote_layers: usize,
    pub start_remote_layer: usize,
    pub cache_activations: bool,
    pub cache_gradients: bool,
    pub interface_name: String,
    pub server_mac: [u8; 6],
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            seq_len: 512,
            embed_dim: 768,
            ffn_dim: 3072,
            num_heads: 12,
            num_kv_heads: 12,
            total_layers: 12,
            local_layers: 6,
            remote_layers: 6,
            start_remote_layer: 6,
            cache_activations: true,
            cache_gradients: true,
            interface_name: "eth0".to_string(),
            server_mac: [0; 6],
        }
    }
}

impl DistributedConfig {
    pub fn validate(&self) -> bool {
        self.local_layers + self.remote_layers == self.total_layers
            && self.start_remote_layer + self.remote_layers == self.total_layers
    }
    
    pub fn create_symmetric(total_layers: usize, embed_dim: usize, ffn_dim: usize, num_heads: usize) -> Self {
        let local = total_layers / 2;
        let remote = total_layers - local;
        
        Self {
            total_layers,
            embed_dim,
            ffn_dim,
            num_heads,
            num_kv_heads: num_heads,
            local_layers: local,
            remote_layers: remote,
            start_remote_layer: local,
            ..Default::default()
        }
    }
}

// Timing stats for benchmarking
#[derive(Default, Clone)]
pub struct TimingStats {
    pub forward_ms: f64,
    pub backward_ms: f64,
    pub total_ms: f64,
    pub elements_processed: usize,
}

#[cfg(target_os = "linux")]
pub mod linux_raw_socket {
    use super::*;
    
    // Linux socket constants
    const AF_PACKET: i32 = 17;
    const SOCK_RAW: i32 = 3;
    const ETH_P_ALL: u16 = 0x0003;
    const SIOCGIFINDEX: u64 = 0x8933;
    const SIOCGIFHWADDR: u64 = 0x8927;
    const ETH_ALEN: usize = 6;
    const IFNAMSIZ: usize = 16;
    
    #[repr(C)]
    struct sockaddr_ll {
        sll_family: u16,
        sll_protocol: u16,
        sll_ifindex: i32,
        sll_hatype: u16,
        sll_pkttype: u8,
        sll_halen: u8,
        sll_addr: [u8; 8],
    }
    
    #[repr(C)]
    struct ifreq {
        ifr_name: [u8; IFNAMSIZ],
        ifr_data: [u8; 24],
    }
    
    extern "C" {
        fn socket(domain: i32, sock_type: i32, protocol: i32) -> i32;
        fn bind(sockfd: i32, addr: *const sockaddr_ll, addrlen: u32) -> i32;
        fn sendto(sockfd: i32, buf: *const u8, len: usize, flags: i32,
                  dest_addr: *const sockaddr_ll, addrlen: u32) -> isize;
        fn recvfrom(sockfd: i32, buf: *mut u8, len: usize, flags: i32,
                    src_addr: *mut sockaddr_ll, addrlen: *mut u32) -> isize;
        fn close(fd: i32) -> i32;
        fn ioctl(fd: i32, request: u64, ...) -> i32;
        fn select(nfds: i32, readfds: *mut libc::fd_set, writefds: *mut libc::fd_set,
                  exceptfds: *mut libc::fd_set, timeout: *mut libc::timeval) -> i32;
    }
    
    pub struct RawSocket {
        fd: i32,
        if_index: i32,
        local_mac: [u8; 6],
        #[allow(dead_code)]
        interface_name: String,
    }
    
    impl RawSocket {
        pub fn new(interface: &str) -> io::Result<Self> {
            let fd = unsafe { socket(AF_PACKET, SOCK_RAW, (ETH_P_ALL as u16).to_be() as i32) };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            
            // Get interface index
            let mut ifr: ifreq = unsafe { std::mem::zeroed() };
            let name_bytes = interface.as_bytes();
            ifr.ifr_name[..name_bytes.len().min(IFNAMSIZ - 1)]
                .copy_from_slice(&name_bytes[..name_bytes.len().min(IFNAMSIZ - 1)]);
            
            if unsafe { ioctl(fd, SIOCGIFINDEX, &mut ifr) } < 0 {
                unsafe { close(fd) };
                return Err(io::Error::last_os_error());
            }
            
            let if_index = i32::from_ne_bytes([
                ifr.ifr_data[0], ifr.ifr_data[1], ifr.ifr_data[2], ifr.ifr_data[3]
            ]);
            
            // Get MAC address
            if unsafe { ioctl(fd, SIOCGIFHWADDR, &mut ifr) } < 0 {
                unsafe { close(fd) };
                return Err(io::Error::last_os_error());
            }
            
            let mut local_mac = [0u8; 6];
            local_mac.copy_from_slice(&ifr.ifr_data[0..6]);
            
            // Bind to interface
            let addr = sockaddr_ll {
                sll_family: AF_PACKET as u16,
                sll_protocol: 0,
                sll_ifindex: if_index,
                sll_hatype: 0,
                sll_pkttype: 0,
                sll_halen: 0,
                sll_addr: [0; 8],
            };
            
            if unsafe { bind(fd, &addr, std::mem::size_of::<sockaddr_ll>() as u32) } < 0 {
                unsafe { close(fd) };
                return Err(io::Error::last_os_error());
            }
            
            Ok(Self {
                fd,
                if_index,
                local_mac,
                interface_name: interface.to_string(),
            })
        }
        
        pub fn local_mac(&self) -> &[u8; 6] {
            &self.local_mac
        }
        
        pub fn send_frame(&self, dest_mac: &[u8; 6], payload: &[u8]) -> io::Result<usize> {
            let mut frame = Vec::with_capacity(14 + payload.len());
            frame.extend_from_slice(dest_mac);
            frame.extend_from_slice(&self.local_mac);
            frame.extend_from_slice(&DTX_ETHERTYPE.to_be_bytes());
            frame.extend_from_slice(payload);
            
            let addr = sockaddr_ll {
                sll_family: AF_PACKET as u16,
                sll_protocol: 0,
                sll_ifindex: self.if_index,
                sll_hatype: 0,
                sll_pkttype: 0,
                sll_halen: ETH_ALEN as u8,
                sll_addr: {
                    let mut arr = [0u8; 8];
                    arr[..6].copy_from_slice(dest_mac);
                    arr
                },
            };
            
            let sent = unsafe {
                sendto(
                    self.fd,
                    frame.as_ptr(),
                    frame.len(),
                    0,
                    &addr,
                    std::mem::size_of::<sockaddr_ll>() as u32,
                )
            };
            
            if sent < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(sent as usize)
            }
        }
        
        pub fn receive_frame(&self, timeout_ms: u32) -> io::Result<Option<EthernetFrame>> {
            // Set up select with timeout
            let mut readfds: libc::fd_set = unsafe { std::mem::zeroed() };
            unsafe {
                libc::FD_ZERO(&mut readfds);
                libc::FD_SET(self.fd, &mut readfds);
            }
            
            let mut tv = libc::timeval {
                tv_sec: (timeout_ms / 1000) as i64,
                tv_usec: ((timeout_ms % 1000) * 1000) as i64,
            };
            
            let ret = unsafe {
                select(self.fd + 1, &mut readfds, std::ptr::null_mut(),
                       std::ptr::null_mut(), &mut tv)
            };
            
            if ret <= 0 {
                return Ok(None);
            }
            
            let mut buffer = vec![0u8; 2048];
            let mut src_addr: sockaddr_ll = unsafe { std::mem::zeroed() };
            let mut addr_len = std::mem::size_of::<sockaddr_ll>() as u32;
            
            let recv_len = unsafe {
                recvfrom(
                    self.fd,
                    buffer.as_mut_ptr(),
                    buffer.len(),
                    0,
                    &mut src_addr,
                    &mut addr_len,
                )
            };
            
            if recv_len < 14 {
                return Ok(None);
            }
            
            let frame = EthernetFrame::from_bytes(&buffer[..recv_len as usize]);
            
            // Filter for our DTX ethertype
            if let Some(ref f) = frame {
                if f.ether_type != DTX_ETHERTYPE {
                    return Ok(None);
                }
            }
            
            Ok(frame)
        }
    }
    
    impl Drop for RawSocket {
        fn drop(&mut self) {
            unsafe { close(self.fd) };
        }
    }
}

// Re-export for Linux
#[cfg(target_os = "linux")]
pub use linux_raw_socket::RawSocket;

// Stub for non-Linux platforms
#[cfg(not(target_os = "linux"))]
pub struct RawSocket;

#[cfg(not(target_os = "linux"))]
impl RawSocket {
    pub fn new(_interface: &str) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Raw sockets only supported on Linux"
        ))
    }
    
    pub fn local_mac(&self) -> &[u8; 6] {
        &[0; 6]
    }
    
    pub fn send_frame(&self, _dest_mac: &[u8; 6], _payload: &[u8]) -> io::Result<usize> {
        Err(io::Error::new(io::ErrorKind::Unsupported, "Not supported"))
    }
    
    pub fn receive_frame(&self, _timeout_ms: u32) -> io::Result<Option<EthernetFrame>> {
        Err(io::Error::new(io::ErrorKind::Unsupported, "Not supported"))
    }
}
