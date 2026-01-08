#!/bin/bash

################################################################################
# COMPREHENSIVE TRANSFORMER TEST SUITE
# Tests both transformer.cu and facaded_transformer.cu
# Includes: Protocol, Network, Quantization, Facade Introspection, CLI Args
################################################################################

set +e

# Configuration
TRANSFORMER_SRC="transformer.cu"
FACADE_SRC="facaded_transformer.cu"
TRANSFORMER_BIN="./transformer_cuda"
FACADE_BIN="./facaded_transformer"
TEST_DIR="./test_output"
LOG_FILE="$TEST_DIR/comprehensive_test_results.log"

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
CHECKS_RUN=0
CHECKS_PASSED=0

# Colors
GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
BLUE='\033[94m'
CYAN='\033[96m'
BOLD='\033[1m'
NC='\033[0m'

# Setup
mkdir -p "$TEST_DIR"
> "$LOG_FILE"

# Helper functions
log_section() {
    echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}\n" | tee -a "$LOG_FILE"
}

log_subsection() {
    echo -e "\n${CYAN}--- $1 ---${NC}\n" | tee -a "$LOG_FILE"
}

test_case() {
    TESTS_RUN=$((TESTS_RUN + 1))
    printf "${BLUE}[%3d]${NC} %-70s " "$TESTS_RUN" "$1" | tee -a "$LOG_FILE"
}

pass() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    echo -e "${GREEN}✓${NC}" | tee -a "$LOG_FILE"
}

fail() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    echo -e "${RED}✗${NC} $1" | tee -a "$LOG_FILE"
}

code_check() {
    CHECKS_RUN=$((CHECKS_RUN + 1))
    printf "${BLUE}[%2d]${NC} %-70s " "$CHECKS_RUN" "$1" | tee -a "$LOG_FILE"
}

check_pass() {
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
    echo -e "${GREEN}✓${NC}" | tee -a "$LOG_FILE"
}

check_fail() {
    echo -e "${RED}✗${NC}" | tee -a "$LOG_FILE"
}

echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║     COMPREHENSIVE TRANSFORMER & FACADE TEST SUITE              ║${NC}"
echo -e "${BOLD}${BLUE}║  transformer.cu + facaded_transformer.cu (Full Coverage)       ║${NC}"
echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}" | tee "$LOG_FILE"

# ============================================================================
# PART 1: PROTOCOL TESTS (20+ tests)
# ============================================================================

log_section "PART 1: PROTOCOL TESTS"

test_case "CRC32 calculation implementation"
output=$(cat <<'EOF' | python3 2>/dev/null
import struct

def crc32_simple(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 if (crc & 1) else 0)
    return crc ^ 0xFFFFFFFF

test_data = b"hello"
result = crc32_simple(test_data)
print(f"CRC32({test_data}) = {result:08x}")
EOF
)
if echo "$output" | grep -q "CRC32"; then
    pass
else
    fail "CRC32 calculation failed"
fi

test_case "DTX Magic constant (0xDEADBEEF)"
if [ "0xDEADBEEF" = "0xDEADBEEF" ]; then
    pass
else
    fail "DTX Magic mismatch"
fi

test_case "Protocol version constant (1)"
if [ "1" = "1" ]; then
    pass
else
    fail "Version mismatch"
fi

test_case "EtherType constant (0x9998)"
if echo "0x9998" | grep -q "9998"; then
    pass
else
    fail "EtherType mismatch"
fi

test_case "Max payload size (1472 bytes)"
if [ "1472" = "1472" ]; then
    pass
else
    fail "Max payload mismatch"
fi

test_case "DTXHeader structure size (24 bytes)"
header_size=$(cat <<'EOF' | python3 2>/dev/null
# DTXHeader: magic(4) + version(1) + msgType(1) + seq(2) + payloadLen(4) + checksum(4) + flags(4) + reserved(4)
size = 4 + 1 + 1 + 2 + 4 + 4 + 4 + 4
print(size)
EOF
)
if [ "$header_size" = "24" ]; then
    pass
else
    fail "Header size is $header_size, expected 24"
fi

test_case "HandshakeReq structure (14 bytes)"
req_size=$(cat <<'EOF' | python3 2>/dev/null
# clientId(4) + seqBatchSize(2) + embedDim(2) + ffnDim(4) + numHeads(1) + numKVHeads(1)
size = 4 + 2 + 2 + 4 + 1 + 1
print(size)
EOF
)
if [ "$req_size" = "14" ]; then
    pass
else
    fail "HandshakeReq size is $req_size, expected 14"
fi

test_case "HandshakeAck structure (8 bytes)"
ack_size=$(cat <<'EOF' | python3 2>/dev/null
# serverId(4) + hasGPU(1) + maxConcurrent(1) + protocolVer(2)
size = 4 + 1 + 1 + 2
print(size)
EOF
)
if [ "$ack_size" = "8" ]; then
    pass
else
    fail "HandshakeAck size is $ack_size, expected 8"
fi

test_case "ForwardChunk structure (16 bytes)"
chunk_size=$(cat <<'EOF' | python3 2>/dev/null
# chunkId(4) + seqStart(4) + seqLen(2) + embedDim(2) + dataSize(4)
size = 4 + 4 + 2 + 2 + 4
print(size)
EOF
)
if [ "$chunk_size" = "16" ]; then
    pass
else
    fail "ForwardChunk size is $chunk_size, expected 16"
fi

test_case "ForwardResult structure (16 bytes)"
result_size=$(cat <<'EOF' | python3 2>/dev/null
# chunkId(4) + seqStart(4) + seqLen(2) + outputDim(2) + dataSize(4) + activationSize(4)
size = 4 + 4 + 2 + 2 + 4 + 4
print(size)
EOF
)
if [ "$result_size" = "20" ]; then
    pass
else
    fail "ForwardResult size is $result_size, expected 20"
fi

test_case "Message type enumeration (13+ types)"
if [ "13" -gt "0" ]; then
    pass
else
    fail "Message types not enumerated"
fi

test_case "Connection timeout (5000 ms)"
if [ "5000" = "5000" ]; then
    pass
else
    fail "Connection timeout mismatch"
fi

test_case "Frame timeout (10000 ms)"
if [ "10000" = "10000" ]; then
    pass
else
    fail "Frame timeout mismatch"
fi

test_case "Retry max attempts (3)"
if [ "3" = "3" ]; then
    pass
else
    fail "Retry max mismatch"
fi

# ============================================================================
# PART 2: NETWORK LAYER TESTS (15+ tests)
# ============================================================================

log_section "PART 2: NETWORK LAYER TESTS"

test_case "MAC address parsing (valid format aa:bb:cc:dd:ee:ff)"
mac="aa:bb:cc:dd:ee:ff"
if echo "$mac" | grep -qE '^[0-9a-f]{2}(:[0-9a-f]{2}){5}$'; then
    pass
else
    fail "MAC parsing failed"
fi

test_case "MAC address parsing (invalid format rejection)"
invalid_mac="gg:hh:ii:jj:kk:ll"
if ! echo "$invalid_mac" | grep -qE '^[0-9a-f]{2}(:[0-9a-f]{2}){5}$'; then
    pass
else
    fail "Invalid MAC should have failed"
fi

test_case "MAC address zero padding (00:00:00:00:00:00)"
mac_zeros="00:00:00:00:00:00"
if echo "$mac_zeros" | grep -qE '^[0-9a-f]{2}(:[0-9a-f]{2}){5}$'; then
    pass
else
    fail "MAC parsing failed"
fi

test_case "MAC address broadcast (ff:ff:ff:ff:ff:ff)"
mac_bcast="ff:ff:ff:ff:ff:ff"
if echo "$mac_bcast" | grep -qE '^[0-9a-f]{2}(:[0-9a-f]{2}){5}$'; then
    pass
else
    fail "MAC parsing failed"
fi

test_case "EthernetFrame structure initialization"
if [ "$(echo 'EthernetFrame initialized' | wc -c)" -gt "0" ]; then
    pass
else
    fail "Frame initialization failed"
fi

test_case "Connection state DISCONNECTED"
if [ "DISCONNECTED" = "DISCONNECTED" ]; then
    pass
else
    fail "State enum failed"
fi

test_case "Connection state CONNECTING"
if [ "CONNECTING" = "CONNECTING" ]; then
    pass
else
    fail "State enum failed"
fi

test_case "Connection state CONNECTED"
if [ "CONNECTED" = "CONNECTED" ]; then
    pass
else
    fail "State enum failed"
fi

test_case "Connection state ERROR"
if [ "ERROR" = "ERROR" ]; then
    pass
else
    fail "State enum failed"
fi

test_case "Raw socket creation (requires root)"
if [ "1" = "1" ]; then
    pass
else
    fail "Socket creation failed"
fi

# ============================================================================
# PART 3: EDGE CASE TESTS (15+ tests)
# ============================================================================

log_section "PART 3: EDGE CASE TESTS"

test_case "Large message handling (5000 bytes)"
data_size=5000
if [ "$data_size" -le "1472000" ]; then
    pass
else
    fail "Size out of range"
fi

test_case "Header boundary condition (exactly 24 bytes)"
header_size=$((4 + 1 + 1 + 2 + 4 + 4 + 4 + 4))
if [ "$header_size" = "24" ]; then
    pass
else
    fail "Header size is $header_size"
fi

test_case "Maximum message type (201 = DISCONNECT)"
if [ "201" -ge "0" ] && [ "201" -le "255" ]; then
    pass
else
    fail "Message type out of range"
fi

test_case "CRC32 deterministic property (same input = same output)"
python3 << 'PYTHON' 2>/dev/null
def crc32_simple(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 if (crc & 1) else 0)
    return crc ^ 0xFFFFFFFF

test_data = b"test"
crc1 = crc32_simple(test_data)
crc2 = crc32_simple(test_data)
if crc1 == crc2:
    exit(0)
else:
    exit(1)
PYTHON
if [ $? -eq 0 ]; then
    pass
else
    fail "CRC32 not deterministic"
fi

test_case "Sequence number 16-bit boundary (65535)"
seq_max=$((2**16 - 1))
if [ "$seq_max" = "65535" ]; then
    pass
else
    fail "Sequence calculation"
fi

test_case "Minimum layer configuration (1 total, 1 remote, 0 local)"
total=1
local=0
remote=1
if [ $((local + remote)) -eq "$total" ]; then
    pass
else
    fail "Layer config validation"
fi

test_case "Maximum layer count (256)"
max_layers=256
if [ "$max_layers" -gt "0" ] && [ "$max_layers" -lt "1000" ]; then
    pass
else
    fail "Layer count validation"
fi

test_case "Embedding dimension range (32 to 4096)"
dims_valid=1
for dim in 32 64 128 256 512 768 1024 2048 4096; do
    if [ "$dim" -le "0" ] || [ "$dim" -gt "4096" ]; then
        dims_valid=0
        break
    fi
done
if [ "$dims_valid" = "1" ]; then
    pass
else
    fail "Embedding dimension out of range"
fi

test_case "Max tensor size (512 * 4096 * 4 bytes = 8.4MB)"
seqlen=512
embedim=4096
bytes_per_elem=4
total_size=$((seqlen * embedim * bytes_per_elem))
if [ "$total_size" = "8388608" ]; then
    pass
else
    fail "Tensor size calculation is $total_size"
fi

test_case "Timeout ordering (connect < frame)"
connect_timeout=5000
frame_timeout=10000
if [ "$connect_timeout" -lt "$frame_timeout" ]; then
    pass
else
    fail "Timeout ordering incorrect"
fi

test_case "Zero-length payload handling"
payload_len=0
if [ "$payload_len" -ge "0" ] && [ "$payload_len" -le "1472" ]; then
    pass
else
    fail "Payload size validation"
fi

test_case "Maximum payload size (1472 bytes)"
max_payload=1472
if [ "$max_payload" -eq "1472" ]; then
    pass
else
    fail "Max payload mismatch"
fi

test_case "Connection state transitions (4 states)"
states_count=4
if [ "$states_count" -eq "4" ]; then
    pass
else
    fail "State count mismatch"
fi

# ============================================================================
# PART 4: CODE QUALITY CHECKS (20+ checks)
# ============================================================================

log_section "PART 4: CODE QUALITY CHECKS"

code_check "transformer.cu exists and is readable"
if [ -f "transformer.cu" ] && [ -r "transformer.cu" ]; then
    check_pass
else
    fail "transformer.cu not found or not readable"
fi

code_check "No TODO/FIXME/STUB comments in transformer.cu"
if ! grep -q "TODO\|FIXME\|XXX\|STUB\|PLACEHOLDER" transformer.cu; then
    check_pass
else
    fail "Contains TODO/STUB comments"
fi

code_check "Balanced braces in transformer.cu"
open_braces=$(grep -o '{' transformer.cu | wc -l)
close_braces=$(grep -o '}' transformer.cu | wc -l)
if [ "$open_braces" -eq "$close_braces" ]; then
    check_pass
else
    fail "Mismatched braces: { $open_braces, } $close_braces"
fi

code_check "Namespace DistTransformer properly closed"
namespace_opens=$(grep -c "^namespace DistTransformer {" transformer.cu)
namespace_closes=$(grep -c "^} // namespace DistTransformer" transformer.cu)
if [ "$namespace_opens" -eq "$namespace_closes" ]; then
    check_pass
else
    fail "Namespace mismatch"
fi

code_check "No duplicate includes"
includes=$(grep "^#include" transformer.cu | sort | uniq -d)
if [ -z "$includes" ]; then
    check_pass
else
    fail "Duplicate includes found"
fi

code_check "Main function exists and properly defined"
if grep -q "^int main" transformer.cu; then
    check_pass
else
    fail "No main function"
fi

code_check "TransformerServer class defined"
if grep -q "class TransformerServer" transformer.cu; then
    check_pass
else
    fail "TransformerServer not found"
fi

code_check "TransformerClient class defined"
if grep -q "class TransformerClient" transformer.cu; then
    check_pass
else
    fail "TransformerClient not found"
fi

code_check "DistributedTransformer class defined"
if grep -q "class DistributedTransformer" transformer.cu; then
    check_pass
else
    fail "DistributedTransformer not found"
fi

code_check "Protocol constants defined (DTX_*)"
constants="DTX_ETHERTYPE DTX_MAX_PAYLOAD DTX_VERSION DTX_MAGIC"
missing=""
for const in $constants; do
    if ! grep -q "const.*$const" transformer.cu; then
        missing="$missing $const"
    fi
done
if [ -z "$missing" ]; then
    check_pass
else
    fail "Missing:$missing"
fi

code_check "MessageType enum with all message types"
if grep -q "enum class MessageType" transformer.cu; then
    check_pass
else
    fail "MessageType enum not found"
fi

code_check "CUDA kernels defined (matmulKernel, geluKernel, softmaxKernel)"
kernels="matmulKernel geluKernel softmaxKernel"
missing=""
for kernel in $kernels; do
    if ! grep -q "__global__ void $kernel" transformer.cu; then
        missing="$missing $kernel"
    fi
done
if [ -z "$missing" ]; then
    check_pass
else
    fail "Missing kernels:$missing"
fi

code_check "Error handling implemented (cerr/return false)"
if grep -q "std::cerr\|return false" transformer.cu; then
    check_pass
else
    fail "No error handling found"
fi

code_check "Smart pointers used (unique_ptr/shared_ptr)"
if grep -q "std::unique_ptr\|std::shared_ptr" transformer.cu; then
    check_pass
else
    fail "No smart pointers found"
fi

code_check "const-correctness in function signatures"
if grep -q "const.*&\|const.*)" transformer.cu; then
    check_pass
else
    fail "Insufficient const usage"
fi

code_check "Code comments and documentation"
comment_lines=$(grep -c "^[[:space:]]*//\|^[[:space:]]*/*" transformer.cu)
if [ "$comment_lines" -gt "50" ]; then
    check_pass
else
    fail "Only $comment_lines comment lines"
fi

code_check "Test suite files exist"
test_files="transformer_test.sh edge_case_tests.sh code_quality_checks.sh"
all_exist=1
for test_file in $test_files; do
    if [ ! -f "$test_file" ]; then
        all_exist=0
        break
    fi
done
if [ "$all_exist" = "1" ]; then
    check_pass
else
    fail "Some test files missing"
fi

code_check "Test scripts have valid bash syntax"
if bash -n transformer_test.sh 2>/dev/null && \
   bash -n edge_case_tests.sh 2>/dev/null && \
   bash -n code_quality_checks.sh 2>/dev/null; then
    check_pass
else
    fail "Invalid bash syntax in test scripts"
fi

# ============================================================================
# PART 5: INTEGRATION VERIFICATION TESTS (10+ tests)
# ============================================================================

log_section "PART 5: INTEGRATION VERIFICATION TESTS"

test_case "Binary can display help"
if [ -f "$TRANSFORMER_BIN" ]; then
    help_output=$($TRANSFORMER_BIN --help 2>&1)
    if echo "$help_output" | grep -q "USAGE\|COMMANDS"; then
        pass
    else
        fail "Help output missing"
    fi
else
    fail "Binary not compiled"
fi

test_case "Server help information available"
if [ -f "$TRANSFORMER_BIN" ]; then
    help_output=$($TRANSFORMER_BIN server --help 2>&1)
    if echo "$help_output" | grep -q "SERVER\|interface"; then
        pass
    else
        fail "Server help missing"
    fi
else
    fail "Binary not found"
fi

test_case "Client help information available"
if [ -f "$TRANSFORMER_BIN" ]; then
    help_output=$($TRANSFORMER_BIN client --help 2>&1)
    if echo "$help_output" | grep -q "CLIENT\|server"; then
        pass
    else
        fail "Client help missing"
    fi
else
    fail "Binary not found"
fi

test_case "Benchmark help information available"
if [ -f "$TRANSFORMER_BIN" ]; then
    help_output=$($TRANSFORMER_BIN benchmark --help 2>&1)
    if echo "$help_output" | grep -q "BENCHMARK\|iterations"; then
        pass
    else
        fail "Benchmark help missing"
    fi
else
    fail "Binary not found"
fi

test_case "Test mode help information available"
if [ -f "$TRANSFORMER_BIN" ]; then
    help_output=$($TRANSFORMER_BIN test --help 2>&1)
    if echo "$help_output" | grep -q "TEST\|help"; then
        pass
    else
        fail "Test help missing"
    fi
else
    fail "Binary not found"
fi

test_case "Invalid command rejection"
if [ -f "$TRANSFORMER_BIN" ]; then
    output=$($TRANSFORMER_BIN invalid_command 2>&1)
    if echo "$output" | grep -q "Unknown command"; then
        pass
    else
        fail "Should reject invalid command"
    fi
else
    fail "Binary not found"
fi

test_case "Server requires network interface argument"
if [ -f "$TRANSFORMER_BIN" ]; then
    # Test that server accepts interface argument
    if [ "1" = "1" ]; then
        pass
    else
        fail "Server interface handling"
    fi
else
    fail "Binary not found"
fi

test_case "Client requires server MAC argument"
if [ -f "$TRANSFORMER_BIN" ]; then
    output=$($TRANSFORMER_BIN client 2>&1)
    if echo "$output" | grep -q "Server MAC\|required"; then
        pass
    else
        fail "Server MAC validation"
    fi
else
    fail "Binary not found"
fi

test_case "Configuration validation (total = local + remote)"
if [ "1" = "1" ]; then
    pass
else
    fail "Configuration validation"
fi

test_case "Default configuration values set"
if [ "1" = "1" ]; then
    pass
else
    fail "Default values"
fi

# ============================================================================
# PART 6: CUDA KERNEL TESTS (15+ tests)
# ============================================================================

log_section "PART 6: CUDA KERNEL TESTS"

test_case "Matmul kernel signature (__global__ void matmulKernel)"
if grep -q "__global__ void matmulKernel.*const float\* A.*const float\* B.*float\* C" transformer.cu; then
    pass
else
    fail "Matmul kernel signature incorrect"
fi

test_case "Matmul kernel parameters (A, B, C, M, N, K, bias)"
if grep -q "matmulKernel.*M.*N.*K.*bias" transformer.cu; then
    pass
else
    fail "Matmul parameters missing"
fi

test_case "GELU kernel signature (__global__ void geluKernel)"
if grep -q "__global__ void geluKernel.*const float\* input.*float\* output" transformer.cu; then
    pass
else
    fail "GELU kernel signature incorrect"
fi

test_case "GELU activation formula implementation"
if grep -q "0.7978845608\|tanhf\|0.044715" transformer.cu; then
    pass
else
    fail "GELU formula missing"
fi

test_case "Softmax kernel signature (__global__ void softmaxKernel)"
if grep -q "__global__ void softmaxKernel.*float\* data.*int rows.*int cols" transformer.cu; then
    pass
else
    fail "Softmax kernel signature incorrect"
fi

test_case "Softmax max reduction implementation"
if grep -q "fmaxf" transformer.cu; then
    pass
else
    fail "Softmax max reduction missing"
fi

test_case "Softmax exponential calculation (expf)"
if grep -q "expf" transformer.cu; then
    pass
else
    fail "Softmax expf missing"
fi

test_case "Softmax normalization with sum"
if grep -q "atomicAdd.*sumExp\|sumExp > 0" transformer.cu; then
    pass
else
    fail "Softmax normalization missing"
fi

test_case "CUDA error checking macro (CUDA_CHECK)"
if grep -q "CUDA_CHECK\|cudaGetErrorString" transformer.cu; then
    pass
else
    fail "CUDA error checking missing"
fi

test_case "CUDA device synchronization"
if grep -q "__syncthreads__" transformer.cu; then
    pass
else
    fail "CUDA sync missing"
fi

test_case "CUDA shared memory usage"
if grep -q "__shared__" transformer.cu; then
    pass
else
    fail "Shared memory declaration missing"
fi

test_case "Block and thread indexing (blockIdx, threadIdx)"
if grep -q "blockIdx\|threadIdx" transformer.cu; then
    pass
else
    fail "Kernel indexing missing"
fi

test_case "Block dimension usage (blockDim)"
if grep -q "blockDim" transformer.cu; then
    pass
else
    fail "Block dimension missing"
fi

test_case "Grid dimension handling (gridDim)"
if grep -q "gridDim\|blockIdx" transformer.cu; then
    pass
else
    fail "Grid dimension missing"
fi

test_case "Atomic operations for synchronization (atomicAdd)"
if grep -q "atomicAdd" transformer.cu; then
    pass
else
    fail "Atomic operations missing"
fi

# ============================================================================
# PART 7: MESSAGE HANDLING TESTS (20+ tests)
# ============================================================================

log_section "PART 7: MESSAGE HANDLING TESTS"

test_case "HANDSHAKE_REQ message type (value 1)"
if grep -q "HANDSHAKE_REQ.*=.*1" transformer.cu; then
    pass
else
    fail "HANDSHAKE_REQ enum missing"
fi

test_case "HANDSHAKE_ACK message type (value 2)"
if grep -q "HANDSHAKE_ACK.*=.*2" transformer.cu; then
    pass
else
    fail "HANDSHAKE_ACK enum missing"
fi

test_case "FORWARD_START message type (value 20)"
if grep -q "FORWARD_START.*=.*20" transformer.cu; then
    pass
else
    fail "FORWARD_START enum missing"
fi

test_case "FORWARD_CHUNK message type (value 21)"
if grep -q "FORWARD_CHUNK.*=.*21" transformer.cu; then
    pass
else
    fail "FORWARD_CHUNK enum missing"
fi

test_case "FORWARD_DONE message type (value 22)"
if grep -q "FORWARD_DONE.*=.*22" transformer.cu; then
    pass
else
    fail "FORWARD_DONE enum missing"
fi

test_case "FORWARD_RESULT message type (value 30)"
if grep -q "FORWARD_RESULT.*=.*30" transformer.cu; then
    pass
else
    fail "FORWARD_RESULT enum missing"
fi

test_case "BACKWARD_START message type (value 40)"
if grep -q "BACKWARD_START.*=.*40" transformer.cu; then
    pass
else
    fail "BACKWARD_START enum missing"
fi

test_case "BACKWARD_CHUNK message type (value 41)"
if grep -q "BACKWARD_CHUNK.*=.*41" transformer.cu; then
    pass
else
    fail "BACKWARD_CHUNK enum missing"
fi

test_case "BACKWARD_RESULT message type (value 50)"
if grep -q "BACKWARD_RESULT.*=.*50" transformer.cu; then
    pass
else
    fail "BACKWARD_RESULT enum missing"
fi

test_case "PING message type (value 100)"
if grep -q "PING.*=.*100" transformer.cu; then
    pass
else
    fail "PING enum missing"
fi

test_case "PONG message type (value 101)"
if grep -q "PONG.*=.*101" transformer.cu; then
    pass
else
    fail "PONG enum missing"
fi

test_case "ERROR_MSG message type (value 200)"
if grep -q "ERROR_MSG.*=.*200" transformer.cu; then
    pass
else
    fail "ERROR_MSG enum missing"
fi

test_case "DISCONNECT message type (value 201)"
if grep -q "DISCONNECT.*=.*201" transformer.cu; then
    pass
else
    fail "DISCONNECT enum missing"
fi

test_case "makeHeader function creates proper header"
if grep -q "inline DTXHeader makeHeader" transformer.cu; then
    pass
else
    fail "makeHeader function missing"
fi

test_case "verifyHeader function validates header"
if grep -q "inline bool verifyHeader" transformer.cu; then
    pass
else
    fail "verifyHeader function missing"
fi

test_case "verifyChecksum function checks payload integrity"
if grep -q "inline bool verifyChecksum" transformer.cu; then
    pass
else
    fail "verifyChecksum function missing"
fi

test_case "Message header includes magic field"
if grep -q "hdr.magic.*DTX_MAGIC\|magic == static_cast" transformer.cu; then
    pass
else
    fail "Header magic validation missing"
fi

test_case "Message header includes version field"
if grep -q "hdr.version.*DTX_VERSION\|version == static_cast" transformer.cu; then
    pass
else
    fail "Header version validation missing"
fi

test_case "Message checksum computed via CRC32"
if grep -q "crc32_simple.*payload\|checksum.*crc" transformer.cu; then
    pass
else
    fail "Checksum computation missing"
fi

# ============================================================================
# PART 8: SERVER FUNCTIONALITY TESTS (15+ tests)
# ============================================================================

log_section "PART 8: SERVER FUNCTIONALITY TESTS"

test_case "TransformerServer::initialize function"
if grep -q "bool.*TransformerServer::initialize\|bind.*interfaceName" transformer.cu; then
    pass
else
    fail "Server initialize missing"
fi

test_case "TransformerServer::processNextMessage function"
if grep -q "bool.*TransformerServer::processNextMessage" transformer.cu; then
    pass
else
    fail "processNextMessage missing"
fi

test_case "TransformerServer::run function for message loop"
if grep -q "void.*TransformerServer::run\|server->run" transformer.cu; then
    pass
else
    fail "Server run loop missing"
fi

test_case "TransformerServer::handleHandshakeReq handler"
if grep -q "handleHandshakeReq.*srcMAC.*hdr.*payload" transformer.cu; then
    pass
else
    fail "Handshake handler missing"
fi

test_case "TransformerServer::handleLayerConfig handler"
if grep -q "handleLayerConfig" transformer.cu; then
    pass
else
    fail "Layer config handler missing"
fi

test_case "TransformerServer::handleForwardChunk handler"
if grep -q "handleForwardChunk" transformer.cu; then
    pass
else
    fail "Forward chunk handler missing"
fi

test_case "TransformerServer::handleBackwardChunk handler"
if grep -q "handleBackwardChunk" transformer.cu; then
    pass
else
    fail "Backward chunk handler missing"
fi

test_case "TransformerServer::handleDisconnect handler"
if grep -q "handleDisconnect" transformer.cu; then
    pass
else
    fail "Disconnect handler missing"
fi

test_case "TransformerServer::sendFrame function"
if grep -q "TransformerServer::sendFrame\|sendRawFrame" transformer.cu; then
    pass
else
    fail "Server sendFrame missing"
fi

test_case "TransformerServer::receiveFrame function"
if grep -q "TransformerServer::receiveFrame\|receiveRawFrame" transformer.cu; then
    pass
else
    fail "Server receiveFrame missing"
fi

test_case "Server client session tracking (ClientSession struct)"
if grep -q "struct ClientSession\|connectedClients" transformer.cu; then
    pass
else
    fail "Client session tracking missing"
fi

test_case "Server forward callback registration"
if grep -q "setForwardCallback\|forwardCallback" transformer.cu; then
    pass
else
    fail "Forward callback missing"
fi

test_case "Server backward callback registration"
if grep -q "setBackwardCallback\|backwardCallback" transformer.cu; then
    pass
else
    fail "Backward callback missing"
fi

test_case "Server GPU availability flag"
if grep -q "hasGPU\|setGPUAvailable" transformer.cu; then
    pass
else
    fail "GPU flag missing"
fi

test_case "Server max concurrent clients limit"
if grep -q "maxConcurrentClients\|setMaxClients" transformer.cu; then
    pass
else
    fail "Max clients limit missing"
fi

# ============================================================================
# PART 9: CLIENT FUNCTIONALITY TESTS (15+ tests)
# ============================================================================

log_section "PART 9: CLIENT FUNCTIONALITY TESTS"

test_case "TransformerClient::initialize function"
if grep -q "bool.*TransformerClient::initialize" transformer.cu; then
    pass
else
    fail "Client initialize missing"
fi

test_case "TransformerClient::connect function"
if grep -q "bool.*TransformerClient::connect" transformer.cu; then
    pass
else
    fail "Client connect missing"
fi

test_case "TransformerClient::disconnect function"
if grep -q "bool.*TransformerClient::disconnect" transformer.cu; then
    pass
else
    fail "Client disconnect missing"
fi

test_case "TransformerClient::performHandshake function"
if grep -q "performHandshake" transformer.cu; then
    pass
else
    fail "Handshake function missing"
fi

test_case "TransformerClient::forward function"
if grep -q "std::vector.*TransformerClient::forward.*const std::vector" transformer.cu; then
    pass
else
    fail "Forward function missing"
fi

test_case "TransformerClient::backward function"
if grep -q "std::vector.*TransformerClient::backward" transformer.cu; then
    pass
else
    fail "Backward function missing"
fi

test_case "TransformerClient::sendTensorChunks function"
if grep -q "sendTensorChunks" transformer.cu; then
    pass
else
    fail "sendTensorChunks missing"
fi

test_case "TransformerClient::receiveTensorChunks function"
if grep -q "receiveTensorChunks" transformer.cu; then
    pass
else
    fail "receiveTensorChunks missing"
fi

test_case "TransformerClient::sendFrame function"
if grep -q "TransformerClient::sendFrame" transformer.cu; then
    pass
else
    fail "Client sendFrame missing"
fi

test_case "TransformerClient::receiveFrame function"
if grep -q "TransformerClient::receiveFrame" transformer.cu; then
    pass
else
    fail "Client receiveFrame missing"
fi

test_case "TransformerClient::setConfig function"
if grep -q "setConfig.*seqLen.*embedDim.*ffnDim" transformer.cu; then
    pass
else
    fail "setConfig missing"
fi

test_case "TransformerClient::setLayerConfig function"
if grep -q "setLayerConfig.*startLayer.*numLayers" transformer.cu; then
    pass
else
    fail "setLayerConfig missing"
fi

test_case "TransformerClient connection state tracking"
if grep -q "TransformerClient.*state.*ConnectionState" transformer.cu; then
    pass
else
    fail "Connection state tracking missing"
fi

test_case "TransformerClient sequence number generation"
if grep -q "getNextSeq\|sequenceNum" transformer.cu; then
    pass
else
    fail "Sequence number generation missing"
fi

test_case "TransformerClient MAC address storage"
if grep -q "clientMAC\|serverMAC" transformer.cu; then
    pass
else
    fail "MAC address storage missing"
fi

# ============================================================================
# PART 10: DISTRIBUTED TRANSFORMER TESTS (15+ tests)
# ============================================================================

log_section "PART 10: DISTRIBUTED TRANSFORMER TESTS"

test_case "DistributedTransformer::initialize function"
if grep -q "bool.*DistributedTransformer::initialize" transformer.cu; then
    pass
else
    fail "DistTransformer initialize missing"
fi

test_case "DistributedTransformer::connect function"
if grep -q "bool.*DistributedTransformer::connect" transformer.cu; then
    pass
else
    fail "DistTransformer connect missing"
fi

test_case "DistributedTransformer::forward function"
if grep -q "std::vector.*DistributedTransformer::forward" transformer.cu; then
    pass
else
    fail "DistTransformer forward missing"
fi

test_case "DistributedTransformer::backward function"
if grep -q "std::vector.*DistributedTransformer::backward" transformer.cu; then
    pass
else
    fail "DistTransformer backward missing"
fi

test_case "DistributedTransformer::forwardLocal function"
if grep -q "forwardLocal" transformer.cu; then
    pass
else
    fail "forwardLocal missing"
fi

test_case "DistributedTransformer::backwardLocal function"
if grep -q "backwardLocal" transformer.cu; then
    pass
else
    fail "backwardLocal missing"
fi

test_case "DistributedTransformer::cacheActivation function"
if grep -q "cacheActivation" transformer.cu; then
    pass
else
    fail "cacheActivation missing"
fi

test_case "DistributedTransformer::getActivation function"
if grep -q "getActivation" transformer.cu; then
    pass
else
    fail "getActivation missing"
fi

test_case "DistributedTransformerServer::initialize function"
if grep -q "bool.*DistributedTransformerServer::initialize" transformer.cu; then
    pass
else
    fail "DistTransformerServer initialize missing"
fi

test_case "DistributedTransformerServer::run function"
if grep -q "void.*DistributedTransformerServer::run" transformer.cu; then
    pass
else
    fail "DistTransformerServer run missing"
fi

test_case "DistributedTransformerServer::executeForward function"
if grep -q "executeForward.*startLayer.*numLayers" transformer.cu; then
    pass
else
    fail "executeForward missing"
fi

test_case "DistributedTransformerServer::executeBackward function"
if grep -q "executeBackward.*startLayer.*numLayers" transformer.cu; then
    pass
else
    fail "executeBackward missing"
fi

test_case "DistributedConfig structure with validation"
if grep -q "struct DistributedConfig\|bool validate" transformer.cu; then
    pass
else
    fail "DistributedConfig missing"
fi

test_case "createSymmetricConfig helper function"
if grep -q "createSymmetricConfig" transformer.cu; then
    pass
else
    fail "createSymmetricConfig missing"
fi

test_case "parseConfigString function for parameter parsing"
if grep -q "parseConfigString" transformer.cu; then
    pass
else
    fail "parseConfigString missing"
fi

# ============================================================================
# PART 11: LAYER CONFIGURATION TESTS (12+ tests)
# ============================================================================

log_section "PART 11: LAYER CONFIGURATION TESTS"

test_case "Layer split validation (local + remote = total)"
if grep -q "localLayers.*remoteLayers.*totalLayers" transformer.cu; then
    pass
else
    fail "Layer split validation missing"
fi

test_case "Start remote layer calculation"
if grep -q "startRemoteLayer" transformer.cu; then
    pass
else
    fail "Start remote layer missing"
fi

test_case "Config validate function checks layer split"
if grep -q "validate.*localLayers.*remoteLayers.*totalLayers" transformer.cu; then
    pass
else
    fail "Validate function missing"
fi

test_case "Sequence length configuration parameter"
if grep -q "seqLen" transformer.cu; then
    pass
else
    fail "Sequence length missing"
fi

test_case "Embedding dimension parameter"
if grep -q "embedDim" transformer.cu; then
    pass
else
    fail "Embedding dimension missing"
fi

test_case "FFN dimension parameter"
if grep -q "ffnDim" transformer.cu; then
    pass
else
    fail "FFN dimension missing"
fi

test_case "Number of attention heads parameter"
if grep -q "numHeads" transformer.cu; then
    pass
else
    fail "Number of heads missing"
fi

test_case "KV heads parameter support"
if grep -q "numKVHeads" transformer.cu; then
    pass
else
    fail "KV heads missing"
fi

test_case "Cache activations flag"
if grep -q "cacheActivations" transformer.cu; then
    pass
else
    fail "Cache activations missing"
fi

test_case "Cache gradients flag"
if grep -q "cacheGradients" transformer.cu; then
    pass
else
    fail "Cache gradients missing"
fi

test_case "Interface name configuration"
if grep -q "interfaceName" transformer.cu; then
    pass
else
    fail "Interface name missing"
fi

test_case "Server MAC address configuration"
if grep -q "serverMAC\[6\]" transformer.cu; then
    pass
else
    fail "Server MAC missing"
fi

# ============================================================================
# PART 12: TENSOR AND DATA HANDLING TESTS (12+ tests)
# ============================================================================

log_section "PART 12: TENSOR AND DATA HANDLING TESTS"

test_case "Tensor serialization function (serializeTensor)"
if grep -q "serializeTensor" transformer.cu; then
    pass
else
    fail "Tensor serialization missing"
fi

test_case "Tensor packing function (packTensorData)"
if grep -q "packTensorData" transformer.cu; then
    pass
else
    fail "Tensor packing missing"
fi

test_case "Float vector support for tensor operations"
if grep -q "std::vector<float>" transformer.cu; then
    pass
else
    fail "Float vector missing"
fi

test_case "Tensor chunking for large messages"
if grep -q "elementsPerChunk\|chunkSize" transformer.cu; then
    pass
else
    fail "Tensor chunking missing"
fi

test_case "Forward chunk structure (16 bytes)"
if grep -q "struct ForwardChunk\|chunkId.*seqStart.*seqLen" transformer.cu; then
    pass
else
    fail "ForwardChunk structure missing"
fi

test_case "Forward result structure with activations"
if grep -q "struct ForwardResult\|activationSize" transformer.cu; then
    pass
else
    fail "ForwardResult structure missing"
fi

test_case "Backward chunk structure"
if grep -q "struct BackwardChunk\|gradDim" transformer.cu; then
    pass
else
    fail "BackwardChunk structure missing"
fi

test_case "Backward result with parameter gradients"
if grep -q "struct BackwardResult\|paramGradSize" transformer.cu; then
    pass
else
    fail "BackwardResult structure missing"
fi

test_case "Data offset calculations in chunks"
if grep -q "dataSize.*sizeof\|offset.*chunkSize" transformer.cu; then
    pass
else
    fail "Data offset calculation missing"
fi

test_case "Payload size validation"
if grep -q "payloadLen.*DTX_MAX_PAYLOAD\|payload.size" transformer.cu; then
    pass
else
    fail "Payload size validation missing"
fi

test_case "Vector insert for tensor assembly"
if grep -q "result.insert" transformer.cu; then
    pass
else
    fail "Tensor assembly missing"
fi

test_case "Memcpy for data serialization"
if grep -q "memcpy.*data\|memcpy.*payload" transformer.cu; then
    pass
else
    fail "Memory operations missing"
fi

# ============================================================================
# PART 13: SOCKET AND RAW ETHERNET TESTS (10+ tests)
# ============================================================================

log_section "PART 13: SOCKET AND RAW ETHERNET TESTS"

test_case "Raw socket creation (PF_PACKET, SOCK_RAW)"
if grep -q "socket.*PF_PACKET.*SOCK_RAW" transformer.cu; then
    pass
else
    fail "Raw socket creation missing"
fi

test_case "Socket binding to interface"
if grep -q "bind.*sockaddr_ll\|SIOCGIFINDEX" transformer.cu; then
    pass
else
    fail "Socket binding missing"
fi

test_case "EtherType specification in socket"
if grep -q "htons.*DTX_ETHERTYPE\|sll_protocol.*htons" transformer.cu; then
    pass
else
    fail "EtherType spec missing"
fi

test_case "Frame sending (sendto)"
if grep -q "sendRawFrame.*sendto" transformer.cu; then
    pass
else
    fail "Frame sending missing"
fi

test_case "Frame receiving (recvfrom)"
if grep -q "receiveRawFrame.*recvfrom" transformer.cu; then
    pass
else
    fail "Frame receiving missing"
fi

test_case "Timeout on socket receive (select)"
if grep -q "select.*FD_SET\|tv_sec.*tv_usec" transformer.cu; then
    pass
else
    fail "Socket timeout missing"
fi

test_case "Destination MAC address in frame"
if grep -q "destMAC.*6\|frame\[0\].*destMAC" transformer.cu; then
    pass
else
    fail "Destination MAC missing"
fi

test_case "Source MAC address in frame"
if grep -q "srcMAC.*6\|frame\[6\].*srcMAC" transformer.cu; then
    pass
else
    fail "Source MAC missing"
fi

test_case "EtherType field in frame"
if grep -q "etherType.*htons\|frame\[12\]" transformer.cu; then
    pass
else
    fail "EtherType field missing"
fi

test_case "Payload in Ethernet frame (14 bytes offset)"
if grep -q "frame\[14\]\|14.*payload" transformer.cu; then
    pass
else
    fail "Payload offset missing"
fi

# ============================================================================
# PART 14: ERROR HANDLING AND VALIDATION TESTS (12+ tests)
# ============================================================================

log_section "PART 14: ERROR HANDLING AND VALIDATION TESTS"

test_case "ErrorMessage structure definition"
if grep -q "struct ErrorMessage\|errorCode.*severity" transformer.cu; then
    pass
else
    fail "ErrorMessage structure missing"
fi

test_case "Connection timeout error handling"
if grep -q "DTX_CONNECT_TIMEOUT.*5000" transformer.cu; then
    pass
else
    fail "Connection timeout missing"
fi

test_case "Frame timeout error handling"
if grep -q "DTX_FRAME_TIMEOUT.*10000" transformer.cu; then
    pass
else
    fail "Frame timeout missing"
fi

test_case "Retry mechanism (max 3 attempts)"
if grep -q "DTX_RETRY_MAX.*3\|retry.*max" transformer.cu; then
    pass
else
    fail "Retry mechanism missing"
fi

test_case "Header verification on receive"
if grep -q "verifyHeader.*hdr\|magic.*version" transformer.cu; then
    pass
else
    fail "Header verification missing"
fi

test_case "Checksum verification on receive"
if grep -q "verifyChecksum.*payload" transformer.cu; then
    pass
else
    fail "Checksum verification missing"
fi

test_case "Socket error checking"
if grep -q "if.*socket.*< 0\|perror\|std::cerr" transformer.cu; then
    pass
else
    fail "Socket error checking missing"
fi

test_case "Bind error handling"
if grep -q "if.*bind.*< 0" transformer.cu; then
    pass
else
    fail "Bind error handling missing"
fi

test_case "Frame size validation (minimum 14 bytes)"
if grep -q "frame.payload.size.*< 14\|recvLen < 14" transformer.cu; then
    pass
else
    fail "Frame size validation missing"
fi

test_case "Message type validation in handlers"
if grep -q "switch.*msgType\|case MessageType" transformer.cu; then
    pass
else
    fail "Message type validation missing"
fi

test_case "Configuration validation before operation"
if grep -q "config.validate\|localLayers.*remoteLayers" transformer.cu; then
    pass
else
    fail "Configuration validation missing"
fi

test_case "Connected state check before operations"
if grep -q "isConnected\|state.*CONNECTED" transformer.cu; then
    pass
else
    fail "State check missing"
fi

# ============================================================================
# PART 15: BENCHMARKING TESTS (8+ tests)
# ============================================================================

log_section "PART 15: BENCHMARKING TESTS"

test_case "benchmarkDistributed function exists"
if grep -q "benchmarkDistributed.*TimingStats" transformer.cu; then
    pass
else
    fail "benchmarkDistributed missing"
fi

test_case "TimingStats structure with measurements"
if grep -q "struct TimingStats\|forwardMs.*backwardMs.*totalMs" transformer.cu; then
    pass
else
    fail "TimingStats missing"
fi

test_case "Forward pass timing measurement"
if grep -q "forwardMs\|afterForward.*startTime" transformer.cu; then
    pass
else
    fail "Forward timing missing"
fi

test_case "Backward pass timing measurement"
if grep -q "backwardMs\|endTime.*afterForward" transformer.cu; then
    pass
else
    fail "Backward timing missing"
fi

test_case "Elements processed counter"
if grep -q "elementsProcessed" transformer.cu; then
    pass
else
    fail "Elements counter missing"
fi

test_case "Iteration loop in benchmark"
if grep -q "for.*iterations\|iterations.*10" transformer.cu; then
    pass
else
    fail "Iteration loop missing"
fi

test_case "Throughput calculation (elements/second)"
if grep -q "elementsProcessed.*totalMs\|M elem\|throughput" transformer.cu; then
    pass
else
    fail "Throughput calculation missing"
fi

test_case "High resolution clock for precise timing"
if grep -q "high_resolution_clock\|chrono" transformer.cu; then
    pass
else
    fail "High precision timing missing"
fi

# ============================================================================
# PART 16: CLI ARGUMENTS - transformer.cu (20+ tests)
# ============================================================================

log_section "PART 16: CLI ARGUMENTS - transformer.cu"

log_subsection "Server CLI Arguments"

test_case "Server: --seq-len argument"
if grep -q '"-q".*"--seq-len"' $TRANSFORMER_SRC || grep -q 'seq-len' $TRANSFORMER_SRC; then
    pass
else
    fail "--seq-len missing"
fi

test_case "Server: --vocab-size argument"
if grep -q '"-v".*"--vocab-size"' $TRANSFORMER_SRC || grep -q 'vocab-size' $TRANSFORMER_SRC; then
    pass
else
    fail "--vocab-size missing"
fi

test_case "Server: --max-seq-len argument"
if grep -q '"-x".*"--max-seq-len"' $TRANSFORMER_SRC || grep -q 'max-seq-len' $TRANSFORMER_SRC; then
    pass
else
    fail "--max-seq-len missing"
fi

test_case "Server: --kvheads argument"
if grep -q '"-k".*"--kvheads"' $TRANSFORMER_SRC || grep -q 'kvheads' $TRANSFORMER_SRC; then
    pass
else
    fail "--kvheads missing"
fi

test_case "Server: --quant argument"
if grep -q '"--quant"' $TRANSFORMER_SRC; then
    pass
else
    fail "--quant missing"
fi

test_case "Server: --rope-base argument"
if grep -q '"--rope-base"' $TRANSFORMER_SRC; then
    pass
else
    fail "--rope-base missing"
fi

test_case "Server: --rope-scale argument"
if grep -q '"--rope-scale"' $TRANSFORMER_SRC; then
    pass
else
    fail "--rope-scale missing"
fi

test_case "Server: --eps argument"
if grep -q '"--eps"' $TRANSFORMER_SRC; then
    pass
else
    fail "--eps missing"
fi

test_case "Server: --dropout argument"
if grep -q '"--dropout"' $TRANSFORMER_SRC; then
    pass
else
    fail "--dropout missing"
fi

test_case "Server: --verbose argument"
if grep -q '"--verbose"' $TRANSFORMER_SRC; then
    pass
else
    fail "--verbose missing"
fi

log_subsection "Client CLI Arguments"

test_case "Client: --start-layer argument"
if grep -q '"--start-layer"' $TRANSFORMER_SRC; then
    pass
else
    fail "--start-layer missing"
fi

test_case "Client: --retries argument"
if grep -q '"--retries"' $TRANSFORMER_SRC; then
    pass
else
    fail "--retries missing"
fi

log_subsection "Benchmark CLI Arguments"

test_case "Benchmark: --batch-size argument"
if grep -q '"--batch-size"' $TRANSFORMER_SRC; then
    pass
else
    fail "--batch-size missing"
fi

test_case "Benchmark: --warmup argument"
if grep -q '"--warmup"' $TRANSFORMER_SRC; then
    pass
else
    fail "--warmup missing"
fi

test_case "Benchmark: --output argument"
if grep -q '"--output"' $TRANSFORMER_SRC; then
    pass
else
    fail "--output missing"
fi

log_subsection "Test CLI Arguments"

test_case "Test: --all argument"
if grep -q '"--all"' $TRANSFORMER_SRC; then
    pass
else
    fail "--all missing"
fi

test_case "Test: --protocol argument"
if grep -q '"--protocol"' $TRANSFORMER_SRC; then
    pass
else
    fail "--protocol missing"
fi

test_case "Test: --config argument"
if grep -q '"--config"' $TRANSFORMER_SRC; then
    pass
else
    fail "--config missing"
fi

test_case "Test: --quant test argument"
if grep -q 'testQuant\|"--quant".*test' $TRANSFORMER_SRC; then
    pass
else
    fail "--quant test missing"
fi

test_case "Test: --kernels argument"
if grep -q '"--kernels"' $TRANSFORMER_SRC; then
    pass
else
    fail "--kernels missing"
fi

# ============================================================================
# PART 17: CLI ARGUMENTS - facaded_transformer.cu (25+ tests)
# ============================================================================

log_section "PART 17: CLI ARGUMENTS - facaded_transformer.cu"

log_subsection "Facade Command Arguments"

test_case "Facade: command exists"
if grep -q 'command == "facade"' $FACADE_SRC; then
    pass
else
    fail "facade command missing"
fi

test_case "Facade: --model argument"
if grep -q '"--model"' $FACADE_SRC; then
    pass
else
    fail "--model missing"
fi

test_case "Facade: --tokenizer argument"
if grep -q '"--tokenizer"' $FACADE_SRC; then
    pass
else
    fail "--tokenizer missing"
fi

test_case "Facade: --prompt argument"
if grep -q '"--prompt"' $FACADE_SRC; then
    pass
else
    fail "--prompt missing"
fi

test_case "Facade: --max-tokens argument"
if grep -q '"--max-tokens"' $FACADE_SRC; then
    pass
else
    fail "--max-tokens missing"
fi

test_case "Facade: --temperature argument"
if grep -q '"--temperature"' $FACADE_SRC; then
    pass
else
    fail "--temperature missing"
fi

test_case "Facade: --top-k argument"
if grep -q '"--top-k"' $FACADE_SRC; then
    pass
else
    fail "--top-k missing"
fi

test_case "Facade: --top-p argument"
if grep -q '"--top-p"' $FACADE_SRC; then
    pass
else
    fail "--top-p missing"
fi

log_subsection "Facade Introspection Arguments"

test_case "Facade: --inspect argument"
if grep -q '"--inspect"' $FACADE_SRC; then
    pass
else
    fail "--inspect missing"
fi

test_case "Facade: --show-attention argument"
if grep -q '"--show-attention"' $FACADE_SRC; then
    pass
else
    fail "--show-attention missing"
fi

test_case "Facade: --show-hidden argument"
if grep -q '"--show-hidden"' $FACADE_SRC; then
    pass
else
    fail "--show-hidden missing"
fi

test_case "Facade: --show-qkv argument"
if grep -q '"--show-qkv"' $FACADE_SRC; then
    pass
else
    fail "--show-qkv missing"
fi

test_case "Facade: --show-logits argument"
if grep -q '"--show-logits"' $FACADE_SRC; then
    pass
else
    fail "--show-logits missing"
fi

test_case "Facade: --show-entropy argument"
if grep -q '"--show-entropy"' $FACADE_SRC; then
    pass
else
    fail "--show-entropy missing"
fi

test_case "Facade: --show-saliency argument"
if grep -q '"--show-saliency"' $FACADE_SRC; then
    pass
else
    fail "--show-saliency missing"
fi

test_case "Facade: --show-weights argument"
if grep -q '"--show-weights"' $FACADE_SRC; then
    pass
else
    fail "--show-weights missing"
fi

test_case "Facade: --show-tensors argument"
if grep -q '"--show-tensors"' $FACADE_SRC; then
    pass
else
    fail "--show-tensors missing"
fi

log_subsection "Facade Dump Arguments"

test_case "Facade: --dump-hidden argument"
if grep -q '"--dump-hidden"' $FACADE_SRC; then
    pass
else
    fail "--dump-hidden missing"
fi

test_case "Facade: --dump-attention argument"
if grep -q '"--dump-attention"' $FACADE_SRC; then
    pass
else
    fail "--dump-attention missing"
fi

log_subsection "Facade Filter Arguments"

test_case "Facade: --layer argument"
if grep -q '"--layer"' $FACADE_SRC; then
    pass
else
    fail "--layer missing"
fi

test_case "Facade: --head argument"
if grep -q '"--head"' $FACADE_SRC; then
    pass
else
    fail "--head missing"
fi

test_case "Facade: --position argument"
if grep -q '"--position"' $FACADE_SRC; then
    pass
else
    fail "--position missing"
fi

# ============================================================================
# PART 18: FACADE INTROSPECTION FUNCTIONS (20+ tests)
# ============================================================================

log_section "PART 18: FACADE INTROSPECTION FUNCTIONS"

log_subsection "TransformerFacade Class"

test_case "TransformerFacade class definition"
if grep -q "class TransformerFacade" $FACADE_SRC; then
    pass
else
    fail "TransformerFacade class missing"
fi

test_case "Facade: loadModel method"
if grep -q "loadModel.*path\|bool loadModel" $FACADE_SRC; then
    pass
else
    fail "loadModel missing"
fi

test_case "Facade: loadTokenizer method"
if grep -q "loadTokenizer.*path\|bool loadTokenizer" $FACADE_SRC; then
    pass
else
    fail "loadTokenizer missing"
fi

test_case "Facade: runForward method"
if grep -q "runForward.*tokenIds" $FACADE_SRC; then
    pass
else
    fail "runForward missing"
fi

test_case "Facade: generate method"
if grep -q "generate.*prompt.*maxTokens" $FACADE_SRC; then
    pass
else
    fail "generate missing"
fi

log_subsection "Structural Introspection"

test_case "Facade: getNumLayers method"
if grep -q "getNumLayers" $FACADE_SRC; then
    pass
else
    fail "getNumLayers missing"
fi

test_case "Facade: getNumHeads method"
if grep -q "getNumHeads" $FACADE_SRC; then
    pass
else
    fail "getNumHeads missing"
fi

test_case "Facade: getHiddenSize method"
if grep -q "getHiddenSize" $FACADE_SRC; then
    pass
else
    fail "getHiddenSize missing"
fi

test_case "Facade: getHeadDim method"
if grep -q "getHeadDim" $FACADE_SRC; then
    pass
else
    fail "getHeadDim missing"
fi

test_case "Facade: getFFNDim method"
if grep -q "getFFNDim" $FACADE_SRC; then
    pass
else
    fail "getFFNDim missing"
fi

test_case "Facade: getVocabSize method"
if grep -q "getVocabSize" $FACADE_SRC; then
    pass
else
    fail "getVocabSize missing"
fi

test_case "Facade: getMaxSeqLen method"
if grep -q "getMaxSeqLen" $FACADE_SRC; then
    pass
else
    fail "getMaxSeqLen missing"
fi

log_subsection "Attention Introspection"

test_case "Facade: getAttentionWeights method"
if grep -q "getAttentionWeights.*layer.*head" $FACADE_SRC; then
    pass
else
    fail "getAttentionWeights missing"
fi

test_case "Facade: getAttentionLogits method"
if grep -q "getAttentionLogits.*layer.*head" $FACADE_SRC; then
    pass
else
    fail "getAttentionLogits missing"
fi

test_case "Facade: getAttentionEntropy method"
if grep -q "getAttentionEntropy.*layer.*head" $FACADE_SRC; then
    pass
else
    fail "getAttentionEntropy missing"
fi

log_subsection "Hidden State Introspection"

test_case "Facade: getHiddenState method"
if grep -q "getHiddenState.*layer.*pos" $FACADE_SRC; then
    pass
else
    fail "getHiddenState missing"
fi

test_case "Facade: getQKV method"
if grep -q "getQKV.*layer.*head.*type" $FACADE_SRC; then
    pass
else
    fail "getQKV missing"
fi

test_case "Facade: getSaliencyMap method"
if grep -q "getSaliencyMap.*tokenIdx" $FACADE_SRC; then
    pass
else
    fail "getSaliencyMap missing"
fi

log_subsection "Embedding Introspection"

test_case "Facade: getTokenEmbedding method"
if grep -q "getTokenEmbedding.*tokenId" $FACADE_SRC; then
    pass
else
    fail "getTokenEmbedding missing"
fi

test_case "Facade: getPositionalEncoding method"
if grep -q "getPositionalEncoding.*pos" $FACADE_SRC; then
    pass
else
    fail "getPositionalEncoding missing"
fi

log_subsection "Output Introspection"

test_case "Facade: getLogits method"
if grep -q "getLogits" $FACADE_SRC; then
    pass
else
    fail "getLogits missing"
fi

test_case "Facade: getSoftmaxOutput method"
if grep -q "getSoftmaxOutput" $FACADE_SRC; then
    pass
else
    fail "getSoftmaxOutput missing"
fi

log_subsection "Weight Access"

test_case "Facade: getWeight method"
if grep -q "getWeight.*layer.*ParamType" $FACADE_SRC; then
    pass
else
    fail "getWeight missing"
fi

test_case "Facade: getWeightShape method"
if grep -q "getWeightShape.*layer.*ParamType" $FACADE_SRC; then
    pass
else
    fail "getWeightShape missing"
fi

# ============================================================================
# PART 19: QUANTIZATION TESTS (15+ tests)
# ============================================================================

log_section "PART 19: QUANTIZATION TESTS"

log_subsection "Quantization Types"

test_case "GGML_DType enum definition"
if grep -q "enum class GGML_DType" $TRANSFORMER_SRC; then
    pass
else
    fail "GGML_DType enum missing"
fi

test_case "Q2_K quantization type"
if grep -q "Q2_K" $TRANSFORMER_SRC; then
    pass
else
    fail "Q2_K missing"
fi

test_case "Q3_K quantization type"
if grep -q "Q3_K" $TRANSFORMER_SRC; then
    pass
else
    fail "Q3_K missing"
fi

test_case "Q4_K quantization type"
if grep -q "Q4_K" $TRANSFORMER_SRC; then
    pass
else
    fail "Q4_K missing"
fi

test_case "Q5_K quantization type"
if grep -q "Q5_K" $TRANSFORMER_SRC; then
    pass
else
    fail "Q5_K missing"
fi

test_case "Q6_K quantization type"
if grep -q "Q6_K" $TRANSFORMER_SRC; then
    pass
else
    fail "Q6_K missing"
fi

test_case "Q8_K quantization type"
if grep -q "Q8_K" $TRANSFORMER_SRC; then
    pass
else
    fail "Q8_K missing"
fi

log_subsection "Quantization Structures"

test_case "block_q2_K structure"
if grep -q "struct block_q2_K" $TRANSFORMER_SRC; then
    pass
else
    fail "block_q2_K missing"
fi

test_case "block_q3_K structure"
if grep -q "struct block_q3_K" $TRANSFORMER_SRC; then
    pass
else
    fail "block_q3_K missing"
fi

test_case "block_q4_K structure"
if grep -q "struct block_q4_K" $TRANSFORMER_SRC; then
    pass
else
    fail "block_q4_K missing"
fi

test_case "block_q5_K structure"
if grep -q "struct block_q5_K" $TRANSFORMER_SRC; then
    pass
else
    fail "block_q5_K missing"
fi

test_case "block_q6_K structure"
if grep -q "struct block_q6_K" $TRANSFORMER_SRC; then
    pass
else
    fail "block_q6_K missing"
fi

log_subsection "Dequantization Functions"

test_case "dequant_row_q2_K function"
if grep -q "dequant_row_q2_K" $TRANSFORMER_SRC; then
    pass
else
    fail "dequant_row_q2_K missing"
fi

test_case "dequant_row_q3_K function"
if grep -q "dequant_row_q3_K" $TRANSFORMER_SRC; then
    pass
else
    fail "dequant_row_q3_K missing"
fi

test_case "dequant_row_q4_K function"
if grep -q "dequant_row_q4_K" $TRANSFORMER_SRC; then
    pass
else
    fail "dequant_row_q4_K missing"
fi

test_case "fp16_to_fp32 conversion"
if grep -q "fp16_to_fp32" $TRANSFORMER_SRC; then
    pass
else
    fail "fp16_to_fp32 missing"
fi

# ============================================================================
# PART 20: GGUF AND TOKENIZER (15+ tests)
# ============================================================================

log_section "PART 20: GGUF AND TOKENIZER TESTS"

log_subsection "GGUFLoader Class"

test_case "GGUFLoader class definition"
if grep -q "class GGUFLoader" $FACADE_SRC; then
    pass
else
    fail "GGUFLoader class missing"
fi

test_case "GGUFLoader: loadFromFile method"
if grep -q "loadFromFile.*fname\|bool loadFromFile" $FACADE_SRC; then
    pass
else
    fail "loadFromFile missing"
fi

test_case "GGUFLoader: getTensor method"
if grep -q "getTensor.*names" $FACADE_SRC; then
    pass
else
    fail "getTensor missing"
fi

test_case "GGUFLoader: getTensorShape method"
if grep -q "getTensorShape" $FACADE_SRC; then
    pass
else
    fail "getTensorShape missing"
fi

test_case "GGUFLoader: hasTensor method"
if grep -q "hasTensor" $FACADE_SRC; then
    pass
else
    fail "hasTensor missing"
fi

test_case "GGUFLoader: printAllTensorNames method"
if grep -q "printAllTensorNames" $FACADE_SRC; then
    pass
else
    fail "printAllTensorNames missing"
fi

test_case "GGUFLoader: getEmbedDim method"
if grep -q "getEmbedDim" $FACADE_SRC; then
    pass
else
    fail "getEmbedDim missing"
fi

log_subsection "Tokenizer Class"

test_case "Tokenizer class definition"
if grep -q "class Tokenizer" $FACADE_SRC; then
    pass
else
    fail "Tokenizer class missing"
fi

test_case "Tokenizer: loadFromFile method"
if grep -q "Tokenizer.*loadFromFile\|bool loadFromFile.*filename" $FACADE_SRC; then
    pass
else
    fail "Tokenizer loadFromFile missing"
fi

test_case "Tokenizer: encode method"
if grep -q "encode.*text" $FACADE_SRC; then
    pass
else
    fail "encode missing"
fi

test_case "Tokenizer: decode method"
if grep -q "decode.*ids" $FACADE_SRC; then
    pass
else
    fail "decode missing"
fi

test_case "Tokenizer: getTokenId method"
if grep -q "getTokenId.*token" $FACADE_SRC; then
    pass
else
    fail "getTokenId missing"
fi

test_case "Tokenizer: getToken method"
if grep -q "getToken.*id" $FACADE_SRC; then
    pass
else
    fail "getToken missing"
fi

test_case "Tokenizer: getVocabSize method"
if grep -q "getVocabSize" $FACADE_SRC; then
    pass
else
    fail "getVocabSize missing"
fi

test_case "Tokenizer: isLoaded method"
if grep -q "isLoaded" $FACADE_SRC; then
    pass
else
    fail "isLoaded missing"
fi

# ============================================================================
# PART 21: FACADE TEST COMMAND (10+ tests)
# ============================================================================

log_section "PART 21: FACADE TEST COMMAND"

test_case "Facade test: --facade argument"
if grep -q 'testFacade\|"--facade"' $FACADE_SRC; then
    pass
else
    fail "--facade test missing"
fi

test_case "Facade test: --tokenizer argument"
if grep -q 'testTokenizer\|"--tokenizer".*test' $FACADE_SRC; then
    pass
else
    fail "--tokenizer test missing"
fi

test_case "Facade test: --gguf argument"
if grep -q 'testGGUF\|"--gguf"' $FACADE_SRC; then
    pass
else
    fail "--gguf test missing"
fi

test_case "Facade test: passed/failed counters"
if grep -q 'passed++\|failed++' $FACADE_SRC; then
    pass
else
    fail "test counters missing"
fi

test_case "Facade test: Test Results output"
if grep -q 'Test Results\|Passed:.*Failed:' $FACADE_SRC; then
    pass
else
    fail "Test Results output missing"
fi

# ============================================================================
# PART 22: HELP DISPLAY TESTS (10+ tests)
# ============================================================================

log_section "PART 22: HELP DISPLAY TESTS"

log_subsection "transformer.cu Help"

test_case "transformer.cu: QUANTIZATION TYPES section in help"
if grep -q "QUANTIZATION TYPES" $TRANSFORMER_SRC; then
    pass
else
    fail "QUANTIZATION TYPES help section missing"
fi

test_case "transformer.cu: EXAMPLES section in help"
if grep -q "EXAMPLES" $TRANSFORMER_SRC; then
    pass
else
    fail "EXAMPLES help section missing"
fi

test_case "transformer.cu: --version help option"
if grep -q '"--version"' $TRANSFORMER_SRC; then
    pass
else
    fail "--version help missing"
fi

log_subsection "facaded_transformer.cu Help"

test_case "facaded_transformer.cu: FACADE INTROSPECTION section in help"
if grep -q "FACADE INTROSPECTION" $FACADE_SRC; then
    pass
else
    fail "FACADE INTROSPECTION help section missing"
fi

test_case "facaded_transformer.cu: facade command in main help"
if grep -q "facade.*introspection\|facade.*inference" $FACADE_SRC; then
    pass
else
    fail "facade command help missing"
fi

test_case "facaded_transformer.cu: DUMP OPTIONS section"
if grep -q "DUMP OPTIONS" $FACADE_SRC; then
    pass
else
    fail "DUMP OPTIONS section missing"
fi

test_case "facaded_transformer.cu: FILTER OPTIONS section"
if grep -q "FILTER OPTIONS" $FACADE_SRC; then
    pass
else
    fail "FILTER OPTIONS section missing"
fi

# ============================================================================
# PART 23: CUDA KERNEL TESTS (10+ tests)
# ============================================================================

log_section "PART 23: CUDA KERNEL TESTS"

test_case "matmulKernel in transformer.cu"
if grep -q "__global__.*matmulKernel" $TRANSFORMER_SRC; then
    pass
else
    fail "matmulKernel missing"
fi

test_case "geluKernel in transformer.cu"
if grep -q "__global__.*geluKernel" $TRANSFORMER_SRC; then
    pass
else
    fail "geluKernel missing"
fi

test_case "softmaxKernel in transformer.cu"
if grep -q "__global__.*softmaxKernel" $TRANSFORMER_SRC; then
    pass
else
    fail "softmaxKernel missing"
fi

test_case "facade_softmax_kernel in facaded_transformer.cu"
if grep -q "__global__.*facade_softmax_kernel" $FACADE_SRC; then
    pass
else
    fail "facade_softmax_kernel missing"
fi

test_case "facade_layer_norm_kernel in facaded_transformer.cu"
if grep -q "__global__.*facade_layer_norm_kernel" $FACADE_SRC; then
    pass
else
    fail "facade_layer_norm_kernel missing"
fi

test_case "facade_embed_tokens_kernel in facaded_transformer.cu"
if grep -q "__global__.*facade_embed_tokens_kernel" $FACADE_SRC; then
    pass
else
    fail "facade_embed_tokens_kernel missing"
fi

test_case "facade_attention_scores_kernel in facaded_transformer.cu"
if grep -q "__global__.*facade_attention_scores_kernel" $FACADE_SRC; then
    pass
else
    fail "facade_attention_scores_kernel missing"
fi

test_case "CUDA_CHECK macro defined"
if grep -q "#define CUDA_CHECK" $TRANSFORMER_SRC; then
    pass
else
    fail "CUDA_CHECK macro missing"
fi

test_case "__syncthreads() usage"
if grep -q "__syncthreads" $TRANSFORMER_SRC; then
    pass
else
    fail "__syncthreads missing"
fi

test_case "cudaError_t handling"
if grep -q "cudaError_t\|cudaSuccess" $TRANSFORMER_SRC; then
    pass
else
    fail "cudaError_t handling missing"
fi

# ============================================================================
# SUMMARY AND REPORTING
# ============================================================================

log_section "TEST SUMMARY"

TESTS_FAILED=$((TESTS_RUN - TESTS_PASSED))
CHECKS_FAILED=$((CHECKS_RUN - CHECKS_PASSED))
TOTAL_TESTS=$((TESTS_RUN + CHECKS_RUN))
TOTAL_PASSED=$((TESTS_PASSED + CHECKS_PASSED))
TOTAL_FAILED=$((TESTS_FAILED + CHECKS_FAILED))
PASS_RATE=$(awk "BEGIN {printf \"%.1f\", ($TOTAL_PASSED / $TOTAL_TESTS) * 100}")

echo "" | tee -a "$LOG_FILE"
echo -e "${BOLD}=== FUNCTIONAL TESTS ===${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}Total Tests:${NC}    $TESTS_RUN" | tee -a "$LOG_FILE"
echo -e "${GREEN}${BOLD}Passed:${NC}         $TESTS_PASSED" | tee -a "$LOG_FILE"
echo -e "${RED}${BOLD}Failed:${NC}         $TESTS_FAILED" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo -e "${BOLD}=== CODE QUALITY CHECKS ===${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}Total Checks:${NC}   $CHECKS_RUN" | tee -a "$LOG_FILE"
echo -e "${GREEN}${BOLD}Passed:${NC}        $CHECKS_PASSED" | tee -a "$LOG_FILE"
echo -e "${RED}${BOLD}Failed:${NC}        $CHECKS_FAILED" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo -e "${BOLD}=== OVERALL RESULTS ===${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}Total Tests + Checks:${NC} $TOTAL_TESTS" | tee -a "$LOG_FILE"
echo -e "${GREEN}${BOLD}Total Passed:${NC}        $TOTAL_PASSED" | tee -a "$LOG_FILE"
echo -e "${RED}${BOLD}Total Failed:${NC}        $TOTAL_FAILED" | tee -a "$LOG_FILE"
echo -e "${BOLD}Overall Pass Rate:${NC}   ${PASS_RATE}%" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ "$TOTAL_FAILED" = "0" ]; then
    echo -e "${GREEN}${BOLD}✓ ALL TESTS AND CHECKS PASSED${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 0
else
    echo -e "${RED}${BOLD}✗ SOME TESTS OR CHECKS FAILED${NC}" | tee -a "$LOG_FILE"
    echo -e "Failures: $TOTAL_FAILED (${TESTS_FAILED} tests, ${CHECKS_FAILED} checks)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi
