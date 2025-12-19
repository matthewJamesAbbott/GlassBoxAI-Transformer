/*
  TransformerFacade - CUDA Port
  Inspection and Manipulation Facade for Transformer
  Matthew Abbott 2025
  
  Compile with: nvcc -o facadedtransformer_cuda facadedtransformer.cu -lcublas -O3
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <climits>
#include <ctime>
#include <cstdint>

#define MAX_SEQ_LEN 1024

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Type definitions
using DoubleArray = std::vector<double>;
using SingleArray = std::vector<float>;
using IntArray = std::vector<int>;
using Int64Array = std::vector<int64_t>;
using Double2DArray = std::vector<DoubleArray>;
using Double3DArray = std::vector<Double2DArray>;

enum ParamType {
    ptQProj, ptKProj, ptVProj, ptOutProj,
    ptFFN1, ptFFN2,
    ptLayerNorm1Weight, ptLayerNorm1Bias,
    ptLayerNorm2Weight, ptLayerNorm2Bias,
    ptTokenEmbed, ptPosEmbed,
    ptFinalNormWeight, ptFinalNormBias
};

enum QKVType { qkvQuery, qkvKey, qkvValue };

// Forward declarations
class GGUFLoader;
class Tokenizer;
class TransformerModel;
class TransformerFacade;

// ==================== CUDA Kernels ====================

__global__ void softmax_kernel(float* input, float* output, int size) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    
    // Find max
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }
    shared[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    max_val = shared[0];
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    sum = shared[0];
    
    // Normalize
    for (int i = tid; i < size; i += blockDim.x) {
        output[i] /= sum;
    }
}

__global__ void layer_norm_kernel(float* input, float* output, float* gamma, float* beta,
                                   int dim, float eps) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int pos = blockIdx.x;
    float* in_pos = input + pos * dim;
    float* out_pos = output + pos * dim;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += in_pos[i];
    }
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float mean = shared[0] / dim;
    
    // Compute variance
    sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = in_pos[i] - mean;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float var = shared[0] / dim;
    float inv_std = rsqrtf(var + eps);
    
    // Normalize
    for (int i = tid; i < dim; i += blockDim.x) {
        float normalized = (in_pos[i] - mean) * inv_std;
        out_pos[i] = normalized * gamma[i] + beta[i];
    }
}

__global__ void gelu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void add_kernel(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void embed_tokens_kernel(float* token_emb, float* pos_emb, int* token_ids,
                                     float* output, int seq_len, int embed_dim) {
    int pos = blockIdx.x;
    int dim = threadIdx.x;
    if (pos < seq_len && dim < embed_dim) {
        int token_id = token_ids[pos];
        output[pos * embed_dim + dim] = token_emb[token_id * embed_dim + dim] + 
                                         pos_emb[pos * embed_dim + dim];
    }
}

__global__ void matmul_kernel(float* A, float* B, float* C, float* bias,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = (bias != nullptr) ? bias[col] : 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void attention_scores_kernel(float* Q, float* K, float* scores,
                                         int seq_len, int head_dim, float scale, int head_offset) {
    int from_pos = blockIdx.x;
    int to_pos = threadIdx.x;
    
    if (from_pos < seq_len && to_pos < seq_len) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            sum += Q[from_pos * head_dim + d + head_offset] * K[to_pos * head_dim + d + head_offset];
        }
        // Causal mask
        if (to_pos > from_pos) sum = -1e9f;
        scores[from_pos * seq_len + to_pos] = sum * scale;
    }
}

__global__ void attention_output_kernel(float* weights, float* V, float* output,
                                          int seq_len, int head_dim, int head_offset) {
    int pos = blockIdx.x;
    int d = threadIdx.x;
    
    if (pos < seq_len && d < head_dim) {
        float sum = 0.0f;
        for (int src = 0; src < seq_len; src++) {
            sum += weights[pos * seq_len + src] * V[src * head_dim + d + head_offset];
        }
        output[pos * head_dim + d + head_offset] = sum;
    }
}

// ==================== GGUFTensor ====================

struct GGUFTensor {
    std::string name;
    Int64Array shape;
    int numDims;
    int dtype;
    int64_t dataOffset;
    bool dataLoaded;
    SingleArray data;
};

// ==================== Tokenizer ====================

class Tokenizer {
private:
    std::map<std::string, int> tokenToId;
    std::vector<std::string> idToToken;
    int vocabSize;
    bool loaded;

public:
    Tokenizer() : vocabSize(0), loaded(false) {}
    
    bool loadFromFile(const std::string& filename);
    IntArray encode(const std::string& text);
    std::string decode(const IntArray& ids);
    int getTokenId(const std::string& token);
    std::string getToken(int id);
    int getVocabSize() const { return vocabSize; }
    bool isLoaded() const { return loaded; }
};

bool Tokenizer::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();
    
    // Find vocab section inside "model": { ... "vocab": { ... } }
    size_t vocabStart = content.find("\"vocab\"");
    if (vocabStart == std::string::npos) return false;
    
    // Skip to the opening brace of vocab object
    vocabStart = content.find("{", vocabStart);
    if (vocabStart == std::string::npos) return false;
    
    // Find matching closing brace, accounting for strings that may contain braces
    int braceCount = 1;
    size_t vocabEnd = vocabStart + 1;
    bool inString = false;
    bool escaped = false;
    
    while (vocabEnd < content.length() && braceCount > 0) {
        char c = content[vocabEnd];
        if (escaped) {
            escaped = false;
        } else if (c == '\\' && inString) {
            escaped = true;
        } else if (c == '"') {
            inString = !inString;
        } else if (!inString) {
            if (c == '{') braceCount++;
            else if (c == '}') braceCount--;
        }
        vocabEnd++;
    }
    
    // Now parse the vocab content character by character
    size_t pos = vocabStart + 1;
    size_t endPos = vocabEnd - 1;
    
    while (pos < endPos) {
        // Skip whitespace and commas
        while (pos < endPos && (content[pos] == ' ' || content[pos] == '\n' || 
               content[pos] == '\r' || content[pos] == '\t' || content[pos] == ',')) {
            pos++;
        }
        if (pos >= endPos) break;
        
        // Expect opening quote for key
        if (content[pos] != '"') { pos++; continue; }
        pos++;
        
        // Extract key, handling escapes
        std::string token;
        while (pos < endPos) {
            char c = content[pos];
            if (c == '\\' && pos + 1 < endPos) {
                char next = content[pos + 1];
                if (next == 'n') { token += '\n'; pos += 2; }
                else if (next == 'r') { token += '\r'; pos += 2; }
                else if (next == 't') { token += '\t'; pos += 2; }
                else if (next == '"') { token += '"'; pos += 2; }
                else if (next == '\\') { token += '\\'; pos += 2; }
                else if (next == 'u' && pos + 5 < endPos) {
                    std::string hex = content.substr(pos + 2, 4);
                    try {
                        int cp = std::stoi(hex, nullptr, 16);
                        if (cp < 0x80) {
                            token += (char)cp;
                        } else if (cp < 0x800) {
                            token += (char)(0xC0 | (cp >> 6));
                            token += (char)(0x80 | (cp & 0x3F));
                        } else {
                            token += (char)(0xE0 | (cp >> 12));
                            token += (char)(0x80 | ((cp >> 6) & 0x3F));
                            token += (char)(0x80 | (cp & 0x3F));
                        }
                    } catch (...) {}
                    pos += 6;
                } else {
                    token += c; pos++;
                }
            } else if (c == '"') {
                pos++;
                break;
            } else {
                token += c;
                pos++;
            }
        }
        
        // Skip whitespace and colon
        while (pos < endPos && (content[pos] == ' ' || content[pos] == ':' ||
               content[pos] == '\n' || content[pos] == '\r' || content[pos] == '\t')) {
            pos++;
        }
        
        // Parse number
        size_t numStart = pos;
        while (pos < endPos && (isdigit(content[pos]) || content[pos] == '-')) {
            pos++;
        }
        
        if (pos > numStart) {
            try {
                int id = std::stoi(content.substr(numStart, pos - numStart));
                tokenToId[token] = id;
                if (id >= (int)idToToken.size()) idToToken.resize(id + 1);
                idToToken[id] = token;
                if (id >= vocabSize) vocabSize = id + 1;
            } catch (...) {}
        }
    }
    
    loaded = vocabSize > 0;
    return loaded;
}

int Tokenizer::getTokenId(const std::string& token) {
    auto it = tokenToId.find(token);
    return (it != tokenToId.end()) ? it->second : -1;
}

std::string Tokenizer::getToken(int id) {
    return (id >= 0 && id < (int)idToToken.size()) ? idToToken[id] : "";
}

IntArray Tokenizer::encode(const std::string& text) {
    IntArray result;
    if (!loaded) return result;
    
    std::string currentWord;
    for (size_t i = 0; i < text.length(); i++) {
        if (text[i] == ' ') {
            if (!currentWord.empty()) {
                int id = getTokenId(currentWord);
                if (id >= 0) result.push_back(id);
                else for (char c : currentWord) {
                    id = getTokenId(std::string(1, c));
                    if (id >= 0) result.push_back(id);
                }
            }
            currentWord = "\xC4\xA0"; // Ġ in UTF-8
        } else {
            currentWord += text[i];
        }
    }
    if (!currentWord.empty()) {
        int id = getTokenId(currentWord);
        if (id >= 0) result.push_back(id);
        else for (char c : currentWord) {
            id = getTokenId(std::string(1, c));
            if (id >= 0) result.push_back(id);
        }
    }
    return result;
}

std::string Tokenizer::decode(const IntArray& ids) {
    std::string result;
    for (int id : ids) {
        std::string token = getToken(id);
        // Replace Ġ with space
        size_t pos;
        while ((pos = token.find("\xC4\xA0")) != std::string::npos) {
            token.replace(pos, 2, " ");
        }
        while ((pos = token.find("\xC4\x8A")) != std::string::npos) {
            token.replace(pos, 2, "\n");
        }
        result += token;
    }
    return result;
}

// ==================== GGUFLoader ====================

class GGUFLoader {
private:
    std::ifstream stream;
    std::string filename;
    std::vector<GGUFTensor> tensors;
    std::map<std::string, int> tensorMap;
    int64_t tensorDataStart;
    
    int embedDim, numLayers, numHeads, ffnDim, vocabSize, maxSeqLen;
    bool loaded;

    uint32_t readUInt32() {
        uint32_t val;
        stream.read(reinterpret_cast<char*>(&val), 4);
        return val;
    }
    
    uint64_t readUInt64() {
        uint64_t val;
        stream.read(reinterpret_cast<char*>(&val), 8);
        return val;
    }
    
    std::string readString() {
        uint64_t len = readUInt64();
        if (len > 10000000) return "";
        std::string str(len, '\0');
        stream.read(&str[0], len);
        return str;
    }
    
    float float16ToFloat32(uint16_t h) {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        
        if (exp == 0) {
            if (mant == 0) return 0.0f;
            float m = mant / 1024.0f;
            float e = -14.0f;
            while (m < 1.0f) { m *= 2.0f; e -= 1.0f; }
            return (sign ? -1.0f : 1.0f) * m * powf(2.0f, e);
        } else if (exp == 31) {
            return (mant != 0) ? NAN : (sign ? -INFINITY : INFINITY);
        }
        return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15);
    }
    
    void skipMetadataValue(int valueType) {
        switch (valueType) {
            case 0: case 1: stream.seekg(1, std::ios::cur); break;
            case 2: case 3: stream.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: stream.seekg(4, std::ios::cur); break;
            case 7: stream.seekg(1, std::ios::cur); break;
            case 8: {
                uint64_t len = readUInt64();
                stream.seekg(len, std::ios::cur);
            } break;
            case 9: {
                uint32_t arrType = readUInt32();
                uint64_t arrCount = readUInt64();
                for (uint64_t i = 0; i < std::min(arrCount, (uint64_t)999999); i++)
                    skipMetadataValue(arrType);
            } break;
            case 10: case 11: case 12: stream.seekg(8, std::ios::cur); break;
        }
    }

    void parseHeader() {
        char magic[4];
        stream.read(magic, 4);
        if (strncmp(magic, "GGUF", 4) != 0)
            throw std::runtime_error("Invalid GGUF magic");
        
        uint32_t version = readUInt32();
        uint64_t tensorCount = readUInt64();
        uint64_t metadataCount = readUInt64();
        
        for (uint64_t i = 0; i < metadataCount; i++) {
            std::string key = readString();
            uint32_t valueType = readUInt32();
            
            if (key == "gpt2.embedding_length" && (valueType == 4 || valueType == 5 || valueType == 10)) {
                embedDim = (valueType == 10) ? readUInt64() : readUInt32();
            } else if (key == "gpt2.block_count" && (valueType == 4 || valueType == 5 || valueType == 10)) {
                numLayers = (valueType == 10) ? readUInt64() : readUInt32();
            } else if (key == "gpt2.attention.head_count" && (valueType == 4 || valueType == 5 || valueType == 10)) {
                numHeads = (valueType == 10) ? readUInt64() : readUInt32();
            } else if (key == "gpt2.feed_forward_length" && (valueType == 4 || valueType == 5 || valueType == 10)) {
                ffnDim = (valueType == 10) ? readUInt64() : readUInt32();
            } else if (key == "gpt2.context_length" && (valueType == 4 || valueType == 5 || valueType == 10)) {
                maxSeqLen = (valueType == 10) ? readUInt64() : readUInt32();
            } else {
                skipMetadataValue(valueType);
            }
        }
        
        tensors.resize(tensorCount);
        for (uint64_t i = 0; i < tensorCount; i++) {
            tensors[i].name = readString();
            tensors[i].numDims = readUInt32();
            tensors[i].shape.resize(tensors[i].numDims);
            for (int d = 0; d < tensors[i].numDims; d++)
                stream.read(reinterpret_cast<char*>(&tensors[i].shape[d]), 8);
            tensors[i].dtype = readUInt32();
            tensors[i].dataOffset = readUInt64();
            tensors[i].dataLoaded = false;
            tensorMap[tensors[i].name] = i;
        }
        
        tensorDataStart = stream.tellg();
        while (tensorDataStart % 32 != 0) tensorDataStart++;
    }

    bool loadTensorByIndex(int idx) {
        if (idx < 0 || idx >= (int)tensors.size()) return false;
        if (tensors[idx].dataLoaded) return true;
        
        int64_t totalElements = 1;
        for (int d = 0; d < tensors[idx].numDims; d++)
            totalElements *= tensors[idx].shape[d];
        
        tensors[idx].data.resize(totalElements);
        stream.seekg(tensorDataStart + tensors[idx].dataOffset);
        
        if (tensors[idx].dtype == 0) {
            stream.read(reinterpret_cast<char*>(tensors[idx].data.data()), totalElements * 4);
        } else if (tensors[idx].dtype == 1) {
            std::vector<uint16_t> fp16Data(totalElements);
            stream.read(reinterpret_cast<char*>(fp16Data.data()), totalElements * 2);
            for (int64_t i = 0; i < totalElements; i++)
                tensors[idx].data[i] = float16ToFloat32(fp16Data[i]);
        } else {
            return false;
        }
        
        tensors[idx].dataLoaded = true;
        return true;
    }

public:
    GGUFLoader() : embedDim(768), numLayers(12), numHeads(12), ffnDim(3072),
                   vocabSize(50257), maxSeqLen(1024), loaded(false) {}
    
    bool loadFromFile(const std::string& fname) {
        filename = fname;
        stream.open(filename, std::ios::binary);
        if (!stream.is_open()) return false;
        
        try {
            parseHeader();
            loaded = true;
        } catch (...) {
            return false;
        }
        return true;
    }
    
    SingleArray getTensor(const std::vector<std::string>& names) {
        for (const auto& name : names) {
            auto it = tensorMap.find(name);
            if (it != tensorMap.end()) {
                if (loadTensorByIndex(it->second))
                    return tensors[it->second].data;
            }
        }
        return SingleArray();
    }
    
    Int64Array getTensorShape(const std::vector<std::string>& names) {
        for (const auto& name : names) {
            auto it = tensorMap.find(name);
            if (it != tensorMap.end())
                return tensors[it->second].shape;
        }
        return Int64Array();
    }
    
    bool hasTensor(const std::string& name) {
        return tensorMap.find(name) != tensorMap.end();
    }
    
    void printAllTensorNames() {
        for (const auto& t : tensors)
            printf("%s\n", t.name.c_str());
    }
    
    int getEmbedDim() const { return embedDim; }
    int getNumLayers() const { return numLayers; }
    int getNumHeads() const { return numHeads; }
    int getFFNDim() const { return ffnDim; }
    int getVocabSize() const { return vocabSize; }
    int getMaxSeqLen() const { return maxSeqLen; }
    bool isLoaded() const { return loaded; }
    std::vector<GGUFTensor>& getTensors() { return tensors; }
};

// ==================== TransformerModel ====================

class TransformerModel {
private:
    GGUFLoader loader;
    Tokenizer tokenizer;
    int embedDim, numHeads, headDim, numLayers, ffnDim, vocabSize;
    
    // GPU memory
    float *d_hidden, *d_Q, *d_K, *d_V, *d_attnOut;
    float *d_ffnHidden, *d_logits;
    float *d_tokenEmb, *d_posEmb;
    float *d_scores, *d_weights;
    
    // CPU inspection arrays
    Double2DArray lastHiddenStates;
    Double2DArray lastAttentionWeights;
    Double2DArray lastAttentionLogits;
    Double2DArray lastQVectors, lastKVectors, lastVVectors;
    Double2DArray lastLayerNormOutputs, lastFFNOutputs;
    Double2DArray lastResidualInputs, lastResidualOutputs;
    DoubleArray lastLogits;
    int lastSeqLen;
    
    cublasHandle_t cublasHandle;
    
    void allocateGPUMemory(int seqLen) {
        size_t hiddenSize = seqLen * embedDim * sizeof(float);
        size_t qkvSize = seqLen * embedDim * sizeof(float);
        size_t ffnSize = seqLen * ffnDim * sizeof(float);
        size_t logitsSize = vocabSize * sizeof(float);
        size_t scoresSize = seqLen * seqLen * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_hidden, hiddenSize));
        CUDA_CHECK(cudaMalloc(&d_Q, qkvSize));
        CUDA_CHECK(cudaMalloc(&d_K, qkvSize));
        CUDA_CHECK(cudaMalloc(&d_V, qkvSize));
        CUDA_CHECK(cudaMalloc(&d_attnOut, hiddenSize));
        CUDA_CHECK(cudaMalloc(&d_ffnHidden, ffnSize));
        CUDA_CHECK(cudaMalloc(&d_logits, logitsSize));
        CUDA_CHECK(cudaMalloc(&d_scores, scoresSize));
        CUDA_CHECK(cudaMalloc(&d_weights, scoresSize));
    }
    
    void freeGPUMemory() {
        if (d_hidden) cudaFree(d_hidden);
        if (d_Q) cudaFree(d_Q);
        if (d_K) cudaFree(d_K);
        if (d_V) cudaFree(d_V);
        if (d_attnOut) cudaFree(d_attnOut);
        if (d_ffnHidden) cudaFree(d_ffnHidden);
        if (d_logits) cudaFree(d_logits);
        if (d_scores) cudaFree(d_scores);
        if (d_weights) cudaFree(d_weights);
        if (d_tokenEmb) cudaFree(d_tokenEmb);
        if (d_posEmb) cudaFree(d_posEmb);
    }

    void copyToInspectionArrays(float* d_src, Double2DArray& dest, int layer, int size) {
        std::vector<float> h_data(size);
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_src, size * sizeof(float), cudaMemcpyDeviceToHost));
        if (dest.size() <= (size_t)layer) dest.resize(layer + 1);
        dest[layer].resize(size);
        for (int i = 0; i < size; i++) dest[layer][i] = h_data[i];
    }

public:
    TransformerModel() : embedDim(0), numHeads(0), headDim(0), numLayers(0),
                         ffnDim(0), vocabSize(0), lastSeqLen(0),
                         d_hidden(nullptr), d_Q(nullptr), d_K(nullptr), d_V(nullptr),
                         d_attnOut(nullptr), d_ffnHidden(nullptr), d_logits(nullptr),
                         d_tokenEmb(nullptr), d_posEmb(nullptr),
                         d_scores(nullptr), d_weights(nullptr) {
        cublasCreate(&cublasHandle);
    }
    
    ~TransformerModel() {
        freeGPUMemory();
        cublasDestroy(cublasHandle);
    }
    
    bool loadModel(const std::string& path) {
        if (!loader.loadFromFile(path)) return false;
        
        embedDim = loader.getEmbedDim();
        numHeads = loader.getNumHeads();
        headDim = embedDim / numHeads;
        numLayers = loader.getNumLayers();
        ffnDim = loader.getFFNDim();
        vocabSize = loader.getVocabSize();
        
        // Upload embeddings to GPU
        SingleArray tokenEmb = loader.getTensor({"token_embd.weight", "wte.weight"});
        SingleArray posEmb = loader.getTensor({"position_embd.weight", "wpe.weight"});
        
        CUDA_CHECK(cudaMalloc(&d_tokenEmb, tokenEmb.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokenEmb, tokenEmb.data(), tokenEmb.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMalloc(&d_posEmb, posEmb.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_posEmb, posEmb.data(), posEmb.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        lastHiddenStates.resize(numLayers + 1);
        lastAttentionWeights.resize(numLayers);
        lastAttentionLogits.resize(numLayers);
        lastQVectors.resize(numLayers);
        lastKVectors.resize(numLayers);
        lastVVectors.resize(numLayers);
        lastLayerNormOutputs.resize(numLayers);
        lastFFNOutputs.resize(numLayers);
        lastResidualInputs.resize(numLayers);
        lastResidualOutputs.resize(numLayers);
        
        return true;
    }
    
    bool loadTokenizer(const std::string& path) {
        return tokenizer.loadFromFile(path);
    }
    
    DoubleArray forward(const IntArray& tokenIds);
    std::string generate(const std::string& prompt, int maxTokens, double temperature = 1.0);
    
    bool isModelLoaded() const { return loader.isLoaded(); }
    bool isTokenizerLoaded() const { return tokenizer.isLoaded(); }
    
    GGUFLoader& getLoader() { return loader; }
    Tokenizer& getTokenizer() { return tokenizer; }
    
    int getEmbedDim() const { return embedDim; }
    int getNumHeads() const { return numHeads; }
    int getHeadDim() const { return headDim; }
    int getNumLayers() const { return numLayers; }
    int getFFNDim() const { return ffnDim; }
    int getVocabSize() const { return vocabSize; }
    int getLastSeqLen() const { return lastSeqLen; }
    
    Double2DArray& getLastHiddenStates() { return lastHiddenStates; }
    Double2DArray& getLastAttentionWeights() { return lastAttentionWeights; }
    Double2DArray& getLastAttentionLogits() { return lastAttentionLogits; }
    Double2DArray& getLastQVectors() { return lastQVectors; }
    Double2DArray& getLastKVectors() { return lastKVectors; }
    Double2DArray& getLastVVectors() { return lastVVectors; }
    Double2DArray& getLastLayerNormOutputs() { return lastLayerNormOutputs; }
    Double2DArray& getLastFFNOutputs() { return lastFFNOutputs; }
    Double2DArray& getLastResidualInputs() { return lastResidualInputs; }
    Double2DArray& getLastResidualOutputs() { return lastResidualOutputs; }
    DoubleArray& getLastLogits() { return lastLogits; }
};

DoubleArray TransformerModel::forward(const IntArray& tokenIds) {
    int seqLen = tokenIds.size();
    lastSeqLen = seqLen;
    
    freeGPUMemory();
    d_tokenEmb = nullptr; d_posEmb = nullptr;
    
    // Re-upload embeddings
    SingleArray tokenEmb = loader.getTensor({"token_embd.weight", "wte.weight"});
    SingleArray posEmb = loader.getTensor({"position_embd.weight", "wpe.weight"});
    CUDA_CHECK(cudaMalloc(&d_tokenEmb, tokenEmb.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokenEmb, tokenEmb.data(), tokenEmb.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_posEmb, posEmb.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_posEmb, posEmb.data(), posEmb.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    allocateGPUMemory(seqLen);
    
    // Upload token IDs
    int* d_tokenIds;
    CUDA_CHECK(cudaMalloc(&d_tokenIds, seqLen * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokenIds, tokenIds.data(), seqLen * sizeof(int), cudaMemcpyHostToDevice));
    
    // Embed tokens
    dim3 embedGrid(seqLen);
    dim3 embedBlock(embedDim);
    embed_tokens_kernel<<<embedGrid, embedBlock>>>(d_tokenEmb, d_posEmb, d_tokenIds, d_hidden, seqLen, embedDim);
    cudaFree(d_tokenIds);
    
    // Copy initial hidden state for inspection
    copyToInspectionArrays(d_hidden, lastHiddenStates, 0, seqLen * embedDim);
    
    // Process layers (simplified - full implementation would use cuBLAS for matmuls)
    for (int layer = 0; layer < numLayers; layer++) {
        // Store residual input
        copyToInspectionArrays(d_hidden, lastResidualInputs, layer, seqLen * embedDim);
        
        // Get weights for this layer
        char weightName[256];
        sprintf(weightName, "blk.%d.attn_qkv.weight", layer);
        SingleArray qkvWeight = loader.getTensor({weightName});
        sprintf(weightName, "blk.%d.attn_norm.weight", layer);
        SingleArray lnWeight = loader.getTensor({weightName});
        sprintf(weightName, "blk.%d.attn_norm.bias", layer);
        SingleArray lnBias = loader.getTensor({weightName});
        
        // Layer norm + QKV projection (simplified CPU fallback for now)
        std::vector<float> h_hidden(seqLen * embedDim);
        CUDA_CHECK(cudaMemcpy(h_hidden.data(), d_hidden, seqLen * embedDim * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::vector<float> h_Q(seqLen * embedDim), h_K(seqLen * embedDim), h_V(seqLen * embedDim);
        
        for (int pos = 0; pos < seqLen; pos++) {
            // Layer norm
            float mean = 0, var = 0;
            for (int i = 0; i < embedDim; i++) mean += h_hidden[pos * embedDim + i];
            mean /= embedDim;
            for (int i = 0; i < embedDim; i++) {
                float diff = h_hidden[pos * embedDim + i] - mean;
                var += diff * diff;
            }
            var /= embedDim;
            float invStd = 1.0f / sqrtf(var + 1e-5f);
            
            std::vector<float> normed(embedDim);
            for (int i = 0; i < embedDim; i++) {
                normed[i] = (h_hidden[pos * embedDim + i] - mean) * invStd;
                if (lnWeight.size() > (size_t)i) normed[i] *= lnWeight[i];
                if (lnBias.size() > (size_t)i) normed[i] += lnBias[i];
            }
            
            // QKV projection
            if (qkvWeight.size() >= (size_t)(3 * embedDim * embedDim)) {
                for (int i = 0; i < embedDim; i++) {
                    float q = 0, k = 0, v = 0;
                    for (int j = 0; j < embedDim; j++) {
                        q += normed[j] * qkvWeight[i * embedDim + j];
                        k += normed[j] * qkvWeight[(embedDim + i) * embedDim + j];
                        v += normed[j] * qkvWeight[(2 * embedDim + i) * embedDim + j];
                    }
                    h_Q[pos * embedDim + i] = q;
                    h_K[pos * embedDim + i] = k;
                    h_V[pos * embedDim + i] = v;
                }
            }
        }
        
        // Store QKV for inspection
        lastQVectors[layer].resize(seqLen * embedDim);
        lastKVectors[layer].resize(seqLen * embedDim);
        lastVVectors[layer].resize(seqLen * embedDim);
        for (int i = 0; i < seqLen * embedDim; i++) {
            lastQVectors[layer][i] = h_Q[i];
            lastKVectors[layer][i] = h_K[i];
            lastVVectors[layer][i] = h_V[i];
        }
        
        // Attention
        float scale = 1.0f / sqrtf((float)headDim);
        std::vector<float> attnWeights(numHeads * seqLen * seqLen);
        std::vector<float> attnLogits(numHeads * seqLen * seqLen);
        std::vector<float> attnOut(seqLen * embedDim, 0);
        
        for (int h = 0; h < numHeads; h++) {
            int headStart = h * headDim;
            for (int fromPos = 0; fromPos < seqLen; fromPos++) {
                // Compute scores
                std::vector<float> scores(seqLen);
                float maxScore = -1e9f;
                for (int toPos = 0; toPos < seqLen; toPos++) {
                    if (toPos > fromPos) {
                        scores[toPos] = -1e9f;
                    } else {
                        float score = 0;
                        for (int d = 0; d < headDim; d++) {
                            score += h_Q[fromPos * embedDim + headStart + d] * 
                                     h_K[toPos * embedDim + headStart + d];
                        }
                        scores[toPos] = score * scale;
                    }
                    attnLogits[h * seqLen * seqLen + fromPos * seqLen + toPos] = scores[toPos];
                    if (scores[toPos] > maxScore) maxScore = scores[toPos];
                }
                
                // Softmax
                float sum = 0;
                for (int toPos = 0; toPos < seqLen; toPos++) {
                    scores[toPos] = expf(scores[toPos] - maxScore);
                    sum += scores[toPos];
                }
                for (int toPos = 0; toPos < seqLen; toPos++) {
                    scores[toPos] /= sum;
                    attnWeights[h * seqLen * seqLen + fromPos * seqLen + toPos] = scores[toPos];
                }
                
                // Weighted sum of values
                for (int d = 0; d < headDim; d++) {
                    float val = 0;
                    for (int toPos = 0; toPos < seqLen; toPos++) {
                        val += scores[toPos] * h_V[toPos * embedDim + headStart + d];
                    }
                    attnOut[fromPos * embedDim + headStart + d] = val;
                }
            }
        }
        
        // Store attention weights for inspection
        lastAttentionWeights[layer].resize(attnWeights.size());
        lastAttentionLogits[layer].resize(attnLogits.size());
        for (size_t i = 0; i < attnWeights.size(); i++) {
            lastAttentionWeights[layer][i] = attnWeights[i];
            lastAttentionLogits[layer][i] = attnLogits[i];
        }
        
        // Output projection + residual
        sprintf(weightName, "blk.%d.attn_output.weight", layer);
        SingleArray projWeight = loader.getTensor({weightName});
        sprintf(weightName, "blk.%d.attn_output.bias", layer);
        SingleArray projBias = loader.getTensor({weightName});
        
        for (int pos = 0; pos < seqLen; pos++) {
            for (int i = 0; i < embedDim; i++) {
                float sum = projBias.size() > (size_t)i ? projBias[i] : 0;
                for (int j = 0; j < embedDim; j++) {
                    if (projWeight.size() > (size_t)(i * embedDim + j))
                        sum += attnOut[pos * embedDim + j] * projWeight[i * embedDim + j];
                }
                h_hidden[pos * embedDim + i] += sum;
            }
        }
        
        // FFN (simplified)
        sprintf(weightName, "blk.%d.ffn_up.weight", layer);
        SingleArray upWeight = loader.getTensor({weightName});
        sprintf(weightName, "blk.%d.ffn_down.weight", layer);
        SingleArray downWeight = loader.getTensor({weightName});
        
        if (upWeight.size() > 0 && downWeight.size() > 0) {
            for (int pos = 0; pos < seqLen; pos++) {
                std::vector<float> ffnHidden(ffnDim);
                for (int i = 0; i < ffnDim; i++) {
                    float sum = 0;
                    for (int j = 0; j < embedDim; j++)
                        sum += h_hidden[pos * embedDim + j] * upWeight[i * embedDim + j];
                    // GELU
                    ffnHidden[i] = 0.5f * sum * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
                }
                for (int i = 0; i < embedDim; i++) {
                    float sum = 0;
                    for (int j = 0; j < ffnDim; j++)
                        sum += ffnHidden[j] * downWeight[i * ffnDim + j];
                    h_hidden[pos * embedDim + i] += sum;
                }
            }
        }
        
        lastFFNOutputs[layer].resize(seqLen * embedDim);
        for (int i = 0; i < seqLen * embedDim; i++)
            lastFFNOutputs[layer][i] = h_hidden[i];
        
        // Store residual output
        copyToInspectionArrays(d_hidden, lastResidualOutputs, layer, seqLen * embedDim);
        
        // Copy back to GPU
        CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(), seqLen * embedDim * sizeof(float), cudaMemcpyHostToDevice));
        
        // Store hidden state
        copyToInspectionArrays(d_hidden, lastHiddenStates, layer + 1, seqLen * embedDim);
    }
    
    // Final layer norm + logits
    std::vector<float> h_hidden(seqLen * embedDim);
    CUDA_CHECK(cudaMemcpy(h_hidden.data(), d_hidden, seqLen * embedDim * sizeof(float), cudaMemcpyDeviceToHost));
    
    SingleArray finalLnW = loader.getTensor({"output_norm.weight", "ln_f.weight"});
    SingleArray finalLnB = loader.getTensor({"output_norm.bias", "ln_f.bias"});
    
    // Normalize last position
    std::vector<float> lastPos(embedDim);
    float mean = 0, var = 0;
    for (int i = 0; i < embedDim; i++) mean += h_hidden[(seqLen - 1) * embedDim + i];
    mean /= embedDim;
    for (int i = 0; i < embedDim; i++) {
        float diff = h_hidden[(seqLen - 1) * embedDim + i] - mean;
        var += diff * diff;
    }
    var /= embedDim;
    float invStd = 1.0f / sqrtf(var + 1e-5f);
    
    for (int i = 0; i < embedDim; i++) {
        lastPos[i] = (h_hidden[(seqLen - 1) * embedDim + i] - mean) * invStd;
        if (finalLnW.size() > (size_t)i) lastPos[i] *= finalLnW[i];
        if (finalLnB.size() > (size_t)i) lastPos[i] += finalLnB[i];
    }
    
    // Compute logits
    lastLogits.resize(vocabSize);
    for (int i = 0; i < vocabSize; i++) {
        float sum = 0;
        for (int j = 0; j < embedDim; j++)
            sum += lastPos[j] * tokenEmb[i * embedDim + j];
        lastLogits[i] = sum;
    }
    
    return lastLogits;
}

std::string TransformerModel::generate(const std::string& prompt, int maxTokens, double temperature) {
    if (!loader.isLoaded() || !tokenizer.isLoaded()) return "";
    
    IntArray tokenIds = tokenizer.encode(prompt);
    if (tokenIds.empty()) return "";
    
    for (int t = 0; t < maxTokens; t++) {
        DoubleArray logits = forward(tokenIds);
        if (logits.empty()) break;
        
        int selectedId;
        if (temperature <= 0.01) {
            selectedId = 0;
            for (int i = 1; i < vocabSize; i++)
                if (logits[i] > logits[selectedId]) selectedId = i;
        } else {
            // Apply temperature and sample
            double maxLogit = logits[0];
            for (int i = 1; i < vocabSize; i++)
                if (logits[i] > maxLogit) maxLogit = logits[i];
            
            double sum = 0;
            std::vector<double> probs(vocabSize);
            for (int i = 0; i < vocabSize; i++) {
                probs[i] = exp((logits[i] - maxLogit) / temperature);
                sum += probs[i];
            }
            for (int i = 0; i < vocabSize; i++) probs[i] /= sum;
            
            double r = (double)rand() / RAND_MAX;
            double cumProb = 0;
            selectedId = 0;
            for (int i = 0; i < vocabSize; i++) {
                cumProb += probs[i];
                if (r <= cumProb) { selectedId = i; break; }
            }
        }
        
        tokenIds.push_back(selectedId);
        if (selectedId == 50256) break; // EOS
    }
    
    return tokenizer.decode(tokenIds);
}

// ==================== TransformerFacade ====================

class TransformerFacade {
private:
    TransformerModel model;
    bool ownsModel;
    Double3DArray kvCacheK, kvCacheV;
    
public:
    TransformerFacade() : ownsModel(true) {}
    
    bool loadModel(const std::string& path) { return model.loadModel(path); }
    bool loadTokenizer(const std::string& path) { return model.loadTokenizer(path); }
    DoubleArray runForward(const IntArray& tokenIds) { return model.forward(tokenIds); }
    std::string generate(const std::string& prompt, int maxTokens, double temp = 1.0) {
        return model.generate(prompt, maxTokens, temp);
    }
    
    // Structural introspection
    int getNumLayers() { return model.getNumLayers(); }
    int getNumHeads(int layer = 0) { return model.getNumHeads(); }
    int getHiddenSize(int layer = 0) { return model.getEmbedDim(); }
    int getHeadDim() { return model.getHeadDim(); }
    int getFFNDim() { return model.getFFNDim(); }
    int getVocabSize() { return model.getVocabSize(); }
    int getMaxSeqLen() { return model.getLoader().getMaxSeqLen(); }
    int getLastSeqLen() { return model.getLastSeqLen(); }
    bool isModelLoaded() { return model.isModelLoaded(); }
    bool isTokenizerLoaded() { return model.isTokenizerLoaded(); }
    
    TransformerModel& getModel() { return model; }
    
    // Token embedding
    DoubleArray getTokenEmbedding(int tokenId) {
        if (!isModelLoaded()) return DoubleArray();
        SingleArray emb = model.getLoader().getTensor({"token_embd.weight", "wte.weight"});
        int dim = model.getEmbedDim();
        DoubleArray result(dim);
        for (int i = 0; i < dim; i++)
            result[i] = emb[tokenId * dim + i];
        return result;
    }
    
    DoubleArray getPositionalEncoding(int pos) {
        if (!isModelLoaded()) return DoubleArray();
        SingleArray emb = model.getLoader().getTensor({"position_embd.weight", "wpe.weight"});
        int dim = model.getEmbedDim();
        DoubleArray result(dim);
        for (int i = 0; i < dim; i++)
            result[i] = emb[pos * dim + i];
        return result;
    }
    
    // Attention inspection
    double getAttentionWeights(int layer, int head, int fromPos, int toPos) {
        if (!isModelLoaded()) return 0;
        auto& weights = model.getLastAttentionWeights();
        if (layer >= (int)weights.size()) return 0;
        int seqLen = model.getLastSeqLen();
        int idx = head * seqLen * seqLen + fromPos * seqLen + toPos;
        return (idx < (int)weights[layer].size()) ? weights[layer][idx] : 0;
    }
    
    double getAttentionLogits(int layer, int head, int fromPos, int toPos) {
        if (!isModelLoaded()) return 0;
        auto& logits = model.getLastAttentionLogits();
        if (layer >= (int)logits.size()) return 0;
        int seqLen = model.getLastSeqLen();
        int idx = head * seqLen * seqLen + fromPos * seqLen + toPos;
        return (idx < (int)logits[layer].size()) ? logits[layer][idx] : 0;
    }
    
    // Hidden state
    DoubleArray getHiddenState(int layer, int pos) {
        if (!isModelLoaded()) return DoubleArray();
        auto& states = model.getLastHiddenStates();
        if (layer >= (int)states.size()) return DoubleArray();
        int dim = model.getEmbedDim();
        DoubleArray result(dim);
        for (int i = 0; i < dim; i++)
            result[i] = states[layer][pos * dim + i];
        return result;
    }
    
    // QKV
    DoubleArray getQKV(int layer, int head, QKVType type, int pos) {
        if (!isModelLoaded()) return DoubleArray();
        Double2DArray* src;
        switch (type) {
            case qkvQuery: src = &model.getLastQVectors(); break;
            case qkvKey: src = &model.getLastKVectors(); break;
            case qkvValue: src = &model.getLastVVectors(); break;
        }
        if (layer >= (int)src->size()) return DoubleArray();
        int headDim = model.getHeadDim();
        int embedDim = model.getEmbedDim();
        int headStart = head * headDim;
        DoubleArray result(headDim);
        for (int i = 0; i < headDim; i++)
            result[i] = (*src)[layer][pos * embedDim + headStart + i];
        return result;
    }
    
    // Layer norm output
    DoubleArray getLayerNormOutput(int layer, int pos) {
        if (!isModelLoaded()) return DoubleArray();
        auto& outputs = model.getLastLayerNormOutputs();
        if (layer >= (int)outputs.size()) return DoubleArray();
        int dim = model.getEmbedDim();
        DoubleArray result(dim);
        for (int i = 0; i < dim; i++)
            result[i] = outputs[layer][pos * dim + i];
        return result;
    }
    
    // FFN output
    DoubleArray getFFNOutput(int layer, int pos) {
        if (!isModelLoaded()) return DoubleArray();
        auto& outputs = model.getLastFFNOutputs();
        if (layer >= (int)outputs.size()) return DoubleArray();
        int dim = model.getEmbedDim();
        DoubleArray result(dim);
        for (int i = 0; i < dim; i++)
            result[i] = outputs[layer][pos * dim + i];
        return result;
    }
    
    // Logits
    DoubleArray getLogits(int pos = -1) {
        return model.getLastLogits();
    }
    
    DoubleArray getSoftmaxOutput(int pos = -1) {
        DoubleArray logits = getLogits(pos);
        if (logits.empty()) return DoubleArray();
        double maxVal = logits[0];
        for (size_t i = 1; i < logits.size(); i++)
            if (logits[i] > maxVal) maxVal = logits[i];
        double sum = 0;
        DoubleArray result(logits.size());
        for (size_t i = 0; i < logits.size(); i++) {
            result[i] = exp(logits[i] - maxVal);
            sum += result[i];
        }
        for (size_t i = 0; i < result.size(); i++)
            result[i] /= sum;
        return result;
    }
    
    // Weight access
    SingleArray getWeight(int layer, ParamType type) {
        if (!isModelLoaded()) return SingleArray();
        char name[256];
        switch (type) {
            case ptQProj: case ptKProj: case ptVProj:
                sprintf(name, "blk.%d.attn_qkv.weight", layer); break;
            case ptOutProj:
                sprintf(name, "blk.%d.attn_output.weight", layer); break;
            case ptFFN1:
                sprintf(name, "blk.%d.ffn_up.weight", layer); break;
            case ptFFN2:
                sprintf(name, "blk.%d.ffn_down.weight", layer); break;
            case ptLayerNorm1Weight:
                sprintf(name, "blk.%d.attn_norm.weight", layer); break;
            case ptLayerNorm1Bias:
                sprintf(name, "blk.%d.attn_norm.bias", layer); break;
            case ptLayerNorm2Weight:
                sprintf(name, "blk.%d.ffn_norm.weight", layer); break;
            case ptLayerNorm2Bias:
                sprintf(name, "blk.%d.ffn_norm.bias", layer); break;
            case ptTokenEmbed:
                strcpy(name, "token_embd.weight"); break;
            case ptPosEmbed:
                strcpy(name, "position_embd.weight"); break;
            case ptFinalNormWeight:
                strcpy(name, "output_norm.weight"); break;
            case ptFinalNormBias:
                strcpy(name, "output_norm.bias"); break;
        }
        return model.getLoader().getTensor({name});
    }
    
    Int64Array getWeightShape(int layer, ParamType type) {
        if (!isModelLoaded()) return Int64Array();
        char name[256];
        switch (type) {
            case ptQProj: case ptKProj: case ptVProj:
                sprintf(name, "blk.%d.attn_qkv.weight", layer); break;
            case ptOutProj:
                sprintf(name, "blk.%d.attn_output.weight", layer); break;
            case ptFFN1:
                sprintf(name, "blk.%d.ffn_up.weight", layer); break;
            case ptFFN2:
                sprintf(name, "blk.%d.ffn_down.weight", layer); break;
            default:
                return Int64Array();
        }
        return model.getLoader().getTensorShape({name});
    }
    
    // KV Cache
    DoubleArray getKeyValueCache(int layer, int head, bool isKey, int pos) {
        if (!isModelLoaded()) return DoubleArray();
        Double2DArray* src = isKey ? &model.getLastKVectors() : &model.getLastVVectors();
        if (layer >= (int)src->size()) return DoubleArray();
        int headDim = model.getHeadDim();
        int embedDim = model.getEmbedDim();
        int headStart = head * headDim;
        DoubleArray result(headDim);
        for (int i = 0; i < headDim; i++)
            result[i] = (*src)[layer][pos * embedDim + headStart + i];
        return result;
    }
    
    // Layer norm stats
    void getLayerNormStats(int layer, double& mean, double& stddev) {
        mean = stddev = 0;
        if (!isModelLoaded()) return;
        auto& outputs = model.getLastLayerNormOutputs();
        if (layer >= (int)outputs.size() || outputs[layer].empty()) return;
        
        double sum = 0;
        for (double v : outputs[layer]) sum += v;
        mean = sum / outputs[layer].size();
        
        double varSum = 0;
        for (double v : outputs[layer]) {
            double diff = v - mean;
            varSum += diff * diff;
        }
        stddev = sqrt(varSum / outputs[layer].size());
    }
    
    // Residual access
    DoubleArray getResidualInput(int layer, int pos) {
        if (!isModelLoaded()) return DoubleArray();
        auto& inputs = model.getLastResidualInputs();
        if (layer >= (int)inputs.size()) return DoubleArray();
        int dim = model.getEmbedDim();
        DoubleArray result(dim);
        for (int i = 0; i < dim; i++)
            result[i] = inputs[layer][pos * dim + i];
        return result;
    }
    
    DoubleArray getResidualOutput(int layer, int pos) {
        if (!isModelLoaded()) return DoubleArray();
        auto& outputs = model.getLastResidualOutputs();
        if (layer >= (int)outputs.size()) return DoubleArray();
        int dim = model.getEmbedDim();
        DoubleArray result(dim);
        for (int i = 0; i < dim; i++)
            result[i] = outputs[layer][pos * dim + i];
        return result;
    }
    
    // Activation histogram
    DoubleArray getActivationHistogram(int layer, int head, int numBins = 50) {
        if (!isModelLoaded()) return DoubleArray();
        auto& states = model.getLastHiddenStates();
        if (layer >= (int)states.size() || states[layer].empty()) return DoubleArray();
        
        int seqLen = model.getLastSeqLen();
        int headDim = model.getHeadDim();
        int embedDim = model.getEmbedDim();
        int headStart = head * headDim;
        
        double minVal = states[layer][headStart];
        double maxVal = minVal;
        for (int pos = 0; pos < seqLen; pos++) {
            for (int d = 0; d < headDim; d++) {
                double v = states[layer][pos * embedDim + headStart + d];
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }
        }
        
        double range = maxVal - minVal;
        if (range == 0) range = 1;
        
        DoubleArray hist(numBins, 0);
        int count = 0;
        for (int pos = 0; pos < seqLen; pos++) {
            for (int d = 0; d < headDim; d++) {
                double v = states[layer][pos * embedDim + headStart + d];
                int bin = (int)((v - minVal) / range * (numBins - 1));
                if (bin >= numBins) bin = numBins - 1;
                if (bin < 0) bin = 0;
                hist[bin]++;
                count++;
            }
        }
        for (int i = 0; i < numBins; i++) hist[i] /= count;
        return hist;
    }
    
    // Attention entropy
    double getAttentionEntropy(int layer, int head) {
        if (!isModelLoaded()) return 0;
        auto& weights = model.getLastAttentionWeights();
        if (layer >= (int)weights.size()) return 0;
        
        int seqLen = model.getLastSeqLen();
        double sum = 0;
        for (int pos = 0; pos < seqLen; pos++) {
            for (int src = 0; src < seqLen; src++) {
                int idx = head * seqLen * seqLen + pos * seqLen + src;
                double w = weights[layer][idx];
                if (w > 1e-10) sum -= w * log(w);
            }
        }
        return sum / seqLen;
    }
    
    // Saliency map
    DoubleArray getSaliencyMap(int tokenIdx, int layer) {
        DoubleArray hidden = getHiddenState(layer, tokenIdx);
        if (hidden.empty()) return DoubleArray();
        
        double maxAbs = 0;
        for (double v : hidden) if (fabs(v) > maxAbs) maxAbs = fabs(v);
        
        DoubleArray result(hidden.size());
        for (size_t i = 0; i < hidden.size(); i++)
            result[i] = (maxAbs > 0) ? fabs(hidden[i]) / maxAbs : 0;
        return result;
    }
    
    // Activation trace
    DoubleArray getActivationTrace(int tokenIdx, int layer) {
        return getHiddenState(layer, tokenIdx);
    }
};

// ==================== Main Program ====================

void printDoubleArray(const DoubleArray& arr, const std::string& name, int maxItems = 20) {
    printf("%s (%zu elements):\n", name.c_str(), arr.size());
    if (arr.empty()) { printf("  <empty>\n"); return; }
    printf("  [");
    for (size_t i = 0; i < std::min((size_t)maxItems, arr.size()); i++) {
        if (i > 0) printf(", ");
        printf("%.6f", arr[i]);
    }
    if (arr.size() > (size_t)maxItems) printf(", ... (%zu more)", arr.size() - maxItems);
    printf("]\n");
}

void printSingleArray(const SingleArray& arr, const std::string& name, int maxItems = 20) {
    printf("%s (%zu elements):\n", name.c_str(), arr.size());
    if (arr.empty()) { printf("  <empty>\n"); return; }
    printf("  [");
    for (size_t i = 0; i < std::min((size_t)maxItems, arr.size()); i++) {
        if (i > 0) printf(", ");
        printf("%.6f", arr[i]);
    }
    if (arr.size() > (size_t)maxItems) printf(", ... (%zu more)", arr.size() - maxItems);
    printf("]\n");
}

void printInt64Array(const Int64Array& arr, const std::string& name) {
    printf("%s: [", name.c_str());
    for (size_t i = 0; i < arr.size(); i++) {
        if (i > 0) printf(", ");
        printf("%ld", arr[i]);
    }
    printf("]\n");
}

void printIntArray(const IntArray& arr, const std::string& name) {
    printf("%s: [", name.c_str());
    for (size_t i = 0; i < arr.size(); i++) {
        if (i > 0) printf(", ");
        printf("%d", arr[i]);
    }
    printf("]\n");
}

ParamType parseParamType(const std::string& s) {
    if (s == "qproj") return ptQProj;
    if (s == "kproj") return ptKProj;
    if (s == "vproj") return ptVProj;
    if (s == "outproj") return ptOutProj;
    if (s == "ffn1") return ptFFN1;
    if (s == "ffn2") return ptFFN2;
    if (s == "ln1w") return ptLayerNorm1Weight;
    if (s == "ln1b") return ptLayerNorm1Bias;
    if (s == "ln2w") return ptLayerNorm2Weight;
    if (s == "ln2b") return ptLayerNorm2Bias;
    if (s == "tokemb") return ptTokenEmbed;
    if (s == "posemb") return ptPosEmbed;
    if (s == "finalnormw") return ptFinalNormWeight;
    if (s == "finalnormb") return ptFinalNormBias;
    return ptQProj;
}

void printUsage() {
    printf("TransformerFacade CUDA - Pascal Transformer with Inspection Facade\n");
    printf("Matthew Abbott 2025\n\n");
    printf("Usage: facadedtransformer_cuda -m <model> [options] [inspection commands]\n\n");
    printf("Required:\n");
    printf("  -m, --model <path>           Path to GGUF model file\n\n");
    printf("Basic Options:\n");
    printf("  -t, --tokenizer <path>       Path to tokenizer.json file\n");
    printf("  -p, --prompt <text>          Input prompt (runs forward pass)\n");
    printf("  -h, --help                   Show this help message\n\n");
    printf("Generation:\n");
    printf("  --generate                   Generate text from prompt\n");
    printf("  -n, --max-tokens <n>         Max tokens (default: 50)\n");
    printf("  -T, --temperature <f>        Temperature (default: 1.0)\n\n");
    printf("Model Inspection:\n");
    printf("  -i, --info                   Show model architecture\n");
    printf("  -l, --list-tensors           List tensor names\n\n");
    printf("Tokenizer:\n");
    printf("  --encode <text>              Encode text to IDs\n");
    printf("  --decode <ids>               Decode IDs to text\n\n");
    printf("Embeddings:\n");
    printf("  --token-embed <id>           Token embedding\n");
    printf("  --pos-embed <pos>            Positional encoding\n\n");
    printf("Attention (requires -p):\n");
    printf("  --attention <L> <H> <F> <T>  Attention weight\n");
    printf("  --attn-logits <L> <H> <F> <T> Attention logit\n");
    printf("  --entropy <layer> <head>     Attention entropy\n");
    printf("  --entropy-all                All entropy values\n\n");
    printf("Activations (requires -p):\n");
    printf("  --hidden <layer> <pos>       Hidden state\n");
    printf("  --qkv <L> <H> <q/k/v> <pos>  Q/K/V vector\n");
    printf("  --logits                     Output logits\n");
    printf("  --softmax                    Softmax output\n\n");
    printf("Weights:\n");
    printf("  --weight <layer> <type>      Weight tensor\n");
    printf("  --weight-shape <layer> <type> Weight shape\n\n");
    printf("Statistics (requires -p):\n");
    printf("  --ln-stats <layer>           Layer norm stats\n");
    printf("  --histogram <L> <H> [bins]   Activation histogram\n");
    printf("  --saliency <token> <layer>   Saliency map\n\n");
    printf("Output:\n");
    printf("  --max-print <n>              Max elements to print\n");
    printf("  --full                       Print full arrays\n");
}

int main(int argc, char** argv) {
    std::string modelPath, tokenizerPath, prompt;
    int maxTokens = 50, maxPrint = 20;
    double temperature = 1.0;
    bool fullPrint = false, doGenerate = false;
    
    enum Mode {
        mNone, mInfo, mListTensors, mEncode, mDecode, mTokenEmbed, mPosEmbed,
        mAttention, mAttnLogits, mEntropy, mEntropyAll, mHidden, mQKV,
        mLogits, mSoftmax, mWeight, mWeightShape, mLNStats, mLNStatsAll,
        mHistogram, mSaliency, mFFNOutput, mResidualIn, mResidualOut
    } mode = mNone;
    
    int argLayer = 0, argHead = 0, argPos = -1, argFrom = 0, argTo = 0, argTokenId = 0;
    int argBins = 50;
    QKVType argQKVType = qkvQuery;
    ParamType argParamType = ptQProj;
    std::string argText, argIDs;
    
    if (argc == 1) { printUsage(); return 0; }
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") { printUsage(); return 0; }
        else if (arg == "-m" || arg == "--model") { if (++i < argc) modelPath = argv[i]; }
        else if (arg == "-t" || arg == "--tokenizer") { if (++i < argc) tokenizerPath = argv[i]; }
        else if (arg == "-p" || arg == "--prompt") { if (++i < argc) prompt = argv[i]; }
        else if (arg == "-n" || arg == "--max-tokens") { if (++i < argc) maxTokens = atoi(argv[i]); }
        else if (arg == "-T" || arg == "--temperature") { if (++i < argc) temperature = atof(argv[i]); }
        else if (arg == "--max-print") { if (++i < argc) maxPrint = atoi(argv[i]); }
        else if (arg == "--full") { fullPrint = true; }
        else if (arg == "-i" || arg == "--info") { mode = mInfo; }
        else if (arg == "-l" || arg == "--list-tensors") { mode = mListTensors; }
        else if (arg == "--generate") { doGenerate = true; }
        else if (arg == "--encode") { mode = mEncode; if (++i < argc) argText = argv[i]; }
        else if (arg == "--decode") { mode = mDecode; if (++i < argc) argIDs = argv[i]; }
        else if (arg == "--token-embed") { mode = mTokenEmbed; if (++i < argc) argTokenId = atoi(argv[i]); }
        else if (arg == "--pos-embed") { mode = mPosEmbed; if (++i < argc) argPos = atoi(argv[i]); }
        else if (arg == "--attention") {
            mode = mAttention;
            if (++i < argc) argLayer = atoi(argv[i]);
            if (++i < argc) argHead = atoi(argv[i]);
            if (++i < argc) argFrom = atoi(argv[i]);
            if (++i < argc) argTo = atoi(argv[i]);
        }
        else if (arg == "--attn-logits") {
            mode = mAttnLogits;
            if (++i < argc) argLayer = atoi(argv[i]);
            if (++i < argc) argHead = atoi(argv[i]);
            if (++i < argc) argFrom = atoi(argv[i]);
            if (++i < argc) argTo = atoi(argv[i]);
        }
        else if (arg == "--entropy") {
            mode = mEntropy;
            if (++i < argc) argLayer = atoi(argv[i]);
            if (++i < argc) argHead = atoi(argv[i]);
        }
        else if (arg == "--entropy-all") { mode = mEntropyAll; }
        else if (arg == "--hidden") {
            mode = mHidden;
            if (++i < argc) argLayer = atoi(argv[i]);
            if (++i < argc) argPos = atoi(argv[i]);
        }
        else if (arg == "--qkv") {
            mode = mQKV;
            if (++i < argc) argLayer = atoi(argv[i]);
            if (++i < argc) argHead = atoi(argv[i]);
            if (++i < argc) {
                if (std::string(argv[i]) == "q") argQKVType = qkvQuery;
                else if (std::string(argv[i]) == "k") argQKVType = qkvKey;
                else if (std::string(argv[i]) == "v") argQKVType = qkvValue;
            }
            if (++i < argc) argPos = atoi(argv[i]);
        }
        else if (arg == "--logits") { mode = mLogits; }
        else if (arg == "--softmax") { mode = mSoftmax; }
        else if (arg == "--weight") {
            mode = mWeight;
            if (++i < argc) argLayer = atoi(argv[i]);
            if (++i < argc) argParamType = parseParamType(argv[i]);
        }
        else if (arg == "--weight-shape") {
            mode = mWeightShape;
            if (++i < argc) argLayer = atoi(argv[i]);
            if (++i < argc) argParamType = parseParamType(argv[i]);
        }
        else if (arg == "--ln-stats") {
            mode = mLNStats;
            if (++i < argc) argLayer = atoi(argv[i]);
        }
        else if (arg == "--ln-stats-all") { mode = mLNStatsAll; }
        else if (arg == "--histogram") {
            mode = mHistogram;
            if (++i < argc) argLayer = atoi(argv[i]);
            if (++i < argc) argHead = atoi(argv[i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') argBins = atoi(argv[++i]);
        }
        else if (arg == "--saliency") {
            mode = mSaliency;
            if (++i < argc) argTokenId = atoi(argv[i]);
            if (++i < argc) argLayer = atoi(argv[i]);
        }
        else if (arg == "--ffn-output") {
            mode = mFFNOutput;
            if (++i < argc) argLayer = atoi(argv[i]);
            if (++i < argc) argPos = atoi(argv[i]);
        }
        else if (arg == "--residual-in") {
            mode = mResidualIn;
            if (++i < argc) argLayer = atoi(argv[i]);
            if (++i < argc) argPos = atoi(argv[i]);
        }
        else if (arg == "--residual-out") {
            mode = mResidualOut;
            if (++i < argc) argLayer = atoi(argv[i]);
            if (++i < argc) argPos = atoi(argv[i]);
        }
        else {
            printf("Unknown option: %s\n", arg.c_str());
            return 1;
        }
    }
    
    if (fullPrint) maxPrint = INT_MAX;
    
    if (modelPath.empty()) {
        printf("Error: Model path required (-m)\n");
        return 1;
    }
    
    TransformerFacade facade;
    
    printf("Loading model: %s\n", modelPath.c_str());
    if (!facade.loadModel(modelPath)) {
        printf("Error: Failed to load model\n");
        return 1;
    }
    printf("Model loaded successfully\n\n");
    
    if (mode == mListTensors) {
        printf("Tensor names:\n");
        facade.getModel().getLoader().printAllTensorNames();
        return 0;
    }
    
    if (mode == mInfo) {
        printf("Model Architecture:\n");
        printf("  Embedding dimension: %d\n", facade.getHiddenSize());
        printf("  Number of layers:    %d\n", facade.getNumLayers());
        printf("  Number of heads:     %d\n", facade.getNumHeads());
        printf("  Head dimension:      %d\n", facade.getHeadDim());
        printf("  FFN dimension:       %d\n", facade.getFFNDim());
        printf("  Vocab size:          %d\n", facade.getVocabSize());
        printf("  Max sequence length: %d\n", facade.getMaxSeqLen());
        return 0;
    }
    
    if (!tokenizerPath.empty()) {
        printf("Loading tokenizer: %s\n", tokenizerPath.c_str());
        if (!facade.loadTokenizer(tokenizerPath)) {
            printf("Error: Failed to load tokenizer\n");
            return 1;
        }
        printf("Tokenizer loaded (vocab: %d)\n\n", facade.getModel().getTokenizer().getVocabSize());
    }
    
    if (mode == mEncode) {
        IntArray ids = facade.getModel().getTokenizer().encode(argText);
        char buf[256];
        sprintf(buf, "Token IDs for \"%s\"", argText.c_str());
        printIntArray(ids, buf);
        return 0;
    }
    
    if (mode == mDecode) {
        IntArray ids;
        std::stringstream ss(argIDs);
        std::string token;
        while (std::getline(ss, token, ',')) ids.push_back(atoi(token.c_str()));
        printf("Decoded: \"%s\"\n", facade.getModel().getTokenizer().decode(ids).c_str());
        return 0;
    }
    
    if (mode == mTokenEmbed) {
        char buf[256];
        sprintf(buf, "Token embedding for ID %d", argTokenId);
        printDoubleArray(facade.getTokenEmbedding(argTokenId), buf, maxPrint);
        return 0;
    }
    
    if (mode == mPosEmbed) {
        char buf[256];
        sprintf(buf, "Positional encoding for pos %d", argPos);
        printDoubleArray(facade.getPositionalEncoding(argPos), buf, maxPrint);
        return 0;
    }
    
    if (mode == mWeight) {
        char buf[256];
        sprintf(buf, "Weight layer %d", argLayer);
        printSingleArray(facade.getWeight(argLayer, argParamType), buf, maxPrint);
        return 0;
    }
    
    if (mode == mWeightShape) {
        char buf[256];
        sprintf(buf, "Weight shape layer %d", argLayer);
        printInt64Array(facade.getWeightShape(argLayer, argParamType), buf);
        return 0;
    }
    
    // Commands requiring forward pass
    bool needForward = (mode >= mAttention && mode <= mSaliency) || mode == mFFNOutput || 
                       mode == mResidualIn || mode == mResidualOut;
    
    if (needForward) {
        if (prompt.empty()) {
            printf("Error: Prompt (-p) required\n");
            return 1;
        }
        if (!facade.isTokenizerLoaded()) {
            printf("Error: Tokenizer (-t) required\n");
            return 1;
        }
        IntArray ids = facade.getModel().getTokenizer().encode(prompt);
        printf("Running forward pass: \"%s\" (%zu tokens)\n", prompt.c_str(), ids.size());
        facade.runForward(ids);
        printf("Forward pass complete\n\n");
    }
    
    switch (mode) {
        case mAttention:
            printf("Attention weight [L=%d,H=%d,F=%d,T=%d]: %f\n",
                argLayer, argHead, argFrom, argTo,
                facade.getAttentionWeights(argLayer, argHead, argFrom, argTo));
            break;
        case mAttnLogits:
            printf("Attention logit [L=%d,H=%d,F=%d,T=%d]: %f\n",
                argLayer, argHead, argFrom, argTo,
                facade.getAttentionLogits(argLayer, argHead, argFrom, argTo));
            break;
        case mEntropy:
            printf("Attention entropy [L=%d,H=%d]: %f\n", argLayer, argHead,
                facade.getAttentionEntropy(argLayer, argHead));
            break;
        case mEntropyAll:
            printf("Attention entropy for all layers/heads:\n");
            for (int l = 0; l < facade.getNumLayers(); l++) {
                printf("  Layer %2d: ", l);
                for (int h = 0; h < facade.getNumHeads(); h++)
                    printf("H%d=%.4f ", h, facade.getAttentionEntropy(l, h));
                printf("\n");
            }
            break;
        case mHidden: {
            char buf[256];
            sprintf(buf, "Hidden state [L=%d,pos=%d]", argLayer, argPos);
            printDoubleArray(facade.getHiddenState(argLayer, argPos), buf, maxPrint);
        } break;
        case mQKV: {
            char buf[256];
            sprintf(buf, "QKV [L=%d,H=%d,type=%d,pos=%d]", argLayer, argHead, (int)argQKVType, argPos);
            printDoubleArray(facade.getQKV(argLayer, argHead, argQKVType, argPos), buf, maxPrint);
        } break;
        case mLogits:
            printDoubleArray(facade.getLogits(), "Logits", maxPrint);
            break;
        case mSoftmax:
            printDoubleArray(facade.getSoftmaxOutput(), "Softmax", maxPrint);
            break;
        case mLNStats: {
            double mean, stddev;
            facade.getLayerNormStats(argLayer, mean, stddev);
            printf("Layer norm [L=%d]: mean=%.6f, stddev=%.6f\n", argLayer, mean, stddev);
        } break;
        case mLNStatsAll:
            printf("Layer norm stats:\n");
            for (int l = 0; l < facade.getNumLayers(); l++) {
                double mean, stddev;
                facade.getLayerNormStats(l, mean, stddev);
                printf("  Layer %2d: mean=%.6f, stddev=%.6f\n", l, mean, stddev);
            }
            break;
        case mHistogram: {
            char buf[256];
            sprintf(buf, "Histogram [L=%d,H=%d,bins=%d]", argLayer, argHead, argBins);
            printDoubleArray(facade.getActivationHistogram(argLayer, argHead, argBins), buf, maxPrint);
        } break;
        case mSaliency: {
            char buf[256];
            sprintf(buf, "Saliency [tok=%d,L=%d]", argTokenId, argLayer);
            printDoubleArray(facade.getSaliencyMap(argTokenId, argLayer), buf, maxPrint);
        } break;
        case mFFNOutput: {
            char buf[256];
            sprintf(buf, "FFN output [L=%d,pos=%d]", argLayer, argPos);
            printDoubleArray(facade.getFFNOutput(argLayer, argPos), buf, maxPrint);
        } break;
        case mResidualIn: {
            char buf[256];
            sprintf(buf, "Residual input [L=%d,pos=%d]", argLayer, argPos);
            printDoubleArray(facade.getResidualInput(argLayer, argPos), buf, maxPrint);
        } break;
        case mResidualOut: {
            char buf[256];
            sprintf(buf, "Residual output [L=%d,pos=%d]", argLayer, argPos);
            printDoubleArray(facade.getResidualOutput(argLayer, argPos), buf, maxPrint);
        } break;
        default:
            break;
    }
    
    if (doGenerate) {
        if (!facade.isTokenizerLoaded()) {
            printf("Error: Tokenizer required for generation\n");
            return 1;
        }
        if (prompt.empty()) {
            printf("Error: Prompt required for generation\n");
            return 1;
        }
        printf("Generating (max=%d, temp=%.2f):\n", maxTokens, temperature);
        srand(time(nullptr));
        std::string output = facade.generate(prompt, maxTokens, temperature);
        printf("=== Generated Output ===\n%s\n========================\n", output.c_str());
    }
    
    if (mode == mNone && !doGenerate) {
        printf("No inspection command. Use --help for options.\n");
    }
    
    return 0;
}
