//
// GGUF f32 CLI Transformer - CUDA Implementation
// Matthew Abbott 2025
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>
#include <iomanip>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

constexpr int MAX_SEQ_LEN = 1024;
constexpr const char* GGUF_MAGIC = "GGUF";
constexpr int BLOCK_SIZE = 256;

// ==================== CUDA Kernels ====================

__global__ void matmulKernel(const float* A, const float* B, float* C,
                              int M, int N, int K, const float* bias) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = (bias != nullptr) ? bias[col] : 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmulTransposeKernel(const float* A, const float* B, float* C,
                                       int M, int N, int K, const float* bias) {
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

__global__ void geluKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        data[idx] = x * cdf;
    }
}

__global__ void layerNormKernel(const float* input, float* output,
                                 const float* gamma, const float* beta,
                                 int seqLen, int dim) {
    int pos = blockIdx.x;
    if (pos >= seqLen) return;
    
    extern __shared__ float shared[];
    float* sdata = shared;
    
    int tid = threadIdx.x;
    int offset = pos * dim;
    
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += input[offset + i];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / dim;
    __syncthreads();
    
    sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = input[offset + i] - mean;
        sum += diff * diff;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float variance = sdata[0] / dim;
    float invStd = rsqrtf(variance + 1e-5f);
    
    for (int i = tid; i < dim; i += blockDim.x) {
        float normalized = (input[offset + i] - mean) * invStd;
        float g = (gamma != nullptr) ? gamma[i] : 1.0f;
        float b = (beta != nullptr) ? beta[i] : 0.0f;
        output[offset + i] = normalized * g + b;
    }
}

__global__ void softmaxKernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int offset = row * cols;
    
    float maxVal = -1e30f;
    for (int i = tid; i < cols; i += blockDim.x) {
        maxVal = fmaxf(maxVal, data[offset + i]);
    }
    shared[tid] = maxVal;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    maxVal = shared[0];
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(data[offset + i] - maxVal);
        data[offset + i] = val;
        sum += val;
    }
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    sum = shared[0];
    
    for (int i = tid; i < cols; i += blockDim.x) {
        data[offset + i] /= sum;
    }
}

__global__ void addResidualKernel(float* output, const float* residual, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += residual[idx];
    }
}

__global__ void embedTokensKernel(const int* tokenIDs, const float* tokenEmb,
                                   const float* posEmb, float* output,
                                   int seqLen, int embedDim) {
    int pos = blockIdx.x;
    int i = threadIdx.x;
    
    if (pos < seqLen && i < embedDim) {
        int tokenID = tokenIDs[pos];
        output[pos * embedDim + i] = tokenEmb[tokenID * embedDim + i] + posEmb[pos * embedDim + i];
    }
}

__global__ void computeQKVKernel(const float* normInput, const float* weight, const float* bias,
                                  float* Q, float* K, float* V,
                                  int seqLen, int embedDim) {
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < seqLen && i < embedDim) {
        int offset = pos * embedDim;
        
        float sumQ = (bias != nullptr) ? bias[i] : 0.0f;
        float sumK = (bias != nullptr) ? bias[embedDim + i] : 0.0f;
        float sumV = (bias != nullptr) ? bias[2 * embedDim + i] : 0.0f;
        
        for (int j = 0; j < embedDim; j++) {
            float inp = normInput[offset + j];
            sumQ += inp * weight[i * embedDim + j];
            sumK += inp * weight[(embedDim + i) * embedDim + j];
            sumV += inp * weight[(2 * embedDim + i) * embedDim + j];
        }
        
        Q[offset + i] = sumQ;
        K[offset + i] = sumK;
        V[offset + i] = sumV;
    }
}

__global__ void attentionScoresKernel(const float* Q, const float* K, float* scores,
                                       int seqLen, int numHeads, int headDim, float scale) {
    int h = blockIdx.z;
    int pos = blockIdx.y;
    int srcPos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < numHeads && pos < seqLen && srcPos < seqLen) {
        if (srcPos > pos) {
            scores[h * seqLen * seqLen + pos * seqLen + srcPos] = -1e9f;
        } else {
            int headStart = h * headDim;
            float sum = 0.0f;
            for (int i = 0; i < headDim; i++) {
                sum += Q[pos * (numHeads * headDim) + headStart + i] *
                       K[srcPos * (numHeads * headDim) + headStart + i];
            }
            scores[h * seqLen * seqLen + pos * seqLen + srcPos] = sum / scale;
        }
    }
}

__global__ void attentionOutputKernel(const float* attnWeights, const float* V, float* output,
                                       int seqLen, int numHeads, int headDim) {
    int h = blockIdx.z;
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < numHeads && pos < seqLen && i < headDim) {
        int headStart = h * headDim;
        float sum = 0.0f;
        for (int srcPos = 0; srcPos < seqLen; srcPos++) {
            sum += attnWeights[h * seqLen * seqLen + pos * seqLen + srcPos] *
                   V[srcPos * (numHeads * headDim) + headStart + i];
        }
        output[pos * (numHeads * headDim) + headStart + i] = sum;
    }
}

__global__ void projectionKernel(const float* input, const float* weight, const float* bias,
                                  float* output, const float* residual,
                                  int seqLen, int embedDim) {
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < seqLen && i < embedDim) {
        float sum = (bias != nullptr) ? bias[i] : 0.0f;
        for (int j = 0; j < embedDim; j++) {
            sum += input[pos * embedDim + j] * weight[i * embedDim + j];
        }
        output[pos * embedDim + i] = residual[pos * embedDim + i] + sum;
    }
}

__global__ void ffnUpKernel(const float* input, const float* weight, const float* bias,
                             float* output, int seqLen, int embedDim, int ffnDim) {
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < seqLen && i < ffnDim) {
        float sum = (bias != nullptr) ? bias[i] : 0.0f;
        for (int j = 0; j < embedDim; j++) {
            sum += input[pos * embedDim + j] * weight[i * embedDim + j];
        }
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
        output[pos * ffnDim + i] = sum * cdf;
    }
}

__global__ void ffnDownKernel(const float* input, const float* weight, const float* bias,
                               float* output, const float* residual,
                               int seqLen, int ffnDim, int embedDim) {
    int pos = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < seqLen && i < embedDim) {
        float sum = (bias != nullptr) ? bias[i] : 0.0f;
        for (int j = 0; j < ffnDim; j++) {
            sum += input[pos * ffnDim + j] * weight[i * ffnDim + j];
        }
        output[pos * embedDim + i] = residual[pos * embedDim + i] + sum;
    }
}

__global__ void computeLogitsKernel(const float* hidden, const float* tokenEmb,
                                     float* logits, int embedDim, int vocabSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < vocabSize) {
        float sum = 0.0f;
        for (int j = 0; j < embedDim; j++) {
            sum += hidden[j] * tokenEmb[i * embedDim + j];
        }
        logits[i] = sum;
    }
}

// ==================== Tokenizer (CPU) ====================

class Tokenizer {
private:
    std::map<std::string, int> tokenToID;
    std::vector<std::string> idToToken;
    int vocabSize = 0;
    bool loaded = false;

public:
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Tokenizer file not found: " << filename << std::endl;
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json = buffer.str();
        file.close();

        size_t vocabPos = json.find("\"vocab\"");
        if (vocabPos == std::string::npos) {
            std::cerr << "No vocab found in tokenizer" << std::endl;
            return false;
        }

        size_t braceStart = json.find('{', vocabPos);
        if (braceStart == std::string::npos) return false;

        int braceCount = 1;
        size_t pos = braceStart + 1;
        
        while (braceCount > 0 && pos < json.size()) {
            if (json[pos] == '{') braceCount++;
            else if (json[pos] == '}') braceCount--;
            
            if (braceCount == 1 && json[pos] == '"') {
                size_t tokenStart = pos + 1;
                size_t tokenEnd = json.find('"', tokenStart);
                while (tokenEnd != std::string::npos && json[tokenEnd - 1] == '\\') {
                    tokenEnd = json.find('"', tokenEnd + 1);
                }
                if (tokenEnd == std::string::npos) break;
                
                std::string token = json.substr(tokenStart, tokenEnd - tokenStart);
                
                size_t escPos;
                while ((escPos = token.find("\\\"")) != std::string::npos)
                    token.replace(escPos, 2, "\"");
                while ((escPos = token.find("\\n")) != std::string::npos)
                    token.replace(escPos, 2, "\n");
                while ((escPos = token.find("\\t")) != std::string::npos)
                    token.replace(escPos, 2, "\t");
                while ((escPos = token.find("\\\\")) != std::string::npos)
                    token.replace(escPos, 2, "\\");
                
                size_t colonPos = json.find(':', tokenEnd);
                if (colonPos == std::string::npos) break;
                
                size_t numStart = colonPos + 1;
                while (numStart < json.size() && (json[numStart] == ' ' || json[numStart] == '\t'))
                    numStart++;
                
                size_t numEnd = numStart;
                while (numEnd < json.size() && (json[numEnd] >= '0' && json[numEnd] <= '9'))
                    numEnd++;
                
                if (numEnd > numStart) {
                    int id = std::stoi(json.substr(numStart, numEnd - numStart));
                    tokenToID[token] = id;
                    
                    while ((int)idToToken.size() <= id)
                        idToToken.push_back("");
                    idToToken[id] = token;
                    
                    if (id >= vocabSize) vocabSize = id + 1;
                }
                
                pos = numEnd;
            } else {
                pos++;
            }
        }

        loaded = vocabSize > 0;
        if (loaded)
            std::cout << "Tokenizer loaded: " << vocabSize << " tokens" << std::endl;
        
        return loaded;
    }

    int getTokenID(const std::string& token) const {
        auto it = tokenToID.find(token);
        return (it != tokenToID.end()) ? it->second : -1;
    }

    std::string getIDToken(int id) const {
        if (id >= 0 && id < (int)idToToken.size())
            return idToToken[id];
        return "";
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> result;
        if (!loaded) return result;

        std::vector<std::string> tokens;
        std::string currentWord;

        for (char ch : text) {
            if (ch == ' ') {
                if (!currentWord.empty())
                    tokens.push_back(currentWord);
                currentWord = "\xC4\xA0";
            } else {
                currentWord += ch;
            }
        }
        if (!currentWord.empty())
            tokens.push_back(currentWord);

        for (const auto& token : tokens) {
            int id = getTokenID(token);
            if (id >= 0) {
                result.push_back(id);
            } else {
                for (char c : token) {
                    std::string charStr(1, c);
                    id = getTokenID(charStr);
                    if (id >= 0) result.push_back(id);
                }
            }
        }
        return result;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            std::string token = getIDToken(id);
            size_t pos;
            while ((pos = token.find("\xC4\xA0")) != std::string::npos)
                token.replace(pos, 2, " ");
            while ((pos = token.find("\xC4\x8A")) != std::string::npos)
                token.replace(pos, 2, "\n");
            result += token;
        }
        return result;
    }

    bool isLoaded() const { return loaded; }
    int getVocabSize() const { return vocabSize; }
};

// ==================== GGUFTensor ====================

struct GGUFTensor {
    std::string name;
    std::vector<int64_t> shape;
    int numDims = 0;
    int dtype = 0;
    int64_t dataOffset = 0;
    bool dataLoaded = false;
    std::vector<float> data;
    float* d_data = nullptr;
};

// ==================== GGUFLoader ====================

class GGUFLoader {
private:
    std::ifstream stream;
    std::string filename;
    std::vector<GGUFTensor> tensors;
    std::map<std::string, size_t> tensorMap;
    int64_t tensorDataStart = 0;
    
    int embedDim = 768;
    int numLayers = 12;
    int numHeads = 12;
    int ffnDim = 3072;
    int vocabSize = 50257;
    int maxSeqLen = 1024;
    bool loaded = false;

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

    float float16ToFloat32(uint16_t h) {
        int sign = (h >> 15) & 1;
        int exponent = (h >> 10) & 0x1F;
        int mantissa = h & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0) return 0.0f;
            double e = -14;
            double m = mantissa / 1024.0;
            while (m < 1) { m *= 2; e -= 1; }
            float val = (float)(m * std::pow(2.0, e));
            return sign ? -val : val;
        } else if (exponent == 31) {
            if (mantissa != 0) return std::numeric_limits<float>::quiet_NaN();
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else {
            float val = (float)((1 + mantissa / 1024.0) * std::pow(2.0, exponent - 15));
            return sign ? -val : val;
        }
    }

    float bfloat16ToFloat32(uint16_t bf) {
        uint32_t f32bits = (uint32_t)bf << 16;
        float result;
        std::memcpy(&result, &f32bits, 4);
        return result;
    }

    std::string readString() {
        uint64_t len = readUInt64();
        if (len > 10000000) return "";
        std::string str(len, '\0');
        if (len > 0)
            stream.read(&str[0], len);
        return str;
    }

    void skipMetadataValue(int valueType) {
        switch (valueType) {
            case 0: case 1: stream.seekg(1, std::ios::cur); break;
            case 2: case 3: stream.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: stream.seekg(4, std::ios::cur); break;
            case 7: stream.seekg(1, std::ios::cur); break;
            case 8: {
                uint64_t strLen = readUInt64();
                stream.seekg(strLen, std::ios::cur);
                break;
            }
            case 9: {
                uint32_t arrType = readUInt32();
                uint64_t arrCount = readUInt64();
                for (uint64_t i = 0; i < std::min(arrCount, (uint64_t)1000000); i++)
                    skipMetadataValue(arrType);
                break;
            }
            case 10: case 11: case 12: stream.seekg(8, std::ios::cur); break;
        }
    }

    void parseHeader() {
        char magic[5] = {0};
        stream.read(magic, 4);
        if (std::string(magic) != GGUF_MAGIC)
            throw std::runtime_error("Invalid GGUF magic: " + std::string(magic));

        uint32_t version = readUInt32();
        uint64_t tensorCount = readUInt64();
        uint64_t metadataCount = readUInt64();

        std::cout << "GGUF Version: " << version << std::endl;
        std::cout << "Tensors: " << tensorCount << std::endl;
        std::cout << "Metadata entries: " << metadataCount << std::endl;

        for (uint64_t i = 0; i < metadataCount; i++) {
            std::string key = readString();
            uint32_t valueType = readUInt32();
            
            if ((key == "gpt2.embedding_length") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                embedDim = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "gpt2.block_count") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                numLayers = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "gpt2.attention.head_count") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                numHeads = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "gpt2.feed_forward_length") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                ffnDim = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else if ((key == "gpt2.context_length") && (valueType == 4 || valueType == 5 || valueType == 10)) {
                maxSeqLen = (valueType == 10) ? (int)readUInt64() : (int)readUInt32();
            } else {
                skipMetadataValue(valueType);
            }
        }

        std::cout << "Model config: embed_dim=" << embedDim << ", layers=" << numLayers
                  << ", heads=" << numHeads << ", ffn=" << ffnDim << std::endl;

        tensors.resize(tensorCount);
        for (uint64_t i = 0; i < tensorCount; i++) {
            tensors[i].name = readString();
            uint32_t numDims = readUInt32();
            tensors[i].numDims = numDims;
            tensors[i].shape.resize(numDims);
            for (uint32_t d = 0; d < numDims; d++)
                tensors[i].shape[d] = readUInt64();
            tensors[i].dtype = readUInt32();
            tensors[i].dataOffset = readUInt64();
            tensors[i].dataLoaded = false;
            tensors[i].d_data = nullptr;
            tensorMap[tensors[i].name] = i;
        }

        int64_t pos = stream.tellg();
        int64_t aligned = ((pos + 31) / 32) * 32;
        tensorDataStart = aligned;
    }

    bool loadTensorByIndex(size_t idx) {
        if (idx >= tensors.size()) return false;
        GGUFTensor& t = tensors[idx];
        if (t.dataLoaded) return true;

        int64_t numElements = 1;
        for (int64_t dim : t.shape)
            numElements *= dim;

        t.data.resize(numElements);
        stream.seekg(tensorDataStart + t.dataOffset);

        if (t.dtype == 0) {
            stream.read(reinterpret_cast<char*>(t.data.data()), numElements * 4);
        } else if (t.dtype == 1) {
            std::vector<uint16_t> f16data(numElements);
            stream.read(reinterpret_cast<char*>(f16data.data()), numElements * 2);
            for (int64_t j = 0; j < numElements; j++)
                t.data[j] = float16ToFloat32(f16data[j]);
        } else if (t.dtype == 30) {
            std::vector<uint16_t> bf16data(numElements);
            stream.read(reinterpret_cast<char*>(bf16data.data()), numElements * 2);
            for (int64_t j = 0; j < numElements; j++)
                t.data[j] = bfloat16ToFloat32(bf16data[j]);
        } else {
            std::cerr << "Unsupported dtype " << t.dtype << " for tensor " << t.name << std::endl;
            return false;
        }

        t.dataLoaded = true;
        return true;
    }

public:
    bool loadFromFile(const std::string& fname) {
        filename = fname;
        stream.open(fname, std::ios::binary);
        if (!stream.is_open()) {
            std::cerr << "Failed to open GGUF file: " << fname << std::endl;
            return false;
        }

        try {
            parseHeader();
            loaded = true;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing GGUF: " << e.what() << std::endl;
            return false;
        }

        return true;
    }

    float* getTensorGPU(const std::vector<std::string>& names) {
        for (const auto& name : names) {
            auto it = tensorMap.find(name);
            if (it != tensorMap.end()) {
                GGUFTensor& t = tensors[it->second];
                if (t.d_data != nullptr) return t.d_data;
                
                if (!loadTensorByIndex(it->second)) return nullptr;
                
                CUDA_CHECK(cudaMalloc(&t.d_data, t.data.size() * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(t.d_data, t.data.data(), t.data.size() * sizeof(float), cudaMemcpyHostToDevice));
                
                t.data.clear();
                t.data.shrink_to_fit();
                
                return t.d_data;
            }
        }
        return nullptr;
    }

    std::vector<float> getTensor(const std::vector<std::string>& names) {
        for (const auto& name : names) {
            auto it = tensorMap.find(name);
            if (it != tensorMap.end()) {
                if (loadTensorByIndex(it->second))
                    return tensors[it->second].data;
            }
        }
        return {};
    }

    bool hasTensor(const std::string& name) const {
        return tensorMap.find(name) != tensorMap.end();
    }

    void printAllTensorNames() {
        std::cout << "\n=== All Tensor Names ===" << std::endl;
        for (const auto& t : tensors) {
            std::cout << t.name << " [";
            for (size_t i = 0; i < t.shape.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << t.shape[i];
            }
            std::cout << "] dtype=" << t.dtype << std::endl;
        }
    }

    void freeGPUMemory() {
        for (auto& t : tensors) {
            if (t.d_data != nullptr) {
                cudaFree(t.d_data);
                t.d_data = nullptr;
            }
        }
    }

    int getEmbedDim() const { return embedDim; }
    int getNumLayers() const { return numLayers; }
    int getNumHeads() const { return numHeads; }
    int getFFNDim() const { return ffnDim; }
    int getVocabSize() const { return vocabSize; }
    int getMaxSeqLen() const { return maxSeqLen; }
    bool isLoaded() const { return loaded; }
};

// ==================== TransformerModel ====================

class TransformerModel {
private:
    GGUFLoader loader;
    Tokenizer tokenizer;
    int embedDim = 0;
    int numHeads = 0;
    int headDim = 0;
    int numLayers = 0;
    int ffnDim = 0;
    int vocabSize = 0;

    std::mt19937 rng;

    float* d_hidden = nullptr;
    float* d_hidden2 = nullptr;
    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_attnOut = nullptr;
    float* d_attnScores = nullptr;
    float* d_ffnHidden = nullptr;
    float* d_logits = nullptr;
    int* d_tokenIDs = nullptr;
    
    int allocatedSeqLen = 0;

    void allocateBuffers(int seqLen) {
        if (seqLen <= allocatedSeqLen) return;
        
        freeBuffers();
        
        CUDA_CHECK(cudaMalloc(&d_hidden, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden2, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attnOut, seqLen * embedDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attnScores, numHeads * seqLen * seqLen * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ffnHidden, seqLen * ffnDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_logits, vocabSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tokenIDs, seqLen * sizeof(int)));
        
        allocatedSeqLen = seqLen;
    }

    void freeBuffers() {
        if (d_hidden) cudaFree(d_hidden);
        if (d_hidden2) cudaFree(d_hidden2);
        if (d_Q) cudaFree(d_Q);
        if (d_K) cudaFree(d_K);
        if (d_V) cudaFree(d_V);
        if (d_attnOut) cudaFree(d_attnOut);
        if (d_attnScores) cudaFree(d_attnScores);
        if (d_ffnHidden) cudaFree(d_ffnHidden);
        if (d_logits) cudaFree(d_logits);
        if (d_tokenIDs) cudaFree(d_tokenIDs);
        d_hidden = d_hidden2 = d_Q = d_K = d_V = d_attnOut = d_attnScores = d_ffnHidden = d_logits = nullptr;
        d_tokenIDs = nullptr;
        allocatedSeqLen = 0;
    }

    void embedTokens(const std::vector<int>& tokenIDs, int seqLen) {
        float* d_tokenEmb = loader.getTensorGPU({"token_embd.weight", "wte.weight"});
        float* d_posEmb = loader.getTensorGPU({"position_embd.weight", "wpe.weight"});
        
        CUDA_CHECK(cudaMemcpy(d_tokenIDs, tokenIDs.data(), seqLen * sizeof(int), cudaMemcpyHostToDevice));
        
        dim3 block(embedDim);
        dim3 grid(seqLen);
        embedTokensKernel<<<grid, block>>>(d_tokenIDs, d_tokenEmb, d_posEmb, d_hidden, seqLen, embedDim);
        CUDA_CHECK(cudaGetLastError());
    }

    void attentionBlock(int seqLen, int layerIdx) {
        std::string prefix = "blk." + std::to_string(layerIdx) + ".";
        
        float* d_ln1g = loader.getTensorGPU({prefix + "attn_norm.weight"});
        float* d_ln1b = loader.getTensorGPU({prefix + "attn_norm.bias"});
        float* d_qkvW = loader.getTensorGPU({prefix + "attn_qkv.weight"});
        float* d_qkvB = loader.getTensorGPU({prefix + "attn_qkv.bias"});
        float* d_projW = loader.getTensorGPU({prefix + "attn_output.weight"});
        float* d_projB = loader.getTensorGPU({prefix + "attn_output.bias"});
        
        int sharedMem = BLOCK_SIZE * sizeof(float);
        layerNormKernel<<<seqLen, BLOCK_SIZE, sharedMem>>>(d_hidden, d_hidden2, d_ln1g, d_ln1b, seqLen, embedDim);
        
        dim3 qkvBlock(BLOCK_SIZE);
        dim3 qkvGrid((embedDim + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen);
        computeQKVKernel<<<qkvGrid, qkvBlock>>>(d_hidden2, d_qkvW, d_qkvB, d_Q, d_K, d_V, seqLen, embedDim);
        
        float scale = sqrtf((float)headDim);
        dim3 scoreBlock(BLOCK_SIZE);
        dim3 scoreGrid((seqLen + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen, numHeads);
        attentionScoresKernel<<<scoreGrid, scoreBlock>>>(d_Q, d_K, d_attnScores, seqLen, numHeads, headDim, scale);
        
        for (int h = 0; h < numHeads; h++) {
            softmaxKernel<<<seqLen, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
                d_attnScores + h * seqLen * seqLen, seqLen, seqLen);
        }
        
        dim3 outBlock(BLOCK_SIZE);
        dim3 outGrid((headDim + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen, numHeads);
        attentionOutputKernel<<<outGrid, outBlock>>>(d_attnScores, d_V, d_attnOut, seqLen, numHeads, headDim);
        
        dim3 projBlock(BLOCK_SIZE);
        dim3 projGrid((embedDim + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen);
        projectionKernel<<<projGrid, projBlock>>>(d_attnOut, d_projW, d_projB, d_hidden2, d_hidden, seqLen, embedDim);
        
        std::swap(d_hidden, d_hidden2);
        
        CUDA_CHECK(cudaGetLastError());
    }

    void ffnBlock(int seqLen, int layerIdx) {
        std::string prefix = "blk." + std::to_string(layerIdx) + ".";
        
        float* d_ln2g = loader.getTensorGPU({prefix + "ffn_norm.weight"});
        float* d_ln2b = loader.getTensorGPU({prefix + "ffn_norm.bias"});
        float* d_upW = loader.getTensorGPU({prefix + "ffn_up.weight"});
        float* d_upB = loader.getTensorGPU({prefix + "ffn_up.bias"});
        float* d_downW = loader.getTensorGPU({prefix + "ffn_down.weight"});
        float* d_downB = loader.getTensorGPU({prefix + "ffn_down.bias"});
        
        int sharedMem = BLOCK_SIZE * sizeof(float);
        layerNormKernel<<<seqLen, BLOCK_SIZE, sharedMem>>>(d_hidden, d_hidden2, d_ln2g, d_ln2b, seqLen, embedDim);
        
        dim3 upBlock(BLOCK_SIZE);
        dim3 upGrid((ffnDim + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen);
        ffnUpKernel<<<upGrid, upBlock>>>(d_hidden2, d_upW, d_upB, d_ffnHidden, seqLen, embedDim, ffnDim);
        
        dim3 downBlock(BLOCK_SIZE);
        dim3 downGrid((embedDim + BLOCK_SIZE - 1) / BLOCK_SIZE, seqLen);
        ffnDownKernel<<<downGrid, downBlock>>>(d_ffnHidden, d_downW, d_downB, d_hidden2, d_hidden, seqLen, ffnDim, embedDim);
        
        std::swap(d_hidden, d_hidden2);
        
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<float> computeLogits(int seqLen) {
        float* d_lnG = loader.getTensorGPU({"output_norm.weight", "ln_f.weight"});
        float* d_lnB = loader.getTensorGPU({"output_norm.bias", "ln_f.bias"});
        float* d_tokenEmb = loader.getTensorGPU({"token_embd.weight", "wte.weight"});
        
        float* d_lastHidden;
        CUDA_CHECK(cudaMalloc(&d_lastHidden, embedDim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_lastHidden, d_hidden + (seqLen - 1) * embedDim, embedDim * sizeof(float), cudaMemcpyDeviceToDevice));
        
        float* d_normed;
        CUDA_CHECK(cudaMalloc(&d_normed, embedDim * sizeof(float)));
        
        int sharedMem = BLOCK_SIZE * sizeof(float);
        layerNormKernel<<<1, BLOCK_SIZE, sharedMem>>>(d_lastHidden, d_normed, d_lnG, d_lnB, 1, embedDim);
        
        dim3 block(BLOCK_SIZE);
        dim3 grid((vocabSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
        computeLogitsKernel<<<grid, block>>>(d_normed, d_tokenEmb, d_logits, embedDim, vocabSize);
        
        CUDA_CHECK(cudaGetLastError());
        
        std::vector<float> logits(vocabSize);
        CUDA_CHECK(cudaMemcpy(logits.data(), d_logits, vocabSize * sizeof(float), cudaMemcpyDeviceToHost));
        
        cudaFree(d_lastHidden);
        cudaFree(d_normed);
        
        return logits;
    }

    std::vector<float> forward(const std::vector<int>& tokenIDs) {
        int seqLen = tokenIDs.size();
        allocateBuffers(seqLen);
        
        embedTokens(tokenIDs, seqLen);
        
        for (int l = 0; l < numLayers; l++) {
            std::cout << "\rLayer " << (l + 1) << "/" << numLayers << "..." << std::flush;
            attentionBlock(seqLen, l);
            ffnBlock(seqLen, l);
        }
        std::cout << " done" << std::endl;
        
        return computeLogits(seqLen);
    }

public:
    TransformerModel() : rng(std::random_device{}()) {}
    
    ~TransformerModel() {
        freeBuffers();
        loader.freeGPUMemory();
    }

    bool loadModel(const std::string& ggufPath) {
        if (!loader.loadFromFile(ggufPath))
            return false;

        embedDim = loader.getEmbedDim();
        numLayers = loader.getNumLayers();
        numHeads = loader.getNumHeads();
        ffnDim = loader.getFFNDim();
        vocabSize = loader.getVocabSize();
        headDim = embedDim / numHeads;

        return true;
    }

    bool loadTokenizer(const std::string& tokenizerPath) {
        return tokenizer.loadFromFile(tokenizerPath);
    }

    std::string generate(const std::string& prompt, int maxTokens, double temperature = 1.0) {
        if (!loader.isLoaded()) {
            std::cerr << "Error: Model not loaded" << std::endl;
            return "";
        }

        if (!tokenizer.isLoaded()) {
            std::cerr << "Error: Tokenizer not loaded" << std::endl;
            return "";
        }

        std::cout << "Encoding prompt..." << std::endl;
        auto tokenIDs = tokenizer.encode(prompt);
        std::cout << "Input tokens: " << tokenIDs.size() << std::endl;
        std::cout << "Temperature: " << std::fixed << std::setprecision(2) << temperature << std::endl;

        if (tokenIDs.empty()) {
            std::cerr << "Error: Could not tokenize input" << std::endl;
            return "";
        }

        std::cout << "Token IDs: ";
        for (size_t i = 0; i < std::min(tokenIDs.size(), (size_t)10); i++)
            std::cout << tokenIDs[i] << " ";
        if (tokenIDs.size() > 10) std::cout << "...";
        std::cout << std::endl;

        auto startTime = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < maxTokens; i++) {
            std::cout << std::endl << "=== Generating token " << (i + 1) << "/" << maxTokens << " ===" << std::endl;

            auto logits = forward(tokenIDs);

            if (logits.empty()) {
                std::cerr << "ERROR: Forward pass failed" << std::endl;
                break;
            }

            int bestID = 0;
            float bestLogit = logits[0];
            for (size_t j = 1; j < logits.size(); j++) {
                if (logits[j] > bestLogit) {
                    bestLogit = logits[j];
                    bestID = j;
                }
            }

            int selectedID;
            if (temperature <= 0.01) {
                selectedID = bestID;
            } else {
                for (float& l : logits) l /= temperature;
                
                float maxVal = *std::max_element(logits.begin(), logits.end());
                float sum = 0;
                for (float& l : logits) {
                    l = std::exp(l - maxVal);
                    sum += l;
                }
                for (float& l : logits) l /= sum;
                
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double r = dist(rng);
                double cumulativeProb = 0.0;
                selectedID = 0;
                for (size_t j = 0; j < logits.size(); j++) {
                    cumulativeProb += logits[j];
                    if (r <= cumulativeProb) {
                        selectedID = j;
                        break;
                    }
                }
            }

            std::cout << "Generated token: " << selectedID << " = \"" << tokenizer.getIDToken(selectedID)
                      << "\" (best was: " << bestID << " logit: " << std::fixed << std::setprecision(4) << bestLogit << ")" << std::endl;

            tokenIDs.push_back(selectedID);

            if (selectedID == 50256) {
                std::cout << "[EOS token reached]" << std::endl;
                break;
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        double elapsedSecs = std::chrono::duration<double>(endTime - startTime).count();
        std::cout << std::endl << "Generation complete in " << std::fixed << std::setprecision(1) << elapsedSecs << " seconds" << std::endl;

        return tokenizer.decode(tokenIDs);
    }

    bool isModelLoaded() const { return loader.isLoaded(); }
    bool isTokenizerLoaded() const { return tokenizer.isLoaded(); }
    void printTensorNames() { loader.printAllTensorNames(); }
};

// ==================== Argument Parser ====================

struct Arguments {
    std::string ggufPath;
    std::string tokenizerPath;
    std::string prompt = "Hello";
    int maxTokens = 5;
    double temperature = 1.0;
    bool listTensors = false;
    bool help = false;
};

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <model.gguf> <tokenizer.json> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  <model.gguf>       Path to the GGUF model file" << std::endl;
    std::cout << "  <tokenizer.json>   Path to the tokenizer JSON file" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -p, --prompt TEXT      Input prompt (default: \"Hello\")" << std::endl;
    std::cout << "  -n, --max-tokens N     Maximum tokens to generate (default: 5)" << std::endl;
    std::cout << "  -t, --temperature T    Sampling temperature (default: 1.0)" << std::endl;
    std::cout << "  --list-tensors         List all tensor names in the model" << std::endl;
    std::cout << "  -h, --help             Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << progName << " gpt2-f32.gguf tokenizer.json -p \"Hello world\" -n 10 -t 0.8" << std::endl;
    std::cout << "  " << progName << " gpt2-f32.gguf tokenizer.json --list-tensors" << std::endl;
}

Arguments parseArguments(int argc, char* argv[]) {
    Arguments args;

    if (argc < 2) {
        args.help = true;
        return args;
    }

    int positionalCount = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            args.help = true;
            return args;
        } else if (arg == "--list-tensors") {
            args.listTensors = true;
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if ((arg == "-n" || arg == "--max-tokens") && i + 1 < argc) {
            args.maxTokens = std::stoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--temperature") && i + 1 < argc) {
            args.temperature = std::stod(argv[++i]);
        } else if (arg[0] != '-') {
            if (positionalCount == 0) {
                args.ggufPath = arg;
                positionalCount++;
            } else if (positionalCount == 1) {
                args.tokenizerPath = arg;
                positionalCount++;
            }
        }
    }

    if (args.ggufPath.empty() || args.tokenizerPath.empty()) {
        args.help = true;
    }

    return args;
}

// ==================== Main ====================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  GPT-2 CLI - CUDA Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
    std::cout << std::endl;

    Arguments args = parseArguments(argc, argv);

    if (args.help) {
        printUsage(argv[0]);
        return 1;
    }

    TransformerModel model;

    std::cout << "Loading model..." << std::endl;
    if (!model.loadModel(args.ggufPath)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    if (args.listTensors) {
        model.printTensorNames();
        return 0;
    }

    std::cout << std::endl << "Loading tokenizer..." << std::endl;
    if (!model.loadTokenizer(args.tokenizerPath)) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Prompt: \"" << args.prompt << "\"" << std::endl;
    std::cout << "Max tokens: " << args.maxTokens << std::endl;
    std::cout << "Temperature: " << std::fixed << std::setprecision(2) << args.temperature << std::endl;
    std::cout << "========================================" << std::endl;

    std::string generatedText = model.generate(args.prompt, args.maxTokens, args.temperature);

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GENERATED TEXT:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << generatedText << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
