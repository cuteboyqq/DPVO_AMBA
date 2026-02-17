#include "update_onnx.hpp"
#include "onnx_env.hpp"
#include "dla_config.hpp"
#include "logger.hpp"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>

// ONNX Runtime includes
#ifdef USE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

DPVOUpdateONNX::DPVOUpdateONNX(Config_S* config)
{
    auto logger = spdlog::get("dpvo");
    if (!logger) {
        logger = spdlog::stdout_color_mt("dpvo");
        logger->set_pattern("[%n] [%^%l%$] %v");
    }

#ifdef USE_ONNX_RUNTIME
    // Use updateModelPath if available, otherwise fallback to modelPath
    m_modelPath = !config->updateModelPath.empty() ? config->updateModelPath : config->modelPath;
    
    if (m_modelPath.empty()) {
        logger->error("DPVOUpdateONNX: No model path provided");
        return;
    }
    
    // Check if file exists
    std::ifstream file(m_modelPath);
    if (!file.good()) {
        logger->error("DPVOUpdateONNX: Model file does not exist: {}", m_modelPath);
        return;
    }
    
    // Read maxEdges from config (default 360)
    if (config != nullptr && config->maxEdges > 0) {
        m_maxEdge = config->maxEdges;
    }
    
    // Buffer sizes (same as DPVOUpdate)
    m_netOutBufferSize = 1 * 384 * m_maxEdge * 1;  // [1, 384, m_maxEdge, 1]
    m_dOutBufferSize = 1 * 2 * m_maxEdge * 1;      // [1, 2, m_maxEdge, 1]
    m_wOutBufferSize = 1 * 2 * m_maxEdge * 1;      // [1, 2, m_maxEdge, 1]
    
    _initModel();
#else
    logger->error("DPVOUpdateONNX: ONNX Runtime not enabled. Compile with -DUSE_ONNX_RUNTIME");
#endif
}

DPVOUpdateONNX::~DPVOUpdateONNX()
{
#ifdef USE_ONNX_RUNTIME
    if (m_session) {
        Ort::Session* session = static_cast<Ort::Session*>(m_session);
        delete session;
        m_session = nullptr;
    }
#endif
}

void DPVOUpdateONNX::_initModel()
{
    auto logger = spdlog::get("dpvo");
    
#ifdef USE_ONNX_RUNTIME
    try {
        // Use singleton Ort::Env (required by ONNX Runtime)
        Ort::Env& env = OnnxEnvSingleton::getInstance();
        Ort::SessionOptions session_options;
        
        // Create session
        Ort::Session* session = new Ort::Session(env, m_modelPath.c_str(), session_options);
        m_session = session;
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info (6 inputs: net, inp, corr, ii, jj, kk)
        size_t num_input_nodes = session->GetInputCount();
        if (num_input_nodes != 6) {
            logger->error("DPVOUpdateONNX: Expected 6 inputs, got {}", num_input_nodes);
            return;
        }
        
        m_inputNames.clear();
        m_inputShapes.clear();
        m_inputTypes.clear();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session->GetInputNameAllocated(i, allocator);
            m_inputNames.push_back(input_name.get());
            
            Ort::TypeInfo input_type_info = session->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = input_tensor_info.GetShape();
            ONNXTensorElementDataType element_type = input_tensor_info.GetElementType();
            m_inputShapes.push_back(shape);
            m_inputTypes.push_back(element_type);
            
            if (logger) {
                const char* type_name = (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ? "float" :
                                       (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) ? "int32" :
                                       (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ? "int64" : "unknown";
                logger->info("DPVOUpdateONNX: Input[{}] '{}' - shape={}, type={}", 
                             i, input_name.get(), 
                             (shape.size() > 0 ? std::to_string(shape[0]) : "?"), type_name);
            }
        }
        
        // Output info (3 outputs: net_out, d_out, w_out)
        size_t num_output_nodes = session->GetOutputCount();
        if (num_output_nodes != 3) {
            logger->error("DPVOUpdateONNX: Expected 3 outputs, got {}", num_output_nodes);
            return;
        }
        
        m_outputNames.clear();
        m_outputShapes.clear();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            m_outputNames.push_back(output_name.get());
            
            Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = output_tensor_info.GetShape();
            m_outputShapes.push_back(shape);
            
            // Log output shape
            std::string shape_str = "[";
            for (size_t j = 0; j < shape.size(); j++) {
                if (j > 0) shape_str += ", ";
                shape_str += std::to_string(shape[j]);
            }
            shape_str += "]";
            logger->info("DPVOUpdateONNX: Output[{}] '{}' - expected shape: {}", 
                         i, output_name.get(), shape_str);
        }
        
        logger->info("DPVOUpdateONNX: Model loaded successfully. Inputs: {}, Outputs: {}", 
                     num_input_nodes, num_output_nodes);
    } catch (const std::exception& e) {
        logger->error("DPVOUpdateONNX: Failed to initialize model: {}", e.what());
        m_session = nullptr;
    }
#else
    if (logger) {
        logger->error("DPVOUpdateONNX: ONNX Runtime not enabled");
    }
#endif
}

bool DPVOUpdateONNX::runInference(float* netData, float* inpData, float* corrData, 
                                   float* iiData, float* jjData, float* kkData, 
                                   int frameIdx, DPVOUpdate_Prediction& pred)
{
    auto logger = spdlog::get("dpvo");
    
#ifdef USE_ONNX_RUNTIME
    if (!m_session) {
        if (logger) logger->error("DPVOUpdateONNX: Session not initialized");
        return false;
    }
    
    try {
        Ort::Session* session = static_cast<Ort::Session*>(m_session);
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Prepare input data
        std::vector<std::vector<float>> input_data;
        if (!_loadInput(netData, inpData, corrData, iiData, jjData, kkData, input_data)) {
            if (logger) logger->error("DPVOUpdateONNX: Failed to load input");
            return false;
        }
        
        // Create input tensors
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> input_tensors;
        
        // Prepare int32 data for index inputs (ii, jj, kk)
        std::vector<int32_t> ii_int32(m_maxEdge);
        std::vector<int32_t> jj_int32(m_maxEdge);
        std::vector<int32_t> kk_int32(m_maxEdge);
        for (int i = 0; i < m_maxEdge; i++) {
            ii_int32[i] = static_cast<int32_t>(iiData[i]);
            jj_int32[i] = static_cast<int32_t>(jjData[i]);
            kk_int32[i] = static_cast<int32_t>(kkData[i]);
        }
        
        for (size_t i = 0; i < m_inputNames.size(); i++) {
            std::vector<int64_t> shape = m_inputShapes[i];
            size_t total_size = 1;
            for (int64_t dim : shape) {
                if (dim > 0) total_size *= dim;
            }
            
            ONNXTensorElementDataType element_type = m_inputTypes[i];
            
            if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                // Inputs 3, 4, 5 are indices (ii, jj, kk) - use int32
                int32_t* int_data = nullptr;
                if (i == 3) int_data = ii_int32.data();
                else if (i == 4) int_data = jj_int32.data();
                else if (i == 5) int_data = kk_int32.data();
                else {
                    if (logger) logger->error("DPVOUpdateONNX: Unexpected int32 input index {}", i);
                    return false;
                }
                
                Ort::Value tensor = Ort::Value::CreateTensor<int32_t>(
                    memory_info, int_data, total_size,
                    shape.data(), shape.size());
                input_tensors.push_back(std::move(tensor));
                
                if (logger) {
                    logger->info("DPVOUpdateONNX: Created int32 tensor for input[{}] '{}', size={}", 
                                 i, m_inputNames[i], total_size);
                }
            } else {
                // Inputs 0, 1, 2 are float (net, inp, corr)
                Ort::Value tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_data[i].data(), total_size,
                    shape.data(), shape.size());
                input_tensors.push_back(std::move(tensor));
                
                if (logger && i < 3) {
                    logger->info("DPVOUpdateONNX: Created float tensor for input[{}] '{}', size={}", 
                                 i, m_inputNames[i], total_size);
                }
            }
        }
        
        // Prepare input/output names
        std::vector<const char*> input_names;
        for (const auto& name : m_inputNames) {
            input_names.push_back(name.c_str());
        }
        
        std::vector<const char*> output_names;
        for (const auto& name : m_outputNames) {
            output_names.push_back(name.c_str());
        }
        
        // Run inference
        auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                          input_names.data(), input_tensors.data(), input_tensors.size(),
                                          output_names.data(), output_names.size());
        
        if (output_tensors.size() != 3) {
            if (logger) logger->error("DPVOUpdateONNX: Expected 3 output tensors, got {}", output_tensors.size());
            return false;
        }
        
        // Verify output shapes match expected
        for (size_t i = 0; i < output_tensors.size(); i++) {
            auto shape_info = output_tensors[i].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> actual_shape = shape_info.GetShape();
            std::vector<int64_t> expected_shape = m_outputShapes[i];
            
            if (logger && frameIdx % 100 == 0) {  // Log every 100 frames to reduce noise
                std::string shape_str = "[";
                for (size_t j = 0; j < actual_shape.size(); j++) {
                    if (j > 0) shape_str += ", ";
                    shape_str += std::to_string(actual_shape[j]);
                }
                shape_str += "]";
                logger->info("DPVOUpdateONNX: Output[{}] '{}' actual shape: {}", 
                            i, m_outputNames[i].c_str(), shape_str);
            }
            
            // Verify shape matches expected
            if (actual_shape.size() != expected_shape.size()) {
                if (logger) logger->error("DPVOUpdateONNX: Output[{}] shape dimension mismatch: expected {} dims, got {} dims",
                                         i, expected_shape.size(), actual_shape.size());
                return false;
            }
            for (size_t j = 0; j < actual_shape.size(); j++) {
                if (actual_shape[j] != expected_shape[j] && expected_shape[j] != -1) {  // -1 means dynamic dimension
                    if (logger) logger->warn("DPVOUpdateONNX: Output[{}] shape[{}] mismatch: expected {}, got {}",
                                            i, j, expected_shape[j], actual_shape[j]);
                }
            }
        }
        
        // Allocate output buffers
        pred.netOutBuff = new float[m_netOutBufferSize];
        pred.dOutBuff = new float[m_dOutBufferSize];
        pred.wOutBuff = new float[m_wOutBufferSize];
        
        // Extract outputs
        // Output 0: net_out [1, 384, 360, 1]
        float* net_out_data = output_tensors[0].GetTensorMutableData<float>();
        size_t net_out_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (net_out_size != m_netOutBufferSize) {
            if (logger) logger->error("DPVOUpdateONNX: net_out size mismatch: expected {}, got {}", 
                                     m_netOutBufferSize, net_out_size);
            return false;
        }
        std::memcpy(pred.netOutBuff, net_out_data, m_netOutBufferSize * sizeof(float));
        
        // Output 1: d_out [1, 2, 360, 1]
        float* d_out_data = output_tensors[1].GetTensorMutableData<float>();
        size_t d_out_size = output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount();
        if (d_out_size != m_dOutBufferSize) {
            if (logger) logger->error("DPVOUpdateONNX: d_out size mismatch: expected {}, got {}", 
                                     m_dOutBufferSize, d_out_size);
            return false;
        }
        std::memcpy(pred.dOutBuff, d_out_data, m_dOutBufferSize * sizeof(float));
        
        // Output 2: w_out [1, 2, 360, 1]
        float* w_out_data = output_tensors[2].GetTensorMutableData<float>();
        size_t w_out_size = output_tensors[2].GetTensorTypeAndShapeInfo().GetElementCount();
        if (w_out_size != m_wOutBufferSize) {
            if (logger) logger->error("DPVOUpdateONNX: w_out size mismatch: expected {}, got {}", 
                                     m_wOutBufferSize, w_out_size);
            return false;
        }
        std::memcpy(pred.wOutBuff, w_out_data, m_wOutBufferSize * sizeof(float));
        
        pred.isProcessed = true;
        
        if (logger) {
            logger->info("\033[33mDPVOUpdateONNX: Inference successful. Frame: {}\033[0m", frameIdx);
        }
        
        return true;
    } catch (const std::exception& e) {
        if (logger) logger->error("DPVOUpdateONNX: Inference failed: {}", e.what());
        return false;
    }
#else
    if (logger) logger->error("DPVOUpdateONNX: ONNX Runtime not enabled");
    return false;
#endif
}

bool DPVOUpdateONNX::_loadInput(float* netData, float* inpData, float* corrData, 
                                 float* iiData, float* jjData, float* kkData,
                                 std::vector<std::vector<float>>& input_data)
{
#ifdef USE_ONNX_RUNTIME
    // Prepare 6 input tensors
    input_data.resize(6);
    
    // Input 0: net [1, 384, 360, 1]
    input_data[0].assign(netData, netData + m_maxEdge * 384);
    
    // Input 1: inp [1, 384, 360, 1]
    input_data[1].assign(inpData, inpData + m_maxEdge * 384);
    
    // Input 2: corr [1, 882, 360, 1] (882 = 2*49*3*3 for P=3)
    input_data[2].assign(corrData, corrData + m_maxEdge * 882);
    
    // Input 3: ii [1, 360, 1] (as float)
    input_data[3].resize(m_maxEdge);
    for (int i = 0; i < m_maxEdge; i++) {
        input_data[3][i] = static_cast<float>(iiData[i]);
    }
    
    // Input 4: jj [1, 360, 1] (as float)
    input_data[4].resize(m_maxEdge);
    for (int i = 0; i < m_maxEdge; i++) {
        input_data[4][i] = static_cast<float>(jjData[i]);
    }
    
    // Input 5: kk [1, 360, 1] (as float)
    input_data[5].resize(m_maxEdge);
    for (int i = 0; i < m_maxEdge; i++) {
        input_data[5][i] = static_cast<float>(kkData[i]);
    }
    
    return true;
#else
    return false;
#endif
}

// Reshape inputs (same interface as DPVOUpdate)
int DPVOUpdateONNX::reshapeInput(
    int num_active,
    float (*m_net)[384],
    const float* ctx,
    const std::vector<float>& corr,
    const int* m_ii,
    const int* m_jj,
    const int* m_kk,
    int D,
    int P,
    std::vector<float>& net_input,
    std::vector<float>& inp_input,
    std::vector<float>& corr_input,
    std::vector<float>& ii_input,
    std::vector<float>& jj_input,
    std::vector<float>& kk_input,
    const int MODEL_EDGE_COUNT,
    const int CORR_DIM
)
{
    // Resize buffers to match model input size [1, DIM, H, 1]
    // Total size: 1 * DIM * MODEL_EDGE_COUNT * 1 = DIM * MODEL_EDGE_COUNT
    net_input.resize(1 * 384 * MODEL_EDGE_COUNT * 1);
    inp_input.resize(1 * 384 * MODEL_EDGE_COUNT * 1);
    corr_input.resize(1 * CORR_DIM * MODEL_EDGE_COUNT * 1);
    ii_input.resize(1 * 1 * MODEL_EDGE_COUNT * 1);
    jj_input.resize(1 * 1 * MODEL_EDGE_COUNT * 1);
    kk_input.resize(1 * 1 * MODEL_EDGE_COUNT * 1);
    
    // Zero out buffers
    std::fill(net_input.begin(), net_input.end(), 0.0f);
    std::fill(inp_input.begin(), inp_input.end(), 0.0f);
    std::fill(corr_input.begin(), corr_input.end(), 0.0f);
    std::fill(ii_input.begin(), ii_input.end(), 0.0f);
    std::fill(jj_input.begin(), jj_input.end(), 0.0f);
    std::fill(kk_input.begin(), kk_input.end(), 0.0f);
    
    // Reshape from [H, DIM] to [1, DIM, H, 1] to match Python's permute(0,2,1).unsqueeze(-1)
    // Python: net.permute(0, 2, 1).unsqueeze(-1) → [1, H, DIM] → [1, DIM, H, 1]
    // Index formula for [1, DIM, H, 1]: idx = n * (DIM * H * 1) + c * (H * 1) + h * 1 + w
    // For [1, 384, MODEL_EDGE_COUNT, 1]: idx = 0 + c * MODEL_EDGE_COUNT + h + 0
    for (int e = 0; e < num_active && e < MODEL_EDGE_COUNT; e++) {
        // Reshape net: [H, DIM] → [1, DIM, H, 1]
        for (int d = 0; d < 384; d++) {
            // [1, 384, MODEL_EDGE_COUNT, 1] layout: channel d, edge e
            int idx = 0 * (384 * MODEL_EDGE_COUNT * 1) + 
                      d * (MODEL_EDGE_COUNT * 1) + 
                      e * 1 + 
                      0;
            net_input[idx] = m_net[e][d];
            inp_input[idx] = ctx[e * 384 + d];
        }
        
        // Reshape correlation: [H, CORR_DIM] → [1, CORR_DIM, H, 1]
        for (int c = 0; c < CORR_DIM; c++) {
            int idx = 0 * (CORR_DIM * MODEL_EDGE_COUNT * 1) + 
                      c * (MODEL_EDGE_COUNT * 1) + 
                      e * 1 + 
                      0;
            corr_input[idx] = corr[e * CORR_DIM + c];
        }
        
        // Reshape indices: [H] → [1, 1, H, 1]
        int idx_ii = 0 * (1 * MODEL_EDGE_COUNT * 1) + 
                     0 * (MODEL_EDGE_COUNT * 1) + 
                     e * 1 + 
                     0;
        int idx_jj = idx_ii;  // Same layout
        int idx_kk = idx_ii;  // Same layout
        
        ii_input[idx_ii] = static_cast<float>(m_ii[e]);
        jj_input[idx_jj] = static_cast<float>(m_jj[e]);
        kk_input[idx_kk] = static_cast<float>(m_kk[e]);
    }
    
    return num_active;
}

