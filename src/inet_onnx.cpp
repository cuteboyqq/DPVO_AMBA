#include "inet_onnx.hpp"
#include "onnx_env.hpp"
#include "logger.hpp"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>

// ONNX Runtime includes
#ifdef USE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

// Ambarella CV28
#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

INetInferenceONNX::INetInferenceONNX(Config_S* config)
{
    auto logger = spdlog::get("inet");
    if (!logger) {
        logger = spdlog::stdout_color_mt("inet");
        logger->set_pattern("[%n] [%^%l%$] %v");
    }

#ifdef USE_ONNX_RUNTIME
    // Use inetModelPath if available, otherwise fallback to modelPath
    m_modelPath = !config->inetModelPath.empty() ? config->inetModelPath : config->modelPath;
    
    if (m_modelPath.empty()) {
        logger->error("INetInferenceONNX: No model path provided");
        return;
    }
    
    // Check if file exists
    std::ifstream file(m_modelPath);
    if (!file.good()) {
        logger->error("INetInferenceONNX: Model file does not exist: {}", m_modelPath);
        return;
    }
    
    _initModel();
#else
    logger->error("INetInferenceONNX: ONNX Runtime not enabled. Compile with -DUSE_ONNX_RUNTIME");
#endif
}

INetInferenceONNX::~INetInferenceONNX()
{
#ifdef USE_ONNX_RUNTIME
    if (m_session) {
        Ort::Session* session = static_cast<Ort::Session*>(m_session);
        delete session;
        m_session = nullptr;
    }
#endif
}

void INetInferenceONNX::_initModel()
{
    auto logger = spdlog::get("inet");
    
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
        
        // Input info
        size_t num_input_nodes = session->GetInputCount();
        if (num_input_nodes != 1) {
            logger->error("INetInferenceONNX: Expected 1 input, got {}", num_input_nodes);
            return;
        }
        
        auto input_name = session->GetInputNameAllocated(0, allocator);
        m_inputNames.push_back(input_name.get());
        
        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        m_inputShape = input_tensor_info.GetShape();
        
        // Handle dynamic dimensions (-1)
        if (m_inputShape.size() == 4) {
            m_inputHeight = m_inputShape[2] > 0 ? static_cast<int>(m_inputShape[2]) : 0;
            m_inputWidth = m_inputShape[3] > 0 ? static_cast<int>(m_inputShape[3]) : 0;
            m_inputChannel = m_inputShape[1] > 0 ? static_cast<int>(m_inputShape[1]) : 3;
        }
        
        // Output info
        size_t num_output_nodes = session->GetOutputCount();
        if (num_output_nodes != 1) {
            logger->error("INetInferenceONNX: Expected 1 output, got {}", num_output_nodes);
            return;
        }
        
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        m_outputNames.push_back(output_name.get());
        
        Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        m_outputShape = output_tensor_info.GetShape();
        
        if (m_outputShape.size() == 4) {
            m_outputHeight = m_outputShape[2] > 0 ? static_cast<int>(m_outputShape[2]) : 0;
            m_outputWidth = m_outputShape[3] > 0 ? static_cast<int>(m_outputShape[3]) : 0;
            m_outputChannel = m_outputShape[1] > 0 ? static_cast<int>(m_outputShape[1]) : 384;
        }
        
        logger->info("INetInferenceONNX: Model loaded successfully. Input: [{}x{}x{}], Output: [{}x{}x{}]",
                     m_inputChannel, m_inputHeight, m_inputWidth,
                     m_outputChannel, m_outputHeight, m_outputWidth);
    } catch (const std::exception& e) {
        logger->error("INetInferenceONNX: Failed to initialize model: {}", e.what());
        m_session = nullptr;
    }
#else
    if (logger) {
        logger->error("INetInferenceONNX: ONNX Runtime not enabled");
    }
#endif
}

#if defined(CV28) || defined(CV28_SIMULATOR)
bool INetInferenceONNX::runInference(ea_tensor_t* imgTensor, float* imap_out)
#else
bool INetInferenceONNX::runInference(void* imgTensor, float* imap_out)
#endif
{
    auto logger = spdlog::get("inet");
    
#ifdef USE_ONNX_RUNTIME
    if (!m_session) {
        if (logger) logger->error("INetInferenceONNX: Session not initialized");
        return false;
    }
    
    try {
        Ort::Session* session = static_cast<Ort::Session*>(m_session);
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Prepare input data
        std::vector<float> input_data;
        if (!_loadInput(imgTensor, input_data)) {
            if (logger) logger->error("INetInferenceONNX: Failed to load input");
            return false;
        }
        
        // Create input tensor
        std::vector<int64_t> input_shape = {1, m_inputChannel, m_inputHeight, m_inputWidth};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());
        
        // Prepare input/output names
        std::vector<const char*> input_names = {m_inputNames[0].c_str()};
        std::vector<const char*> output_names = {m_outputNames[0].c_str()};
        
        // Run inference
        auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                          input_names.data(), &input_tensor, 1,
                                          output_names.data(), 1);
        
        if (output_tensors.empty()) {
            if (logger) logger->error("INetInferenceONNX: No output tensors");
            return false;
        }
        
        // Extract output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = m_outputChannel * m_outputHeight * m_outputWidth;
        std::memcpy(imap_out, output_data, output_size * sizeof(float));
        
        if (logger) {
            logger->info("\033[33mINetInferenceONNX: Inference successful. Output size: {}\033[0m", output_size);
        }
        
        return true;
    } catch (const std::exception& e) {
        if (logger) logger->error("INetInferenceONNX: Inference failed: {}", e.what());
        return false;
    }
#else
    if (logger) logger->error("INetInferenceONNX: ONNX Runtime not enabled");
    return false;
#endif
}

#if defined(CV28) || defined(CV28_SIMULATOR)
bool INetInferenceONNX::_loadInput(ea_tensor_t* imgTensor, std::vector<float>& input_data)
#else
bool INetInferenceONNX::_loadInput(void* imgTensor, std::vector<float>& input_data)
#endif
{
#ifdef USE_ONNX_RUNTIME
#if defined(CV28) || defined(CV28_SIMULATOR)
    // Get image dimensions from tensor
    const size_t* shape = ea_tensor_shape(imgTensor);
    int H = static_cast<int>(shape[EA_H]);
    int W = static_cast<int>(shape[EA_W]);
    void* tensor_data = ea_tensor_data(imgTensor);
    
    if (!tensor_data) {
        return false;
    }
    
    // Resize input_data if needed
    input_data.resize(m_inputChannel * m_inputHeight * m_inputWidth);
    
    // Convert from uint8_t [C, H, W] to float [C, H, W] normalized
    // Normalization: 2 * (image / 255.0) - 0.5 = (2 * image - 127.5) / 255.0
    const uint8_t* src = static_cast<const uint8_t*>(tensor_data);
    
    // Resize if dimensions don't match (using nearest neighbor)
    if (H == m_inputHeight && W == m_inputWidth) {
        // Direct copy when dimensions match
        for (int c = 0; c < m_inputChannel; c++) {
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int src_idx = c * H * W + y * W + x;
                    int dst_idx = c * m_inputHeight * m_inputWidth + y * m_inputWidth + x;
                    float normalized = (2.0f * src[src_idx] - 127.5f) / 255.0f;
                    input_data[dst_idx] = normalized;
                }
            }
        }
    } else {
        // Resize using nearest neighbor interpolation
        float scale_x = static_cast<float>(W) / static_cast<float>(m_inputWidth);
        float scale_y = static_cast<float>(H) / static_cast<float>(m_inputHeight);
        
        for (int c = 0; c < m_inputChannel; c++) {
            for (int y = 0; y < m_inputHeight; y++) {
                for (int x = 0; x < m_inputWidth; x++) {
                    // Map destination coordinates to source coordinates
                    int src_x = static_cast<int>(x * scale_x);
                    int src_y = static_cast<int>(y * scale_y);
                    
                    // Clamp to valid range
                    src_x = std::max(0, std::min(src_x, W - 1));
                    src_y = std::max(0, std::min(src_y, H - 1));
                    
                    int src_idx = c * H * W + src_y * W + src_x;
                    int dst_idx = c * m_inputHeight * m_inputWidth + y * m_inputWidth + x;
                    float normalized = (2.0f * src[src_idx] - 127.5f) / 255.0f;
                    input_data[dst_idx] = normalized;
                }
            }
        }
    }
    
    return true;
#else
    // Fallback for non-CV28: imgTensor is uint8_t* [C, H, W]
    // This would need to be implemented based on your non-CV28 image format
    return false;  // Not implemented for non-CV28
#endif
#else
    return false;
#endif
}

