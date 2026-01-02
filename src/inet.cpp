#include "inet.hpp"
#include "dla_config.hpp"
#include "logger.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// =================================================================================================
// INet Inference Implementation
// =================================================================================================
INetInference::INetInference(Config_S *config)
{
    // Check if logger already exists (to avoid "logger already exists" error)
    auto logger = spdlog::get("inet");
    if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("inet", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
        logger = spdlog::stdout_color_mt("inet");
        logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    }

    logger->set_level(config->stDebugConfig.AIModel ? spdlog::level::debug : spdlog::level::info);

#if defined(CV28) || defined(CV28_SIMULATOR)
    // Use inetModelPath if available, otherwise fallback to modelPath
    m_modelPathStr = !config->inetModelPath.empty() ? config->inetModelPath : config->modelPath;
    m_ptrModelPath = const_cast<char *>(m_modelPathStr.c_str());

    // Initialize network parameters
    ea_net_params_t net_params;
    memset(&net_params, 0, sizeof(net_params));
    net_params.acinf_gpu_id = -1;

    // Create network instance
    m_model = ea_net_new(&net_params);
    if (m_model == NULL)
    {
        logger->error("Creating INet model failed");
    }

    m_inputTensor = nullptr;
    m_outputTensor = nullptr;

    _initModelIO();
#endif
}

INetInference::~INetInference()
{
#if defined(CV28) || defined(CV28_SIMULATOR)
    _releaseModel();
    if (m_outputBuffer)
    {
        delete[] m_outputBuffer;
    }
#endif
}

bool INetInference::runInference(const float *image, int H, int W, float *imap_out)
{
#if defined(CV28) || defined(CV28_SIMULATOR)
    auto logger = spdlog::get("inet");

    if (!_loadInput(image, H, W))
    {
        logger->info("INet: Load Input Data Failed");
        return false;
    }else{
        logger->info("INet: Load Input Data successful");
    }

    // Run inference
    if (EA_SUCCESS != ea_net_forward(m_model, 1))
    {
        logger->info("INet: Inference failed");
        return false;
    }else{
        logger->info("INet: Inference successful");
    }

    // Sync output tensor
#if defined(CV28)
    int rval = ea_tensor_sync_cache(m_outputTensor, EA_VP, EA_CPU);
    if (rval != EA_SUCCESS)
    {
        logger->info("INet: Failed to sync output tensor");
    }else{
        logger->info("INet: sync output tensor successful");
    }
#endif

    m_outputTensor = ea_net_output_by_index(m_model, 0);
    if (logger) logger->info("INet: Got output tensor");

    // Copy output to imap_out
    // Model output: [1, 384, H/4, W/4] (channel-first in tensor)
    // Python works at 1/4 resolution (RES=4), so use model output directly without upsampling
    // Output format: [384, H/4, W/4] where H/4, W/4 match the model output resolution
    const int outH = m_outputHeight;  // e.g., 120 (H/4)
    const int outW = m_outputWidth;   // e.g., 160 (W/4)
    const int outC = m_outputChannel; // 384

    if (logger) logger->info("INet: About to get tensor_data, outH={}, outW={}, outC={}", outH, outW, outC);

    // Get actual tensor size to verify
    const size_t *tensor_shape = ea_tensor_shape(m_outputTensor);
    const size_t tensor_N = tensor_shape[EA_N];
    const size_t tensor_C = tensor_shape[EA_C];
    const size_t tensor_H = tensor_shape[EA_H];
    const size_t tensor_W = tensor_shape[EA_W];
    const size_t tensor_total_elements = tensor_N * tensor_C * tensor_H * tensor_W;
    const size_t tensor_total_bytes = tensor_total_elements * sizeof(float);
    
    if (logger) logger->info("INet: Tensor shape: N={}, C={}, H={}, W={}, total_elements={}, total_bytes={}", 
                              tensor_N, tensor_C, tensor_H, tensor_W, tensor_total_elements, tensor_total_bytes);

    // Copy from tensor directly to imap_out (no upsampling needed)
    // Tensor layout: [N, C, H, W] = [1, 384, 120, 160]
    // Output layout: [C, H, W] = [384, 120, 160]
    float *tensor_data = (float *)ea_tensor_data(m_outputTensor);
    if (tensor_data == nullptr) {
        if (logger) logger->error("INet: tensor_data is nullptr!");
        return false;
    }
    if (logger) logger->info("INet: Got tensor_data pointer: {}", (void*)tensor_data);
    
    if (imap_out == nullptr) {
        if (logger) logger->info("INet: imap_out is nullptr!");
        return false;
    }
    if (logger) logger->info("INet: imap_out pointer: {}", (void*)imap_out);
    
    const size_t expected_size = static_cast<size_t>(outC) * outH * outW;
    const size_t expected_bytes = expected_size * sizeof(float);
    if (logger) logger->info("INet: Expected output size: {} elements, {} bytes", expected_size, expected_bytes);
    
    // Verify tensor size matches expected size (accounting for N=1 dimension)
    if (tensor_total_elements != expected_size) {
        if (logger) logger->info("INet: Size mismatch! tensor_total_elements={}, expected_size={}", 
                                  tensor_total_elements, expected_size);
        // Still try to copy, but use the smaller size
    }
    
    // Try to access first element to test if memory is valid
    if (logger) logger->info("INet: About to test first element access");
    volatile float test_val = tensor_data[0];  // Use volatile to prevent optimization
    if (logger) logger->info("INet: First tensor element read: {}", test_val);
    
    // Copy data using the nested loop approach (safer for layout conversion)
    // Tensor layout: [N=0, C, H, W]
    // Output layout: [C, H, W]
    if (logger) logger->info("INet: Starting copy loop, outC={}, outH={}, outW={}", outC, outH, outW);
    
    // Test first iteration separately with extensive logging
    int c = 0, y = 0, x = 0;
    if (logger) logger->info("INet: About to calculate first indices");
    int tensor_idx = 0 * outC * outH * outW + c * outH * outW + y * outW + x;
    int dst_idx = c * outH * outW + y * outW + x;
    if (logger) logger->info("INet: First indices calculated: tensor_idx={}, dst_idx={}", tensor_idx, dst_idx);
    
    if (static_cast<size_t>(tensor_idx) >= tensor_total_elements) {
        if (logger) logger->info("INet: tensor_idx out of bounds! tensor_idx={}, tensor_total={}", 
                                  tensor_idx, tensor_total_elements);
        return false;
    }
    if (static_cast<size_t>(dst_idx) >= expected_size) {
        if (logger) logger->info("INet: dst_idx out of bounds! dst_idx={}, expected={}", 
                                  dst_idx, expected_size);
        return false;
    }
    
    if (logger) logger->info("INet: About to access tensor_data[{}]", tensor_idx);
    volatile float test_tensor_val = tensor_data[tensor_idx];
    if (logger) logger->info("INet: tensor_data[{}] = {}", tensor_idx, test_tensor_val);
    
    if (logger) logger->info("INet: About to write to imap_out[{}]", dst_idx);
    imap_out[dst_idx] = test_tensor_val / 4.0f;
    if (logger) logger->info("INet: First element written successfully");
    
    // Continue with the rest of the loop
    if (logger) logger->info("INet: Starting main copy loop");
    for (c = 0; c < outC; c++)
    {
        if (c == 0 && logger) logger->info("INet: Channel 0, starting y loop");
        for (y = 0; y < outH; y++)
        {
            if (c == 0 && y == 0 && logger) logger->info("INet: Channel 0, Row 0, starting x loop");
            for (x = 0; x < outW; x++)
            {
                // Skip first element (already done)
                if (c == 0 && y == 0 && x == 0) {
                    if (logger) logger->info("INet: Skipping first element (c=0, y=0, x=0)");
                    continue;
                }
                
                // Tensor layout: [N=0, C, H, W]
                tensor_idx = 0 * outC * outH * outW + c * outH * outW + y * outW + x;
                // Output layout: [C, H, W]
                dst_idx = c * outH * outW + y * outW + x;
                
                if (static_cast<size_t>(tensor_idx) >= tensor_total_elements) {
                    if (logger) {
                        logger->info("INet: tensor_idx out of bounds! c={}, y={}, x={}, tensor_idx={}, tensor_total={}", 
                                      c, y, x, tensor_idx, tensor_total_elements);
                    }
                    return false;
                }
                if (static_cast<size_t>(dst_idx) >= expected_size) {
                    if (logger) {
                        logger->info("INet: dst_idx out of bounds! c={}, y={}, x={}, dst_idx={}, expected={}", 
                                      c, y, x, dst_idx, expected_size);
                    }
                    return false;
                }
                
                if (c == 0 && y == 0 && x < 5 && logger) {
                    logger->info("INet: About to read tensor_data[{}]", tensor_idx);
                }
                volatile float val = tensor_data[tensor_idx];
                
                imap_out[dst_idx] = val / 4.0f;
            }
        }
    }
    if (logger) logger->info("INet: Copy loop completed, about to return");

    return true;
#else
    return false;
#endif
}

#if defined(CV28) || defined(CV28_SIMULATOR)
void INetInference::_initModelIO()
{
    auto logger = spdlog::get("inet");

    int rval = EA_SUCCESS;
    logger->info("-------------------------------------------");
    logger->info("Configure INet Model Input/Output");

    // Configure input tensor
    logger->info("Input Name: {}", m_inputTensorName);
    rval = ea_net_config_input(m_model, m_inputTensorName.c_str());

    // Configure output tensor
    logger->info("Output Name: {}", m_outputTensorName);
    rval = ea_net_config_output(m_model, m_outputTensorName.c_str());

    // Load model
    logger->info("Model Path: {}", m_ptrModelPath);
    FILE *file = fopen(m_ptrModelPath, "r");
    if (file == nullptr)
    {
        logger->error("INet model file does not exist at path: {}", m_ptrModelPath);
        return;
    }
    fclose(file);

    rval = ea_net_load(m_model, EA_NET_LOAD_FILE, (void *)m_ptrModelPath, 1);

    // Get input tensor
    m_inputTensor = ea_net_input(m_model, m_inputTensorName.c_str());
    m_inputHeight = ea_tensor_shape(m_inputTensor)[EA_H];
    m_inputWidth = ea_tensor_shape(m_inputTensor)[EA_W];
    m_inputChannel = ea_tensor_shape(m_inputTensor)[EA_C];
    logger->info("INet Input H: {}, W: {}, C: {}", m_inputHeight, m_inputWidth, m_inputChannel);

    // Get output tensor
    m_outputTensor = ea_net_output_by_index(m_model, 0);
    m_outputHeight = ea_tensor_shape(m_outputTensor)[EA_H];
    m_outputWidth = ea_tensor_shape(m_outputTensor)[EA_W];
    m_outputChannel = ea_tensor_shape(m_outputTensor)[EA_C];
    logger->info("INet Output H: {}, W: {}, C: {}", m_outputHeight, m_outputWidth, m_outputChannel);
}

bool INetInference::_releaseModel()
{
    if (m_model)
    {
        ea_net_free(m_model);
        m_model = nullptr;
    }
    return true;
}

bool INetInference::_loadInput(const float *image, int H, int W)
{
    // Convert normalized float image [C, H, W] (channel-first, range [-0.5, 1.5]) 
    // to model input format [1, 3, H, W]
    // Python: image = 2 * (image / 255.0) - 0.5 -> range [-0.5, 1.5]
    // 
    // AMBA model configuration (from YAML):
    //   - Input format: uint8 [0, 255]
    //   - std: 256 (model normalizes input by dividing by 256 during inference, handled automatically by AMBA runtime)
    // 
    // So we need to convert normalized float [-0.5, 1.5] back to uint8 [0, 255]
    // Conversion: (normalized + 0.5) / 2.0 * 255.0 -> [0, 255]
    // The model will then automatically normalize uint8/256 -> [0.0, 1.0] during inference
    
    if (m_inputTensor == nullptr) {
        auto logger = spdlog::get("inet");
        if (logger) logger->error("INet: m_inputTensor is nullptr!");
        return false;
    }
    
    const int inputSize = m_inputChannel * m_inputHeight * m_inputWidth;
    std::vector<uint8_t> inputBuffer(inputSize);
    
    // Resize image from [C, H, W] to [C, inputH, inputW] and convert normalized float to uint8
    for (int c = 0; c < m_inputChannel; c++)
    {
        for (int y = 0; y < m_inputHeight; y++)
        {
            for (int x = 0; x < m_inputWidth; x++)
            {
                int src_y = (y * H) / m_inputHeight;
                int src_x = (x * W) / m_inputWidth;
                int src_idx = c * H * W + src_y * W + src_x;
                
                // Convert normalized float [-0.5, 1.5] to uint8 [0, 255]
                // Formula: (normalized + 0.5) / 2.0 * 255.0
                float normalized_val = image[src_idx];
                float mapped_val = (normalized_val + 0.5f) / 2.0f * 255.0f;
                uint8_t uint8_val = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, mapped_val)));
                
                int dst_idx = 0 * m_inputChannel * m_inputHeight * m_inputWidth +
                              c * m_inputHeight * m_inputWidth +
                              y * m_inputWidth + x;
                inputBuffer[dst_idx] = uint8_val;
            }
        }
    }
    
    void* tensor_data = ea_tensor_data(m_inputTensor);
    if (tensor_data == nullptr) {
        auto logger = spdlog::get("inet");
        if (logger) logger->error("INet: ea_tensor_data returned nullptr!");
        return false;
    }
    std::memcpy(tensor_data, inputBuffer.data(), inputSize * sizeof(uint8_t));

    return true;
}
#endif

int INetInference::getInputHeight() const {
#if defined(CV28) || defined(CV28_SIMULATOR)
    return m_inputHeight;
#else
    return 0;
#endif
}

int INetInference::getInputWidth() const {
#if defined(CV28) || defined(CV28_SIMULATOR)
    return m_inputWidth;
#else
    return 0;
#endif
}

int INetInference::getOutputHeight() const {
#if defined(CV28) || defined(CV28_SIMULATOR)
    return m_outputHeight;
#else
    return 0;
#endif
}

int INetInference::getOutputWidth() const {
#if defined(CV28) || defined(CV28_SIMULATOR)
    return m_outputWidth;
#else
    return 0;
#endif
}

