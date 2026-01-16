#include "inet.hpp"
#include "dla_config.hpp"
#include "logger.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>
#include <sstream>

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

#if defined(CV28) || defined(CV28_SIMULATOR)
// Tensor-based _loadInput - uses tensor directly, avoids conversion
bool INetInference::_loadInput(ea_tensor_t* imgTensor)
{
    if (imgTensor == nullptr) {
        auto logger = spdlog::get("inet");
        if (logger) logger->error("INet: imgTensor is nullptr!");
        return false;
    }
    
    if (m_inputTensor == nullptr) {
        auto logger = spdlog::get("inet");
        if (logger) logger->error("INet: m_inputTensor is nullptr!");
        return false;
    }
    
    auto logger = spdlog::get("inet");
    
    // Use ea_cvt_color_resize directly on the input tensor
    // The input tensor from ea_img_resource is typically in RGB format
    // Use EA_COLOR_BGR2RGB (same as uint8_t* path) for consistency
    int rval = EA_SUCCESS;
#if defined(CV28)
    rval = ea_cvt_color_resize(imgTensor, m_inputTensor, EA_COLOR_BGR2RGB, EA_VP);
#elif defined(CV28_SIMULATOR)
    rval = ea_cvt_color_resize(imgTensor, m_inputTensor, EA_COLOR_BGR2RGB, EA_CPU);
#endif
    
    if (rval != EA_SUCCESS) {
        if (logger) logger->error("INet: ea_cvt_color_resize (tensor) failed with error: {}", rval);
        return false;
    }
    
    if (logger) logger->info("INet: _loadInput (tensor) successful using ea_cvt_color_resize");
    return true;
}

// Tensor-based runInference overload
bool INetInference::runInference(ea_tensor_t* imgTensor, float* imap_out)
{
    if (imgTensor == nullptr || imap_out == nullptr) {
        return false;
    }
    
    auto logger = spdlog::get("inet");
    
    if (!_loadInput(imgTensor)) {
        if (logger) logger->info("INet: Load Input Data Failed (tensor)");
        return false;
    } else {
        if (logger) logger->info("INet: Load Input Data successful (tensor)");
    }
    
    // Run inference (same as uint8_t* version)
    if (EA_SUCCESS != ea_net_forward(m_model, 1)) {
        if (logger) logger->info("INet: Inference failed");
        return false;
    } else {
        if (logger) logger->info("\033[33mINet: Inference successful\033[0m");
    }
    
    // Sync output tensor
#if defined(CV28)
    int rval = ea_tensor_sync_cache(m_outputTensor, EA_VP, EA_CPU);
    if (rval != EA_SUCCESS) {
        if (logger) logger->info("INet: Failed to sync output tensor");
    } else {
        if (logger) logger->info("INet: sync output tensor successful");
    }
#endif
    
    m_outputTensor = ea_net_output_by_index(m_model, 0);
    if (logger) logger->info("INet: Got output tensor");
    
    // Get actual tensor shape to verify layout
    // NOTE: Unlike YOLOv8 which uses serialized format due to SNPE converter issues,
    // FNet/INet should use standard NCHW format [N, C, H, W]
    const size_t *tensor_shape = ea_tensor_shape(m_outputTensor);
    const size_t tensor_N = tensor_shape[EA_N];
    const size_t tensor_C = tensor_shape[EA_C];
    const size_t tensor_H = tensor_shape[EA_H];
    const size_t tensor_W = tensor_shape[EA_W];
    
    // Verify tensor shape matches expected dimensions
    if (tensor_N != 1 || tensor_C != static_cast<size_t>(m_outputChannel) || 
        tensor_H != static_cast<size_t>(m_outputHeight) || tensor_W != static_cast<size_t>(m_outputWidth)) {
        if (logger) logger->error("INet: Tensor shape mismatch! Expected [1, {}, {}, {}], got [{}, {}, {}, {}]",
                     m_outputChannel, m_outputHeight, m_outputWidth,
                     tensor_N, tensor_C, tensor_H, tensor_W);
    }
    
    // Copy output
    // Assumes standard NCHW format: [N=1, C, H, W]
    // If AMBA CV28 uses serialized format like YOLOv8, this would need to be changed
    const int outH = m_outputHeight;
    const int outW = m_outputWidth;
    const int outC = m_outputChannel;
    
    float *tensor_data = (float *)ea_tensor_data(m_outputTensor);
    if (tensor_data == nullptr) {
        if (logger) logger->error("INet: tensor_data is nullptr!");
        return false;
    }
    
    if (logger) {
        logger->info("INet: Tensor shape from ea_tensor_shape: N={}, C={}, H={}, W={}", 
                      tensor_N, tensor_C, tensor_H, tensor_W);
        logger->info("INet: First 5 tensor values (raw): [{}, {}, {}, {}, {}]",
                     tensor_data[0], tensor_data[1], tensor_data[2], tensor_data[3], tensor_data[4]);
    }
    
    // Copy assuming standard NCHW format: [N=0, C, H, W]
    for (int c = 0; c < outC; c++) {
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                // Standard NCHW indexing: [N=0, C, H, W]
                int tensor_idx = 0 * outC * outH * outW + c * outH * outW + y * outW + x;
                // Output layout: [C, H, W] (removing batch dimension)
                int dst_idx = c * outH * outW + y * outW + x;
                imap_out[dst_idx] = tensor_data[tensor_idx];
            }
        }
    }
    
    return true;
}
#endif

int INetInference::getInputHeight() const {
    return m_inputHeight;
}

int INetInference::getInputWidth() const {
    return m_inputWidth;
}

int INetInference::getOutputHeight() const {
    return m_outputHeight;
}

int INetInference::getOutputWidth() const {
    return m_outputWidth;
}

