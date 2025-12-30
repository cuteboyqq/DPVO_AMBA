#include "net.hpp"
#include "correlation_kernel.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

// =================================================================================================
// FNet Inference Implementation
// =================================================================================================
FNetInference::FNetInference(Config_S *config)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::syslog_logger_mt("fnet", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
    auto logger = spdlog::stdout_color_mt("fnet");
    logger->set_pattern("[%n] [%^%l%$] %v");
#endif

    logger->set_level(config->stDebugConfig.AIModel ? spdlog::level::debug : spdlog::level::info);

#if defined(CV28) || defined(CV28_SIMULATOR)
    // Use fnetModelPath if available, otherwise fallback to modelPath
    m_modelPathStr = !config->fnetModelPath.empty() ? config->fnetModelPath : config->modelPath;
    m_ptrModelPath = const_cast<char *>(m_modelPathStr.c_str());

    // Initialize network parameters
    ea_net_params_t net_params;
    memset(&net_params, 0, sizeof(net_params));
    net_params.acinf_gpu_id = -1;

    // Create network instance
    m_model = ea_net_new(&net_params);
    if (m_model == NULL)
    {
        logger->info("Creating FNet model failed");
    }else{
        logger->info("Creating FNet model successful");
    }

    m_inputTensor = nullptr;
    m_outputTensor = nullptr;

    _initModelIO();
#endif
}

FNetInference::~FNetInference()
{
#if defined(CV28) || defined(CV28_SIMULATOR)
    _releaseModel();
    if (m_outputBuffer)
    {
        delete[] m_outputBuffer;
    }
#endif
}

bool FNetInference::runInference(const uint8_t *image, int H, int W, float *fmap_out)
{
#if defined(CV28) || defined(CV28_SIMULATOR)
    auto logger = spdlog::get("fnet");

    if (!_loadInput(image, H, W))
    {
        logger->info("FNet: Load Input Data Failed");
        return false;
    }else{
        logger->info("FNet: Load Input Data successful");
    }

    // Run inference
    if (EA_SUCCESS != ea_net_forward(m_model, 1))
    {
        logger->info("FNet: Inference failed");
        return false;
    }else{logger->info("FNet: Inference successful");}

    // Sync output tensor
#if defined(CV28)
    int rval = ea_tensor_sync_cache(m_outputTensor, EA_VP, EA_CPU);
    if (rval != EA_SUCCESS)
    {
        logger->info("FNet: Failed to sync output tensor");
    }
#endif

    m_outputTensor = ea_net_output_by_index(m_model, 0);

    // Copy output to fmap_out
    // Model output: [1, 128, H/4, W/4] (channel-first in tensor)
    // Python works at 1/4 resolution (RES=4), so use model output directly without upsampling
    // Output format: [128, H/4, W/4] where H/4, W/4 match the model output resolution
    const int outH = m_outputHeight;  // e.g., 120 (H/4)
    const int outW = m_outputWidth;   // e.g., 160 (W/4)
    const int outC = m_outputChannel; // 128

    // Copy from tensor directly to fmap_out (no upsampling needed)
    // Tensor layout: [N, C, H, W] = [1, 128, 120, 160]
    // Output layout: [C, H, W] = [128, 120, 160]
    float *tensor_data = (float *)ea_tensor_data(m_outputTensor);
    for (int c = 0; c < outC; c++)
    {
        for (int y = 0; y < outH; y++)
        {
            for (int x = 0; x < outW; x++)
            {
                // Tensor layout: [N=0, C, H, W]
                int tensor_idx = 0 * outC * outH * outW + c * outH * outW + y * outW + x;
                // Output layout: [C, H, W] - same resolution as model output
                int dst_idx = c * outH * outW + y * outW + x;
                fmap_out[dst_idx] = tensor_data[tensor_idx] / 4.0f; // Divide by 4.0 as in Python
            }
        }
    }

    return true;
#else
    return false;
#endif
}

#if defined(CV28) || defined(CV28_SIMULATOR)
void FNetInference::_initModelIO()
{
    auto logger = spdlog::get("fnet");

    int rval = EA_SUCCESS;
    logger->info("-------------------------------------------");
    logger->info("Configure FNet Model Input/Output");

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
        logger->error("FNet model file does not exist at path: {}", m_ptrModelPath);
        return;
    }else{
        logger->error("FNet model file exist at path: {}", m_ptrModelPath);
    }
    fclose(file);

    rval = ea_net_load(m_model, EA_NET_LOAD_FILE, (void *)m_ptrModelPath, 1);

    // Get input tensor
    m_inputTensor = ea_net_input(m_model, m_inputTensorName.c_str());
    m_inputHeight = ea_tensor_shape(m_inputTensor)[EA_H];
    m_inputWidth = ea_tensor_shape(m_inputTensor)[EA_W];
    m_inputChannel = ea_tensor_shape(m_inputTensor)[EA_C];
    logger->info("FNet Input H: {}, W: {}, C: {}", m_inputHeight, m_inputWidth, m_inputChannel);

    // Get output tensor
    m_outputTensor = ea_net_output_by_index(m_model, 0);
    m_outputHeight = ea_tensor_shape(m_outputTensor)[EA_H];
    m_outputWidth = ea_tensor_shape(m_outputTensor)[EA_W];
    m_outputChannel = ea_tensor_shape(m_outputTensor)[EA_C];
    logger->info("FNet Output H: {}, W: {}, C: {}", m_outputHeight, m_outputWidth, m_outputChannel);
}

bool FNetInference::_releaseModel()
{
    if (m_model)
    {
        ea_net_free(m_model);
        m_model = nullptr;
    }
    return true;
}

bool FNetInference::_loadInput(const uint8_t *image, int H, int W)
{
    // Convert uint8 image [C, H, W] (channel-first) to model input format [1, 3, H, W]
    // Model expects: uint8, normalized by std=256 (divide by 256) - handled by model

    // Allocate input buffer
    const int inputSize = m_inputChannel * m_inputHeight * m_inputWidth;
    std::vector<uint8_t> inputBuffer(inputSize);

    // Resize image from [C, H, W] to [C, inputH, inputW]
    // Simple nearest neighbor resize if needed
    for (int c = 0; c < m_inputChannel; c++)
    {
        for (int y = 0; y < m_inputHeight; y++)
        {
            for (int x = 0; x < m_inputWidth; x++)
            {
                int src_y = (y * H) / m_inputHeight;
                int src_x = (x * W) / m_inputWidth;
                // Source image is channel-first: [C, H, W]
                int src_idx = c * H * W + src_y * W + src_x;
                // Tensor layout: [N=0, C, H, W]
                int dst_idx = 0 * m_inputChannel * m_inputHeight * m_inputWidth +
                              c * m_inputHeight * m_inputWidth +
                              y * m_inputWidth + x;
                inputBuffer[dst_idx] = image[src_idx];
            }
        }
    }

    // Copy to input tensor (model will normalize by std=256 internally)
    std::memcpy(ea_tensor_data(m_inputTensor), inputBuffer.data(), inputSize * sizeof(uint8_t));

    return true;
}
#endif

// =================================================================================================
// INet Inference Implementation
// =================================================================================================
INetInference::INetInference(Config_S *config)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::syslog_logger_mt("inet", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
    auto logger = spdlog::stdout_color_mt("inet");
    logger->set_pattern("[%n] [%^%l%$] %v");
#endif

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

bool INetInference::runInference(const uint8_t *image, int H, int W, float *imap_out)
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
                // Log every iteration for first few elements to catch the crash
                // if (c == 0 && y == 0 && x < 5 && logger) {
                //     logger->info("INet: Loop iteration: c={}, y={}, x={}", c, y, x);
                // }
                
                // Skip first element (already done)
                if (c == 0 && y == 0 && x == 0) {
                    if (logger) logger->info("INet: Skipping first element (c=0, y=0, x=0)");
                    continue;
                }
                
                // if (c == 0 && y == 0 && x < 5 && logger) {
                //     logger->info("INet: Processing element (c={}, y={}, x={})", c, y, x);
                // }
                
                // Tensor layout: [N=0, C, H, W]
                tensor_idx = 0 * outC * outH * outW + c * outH * outW + y * outW + x;
                // Output layout: [C, H, W]
                dst_idx = c * outH * outW + y * outW + x;
                
                // if (c == 0 && y == 0 && x < 5 && logger) {
                //     logger->info("INet: Calculated indices: tensor_idx={}, dst_idx={}", tensor_idx, dst_idx);
                // }
                
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
                // if (c == 0 && y == 0 && x < 5 && logger) {
                //     logger->info("INet: Read tensor_data[{}] = {}", tensor_idx, val);
                // }
                
                // if (c == 0 && y == 0 && x < 5 && logger) {
                //     logger->info("INet: About to write imap_out[{}]", dst_idx);
                // }
                imap_out[dst_idx] = val / 4.0f;
                // if (c == 0 && y == 0 && x < 5 && logger) {
                //     logger->info("INet: Wrote imap_out[{}] successfully", dst_idx);
                // }
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

bool INetInference::_loadInput(const uint8_t *image, int H, int W)
{
    // Convert uint8 image [C, H, W] (channel-first) to model input format [1, 3, H, W]
    // Model expects: uint8, normalized by std=256 (divide by 256) - handled by model

    // Allocate input buffer
    const int inputSize = m_inputChannel * m_inputHeight * m_inputWidth;
    std::vector<uint8_t> inputBuffer(inputSize);

    // Resize image from [C, H, W] to [C, inputH, inputW]
    // Simple nearest neighbor resize if needed
    for (int c = 0; c < m_inputChannel; c++)
    {
        for (int y = 0; y < m_inputHeight; y++)
        {
            for (int x = 0; x < m_inputWidth; x++)
            {
                int src_y = (y * H) / m_inputHeight;
                int src_x = (x * W) / m_inputWidth;
                // Source image is channel-first: [C, H, W]
                int src_idx = c * H * W + src_y * W + src_x;
                // Tensor layout: [N=0, C, H, W]
                int dst_idx = 0 * m_inputChannel * m_inputHeight * m_inputWidth +
                              c * m_inputHeight * m_inputWidth +
                              y * m_inputWidth + x;
                inputBuffer[dst_idx] = image[src_idx];
            }
        }
    }

    // Copy to input tensor (model will normalize by std=256 internally)
    std::memcpy(ea_tensor_data(m_inputTensor), inputBuffer.data(), inputSize * sizeof(uint8_t));

    return true;
}
#endif

// =================================================================================================
// Patchifier Implementation
// =================================================================================================
Patchifier::Patchifier(int patch_size, int DIM)
    : m_patch_size(patch_size), m_DIM(DIM), m_fnet(nullptr), m_inet(nullptr)
{
}

Patchifier::Patchifier(int patch_size, int DIM, Config_S *config)
    : m_patch_size(patch_size), m_DIM(DIM)
{
    // Models will be set via setModels() if config provided
    if (config != nullptr)
    {
        // Note: You'll need separate configs for fnet and inet
        // For now, assuming same config path structure
    }
}

Patchifier::~Patchifier()
{
    // Models will be automatically destroyed by unique_ptr
}

void Patchifier::setModels(Config_S *fnetConfig, Config_S *inetConfig)
{
    // Drop existing loggers if they exist (in case models were created elsewhere)
    // This prevents "logger with name already exists" errors
#ifdef SPDLOG_USE_SYSLOG
    spdlog::drop("fnet");
    spdlog::drop("inet");
#else
    spdlog::drop("fnet");
    spdlog::drop("inet");
#endif
    
    if (fnetConfig != nullptr)
    {
        m_fnet = std::make_unique<FNetInference>(fnetConfig);
    }
    if (inetConfig != nullptr)
    {
        m_inet = std::make_unique<INetInference>(inetConfig);
    }
}

// Forward pass: fill fmap, imap, gmap, patches, clr
// Note: fmap and imap are at 1/4 resolution (RES=4), but image and coords are at full resolution
void Patchifier::forward(
    const uint8_t *image,
    int H, int W,   // Actual image dimensions (may differ from model input)
    float *fmap,    // [128, model_H/4, model_W/4] - at 1/4 resolution of model input
    float *imap,    // [DIM, model_H/4, model_W/4] - at 1/4 resolution of model input
    float *gmap,    // [M, 128, P, P]
    float *patches, // [M, 3, P, P]
    uint8_t *clr,   // [M, 3]
    int M)
{
    const int RES = 4;  // Resolution factor (Python RES=4)
    const int inet_output_channels = 384;  // INet model output channels (not m_DIM which defaults to 64)
    
    // Get model output dimensions (models resize input internally)
    // fmap/imap buffers are sized based on model output, not actual image size
    int fmap_H = 0, fmap_W = 0;
    if (m_fnet != nullptr && m_inet != nullptr)
    {
        // Use model output dimensions (models output at 1/4 of their input size)
        fmap_H = m_fnet->getOutputHeight();  // e.g., 120 (480/4)
        fmap_W = m_fnet->getOutputWidth();   // e.g., 160 (640/4)
        
        // Validate that fnet and inet have same output dimensions
        if (fmap_H != m_inet->getOutputHeight() || fmap_W != m_inet->getOutputWidth())
        {
            throw std::runtime_error("FNet and INet output dimension mismatch");
        }
    }
    else
    {
        // Fallback: calculate from actual image dimensions (should not happen if models are set)
        fmap_H = H / RES;
        fmap_W = W / RES;
    }

    // ------------------------------------------------
    // 1. Run fnet and inet inference to get fmap and imap
    // ------------------------------------------------
    if (m_fnet != nullptr && m_inet != nullptr)
    {
        // Allocate temporary buffers for model output size (1/4 resolution)
        if (m_fmap_buffer.size() != 128 * fmap_H * fmap_W)
        {
            m_fmap_buffer.resize(128 * fmap_H * fmap_W);
        }
        // INet outputs 384 channels, so use that instead of m_DIM
        // m_DIM might be 64 (default), but we need 384 for INet output
        if (m_imap_buffer.size() != inet_output_channels * fmap_H * fmap_W)
        {
            m_imap_buffer.resize(inet_output_channels * fmap_H * fmap_W);
        }
        auto logger_patch = spdlog::get("fnet");
        if (!logger_patch) {
            logger_patch = spdlog::get("inet");
        }


        
        if (logger_patch) logger_patch->error("[Patchifier] About to call fnet->runInference");
        // Run fnet inference (models will resize input internally)
        if (!m_fnet->runInference(image, H, W, m_fmap_buffer.data()))
        {
            if (logger_patch) logger_patch->error("[Patchifier] fnet->runInference failed");
            // Fallback: zero fill if inference fails
            std::fill(m_fmap_buffer.begin(), m_fmap_buffer.end(), 0.0f);
        }else{
            if (logger_patch) logger_patch->error("[Patchifier] fnet->runInference successful");
        }

            // Run inet inference (models will resize input internally)
        



        if (logger_patch) logger_patch->error("[Patchifier] About to call inet->runInference");
        
        if (!m_inet->runInference(image, H, W, m_imap_buffer.data()))
        {
            if (logger_patch) logger_patch->error("[Patchifier] inet->runInference failed");
            // Fallback: zero fill if inference fails
            std::fill(m_imap_buffer.begin(), m_imap_buffer.end(), 0.0f);
        } else {
            if (logger_patch) logger_patch->error("[Patchifier] inet->runInference successful");
        }





        if (logger_patch) logger_patch->error("[Patchifier] About to memcpy fmap, size={}", 128 * fmap_H * fmap_W * sizeof(float));
        // Copy to output buffers (already at 1/4 resolution of model input)
        std::memcpy(fmap, m_fmap_buffer.data(), 128 * fmap_H * fmap_W * sizeof(float));
        if (logger_patch) logger_patch->error("[Patchifier] fmap memcpy completed");
        
        // NOTE: imap parameter is actually a buffer for patch features [M, DIM], not the full feature map
        // We need to extract patches from the full feature map [384, 120, 160] using patchify_cpu_safe
        // So we don't memcpy the full feature map to imap - instead we'll extract patches later
        if (logger_patch) logger_patch->error("[Patchifier] Skipping imap memcpy - will extract patches later");
    }
    else
    {
        // Fallback: zero fill if models not available
        std::fill(fmap, fmap + 128 * fmap_H * fmap_W, 0.0f);
        // imap will be zero-filled later in patchify_cpu_safe fallback
    }

    printf("[Patchifier] About to create grid, H=%d, W=%d\n", H, W);
    fflush(stdout);
    
    // ------------------------------------------------
    // 2. Image → float grid (for patches) - full resolution
    // ------------------------------------------------
    std::vector<float> grid(3 * H * W);
    printf("[Patchifier] Grid created, size=%zu\n", grid.size());
    fflush(stdout);
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < H * W; i++)
            grid[c * H * W + i] = image[c * H * W + i] / 255.0f;

    printf("[Patchifier] About to create coords, M=%d\n", M);
    fflush(stdout);
    
    // ------------------------------------------------
    // 3. Generate RANDOM coords (Python RANDOM mode) - full resolution
    // ------------------------------------------------
    std::vector<float> coords(M * 2);
    for (int m = 0; m < M; m++)
    {
        coords[m * 2 + 0] = 1 + rand() % (W - 2); // Full resolution coordinates
        coords[m * 2 + 1] = 1 + rand() % (H - 2); // Full resolution coordinates
    }
    printf("[Patchifier] Coords created\n");
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (patches)\n");
    fflush(stdout);
    
    // ------------------------------------------------
    // 4. Patchify grid → patches (RGB) - full resolution
    // ------------------------------------------------
    patchify_cpu_safe(
        grid.data(), coords.data(),
        M, 3, H, W,
        m_patch_size / 2,
        patches);
    
    printf("[Patchifier] patchify_cpu_safe (patches) completed\n");
    fflush(stdout);

    printf("[Patchifier] About to create fmap_coords\n");
    fflush(stdout);
    
    // ------------------------------------------------
    // 5. Patchify fmap → gmap - scale coords to 1/4 resolution
    // ------------------------------------------------
    // fmap is at 1/4 resolution, so scale coordinates
    std::vector<float> fmap_coords(M * 2);
    for (int m = 0; m < M; m++)
    {
        fmap_coords[m * 2 + 0] = coords[m * 2 + 0] / RES; // Scale to 1/4 resolution
        fmap_coords[m * 2 + 1] = coords[m * 2 + 1] / RES; // Scale to 1/4 resolution
    }
    printf("[Patchifier] fmap_coords created\n");
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (gmap)\n");
    fflush(stdout);
    patchify_cpu_safe(
        fmap, fmap_coords.data(),
        M, 128, fmap_H, fmap_W, // Use 1/4 resolution dimensions
        m_patch_size / 2,
        gmap);
    printf("[Patchifier] patchify_cpu_safe (gmap) completed\n");
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (imap)\n");
    fflush(stdout);
    // ------------------------------------------------
    // 6. imap sampling (radius = 0) - scale coords to 1/4 resolution
    // Extract patches from the full feature map stored in m_imap_buffer
    // ------------------------------------------------
    // Use m_imap_buffer (full feature map [384, 120, 160]) as source
    // Extract patches and write to imap (which is [M, DIM] = [8, 384])
    if (m_fnet != nullptr && m_inet != nullptr) {
        // Extract patches from the full feature map
        patchify_cpu_safe(
            m_imap_buffer.data(), fmap_coords.data(),  // Source: full feature map, coords at 1/4 resolution
            M, inet_output_channels, fmap_H, fmap_W,   // M patches, 384 channels, 120x160 feature map
            0,                                          // radius = 0 (single pixel)
            imap                                        // Output: [M, DIM, 1, 1] = [8, 384, 1, 1]
        );
    } else {
        // Fallback: zero fill if models not available
        std::fill(imap, imap + M * m_DIM, 0.0f);
    }
    printf("[Patchifier] patchify_cpu_safe (imap) completed\n");
    fflush(stdout);

    printf("[Patchifier] About to extract colors\n");
    fflush(stdout);
    // ------------------------------------------------
    // 7. Color for visualization - full resolution
    // ------------------------------------------------
    for (int m = 0; m < M; m++)
    {
        int x = static_cast<int>(coords[m * 2 + 0]);
        int y = static_cast<int>(coords[m * 2 + 1]);
        for (int c = 0; c < 3; c++)
            clr[m * 3 + c] = image[c * H * W + y * W + x];
    }
    printf("[Patchifier] Colors extracted\n");
    fflush(stdout);
    
    printf("[Patchifier] forward() about to return\n");
    fflush(stdout);
}

// void Patchifier::forward(const uint8_t* image, int H, int W,
//                          float* fmap, float* imap, float* gmap,
//                          float* patches, uint8_t* clr,
//                          int patches_per_image) {

//     // 1. Fill fmap & imap with dummy features (replace with real network if needed)
//     int fmap_size = 128 * H * W;
//     int imap_size = m_DIM * H * W;
//     for (int i = 0; i < fmap_size; i++) fmap[i] = static_cast<float>(rand())/RAND_MAX;
//     for (int i = 0; i < imap_size; i++) imap[i] = static_cast<float>(rand())/RAND_MAX;

//     // 2. Extract patches from image and imap/gmap
//     extractPatches(image, H, W, patches, clr, patches_per_image);

//     // 3. Fill gmap with dummy patch features
//     for (int i = 0; i < patches_per_image * 128 * m_patch_size * m_patch_size; i++)
//         gmap[i] = static_cast<float>(rand())/RAND_MAX;
// }

// // Simple patch extraction (random centroids)
// void Patchifier::extractPatches(const uint8_t* image, int H, int W,
//                                 float* patches, uint8_t* clr,
//                                 int patches_per_image) {
//     for (int p = 0; p < patches_per_image; p++) {
//         int cx = rand() % (W - m_patch_size);
//         int cy = rand() % (H - m_patch_size);

//         // extract patch
//         for (int c = 0; c < 3; c++) {
//             for (int y = 0; y < m_patch_size; y++) {
//                 for (int x = 0; x < m_patch_size; x++) {
//                     int idx_img = (c * H + (cy+y)) * W + (cx+x);
//                     int idx_patch = (p*3 + c)*m_patch_size*m_patch_size + y*m_patch_size + x;
//                     patches[idx_patch] = static_cast<float>(image[idx_img]) / 255.0f;
//                 }
//             }

//             // color for patch
//             clr[p*3 + c] = image[(c*H + cy) * W + cx];
//         }
//     }
// }

// =================================================================================================
// DPVO Update Model Implementation
// =================================================================================================

DPVOUpdate::DPVOUpdate(Config_S *config, WakeCallback wakeFunc)
    : m_netBufferSize(1 * 384 * 768 * 1),
      m_inpBufferSize(1 * 384 * 768 * 1),
      m_corrBufferSize(1 * 882 * 768 * 1),
      m_iiBufferSize(1 * 768 * 1),
      m_jjBufferSize(1 * 768 * 1),
      m_kkBufferSize(1 * 768 * 1),
      m_netOutBufferSize(1 * 384 * 768 * 1),
      m_dOutBufferSize(1 * 2 * 768 * 1),
      m_wOutBufferSize(1 * 2 * 768 * 1),
      m_estimateTime(config->stShowProcTimeConfig.AIModel)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::syslog_logger_mt("dpvo_update", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
    auto logger = spdlog::stdout_color_mt("dpvo_update");
    logger->set_pattern("[%n] [%^%l%$] %v");
#endif

    logger->set_level(config->stDebugConfig.AIModel ? spdlog::level::debug : spdlog::level::info);

    // ==================================
    // (Ambarella CV28) Model Initialization
    // ==================================
#if defined(CV28) || defined(CV28_SIMULATOR)

    // Use updateModelPath if available, otherwise fallback to modelPath
    m_modelPathStr = !config->updateModelPath.empty() ? config->updateModelPath : config->modelPath;
    m_ptrModelPath = const_cast<char *>(m_modelPathStr.c_str());

    // Initialize network parameters
    ea_net_params_t net_params;
    memset(&net_params, 0, sizeof(net_params));

    // Set GPU ID to -1 to use CPU
    net_params.acinf_gpu_id = -1;

    // Create network instance
    m_model = ea_net_new(&net_params);
    if (m_model == NULL)
    {
        logger->error("Creating DPVO Update model failed");
    }

    m_inputNetTensor = NULL;
    m_inputInpTensor = NULL;
    m_inputCorrTensor = NULL;
    m_inputIiTensor = NULL;
    m_inputJjTensor = NULL;
    m_inputKkTensor = NULL;

    m_outputTensors = std::vector<ea_tensor_t *>(m_outputTensorList.size());

    // Allocate working buffers for input data
    m_netBuff = new float[m_netBufferSize];
    m_inpBuff = new float[m_inpBufferSize];
    m_corrBuff = new float[m_corrBufferSize];
    m_iiBuff = new int32_t[m_iiBufferSize];
    m_jjBuff = new int32_t[m_jjBufferSize];
    m_kkBuff = new int32_t[m_kkBufferSize];

    // Allocate working buffers for output data
    m_netOutBuff = new float[m_netOutBufferSize];
    m_dOutBuff = new float[m_dOutBufferSize];
    m_wOutBuff = new float[m_wOutBufferSize];
#endif
    // ==================================

    m_wakeFunc = wakeFunc;

    // Init Model Input/Output Tensor
    _initModelIO();
}

bool DPVOUpdate::_releaseModel()
{
    if (m_model)
    {
        ea_net_free(m_model);
        m_model = NULL;
    }

    return true;
}

bool DPVOUpdate::createDirectory(const std::string &path)
{
    return mkdir(path.c_str(), 0755) == 0; // Create directory with rwxr-xr-x permissions
}

bool DPVOUpdate::directoryExists(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        return false; // Directory doesn't exist
    }
    return (info.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

DPVOUpdate::~DPVOUpdate()
{
#ifdef SPDLOG_USE_SYSLOG
    spdlog::drop("dpvo_update");
#else
    spdlog::drop("dpvo_update");
#endif

    // ==================================
    // Ambarella CV28
    // ==================================
#if defined(CV28) || defined(CV28_SIMULATOR)

    _releaseInputTensors();
    _releaseOutputTensors();
    _releaseTensorBuffers();
    _releaseModel();

#endif
    // ==================================
}

// ============================================
//               Tensor Settings
// ============================================
void DPVOUpdate::_initModelIO()
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("dpvo_update");
#else
    auto logger = spdlog::get("dpvo_update");
#endif

    // ==================================
    // (Ambarella CV28) Create Model Output Buffers
    // ==================================
#if defined(CV28) || defined(CV28_SIMULATOR)

    int rval = EA_SUCCESS;
    logger->info("-------------------------------------------");
    logger->info("Configure Model Input/Output");

    // Configure input tensors
    logger->info("-------------------------------------------");
    logger->info("Configure Input Tensors");
    logger->info("Input Net Name: {}", m_inputNetTensorName);
    rval = ea_net_config_input(m_model, m_inputNetTensorName.c_str());

    logger->info("Input Inp Name: {}", m_inputInpTensorName);
    rval = ea_net_config_input(m_model, m_inputInpTensorName.c_str());

    logger->info("Input Corr Name: {}", m_inputCorrTensorName);
    rval = ea_net_config_input(m_model, m_inputCorrTensorName.c_str());

    logger->info("Input Ii Name: {}", m_inputIiTensorName);
    rval = ea_net_config_input(m_model, m_inputIiTensorName.c_str());

    logger->info("Input Jj Name: {}", m_inputJjTensorName);
    rval = ea_net_config_input(m_model, m_inputJjTensorName.c_str());

    logger->info("Input Kk Name: {}", m_inputKkTensorName);
    rval = ea_net_config_input(m_model, m_inputKkTensorName.c_str());

    // Configure output tensors
    logger->info("-------------------------------------------");
    logger->info("Configure Output Tensors");

    for (size_t i = 0; i < m_outputTensorList.size(); ++i)
    {
        logger->info("Output Name: {}", m_outputTensorList[i]);
        rval = ea_net_config_output(m_model, m_outputTensorList[i].c_str());
    }

    // Configure model path
    logger->info("-------------------------------------------");
    logger->info("Load Model");
    logger->info("Model Path: {}", m_ptrModelPath);

    // Check if model path exists before loading
    if (m_ptrModelPath == nullptr || strlen(m_ptrModelPath) == 0)
    {
        logger->error("Model path is null or empty");
        return;
    }

    // Check if file exists
    FILE *file = fopen(m_ptrModelPath, "r");
    if (file == nullptr)
    {
        logger->error("Model file does not exist at path: {}", m_ptrModelPath);
        return;
    }
    fclose(file);

    logger->info("Model file exists, proceeding with loading");

    rval = ea_net_load(m_model, EA_NET_LOAD_FILE, (void *)m_ptrModelPath, 1 /*max_batch*/);

    // Get input tensors
    logger->info("-------------------------------------------");
    logger->info("Create Model Input Tensors");

    m_inputNetTensor = ea_net_input(m_model, m_inputNetTensorName.c_str());
    m_inputInpTensor = ea_net_input(m_model, m_inputInpTensorName.c_str());
    m_inputCorrTensor = ea_net_input(m_model, m_inputCorrTensorName.c_str());
    m_inputIiTensor = ea_net_input(m_model, m_inputIiTensorName.c_str());
    m_inputJjTensor = ea_net_input(m_model, m_inputJjTensorName.c_str());
    m_inputKkTensor = ea_net_input(m_model, m_inputKkTensorName.c_str());

    logger->info("Input Net Shape: {}x{}x{}x{}",
                 ea_tensor_shape(m_inputNetTensor)[EA_N],
                 ea_tensor_shape(m_inputNetTensor)[EA_H],
                 ea_tensor_shape(m_inputNetTensor)[EA_W],
                 ea_tensor_shape(m_inputNetTensor)[EA_C]);

    logger->info("Input Inp Shape: {}x{}x{}x{}",
                 ea_tensor_shape(m_inputInpTensor)[EA_N],
                 ea_tensor_shape(m_inputInpTensor)[EA_H],
                 ea_tensor_shape(m_inputInpTensor)[EA_W],
                 ea_tensor_shape(m_inputInpTensor)[EA_C]);

    logger->info("Input Corr Shape: {}x{}x{}x{}",
                 ea_tensor_shape(m_inputCorrTensor)[EA_N],
                 ea_tensor_shape(m_inputCorrTensor)[EA_H],
                 ea_tensor_shape(m_inputCorrTensor)[EA_W],
                 ea_tensor_shape(m_inputCorrTensor)[EA_C]);

    // Get output tensors
    logger->info("-------------------------------------------");
    logger->info("Create Model Output Tensors");
    m_outputTensors[0] = ea_net_output_by_index(m_model, 0);
    m_outputTensors[1] = ea_net_output_by_index(m_model, 1);
    m_outputTensors[2] = ea_net_output_by_index(m_model, 2);

    for (size_t i = 0; i < ea_net_output_num(m_model); i++)
    {
        const char *tensorName = static_cast<const char *>(ea_net_output_name(m_model, i));
        const size_t *tensorShape = static_cast<const size_t *>(ea_tensor_shape(ea_net_output_by_index(m_model, i)));
        size_t tensorSize = static_cast<size_t>(ea_tensor_size(ea_net_output_by_index(m_model, i)));

        std::string shapeStr = std::to_string(tensorShape[0]) + "x" +
                               std::to_string(tensorShape[1]) + "x" +
                               std::to_string(tensorShape[2]) + "x" +
                               std::to_string(tensorShape[3]);

        logger->info("Output Tensor Name: {}", tensorName);
        logger->info("Output Tensor Shape: ({})", shapeStr);
        logger->info("Output Tensor Size: {}", tensorSize);
    }

#endif
    // ==================================
    return;
}

// =================================================================================================
// Ambarella CV28 Tensor Functions
// =================================================================================================
#if defined(CV28) || defined(CV28_SIMULATOR)
bool DPVOUpdate::_checkSavedTensor(int frameIdx)
{
    auto logger = spdlog::get("dpvo_update");

    auto time_0 = std::chrono::high_resolution_clock::now();

    // Define file paths for each tensor
    std::string netOutFilePath = m_tensorPath + "/frame_" + std::to_string(frameIdx - 1) + "_tensor0.bin";
    std::string dOutFilePath = m_tensorPath + "/frame_" + std::to_string(frameIdx - 1) + "_tensor1.bin";
    std::string wOutFilePath = m_tensorPath + "/frame_" + std::to_string(frameIdx - 1) + "_tensor2.bin";

    // Function to load tensor data from a binary file
    auto loadTensorFromBinaryFile = [](const std::string &filePath, float *buffer, size_t size)
    {
        std::ifstream inFile(filePath, std::ios::binary);
        if (inFile.is_open())
        {
            inFile.read(reinterpret_cast<char *>(buffer), size * sizeof(float));
            inFile.close();
            return true;
        }
        else
        {
            std::cerr << "Failed to open file: " << filePath << std::endl;
            return false;
        }
    };

    // Check if tensor files exist and load them if they do
    std::ifstream netOutFile(netOutFilePath, std::ios::binary);
    std::ifstream dOutFile(dOutFilePath, std::ios::binary);
    std::ifstream wOutFile(wOutFilePath, std::ios::binary);

    if (netOutFile && loadTensorFromBinaryFile(netOutFilePath, m_netOutBuff, m_netOutBufferSize) &&
        dOutFile && loadTensorFromBinaryFile(dOutFilePath, m_dOutBuff, m_dOutBufferSize) &&
        wOutFile && loadTensorFromBinaryFile(wOutFilePath, m_wOutBuff, m_wOutBufferSize))
    {
        if (m_estimateTime)
        {
            auto time_1 = std::chrono::high_resolution_clock::now();
            logger->info("[_checkSavedTensor]: {} ms",
                         std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
        }

        return true; // true means the tensors are already saved in the specific path
    }

    return false;
}

#if defined(SAVE_OUTPUT_TENSOR)
bool DPVOUpdate::_saveOutputTensor(int frameIdx)
{
    auto logger = spdlog::get("dpvo_update");

    std::string netOutFilePath = m_tensorPath + "/frame_" + std::to_string(frameIdx - 1) + "_tensor0.bin";
    std::string dOutFilePath = m_tensorPath + "/frame_" + std::to_string(frameIdx - 1) + "_tensor1.bin";
    std::string wOutFilePath = m_tensorPath + "/frame_" + std::to_string(frameIdx - 1) + "_tensor2.bin";

    logger->debug("========================================");
    logger->debug("Model Frame Index: {}", frameIdx);
    logger->debug("========================================");

    // Save tensors to binary files
    auto saveTensorToBinaryFile = [](const std::string &filePath, float *buffer, size_t size)
    {
        std::ofstream outFile(filePath, std::ios::binary);
        if (outFile.is_open())
        {
            outFile.write(reinterpret_cast<char *>(buffer), size * sizeof(float));
            outFile.close();
        }
        else
        {
            std::cerr << "Failed to open file: " << filePath << std::endl;
        }
    };

    // Save each tensor to its corresponding binary file
    saveTensorToBinaryFile(netOutFilePath, m_pred.netOutBuff, m_netOutBufferSize);
    saveTensorToBinaryFile(dOutFilePath, m_pred.dOutBuff, m_dOutBufferSize);
    saveTensorToBinaryFile(wOutFilePath, m_pred.wOutBuff, m_wOutBufferSize);

    return true;
}
#endif

bool DPVOUpdate::_releaseInputTensors()
{
    if (m_inputNetTensor)
    {
        m_inputNetTensor = nullptr;
    }
    if (m_inputInpTensor)
    {
        m_inputInpTensor = nullptr;
    }
    if (m_inputCorrTensor)
    {
        m_inputCorrTensor = nullptr;
    }
    if (m_inputIiTensor)
    {
        m_inputIiTensor = nullptr;
    }
    if (m_inputJjTensor)
    {
        m_inputJjTensor = nullptr;
    }
    if (m_inputKkTensor)
    {
        m_inputKkTensor = nullptr;
    }
    return true;
}

bool DPVOUpdate::_releaseOutputTensors()
{
    for (size_t i = 0; i < m_outputTensorList.size(); i++)
    {
        if (m_outputTensors[i])
        {
            m_outputTensors[i] = nullptr;
        }
    }
    return true;
}

bool DPVOUpdate::_releaseTensorBuffers()
{
    // Release Input Buffers
    delete[] m_netBuff;
    delete[] m_inpBuff;
    delete[] m_corrBuff;
    delete[] m_iiBuff;
    delete[] m_jjBuff;
    delete[] m_kkBuff;

    // Release Output Buffers
    delete[] m_netOutBuff;
    delete[] m_dOutBuff;
    delete[] m_wOutBuff;

    m_netBuff = nullptr;
    m_inpBuff = nullptr;
    m_corrBuff = nullptr;
    m_iiBuff = nullptr;
    m_jjBuff = nullptr;
    m_kkBuff = nullptr;

    m_netOutBuff = nullptr;
    m_dOutBuff = nullptr;
    m_wOutBuff = nullptr;

    return true;
}
#endif
// =================================================================================================

// =================================================================================================
// Synchronous Inference (Public API)
// =================================================================================================
bool DPVOUpdate::runInference(float *netData, float *inpData, float *corrData,
                              int32_t *iiData, int32_t *jjData, int32_t *kkData,
                              int frameIdx, DPVOUpdate_Prediction &pred)
{
    // Reset prediction structure
    pred = DPVOUpdate_Prediction();

    // Call internal _run method
    if (!_run(netData, inpData, corrData, iiData, jjData, kkData, frameIdx))
    {
        return false;
    }

    // Copy results directly from m_pred to output
    pred.isProcessed = m_pred.isProcessed;
    pred.netOutBuff = m_pred.netOutBuff;
    pred.dOutBuff = m_pred.dOutBuff;
    pred.wOutBuff = m_pred.wOutBuff;

    // Clear m_pred buffers so they don't get double-freed
    m_pred.netOutBuff = nullptr;
    m_pred.dOutBuff = nullptr;
    m_pred.wOutBuff = nullptr;

    return true;
}

// =================================================================================================
// Inference Entrypoint (Internal)
// =================================================================================================
bool DPVOUpdate::_run(float *netData, float *inpData, float *corrData,
                      int32_t *iiData, int32_t *jjData, int32_t *kkData, int frameIdx)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("dpvo_update");
#else
    auto logger = spdlog::get("dpvo_update");
#endif

    auto time_0 = std::chrono::high_resolution_clock::now();
    auto time_1 = std::chrono::time_point<std::chrono::high_resolution_clock>{};
    auto time_2 = std::chrono::high_resolution_clock::now();

    logger->debug(" =========== Model Frame Index: {} ===========", frameIdx);

    if (m_estimateTime)
    {
        logger->info("[AI Processing Time]");
        logger->info("-----------------------------------------");
    }

    // ==================================
    // Ambarella CV28 Inference
    // ==================================
#if defined(CV28) || defined(CV28_SIMULATOR)

    int rval = EA_SUCCESS;

    m_bProcessed = false;

    // STEP 1: load input tensors
    if (!_loadInput(netData, inpData, corrData, iiData, jjData, kkData))
    {
        logger->error("Load Input Data Failed");
        return false;
    }

    // STEP 2: inference
    // STEP 2-1: check if the tensors are already saved, pass the inference
    if (_checkSavedTensor(frameIdx))
    {
        // Allocate memory for all output tensors
        m_pred.netOutBuff = new float[m_netOutBufferSize];
        m_pred.dOutBuff = new float[m_dOutBufferSize];
        m_pred.wOutBuff = new float[m_wOutBufferSize];

        // Copy output tensors to prediction buffers
        std::memcpy(m_pred.netOutBuff, (float *)m_netOutBuff, m_netOutBufferSize * sizeof(float));
        std::memcpy(m_pred.dOutBuff, (float *)m_dOutBuff, m_dOutBufferSize * sizeof(float));
        std::memcpy(m_pred.wOutBuff, (float *)m_wOutBuff, m_wOutBufferSize * sizeof(float));
    }
    else
    {
        // STEP 2-2: run inference using Ambarella's eazyai library
        if (m_estimateTime)
            time_1 = std::chrono::high_resolution_clock::now();

        if (EA_SUCCESS != ea_net_forward(m_model, 1))
        {
            _releaseInputTensors();
            _releaseOutputTensors();
            _releaseTensorBuffers();
            _releaseModel();
        }
        else
        {
            // Sync output tensors between VP and CPU
#if defined(CV28)
            rval = ea_tensor_sync_cache(m_outputTensors[0], EA_VP, EA_CPU);
            if (rval != EA_SUCCESS)
            {
                logger->error("Failed to sync output tensor 0");
            }
            rval = ea_tensor_sync_cache(m_outputTensors[1], EA_VP, EA_CPU);
            if (rval != EA_SUCCESS)
            {
                logger->error("Failed to sync output tensor 1");
            }
            rval = ea_tensor_sync_cache(m_outputTensors[2], EA_VP, EA_CPU);
            if (rval != EA_SUCCESS)
            {
                logger->error("Failed to sync output tensor 2");
            }
#endif

            m_outputTensors[0] = ea_net_output_by_index(m_model, 0);
            m_outputTensors[1] = ea_net_output_by_index(m_model, 1);
            m_outputTensors[2] = ea_net_output_by_index(m_model, 2);

            // Allocate memory for all output tensors
            m_pred.netOutBuff = new float[m_netOutBufferSize];
            m_pred.dOutBuff = new float[m_dOutBufferSize];
            m_pred.wOutBuff = new float[m_wOutBufferSize];

            // Copy output tensors to prediction buffers
            std::memcpy(m_pred.netOutBuff, (float *)ea_tensor_data(m_outputTensors[0]), m_netOutBufferSize * sizeof(float));
            std::memcpy(m_pred.dOutBuff, (float *)ea_tensor_data(m_outputTensors[1]), m_dOutBufferSize * sizeof(float));
            std::memcpy(m_pred.wOutBuff, (float *)ea_tensor_data(m_outputTensors[2]), m_wOutBufferSize * sizeof(float));

#if defined(SAVE_OUTPUT_TENSOR)
            _saveOutputTensor(frameIdx);
#endif
        }
    }

    m_bProcessed = true;

    logger->debug(" ============= Model Frame Index: {} =============", frameIdx);

    if (m_estimateTime)
    {
        time_2 = std::chrono::high_resolution_clock::now();
        logger->info("[Inference]: {} ms",
                     std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_1).count() / (1000.0 * 1000));
    }
#endif
    // ==================================

    time_2 = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_0);
    m_inferenceTime = static_cast<float>(nanoseconds.count()) / 1e9f;

    if (m_estimateTime)
    {
        logger->info("[Total]: {} ms", m_inferenceTime);
        logger->info("-----------------------------------------");
    }

    logger->debug("End AI Model Part");
    logger->debug("========================================");

    return true;
}

// =================================================================================================
// Load Inputs
// =================================================================================================
bool DPVOUpdate::_loadInput(float *netData, float *inpData, float *corrData,
                            int32_t *iiData, int32_t *jjData, int32_t *kkData)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("dpvo_update");
#else
    auto logger = spdlog::get("dpvo_update");
#endif

    auto time_0 = m_estimateTime ? std::chrono::high_resolution_clock::now()
                                 : std::chrono::time_point<std::chrono::high_resolution_clock>{};
    auto time_1 = std::chrono::time_point<std::chrono::high_resolution_clock>{};

    // Copy input data to working buffers
    std::memcpy(m_netBuff, netData, m_netBufferSize * sizeof(float));
    std::memcpy(m_inpBuff, inpData, m_inpBufferSize * sizeof(float));
    std::memcpy(m_corrBuff, corrData, m_corrBufferSize * sizeof(float));
    std::memcpy(m_iiBuff, iiData, m_iiBufferSize * sizeof(int32_t));
    std::memcpy(m_jjBuff, jjData, m_jjBufferSize * sizeof(int32_t));
    std::memcpy(m_kkBuff, kkData, m_kkBufferSize * sizeof(int32_t));

    // Copy data to input tensors
#if defined(CV28) || defined(CV28_SIMULATOR)
    // Copy float tensors
    std::memcpy(ea_tensor_data(m_inputNetTensor), m_netBuff, m_netBufferSize * sizeof(float));
    std::memcpy(ea_tensor_data(m_inputInpTensor), m_inpBuff, m_inpBufferSize * sizeof(float));
    std::memcpy(ea_tensor_data(m_inputCorrTensor), m_corrBuff, m_corrBufferSize * sizeof(float));

    // Copy int32 tensors
    std::memcpy(ea_tensor_data(m_inputIiTensor), m_iiBuff, m_iiBufferSize * sizeof(int32_t));
    std::memcpy(ea_tensor_data(m_inputJjTensor), m_jjBuff, m_jjBufferSize * sizeof(int32_t));
    std::memcpy(ea_tensor_data(m_inputKkTensor), m_kkBuff, m_kkBufferSize * sizeof(int32_t));
#endif

    if (m_estimateTime)
    {
        time_1 = std::chrono::high_resolution_clock::now();
        logger->info("[Load Input]: {} ms",
                     std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
    }

    return true;
}

void DPVOUpdate::updateTensorPath(const std::string &path)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("dpvo_update");
#else
    auto logger = spdlog::get("dpvo_update");
#endif

    m_tensorPath = path; // Assuming m_tensorPath is a member variable to store the path
    logger->info("Updated tensor path to: {}", m_tensorPath);
}
