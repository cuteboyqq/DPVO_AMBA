#include "fnet.hpp"
#include "dla_config.hpp"
#include "logger.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// =================================================================================================
// FNet Inference Implementation
// =================================================================================================
FNetInference::FNetInference(Config_S *config)
{
    // Check if logger already exists (to avoid "logger already exists" error)
    auto logger = spdlog::get("fnet");
    if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("fnet", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
        logger = spdlog::stdout_color_mt("fnet");
        logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    }

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

bool FNetInference::runInference(const float *image, int H, int W, float *fmap_out)
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

bool FNetInference::_loadInput(const float *image, int H, int W)
{
    // Convert normalized float image [C, H, W] (channel-first, range [-0.5, 1.5]) to model input format [1, 3, H, W]
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
        auto logger = spdlog::get("fnet");
        if (logger) logger->error("FNet: m_inputTensor is nullptr!");
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
        auto logger = spdlog::get("fnet");
        if (logger) logger->error("FNet: ea_tensor_data returned nullptr!");
        return false;
    }
    std::memcpy(tensor_data, inputBuffer.data(), inputSize * sizeof(uint8_t));

    return true;
}
#endif

int FNetInference::getInputHeight() const {
#if defined(CV28) || defined(CV28_SIMULATOR)
    return m_inputHeight;
#else
    return 0;
#endif
}

int FNetInference::getInputWidth() const {
#if defined(CV28) || defined(CV28_SIMULATOR)
    return m_inputWidth;
#else
    return 0;
#endif
}

int FNetInference::getOutputHeight() const {
#if defined(CV28) || defined(CV28_SIMULATOR)
    return m_outputHeight;
#else
    return 0;
#endif
}

int FNetInference::getOutputWidth() const {
#if defined(CV28) || defined(CV28_SIMULATOR)
    return m_outputWidth;
#else
    return 0;
#endif
}

