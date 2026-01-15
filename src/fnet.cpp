#include "fnet.hpp"
#include "dla_config.hpp"
#include "logger.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>
#include <sstream>

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
    }else{
        logger->info("\033[33mFNet: Inference successful\033[0m");
    }

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
    const int outH = m_outputHeight;  // e.g., 132 (528/4)
    const int outW = m_outputWidth;   // e.g., 240 (960/4)
    const int outC = m_outputChannel; // 128

    // Copy from tensor directly to fmap_out (no upsampling needed)
    // Tensor layout: [N, C, H, W] = [1, 128, 132, 240]
    // Output layout: [C, H, W] = [128, 132, 240]
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
                fmap_out[dst_idx] = tensor_data[tensor_idx]; // Models already handle /4 internally, no need to divide
            }
        }
    }

    // Log output statistics
    const size_t total_size = static_cast<size_t>(outC) * outH * outW;
    int zero_count = 0;
    int nonzero_count = 0;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    
    // Sample first 20 values for logging
    std::vector<float> sample_values;
    const int sample_count = std::min(20, static_cast<int>(total_size));
    
    for (size_t i = 0; i < total_size; i++) {
        float val = fmap_out[i];
        if (val == 0.0f) {
            zero_count++;
        } else {
            nonzero_count++;
        }
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
        
        if (static_cast<int>(i) < sample_count) {
            sample_values.push_back(val);
        }
    }
    
    double mean = (total_size > 0) ? (sum / total_size) : 0.0;
    
    logger->info("FNet: Output statistics - size={}, zero_count={}, nonzero_count={}, min={}, max={}, mean={}", 
                 total_size, zero_count, nonzero_count, min_val, max_val, mean);
    
    // Format sample values as string
    std::ostringstream sample_oss;
    for (size_t i = 0; i < sample_values.size(); i++) {
        if (i > 0) sample_oss << ", ";
        sample_oss << sample_values[i];
    }
    logger->info("FNet: First {} sample values: [{}]", sample_count, sample_oss.str());

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
    // Process original uint8 image [C, H, W] (channel-first, range [0, 255])
    // AMBA CV28 model will handle normalization internally (std: 256, normalizes by dividing by 256)
    // Use ea_cvt_color_resize to resize (similar to YOLOv8::_preProcessingMemory)
    
    if (m_inputTensor == nullptr) {
        auto logger = spdlog::get("fnet");
        if (logger) logger->error("FNet: m_inputTensor is nullptr!");
        return false;
    }
    
    auto logger = spdlog::get("fnet");
    
    // Step 1: Create a temporary tensor from raw image data [C, H, W]
    // Tensor shape format: [C, H, W] (no batch dimension in shape array)
    const int C = m_inputChannel;  // 3 for RGB
    const size_t srcSize = C * H * W;
    
    // Step 2: Create temporary source tensor
    // Shape array: [N, C, H, W] = [1, C, H, W] (tensor format)
    size_t srcShape[4] = {1, static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W)};
    // Pitch: bytes per row = W * C (for channel-first format)
    // size_t srcPitch = W * C;
    size_t srcPitch = W * sizeof(uint8_t);
    
    ea_tensor_t* srcTensor = ea_tensor_new(EA_U8, srcShape, 0);
    // ea_tensor_t* srcTensor = ea_tensor_new(EA_U8, srcShape, 0);
    if (srcTensor == nullptr) {
        if (logger) logger->error("FNet: Failed to create source tensor!");
        return false;
    }
    
    // Step 3: Copy image data to tensor, converting RGB to BGR
    // Input image is RGB [C, H, W]: c=0=R, c=1=G, c=2=B
    // ea_cvt_color_resize with EA_COLOR_BGR2RGB expects BGR input: c=0=B, c=1=G, c=2=R
    // So we need to swap R and B channels when copying
    void* tensor_data = ea_tensor_data(srcTensor);
    if (tensor_data == nullptr) {
        if (logger) logger->error("FNet: Failed to get tensor data pointer!");
        ea_tensor_free(srcTensor);
        return false;
    }
    
    uint8_t* dst = static_cast<uint8_t*>(tensor_data);
    // Convert RGB to BGR by swapping channels 0 and 2
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            // Source RGB: c=0=R, c=1=G, c=2=B
            // Destination BGR: c=0=B, c=1=G, c=2=R
            int src_idx_base = y * W + x;
            int dst_idx_base = 0 * H * W + y * W + x;  // Channel 0 (B in BGR)
            
            // Copy B channel (source c=2) to destination c=0
            dst[dst_idx_base] = image[2 * H * W + src_idx_base];  // B
            
            // Copy G channel (source c=1) to destination c=1
            dst[1 * H * W + y * W + x] = image[1 * H * W + src_idx_base];  // G
            
            // Copy R channel (source c=0) to destination c=2
            dst[2 * H * W + y * W + x] = image[0 * H * W + src_idx_base];  // R
        }
    }
    
    // Step 4: Use ea_cvt_color_resize to resize (similar to YOLOv8::_preProcessingMemory)
    // srcTensor is now in BGR format, EA_COLOR_BGR2RGB will convert BGR->RGB and resize
    int rval = EA_SUCCESS;
#if defined(CV28)
    rval = ea_cvt_color_resize(srcTensor, m_inputTensor, EA_COLOR_BGR2RGB, EA_VP);
#elif defined(CV28_SIMULATOR)
    rval = ea_cvt_color_resize(srcTensor, m_inputTensor, EA_COLOR_BGR2RGB, EA_CPU);
#endif
    
    // Step 5: Free temporary tensor
    ea_tensor_free(srcTensor);
    
    if (rval != EA_SUCCESS) {
        if (logger) logger->error("FNet: ea_cvt_color_resize failed with error: {}", rval);
        return false;
    }
    
    if (logger) logger->info("FNet: _loadInput successful using ea_cvt_color_resize");
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

