#include "fnet.hpp"
#include "dla_config.hpp"
#include "logger.hpp"
#include "target_frame.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>

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

#if defined(CV28) || defined(CV28_SIMULATOR)
// Tensor-based _loadInput — uses OpenCV for preprocessing to match fnet_onnx.cpp exactly.
// CRITICAL: ea_cvt_color_resize (AMBA hardware) produces different pixels than OpenCV,
// causing wrong DPVO poses. Using OpenCV here ensures identical preprocessing to the
// ONNX path, which is proven to give correct poses.
bool FNetInference::_loadInput(ea_tensor_t* imgTensor)
{
    if (imgTensor == nullptr) {
        auto logger = spdlog::get("fnet");
        if (logger) logger->error("FNet: imgTensor is nullptr!");
        return false;
    }
    
    if (m_inputTensor == nullptr) {
        auto logger = spdlog::get("fnet");
        if (logger) logger->error("FNet: m_inputTensor is nullptr!");
        return false;
    }
    
    auto logger = spdlog::get("fnet");
    //     🧠 Why pitch exists
    // On hardware accelerators (like CV28 VP), memory is usually aligned to:
    // 16 bytes
    // 32 bytes
    // 64 bytes
    // 128 bytes
    // to improve DMA / cache performance.
    // So even if your image width is:
    // W = 640 pixels
    // Memory may be aligned to:
    // pitch = 672 bytes
    // |<---- 640 valid pixels ---->|<-- 32 padding -->|
//     Width = 5 pixels
// Height = 3
// Channels = 1
// But hardware aligns to 8 bytes.

// Memory layout becomes:
// Row 0: [ P0 P1 P2 P3 P4 X X X ]
// Row 1: [ P5 P6 P7 P8 P9 X X X ]
// Row 2: [ P10 P11 P12 P13 P14 X X X ]


// 🧠 Summary
// ea_tensor_pitch(imgTensor)
//     = number of bytes per row in memory
//     = width + alignment padding


// It ensures:

// ✔ correct row stepping
// ✔ correct DMA alignment
// ✔ correct CV28 hardware access

// Without it → corrupted tensor.


    // ── Step 1: Read source image from ea_tensor → cv::Mat ──
    // (Same approach as fnet_onnx.cpp lines 209-234)
#if defined(CV28)
    ea_tensor_sync_cache(imgTensor, EA_VP, EA_CPU);
#endif
    const size_t* src_shape = ea_tensor_shape(imgTensor);
    int srcH = static_cast<int>(src_shape[EA_H]);
    int srcW = static_cast<int>(src_shape[EA_W]);
    size_t src_pitch = ea_tensor_pitch(imgTensor);
    const uint8_t* src_data = static_cast<const uint8_t*>(ea_tensor_data(imgTensor));
    
    if (!src_data) {
        if (logger) logger->error("FNet: ea_tensor_data returned nullptr for imgTensor");
        return false;
    }
    
    // Convert NCHW tensor → HWC cv::Mat (BGR), handling pitch
    cv::Mat img_bgr(srcH, srcW, CV_8UC3);
    size_t src_pitch_elem = src_pitch / sizeof(uint8_t);  // pitch in uint8 elements
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < srcH; y++) {
            const uint8_t* row = src_data + (c * srcH + y) * src_pitch_elem;
            for (int x = 0; x < srcW; x++) {
                img_bgr.at<cv::Vec3b>(y, x)[c] = row[x];
            }
        }
    }
    
    // ── Step 2: Resize with OpenCV (matches Python DPVO / fnet_onnx.cpp) ──
    int dstH = m_inputHeight;
    int dstW = m_inputWidth;
    cv::Mat img_resized;
    if (srcH == dstH && srcW == dstW) {
        img_resized = img_bgr;
    } else {
        cv::resize(img_bgr, img_resized, cv::Size(dstW, dstH), 0, 0, cv::INTER_LINEAR);
    }
    
    // ── Step 3: BGR → RGB (matches fnet_onnx.cpp line 246) ──
    cv::Mat img_rgb;
    cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);
    
    // ── Step 4: Write uint8 RGB pixels into m_inputTensor (NCHW, pitch-aware) ──
    // The AMBA model has normalization baked in (mean=63.75, std=127.5),
    // so we write uint8 [0,255] values — the model handles normalization internally.
    uint8_t* dst_data = static_cast<uint8_t*>(ea_tensor_data(m_inputTensor));
    size_t dst_pitch = ea_tensor_pitch(m_inputTensor);
    size_t dst_pitch_elem = dst_pitch / sizeof(uint8_t);
    
    if (!dst_data) {
        if (logger) logger->error("FNet: ea_tensor_data returned nullptr for m_inputTensor");
        return false;
    }
    
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < dstH; y++) {
            uint8_t* dst_row = dst_data + (c * dstH + y) * dst_pitch_elem;
            for (int x = 0; x < dstW; x++) {
                dst_row[x] = img_rgb.at<cv::Vec3b>(y, x)[c];
            }
        }
    }
    
    // Sync cache: CPU → VP (we wrote on CPU, model runs on VP)
#if defined(CV28)
    ea_tensor_sync_cache(m_inputTensor, EA_CPU, EA_VP);
#endif
    
    if (logger) logger->info("FNet: _loadInput successful using OpenCV preprocessing (resize + BGR→RGB)");
    
    // ── Save preprocessed input tensor at TARGET_FRAME for debugging ──
    {
        static int s_fnet_load_frame = 0;
        if (TARGET_FRAME >= 0 && s_fnet_load_frame == TARGET_FRAME) {
            const size_t* in_shape = ea_tensor_shape(m_inputTensor);
            size_t in_C = in_shape[EA_C], in_H = in_shape[EA_H], in_W = in_shape[EA_W];
            
            if (logger) {
                logger->info("FNet: Preprocessed input tensor info:");
                logger->info("  Shape: C={}, H={}, W={}", in_C, in_H, in_W);
                logger->info("  Pitch: {} bytes", dst_pitch);
                logger->info("  First 8 bytes: [{}, {}, {}, {}, {}, {}, {}, {}]",
                             dst_data[0], dst_data[1], dst_data[2], dst_data[3],
                             dst_data[4], dst_data[5], dst_data[6], dst_data[7]);
            }
            
            // Ensure bin_file directory exists
            struct stat st;
            if (stat("bin_file", &st) != 0) {
                mkdir("bin_file", 0755);
            }
            
            // Save dense CHW tensor (strip pitch padding for clean comparison)
            std::string fn = "bin_file/amba_fnet_preprocessed_frame"
                             + std::to_string(TARGET_FRAME) + ".bin";
            std::ofstream f(fn, std::ios::binary);
            if (f.is_open()) {
                for (size_t c = 0; c < in_C; c++) {
                    for (size_t y = 0; y < in_H; y++) {
                        const uint8_t* row = dst_data + (c * in_H + y) * dst_pitch_elem;
                        f.write(reinterpret_cast<const char*>(row), in_W * sizeof(uint8_t));
                    }
                }
                f.close();
                if (logger) logger->info("  Saved dense {} bytes to {}", in_C * in_H * in_W, fn);
            }
            
            // Save metadata
            std::string mfn = "bin_file/amba_fnet_preprocessed_meta_frame"
                              + std::to_string(TARGET_FRAME) + ".txt";
            std::ofstream mf(mfn);
            if (mf.is_open()) {
                mf << "C=" << in_C << "\n";
                mf << "H=" << in_H << "\n";
                mf << "W=" << in_W << "\n";
                mf << "pitch_bytes=" << dst_pitch << "\n";
                mf << "total_bytes=" << in_C * in_H * in_W << "\n";
                mf.close();
                if (logger) logger->info("  Saved metadata to {}", mfn);
            }
        }
        s_fnet_load_frame++;
    }
    
    return true;
}
#endif
// Tensor-based runInference overload
bool FNetInference::runInference(ea_tensor_t* imgTensor, float* fmap_out)
{
    if (imgTensor == nullptr || fmap_out == nullptr) {
        return false;
    }
    
    auto logger = spdlog::get("fnet");
    
    if (!_loadInput(imgTensor)) {
        if (logger) logger->info("FNet: Load Input Data Failed (tensor)");
        return false;
    } else {
        if (logger) logger->info("FNet: Load Input Data successful (tensor)");
    }
    
    // Run inference (same as uint8_t* version)
    if (EA_SUCCESS != ea_net_forward(m_model, 1)) {
        if (logger) logger->info("FNet: Inference failed");
        return false;
    } else {
        if (logger) logger->info("\033[33mFNet: Inference successful\033[0m");
    }
    
    // Sync output tensor
#if defined(CV28)
    int rval = ea_tensor_sync_cache(m_outputTensor, EA_VP, EA_CPU);
    if (rval != EA_SUCCESS) {
        if (logger) logger->info("FNet: Failed to sync output tensor");
    }
#endif
    
    m_outputTensor = ea_net_output_by_index(m_model, 0);
    
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
        if (logger) logger->error("FNet: Tensor shape mismatch! Expected [1, {}, {}, {}], got [{}, {}, {}, {}]",
                     m_outputChannel, m_outputHeight, m_outputWidth,
                     tensor_N, tensor_C, tensor_H, tensor_W);
    }
    
    // Copy output using pitch-aware NCHW reading
    const int outH = m_outputHeight;
    const int outW = m_outputWidth;
    const int outC = m_outputChannel;
    
    float *tensor_data = (float *)ea_tensor_data(m_outputTensor);
    
    // CRITICAL: Get tensor pitch to handle row padding on CV28 hardware.
    // ea_tensor_pitch returns the row stride in BYTES. On the simulator it equals W * sizeof(float),
    // but on real CV28 hardware it may be larger due to memory alignment requirements.
    // Without using pitch, every row after the first is read from the wrong offset,
    // causing progressive data corruption across channels.
    size_t pitch_bytes = ea_tensor_pitch(m_outputTensor);
    size_t pitch_floats = pitch_bytes / sizeof(float);
    
    if (logger && tensor_data != nullptr) {
        logger->info("FNet: Tensor shape from ea_tensor_shape: N={}, C={}, H={}, W={}", 
                      tensor_N, tensor_C, tensor_H, tensor_W);
        logger->info("FNet: Output tensor pitch: {} bytes ({} floats), W={}", 
                      pitch_bytes, pitch_floats, outW);
        if (pitch_floats != static_cast<size_t>(outW)) {
            logger->warn("FNet: ⚠️  Pitch ({}) != W ({})! Row padding detected. Using pitch-aware copy.",
                         pitch_floats, outW);
        }
        logger->info("FNet: First 5 tensor values (raw): [{}, {}, {}, {}, {}]",
                     tensor_data[0], tensor_data[1], tensor_data[2], tensor_data[3], tensor_data[4]);
    }
    
    // Pitch-aware NCHW copy: each row has pitch_floats stride (>= outW) in memory
    for (int c = 0; c < outC; c++) {
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                // Pitch-aware NCHW indexing: row stride = pitch_floats, channel stride = outH * pitch_floats
                size_t tensor_idx = static_cast<size_t>(c) * outH * pitch_floats 
                                  + static_cast<size_t>(y) * pitch_floats + x;
                // Output layout: [C, H, W] dense (no padding)
                int dst_idx = c * outH * outW + y * outW + x;
                fmap_out[dst_idx] = tensor_data[tensor_idx];
            }
        }
    }
    
    return true;
}

int FNetInference::getInputHeight() const {
    return m_inputHeight;
}

int FNetInference::getInputWidth() const {
    return m_inputWidth;
}

int FNetInference::getOutputHeight() const {
    return m_outputHeight;
}

int FNetInference::getOutputWidth() const {
    return m_outputWidth;
}

