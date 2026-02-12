#include "update.hpp"
#include "dla_config.hpp"
#include "logger.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <chrono>
#include <limits>
#include <spdlog/spdlog.h>

// =================================================================================================
// DPVO Update Model Implementation
// =================================================================================================

DPVOUpdate::DPVOUpdate(Config_S *config, WakeCallback wakeFunc)
    : m_maxEdge(360),  // Initialize max edge count (matches MAX_EDGES = 360)
      m_netBufferSize(1 * 384 * m_maxEdge * 1),
      m_inpBufferSize(1 * 384 * m_maxEdge * 1),
      m_corrBufferSize(1 * 882 * m_maxEdge * 1),
      m_iiBufferSize(1 * m_maxEdge * 1),
      m_jjBufferSize(1 * m_maxEdge * 1),
      m_kkBufferSize(1 * m_maxEdge * 1),
      m_netOutBufferSize(1 * 384 * m_maxEdge * 1),
      m_dOutBufferSize(1 * 2 * m_maxEdge * 1),
      m_wOutBufferSize(1 * 2 * m_maxEdge * 1),
      m_estimateTime(config->stShowProcTimeConfig.AIModel)
{
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
    m_iiBuff = new float[m_iiBufferSize];
    m_jjBuff = new float[m_jjBufferSize];
    m_kkBuff = new float[m_kkBufferSize];

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
    auto logger = spdlog::get("dpvo_update");
    if (!logger) {
        logger = spdlog::get("dpvo");
    }

    // ==================================
    // (Ambarella CV28) Create Model Output Buffers
    // ==================================
#if defined(CV28) || defined(CV28_SIMULATOR)

    int rval = EA_SUCCESS;

    // Configure input tensors
    rval = ea_net_config_input(m_model, m_inputNetTensorName.c_str());
    rval = ea_net_config_input(m_model, m_inputInpTensorName.c_str());
    rval = ea_net_config_input(m_model, m_inputCorrTensorName.c_str());
    rval = ea_net_config_input(m_model, m_inputIiTensorName.c_str());
    rval = ea_net_config_input(m_model, m_inputJjTensorName.c_str());
    rval = ea_net_config_input(m_model, m_inputKkTensorName.c_str());

    // Configure output tensors
    for (size_t i = 0; i < m_outputTensorList.size(); ++i)
    {
        rval = ea_net_config_output(m_model, m_outputTensorList[i].c_str());
    }

    // Check if model path exists before loading
    if (m_ptrModelPath == nullptr || strlen(m_ptrModelPath) == 0)
    {
        if (logger) logger->error("DPVOUpdate::_initModelIO: Model path is null or empty");
        return;
    }

    // Check if file exists
    FILE *file = fopen(m_ptrModelPath, "r");
    if (file == nullptr)
    {
        if (logger) logger->error("DPVOUpdate::_initModelIO: Model file does not exist at path: {}", m_ptrModelPath);
        return;
    }
    fclose(file);

    rval = ea_net_load(m_model, EA_NET_LOAD_FILE, (void *)m_ptrModelPath, 1 /*max_batch*/);
    if (rval != EA_SUCCESS) {
        if (logger) logger->error("DPVOUpdate::_initModelIO: Failed to load model from '{}', rval={}", m_ptrModelPath, rval);
        return;
    }

    // Get input tensors
    m_inputNetTensor = ea_net_input(m_model, m_inputNetTensorName.c_str());
    m_inputInpTensor = ea_net_input(m_model, m_inputInpTensorName.c_str());
    m_inputCorrTensor = ea_net_input(m_model, m_inputCorrTensorName.c_str());
    m_inputIiTensor = ea_net_input(m_model, m_inputIiTensorName.c_str());
    m_inputJjTensor = ea_net_input(m_model, m_inputJjTensorName.c_str());
    m_inputKkTensor = ea_net_input(m_model, m_inputKkTensorName.c_str());

    // Validate input tensors were retrieved successfully
    if (m_inputNetTensor == nullptr || m_inputInpTensor == nullptr || m_inputCorrTensor == nullptr ||
        m_inputIiTensor == nullptr || m_inputJjTensor == nullptr || m_inputKkTensor == nullptr) {
        if (logger) logger->error("DPVOUpdate::_initModelIO: Failed to get one or more input tensors");
        return;
    }

    // Get output tensors
    m_outputTensors[0] = ea_net_output_by_index(m_model, 0);
    m_outputTensors[1] = ea_net_output_by_index(m_model, 1);
    m_outputTensors[2] = ea_net_output_by_index(m_model, 2);

#endif
    // ==================================
    return;
}


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

// =================================================================================================

// =================================================================================================
// Synchronous Inference (Public API)
// =================================================================================================
bool DPVOUpdate::runInference(float *netData, float *inpData, float *corrData,
                              float *iiData, float *jjData, float *kkData,
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
                      float *iiData, float *jjData, float *kkData, int frameIdx)
{
    auto logger = spdlog::get("dpvo_update");
    if (!logger) {
        logger = spdlog::get("dpvo");
    }

    auto time_0 = std::chrono::high_resolution_clock::now();
    auto time_1 = std::chrono::time_point<std::chrono::high_resolution_clock>{};
    auto time_2 = std::chrono::high_resolution_clock::now();

    // ==================================
    // Ambarella CV28 Inference
    // ==================================
#if defined(CV28) || defined(CV28_SIMULATOR)

    int rval = EA_SUCCESS;

    m_bProcessed = false;

    // STEP 1: load input tensors
    if (!_loadInput(netData, inpData, corrData, iiData, jjData, kkData))
    {
        if (logger) logger->error("DPVOUpdate::_run: _loadInput failed");
        return false;
    }

    // STEP 2: run inference using Ambarella's eazyai library
    {
        if (m_estimateTime)
            time_1 = std::chrono::high_resolution_clock::now();

        // Validate model before calling forward
        if (m_model == nullptr) {
            if (logger) logger->error("DPVOUpdate::_run: m_model is null");
            return false;
        }
        
        // Validate input tensors are still valid
        if (m_inputNetTensor == nullptr || m_inputInpTensor == nullptr || m_inputCorrTensor == nullptr ||
            m_inputIiTensor == nullptr || m_inputJjTensor == nullptr || m_inputKkTensor == nullptr) {
            if (logger) logger->error("DPVOUpdate::_run: Input tensors are null");
            return false;
        }
        
        
        int forward_result = ea_net_forward(m_model, 1);
        
        if (EA_SUCCESS != forward_result)
        {
            if (logger) logger->error("DPVOUpdate::_run: ea_net_forward failed with error code: {}", forward_result);
            return false;
        } else {
            if (logger) logger->info("\033[33mDPVOUpdate: Inference successful\033[0m");
        }
        
        // Sync output tensors between VP and CPU (sync existing tensors from initialization, matching YOLOv8 pattern)
#if defined(CV28)
        if (m_outputTensors[0] != nullptr) {
            rval = ea_tensor_sync_cache(m_outputTensors[0], EA_VP, EA_CPU);
            if (rval != EA_SUCCESS && logger) {
                logger->error("DPVOUpdate::_run: Failed to sync output tensor 0");
            }
        }
        if (m_outputTensors[1] != nullptr) {
            rval = ea_tensor_sync_cache(m_outputTensors[1], EA_VP, EA_CPU);
            if (rval != EA_SUCCESS && logger) {
                logger->error("DPVOUpdate::_run: Failed to sync output tensor 1");
            }
        }
        if (m_outputTensors[2] != nullptr) {
            rval = ea_tensor_sync_cache(m_outputTensors[2], EA_VP, EA_CPU);
            if (rval != EA_SUCCESS && logger) {
                logger->error("DPVOUpdate::_run: Failed to sync output tensor 2");
            }
        }
#endif

        // Get output tensors AFTER forward pass (re-retrieve them, matching YOLOv8 pattern)
        m_outputTensors[0] = ea_net_output_by_index(m_model, 0);
        m_outputTensors[1] = ea_net_output_by_index(m_model, 1);
        m_outputTensors[2] = ea_net_output_by_index(m_model, 2);

        // Validate output tensors
        if (m_outputTensors[0] == nullptr || m_outputTensors[1] == nullptr || m_outputTensors[2] == nullptr) {
            if (logger) logger->error("DPVOUpdate::_run: Output tensors are null after retrieval");
            return false;
        }

        // Allocate memory for all output tensors
        m_pred.netOutBuff = new float[m_netOutBufferSize];
        m_pred.dOutBuff = new float[m_dOutBufferSize];
        m_pred.wOutBuff = new float[m_wOutBufferSize];

        // Get tensor data pointers and validate
        void* tensor0_data = ea_tensor_data(m_outputTensors[0]);
        void* tensor1_data = ea_tensor_data(m_outputTensors[1]);
        void* tensor2_data = ea_tensor_data(m_outputTensors[2]);
        
        if (tensor0_data == nullptr || tensor1_data == nullptr || tensor2_data == nullptr) {
            if (logger) logger->error("DPVOUpdate::_run: Output tensor data pointers are null");
            delete[] m_pred.netOutBuff;
            delete[] m_pred.dOutBuff;
            delete[] m_pred.wOutBuff;
            m_pred.netOutBuff = nullptr;
            m_pred.dOutBuff = nullptr;
            m_pred.wOutBuff = nullptr;
            return false;
        }

        // Diagnostic: Check tensor shapes and raw data before copying
        if (logger) {
            const size_t* shape0 = ea_tensor_shape(m_outputTensors[0]);
            const size_t* shape1 = ea_tensor_shape(m_outputTensors[1]);
            const size_t* shape2 = ea_tensor_shape(m_outputTensors[2]);
            
            logger->info("\033[35mDPVOUpdate::_run: Output tensor shapes - tensor0=[{}x{}x{}x{}], tensor1=[{}x{}x{}x{}], tensor2=[{}x{}x{}x{}]\033[0m",
                         shape0[EA_N], shape0[EA_C], shape0[EA_H], shape0[EA_W],
                         shape1[EA_N], shape1[EA_C], shape1[EA_H], shape1[EA_W],
                         shape2[EA_N], shape2[EA_C], shape2[EA_H], shape2[EA_W]);
            
            // Check raw weight tensor data (tensor2) using pitch-aware indexing
            {
                size_t w_pitch_bytes = ea_tensor_pitch(m_outputTensors[2]);
                size_t w_pitch_floats = w_pitch_bytes / sizeof(float);
                int w_C = static_cast<int>(shape2[EA_C]);
                int w_H = static_cast<int>(shape2[EA_H]);
                int w_W = static_cast<int>(shape2[EA_W]);
                const float* raw_wOut = static_cast<const float*>(tensor2_data);
                
                logger->info("\033[35mDPVOUpdate::_run: Weight tensor pitch={} bytes ({} floats), W={}\033[0m",
                             w_pitch_bytes, w_pitch_floats, w_W);
                
                float raw_w_min = std::numeric_limits<float>::max();
                float raw_w_max = std::numeric_limits<float>::lowest();
                size_t raw_w_zero_count = 0;
                size_t raw_w_nonzero_count = 0;
                for (int c = 0; c < w_C; c++) {
                    for (int h = 0; h < w_H; h++) {
                        for (int w = 0; w < w_W; w++) {
                            size_t idx = static_cast<size_t>(c) * w_H * w_pitch_floats
                                       + static_cast<size_t>(h) * w_pitch_floats + w;
                            float val = raw_wOut[idx];
                            if (val < raw_w_min) raw_w_min = val;
                            if (val > raw_w_max) raw_w_max = val;
                            if (val == 0.0f) raw_w_zero_count++;
                            else raw_w_nonzero_count++;
                        }
                    }
                }
                logger->info("\033[35mDPVOUpdate::_run: Raw weight tensor (tensor2) BEFORE copy - elements={}, range=[{}, {}], zero_count={}, nonzero_count={}\033[0m",
                             w_C * w_H * w_W, raw_w_min, raw_w_max, raw_w_zero_count, raw_w_nonzero_count);
                
                // Show first 10 actual values (pitch-aware)
                int shown = 0;
                std::vector<float> first_vals;
                for (int c = 0; c < w_C && shown < 10; c++) {
                    for (int h = 0; h < w_H && shown < 10; h++) {
                        for (int w = 0; w < w_W && shown < 10; w++) {
                            size_t idx = static_cast<size_t>(c) * w_H * w_pitch_floats
                                       + static_cast<size_t>(h) * w_pitch_floats + w;
                            first_vals.push_back(raw_wOut[idx]);
                            shown++;
                        }
                    }
                }
                while (first_vals.size() < 10) first_vals.push_back(0.0f);
                logger->info("\033[35mDPVOUpdate::_run: Raw weight tensor first 10 values (pitch-aware): [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]\033[0m",
                             first_vals[0], first_vals[1], first_vals[2], first_vals[3], first_vals[4],
                             first_vals[5], first_vals[6], first_vals[7], first_vals[8], first_vals[9]);
                
                if (raw_w_nonzero_count == 0) {
                    logger->warn("\033[33mDPVOUpdate::_run: WARNING - Model is outputting all zeros for weights (tensor2).\033[0m");
                }
            }
        }

        // CRITICAL: Use pitch-aware copying for ALL output tensors.
        // Same pitch issue as inputs — W=1 tensors have padded rows on CV28.
        {
            // Helper lambda: pitch-aware copy from ea_tensor (NCHW) → dense buffer
            auto pitchAwareCopyFromTensor = [&logger](ea_tensor_t* tensor, const void* raw_data, float* dst, size_t expected_size, const char* name) {
                const size_t* shape = ea_tensor_shape(tensor);
                int C = static_cast<int>(shape[EA_C]);
                int H = static_cast<int>(shape[EA_H]);
                int W = static_cast<int>(shape[EA_W]);
                size_t pitch_bytes = ea_tensor_pitch(tensor);
                size_t pitch_floats = pitch_bytes / sizeof(float);
                const float* src = static_cast<const float*>(raw_data);
                
                if (logger) {
                    logger->info("DPVOUpdate::_run: {} output tensor shape=[{},{},{}], pitch={} bytes ({} floats), W={}",
                                 name, C, H, W, pitch_bytes, pitch_floats, W);
                }
                
                if (pitch_floats == static_cast<size_t>(W)) {
                    // No padding — fast path: direct memcpy
                    std::memcpy(dst, src, expected_size * sizeof(float));
                } else {
                    // Pitch-aware copy: read W elements per row, skip padding
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            size_t tensor_row = static_cast<size_t>(c) * H * pitch_floats
                                              + static_cast<size_t>(h) * pitch_floats;
                            int dst_row = c * H * W + h * W;
                            for (int w = 0; w < W; w++) {
                                dst[dst_row + w] = src[tensor_row + w];
                            }
                        }
                    }
                }
            };
            
            pitchAwareCopyFromTensor(m_outputTensors[0], tensor0_data, m_pred.netOutBuff, m_netOutBufferSize, "net_out");
            pitchAwareCopyFromTensor(m_outputTensors[1], tensor1_data, m_pred.dOutBuff,   m_dOutBufferSize,   "d_out");
            pitchAwareCopyFromTensor(m_outputTensors[2], tensor2_data, m_pred.wOutBuff,   m_wOutBufferSize,   "w_out");
        }

        // Log delta and weight output values with blue text
        if (logger) {
            // Calculate statistics for delta values (dOutBuff: [1, 2, m_maxEdge, 1])
            float* dOut = m_pred.dOutBuff;
            float d_min = std::numeric_limits<float>::max();
            float d_max = std::numeric_limits<float>::lowest();
            float d_sum = 0.0f;
            size_t d_zero_count = 0;
            size_t d_nonzero_count = 0;
            
            for (size_t i = 0; i < m_dOutBufferSize; i++) {
                float val = dOut[i];
                if (val < d_min) d_min = val;
                if (val > d_max) d_max = val;
                d_sum += val;
                if (val == 0.0f) d_zero_count++;
                else d_nonzero_count++;
            }
            float d_mean = (m_dOutBufferSize > 0) ? d_sum / m_dOutBufferSize : 0.0f;
            
            // Calculate statistics for weight values (wOutBuff: [1, 2, m_maxEdge, 1])
            float* wOut = m_pred.wOutBuff;
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            float w_sum = 0.0f;
            size_t w_zero_count = 0;
            size_t w_nonzero_count = 0;
            
            for (size_t i = 0; i < m_wOutBufferSize; i++) {
                float val = wOut[i];
                if (val < w_min) w_min = val;
                if (val > w_max) w_max = val;
                w_sum += val;
                if (val == 0.0f) w_zero_count++;
                else w_nonzero_count++;
            }
            float w_mean = (m_wOutBufferSize > 0) ? w_sum / m_wOutBufferSize : 0.0f;
            
            // Log with blue text (\033[34m for blue, \033[0m to reset)
            logger->info("\033[34mDPVOUpdate::_run: Delta output (dOut) - size={}, range=[{}, {}], mean={:.6f}, zero_count={}, nonzero_count={}\033[0m",
                         m_dOutBufferSize, d_min, d_max, d_mean, d_zero_count, d_nonzero_count);
            logger->info("\033[34mDPVOUpdate::_run: Weight output (wOut) - size={}, range=[{}, {}], mean={:.6f}, zero_count={}, nonzero_count={}\033[0m",
                         m_wOutBufferSize, w_min, w_max, w_mean, w_zero_count, w_nonzero_count);
            
            // Additional debugging: Check if weights are in a different channel or need processing
            // Check first few values from both channels
            if (m_wOutBufferSize >= 20) {
                logger->info("\033[34mDPVOUpdate::_run: Weight tensor sample - channel0[0-4]: [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}], "
                            "channel1[0-4]: [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]\033[0m",
                            wOut[0], wOut[1], wOut[2], wOut[3], wOut[4],
                            (m_wOutBufferSize > m_maxEdge ? wOut[m_maxEdge] : 0.0f),
                            (m_wOutBufferSize > m_maxEdge + 1 ? wOut[m_maxEdge + 1] : 0.0f),
                            (m_wOutBufferSize > m_maxEdge + 2 ? wOut[m_maxEdge + 2] : 0.0f),
                            (m_wOutBufferSize > m_maxEdge + 3 ? wOut[m_maxEdge + 3] : 0.0f),
                            (m_wOutBufferSize > m_maxEdge + 4 ? wOut[m_maxEdge + 4] : 0.0f));
            }
            
            // Show sample values for first few edges (each edge has 2 delta values and 2 weight values)
            const int num_sample_edges = std::min(5, static_cast<int>(m_maxEdge));
            logger->info("\033[34mDPVOUpdate::_run: Sample delta and weight values (first {} edges):\033[0m", num_sample_edges);
            for (int e = 0; e < num_sample_edges; e++) {
                // dOut shape: [1, 2, m_maxEdge, 1] -> index = 0*2*m_maxEdge*1 + c*1*m_maxEdge*1 + e*1 + 0 = c*m_maxEdge + e
                // For c=0: idx = e, for c=1: idx = m_maxEdge + e
                size_t d_idx0 = e;  // First delta value for edge e
                size_t d_idx1 = m_maxEdge + e;  // Second delta value for edge e
                size_t w_idx0 = e;  // First weight value for edge e
                size_t w_idx1 = m_maxEdge + e;  // Second weight value for edge e
                
                logger->info("\033[34m  Edge[{}]: delta=[{:.6f}, {:.6f}], weight=[{:.6f}, {:.6f}]\033[0m",
                             e, 
                             (d_idx0 < m_dOutBufferSize ? dOut[d_idx0] : 0.0f),
                             (d_idx1 < m_dOutBufferSize ? dOut[d_idx1] : 0.0f),
                             (w_idx0 < m_wOutBufferSize ? wOut[w_idx0] : 0.0f),
                             (w_idx1 < m_wOutBufferSize ? wOut[w_idx1] : 0.0f));
            }
        }

    }

    m_bProcessed = true;

    if (m_estimateTime)
    {
        time_2 = std::chrono::high_resolution_clock::now();
    }
#endif
    // ==================================

    time_2 = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_0);
    m_inferenceTime = static_cast<float>(nanoseconds.count()) / 1e9f;

    return true;
}

// =================================================================================================
// Load Inputs
// =================================================================================================
bool DPVOUpdate::_loadInput(float *netData, float *inpData, float *corrData,
                            float *iiData, float *jjData, float *kkData)
{
    auto logger = spdlog::get("dpvo_update");
    if (!logger) {
        logger = spdlog::get("dpvo");
    }

    auto time_0 = m_estimateTime ? std::chrono::high_resolution_clock::now()
                                 : std::chrono::time_point<std::chrono::high_resolution_clock>{};
    auto time_1 = std::chrono::time_point<std::chrono::high_resolution_clock>{};

    // Validate input data pointers
    if (netData == nullptr || inpData == nullptr || corrData == nullptr ||
        iiData == nullptr || jjData == nullptr || kkData == nullptr) {
        if (logger) logger->error("DPVOUpdate::_loadInput: One or more input data pointers are null");
        return false;
    }
    
    // Copy input data to working buffers
    std::memcpy(m_netBuff, netData, m_netBufferSize * sizeof(float));
    std::memcpy(m_inpBuff, inpData, m_inpBufferSize * sizeof(float));
    std::memcpy(m_corrBuff, corrData, m_corrBufferSize * sizeof(float));
    std::memcpy(m_iiBuff, iiData, m_iiBufferSize * sizeof(float));
    std::memcpy(m_jjBuff, jjData, m_jjBufferSize * sizeof(float));
    std::memcpy(m_kkBuff, kkData, m_kkBufferSize * sizeof(float));

    // Log input data statistics and first 10 values
    if (logger) {
        // Calculate statistics for net input
        size_t net_zero_count = 0;
        size_t net_nonzero_count = 0;
        for (size_t i = 0; i < m_netBufferSize; i++) {
            if (m_netBuff[i] == 0.0f) net_zero_count++;
            else net_nonzero_count++;
        }
        
        // Calculate statistics for inp input
        size_t inp_zero_count = 0;
        size_t inp_nonzero_count = 0;
        for (size_t i = 0; i < m_inpBufferSize; i++) {
            if (m_inpBuff[i] == 0.0f) inp_zero_count++;
            else inp_nonzero_count++;
        }
        
        // Calculate statistics for corr input
        size_t corr_zero_count = 0;
        size_t corr_nonzero_count = 0;
        for (size_t i = 0; i < m_corrBufferSize; i++) {
            if (m_corrBuff[i] == 0.0f) corr_zero_count++;
            else corr_nonzero_count++;
        }
        
        // Log statistics with green text (\033[32m for green, \033[0m to reset)
        logger->info("\033[32mDPVOUpdate::_loadInput: net input - size={}, zero_count={}, nonzero_count={}\033[0m",
                     m_netBufferSize, net_zero_count, net_nonzero_count);
        logger->info("\033[32mDPVOUpdate::_loadInput: inp input - size={}, zero_count={}, nonzero_count={}\033[0m",
                     m_inpBufferSize, inp_zero_count, inp_nonzero_count);
        logger->info("\033[32mDPVOUpdate::_loadInput: corr input - size={}, zero_count={}, nonzero_count={}\033[0m",
                     m_corrBufferSize, corr_zero_count, corr_nonzero_count);
        
        // Log first 10 values with green text
        const int num_samples = 10;
        logger->info("\033[32mDPVOUpdate::_loadInput: First {} net values: [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]\033[0m",
                     num_samples,
                     (m_netBufferSize > 0 ? m_netBuff[0] : 0.0f),
                     (m_netBufferSize > 1 ? m_netBuff[1] : 0.0f),
                     (m_netBufferSize > 2 ? m_netBuff[2] : 0.0f),
                     (m_netBufferSize > 3 ? m_netBuff[3] : 0.0f),
                     (m_netBufferSize > 4 ? m_netBuff[4] : 0.0f),
                     (m_netBufferSize > 5 ? m_netBuff[5] : 0.0f),
                     (m_netBufferSize > 6 ? m_netBuff[6] : 0.0f),
                     (m_netBufferSize > 7 ? m_netBuff[7] : 0.0f),
                     (m_netBufferSize > 8 ? m_netBuff[8] : 0.0f),
                     (m_netBufferSize > 9 ? m_netBuff[9] : 0.0f));
        
        logger->info("\033[32mDPVOUpdate::_loadInput: First {} inp values: [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]\033[0m",
                     num_samples,
                     (m_inpBufferSize > 0 ? m_inpBuff[0] : 0.0f),
                     (m_inpBufferSize > 1 ? m_inpBuff[1] : 0.0f),
                     (m_inpBufferSize > 2 ? m_inpBuff[2] : 0.0f),
                     (m_inpBufferSize > 3 ? m_inpBuff[3] : 0.0f),
                     (m_inpBufferSize > 4 ? m_inpBuff[4] : 0.0f),
                     (m_inpBufferSize > 5 ? m_inpBuff[5] : 0.0f),
                     (m_inpBufferSize > 6 ? m_inpBuff[6] : 0.0f),
                     (m_inpBufferSize > 7 ? m_inpBuff[7] : 0.0f),
                     (m_inpBufferSize > 8 ? m_inpBuff[8] : 0.0f),
                     (m_inpBufferSize > 9 ? m_inpBuff[9] : 0.0f));
        
        logger->info("\033[32mDPVOUpdate::_loadInput: First {} corr values: [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]\033[0m",
                     num_samples,
                     (m_corrBufferSize > 0 ? m_corrBuff[0] : 0.0f),
                     (m_corrBufferSize > 1 ? m_corrBuff[1] : 0.0f),
                     (m_corrBufferSize > 2 ? m_corrBuff[2] : 0.0f),
                     (m_corrBufferSize > 3 ? m_corrBuff[3] : 0.0f),
                     (m_corrBufferSize > 4 ? m_corrBuff[4] : 0.0f),
                     (m_corrBufferSize > 5 ? m_corrBuff[5] : 0.0f),
                     (m_corrBufferSize > 6 ? m_corrBuff[6] : 0.0f),
                     (m_corrBufferSize > 7 ? m_corrBuff[7] : 0.0f),
                     (m_corrBufferSize > 8 ? m_corrBuff[8] : 0.0f),
                     (m_corrBufferSize > 9 ? m_corrBuff[9] : 0.0f));
    }

    // Copy data to input tensors
#if defined(CV28) || defined(CV28_SIMULATOR)
    // Validate input tensors before using them
    if (m_inputNetTensor == nullptr || m_inputInpTensor == nullptr || m_inputCorrTensor == nullptr ||
        m_inputIiTensor == nullptr || m_inputJjTensor == nullptr || m_inputKkTensor == nullptr) {
        if (logger) logger->error("DPVOUpdate::_loadInput: One or more input tensors are null");
        return false;
    }
    
    // Get tensor data pointers for writing (need to sync cache manually after writing)
    void* net_data = ea_tensor_data(m_inputNetTensor);
    void* inp_data = ea_tensor_data(m_inputInpTensor);
    void* corr_data = ea_tensor_data(m_inputCorrTensor);
    void* ii_data = ea_tensor_data(m_inputIiTensor);
    void* jj_data = ea_tensor_data(m_inputJjTensor);
    void* kk_data = ea_tensor_data(m_inputKkTensor);
    
    if (net_data == nullptr || inp_data == nullptr || corr_data == nullptr ||
        ii_data == nullptr || jj_data == nullptr || kk_data == nullptr) {
        if (logger) logger->error("DPVOUpdate::_loadInput: One or more tensor data pointers are null");
        return false;
    }
    

    // CRITICAL: Use pitch-aware copying for ALL input tensors.
    // AMBA CV28 pads tensor rows to 32-byte alignment. For W=1 tensors (like [1,384,360,1]),
    // ea_tensor_pitch returns 32 bytes (8 floats) instead of 4 bytes (1 float).
    // Using raw memcpy would write data into padding bytes, corrupting the input.
    {
        // Helper lambda: pitch-aware copy from dense buffer → ea_tensor (NCHW)
        auto pitchAwareCopyToTensor = [&logger](ea_tensor_t* tensor, const float* src, size_t expected_size, const char* name) {
            const size_t* shape = ea_tensor_shape(tensor);
            int C = static_cast<int>(shape[EA_C]);
            int H = static_cast<int>(shape[EA_H]);
            int W = static_cast<int>(shape[EA_W]);
            size_t pitch_bytes = ea_tensor_pitch(tensor);
            size_t pitch_floats = pitch_bytes / sizeof(float);
            float* dst = static_cast<float*>(ea_tensor_data(tensor));
            
            if (logger) {
                logger->info("DPVOUpdate::_loadInput: {} tensor shape=[{},{},{}], pitch={} bytes ({} floats), W={}",
                             name, C, H, W, pitch_bytes, pitch_floats, W);
            }
            
            if (pitch_floats == static_cast<size_t>(W)) {
                // No padding — fast path: direct memcpy
                std::memcpy(dst, src, expected_size * sizeof(float));
            } else {
                // Pitch-aware copy: write W elements per row, skip padding
                for (int c = 0; c < C; c++) {
                    for (int h = 0; h < H; h++) {
                        size_t tensor_row = static_cast<size_t>(c) * H * pitch_floats
                                          + static_cast<size_t>(h) * pitch_floats;
                        int src_row = c * H * W + h * W;
                        for (int w = 0; w < W; w++) {
                            dst[tensor_row + w] = src[src_row + w];
                        }
                    }
                }
            }
        };
        
        pitchAwareCopyToTensor(m_inputNetTensor,  m_netBuff,  m_netBufferSize,  "net");
        pitchAwareCopyToTensor(m_inputInpTensor,  m_inpBuff,  m_inpBufferSize,  "inp");
        pitchAwareCopyToTensor(m_inputCorrTensor, m_corrBuff, m_corrBufferSize, "corr");
        pitchAwareCopyToTensor(m_inputIiTensor,   m_iiBuff,   m_iiBufferSize,   "ii");
        pitchAwareCopyToTensor(m_inputJjTensor,   m_jjBuff,   m_jjBufferSize,   "jj");
        pitchAwareCopyToTensor(m_inputKkTensor,   m_kkBuff,   m_kkBufferSize,   "kk");
    }
    
    // Sync input tensors from CPU to VP (required when using ea_tensor_data instead of ea_tensor_data_for_write)
#if defined(CV28)
    ea_tensor_sync_cache(m_inputNetTensor, EA_CPU, EA_VP);
    ea_tensor_sync_cache(m_inputInpTensor, EA_CPU, EA_VP);
    ea_tensor_sync_cache(m_inputCorrTensor, EA_CPU, EA_VP);
    ea_tensor_sync_cache(m_inputIiTensor, EA_CPU, EA_VP);
    ea_tensor_sync_cache(m_inputJjTensor, EA_CPU, EA_VP);
    ea_tensor_sync_cache(m_inputKkTensor, EA_CPU, EA_VP);
#endif
#endif

    if (m_estimateTime)
    {
        time_1 = std::chrono::high_resolution_clock::now();
    }

    return true;
}

// Stub implementations for optional methods
void DPVOUpdate::runThread() {}
void DPVOUpdate::stopThread() {}
void DPVOUpdate::updateInputData(float* netData, float* inpData, float* corrData, 
                                 float* iiData, float* jjData, float* kkData, int frameIdx) {}
void DPVOUpdate::notifyProcessingComplete() {}
bool DPVOUpdate::getLastestPrediction(DPVOUpdate_Prediction& pred, int& frameIdx) { return false; }
bool DPVOUpdate::isInputBufferEmpty() const { return true; }
bool DPVOUpdate::isPredictionBufferEmpty() const { return true; }
void DPVOUpdate::getDebugProfiles(float& inferenceTime, int& inputBufferSize, int& outBufferSize) {
    inferenceTime = m_inferenceTime;
    inputBufferSize = static_cast<int>(m_netBufferSize + m_inpBufferSize + m_corrBufferSize);
    outBufferSize = static_cast<int>(m_netOutBufferSize + m_dOutBufferSize + m_wOutBufferSize);
}
bool DPVOUpdate::_runInferenceFunc() { return false; }

// =================================================================================================
// Reshape inputs for DPVO update model (Member function)
// =================================================================================================
// Uses pre-allocated buffers to avoid memory allocation overhead

// ----------What is float (*m_net)[384] means ??----------------
// Row 0:  [0][0] [0][1] ... [0][383]
// Row 1:  [1][0] [1][1] ... [1][383]
// Row 2:  [2][0] [2][1] ... [2][383]
// ...
// address = base + (i * 384 + j) * sizeof(float)
// The compiler needs that 384 to compute offsets correctly.
// That is exactly why the function must declare:
// float (*m_net)[384] means:
// ✅ Pointer to array of 384 floats
// ✅ Used for 2D contiguous arrays
// ✅ Required so compiler knows row size
// ✅ Matches float m_net[MAX_EDGES][384]
int DPVOUpdate::reshapeInput(
    int num_active,
    float (*m_net)[384],
    const float* ctx,
    const std::vector<float>& corr,
    const int* m_ii,
    const int* m_jj,
    const int* m_kk,
    int D,
    int P,
    // Output buffers (pre-allocated, reused)
        std::vector<float>& net_input,
        std::vector<float>& inp_input,
        std::vector<float>& corr_input,
        std::vector<float>& ii_input,
        std::vector<float>& jj_input,
        std::vector<float>& kk_input,
    const int MODEL_EDGE_COUNT,
    const int CORR_DIM)
{
    // Prepare input data - pad or truncate to MODEL_EDGE_COUNT
    const int num_edges_to_process = std::min(num_active, MODEL_EDGE_COUNT);
    
    // Resize buffers to match model input size [1, DIM, H, 1]
    // Total size: 1 * DIM * MODEL_EDGE_COUNT * 1 = DIM * MODEL_EDGE_COUNT
    net_input.resize(1 * 384 * MODEL_EDGE_COUNT * 1);
    inp_input.resize(1 * 384 * MODEL_EDGE_COUNT * 1);
    corr_input.resize(1 * CORR_DIM * MODEL_EDGE_COUNT * 1);
    ii_input.resize(1 * 1 * MODEL_EDGE_COUNT * 1);
    jj_input.resize(1 * 1 * MODEL_EDGE_COUNT * 1);
    kk_input.resize(1 * 1 * MODEL_EDGE_COUNT * 1);
    
    // Zero-fill ALL buffers — matching ONNX path exactly (update_onnx.cpp lines 407-413).
    // CRITICAL: The previous code only zeroed an incorrect contiguous range using NCHW offsets
    // (384 * num_edges_to_process) which doesn't correspond to inactive edge positions in the
    // channel-first layout. The ONNX path zeros EVERYTHING first, then overwrites active edges.
    // Inactive edges must be zero (the model was trained with zero-padded inactive edges).
    std::fill(net_input.begin(), net_input.end(), 0.0f);
    std::fill(inp_input.begin(), inp_input.end(), 0.0f);
    std::fill(corr_input.begin(), corr_input.end(), 0.0f);
    std::fill(ii_input.begin(), ii_input.end(), 0.0f);
    std::fill(jj_input.begin(), jj_input.end(), 0.0f);
    std::fill(kk_input.begin(), kk_input.end(), 0.0f);
    
    // Check net state before copying
    int net_zero_count = 0;
    int net_nonzero_count = 0;
    float net_min = std::numeric_limits<float>::max();
    float net_max = std::numeric_limits<float>::lowest();
    for (int e = 0; e < std::min(num_edges_to_process, num_active); e++) {
        for (int d = 0; d < 384; d++) {
            float val = m_net[e][d];
            if (val == 0.0f) net_zero_count++;
            else net_nonzero_count++;
            if (val < net_min) net_min = val;
            if (val > net_max) net_max = val;
        }
    }
    
    // NOTE: Do NOT modify m_net when it's all zeros. Python starts with self.pg.net = torch.zeros(...)
    // and passes zeros on the first iteration. The ONNX path (update_onnx.cpp) does the same.
    // Previously there was a workaround here that corrupted the initial hidden state by setting
    // m_net = ctx * 0.1, which caused diverging behavior vs the ONNX model.
    
    // Reshape net and inp data: [num_active, 384] -> [1, 384, 360, 1] (384=NET_DIM, 360=MAX_EDGES)
    for (int e = 0; e < num_edges_to_process; e++) {
        // Validate edge index
        if (e < 0 || e >= num_active) {
            continue;
        }
        for (int d = 0; d < 384; d++) {
            // YAML layout: [N, C, H, W] = [1, 384, 360, 1] (384=NET_DIM channels, 360=MAX_EDGES)
            // Index calculation: n * C * H * W + c * H * W + h * W + w
            // For net/inp: n=0, c=d (channel), h=e (edge index), w=0
            int idx = 0 * 384 * MODEL_EDGE_COUNT * 1 + d * MODEL_EDGE_COUNT * 1 + e * 1 + 0;
            net_input[idx] = m_net[e][d];
            inp_input[idx] = ctx[e * 384 + d];
        }
    }
    
    // Check input data statistics after copying
    float net_input_min = *std::min_element(net_input.begin(), net_input.end());
    float net_input_max = *std::max_element(net_input.begin(), net_input.end());
    float inp_input_min = *std::min_element(inp_input.begin(), inp_input.end());
    float inp_input_max = *std::max_element(inp_input.begin(), inp_input.end());
    float corr_input_min = *std::min_element(corr_input.begin(), corr_input.end());
    float corr_input_max = *std::max_element(corr_input.begin(), corr_input.end());
    
    // Reshape correlation: [num_active, CORR_DIM] -> [1, CORR_DIM, MODEL_EDGE_COUNT, 1]
    // CRITICAL: Use simple flat copy matching the ONNX reshapeInput (update_onnx.cpp line 432).
    // computeCorrelation outputs correlation data with 882 elements per edge (D_output=7, P=3, 2 levels).
    // The flat layout already matches what the model expects — DO NOT decompose and reorder!
    // Previous code had a 5-nested loop that permuted features into the wrong order.
    
    // Check correlation data before reshaping
    float corr_min = *std::min_element(corr.begin(), corr.end());
    float corr_max = *std::max_element(corr.begin(), corr.end());
    int corr_zero_count = 0;
    int corr_nonzero_count = 0;
    for (size_t i = 0; i < corr.size(); i++) {
        if (corr[i] == 0.0f) corr_zero_count++;
        else corr_nonzero_count++;
    }
    
    int corr_copied_count = 0;
    int corr_skipped_count = 0;
    for (int e = 0; e < num_edges_to_process; e++) {
        if (e < 0 || e >= num_active) {
            continue;
        }
        // Flat copy: corr[e * CORR_DIM + c] → corr_input[c * MODEL_EDGE_COUNT + e]
        // This matches update_onnx.cpp: corr_input[idx] = corr[e * CORR_DIM + c]
        for (int c = 0; c < CORR_DIM; c++) {
            int src_idx = e * CORR_DIM + c;
            if (src_idx < 0 || src_idx >= static_cast<int>(corr.size())) {
                corr_skipped_count++;
                continue;
            }
            // [1, CORR_DIM, MODEL_EDGE_COUNT, 1] layout
            int idx = c * MODEL_EDGE_COUNT + e;
            if (idx >= 0 && idx < static_cast<int>(corr_input.size())) {
                corr_input[idx] = corr[src_idx];
                corr_copied_count++;
            } else {
                corr_skipped_count++;
            }
        }
    }
    
    // Copy indices: matching ONNX path exactly (update_onnx.cpp lines 440-451)
    // Layout: [1, 1, MODEL_EDGE_COUNT, 1] → index = e
    // CRITICAL: Pass raw indices WITHOUT clamping — the ONNX model expects raw values.
    // Previously this code clamped ii to [0,3], jj to [0,35], kk to [0,∞) and padded
    // inactive edges with non-zero last-valid indices. This caused divergence because:
    //   1. The model was trained with zero-padded inactive edges (Python uses zero padding)
    //   2. Non-zero indices in inactive edges cause Gather to pull different data
    //   3. Index clamping changes values the model was designed to receive
    // Inactive edges remain zero from the std::fill above (matching ONNX path).
    for (int e = 0; e < num_edges_to_process; e++) {
        if (e >= num_active) continue;
        
        // [1, 1, MODEL_EDGE_COUNT, 1] layout: idx = e
        int idx = e;
        
        ii_input[idx] = static_cast<float>(m_ii[e]);
        jj_input[idx] = static_cast<float>(m_jj[e]);
        kk_input[idx] = static_cast<float>(m_kk[e]);
    }
    
    return num_edges_to_process;
}

