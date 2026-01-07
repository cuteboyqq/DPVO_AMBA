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
    : m_maxEdge(768),  // Initialize max edge count (change this value to modify model input size)
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

// =================================================================================================
// Ambarella CV28 Tensor Functions
// =================================================================================================
#if defined(CV28) || defined(CV28_SIMULATOR)
bool DPVOUpdate::_checkSavedTensor(int frameIdx)
{
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
        return true; // true means the tensors are already saved in the specific path
    }

    return false;
}

#if defined(SAVE_OUTPUT_TENSOR)
bool DPVOUpdate::_saveOutputTensor(int frameIdx)
{
    std::string netOutFilePath = m_tensorPath + "/frame_" + std::to_string(frameIdx - 1) + "_tensor0.bin";
    std::string dOutFilePath = m_tensorPath + "/frame_" + std::to_string(frameIdx - 1) + "_tensor1.bin";
    std::string wOutFilePath = m_tensorPath + "/frame_" + std::to_string(frameIdx - 1) + "_tensor2.bin";

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

        // Copy output tensors to prediction buffers
        std::memcpy(m_pred.netOutBuff, (float *)tensor0_data, m_netOutBufferSize * sizeof(float));
        std::memcpy(m_pred.dOutBuff, (float *)tensor1_data, m_dOutBufferSize * sizeof(float));
        std::memcpy(m_pred.wOutBuff, (float *)tensor2_data, m_wOutBufferSize * sizeof(float));

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
            logger->info("\033[34mDPVOUpdate::_run: Delta output (dOut) - size={}, range=[{}, {}], mean={}, zero_count={}, nonzero_count={}\033[0m",
                         m_dOutBufferSize, d_min, d_max, d_mean, d_zero_count, d_nonzero_count);
            logger->info("\033[34mDPVOUpdate::_run: Weight output (wOut) - size={}, range=[{}, {}], mean={}, zero_count={}, nonzero_count={}\033[0m",
                         m_wOutBufferSize, w_min, w_max, w_mean, w_zero_count, w_nonzero_count);
            
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
                
                logger->info("\033[34m  Edge[{}]: delta=[{}, {}], weight=[{}, {}]\033[0m",
                             e, 
                             (d_idx0 < m_dOutBufferSize ? dOut[d_idx0] : 0.0f),
                             (d_idx1 < m_dOutBufferSize ? dOut[d_idx1] : 0.0f),
                             (w_idx0 < m_wOutBufferSize ? wOut[w_idx0] : 0.0f),
                             (w_idx1 < m_wOutBufferSize ? wOut[w_idx1] : 0.0f));
            }
        }

#if defined(SAVE_OUTPUT_TENSOR)
        _saveOutputTensor(frameIdx);
#endif
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
                            int32_t *iiData, int32_t *jjData, int32_t *kkData)
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
    std::memcpy(m_iiBuff, iiData, m_iiBufferSize * sizeof(int32_t));
    std::memcpy(m_jjBuff, jjData, m_jjBufferSize * sizeof(int32_t));
    std::memcpy(m_kkBuff, kkData, m_kkBufferSize * sizeof(int32_t));

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
    

    // Copy float tensors
    std::memcpy(net_data, m_netBuff, m_netBufferSize * sizeof(float));
    std::memcpy(inp_data, m_inpBuff, m_inpBufferSize * sizeof(float));
    std::memcpy(corr_data, m_corrBuff, m_corrBufferSize * sizeof(float));

    // Copy int32 tensors
    std::memcpy(ii_data, m_iiBuff, m_iiBufferSize * sizeof(int32_t));
    std::memcpy(jj_data, m_jjBuff, m_jjBufferSize * sizeof(int32_t));
    std::memcpy(kk_data, m_kkBuff, m_kkBufferSize * sizeof(int32_t));
    
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

void DPVOUpdate::updateTensorPath(const std::string &path)
{
    m_tensorPath = path;
}

// Stub implementations for optional methods
void DPVOUpdate::runThread() {}
void DPVOUpdate::stopThread() {}
void DPVOUpdate::updateInputData(float* netData, float* inpData, float* corrData, 
                                 int32_t* iiData, int32_t* jjData, int32_t* kkData, int frameIdx) {}
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
    std::vector<int32_t>& ii_input,
    std::vector<int32_t>& jj_input,
    std::vector<int32_t>& kk_input,
    const int MODEL_EDGE_COUNT,
    const int CORR_DIM)
{
    // Prepare input data - pad or truncate to MODEL_EDGE_COUNT
    const int num_edges_to_process = std::min(num_active, MODEL_EDGE_COUNT);
    
    // Resize buffers if needed (they should be pre-allocated, but resize is safe)
    net_input.resize(1 * 384 * MODEL_EDGE_COUNT * 1, 0.0f);
    inp_input.resize(1 * 384 * MODEL_EDGE_COUNT * 1, 0.0f);
    corr_input.resize(1 * CORR_DIM * MODEL_EDGE_COUNT * 1, 0.0f);
    ii_input.resize(1 * 1 * MODEL_EDGE_COUNT * 1, 0);
    jj_input.resize(1 * 1 * MODEL_EDGE_COUNT * 1, 0);
    kk_input.resize(1 * 1 * MODEL_EDGE_COUNT * 1, 0);
    
    // Zero-fill buffers (faster than resize with value)
    std::fill(net_input.begin(), net_input.end(), 0.0f);
    std::fill(inp_input.begin(), inp_input.end(), 0.0f);
    std::fill(corr_input.begin(), corr_input.end(), 0.0f);
    std::fill(ii_input.begin(), ii_input.end(), 0);
    std::fill(jj_input.begin(), jj_input.end(), 0);
    std::fill(kk_input.begin(), kk_input.end(), 0);
    
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
    
    // WORKAROUND: If net is all zeros, initialize it from context (inp) to break the cycle
    bool net_all_zero = (net_nonzero_count == 0);
    if (net_all_zero) {
        // Initialize net from context (inp) - this gives the model some initial state
        for (int e = 0; e < num_edges_to_process; e++) {
            if (e < 0 || e >= num_active) continue;
            for (int d = 0; d < 384; d++) {
                // Use context as initial net state (scaled down to avoid large values)
                m_net[e][d] = ctx[e * 384 + d] * 0.1f;
            }
        }
    }
    
    // Reshape net and inp data: [num_active, 384] -> [1, 384, 384, 1]
    for (int e = 0; e < num_edges_to_process; e++) {
        // Validate edge index
        if (e < 0 || e >= num_active) {
            continue;
        }
        for (int d = 0; d < 384; d++) {
            // YAML layout: [N, C, H, W] = [1, 384, 384, 1]
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
    
    // Reshape correlation: [num_active, D, D, P, P, 2] -> [1, 882, 384, 1]
    const int target_corr_dim = CORR_DIM; // 882
    const int D_target = 7;  // Target window size for model (7×7 instead of 8×8)
    const int offset = (D - D_target) / 2;  // Center offset: (8-7)/2 = 0 (integer division)
    
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
        // Validate edge index
        if (e < 0 || e >= num_active) {
            continue;
        }
        // Source layout: corr[e][di][dj][pi][pj][c] = [num_active, D, D, P, P, 2]
        // We need to extract center 7×7 region from 8×8 window
        for (int c = 0; c < 2; c++) {
            for (int di = 0; di < D_target && (di + offset) < D; di++) {
                for (int dj = 0; dj < D_target && (dj + offset) < D; dj++) {
                    for (int pi = 0; pi < P; pi++) {
                        for (int pj = 0; pj < P; pj++) {
                            // Source: corr[e][di+offset][dj+offset][pi][pj][c]
                            // Layout: [num_active, D, D, P, P, 2]
                            int src_idx = e * D * D * P * P * 2 +
                                         (di + offset) * D * P * P * 2 +
                                         (dj + offset) * P * P * 2 +
                                         pi * P * 2 +
                                         pj * 2 +
                                         c;  // Channel last
                            
                            // Validate source index
                            if (src_idx < 0 || src_idx >= static_cast<int>(corr.size())) {
                                corr_skipped_count++;
                                continue;
                            }
                            
                            // YAML layout: [N, C, H, W] = [1, 882, 384, 1]
                            // Model expects: [1, 882, 384, 1] where 882 = 2 * 7 * 7 * 3 * 3
                            // Index calculation: n * C * H * W + c * H * W + h * W + w
                            // Where: n=0, c=dst_corr_idx (feature index), h=e (edge index), w=0
                            int dst_corr_idx = c * D_target * D_target * P * P +
                                              di * D_target * P * P +
                                              dj * P * P +
                                              pi * P + pj;
                            
                            if (dst_corr_idx < target_corr_dim) {
                                // [N, C, H, W] = [1, 882, 384, 1]
                                int idx = 0 * CORR_DIM * MODEL_EDGE_COUNT * 1 + 
                                         dst_corr_idx * MODEL_EDGE_COUNT * 1 + 
                                         e * 1 + 
                                         0;
                                
                                    // Validate destination index
                                    if (idx >= 0 && idx < static_cast<int>(corr_input.size())) {
                                        corr_input[idx] = corr[src_idx];
                                        corr_copied_count++;
                                    } else {
                                        corr_skipped_count++;
                                    }
                            } else {
                                corr_skipped_count++;
                            }
                        }
                    }
                }
            }
        }
        // Rest of CORR_DIM is zero-padded (already initialized to 0)
    }
    
    // Copy indices: [num_active] -> [1, 1, 384, 1] (YAML now specifies 4D shape)
    // AMBA tensor shape: [N, C, H, W] = [1, 1, 384, 1]
    // CRITICAL: For zero-padded edges, we need to set valid indices to prevent
    // neighbors_tensor from producing invalid gather indices that cause AMBA CV28 to fail.
    // Strategy: Duplicate the last valid edge's indices for all zero-padded edges.
    
    // Get last valid indices for padding (if we have valid edges)
    // CRITICAL: Ensure all indices are non-negative, especially kk which can become negative
    // after keyframe removal (m_kk[e] -= PatchGraph::M)
    int32_t last_ii = 0, last_jj = 0, last_kk = 0;
    if (num_edges_to_process > 0 && num_active > 0) {
        // Find the last valid edge with non-negative kk
        int last_valid_e = -1;
        for (int e = std::min(num_edges_to_process - 1, num_active - 1); e >= 0; e--) {
            if (m_kk[e] >= 0) {
                last_valid_e = e;
                break;
            }
        }
        
        if (last_valid_e >= 0) {
            int32_t raw_ii = static_cast<int32_t>(m_ii[last_valid_e]);
            int32_t raw_jj = static_cast<int32_t>(m_jj[last_valid_e]);
            int32_t raw_kk = static_cast<int32_t>(m_kk[last_valid_e]);
            
            // Clamp to valid ranges before using as padding values
            const int M = 8;  // PATCHES_PER_FRAME
            const int MAX_FRAMES = 36;  // BUFFER_SIZE
            last_ii = (raw_ii < 0) ? 0 : ((raw_ii >= M) ? M - 1 : raw_ii);
            last_jj = (raw_jj < 0) ? 0 : ((raw_jj >= MAX_FRAMES) ? MAX_FRAMES - 1 : raw_jj);
            last_kk = (raw_kk < 0) ? 0 : raw_kk;
        } else {
            // Fallback: use first edge if all have negative kk (shouldn't happen, but be safe)
            if (num_active > 0) {
                const int M = 8;
                const int MAX_FRAMES = 36;
                int32_t raw_ii = static_cast<int32_t>(m_ii[0]);
                int32_t raw_jj = static_cast<int32_t>(m_jj[0]);
                last_ii = (raw_ii < 0) ? 0 : ((raw_ii >= M) ? M - 1 : raw_ii);
                last_jj = (raw_jj < 0) ? 0 : ((raw_jj >= MAX_FRAMES) ? MAX_FRAMES - 1 : raw_jj);
                last_kk = 0;  // Force to 0 if all kk are negative
            }
        }
    }
    
    for (int e = 0; e < MODEL_EDGE_COUNT; e++) {
        // AMBA 4D tensor layout: [N, C, H, W] = [1, 1, 384, 1]
        // Index formula: idx = n * C * H * W + c * H * W + h * W + w
        // Where: n=0, c=0, h=e (edge index), w=0
        // idx = 0 * 1 * 384 * 1 + 0 * 384 * 1 + e * 1 + 0 = e
        int idx = 0 * 1 * MODEL_EDGE_COUNT * 1 + 0 * MODEL_EDGE_COUNT * 1 + e * 1 + 0;
        
        if (e < num_edges_to_process && e < num_active) {
            // Valid edge: copy actual indices, but clamp to valid ranges
            // CRITICAL: AMBA CV28 requires valid indices for gather operations
            // ii should be patch index in [0, M-1] where M=8 (patches per frame)
            // jj should be frame index in [0, n-1] where n is number of frames
            // kk can become negative after keyframe removal (m_kk[e] -= PatchGraph::M)
            const int M = 8;  // PATCHES_PER_FRAME
            const int MAX_FRAMES = 36;  // BUFFER_SIZE, but we use actual n from context
            
            int32_t ii_val = static_cast<int32_t>(m_ii[e]);
            int32_t jj_val = static_cast<int32_t>(m_jj[e]);
            int32_t kk_val = static_cast<int32_t>(m_kk[e]);
            
            // Clamp ii to valid patch index range [0, M-1] = [0, 7]
            // CRITICAL: ii values > M-1 will cause neighbors_tensor to compute invalid Gather indices
            if (ii_val < 0) {
                ii_val = 0;
            } else if (ii_val >= M) {
                ii_val = M - 1;  // Clamp to max valid patch index
            }
            
            // Clamp jj to valid frame index range [0, MAX_FRAMES-1]
            // Note: We use MAX_FRAMES as upper bound, actual n might be smaller
            if (jj_val < 0) {
                jj_val = 0;
            } else if (jj_val >= MAX_FRAMES) {
                jj_val = MAX_FRAMES - 1;
            }
            
            // Clamp kk to non-negative (AMBA CV28 Gather requirement)
            if (kk_val < 0) {
                kk_val = 0;  // Clamp to 0 to prevent AMBA CV28 gather failures
            }
            
            ii_input[idx] = ii_val;
            jj_input[idx] = jj_val;
            kk_input[idx] = kk_val;
        } else {
            // Zero-padded edge: use last valid indices (already clamped to valid ranges)
            // This ensures neighbors_tensor produces valid gather indices (prevents AMBA CV28 gather failures)
            ii_input[idx] = last_ii;  // Already clamped to [0, M-1]
            jj_input[idx] = last_jj;  // Already clamped to [0, MAX_FRAMES-1]
            kk_input[idx] = last_kk;  // Already clamped to >= 0
        }
    }
    
    return num_edges_to_process;
}

