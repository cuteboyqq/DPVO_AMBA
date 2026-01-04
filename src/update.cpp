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
    if (rval != EA_SUCCESS) {
        logger->error("Failed to config input '{}', rval={}", m_inputNetTensorName, rval);
    }

    logger->info("Input Inp Name: {}", m_inputInpTensorName);
    rval = ea_net_config_input(m_model, m_inputInpTensorName.c_str());
    if (rval != EA_SUCCESS) {
        logger->error("Failed to config input '{}', rval={}", m_inputInpTensorName, rval);
    }

    logger->info("Input Corr Name: {}", m_inputCorrTensorName);
    rval = ea_net_config_input(m_model, m_inputCorrTensorName.c_str());
    if (rval != EA_SUCCESS) {
        logger->error("Failed to config input '{}', rval={}", m_inputCorrTensorName, rval);
    }

    logger->info("Input Ii Name: {}", m_inputIiTensorName);
    rval = ea_net_config_input(m_model, m_inputIiTensorName.c_str());
    if (rval != EA_SUCCESS) {
        logger->error("Failed to config input '{}', rval={}", m_inputIiTensorName, rval);
    }

    logger->info("Input Jj Name: {}", m_inputJjTensorName);
    rval = ea_net_config_input(m_model, m_inputJjTensorName.c_str());
    if (rval != EA_SUCCESS) {
        logger->error("Failed to config input '{}', rval={}", m_inputJjTensorName, rval);
    }

    logger->info("Input Kk Name: {}", m_inputKkTensorName);
    rval = ea_net_config_input(m_model, m_inputKkTensorName.c_str());
    if (rval != EA_SUCCESS) {
        logger->error("Failed to config input '{}', rval={}", m_inputKkTensorName, rval);
    }

    // Configure output tensors
    logger->info("-------------------------------------------");
    logger->info("Configure Output Tensors");

    for (size_t i = 0; i < m_outputTensorList.size(); ++i)
    {
        logger->info("Output Name: {}", m_outputTensorList[i]);
        rval = ea_net_config_output(m_model, m_outputTensorList[i].c_str());
        if (rval != EA_SUCCESS) {
            logger->error("Failed to config output '{}', rval={}", m_outputTensorList[i], rval);
        } else {
            logger->info("Successfully configured output '{}'", m_outputTensorList[i]);
        }
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
    if (rval != EA_SUCCESS) {
        logger->error("Failed to load model from '{}', rval={} (EA_SUCCESS=0)", m_ptrModelPath, rval);
        logger->error("Model load failed - inference will not work until model is loaded successfully");
        return;
    } else {
        logger->info("Model loaded successfully from '{}'", m_ptrModelPath);
    }

    // Get input tensors
    logger->info("-------------------------------------------");
    logger->info("Create Model Input Tensors");

    m_inputNetTensor = ea_net_input(m_model, m_inputNetTensorName.c_str());
    m_inputInpTensor = ea_net_input(m_model, m_inputInpTensorName.c_str());
    m_inputCorrTensor = ea_net_input(m_model, m_inputCorrTensorName.c_str());
    m_inputIiTensor = ea_net_input(m_model, m_inputIiTensorName.c_str());
    m_inputJjTensor = ea_net_input(m_model, m_inputJjTensorName.c_str());
    m_inputKkTensor = ea_net_input(m_model, m_inputKkTensorName.c_str());

    // Validate input tensors were retrieved successfully
    if (m_inputNetTensor == nullptr || m_inputInpTensor == nullptr || m_inputCorrTensor == nullptr ||
        m_inputIiTensor == nullptr || m_inputJjTensor == nullptr || m_inputKkTensor == nullptr) {
        logger->error("Failed to get one or more input tensors - model may not be loaded correctly");
        logger->error("Input tensors: net={}, inp={}, corr={}, ii={}, jj={}, kk={}",
                     (void*)m_inputNetTensor, (void*)m_inputInpTensor, (void*)m_inputCorrTensor,
                     (void*)m_inputIiTensor, (void*)m_inputJjTensor, (void*)m_inputKkTensor);
        return;
    }

    logger->info("Input Net Shape: {}x{}x{}x{}",
                 ea_tensor_shape(m_inputNetTensor)[EA_N],
                 ea_tensor_shape(m_inputNetTensor)[EA_C],
                 ea_tensor_shape(m_inputNetTensor)[EA_H],
                 ea_tensor_shape(m_inputNetTensor)[EA_W]);

    logger->info("Input Inp Shape: {}x{}x{}x{}",
                 ea_tensor_shape(m_inputInpTensor)[EA_N],
                 ea_tensor_shape(m_inputInpTensor)[EA_C],
                 ea_tensor_shape(m_inputInpTensor)[EA_H],
                 ea_tensor_shape(m_inputInpTensor)[EA_W]);

    logger->info("Input Corr Shape: {}x{}x{}x{}",
                 ea_tensor_shape(m_inputCorrTensor)[EA_N],
                 ea_tensor_shape(m_inputCorrTensor)[EA_C],
                 ea_tensor_shape(m_inputCorrTensor)[EA_H],
                 ea_tensor_shape(m_inputCorrTensor)[EA_W]);

    logger->info("Input Ii Shape: {}x{}x{}x{}",
                 ea_tensor_shape(m_inputIiTensor)[EA_N],
                 ea_tensor_shape(m_inputIiTensor)[EA_C],
                 ea_tensor_shape(m_inputIiTensor)[EA_H],
                 ea_tensor_shape(m_inputIiTensor)[EA_W]);

    logger->info("Input Jj Shape: {}x{}x{}x{}",
                 ea_tensor_shape(m_inputJjTensor)[EA_N],
                 ea_tensor_shape(m_inputJjTensor)[EA_C],
                 ea_tensor_shape(m_inputJjTensor)[EA_H],
                 ea_tensor_shape(m_inputJjTensor)[EA_W]);

    logger->info("Input Kk Shape: {}x{}x{}x{}",
                 ea_tensor_shape(m_inputKkTensor)[EA_N],
                 ea_tensor_shape(m_inputKkTensor)[EA_C],
                 ea_tensor_shape(m_inputKkTensor)[EA_H],
                 ea_tensor_shape(m_inputKkTensor)[EA_W]);

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
    logger->info("DPVOUpdate::_run: About to call _loadInput, frameIdx={}", frameIdx);
    if (!_loadInput(netData, inpData, corrData, iiData, jjData, kkData))
    {
        logger->error("Load Input Data Failed");
        return false;
    }
    logger->info("DPVOUpdate::_run: _loadInput completed");

    // STEP 2: inference
    // STEP 2-1: check if the tensors are already saved, pass the inference
    logger->info("DPVOUpdate::_run: About to check saved tensor, frameIdx={}", frameIdx);
    if (_checkSavedTensor(frameIdx))
    {
        logger->info("DPVOUpdate::_run: Using saved tensor, allocating buffers");
        // Allocate memory for all output tensors
        m_pred.netOutBuff = new float[m_netOutBufferSize];
        m_pred.dOutBuff = new float[m_dOutBufferSize];
        m_pred.wOutBuff = new float[m_wOutBufferSize];

        // Copy output tensors to prediction buffers
        std::memcpy(m_pred.netOutBuff, (float *)m_netOutBuff, m_netOutBufferSize * sizeof(float));
        std::memcpy(m_pred.dOutBuff, (float *)m_dOutBuff, m_dOutBufferSize * sizeof(float));
        std::memcpy(m_pred.wOutBuff, (float *)m_wOutBuff, m_wOutBufferSize * sizeof(float));
        logger->info("DPVOUpdate::_run: Saved tensor copy completed");
    }
    else
    {
        logger->info("DPVOUpdate::_run: Saved tensor not found, running inference");
        // STEP 2-2: run inference using Ambarella's eazyai library
        if (m_estimateTime)
            time_1 = std::chrono::high_resolution_clock::now();

        // Validate model before calling forward
        logger->info("DPVOUpdate::_run: Validating model before forward pass");
        if (m_model == nullptr) {
            logger->error("DPVOUpdate::_run: m_model is null - cannot run inference");
            return false;
        }
        logger->info("DPVOUpdate::_run: Model is valid, m_model={}", (void*)m_model);
        
        // Validate input tensors are still valid
        logger->info("DPVOUpdate::_run: Re-validating input tensors before forward");
        if (m_inputNetTensor == nullptr || m_inputInpTensor == nullptr || m_inputCorrTensor == nullptr ||
            m_inputIiTensor == nullptr || m_inputJjTensor == nullptr || m_inputKkTensor == nullptr) {
            logger->error("DPVOUpdate::_run: Input tensors became null before forward pass");
            return false;
        }
        logger->info("DPVOUpdate::_run: All input tensors are still valid");
        
        // Log input tensor data pointers one more time
        void* net_data_check = ea_tensor_data(m_inputNetTensor);
        void* inp_data_check = ea_tensor_data(m_inputInpTensor);
        void* corr_data_check = ea_tensor_data(m_inputCorrTensor);
        logger->info("DPVOUpdate::_run: Input tensor data pointers - net={}, inp={}, corr={}",
                     (void*)net_data_check, (void*)inp_data_check, (void*)corr_data_check);
        
        if (net_data_check == nullptr || inp_data_check == nullptr || corr_data_check == nullptr) {
            logger->error("DPVOUpdate::_run: Input tensor data pointers are null before forward");
            return false;
        }
        
        // Test read from input tensors to ensure they're accessible
        logger->info("DPVOUpdate::_run: Testing read access to input tensor data");
        volatile float test_net = static_cast<float*>(net_data_check)[0];
        volatile float test_inp = static_cast<float*>(inp_data_check)[0];
        volatile float test_corr = static_cast<float*>(corr_data_check)[0];
        logger->info("DPVOUpdate::_run: Test read successful - net[0]={}, inp[0]={}, corr[0]={}",
                     test_net, test_inp, test_corr);

        // Check if net input is all zeros (which might cause inference to fail)
        bool net_all_zero = true;
        for (size_t i = 0; i < m_netBufferSize && net_all_zero; i++) {
            if (m_netBuff[i] != 0.0f) {
                net_all_zero = false;
            }
        }
        if (net_all_zero) {
            logger->warn("DPVOUpdate::_run: WARNING - net input buffer is all zeros ({} elements)", m_netBufferSize);
            logger->warn("DPVOUpdate::_run: This might cause ea_net_forward to fail - net state may not be initialized");
        }
        
        // Log input tensor statistics one more time before forward
        if (net_data_check != nullptr && m_netBufferSize > 0) {
            float* net_array = static_cast<float*>(net_data_check);
            float net_tensor_min = *std::min_element(net_array, net_array + m_netBufferSize);
            float net_tensor_max = *std::max_element(net_array, net_array + m_netBufferSize);
            logger->info("DPVOUpdate::_run: Net tensor data range before forward: [{}, {}]", net_tensor_min, net_tensor_max);
        }
        
        // Verify output tensors are valid before forward (they should be from initialization)
        logger->info("DPVOUpdate::_run: Verifying output tensors before forward");
        if (m_outputTensors[0] == nullptr || m_outputTensors[1] == nullptr || m_outputTensors[2] == nullptr) {
            logger->warn("DPVOUpdate::_run: Output tensors are null before forward - retrieving them");
            m_outputTensors[0] = ea_net_output_by_index(m_model, 0);
            m_outputTensors[1] = ea_net_output_by_index(m_model, 1);
            m_outputTensors[2] = ea_net_output_by_index(m_model, 2);
            if (m_outputTensors[0] == nullptr || m_outputTensors[1] == nullptr || m_outputTensors[2] == nullptr) {
                logger->error("DPVOUpdate::_run: Failed to retrieve output tensors before forward");
                return false;
            }
            logger->info("DPVOUpdate::_run: Output tensors retrieved successfully before forward");
        } else {
            logger->info("DPVOUpdate::_run: Output tensors are valid before forward");
        }
        
        // Final validation before forward
        logger->info("DPVOUpdate::_run: Final validation before forward");
        logger->info("DPVOUpdate::_run: Model pointer: {}", (void*)m_model);
        logger->info("DPVOUpdate::_run: Input tensor count: {}", ea_net_input_num(m_model));
        logger->info("DPVOUpdate::_run: Output tensor count: {}", ea_net_output_num(m_model));
        
        // Verify all input tensors are still valid
        bool all_inputs_valid = (m_inputNetTensor != nullptr && m_inputInpTensor != nullptr && 
                                m_inputCorrTensor != nullptr && m_inputIiTensor != nullptr && 
                                m_inputJjTensor != nullptr && m_inputKkTensor != nullptr);
        logger->info("DPVOUpdate::_run: All input tensors valid: {}", all_inputs_valid);
        
        logger->info("DPVOUpdate::_run: About to call ea_net_forward(m_model={}, batch=1)", (void*)m_model);
        int forward_result = ea_net_forward(m_model, 1);
        logger->info("DPVOUpdate::_run: ea_net_forward returned: {} (EA_SUCCESS={})", forward_result, EA_SUCCESS);
        
        if (EA_SUCCESS != forward_result)
        {
            logger->error("DPVOUpdate::_run: ea_net_forward failed with error code: {} (EA_SUCCESS={})", 
                         forward_result, EA_SUCCESS);
            logger->error("DPVOUpdate::_run: Error code meanings: 0=EA_SUCCESS, -1=EA_FAIL, -11=EAGAIN, -4=EINTR");
            logger->error("DPVOUpdate::_run: Model state - m_model={}, input tensors valid={}",
                         (void*)m_model,
                         (m_inputNetTensor != nullptr && m_inputInpTensor != nullptr && m_inputCorrTensor != nullptr));
            
            // Check if this is a recoverable error (EAGAIN or EINTR)
            if (forward_result == -11) {  // EAGAIN
                logger->warn("DPVOUpdate::_run: Inference interrupted by other net (EAGAIN) - this is recoverable");
            } else if (forward_result == -4) {  // EINTR
                logger->warn("DPVOUpdate::_run: Inference interrupted by signal (EINTR) - this is recoverable");
            } else {
                logger->error("DPVOUpdate::_run: Inference failed with unrecoverable error (likely EA_FAIL=-1)");
                logger->error("DPVOUpdate::_run: Possible causes: invalid input data, model state corruption, or hardware error");
            }
            
            // Don't release model/tensors on single inference failure - allow retry on next call
            // Only release in destructor or on critical errors
            return false;
        }
        logger->info("DPVOUpdate::_run: ea_net_forward completed successfully");
        
        // Get output tensors AFTER forward pass (they may have changed)
        logger->info("DPVOUpdate::_run: About to get output tensors by index");
        m_outputTensors[0] = ea_net_output_by_index(m_model, 0);
        m_outputTensors[1] = ea_net_output_by_index(m_model, 1);
        m_outputTensors[2] = ea_net_output_by_index(m_model, 2);
        logger->info("DPVOUpdate::_run: Output tensors retrieved");

        // Validate output tensors before syncing
        if (m_outputTensors[0] == nullptr || m_outputTensors[1] == nullptr || m_outputTensors[2] == nullptr) {
            logger->error("DPVOUpdate::_run: One or more output tensors are null after retrieval");
            return false;
        }
        
        // Sync output tensors between VP and CPU (AFTER getting them)
#if defined(CV28)
        logger->info("DPVOUpdate::_run: About to sync output tensors");
        rval = ea_tensor_sync_cache(m_outputTensors[0], EA_VP, EA_CPU);
        if (rval != EA_SUCCESS)
        {
            logger->error("Failed to sync output tensor 0, rval={}", rval);
        }
        rval = ea_tensor_sync_cache(m_outputTensors[1], EA_VP, EA_CPU);
        if (rval != EA_SUCCESS)
        {
            logger->error("Failed to sync output tensor 1, rval={}", rval);
        }
        rval = ea_tensor_sync_cache(m_outputTensors[2], EA_VP, EA_CPU);
        if (rval != EA_SUCCESS)
        {
            logger->error("Failed to sync output tensor 2, rval={}", rval);
        }
        logger->info("DPVOUpdate::_run: Tensor sync completed");
#endif

        // Validate output tensors
        if (m_outputTensors[0] == nullptr || m_outputTensors[1] == nullptr || m_outputTensors[2] == nullptr) {
            logger->error("DPVOUpdate::_run: One or more output tensors are null");
            return false;
        }

        logger->info("DPVOUpdate::_run: About to allocate prediction buffers");
        // Allocate memory for all output tensors
        m_pred.netOutBuff = new float[m_netOutBufferSize];
        m_pred.dOutBuff = new float[m_dOutBufferSize];
        m_pred.wOutBuff = new float[m_wOutBufferSize];
        logger->info("DPVOUpdate::_run: Prediction buffers allocated");

        // Get tensor data pointers and validate
        logger->info("DPVOUpdate::_run: About to get tensor data pointers");
        void* tensor0_data = ea_tensor_data(m_outputTensors[0]);
        void* tensor1_data = ea_tensor_data(m_outputTensors[1]);
        void* tensor2_data = ea_tensor_data(m_outputTensors[2]);
        
        if (tensor0_data == nullptr || tensor1_data == nullptr || tensor2_data == nullptr) {
            logger->error("DPVOUpdate::_run: One or more tensor data pointers are null");
            delete[] m_pred.netOutBuff;
            delete[] m_pred.dOutBuff;
            delete[] m_pred.wOutBuff;
            m_pred.netOutBuff = nullptr;
            m_pred.dOutBuff = nullptr;
            m_pred.wOutBuff = nullptr;
            return false;
        }
        logger->info("DPVOUpdate::_run: Tensor data pointers retrieved, about to memcpy");

        // Copy output tensors to prediction buffers
        std::memcpy(m_pred.netOutBuff, (float *)tensor0_data, m_netOutBufferSize * sizeof(float));
        logger->info("DPVOUpdate::_run: Copied tensor 0");
        std::memcpy(m_pred.dOutBuff, (float *)tensor1_data, m_dOutBufferSize * sizeof(float));
        logger->info("DPVOUpdate::_run: Copied tensor 1");
        std::memcpy(m_pred.wOutBuff, (float *)tensor2_data, m_wOutBufferSize * sizeof(float));
        logger->info("DPVOUpdate::_run: Copied tensor 2, memcpy completed");

#if defined(SAVE_OUTPUT_TENSOR)
        _saveOutputTensor(frameIdx);
#endif
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

    // Validate input data pointers
    if (netData == nullptr || inpData == nullptr || corrData == nullptr ||
        iiData == nullptr || jjData == nullptr || kkData == nullptr) {
        logger->error("DPVOUpdate::_loadInput: One or more input data pointers are null");
        return false;
    }
    
    logger->info("DPVOUpdate::_loadInput: Buffer sizes - net={}, inp={}, corr={}, ii={}, jj={}, kk={}",
                 m_netBufferSize, m_inpBufferSize, m_corrBufferSize, m_iiBufferSize, m_jjBufferSize, m_kkBufferSize);
    
    // Copy input data to working buffers
    logger->info("DPVOUpdate::_loadInput: Copying data to working buffers");
    std::memcpy(m_netBuff, netData, m_netBufferSize * sizeof(float));
    std::memcpy(m_inpBuff, inpData, m_inpBufferSize * sizeof(float));
    std::memcpy(m_corrBuff, corrData, m_corrBufferSize * sizeof(float));
    std::memcpy(m_iiBuff, iiData, m_iiBufferSize * sizeof(int32_t));
    std::memcpy(m_jjBuff, jjData, m_jjBufferSize * sizeof(int32_t));
    std::memcpy(m_kkBuff, kkData, m_kkBufferSize * sizeof(int32_t));
    logger->info("DPVOUpdate::_loadInput: Data copied to working buffers");
    
    // Log sample values for debugging
    if (m_netBufferSize > 0) {
        float net_min = *std::min_element(m_netBuff, m_netBuff + m_netBufferSize);
        float net_max = *std::max_element(m_netBuff, m_netBuff + m_netBufferSize);
        logger->info("DPVOUpdate::_loadInput: net buffer range: [{}, {}], first={}, last={}",
                     net_min, net_max, m_netBuff[0], m_netBuff[m_netBufferSize - 1]);
    }
    if (m_iiBufferSize > 0) {
        int32_t ii_min = *std::min_element(m_iiBuff, m_iiBuff + m_iiBufferSize);
        int32_t ii_max = *std::max_element(m_iiBuff, m_iiBuff + m_iiBufferSize);
        logger->info("DPVOUpdate::_loadInput: ii buffer range: [{}, {}], first={}, last={}",
                     ii_min, ii_max, m_iiBuff[0], m_iiBuff[m_iiBufferSize - 1]);
    }

    // Copy data to input tensors
#if defined(CV28) || defined(CV28_SIMULATOR)
    // Validate input tensors before using them
    logger->info("DPVOUpdate::_loadInput: Validating input tensors");
    if (m_inputNetTensor == nullptr || m_inputInpTensor == nullptr || m_inputCorrTensor == nullptr ||
        m_inputIiTensor == nullptr || m_inputJjTensor == nullptr || m_inputKkTensor == nullptr) {
        logger->error("DPVOUpdate::_loadInput: One or more input tensors are null - model may have been released");
        logger->error("DPVOUpdate::_loadInput: net={}, inp={}, corr={}, ii={}, jj={}, kk={}",
                     (void*)m_inputNetTensor, (void*)m_inputInpTensor, (void*)m_inputCorrTensor,
                     (void*)m_inputIiTensor, (void*)m_inputJjTensor, (void*)m_inputKkTensor);
        return false;
    }
    logger->info("DPVOUpdate::_loadInput: All input tensors are valid");
    
    // Log tensor shapes
    const size_t* net_shape = ea_tensor_shape(m_inputNetTensor);
    const size_t* inp_shape = ea_tensor_shape(m_inputInpTensor);
    const size_t* corr_shape = ea_tensor_shape(m_inputCorrTensor);
    const size_t* ii_shape = ea_tensor_shape(m_inputIiTensor);
    const size_t* jj_shape = ea_tensor_shape(m_inputJjTensor);
    const size_t* kk_shape = ea_tensor_shape(m_inputKkTensor);
    if (net_shape && inp_shape && corr_shape && ii_shape && jj_shape && kk_shape) {
        logger->info("DPVOUpdate::_loadInput: Tensor shapes - net=[{}x{}x{}x{}], inp=[{}x{}x{}x{}], corr=[{}x{}x{}x{}]",
                     net_shape[0], net_shape[1], net_shape[2], net_shape[3],
                     inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3],
                     corr_shape[0], corr_shape[1], corr_shape[2], corr_shape[3]);
        logger->info("DPVOUpdate::_loadInput: Int32 tensor shapes - ii=[{}x{}x{}x{}], jj=[{}x{}x{}x{}], kk=[{}x{}x{}x{}]",
                     ii_shape[0], ii_shape[1], ii_shape[2], ii_shape[3],
                     jj_shape[0], jj_shape[1], jj_shape[2], jj_shape[3],
                     kk_shape[0], kk_shape[1], kk_shape[2], kk_shape[3]);
    }
    
    // Get tensor data pointers for writing (using ea_tensor_data_for_write handles cache coherency automatically)
    logger->info("DPVOUpdate::_loadInput: Getting tensor data pointers for write");
    void* net_data = ea_tensor_data_for_write(m_inputNetTensor, EA_CPU);
    void* inp_data = ea_tensor_data_for_write(m_inputInpTensor, EA_CPU);
    void* corr_data = ea_tensor_data_for_write(m_inputCorrTensor, EA_CPU);
    void* ii_data = ea_tensor_data_for_write(m_inputIiTensor, EA_CPU);
    void* jj_data = ea_tensor_data_for_write(m_inputJjTensor, EA_CPU);
    void* kk_data = ea_tensor_data_for_write(m_inputKkTensor, EA_CPU);
    
    if (net_data == nullptr || inp_data == nullptr || corr_data == nullptr ||
        ii_data == nullptr || jj_data == nullptr || kk_data == nullptr) {
        logger->error("DPVOUpdate::_loadInput: One or more tensor data pointers are null");
        logger->error("DPVOUpdate::_loadInput: net_data={}, inp_data={}, corr_data={}, ii_data={}, jj_data={}, kk_data={}",
                     (void*)net_data, (void*)inp_data, (void*)corr_data,
                     (void*)ii_data, (void*)jj_data, (void*)kk_data);
        return false;
    }
    logger->info("DPVOUpdate::_loadInput: All tensor data pointers are valid");
    
    // Log tensor sizes
    size_t net_tensor_size = ea_tensor_size(m_inputNetTensor);
    size_t inp_tensor_size = ea_tensor_size(m_inputInpTensor);
    size_t corr_tensor_size = ea_tensor_size(m_inputCorrTensor);
    logger->info("DPVOUpdate::_loadInput: Tensor sizes - net={} bytes, inp={} bytes, corr={} bytes",
                 net_tensor_size, inp_tensor_size, corr_tensor_size);
    logger->info("DPVOUpdate::_loadInput: Expected buffer sizes - net={} bytes, inp={} bytes, corr={} bytes",
                 m_netBufferSize * sizeof(float), m_inpBufferSize * sizeof(float), m_corrBufferSize * sizeof(float));
    
    // Validate input data for NaN/Inf before copying
    logger->info("DPVOUpdate::_loadInput: Validating input data for NaN/Inf");
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < m_netBufferSize; i++) {
        if (std::isnan(m_netBuff[i])) nan_count++;
        if (std::isinf(m_netBuff[i])) inf_count++;
    }
    for (size_t i = 0; i < m_inpBufferSize; i++) {
        if (std::isnan(m_inpBuff[i])) nan_count++;
        if (std::isinf(m_inpBuff[i])) inf_count++;
    }
    for (size_t i = 0; i < m_corrBufferSize; i++) {
        if (std::isnan(m_corrBuff[i])) nan_count++;
        if (std::isinf(m_corrBuff[i])) inf_count++;
    }
    if (nan_count > 0 || inf_count > 0) {
        logger->error("DPVOUpdate::_loadInput: Found {} NaN and {} Inf values in input data - this will cause inference to fail", nan_count, inf_count);
        return false;
    }
    logger->info("DPVOUpdate::_loadInput: Input data validation passed (no NaN/Inf)");
    
    // Copy float tensors
    logger->info("DPVOUpdate::_loadInput: Copying float tensors (net, inp, corr)");
    std::memcpy(net_data, m_netBuff, m_netBufferSize * sizeof(float));
    logger->info("DPVOUpdate::_loadInput: Copied net tensor ({} bytes)", m_netBufferSize * sizeof(float));
    std::memcpy(inp_data, m_inpBuff, m_inpBufferSize * sizeof(float));
    logger->info("DPVOUpdate::_loadInput: Copied inp tensor ({} bytes)", m_inpBufferSize * sizeof(float));
    std::memcpy(corr_data, m_corrBuff, m_corrBufferSize * sizeof(float));
    logger->info("DPVOUpdate::_loadInput: Copied corr tensor ({} bytes)", m_corrBufferSize * sizeof(float));

    // Copy int32 tensors
    logger->info("DPVOUpdate::_loadInput: Copying int32 tensors (ii, jj, kk)");
    std::memcpy(ii_data, m_iiBuff, m_iiBufferSize * sizeof(int32_t));
    logger->info("DPVOUpdate::_loadInput: Copied ii tensor ({} bytes)", m_iiBufferSize * sizeof(int32_t));
    std::memcpy(jj_data, m_jjBuff, m_jjBufferSize * sizeof(int32_t));
    logger->info("DPVOUpdate::_loadInput: Copied jj tensor ({} bytes)", m_jjBufferSize * sizeof(int32_t));
    std::memcpy(kk_data, m_kkBuff, m_kkBufferSize * sizeof(int32_t));
    logger->info("DPVOUpdate::_loadInput: Copied kk tensor ({} bytes)", m_kkBufferSize * sizeof(int32_t));
    
    // Validate int32 tensor data after copying
    int32_t* ii_data_check = static_cast<int32_t*>(ii_data);
    int32_t* jj_data_check = static_cast<int32_t*>(jj_data);
    int32_t* kk_data_check = static_cast<int32_t*>(kk_data);
    if (m_iiBufferSize > 0 && m_jjBufferSize > 0 && m_kkBufferSize > 0) {
        int32_t ii_min = *std::min_element(ii_data_check, ii_data_check + m_iiBufferSize);
        int32_t ii_max = *std::max_element(ii_data_check, ii_data_check + m_iiBufferSize);
        int32_t jj_min = *std::min_element(jj_data_check, jj_data_check + m_jjBufferSize);
        int32_t jj_max = *std::max_element(jj_data_check, jj_data_check + m_jjBufferSize);
        int32_t kk_min = *std::min_element(kk_data_check, kk_data_check + m_kkBufferSize);
        int32_t kk_max = *std::max_element(kk_data_check, kk_data_check + m_kkBufferSize);
        logger->info("DPVOUpdate::_loadInput: Int32 tensor data ranges - ii=[{}, {}], jj=[{}, {}], kk=[{}, {}]",
                     ii_min, ii_max, jj_min, jj_max, kk_min, kk_max);
        logger->info("DPVOUpdate::_loadInput: Int32 tensor first values - ii[0]={}, jj[0]={}, kk[0]={}",
                     ii_data_check[0], jj_data_check[0], kk_data_check[0]);
    }
    
    // Note: When using ea_tensor_data_for_write(), cache coherency is handled automatically
    // No need to call ea_tensor_sync_cache() separately - the write operation ensures
    // the data is properly flushed to VP memory before inference
    logger->info("DPVOUpdate::_loadInput: All tensors copied successfully (cache coherency handled by ea_tensor_data_for_write)");
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
// Helper function: Reshape inputs for DPVO update model
// =================================================================================================
// Uses pre-allocated buffers to avoid memory allocation overhead
int reshapeInput(
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
    auto logger = spdlog::get("dpvo_update");
    if (!logger) {
        logger = spdlog::get("dpvo");
    }
    
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
    
    if (logger) logger->info("reshapeInput: Reshaping net/inp data, num_edges_to_process={}", num_edges_to_process);
    
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
    if (logger) {
        logger->info("reshapeInput: Net state stats - zero_count={}, nonzero_count={}, min={}, max={}",
                     net_zero_count, net_nonzero_count, net_min, net_max);
    }
    
    // WORKAROUND: If net is all zeros, initialize it from context (inp) to break the cycle
    bool net_all_zero = (net_nonzero_count == 0);
    if (net_all_zero && logger) {
        logger->warn("reshapeInput: Net state is all zeros - initializing from context (inp) as workaround");
        // Initialize net from context (inp) - this gives the model some initial state
        for (int e = 0; e < num_edges_to_process; e++) {
            if (e < 0 || e >= num_active) continue;
            for (int d = 0; d < 384; d++) {
                // Use context as initial net state (scaled down to avoid large values)
                m_net[e][d] = ctx[e * 384 + d] * 0.1f;
            }
        }
        if (logger) logger->info("reshapeInput: Net state initialized from context for {} edges", num_edges_to_process);
    }
    
    // Reshape net and inp data: [num_active, 384] -> [1, 384, 768, 1]
    for (int e = 0; e < num_edges_to_process; e++) {
        // Validate edge index
        if (e < 0 || e >= num_active) {
            if (logger && e < 10) logger->error("reshapeInput: Invalid edge index e={}, num_active={}", e, num_active);
            continue;
        }
        for (int d = 0; d < 384; d++) {
            // YAML layout: [N, C, H, W] = [1, 384, 768, 1]
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
    
    if (logger) {
        logger->info("reshapeInput: Input data ranges - net=[{}, {}], inp=[{}, {}], corr=[{}, {}]",
                     net_input_min, net_input_max, inp_input_min, inp_input_max, corr_input_min, corr_input_max);
    }
    
    if (logger) logger->info("reshapeInput: Net/inp data reshaping completed");
    
    // Reshape correlation: [num_active, D, D, P, P, 2] -> [1, 882, 768, 1]
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
    if (logger) {
        logger->info("reshapeInput: Correlation data stats - zero_count={}, nonzero_count={}, min={}, max={}, size={}",
                     corr_zero_count, corr_nonzero_count, corr_min, corr_max, corr.size());
    }
    
    if (logger) logger->info("reshapeInput: Reshaping correlation data, D={}, D_target={}, offset={}", D, D_target, offset);
    
    // Debug: Check a few source correlation values before reshaping
    if (logger && num_edges_to_process > 0) {
        int sample_e = 0;
        int sample_di = 0, sample_dj = 0, sample_pi = 0, sample_pj = 0, sample_c = 0;
        int sample_src_idx = sample_e * D * D * P * P * 2 +
                            sample_di * D * P * P * 2 +
                            sample_dj * P * P * 2 +
                            sample_pi * P * 2 +
                            sample_pj * 2 +
                            sample_c;
        if (sample_src_idx < static_cast<int>(corr.size())) {
            logger->info("reshapeInput: Sample corr[{}] = {}", sample_src_idx, corr[sample_src_idx]);
        }
    }
    
    int corr_copied_count = 0;
    int corr_skipped_count = 0;
    for (int e = 0; e < num_edges_to_process; e++) {
        // Validate edge index
        if (e < 0 || e >= num_active) {
            if (logger && e < 10) logger->error("reshapeInput: Invalid edge index e={} in correlation reshape, num_active={}", e, num_active);
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
                            
                            // YAML layout: [N, C, H, W] = [1, 882, 768, 1]
                            // Model expects: [1, 882, 768, 1] where 882 = 2 * 7 * 7 * 3 * 3
                            // Index calculation: n * C * H * W + c * H * W + h * W + w
                            // Where: n=0, c=dst_corr_idx (feature index), h=e (edge index), w=0
                            int dst_corr_idx = c * D_target * D_target * P * P +
                                              di * D_target * P * P +
                                              dj * P * P +
                                              pi * P + pj;
                            
                            if (dst_corr_idx < target_corr_dim) {
                                // [N, C, H, W] = [1, 882, 768, 1]
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
    
    if (logger) {
        logger->info("reshapeInput: Correlation reshaping stats - copied={}, skipped={}, total_expected={}",
                     corr_copied_count, corr_skipped_count, num_edges_to_process * 2 * D_target * D_target * P * P);
        // Check a sample of corr_input after reshaping
        float corr_input_sample_min = *std::min_element(corr_input.begin(), corr_input.end());
        float corr_input_sample_max = *std::max_element(corr_input.begin(), corr_input.end());
        int corr_input_nonzero = 0;
        for (size_t i = 0; i < corr_input.size(); i++) {
            if (corr_input[i] != 0.0f) corr_input_nonzero++;
        }
        logger->info("reshapeInput: corr_input after reshaping - min={}, max={}, nonzero_count={}/{}",
                     corr_input_sample_min, corr_input_sample_max, corr_input_nonzero, corr_input.size());
    }
    if (logger) logger->info("reshapeInput: Correlation reshaping completed");
    
    // Copy indices: [num_active] -> [1, 1, 768, 1] (YAML now specifies 4D shape)
    // AMBA tensor shape: [N, C, H, W] = [1, 1, 768, 1]
    // CRITICAL: For zero-padded edges, we need to set valid indices to prevent
    // neighbors_tensor from producing invalid gather indices that cause AMBA CV28 to fail.
    // Strategy: Duplicate the last valid edge's indices for all zero-padded edges.
    if (logger) logger->info("reshapeInput: Copying indices to shape [1, 1, 768, 1], num_edges_to_process={}, num_active={}", num_edges_to_process, num_active);
    
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
            last_ii = static_cast<int32_t>(m_ii[last_valid_e]);
            last_jj = static_cast<int32_t>(m_jj[last_valid_e]);
            last_kk = static_cast<int32_t>(m_kk[last_valid_e]);
        } else {
            // Fallback: use first edge if all have negative kk (shouldn't happen, but be safe)
            if (num_active > 0) {
                last_ii = static_cast<int32_t>(m_ii[0]);
                last_jj = static_cast<int32_t>(m_jj[0]);
                last_kk = 0;  // Force to 0 if all kk are negative
            }
        }
    }
    
    for (int e = 0; e < MODEL_EDGE_COUNT; e++) {
        // AMBA 4D tensor layout: [N, C, H, W] = [1, 1, 768, 1]
        // Index formula: idx = n * C * H * W + c * H * W + h * W + w
        // Where: n=0, c=0, h=e (edge index), w=0
        // idx = 0 * 1 * 768 * 1 + 0 * 768 * 1 + e * 1 + 0 = e
        int idx = 0 * 1 * MODEL_EDGE_COUNT * 1 + 0 * MODEL_EDGE_COUNT * 1 + e * 1 + 0;
        
        if (e < num_edges_to_process && e < num_active) {
            // Valid edge: copy actual indices, but clamp kk to be non-negative
            // CRITICAL: kk can become negative after keyframe removal (m_kk[e] -= PatchGraph::M),
            // but AMBA CV28 requires non-negative indices for gather operations
            int32_t kk_val = static_cast<int32_t>(m_kk[e]);
            if (kk_val < 0) {
                if (logger && e < 10) {
                    logger->warn("reshapeInput: Edge e={} has negative kk={}, clamping to 0", e, kk_val);
                }
                kk_val = 0;  // Clamp to 0 to prevent AMBA CV28 gather failures
            }
            
            ii_input[idx] = static_cast<int32_t>(m_ii[e]);
            jj_input[idx] = static_cast<int32_t>(m_jj[e]);
            kk_input[idx] = kk_val;
            
            // Log first few values for debugging
            if (logger && e < 5) {
                logger->info("reshapeInput: Edge e={}, idx={}, ii={}, jj={}, kk={}", 
                            e, idx, ii_input[idx], jj_input[idx], kk_input[idx]);
            }
        } else {
            // Zero-padded edge: use last valid indices to ensure neighbors_tensor
            // produces valid gather indices (prevents AMBA CV28 gather failures)
            // Ensure last_kk is also non-negative
            int32_t safe_kk = (last_kk >= 0) ? last_kk : 0;
            ii_input[idx] = last_ii;
            jj_input[idx] = last_jj;
            kk_input[idx] = safe_kk;
            
            if (logger && e < num_edges_to_process + 3) {
                logger->info("reshapeInput: Zero-padded edge e={}, using last valid indices: ii={}, jj={}, kk={}", 
                            e, ii_input[idx], jj_input[idx], kk_input[idx]);
            }
        }
    }
    
    // Log sample values from the filled portion
    if (logger && num_edges_to_process > 0) {
        int32_t ii_min = *std::min_element(ii_input.begin(), ii_input.end());
        int32_t ii_max = *std::max_element(ii_input.begin(), ii_input.end());
        int32_t jj_min = *std::min_element(jj_input.begin(), jj_input.end());
        int32_t jj_max = *std::max_element(jj_input.begin(), jj_input.end());
        int32_t kk_min = *std::min_element(kk_input.begin(), kk_input.end());
        int32_t kk_max = *std::max_element(kk_input.begin(), kk_input.end());
        logger->info("reshapeInput: Indices copied - ii=[{}, {}], jj=[{}, {}], kk=[{}, {}] (first {} edges filled, {} padded with last valid indices)",
                    ii_min, ii_max, jj_min, jj_max, kk_min, kk_max, num_edges_to_process, MODEL_EDGE_COUNT - num_edges_to_process);
    }
    if (logger) logger->info("reshapeInput: Indices copied, reshaping completed");
    
    return num_edges_to_process;
}

