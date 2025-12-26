#include "net.hpp"
#include "correlation_kernel.hpp"
#include <sys/stat.h>
#include <cstdio>

Patchifier::Patchifier(int patch_size, int DIM)
    : m_patch_size(patch_size), m_DIM(DIM)
{}

// Forward pass: fill fmap, imap, gmap, patches, clr
void Patchifier::forward(
    const uint8_t* image,
    int H, int W,
    float* fmap,     // [128, H, W]
    float* imap,     // [DIM, H, W]
    float* gmap,     // [M, 128, P, P]
    float* patches,  // [M, 3, P, P]
    uint8_t* clr,    // [M, 3]
    int M
) {
    // ------------------------------------------------
    // 1. Image → float grid (for patches)
    // ------------------------------------------------
    std::vector<float> grid(3 * H * W);
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < H*W; i++)
            grid[c*H*W + i] = image[c*H*W + i] / 255.0f;

    // ------------------------------------------------
    // 2. Generate RANDOM coords (Python RANDOM mode)
    // ------------------------------------------------
    std::vector<float> coords(M * 2);
    for (int m = 0; m < M; m++) {
        coords[m*2 + 0] = 1 + rand() % (W - 2);
        coords[m*2 + 1] = 1 + rand() % (H - 2);
    }

    // ------------------------------------------------
    // 3. Patchify grid → patches (RGB)
    // ------------------------------------------------
    patchify_cpu_safe(
        grid.data(), coords.data(),
        M, 3, H, W,
        m_patch_size / 2,
        patches
    );

    // ------------------------------------------------
    // 4. Patchify fmap → gmap
    // ------------------------------------------------
    patchify_cpu_safe(
        fmap, coords.data(),
        M, 128, H, W,
        m_patch_size / 2,
        gmap
    );

    // ------------------------------------------------
    // 5. imap sampling (radius = 0)
    // ------------------------------------------------
    patchify_cpu_safe(
        imap, coords.data(),
        M, m_DIM, H, W,
        0,
        imap   // reuse buffer shape [M, DIM, 1, 1]
    );

    // ------------------------------------------------
    // 6. Color for visualization
    // ------------------------------------------------
    for (int m = 0; m < M; m++) {
        int x = static_cast<int>(coords[m*2 + 0]);
        int y = static_cast<int>(coords[m*2 + 1]);
        for (int c = 0; c < 3; c++)
            clr[m*3 + c] = image[c*H*W + y*W + x];
    }
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

    m_ptrModelPath = const_cast<char *>(config->modelPath.c_str());

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

    m_inputNetTensor  = NULL;
    m_inputInpTensor  = NULL;
    m_inputCorrTensor = NULL;
    m_inputIiTensor   = NULL;
    m_inputJjTensor   = NULL;
    m_inputKkTensor   = NULL;

    m_outputTensors = std::vector<ea_tensor_t*>(m_outputTensorList.size());

    // Allocate working buffers for input data
    m_netBuff  = new float[m_netBufferSize];
    m_inpBuff  = new float[m_inpBufferSize];
    m_corrBuff = new float[m_corrBufferSize];
    m_iiBuff   = new int32_t[m_iiBufferSize];
    m_jjBuff   = new int32_t[m_jjBufferSize];
    m_kkBuff   = new int32_t[m_kkBufferSize];

    // Allocate working buffers for output data
    m_netOutBuff = new float[m_netOutBufferSize];
    m_dOutBuff   = new float[m_dOutBufferSize];
    m_wOutBuff   = new float[m_wOutBufferSize];
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

    for (size_t i = 0; i < m_outputTensorList.size(); ++i) {
        logger->info("Output Name: {}", m_outputTensorList[i]);
        rval = ea_net_config_output(m_model, m_outputTensorList[i].c_str());
    }

    // Configure model path
    logger->info("-------------------------------------------");
    logger->info("Load Model");
    logger->info("Model Path: {}", m_ptrModelPath);

    // Check if model path exists before loading
    if (m_ptrModelPath == nullptr || strlen(m_ptrModelPath) == 0) {
        logger->error("Model path is null or empty");
        return;
    }
    
    // Check if file exists
    FILE* file = fopen(m_ptrModelPath, "r");
    if (file == nullptr) {
        logger->error("Model file does not exist at path: {}", m_ptrModelPath);
        return;
    }
    fclose(file);
    
    logger->info("Model file exists, proceeding with loading");

    rval = ea_net_load(m_model, EA_NET_LOAD_FILE, (void *)m_ptrModelPath, 1/*max_batch*/);

    // Get input tensors
    logger->info("-------------------------------------------");
    logger->info("Create Model Input Tensors");
    
    m_inputNetTensor  = ea_net_input(m_model, m_inputNetTensorName.c_str());
    m_inputInpTensor  = ea_net_input(m_model, m_inputInpTensorName.c_str());
    m_inputCorrTensor = ea_net_input(m_model, m_inputCorrTensorName.c_str());
    m_inputIiTensor   = ea_net_input(m_model, m_inputIiTensorName.c_str());
    m_inputJjTensor   = ea_net_input(m_model, m_inputJjTensorName.c_str());
    m_inputKkTensor   = ea_net_input(m_model, m_inputKkTensorName.c_str());
    
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

    for (size_t i=0; i<ea_net_output_num(m_model); i++)
    {
        const char* tensorName = static_cast<const char*>(ea_net_output_name(m_model, i));
        const size_t* tensorShape = static_cast<const size_t*>(ea_tensor_shape(ea_net_output_by_index(m_model, i)));
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
    std::string netOutFilePath = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor0.bin";
    std::string dOutFilePath    = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor1.bin";
    std::string wOutFilePath    = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor2.bin";

    // Function to load tensor data from a binary file
    auto loadTensorFromBinaryFile = [](const std::string& filePath, float* buffer, size_t size) {
        std::ifstream inFile(filePath, std::ios::binary);
        if (inFile.is_open()) {
            inFile.read(reinterpret_cast<char*>(buffer), size * sizeof(float));
            inFile.close();
            return true;
        } else {
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

    std::string netOutFilePath = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor0.bin";
    std::string dOutFilePath    = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor1.bin";
    std::string wOutFilePath    = m_tensorPath + "/frame_" + std::to_string(frameIdx-1) + "_tensor2.bin";

    logger->debug("========================================");
    logger->debug("Model Frame Index: {}", frameIdx);
    logger->debug("Out Buffer Size: {}", m_predictionBuffer.size());
    logger->debug("========================================");

    // Save tensors to binary files
    auto saveTensorToBinaryFile = [](const std::string& filePath, float* buffer, size_t size) {
        std::ofstream outFile(filePath, std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(reinterpret_cast<char*>(buffer), size * sizeof(float));
            outFile.close();
        } else {
            std::cerr << "Failed to open file: " << filePath << std::endl;
        }
    };
    
    // Save each tensor to its corresponding binary file
    saveTensorToBinaryFile(netOutFilePath, m_pred.netOutBuff, m_netOutBufferSize);
    saveTensorToBinaryFile(dOutFilePath,   m_pred.dOutBuff,   m_dOutBufferSize);
    saveTensorToBinaryFile(wOutFilePath,   m_pred.wOutBuff,   m_wOutBufferSize);

    return true;
}
#endif

bool DPVOUpdate::_releaseInputTensors()
{
    if (m_inputNetTensor)  { m_inputNetTensor  = nullptr; }
    if (m_inputInpTensor)  { m_inputInpTensor  = nullptr; }
    if (m_inputCorrTensor) { m_inputCorrTensor = nullptr; }
    if (m_inputIiTensor)   { m_inputIiTensor   = nullptr; }
    if (m_inputJjTensor)   { m_inputJjTensor   = nullptr; }
    if (m_inputKkTensor)   { m_inputKkTensor  = nullptr; }
    return true;
}

bool DPVOUpdate::_releaseOutputTensors()
{
    for (size_t i=0; i<m_outputTensorList.size(); i++)
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

    m_netBuff  = nullptr;
    m_inpBuff  = nullptr;
    m_corrBuff = nullptr;
    m_iiBuff   = nullptr;
    m_jjBuff   = nullptr;
    m_kkBuff   = nullptr;

    m_netOutBuff = nullptr;
    m_dOutBuff   = nullptr;
    m_wOutBuff   = nullptr;

    return true;
}
#endif
// =================================================================================================

// =================================================================================================
// Synchronous Inference (Public API)
// =================================================================================================
bool DPVOUpdate::runInference(float* netData, float* inpData, float* corrData, 
                              int32_t* iiData, int32_t* jjData, int32_t* kkData, 
                              int frameIdx, DPVOUpdate_Prediction& pred)
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
    pred.dOutBuff   = m_pred.dOutBuff;
    pred.wOutBuff   = m_pred.wOutBuff;
    
    // Clear m_pred buffers so they don't get double-freed
    m_pred.netOutBuff = nullptr;
    m_pred.dOutBuff   = nullptr;
    m_pred.wOutBuff   = nullptr;
    
    return true;
}

// =================================================================================================
// Inference Entrypoint (Internal)
// =================================================================================================
bool DPVOUpdate::_run(float* netData, float* inpData, float* corrData, 
                      int32_t* iiData, int32_t* jjData, int32_t* kkData, int frameIdx)
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
    logger->debug(" =========== Buffer Size: {} ===========", m_inputFrameBuffer.size());

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
        m_pred.dOutBuff   = new float[m_dOutBufferSize];
        m_pred.wOutBuff   = new float[m_wOutBufferSize];

        // Copy output tensors to prediction buffers
        std::memcpy(m_pred.netOutBuff, (float *)m_netOutBuff, m_netOutBufferSize * sizeof(float));
        std::memcpy(m_pred.dOutBuff,   (float *)m_dOutBuff,   m_dOutBufferSize * sizeof(float));
        std::memcpy(m_pred.wOutBuff,   (float *)m_wOutBuff,   m_wOutBufferSize * sizeof(float));
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
            m_pred.dOutBuff   = new float[m_dOutBufferSize];
            m_pred.wOutBuff   = new float[m_wOutBufferSize];

            // Copy output tensors to prediction buffers
            std::memcpy(m_pred.netOutBuff, (float *)ea_tensor_data(m_outputTensors[0]), m_netOutBufferSize * sizeof(float));
            std::memcpy(m_pred.dOutBuff,   (float *)ea_tensor_data(m_outputTensors[1]), m_dOutBufferSize * sizeof(float));
            std::memcpy(m_pred.wOutBuff,   (float *)ea_tensor_data(m_outputTensors[2]), m_wOutBufferSize * sizeof(float));

#if defined(SAVE_OUTPUT_TENSOR)
            _saveOutputTensor(frameIdx);
#endif
        }
    }

    m_bProcessed = true;

    logger->debug(" ============= Model Frame Index: {} =============", frameIdx);
    logger->debug(" ============= Out Buffer Size: {} =============", m_predictionBuffer.size());

    if (m_estimateTime)
    {
        time_2 = std::chrono::high_resolution_clock::now();
        logger->info("[Inference]: {} ms",
            std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_1).count() / (1000.0 * 1000));
    }
#endif
    // ==================================

    time_2           = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_0);
    m_inferenceTime  = static_cast<float>(nanoseconds.count()) / 1e9f;

    if (m_estimateTime)
    {
        logger->info("[Total]: {} ms", m_inferenceTime);
        logger->info("-----------------------------------------");
    }

    logger->debug("End AI Model Part");
    logger->debug("========================================");
    m_bDone = true;

    return true;
}

void DPVOUpdate::runThread()
{
    m_threadInference = std::thread(&DPVOUpdate::_runInferenceFunc, this);
    return;
}

void DPVOUpdate::stopThread()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_threadTerminated = true;
    }

    m_condition.notify_all(); // Wake up the thread if it's waiting
    if (m_threadInference.joinable())
    {
        m_threadInference.join();
    }
}

bool DPVOUpdate::_runInferenceFunc()
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("dpvo_update");
#else
    auto logger = spdlog::get("dpvo_update");
#endif

    while (!m_threadTerminated)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_condition.wait(lock,
            [this]() { return m_threadTerminated || (!m_inputFrameBuffer.empty() && m_threadStarted); });

        if (m_threadTerminated)
            break;

        // Read input data from buffer
        auto    pair     = m_inputFrameBuffer.front();
        int     frameIdx = pair.first;
        InputData inputData = pair.second;
        m_inputFrameBuffer.pop_front();
        lock.unlock();

        // Perform AI Inference
        if (!_run(inputData.netData, inputData.inpData, inputData.corrData, 
                  inputData.iiData, inputData.jjData, inputData.kkData, frameIdx))
        {
            logger->error("Failed in AI Inference");
            // Free allocated memory before returning
            delete[] inputData.netData;
            delete[] inputData.inpData;
            delete[] inputData.corrData;
            delete[] inputData.iiData;
            delete[] inputData.jjData;
            delete[] inputData.kkData;
            return true;
        }
        
        // Free allocated memory after inference
        delete[] inputData.netData;
        delete[] inputData.inpData;
        delete[] inputData.corrData;
        delete[] inputData.iiData;
        delete[] inputData.jjData;
        delete[] inputData.kkData;
    }
    return true;
}

void DPVOUpdate::notifyProcessingComplete()
{
    if (m_wakeFunc)
        m_wakeFunc();
}

// =================================================================================================
// Load Inputs
// =================================================================================================
bool DPVOUpdate::_loadInput(float* netData, float* inpData, float* corrData, 
                            int32_t* iiData, int32_t* jjData, int32_t* kkData)
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
    std::memcpy(m_netBuff,  netData,  m_netBufferSize * sizeof(float));
    std::memcpy(m_inpBuff,  inpData,  m_inpBufferSize * sizeof(float));
    std::memcpy(m_corrBuff, corrData, m_corrBufferSize * sizeof(float));
    std::memcpy(m_iiBuff,   iiData,   m_iiBufferSize * sizeof(int32_t));
    std::memcpy(m_jjBuff,   jjData,   m_jjBufferSize * sizeof(int32_t));
    std::memcpy(m_kkBuff,   kkData,   m_kkBufferSize * sizeof(int32_t));

    // Copy data to input tensors
#if defined(CV28) || defined(CV28_SIMULATOR)
    // Copy float tensors
    std::memcpy(ea_tensor_data(m_inputNetTensor),  m_netBuff,  m_netBufferSize * sizeof(float));
    std::memcpy(ea_tensor_data(m_inputInpTensor),  m_inpBuff,  m_inpBufferSize * sizeof(float));
    std::memcpy(ea_tensor_data(m_inputCorrTensor), m_corrBuff, m_corrBufferSize * sizeof(float));
    
    // Copy int32 tensors
    std::memcpy(ea_tensor_data(m_inputIiTensor),   m_iiBuff,   m_iiBufferSize * sizeof(int32_t));
    std::memcpy(ea_tensor_data(m_inputJjTensor),   m_jjBuff,   m_jjBufferSize * sizeof(int32_t));
    std::memcpy(ea_tensor_data(m_inputKkTensor),   m_kkBuff,   m_kkBufferSize * sizeof(int32_t));
#endif

    if (m_estimateTime)
    {
        time_1 = std::chrono::high_resolution_clock::now();
        logger->info("[Load Input]: {} ms",
            std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
    }

    return true;
}

void DPVOUpdate::updateInputData(float* netData, float* inpData, float* corrData, 
                                 int32_t* iiData, int32_t* jjData, int32_t* kkData, int frameIdx)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("dpvo_update");
#else
    auto logger = spdlog::get("dpvo_update");
#endif

    std::unique_lock<std::mutex> lock(m_mutex);

    // Allocate and copy input data
    InputData inputData;
    inputData.netData  = new float[m_netBufferSize];
    inputData.inpData  = new float[m_inpBufferSize];
    inputData.corrData = new float[m_corrBufferSize];
    inputData.iiData   = new int32_t[m_iiBufferSize];
    inputData.jjData   = new int32_t[m_jjBufferSize];
    inputData.kkData   = new int32_t[m_kkBufferSize];

    std::memcpy(inputData.netData,  netData,  m_netBufferSize * sizeof(float));
    std::memcpy(inputData.inpData,  inpData,  m_inpBufferSize * sizeof(float));
    std::memcpy(inputData.corrData, corrData, m_corrBufferSize * sizeof(float));
    std::memcpy(inputData.iiData,   iiData,   m_iiBufferSize * sizeof(int32_t));
    std::memcpy(inputData.jjData,   jjData,   m_jjBufferSize * sizeof(int32_t));
    std::memcpy(inputData.kkData,   kkData,   m_kkBufferSize * sizeof(int32_t));

    m_inputFrameBuffer.emplace_back(frameIdx, inputData);
    m_threadStarted = true;
    m_bDone         = false;
    lock.unlock();

    m_condition.notify_one();
}

// =================================================================================================
// Post Processing
// =================================================================================================
bool DPVOUpdate::getLastestPrediction(DPVOUpdate_Prediction& pred, int& frameIdx)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("dpvo_update");
#else
    auto logger = spdlog::get("dpvo_update");
#endif

    std::unique_lock<std::mutex> lock(m_mutex);
    const auto                   bufferSize = m_predictionBuffer.size();

    if (bufferSize == 0)
        return false;

    if (bufferSize > 0)
    {
        auto pair = m_predictionBuffer.front();
        frameIdx  = pair.first;
        pred      = pair.second;
        m_predictionBuffer.pop_front();
        return true;
    }

    lock.unlock();

    logger->warn("buffSize is negative = {}", m_predictionBuffer.size());
    return false;
}

// =================================================================================================
// Utility Functions
// =================================================================================================
void DPVOUpdate::getDebugProfiles(float& inferenceTime, int& inputBufferSize, int& outputBufferSize)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    inputBufferSize  = m_inputFrameBuffer.size();
    outputBufferSize = m_predictionBuffer.size();
    inferenceTime    = m_inferenceTime;
    lock.unlock();
}

bool DPVOUpdate::isInputBufferEmpty() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_inputFrameBuffer.empty();
}

bool DPVOUpdate::isPredictionBufferEmpty() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_predictionBuffer.empty();
}

void DPVOUpdate::updateTensorPath(const std::string& path)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("dpvo_update");
#else
    auto logger = spdlog::get("dpvo_update");
#endif

    m_tensorPath = path; // Assuming m_tensorPath is a member variable to store the path
    logger->info("Updated tensor path to: {}", m_tensorPath);
}
