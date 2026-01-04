#pragma once
#include <vector>
#include <string>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <cstdint>
#include "dla_config.hpp"

// Ambarella CV28
#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

// DPVO Update Model Prediction Structure
struct DPVOUpdate_Prediction
{
    bool    isProcessed = false;

    float*  netOutBuff;  // [1, 384, 768, 1]
    float*  dOutBuff;    // [1, 2, 768, 1]
    float*  wOutBuff;    // [1, 2, 768, 1]

    DPVOUpdate_Prediction()
        : isProcessed(false),
          netOutBuff(nullptr),
          dOutBuff(nullptr),
          wOutBuff(nullptr)
    {
    }
};

// DPVO Update Model Inference Class
class DPVOUpdate
{
public:
    using WakeCallback = std::function<void()>;
    DPVOUpdate(Config_S* config, WakeCallback wakeFunc = nullptr);
    ~DPVOUpdate();

    // Synchronous inference (main method for sequential execution)
    bool runInference(float* netData, float* inpData, float* corrData, 
                      int32_t* iiData, int32_t* jjData, int32_t* kkData, 
                      int frameIdx, DPVOUpdate_Prediction& pred);
    
    // Multi-threading (optional, for async execution)
    void runThread();
    void stopThread();
    void updateInputData(float* netData, float* inpData, float* corrData, 
                         int32_t* iiData, int32_t* jjData, int32_t* kkData, int frameIdx);
    void notifyProcessingComplete();
    
    // Prediction (for async mode)
    bool getLastestPrediction(DPVOUpdate_Prediction& pred, int& frameIdx);

    // Utility Functions
    bool isInputBufferEmpty() const;
    bool isPredictionBufferEmpty() const;
    void updateTensorPath(const std::string& path);
    bool createDirectory(const std::string& path);
    bool directoryExists(const std::string& path);

    // Debug
    void getDebugProfiles(float& inferenceTime, int& inputBufferSize, int& outBufferSize);

    // Thread Management
    bool m_bInferenced      = true;
    bool m_bProcessed       = true;
    bool m_threadTerminated = false;
    bool m_threadStarted    = false;
    bool m_bDone            = false;

    // Debug
    int         m_saveRawImage       = 0;

private:

#if defined(CV28) || defined(CV28_SIMULATOR)
    bool _checkSavedTensor(int frameIdx);
    bool _loadInput(float* netData, float* inpData, float* corrData, 
                    int32_t* iiData, int32_t* jjData, int32_t* kkData);
    bool _run(float* netData, float* inpData, float* corrData, 
              int32_t* iiData, int32_t* jjData, int32_t* kkData, int frameIdx);
    bool _runInferenceFunc();
    void _initModelIO();
    bool _releaseModel();
    bool _releaseInputTensors();
    bool _releaseOutputTensors();
    bool _releaseTensorBuffers();
#endif

#if defined(SAVE_OUTPUT_TENSOR)
    bool _saveOutputTensor(int frameIdx);
#endif

    // === Thread Management === //
    std::thread             m_threadInference;
    mutable std::mutex      m_pred_mutex;
    mutable std::mutex      m_mutex;
    std::condition_variable m_condition;
    WakeCallback            m_wakeFunc;

    // Input buffer sizes (used in constructor, must be outside conditional)
    size_t m_netBufferSize   = 0;  // 1 * 384 * 768 * 1
    size_t m_inpBufferSize   = 0;  // 1 * 384 * 768 * 1
    size_t m_corrBufferSize  = 0;  // 1 * 882 * 768 * 1
    size_t m_iiBufferSize    = 0;  // 1 * 768 * 1
    size_t m_jjBufferSize    = 0;  // 1 * 768 * 1
    size_t m_kkBufferSize    = 0;  // 1 * 768 * 1
    
    // Output buffer sizes
    size_t m_netOutBufferSize = 0;  // 1 * 384 * 768 * 1
    size_t m_dOutBufferSize   = 0;  // 1 * 2 * 768 * 1
    size_t m_wOutBufferSize   = 0;  // 1 * 2 * 768 * 1

#if defined(CV28) || defined(CV28_SIMULATOR)
    std::string m_modelPathStr;    // Store model path as string
    char*       	m_ptrModelPath  = NULL;
    ea_net_t*   	m_model         = NULL;
    
    // Input tensors
    ea_tensor_t* 	m_inputNetTensor   = NULL;
    ea_tensor_t* 	m_inputInpTensor   = NULL;
    ea_tensor_t* 	m_inputCorrTensor  = NULL;
    ea_tensor_t* 	m_inputIiTensor    = NULL;
    ea_tensor_t* 	m_inputJjTensor    = NULL;
    ea_tensor_t* 	m_inputKkTensor    = NULL;
    
    // Output tensors
    std::vector<ea_tensor_t*> 	m_outputTensors;
    
    // Working buffers for input data
    float* m_netBuff;
    float* m_inpBuff;
    float* m_corrBuff;
    int32_t* m_iiBuff;
    int32_t* m_jjBuff;
    int32_t* m_kkBuff;
    
    // Working buffers for output data
    float* m_netOutBuff;
    float* m_dOutBuff;
    float* m_wOutBuff;
#endif

    // Input tensor names
    std::string m_inputNetTensorName  = "net";
    std::string m_inputInpTensorName  = "inp";
    std::string m_inputCorrTensorName = "corr";
    std::string m_inputIiTensorName   = "ii";
    std::string m_inputJjTensorName   = "jj";
    std::string m_inputKkTensorName    = "kk";

    // Output tensor names
    std::vector<std::string> m_outputTensorList = {
        "net_out", "d_out", "w_out"
    };

    // Input data structure for buffer
    struct InputData {
        float*   netData;
        float*   inpData;
        float*   corrData;
        int32_t* iiData;
        int32_t* jjData;
        int32_t* kkData;
    };

    // Prediction Buffer
    std::deque<std::pair<int, DPVOUpdate_Prediction>> m_predictionBuffer;
    std::deque<std::pair<int, InputData>> m_inputFrameBuffer;
    DPVOUpdate_Prediction m_pred;

    // Read Saved Tensor
    std::string m_tensorPath;

    // Debug
    float m_inferenceTime = 0.0f;
    bool m_estimateTime = false;
};

// Helper function to reshape inputs for DPVO update model
// This function handles all the reshaping and copying logic from DPVO::update
// Uses pre-allocated buffers to avoid memory allocation overhead
int reshapeInput(
    int num_active,
    float (*m_net)[384],  // Pointer to 2D array [MAX_EDGES][384]
    const float* ctx,     // Context data [num_active * 384]
    const std::vector<float>& corr,  // Correlation data [num_active * D * D * P * P * 2]
    const int* m_ii,      // Indices [num_active]
    const int* m_jj,      // Indices [num_active]
    const int* m_kk,      // Indices [num_active]
    int D,                // Correlation window size (typically 8)
    int P,                // Patch size (typically 3)
    // Output buffers (pre-allocated, reused)
    std::vector<float>& net_input,
    std::vector<float>& inp_input,
    std::vector<float>& corr_input,
    std::vector<int32_t>& ii_input,
    std::vector<int32_t>& jj_input,
    std::vector<int32_t>& kk_input,
    const int MODEL_EDGE_COUNT = 768,
    const int CORR_DIM = 882
);

