#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iostream>
#include <string>
#include <thread>
#include <fstream>
#include <mutex>
#include <functional>

// Ambarella CV28
#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

// WNC
#include "dla_config.hpp"
#include "logger.hpp"
#include "utils.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// Simple struct for patch
struct Patch {
    std::vector<std::vector<std::vector<float>>> data; // [C][D][D]
    Patch(int C, int D) : data(C, std::vector<std::vector<float>>(D, std::vector<float>(D, 0.0f))) {}
};

struct ColorPatch {
    std::array<uint8_t,3> rgb;
};

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

// Simple inference classes for fnet and inet
class FNetInference {
public:
    FNetInference(Config_S* config);
    ~FNetInference();
    bool runInference(const uint8_t* image, int H, int W, float* fmap_out);
    
    // Getters for model input dimensions
    int getInputHeight() const { 
#if defined(CV28) || defined(CV28_SIMULATOR)
        return m_inputHeight; 
#else
        return 0;
#endif
    }
    int getInputWidth() const { 
#if defined(CV28) || defined(CV28_SIMULATOR)
        return m_inputWidth; 
#else
        return 0;
#endif
    }
    int getOutputHeight() const { 
#if defined(CV28) || defined(CV28_SIMULATOR)
        return m_outputHeight; 
#else
        return 0;
#endif
    }
    int getOutputWidth() const { 
#if defined(CV28) || defined(CV28_SIMULATOR)
        return m_outputWidth; 
#else
        return 0;
#endif
    }
    
private:
#if defined(CV28) || defined(CV28_SIMULATOR)
    void _initModelIO();
    bool _releaseModel();
    bool _loadInput(const uint8_t* image, int H, int W);
    
    std::string m_modelPathStr;    // Store model path as string
    char* m_ptrModelPath = nullptr;
    ea_net_t* m_model = nullptr;
    ea_tensor_t* m_inputTensor = nullptr;
    ea_tensor_t* m_outputTensor = nullptr;
    std::string m_inputTensorName = "images";
    std::string m_outputTensorName = "fmap";
    int m_inputHeight = 0;
    int m_inputWidth = 0;
    int m_inputChannel = 3;
    int m_outputHeight = 0;
    int m_outputWidth = 0;
    int m_outputChannel = 128;
    float* m_outputBuffer = nullptr;
#endif
};

class INetInference {
public:
    INetInference(Config_S* config);
    ~INetInference();
    bool runInference(const uint8_t* image, int H, int W, float* imap_out);
    
    // Getters for model input dimensions
    int getInputHeight() const { 
#if defined(CV28) || defined(CV28_SIMULATOR)
        return m_inputHeight; 
#else
        return 0;
#endif
    }
    int getInputWidth() const { 
#if defined(CV28) || defined(CV28_SIMULATOR)
        return m_inputWidth; 
#else
        return 0;
#endif
    }
    int getOutputHeight() const { 
#if defined(CV28) || defined(CV28_SIMULATOR)
        return m_outputHeight; 
#else
        return 0;
#endif
    }
    int getOutputWidth() const { 
#if defined(CV28) || defined(CV28_SIMULATOR)
        return m_outputWidth; 
#else
        return 0;
#endif
    }
    
private:
#if defined(CV28) || defined(CV28_SIMULATOR)
    void _initModelIO();
    bool _releaseModel();
    bool _loadInput(const uint8_t* image, int H, int W);
    
    std::string m_modelPathStr;    // Store model path as string
    char* m_ptrModelPath = nullptr;
    ea_net_t* m_model = nullptr;
    ea_tensor_t* m_inputTensor = nullptr;
    ea_tensor_t* m_outputTensor = nullptr;
    std::string m_inputTensorName = "images";
    std::string m_outputTensorName = "imap";
    int m_inputHeight = 0;
    int m_inputWidth = 0;
    int m_inputChannel = 3;
    int m_outputHeight = 0;
    int m_outputWidth = 0;
    int m_outputChannel = 384;
    float* m_outputBuffer = nullptr;
#endif
};

class Patchifier {
public:
    Patchifier(int patch_size = 3, int DIM = 64);
    Patchifier(int patch_size, int DIM, Config_S* config); // Constructor with models
    ~Patchifier();
    
    void setModels(Config_S* fnetConfig, Config_S* inetConfig);

    // forward function
    void forward(const uint8_t* image, int H, int W,
                 float* fmap, float* imap, float* gmap,
                 float* patches, uint8_t* clr,
                 int patches_per_image = 8);

private:
    int m_patch_size;
    int m_DIM;
    
    // Model inference objects
    std::unique_ptr<FNetInference> m_fnet;
    std::unique_ptr<INetInference> m_inet;
    
    // Temporary buffers for model outputs
    std::vector<float> m_fmap_buffer;
    std::vector<float> m_imap_buffer;
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

#ifdef __cplusplus
}
#endif
