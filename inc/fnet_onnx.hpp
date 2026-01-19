#pragma once

#include <string>
#include <vector>
#include <memory>
#include "dla_config.hpp"

// Ambarella CV28
#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

// ONNX Runtime FNet Inference Class
// This provides the same interface as FNetInference but uses ONNX Runtime
class FNetInferenceONNX {
public:
    FNetInferenceONNX(Config_S* config);
    ~FNetInferenceONNX();
    
    // Interface matching FNetInference
    // Input: imgTensor is treated as uint8_t* image data [C, H, W] format
#if defined(CV28) || defined(CV28_SIMULATOR)
    bool runInference(ea_tensor_t* imgTensor, float* fmap_out);
#else
    bool runInference(void* imgTensor, float* fmap_out);  // Fallback for non-CV28
#endif
    
    // Getters for model input dimensions
    int getInputHeight() const { return m_inputHeight; }
    int getInputWidth() const { return m_inputWidth; }
    int getOutputHeight() const { return m_outputHeight; }
    int getOutputWidth() const { return m_outputWidth; }
    
private:
    void _initModel();
#if defined(CV28) || defined(CV28_SIMULATOR)
    bool _loadInput(ea_tensor_t* imgTensor, std::vector<float>& input_data);
#else
    bool _loadInput(void* imgTensor, std::vector<float>& input_data);  // Fallback for non-CV28
#endif
    
    std::string m_modelPath;
    int m_inputHeight = 0;
    int m_inputWidth = 0;
    int m_inputChannel = 3;
    int m_outputHeight = 0;
    int m_outputWidth = 0;
    int m_outputChannel = 128;
    
    // ONNX Runtime session (will be void* to avoid requiring ONNX headers in header)
    void* m_session = nullptr;
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::vector<int64_t> m_inputShape;
    std::vector<int64_t> m_outputShape;
};

