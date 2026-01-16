#pragma once
#include <string>
#include "dla_config.hpp"

// Ambarella CV28
#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

// FNet Inference Class
class FNetInference {
public:
    FNetInference(Config_S* config);
    ~FNetInference();
    // Tensor-based version (preferred, avoids conversion)
    bool runInference(ea_tensor_t* imgTensor, float* fmap_out);
    
    // Getters for model input dimensions
    int getInputHeight() const;
    int getInputWidth() const;
    int getOutputHeight() const;
    int getOutputWidth() const;
    
private:
#if defined(CV28) || defined(CV28_SIMULATOR)
    void _initModelIO();
    bool _releaseModel();
    bool _loadInput(ea_tensor_t* imgTensor);  // Tensor input (preferred, avoids conversion)
    
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

