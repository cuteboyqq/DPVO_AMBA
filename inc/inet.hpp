#pragma once
#include <string>
#include "dla_config.hpp"

// Ambarella CV28
#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

// INet Inference Class
class INetInference {
public:
    INetInference(Config_S* config);
    ~INetInference();
    // image: original uint8 image [C, H, W] with values in range [0, 255]
    // AMBA CV28 model will handle normalization internally (std: 256, normalizes by dividing by 256)
    bool runInference(const uint8_t* image, int H, int W, float* imap_out);
    
    // Getters for model input dimensions
    int getInputHeight() const;
    int getInputWidth() const;
    int getOutputHeight() const;
    int getOutputWidth() const;
    
private:
#if defined(CV28) || defined(CV28_SIMULATOR)
    void _initModelIO();
    bool _releaseModel();
    bool _loadInput(const uint8_t* image, int H, int W);  // Original uint8 input [0, 255] - AMBA CV28 handles normalization
    
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

