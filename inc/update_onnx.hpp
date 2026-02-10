#pragma once

#include <string>
#include <vector>
#include <memory>
#include "update.hpp"  // For DPVOUpdate_Prediction struct
#include "dla_config.hpp"  // For Config_S typedef

#ifdef USE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

// ONNX Runtime Update Model Inference Class
// This provides the same interface as DPVOUpdate but uses ONNX Runtime
class DPVOUpdateONNX {
public:
    DPVOUpdateONNX(Config_S* config);
    ~DPVOUpdateONNX();
    
    // Interface matching DPVOUpdate::runInference
    bool runInference(float* netData, float* inpData, float* corrData, 
                      float* iiData, float* jjData, float* kkData, 
                      int frameIdx, DPVOUpdate_Prediction& pred);
    
    // Reshape inputs (same interface as DPVOUpdate)
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
        std::vector<float>& net_input,
        std::vector<float>& inp_input,
        std::vector<float>& corr_input,
        std::vector<float>& ii_input,
        std::vector<float>& jj_input,
        std::vector<float>& kk_input,
        const int MODEL_EDGE_COUNT = 360,
        const int CORR_DIM = 882
    );
    
private:
    void _initModel();
    bool _loadInput(float* netData, float* inpData, float* corrData, 
                    float* iiData, float* jjData, float* kkData,
                    std::vector<std::vector<float>>& input_data);
    
    std::string m_modelPath;
    int m_maxEdge = 360;
    
    // ONNX Runtime session
    void* m_session = nullptr;
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::vector<std::vector<int64_t>> m_inputShapes;
    std::vector<std::vector<int64_t>> m_outputShapes;
#ifdef USE_ONNX_RUNTIME
    std::vector<ONNXTensorElementDataType> m_inputTypes;  // Store input tensor element types
#endif
    
    // Buffer sizes
    size_t m_netOutBufferSize = 0;
    size_t m_dOutBufferSize = 0;
    size_t m_wOutBufferSize = 0;
};

