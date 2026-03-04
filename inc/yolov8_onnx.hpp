/*
 * YOLOv8 ONNX Runtime inference for DPVO.
 * When UseOnnxRuntime = 1 and Yolov8ModelPath points to a .onnx file, use this instead of AMBA YOLOv8.
 */
#pragma once

#include <string>
#include <vector>
#include <memory>
#include "dla_config.hpp"
#include "dataStructures.h"

#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

#ifdef USE_ONNX_RUNTIME
// Forward declare to avoid pulling ONNX headers into all translation units
struct OrtSession;
struct OrtValue;
#endif

class YOLOv8ONNX {
public:
    explicit YOLOv8ONNX(Config_S* config);
    ~YOLOv8ONNX();

    /** Same interface as YOLOv8 (AMBA): synchronous inference, fills out_pred. Caller owns out_pred buffers after return. */
    bool runSync(
#if defined(CV28) || defined(CV28_SIMULATOR)
        ea_tensor_t* imgTensor,
#else
        void* imgTensor,
#endif
        int frameIdx,
        YOLOv8_Prediction& out_pred);

    /** Number of anchor boxes (from model output shape). Used by decoder. */
    int getNumAnchorBox() const { return m_numAnchorBox; }

    /** Input size (for decoder / viewer scaling). */
    int getInputHeight() const { return m_inputHeight; }
    int getInputWidth() const { return m_inputWidth; }

    /** Whether the model loaded successfully. */
    bool isReady() const {
#ifdef USE_ONNX_RUNTIME
        return m_session != nullptr;
#else
        return false;
#endif
    }

private:
    void _initModel();
#if defined(CV28) || defined(CV28_SIMULATOR)
    bool _loadInput(ea_tensor_t* imgTensor, std::vector<float>& input_data);
#else
    bool _loadInput(void* imgTensor, std::vector<float>& input_data);
#endif

    std::string m_modelPath;
    int m_inputHeight  = 288;
    int m_inputWidth   = 512;
    int m_inputChannel = 3;
    int m_numAnchorBox = 0;
    int m_boxBufferSize  = 0;
    int m_confBufferSize = 0;
    int m_classBufferSize = 0;

#ifdef USE_ONNX_RUNTIME
    void* m_session = nullptr;
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::vector<int64_t> m_inputShape;
    std::vector<std::vector<int64_t>> m_outputShapes;
#endif
};
