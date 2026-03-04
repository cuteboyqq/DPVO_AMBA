/*
  (C) 2023-2025 Wistron NeWeb Corporation (WNC) - All Rights Reserved
  YOLOv8 integration for DPVO: synchronous run on same thread as DPVO processing.
*/

#ifndef __YOLOV8__
#define __YOLOV8__

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <fstream>

#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

#include "bounding_box.hpp"
#include "dla_config.hpp"
#include "img_util.hpp"
#include "dataStructures.h"
#include "yolov8_decoder.hpp"

constexpr int YOLOV8_MODEL_HEIGHT = 288;
constexpr int YOLOV8_MODEL_WIDTH  = 512;

class YOLOv8
{
public:
    using WakeCallback = std::function<void()>;
    explicit YOLOv8(Config_S* config, WakeCallback wakeFunc = nullptr);
    ~YOLOv8();

    // Synchronous inference (for DPVO: call from processing thread; fills out_pred; frees previous out_pred buffers)
    bool runSync(ea_tensor_t* imgTensor, int frameIdx, YOLOv8_Prediction& out_pred);
    int getNumAnchorBox() const { return m_numAnchorBox; }
    int getInputWidth() const { return m_inputWidth > 0 ? m_inputWidth : YOLOV8_MODEL_WIDTH; }
    int getInputHeight() const { return m_inputHeight > 0 ? m_inputHeight : YOLOV8_MODEL_HEIGHT; }

    // Optional threaded API (not used by DPVO integration)
    void runThread();
    void stopThread();
    void updateInputFrame(ea_tensor_t* imgTensor, int frameIdx);
    bool getLastestPrediction(YOLOv8_Prediction& pred, int& frameIdx);
    bool isInputBufferEmpty() const;
    bool isPredictionBufferEmpty() const;

#if defined(CV28) || defined(CV28_SIMULATOR)
    std::deque<std::pair<int, ea_tensor_t*>> m_inputFrameBuffer;
#endif
    bool m_bInferenced      = true;
    bool m_bProcessed       = true;
    bool m_threadTerminated  = false;
    bool m_threadStarted     = false;
    bool m_bDone             = false;
    int  m_inputMode         = DETECTION_MODE_FILE;

private:
#if defined(CV28) || defined(CV28_SIMULATOR)
    bool _loadInput(ea_tensor_t* imgTensor);
    int _preProcessingMemory(ea_tensor_t* imgTensor);
    bool _run(ea_tensor_t* imgTensor, int frameIdx);
    void _runInferenceFunc();
    void _initModelIO();
    bool _releaseModel();
    bool _releaseInputTensor();
    bool _releaseOutputTensor();
    bool _releaseTensorBuffers();
    bool _checkSavedTensor(int frameIdx);
#endif
    void notifyProcessingComplete();

    std::thread             m_threadInference;
    mutable std::mutex      m_pred_mutex;
    mutable std::mutex      m_mutex;
    std::condition_variable m_condition;
    WakeCallback            m_wakeFunc;

#if defined(CV28) || defined(CV28_SIMULATOR)
    char*                   m_ptrModelPath  = nullptr;
    std::string             m_modelPathStr;
    ea_net_t*               m_model         = nullptr;
    ea_tensor_t*            m_img           = nullptr;
    ea_tensor_t*            m_inputTensor   = nullptr;
    std::vector<ea_tensor_t*> m_outputTensors;
#endif

    int m_inputChannel  = 0;
    int m_inputHeight   = 0;
    int m_inputWidth    = 0;
    std::string m_inputTensorName = "images";
    int m_numAnchorBox  = 0;
    std::deque<std::pair<int, YOLOv8_Prediction>> m_predictionBuffer;
    YOLOv8_Prediction m_pred;
    int m_boxBufferSize   = 0;
    int m_confBufferSize  = 0;
    int m_classBufferSize = 0;
    float* m_objBoxBuff   = nullptr;
    float* m_objConfBuff  = nullptr;
    float* m_objClsBuff   = nullptr;
    std::vector<std::string> m_outputTensorList = {"dbox", "conf", "cls_id"};
    std::string m_tensorPath;
    bool m_estimateTime = false;
    int m_saveRawImage = 0;
};

#endif
