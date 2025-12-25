/*
  (C) 2023-2025 Wistron NeWeb Corporation (WNC) - All Rights Reserved

  This software and its associated documentation are the confidential and
  proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
  may not be copied, modified, distributed, or otherwise disclosed to third
  parties without the express written consent of the Company.

  Unauthorized reproduction, distribution, or disclosure of this software and
  its associated documentation or the information contained herein is a
  violation of applicable laws and may result in severe legal penalties.
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

// Ambarella CV28
#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

// WNC
#include "bounding_box.hpp" //TODO:
#include "dla_config.hpp"
#include "img_util.hpp" //TODO:
#include "dataStructures.h"
#include "logger.hpp"
#include "utils.hpp"
#include "yolov8_decoder.hpp"

#ifdef __cplusplus
extern "C" {
#endif

constexpr int MODEL_HEIGHT     = 288;
constexpr int MODEL_WIDTH      = 512;
constexpr int FRAME_HEIGHT     = 288;
constexpr int FRAME_WIDTH      = 512;
constexpr int NUM_OBJ_CLASSES  = 2;
constexpr int NUM_LANE_CLASSES = 3;
constexpr int NUM_POSE_CLASSES = 1;

class YOLOv8
{
public:
    using WakeCallback = std::function<void()>;
    YOLOv8(Config_S* config, WakeCallback wakeFunc = nullptr);
    ~YOLOv8();

    // Multi-threading
    void runThread();
    void stopThread();
    void updateInputFrame(ea_tensor_t* imgTensor, int frameIdx);
    void notifyProcessingComplete();
    
    // Prediction
    bool getLastestPrediction(YOLOv8_Prediction& pred, int& frameIdx);

    // Utility Functions
    bool isInputBufferEmpty() const;
    bool isPredictionBufferEmpty() const;
    void updateTensorPath(const std::string& path);
    bool createDirectory(const std::string& path);
    bool directoryExists(const std::string& path);

    // Historical Mode //TODO:
    // bool run_sequential(cv::Mat& imgFrame, YOLOv8_Prediction& pred);

    // Debug
    void getDebugProfiles(float& inferenceTime, int& inputBufferSize, int& outBufferSize);

    // Inference
#if defined(CV28) || defined(CV28_SIMULATOR)
    std::deque<std::pair<int, ea_tensor_t*>> m_inputFrameBuffer;
#else
    std::deque<std::pair<int, cv::Mat>> m_inputFrameBuffer;
#endif

    // Thread Management
    bool m_bInferenced      = true;
    bool m_bProcessed       = true;
    bool m_threadTerminated = false;
    bool m_threadStarted    = false;
    bool m_bDone            = false;

    // Historical Mode
    int         m_inputMode          = DETECTION_MODE_LIVE; // ADAS_MODE_LIVE or ADAS_MODE_HISTORICAL
    std::string m_dbg_dateTime       = "";
    std::string m_dbg_rawImgsDirPath = "";

    // Debug
    int         m_saveRawImage       = 0;

private:

#if defined(CV28) || defined(CV28_SIMULATOR)
    bool _checkSavedTensor(int frameIdx);
    bool _loadInput(ea_tensor_t* imgTensor);
    int _preProcessingMemory(ea_tensor_t* imgTensor);
    bool _run(ea_tensor_t* imgTensor, int frameIdx);
    bool _runInferenceFunc();
    void _initModelIO();
    bool _releaseModel();
    bool _releaseInputTensor();
    bool _releaseOutputTensor();
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

#if defined(CV28) || defined(CV28_SIMULATOR)
    char*       	m_ptrModelPath  = NULL;
    ea_net_t*   	m_model         = NULL;
    ea_tensor_t* 	m_img           = NULL; //TODO:
    ea_tensor_t* 	m_inputTensor   = NULL;
    std::vector<ea_tensor_t*> 	m_outputTensors;
#endif

    // I/O information
    int m_inputChannel    = 0;
    int m_inputHeight     = 0;
    int m_inputWidth      = 0;
    std::string m_inputTensorName = "images";

    // Number of BBox
    int m_numAnchorBox    = 0;

    // Number of keypoints
    int m_numKeypoints    = 17; 

    // Prediction Buffer
    std::deque<std::pair<int, YOLOv8_Prediction>> m_predictionBuffer;
    YOLOv8_Prediction m_pred;

    // Buffer Size
    int m_boxBufferSize   = 0;
    int m_confBufferSize  = 0;
    int m_classBufferSize = 0;
    int m_kptsBufferSize  = 0;

    // Buffer
    float* m_objBoxBuff;
    float* m_objConfBuff;
    float* m_objClsBuff;

    float* m_laneBoxBuff;
    float* m_laneConfBuff;
    float* m_laneClsBuff;

    float* m_poseBoxBuff;
    float* m_poseConfBuff;
    float* m_poseClsBuff;
    float* m_poseKptsBuff;

    // Output Tensor List
    // output_names=['obj_box', 'obj_conf', 'obj_cls', 'lane_box', 'lane_conf', 'lane_cls', 'pose_box', 'pose_conf', 'pose_cls', 'pose_kpt']

    std::vector<std::string> m_outputTensorList = {
        // "dbox",  "conf",  "cls_id"              // Object Detection
        "obj_box", "obj_conf", "obj_cls",   // Object Detection
        "lane_box", "lane_conf", "lane_cls", // Lane Detection
        "pose_box", "pose_conf", "pose_cls", "pose_kpt" // Pose Detection
    };
    // Read Saved Tensor
    std::string m_tensorPath;

    // Debug
    float m_inferenceTime = 0.0f;
    bool m_estimateTime = false;
};

#ifdef __cplusplus
}
#endif

#endif

