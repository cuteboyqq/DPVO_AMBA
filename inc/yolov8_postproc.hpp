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

#ifndef __YOLOv8_POSTPROC__
#define __YOLOv8_POSTPROC__

#include <chrono>
#include <iostream>
#include <string>
#include <deque>
#include <condition_variable>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// WNC
#include "bounding_box.hpp"
#include "yolov8.hpp"
#include "yolov8_decoder.hpp"
#include "img_util.hpp"
#include "utils.hpp"
#include "dla_config.hpp"
#include "logger.hpp"

using namespace std;

constexpr int    FILE_MODE            = 0;
constexpr int    MAX_OBJ_BOX          = 500;
constexpr int    MAX_AREA_BOX         = 10; //30
constexpr size_t MAX_BUFFER_SIZE      = 5;

struct POST_PROC_RESULTS
{
    // Traffic Object Bounding Boxes
    std::vector<BoundingBox> humanBBoxList;
    std::vector<BoundingBox> vehicleBBoxList;
    std::vector<BoundingBox> faceBBoxList;
    std::vector<BoundingBox> skeletonBBoxList;
    std::vector<BoundingBox> riderBBoxList;
    bool                     bCopied = false;
    cv::Mat img;

    POST_PROC_RESULTS()
        : humanBBoxList(),
          vehicleBBoxList(),
          faceBBoxList(),
          skeletonBBoxList(),
          riderBBoxList(),
          bCopied()
    {
        // Reserve capacity for vectors
        humanBBoxList.reserve(MAX_OBJ_BOX);
        vehicleBBoxList.reserve(MAX_OBJ_BOX);
        faceBBoxList.reserve(MAX_OBJ_BOX);
        skeletonBBoxList.reserve(MAX_OBJ_BOX);
        riderBBoxList.reserve(MAX_OBJ_BOX);
        bCopied = false;
    }
};

class YOLOv8_POSTPROC
{
public:
    using WakeCallback = std::function<void()>;

    YOLOv8_POSTPROC(Config_S* config, WakeCallback wakeFunc = nullptr);
    ~YOLOv8_POSTPROC();

    // =========================
    // Thread Management
    // =========================
    void runThread();
    void stopThread();
    void notifyProcessingComplete();
    void updatePredictionBuffer(YOLOv8_Prediction& pred, int frameIdx);
    bool getLastestResult(POST_PROC_RESULTS& result, int& resultFrameIdx);
    bool isInputBufferEmpty() const;
    bool isOutputBufferEmpty() const;
    void getDebugProfiles(int& inputBufferSize, int& outputBufferSize);

    // =========================
    // Historical Mode
    // =========================
    // Sequential
    bool run_sequential(YOLOv8_Prediction& pred, POST_PROC_RESULTS& res);

    // =========================
    // Thread Management
    // =========================
    bool m_threadTerminated = false;
    bool m_threadStarted    = false;

    // =========================
    // Prediction Related
    // =========================
    YOLOv8_Prediction m_preds;
    bool m_bDone            = false;

private:
    // =========================
    // Thread Management
    // =========================
    void _runProcessingFunc();

    // =========================
    // Post Processing Related
    // =========================

    // Check model output buffers
    bool _areObjectBuffersValid();

    // Check prediction
    bool _checkPrediction(const YOLOv8_Prediction& pred);

    // Post Processing
    bool _postProcessing(YOLOv8_Prediction& pred);

    // =========================
    // Object Detection
    // =========================
    bool _objectPostProcessing();
    bool _getHumanBoundingBox(vector<BoundingBox>& _outBboxList, float confidenceHuman);
    bool _getVehicleBoundingBox(vector<BoundingBox>& _outBboxList, float confidenceVehicle);
    bool _getFaceBoundingBox(vector<BoundingBox>& _outBboxList, float confidence);
    bool _getSkeletonBoundingBox(vector<BoundingBox>& _outBboxList, float confidence);
    bool _getRiderBoundingBox(vector<BoundingBox>& _outBboxList, float confidence);

    // =========================
    // Lane Detection
    // =========================
    bool _lanePostProcessing();
    bool _areLaneBuffersValid();
    bool _arePoseBuffersValid();

    // =========================
    // Output Prediction Related
    // =========================
    void _updateResultBuffer(int predFrameIdx);

    // =========================
    // Thread Management
    // =========================
    std::thread             m_threadPostProcessing;
    mutable std::mutex      m_mutex;
    mutable std::mutex      m_result_mutex;
    std::condition_variable m_condition;
    std::condition_variable m_result_cond;
    WakeCallback m_wakeFunc;
    
    // =========================
    // Config
    // =========================
    int m_saveRawImage = 0;
    int m_inputMode    = 0;

    // =========================
    // Input Related
    // =========================

    // Input Image
    cv::Mat m_img;

    // Input Frame Index
    int m_frameIndex = 0;

    // Input Prediction Buffer
    std::deque<std::pair<int, YOLOv8_Prediction>> m_predictionBuffer;

    // =========================
    // Output Related
    // =========================

    // Number of BBox
    int m_numAnchorBox    = 0;

    // Number of Keypoints
    int m_numKeypoints    = 17;

    // Output Tensor Decoder
    YOLOv8_Decoder* m_decoder;

    // Output Tensor Buffer Size
    int m_boxBufferSize   = 0;
    int m_confBufferSize  = 0;
    int m_classBufferSize = 0;
    int m_kptsBufferSize  = 0;

    // Object Detection Buffer
    // float* = pointer to float(s).
    float* m_objBoxBuff;
    float* m_objConfBuff;
    float* m_objClsBuff;

    // Lane Line Detection Buffer
    float* m_laneBoxBuff;
    float* m_laneConfBuff;
    float* m_laneClsBuff;

    // Pose Detection Buffer
    float* m_poseBoxBuff;
    float* m_poseConfBuff;
    float* m_poseClsBuff;
    float* m_poseKptsBuff;

    // Bounding Box
    int m_numObjBox  = 0;
    int m_numLaneBox = 0;
    int m_numPoseBox = 0;
    std::vector<std::vector<v8xyxy>> m_objectDetectionOut;
    std::vector<std::vector<v8xyxy>> m_laneDetectionOut;
    std::vector<std::vector<v8xyxy>> m_poseDetectionOut;

    // Thresholding
    const float m_confidenceThreshold  = 0.1; // original : 0.1
    const float m_iouThreshold         = 0.50; //0.5
    float       m_humanConfidence      = 0.1; //0.1
    float       m_vehicleConfidence    = 0.1; //0.1
    float       m_faceConfidence       = 0.1; //0.1
    float       m_skeletonConfidence   = 0.1; // 0.1
    float       m_riderConfidence      = 0.1; // 0.1

    // Output Result Buffer
    std::deque<std::pair<int, POST_PROC_RESULTS>> m_resultBuffer;

    // Estimate Time
    bool m_estimateTime = false;
};

#endif
