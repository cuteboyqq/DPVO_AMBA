/*
  (C) 2023-2024 Wistron NeWeb Corporation (WNC) - All Rights Reserved

  This software and its associated documentation are the confidential and
  proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
  may not be copied, modified, distributed, or otherwise disclosed to third
  parties without the express written consent of the Company.

  Unauthorized reproduction, distribution, or disclosure of this software and
  its associated documentation or the information contained herein is a
  violation of applicable laws and may result in severe legal penalties.
*/

#ifndef __DATA_STRUCTURES__
#define __DATA_STRUCTURES__

#include <vector>
#include <map>
#include <opencv2/core.hpp>

// DEPRECATED: bounding_box.hpp, object.hpp removed
// #include "bounding_box.hpp"
// #include "object.hpp"
#include "syslog.h"

#define MAX_NUM_OBJ 100 // 32


enum ObjectDetectionLabel
{
    HUMAN           = 0,
    CAT           = 1,
    DOG           = 2,
    // VEHICLE         = 1
};

enum LaneDetectionLabel
{
    BIG_VEHICLE         = 0,
    RIDER       = 1,
    FACE        = 2,

};

enum PoseDetectionLabel
{
    SKELETON         = 0
};

enum TrackingTask
{
    TRACK_HUMAN       = 0,
    TRACK_CAR         = 1
};

enum DETECTION_MODE
{
    DETECTION_MODE_LIVE = 0,
    DETECTION_MODE_FILE = 1,
    DETECTION_MODE_HISTORICAL = 2
};

struct YOLOv8_Prediction
{
    bool    isProcessed = false;

    float*  objBoxBuff;
    float*  objConfBuff;
    float*  objClsBuff;

    float*  laneBoxBuff;
    float*  laneConfBuff;
    float*  laneClsBuff;

    float*  poseBoxBuff;
    float*  poseConfBuff;
    float*  poseClsBuff;
    float*  poseKptsBuff;

    cv::Mat img;

    YOLOv8_Prediction()
        : isProcessed(false),
          objBoxBuff(nullptr),
          objConfBuff(nullptr),
          objClsBuff(nullptr),

          laneBoxBuff(nullptr),
          laneConfBuff(nullptr),
          laneClsBuff(nullptr),

          poseBoxBuff(nullptr),
          poseConfBuff(nullptr),
          poseClsBuff(nullptr),
          poseKptsBuff(nullptr)
    {
    }
};

struct ROI
{
    int       x1 = 0;
    int       y1 = 0;
    int       x2 = 0;
    int       y2 = 0;
    ROI()
        : x1(),
          y1(),
          x2(),
          y2(){};
};

struct WNC_APP_Results
{
    bool                isDetectLine      = false;
    int                 eventType         = 0;
    int                 frameID           = 0;
    // DEPRECATED: BoundingBox and Object removed
    // std::vector<BoundingBox>  detectObjList;
    // std::vector<Object>       trackObjList;
    // std::vector<BoundingBox>  faceObjList;
    // std::vector<BoundingBox>  vehicleObjList;
    // std::vector<BoundingBox>  riderObjList;
    // std::vector<BoundingBox>  poseObjList;

    WNC_APP_Results()
        : isDetectLine(),
          eventType()
          // DEPRECATED: trackObjList, faceObjList, etc. removed
          // trackObjList(),
          // faceObjList(),
          // vehicleObjList(),
          // riderObjList(),
          // poseObjList()
    {   
        // DEPRECATED: reserve calls removed
        // detectObjList.reserve(MAX_NUM_OBJ);
        // trackObjList.reserve(MAX_NUM_OBJ);
        // faceObjList.reserve(MAX_NUM_OBJ);
        // vehicleObjList.reserve(MAX_NUM_OBJ);
        // riderObjList.reserve(MAX_NUM_OBJ);
        // poseObjList.reserve(MAX_NUM_OBJ);
    }
};

struct DebugProfile
{
    int   yoloADAS_InputBufferSize  = 0;
    int   yoloADAS_OutputBufferSize = 0;
    int   postProc_InputBufferSize  = 0;
    int   postProc_OutputBufferSize = 0;
    int   adasResultBufferSize      = 0;
    float AIInfrerenceTime          = 0.0f;
};

#endif
