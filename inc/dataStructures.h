/*
  (C) 2023-2024 Wistron NeWeb Corporation (WNC) - All Rights Reserved
  Minimal subset for YOLOv8 integration with DPVO.
*/

#ifndef __DATA_STRUCTURES__
#define __DATA_STRUCTURES__

#include <vector>
#include <opencv2/core.hpp>

enum DETECTION_MODE
{
    DETECTION_MODE_LIVE = 0,
    DETECTION_MODE_FILE = 1,
    DETECTION_MODE_HISTORICAL = 2
};

struct ROI
{
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

struct YOLOv8_Prediction
{
    bool    isProcessed = false;
    float*  objBoxBuff  = nullptr;
    float*  objConfBuff  = nullptr;
    float*  objClsBuff   = nullptr;
    cv::Mat img;

    YOLOv8_Prediction()
        : isProcessed(false),
          objBoxBuff(nullptr),
          objConfBuff(nullptr),
          objClsBuff(nullptr)
    {
    }
};

#endif
