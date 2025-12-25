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

#pragma once

#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "point.hpp"
#include "bounding_box.hpp"
#include "img_util.hpp"

class Object
{
public:
    Object();
    ~Object();

    void init(int _frameStamp);
    int  getStatus() const;
    void updateStatus(int status);
    void updateBoundingBox(BoundingBox &_box);
    int  getTrackedCount() const;
    float getWeightedProductDistance();

    int                      id;                  // Human ID
    BoundingBox              bbox;                // Bounding Box (x1, y1, x2, y2, label)
    std::vector<BoundingBox> bboxList;            // Bounding Box list
    float                    distanceToCamera;    // Distance to camera
    float                    preDistanceToCamera; // Distance to camera
    float                    currTTC;             // Current Time To Collision (TTC) in float-point seconds
    int                      ttcCounter;          // Counter for TTC
    bool                     needWarn;            // Does this object need to be warning?
    std::string              classStr;
    std::vector<float> vDistanceList;
    std::vector<float> vTTCList;

    // Debug
    int debugMode = false;

    // ema window sizes
    static constexpr int emaWindowSizeShort = 10;
    static constexpr int emaWindowSizeMid = 50;
    static constexpr int emaWindowSizeLong = 100;

    // self IoU threshold
    static constexpr float selfIoUThresh = 0.75f;

    // coeff for weighted distance
    static constexpr float productDistanceWeight = 0.3; // must range in [0.0 ~ 1.0]

private:
    void _updateProductEma(float productW, float productH);
    void _updateSelfIoU();
    void _resetSelfIoU();
    void _updateDistanceBoxProduct();
    float _getDistanceBasedOnBoxWidth() const;
    float _getDistanceBasedOnBoxHeight() const;
    float _getSelfIoU() const;

    int m_count;                // count of tracklet frames

    int    status;              // 0: deactivate / 1: activate
    float selfIoU;              // self IoU of boxes between two consecutive frames
    double emaProductW;         // exponential moving average of distance-box width product
    double emaProductH;         // exponential moving average of distance-box height product

    float distanceBasedOnBoxWidth;  // distance (distance-box product ema)/(calibrated width)
    float distanceBasedOnBoxHeight; // distance (distance-box product ema)/(calibrated height)
};
