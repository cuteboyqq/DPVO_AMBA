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

#include "object.hpp"
#include "utils.hpp"

Object::Object()
{
    init(-1);
};

Object::~Object(){};

void Object::init(int _frameStamp)
{
    status = 0;
    bbox   = BoundingBox(-1, -1, -1, -1, -1);
    bboxList.clear();
    distanceToCamera    = -1;
    preDistanceToCamera = -1;
    ttcCounter          = 0;
    currTTC             = -1;
    id                  = -1;
    classStr            = "";

    emaProductW = 0.0f;
    emaProductH = 0.0f;
    m_count     = 0;

    selfIoU = 0.0f;

    distanceBasedOnBoxWidth  = -1.0f;
    distanceBasedOnBoxHeight = -1.0f;
    vDistanceList.clear();
    vTTCList.clear();

    bbox.setFrameStamp(_frameStamp);
    needWarn = false;
}

void Object::updateStatus(int _status)
{
    status = _status;
}

int Object::getStatus() const
{
    return status;
}

int Object::getTrackedCount() const
{
    return m_count;
}

void Object::updateBoundingBox(BoundingBox& _bbox)
{
    bbox = _bbox;
}

float Object::_getSelfIoU() const
{
    return selfIoU;
}

void Object::_resetSelfIoU()
{
    selfIoU = 0.0f;
}

void Object::_updateSelfIoU()
{
    // self IoU: tracklet box self IoU between two consecutive frames
    int size = this->bboxList.size();
    if (size <= 1)
    {
        // need at least 2 box to calculate IoU
        this->selfIoU = 0.0f;
    }

    // this->bboxList.size() > 1, size - 1 > 0
    // self IoU of boxes between this->bboxList[size-2] and this->bboxList[size-1]
    this->selfIoU = imgUtil::iou(this->bboxList[size - 2], this->bboxList[size - 1]);

    // box top-left shifted version
    // BoundingBox box1_shift(this->bboxList[size-2]);
    // BoundingBox box2_shift(this->bboxList[size-1]);
    // box1_shift.shiftTopLeft();
    // box2_shift.shiftTopLeft();

    // selfIoU = imgUtil::iou(box1_shift, box2_shift);
}

// this function only works for vehicle
void Object::_updateDistanceBoxProduct()
{
    /*
    this function only works for vehicle

    // products of rawDistance and calibrated box width, height
    productW = latestBox.rawDistance * latestBox.calibratedW;
    productH = latestBox.rawDistance * latestBox.calibratedH;

    productW and productH can be constants under the following requirements:
        a) the bounding box is in the front / mainlane
        b) bounding box is accurate
        c) vanishing line is accurate

    in real case, when the box is within 40 meters, the vanishing line and box are mostly accurate
    */

    // ALWAYS reset the following distances and self IoU, under all circumstances
    this->distanceBasedOnBoxWidth  = -1.0f;
    this->distanceBasedOnBoxHeight = -1.0f;
    this->_resetSelfIoU();

    if (this->bboxList.size() == 0)
    {
        // no box, return
        return;
    }

    BoundingBox& latestBox = this->bboxList.back();
    if (latestBox.rawDistance < 0.0f)
    {
        // no box or distance is negative. return
        return;
    }

    // now rawDistance is greater or equal than 0
    // bounding box is invalid or too small
    // range of calibrated box is in ref size (0 ~ a few mm)
    if (latestBox.calibratedW <= 0.0001f || latestBox.calibratedH <= 0.0001f)
    {
        return;
    }

    // get distance when the ema is initialized after 2*emaWindowSizeShort frames
    if (this->m_count > 2 * emaWindowSizeShort)
    {
        // get distances BEFORE updating product sum ema
        this->distanceBasedOnBoxWidth  = this->emaProductW / (float)latestBox.calibratedW;
        this->distanceBasedOnBoxHeight = this->emaProductH / (float)latestBox.calibratedH;
    }

    // get self IoU of boxes between two consecutive frames
    this->_updateSelfIoU();
    if (this->selfIoU < selfIoUThresh)
    {
        // Self box IoU between two consecutive frames is too low.
        // Heavy camera shaking / Occlusion are likely happening. Avoid updating product
        return;
    }

    // MUST get distances BEFORE updating product sum ema
    // the products (rawDistance * calibrated box size) are generally constants
    // get productW and productH as products of rawDistance and calibrated box width, height
    float productW = latestBox.rawDistance * latestBox.calibratedW;
    float productH = latestBox.rawDistance * latestBox.calibratedH;
    // cout<<"rawDistance "<<latestBox.rawDistance<<" calibratedW "<<latestBox.calibratedW<<" calibratedH
    // "<<latestBox.calibratedH<<endl;

    // update product ema and count when rawDistance>=0
    this->_updateProductEma(productW, productH);
}

// this function only works for vehicle
float Object::_getDistanceBasedOnBoxWidth() const
{
    // OOP capsulation, interact with method instead of variables
    return this->distanceBasedOnBoxWidth;
}

// this function only works for vehicle
float Object::_getDistanceBasedOnBoxHeight() const
{
    // OOP capsulation, interact with method instead of variables
    return this->distanceBasedOnBoxHeight;
}

// this function only works for vehicle
void Object::_updateProductEma(float productW, float productH)
{
    // update the exponential moving average of products

    float alpha = 0.0f;
    // Assign alpha according to window sizes and to m_count
    // in the beginning, ema window should be very short
    // once the tracking is stable, increase the size of ema window
    if (this->m_count < 2 * emaWindowSizeShort)
    {
        // this->m_count < 2*emaWindowSizeShort
        alpha = 2.0f / (emaWindowSizeShort + 1);
    }
    else
    {
        if (this->m_count < 2 * emaWindowSizeLong)
        {
            // 2*emaWindowSizeShort <= this->m_count < 2*emaWindowSizeLong
            alpha = 2.0f / (emaWindowSizeMid + 1);
        }
        else
        {
            // 2*emaWindowSizeLong <= this->m_count
            alpha = 2.0f / (emaWindowSizeLong + 1);
        }
    }

    // update ema
    // let alpha = 2 / (n + 1)
    // Exponential Moving Average = (C – P) * (2 / (n + 1)) + P = alpha * C +(1-alpha) * P
    // where C and P are current data point and an exponential moving average of the previous period
    emaProductW = alpha * productW + (1 - alpha) * emaProductW;
    emaProductH = alpha * productH + (1 - alpha) * emaProductH;

    // count increment
    this->m_count++;
}

// this function only works for vehicle
float Object::getWeightedProductDistance()
{
    // get weighted distance from ema distance-box product
    float distance    = -1.0f;
    float rawDistance = this->bboxList.back().rawDistance;

    // task must be TRACK_CAR
    // update distance-box product ema
    this->_updateDistanceBoxProduct();
    // get distance based on product
    float distW = this->_getDistanceBasedOnBoxWidth();
    float distH = this->_getDistanceBasedOnBoxHeight();

    // logger->debug("rawDistance {} distW {} distH {}", this->bboxList.back().rawDistance, distW, distH);
    // logger->debug("calibratedW {} calibratedH {}", this->bboxList.back().calibratedW,
    // this->bboxList.back().calibratedH);

    if (distW >= 0.0f && distH >= 0.0f)
    {
        float distBasedOnBox = sqrt(distW * distH);
        // float distBasedOnBox = distW;
        // float distBasedOnBox = (distW + distH)/2.0f;

        distance = productDistanceWeight * distBasedOnBox + (1 - productDistanceWeight) * rawDistance;
    }
    else
    {
        // use rawDistance only
        distance = rawDistance;
    }

    return distance;
}