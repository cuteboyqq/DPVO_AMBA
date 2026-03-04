/*
  (C) 2025-2026 Wistron NeWeb Corporation (WNC) - All Rights Reserved
  YOLOv8 decoder: raw model outputs -> bounding boxes (v8xyxy) with NMS.
*/

#ifndef __YOLOV8_DECODER__
#define __YOLOV8_DECODER__

#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

constexpr float overlapThreshold = 0.60f;

struct v8xyxy
{
    int   x1     = 0;
    int   y1     = 0;
    int   x2     = 0;
    int   y2     = 0;
    float c_prob = 0.0f;
    int   c      = 0;
    int   area   = 0;
};

class YOLOv8_Decoder
{
public:
    YOLOv8_Decoder(int inputH, int inputW, std::string loggerStr);
    ~YOLOv8_Decoder();

    unsigned int decodeBox(const float *m_detection_box_buff,
                           const float *m_detection_conf_buff,
                           const float *m_detection_cls_buff,
                           int numBbox,
                           float confThreshold,
                           float iouThreshold,
                           int num_Cls,
                           std::vector<std::vector<v8xyxy>> &out);

private:
    float iou(const v8xyxy &a, const v8xyxy &b);
    int getIntersectArea(const v8xyxy &a, const v8xyxy &b);
    float getBboxOverlapRatio(const v8xyxy &boxA, const v8xyxy &boxB);
    int doNMS(std::vector<std::vector<v8xyxy>> &bboxList, const float iouThreshold,
              std::vector<std::vector<v8xyxy>> &classwisePicked, int num_Cls);
    int getCandidates(const float *detectionBox,
                      const float *detectionConf,
                      const float *detectionClass,
                      int numBbox,
                      float confThreshold, std::vector<std::vector<v8xyxy>> &bboxList);

    int m_inputH = 320;
    int m_inputW = 320;
    bool debugMode = false;
    std::string m_loggerStr;
};

#endif
