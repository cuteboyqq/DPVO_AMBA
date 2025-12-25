/*
  (C) 2025-2026 Wistron NeWeb Corporation (WNC) - All Rights Reserved

  This software and its associated documentation are the confidential and
  proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
  may not be copied, modified, distributed, or otherwise disclosed to third
  parties without the express written consent of the Company.

  Unauthorized reproduction, distribution, or disclosure of this software and
  its associated documentation or the information contained herein is a
  violation of applicable laws and may result in severe legal penalties.
*/

#ifndef __YOLOv8_DECODER__
#define __YOLOv8_DECODER__

#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "yolov8.hpp"
#include "img_util.hpp"

using namespace std;

constexpr float overlapThreshold = 0.60;//0.60
#define MAX_KEYPOINTS_NUM 17

struct keypoints
{
    int   x;
    int   y;
    float c;
};

struct v8xyxy
{
    int   x1     = 0;
    int   y1     = 0;
    int   x2     = 0;
    int   y2     = 0;
    float c_prob = 0.0f; // class probability, apply sigmoid()
    int   c      = 0;    // class
    int   area   = 0;    // area according to x1,x2,y1,y2
    std::vector<std::pair<int,int>> pose_kpts;

    // Constructor to reserve space
    v8xyxy() {
        pose_kpts.reserve((MAX_KEYPOINTS_NUM*3)); // Reserve space for MAX_KEYPOINTS_NUM * 2 keypoints
    }

};

class YOLOv8_Decoder
{
public:
    YOLOv8_Decoder(int inputH, int inputW, std::string loggerStr);
    ~YOLOv8_Decoder();

    ///////////////////////////
    /// Member Functions
    //////////////////////////
    unsigned int decodeBox(const float *m_detection_box_buff, 
                        const float *m_detection_conf_buff,
                        const float *m_detection_cls_buff, 
                        int numBbox, 
                        float confThreshold, 
                        float iouThreshold,
                        int num_Cls, 
                        std::vector<std::vector<v8xyxy>> &out);

    
    unsigned int decodeBoxAndKpt(
                        const float *m_detection_box_buff, 
                        const float *m_detection_conf_buff,
                        const float *m_detection_class_buff, 
                        const float *m_keypoint_buff,
                        int numBbox, 
                        float confThreshold,
                        float iouThreshold, 
                        int num_Cls,
                        std::vector<std::vector<v8xyxy>> &classwisePicked);
    

    int getCandidatesWithKpt(
                        const float *detectionBox, 
                        const float *detectionConf, 
                        const float *detectionClass,
                        const float *detectionKeypoints,
                        int numBbox, 
                        const float conf_thr, 
                        vector<vector<v8xyxy>> &bbox_list);
    
    int getCandidatesWithKpt_v2(
                    const float *detectionBox, 
                    const float *detectionConf, 
                    const float *detectionClass,
                    const float *detectionKeypoints,
                    int numBbox, 
                    const float conf_thr, 
                    vector<vector<v8xyxy>> &bbox_list);
    ///////////////////////////
    /// Member Variables
    //////////////////////////
    int m_numOfKeypoints = 17;
private:
    ///////////////////////////
    /// Member Functions
    //////////////////////////
    float iou(const v8xyxy &a, const v8xyxy &b);
    int getIntersectArea(const v8xyxy &a, const v8xyxy &b);
    float getBboxOverlapRatio(const v8xyxy &boxA, const v8xyxy &boxB);
    int doNMS(std::vector<std::vector<v8xyxy>> &bboxList, const float iouThreshold,
              std::vector<std::vector<v8xyxy>> &classwisePicked, int num_Cls);

    int getCandidates(const float *detectionBox, 
                      const float *detectionConf, 
                      const float *detectionClass,
                      int numBbox,
                      float confThreshold, 
                      std::vector<std::vector<v8xyxy>> &bboxList);

    ///////////////////////////
    /// Member Variables
    //////////////////////////
    int m_inputH = 288; //TODO:
    int m_inputW = 512; //TODO:

    // for YOLO-ADAS v0.6.0+
    std::vector<int> m_rawKeypointXList;
    std::vector<int> m_rawKeypointYList;
    bool m_isRawKeypointInitialized = false;

    // debug
    bool   debugMode = true;
    string m_loggerStr;
};

#endif
