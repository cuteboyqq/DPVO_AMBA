/*
  (C) 2025-2026 Wistron NeWeb Corporation (WNC) - All Rights Reserved
*/

#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <spdlog/spdlog.h>
#include "yolov8_decoder.hpp"

using namespace cv;
using namespace std;

static bool xyxyobj_compare(const v8xyxy &a, const v8xyxy &b) {
    return a.c_prob > b.c_prob;
}

// Default valid classes for DPVO integration (single class 0)
static const std::vector<int> validClasses = {0};
static const std::unordered_map<int, int> classToIndex = []() {
    std::unordered_map<int, int> m;
    for (size_t i = 0; i < validClasses.size(); ++i)
        m[validClasses[i]] = static_cast<int>(i);
    return m;
}();

template<typename Container, typename T>
static bool contains(const Container &c, const T &value) {
    return std::find(c.begin(), c.end(), value) != c.end();
}

YOLOv8_Decoder::YOLOv8_Decoder(int inputH, int inputW, std::string loggerStr)
{
    m_loggerStr = std::move(loggerStr);
    m_inputH = inputH;
    m_inputW = inputW;
}

YOLOv8_Decoder::~YOLOv8_Decoder()
{
    spdlog::drop(m_loggerStr);
}

unsigned int YOLOv8_Decoder::decodeBox(
    const float *m_detection_box_buff,
    const float *m_detection_conf_buff,
    const float *m_detection_class_buff,
    int numBbox,
    float confThreshold,
    float iouThreshold,
    int num_Cls,
    std::vector<std::vector<v8xyxy>> &classwisePicked)
{
    auto logger = spdlog::get(m_loggerStr);
    if (classwisePicked.empty())
        classwisePicked = std::vector<std::vector<v8xyxy>>(num_Cls);
    std::vector<std::vector<v8xyxy>> bboxlist(num_Cls);
    getCandidates(m_detection_box_buff, m_detection_conf_buff, m_detection_class_buff,
                  numBbox, confThreshold, bboxlist);
    doNMS(bboxlist, iouThreshold, classwisePicked, num_Cls);
    int size = 0;
    for (int cls = 0; cls < num_Cls; cls++)
        size += static_cast<int>(classwisePicked[cls].size());
    return size;
}

int YOLOv8_Decoder::getCandidates(const float *detectionBox, const float *detectionConf, const float *detectionClass,
                                 int numBbox, const float conf_thr, vector<vector<v8xyxy>> &bbox_list)
{
    auto logger = spdlog::get(m_loggerStr);
    if (numBbox <= 0) {
        if (logger) logger->error("Invalid numBbox: {}", numBbox);
        return 0;
    }
    int numThresholdedBbx = 0;
    for (int i = 0; i < numBbox; i++) {
        if (detectionConf[i] > conf_thr && contains(validClasses, static_cast<int>(detectionClass[i]))) {
            v8xyxy box;
            box.x1 = static_cast<int>(detectionBox[i]);
            box.y1 = static_cast<int>(detectionBox[numBbox + i]);
            box.x2 = static_cast<int>(detectionBox[numBbox * 2 + i]);
            box.y2 = static_cast<int>(detectionBox[numBbox * 3 + i]);
            box.c = static_cast<int>(detectionClass[i]);
            box.c_prob = detectionConf[i];
            box.area = (box.x2 - box.x1) * (box.y2 - box.y1);
            bbox_list[classToIndex.at(box.c)].push_back(box);
            numThresholdedBbx++;
        }
    }
    return numThresholdedBbx;
}

int YOLOv8_Decoder::getIntersectArea(const v8xyxy &a, const v8xyxy &b)
{
    int intersection_w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    if (intersection_w <= 0) return 0;
    int intersection_h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (intersection_h <= 0) return 0;
    return intersection_w * intersection_h;
}

float YOLOv8_Decoder::iou(const v8xyxy &a, const v8xyxy &b)
{
    int intersection_area = getIntersectArea(a, b);
    if (intersection_area == 0) return 0.0f;
    return static_cast<float>(intersection_area) / static_cast<float>(a.area + b.area - intersection_area);
}

float YOLOv8_Decoder::getBboxOverlapRatio(const v8xyxy &boxA, const v8xyxy &boxB)
{
    int iouArea = getIntersectArea(boxA, boxB);
    return static_cast<float>(iouArea) / static_cast<float>(boxA.area);
}

int YOLOv8_Decoder::doNMS(std::vector<std::vector<v8xyxy>> &bboxlist, const float iou_thr,
                          std::vector<std::vector<v8xyxy>> &classwisePicked, int num_Cls)
{
    auto logger = spdlog::get(m_loggerStr);
    for (int cls = 0; cls < num_Cls; cls++) {
        std::vector<v8xyxy> &currBboxList = bboxlist[cls];
        if (!classwisePicked[cls].empty())
            classwisePicked[cls].clear();
        std::vector<v8xyxy> &currPicked = classwisePicked[cls];
        sort(currBboxList.begin(), currBboxList.end(), xyxyobj_compare);
        for (size_t i = 0; i < currBboxList.size(); ++i) {
            bool keep = true;
            for (size_t j = 0; j < currPicked.size(); ++j) {
                float iouValue = iou(currBboxList[i], currPicked[j]);
                if (iouValue >= iou_thr) {
                    currBboxList[i].c_prob = 0.0f;
                    keep = false;
                    break;
                }
                float bboxOverlapRatio = getBboxOverlapRatio(currBboxList[i], currPicked[j]);
                if (bboxOverlapRatio >= overlapThreshold) {
                    currBboxList[i].c_prob = 0.0f;
                    keep = false;
                    break;
                }
            }
            if (keep)
                currPicked.push_back(currBboxList[i]);
        }
    }
    int size = 0;
    for (int cls = 0; cls < num_Cls; cls++)
        size += static_cast<int>(classwisePicked[cls].size());
    return size;
}
