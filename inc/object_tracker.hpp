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

#ifndef __OBJECT_TRACKER__
#define __OBJECT_TRACKER__

#include <algorithm>
#include <string>
#include <chrono>
#include <cmath>
#include <vector>
#include <map>
#include <unordered_set>
#include <deque>
#include <numeric>
#include <limits>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>
#include "opencv2/features2d.hpp"

#include "img_util.hpp"
#include "dla_config.hpp"
#include "dataStructures.h"
#include "logger.hpp"
// #include "dataType.hpp" //TODO:
#include "sTrack.h"
#include "baseTracker.h"
#include "yolov8.hpp"

using namespace std;
constexpr int MAX_BBOX_LIST_SIZE = 100;


class ObjectTracker
{
public:
    ObjectTracker(Config_S *_config, string _task);
    ~ObjectTracker();

    void run(vector<BoundingBox> &bboxList, unordered_map<int, Object> &objectUmap, vector<Object> &trackedObjList);
    int m_maxTracking = 0; // Max tracking object at same time

private:
    BaseTracker        m_BaseTracker;
    unordered_set<int> activeIdSet;
    std::vector<STrack> _boundingBoxToSTrack(std::vector<BoundingBox>& bboxList, std::string logStr);
    void _tracking(std::vector<BoundingBox>& bboxList);
    void _updateObjectUmap(unordered_map<int, Object> &objectUmap);
    void _distanceCalc(unordered_map<int, Object> &objectUmap, vector<Object> &trackedObjList);

    int           m_task;
    int           m_frameWidth = 0;
    int           m_frameRate  = 30; // Fps
    int           m_minWidth   = 0;  // BoundingBox's width must > m_minWidth
    int           m_minHeight  = 0;  // BoundingBox's height must > m_minHeight
    unsigned long m_frameStamp = 0;  // Some kind of timestamp, increase frame by frame

    float m_cameraHeight = 0.0f; // Camera height

    string m_loggerStr;
    bool   m_estimateTime = false;

    // === Threshold === //
    const float m_tTrackThresh = 0.1; // track thresh (detections score >= track_thresh, detections_low score <= track_thresh)
    const float m_tActivateThresh = 0.1; // for tracklet activation 0.3
    const float m_tMatchThresh    = 0.7; // for hungarian matching 0.7

    vector<STrack> output_stracks;
};

#endif
