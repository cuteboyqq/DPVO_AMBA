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

#ifndef __BOUNDING_BOX__
#define __BOUNDING_BOX__

#include <vector>
#include <opencv2/core.hpp>
#include "point.hpp"


#define DEFAULT_POSE_KEYPOINTS_SIZE 51

class BoundingBox
{
public:
    enum BoundingBoxPosition {
        IN_MAIN_LANE = 1,
        RIGHT_EDGE_TOUCHES_LANE = 2,
        LEFT_EDGE_TOUCHES_LANE = 3,
        OUTSIDE_LANE = 4
    };

    BoundingBoxPosition boxPosition = OUTSIDE_LANE; // Default value

    BoundingBox();
    BoundingBox(int x1, int y1, int x2, int y2, int label);
    BoundingBox(int x1, int y1, int x2, int y2, int label, std::vector<std::pair<int,int>> pose_kpts);
    BoundingBox(int x1, int y1, int x2, int y2, int label, std::vector<std::pair<int,int>> pose_kpts, std::string skeletonAction); // Alister add 2025-01-12
    BoundingBox(const BoundingBox& box);
    BoundingBox& operator=(const BoundingBox& other); // Add copy assignment operator
    void updatePoseKeypoints(const std::vector<std::pair<int,int>>& new_pose_kpts); 
    ~BoundingBox();

    int                getHeight();
    int                getWidth();
    int                getArea() const;
    float              getAspectRatio();
    Point              getCenterPoint();
    std::vector<Point> getCornerPoints();
    std::vector<std::pair<int,int>> pose_kpts; // Add pose keypoints, Alister add 2025-01-12
    std::vector<std::pair<int, int>> prev_pose_kpts; // previous keypoints
    std::string skeletonAction;               // new: store classified action
    void shiftTopLeft();
    void setFrameStamp(int _frameStamp);

    // === Default value === //
    int      x1               = -1;    // Bounding Box x1
    int      y1               = -1;    // Bounding Box y1
    int      x2               = -1;    // Bounding Box x2
    int      y2               = -1;    // Bounding Box y2
    int      label            = -1;    // Bounding Box label
    int      frameStamp       = 0;     // Some kind of timestamp
    int      objID            = -1;    // Assigned by object tracking
    int      boxID            = -1;    // Assigned by object tracking
    float    rawDistance      = -1.0f; // Raw distance without regression or component
    float    calibratedW      = -1.0f;
    float    calibratedH      = -1.0f;
    float    confidence       = -1.0f; // Confidence score

// In order to save memory, disable roi for now.
#ifdef ADAS_KPTMATCH
    cv::Rect roi;                      // 2D region-of-interest in image coordinates
#endif

private:
    int debugMode = false;
};
#endif
