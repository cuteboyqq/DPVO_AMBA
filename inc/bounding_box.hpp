/*
  (C) 2023-2024 Wistron NeWeb Corporation (WNC) - All Rights Reserved
  Minimal BoundingBox for YOLOv8 decoder/postproc compatibility.
*/

#ifndef __BOUNDING_BOX__
#define __BOUNDING_BOX__

#include <vector>
#include <opencv2/core.hpp>
#include "point.hpp"

#define DEFAULT_POSE_KEYPOINTS_SIZE 30

class BoundingBox
{
public:
    enum BoundingBoxPosition {
        IN_MAIN_LANE = 1,
        RIGHT_EDGE_TOUCHES_LANE = 2,
        LEFT_EDGE_TOUCHES_LANE = 3,
        OUTSIDE_LANE = 4
    };

    BoundingBoxPosition boxPosition = OUTSIDE_LANE;

    BoundingBox();
    BoundingBox(int x1, int y1, int x2, int y2, int label);
    BoundingBox(int x1, int y1, int x2, int y2, int label, std::vector<std::pair<int,int>> pose_kpts);
    BoundingBox(const BoundingBox& box);
    BoundingBox& operator=(const BoundingBox& other);
    ~BoundingBox();

    int                getHeight();
    int                getWidth();
    int                getArea();
    float              getAspectRatio();
    Point              getCenterPoint();
    std::vector<Point> getCornerPoints();
    std::vector<std::pair<int,int>> pose_kpts;
    void shiftTopLeft();
    void setFrameStamp(int _frameStamp);

    int      x1               = -1;
    int      y1               = -1;
    int      x2               = -1;
    int      y2               = -1;
    int      label            = -1;
    int      frameStamp       = 0;
    int      objID            = -1;
    int      boxID            = -1;
    float    rawDistance      = -1.0f;
    float    calibratedW      = -1.0f;
    float    calibratedH      = -1.0f;
    float    confidence       = -1.0f;

private:
    int debugMode = false;
};
#endif
