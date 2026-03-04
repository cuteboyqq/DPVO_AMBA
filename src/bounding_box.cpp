/*
  (C) 2023-2024 Wistron NeWeb Corporation (WNC) - All Rights Reserved
*/

#include "bounding_box.hpp"

BoundingBox::BoundingBox() : x1(-1), y1(-1), x2(-1), y2(-1), label(-1)
{
    pose_kpts.reserve(DEFAULT_POSE_KEYPOINTS_SIZE);
}

BoundingBox::BoundingBox(int x1, int y1, int x2, int y2, int label) : x1(x1), y1(y1), x2(x2), y2(y2), label(label)
{
    pose_kpts.reserve(DEFAULT_POSE_KEYPOINTS_SIZE);
}

BoundingBox::BoundingBox(
    int x1, int y1, int x2, int y2, int label,
    std::vector<std::pair<int,int>> pose_kpts)
{
    this->x1 = x1;
    this->y1 = y1;
    this->x2 = x2;
    this->y2 = y2;
    this->label = label;
    this->confidence = 0.0f;
    this->objID = -1;
    this->boxID = -1;
    this->pose_kpts = std::move(pose_kpts);
    if (this->pose_kpts.capacity() < DEFAULT_POSE_KEYPOINTS_SIZE)
        this->pose_kpts.reserve(DEFAULT_POSE_KEYPOINTS_SIZE);
}

BoundingBox::BoundingBox(const BoundingBox& box)
{
    this->x1          = box.x1;
    this->y1          = box.y1;
    this->x2          = box.x2;
    this->y2          = box.y2;
    this->label       = box.label;
    this->rawDistance = box.rawDistance;
    this->calibratedW = box.calibratedW;
    this->calibratedH = box.calibratedH;
    this->confidence  = box.confidence;
    this->boxPosition = box.boxPosition;
    this->frameStamp  = box.frameStamp;
    this->objID       = box.objID;
    this->boxID       = box.boxID;
    this->pose_kpts   = box.pose_kpts;
}

BoundingBox& BoundingBox::operator=(const BoundingBox& other)
{
    if (this != &other) {
        x1          = other.x1;
        y1          = other.y1;
        x2          = other.x2;
        y2          = other.y2;
        label       = other.label;
        rawDistance = other.rawDistance;
        calibratedW = other.calibratedW;
        calibratedH = other.calibratedH;
        confidence  = other.confidence;
        boxPosition = other.boxPosition;
        frameStamp  = other.frameStamp;
        objID       = other.objID;
        boxID       = other.boxID;
        pose_kpts   = other.pose_kpts;
    }
    return *this;
}

BoundingBox::~BoundingBox() {}

int BoundingBox::getHeight() { return y2 - y1; }
int BoundingBox::getWidth()  { return x2 - x1; }
int BoundingBox::getArea()   { return (x2 - x1) * (y2 - y1); }
float BoundingBox::getAspectRatio() { return static_cast<float>(y2 - y1) / (x2 - x1); }
Point BoundingBox::getCenterPoint() { return Point((x1 + x2) / 2, (y1 + y2) / 2); }

std::vector<Point> BoundingBox::getCornerPoints()
{
    std::vector<Point> cornerPoints;
    cornerPoints.reserve(4);
    cornerPoints.emplace_back(Point(x1, y1));
    cornerPoints.emplace_back(Point(x2, y1));
    cornerPoints.emplace_back(Point(x1, y2));
    cornerPoints.emplace_back(Point(x2, y2));
    return cornerPoints;
}

void BoundingBox::setFrameStamp(int _frameStamp) { frameStamp = _frameStamp; }
void BoundingBox::shiftTopLeft()
{
    int dx = this->x1, dy = this->y1;
    this->x1 -= dx;
    this->x2 -= dx;
    this->y1 -= dy;
    this->y2 -= dy;
}
