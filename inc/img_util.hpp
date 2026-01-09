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

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
// DEPRECATED: bounding_box.hpp removed
// #include "bounding_box.hpp"
// #include <Eigen/Dense>     // For linear algebra computations

// TODO: Andy add
#include <numeric>  // For std::accumulate and std::inner_product
#include <algorithm> // For std::max_element
#include <cmath>    // For mathematical functions
#include <map>

#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

// Define M_PI if not defined already
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// DEPRECATED: Forward declarations removed
// class Point;
// class BoundingBox;
class LaneLine;

constexpr int COLLINEAR        = 0;
constexpr int CLOCKWISE        = 1;
constexpr int COUNTERCLOCKWISE = 2;

namespace imgUtil
{
void calcLinearEquation(cv::Point &pA, cv::Point &pB, std::vector<float> &equation);
void calcLinearEquation(cv::Point &pA, cv::Point &pB, float &a, float &b);
void calcLinearEquation(cv::Point &pA, cv::Point &pB, float &a, float &b,cv::Mat& image);
void calcLinearEquation(cv::Point& pA, cv::Point& pB, float& a, float& b,cv::Mat& image, cv::Scalar color);
void calcNonLinearEquation(std::vector<cv::Point>& pList, float& a, float& b, float& c, 
                           int imgWidth, int imgHeight, cv::Mat& image); // Alister add 2025-01-24
// DEPRECATED: Function using Point removed
// int checkPointOnWhichLineSide(Point &p, std::vector<float> &linearEquation);
int checkPointOnWhichLineSide(cv::Point &p, cv::Point &pA, cv::Point &pB);
float findXGivenY(int y, float a, float b);
cv::Point2f getLine(cv::Point &p1, cv::Point &p2);
cv::Point getIntersectionPoint(cv::Point &pLeftBot, cv::Point &pLeftTop, cv::Point &pRightBot, cv::Point &pRightTop);
cv::Point getIntersectionPoint(float m1, float c1, float m2, float c2);
// DEPRECATED: Function using Point removed
// float calcEuclideanDistance(Point &pA, Point &pB);
float calcEuclideanDistance(cv::Point &pA, cv::Point &pB);
float calcEuclideanDistance(cv::Point2f &pA, cv::Point2f &pB);
double angleBetweenPoints(cv::Point &pA, cv::Point &pB);
void rotateVector(cv::Point2f &v, cv::Point2f &vRotated, double angle);
void polyFitX(const std::vector<double> &fity, std::vector<double> &fitx, const cv::Mat &line_fit);
void polyFit(const cv::Mat &src_x, const cv::Mat &src_y, cv::Mat &dst, int order);
bool sortByContourSize(const std::vector<cv::Point> &a, const std::vector<cv::Point> &b);
void findMaxContour(cv::Mat &src, cv::Mat &dst);
void findClosestContour(cv::Mat &src, int y, cv::Mat &dst);
void removeSmallContour(cv::Mat &src, cv::Mat &dst, int sizeThreshold);
void dilate(cv::Mat &img, int kernelSize);
void erode(cv::Mat &img, int kernelSize);
// DEPRECATED: Function using BoundingBox removed
// float getBboxOverlapRatio(BoundingBox &bA, BoundingBox &bB);
void roundedRectangle(cv::Mat &src, const cv::Point &topLeft, const cv::Point &bottomRight, const cv::Scalar &lineColor,
                      int thickness, int lineType, int cornerRadius, bool filled);
void PoseKeyPoints(cv::Mat &src, std::vector<std::pair<int,int>> pose_kpts, const cv::Scalar &lineColor, int thickness); // Alister add 2025-01-21
// void drawSkeletonAction(cv::Mat& src, BoundingBox& box, const cv::Scalar& color); // Alsiter add 2025-08-28
cv::Scalar getPoseColor(const std::string& action);
void efficientRectangle(cv::Mat &src, const cv::Point &topLeft, const cv::Point &bottomRight,
                        const cv::Scalar &lineColor, int thickness, int lineType, int cornerRadius, bool filled);
// DEPRECATED: Functions using BoundingBox removed
// void cropImages(cv::Mat &img, cv::Mat &imgCrop, BoundingBox &box);
// float calcDistanceToCamera(float focalLen, float camHeight, int yVanish, BoundingBox &rescaledBox);
int orientation(const cv::Point &p, const cv::Point &q, const cv::Point &r);
bool onSegment(const cv::Point &p, const cv::Point &q, const cv::Point &r);
bool isLinesIntersected(const cv::Point &p1, const cv::Point &q1, const cv::Point &p2, const cv::Point &q2);
// DEPRECATED: Functions using BoundingBox removed
// int getIntersectArea(const BoundingBox &a, const BoundingBox &b);
// float getArea(const BoundingBox &box);
// float iou(const BoundingBox &a, const BoundingBox &b);

// ----- After v0,8.0+ ----- //
double robustCurvatureEstimation(const std::vector<cv::Point>& points);
cv::Point getCenterPoint(const cv::Point& p1, const cv::Point& p2);
cv::Point getSmoothedPoint(const cv::Point& prevPoint, const cv::Point& currPoint, float alpha);
bool areSectionsClose(
    const std::vector<cv::Point>& firstLine,
    const std::vector<cv::Point>& secondLine,
    int imageWidth,
    int idxStart,
    int idxEnd,
    float proximityThreshold);
bool areUpperSectionsClose(
    const std::vector<cv::Point>& firstLine,
    const std::vector<cv::Point>& secondLine,
    int imageWidth,
    float proximityThreshold);
bool areLowerSectionsClose(
    const std::vector<cv::Point>& firstLine,
    const std::vector<cv::Point>& secondLine,
    int imageWidth,
    float proximityThreshold);
bool areMiddleSectionsClose(
    const std::vector<cv::Point>& firstLine,
    const std::vector<cv::Point>& secondLine,
    int imageWidth,
    float proximityThreshold);

//TODO:
std::vector<cv::Point> fitAndResampleCurve(
    const std::vector<cv::Point>& points, int imgWidth, int numSamples);

void fitAndResampleCurveInPlace(
    const std::vector<cv::Point>& points,
    std::vector<cv::Point>& result,       // Pre-allocated output buffer
    std::vector<cv::Point>& sortedBuffer, // Pre-allocated buffer for sorting
    std::vector<cv::Point>& smoothBuffer, // Pre-allocated buffer for smoothing
    cv::Mat& A,                           // Pre-allocated matrix for polynomial fitting
    cv::Mat& bx,                          // Pre-allocated matrix for polynomial fitting
    cv::Mat& xCoeffs,                     // Pre-allocated matrix for coefficients
    int imgWidth, 
    int numSamples);

std::vector<cv::Point> fitAndResampleCurveWithRange(
    const std::vector<cv::Point>& points, double minY, double maxY, int imgWidth, int numSamples);

std::vector<cv::Point> mergeAndPreserveLaneExtent(
    const std::vector<cv::Point>& currentPoints, 
    const std::vector<cv::Point>& previousPoints,
    int imgWidth,
    int numSamples);

void fitAndResampleCurveWithRangeInPlace(
    const std::vector<cv::Point>& points, 
    double minY, 
    double maxY, 
    std::vector<cv::Point>& result,       // Pre-allocated output buffer
    std::vector<cv::Point>& sortedBuffer, // Pre-allocated sorting buffer
    cv::Mat& matrixA,                     // Pre-allocated matrix for polynomial fitting
    cv::Mat& matrixB,                     // Pre-allocated vector for polynomial fitting
    cv::Mat& coeffsMatrix,                // Pre-allocated matrix for coefficients
    int imgWidth, 
    int numSamples);

void mergeAndPreserveLaneExtentInPlace(
    const std::vector<cv::Point>& currentPoints, 
    const std::vector<cv::Point>& previousPoints,
    std::vector<cv::Point>& outputPoints,        // Output buffer
    std::vector<cv::Point>& sortedBuffer,        // Reusable sorting buffer
    std::vector<cv::Point>& mergedBuffer,        // Reusable merge buffer
    cv::Mat& matrixA,                           
    cv::Mat& matrixB,
    cv::Mat& coeffsMatrix,
    int imgWidth,
    int numSamples);

std::vector<cv::Point> weightedMergeAndPreserveLaneExtent(
    const std::vector<cv::Point>& currentPoints, 
    const std::vector<cv::Point>& previousPoints,
    double currentWeight, // Weight for current frame (e.g., 0.7)
    double previousWeight, // Weight for previous frame (e.g., 0.3)
    int imgWidth,
    int numSamples);

void weightedMergeAndPreserveLaneExtentInPlace(
    const std::vector<cv::Point>& currentPoints, 
    const std::vector<cv::Point>& previousPoints,
    std::vector<cv::Point>& result,         // Pre-allocated result buffer
    std::vector<cv::Point>& tempBuffer,     // Pre-allocated temp buffer for smoothing
    std::map<int, int>& currentLookup,      // Pre-allocated map
    std::map<int, int>& previousLookup,     // Pre-allocated map
    double currentWeight,                   // Weight for current frame (e.g., 0.7)
    double previousWeight,                  // Weight for previous frame (e.g., 0.3)
    int imgWidth,
    int numSamples);

bool isPolylinesIntersect(
    const std::vector<cv::Point>& polyline1,
    const std::vector<cv::Point>& polyline2,
    int idxStart,
    double distanceTolerance);
bool arePolylinesIntersecting(
    const std::vector<cv::Point>& polyline1,
    const std::vector<cv::Point>& polyline2,
    double epsilon,
    bool debugOutput);
std::vector<cv::Point> smoothPoints(
    const std::vector<cv::Point>& pointList, 
    int imageWidth, 
    float smoothingStrength);
void smoothPointsInPlace(
    const std::vector<cv::Point>& inputPoints,
    std::vector<cv::Point>& outputPoints,
    std::vector<cv::Point>& tempBuffer,
    int imageWidth,
    float smoothingStrength);
bool lineIntersection(
    const cv::Point& p1, const cv::Point& p2, 
    const cv::Point& p3, const cv::Point& p4,
    cv::Point& intersection);
bool findPolylineIntersection(
    const std::vector<cv::Point>& polyline1, 
    const std::vector<cv::Point>& polyline2,
    cv::Point& intersection);
std::vector<cv::Point> findAllPolylineIntersections(
    const std::vector<cv::Point>& polyline1, 
    const std::vector<cv::Point>& polyline2);
cv::Point2f findLineIntersection(
    const cv::Point& p1_int, const cv::Point& p2_int,
    const cv::Point& p3_int, const cv::Point& p4_int);
double cross_product(const cv::Point2f& a, const cv::Point2f& b);
void calculateLaneAnglesSimple(
    const cv::Point& pLeftFar,
    const cv::Point& pLeftCarhood,
    const cv::Point& pRightFar,
    const cv::Point& pRightCarhood,
    float& angleLeft,
    float& angleRight);
cv::Point rotatePointClockwise(
    const cv::Point& point,
    float angleDegrees,
    const cv::Point& center);
float calculateVectorAngle(
    const cv::Point2f& vectorA,
    const cv::Point2f& vectorB,
    bool inDegrees);
float calculateAngleBetweenPoints(
    const cv::Point& pointO,
    const cv::Point& pointA,
    const cv::Point& pointB,
    bool inDegrees);
void generateLinearPoints(
    const cv::Point& startPoint, 
    const cv::Point& endPoint, 
    int numPoints,
    std::vector<cv::Point>& resultPoints); 
bool isLaneShapeUnnatural(const std::vector<cv::Point>& lane, float slopeChangeThreshold);
float calculateLineSlope(const std::vector<cv::Point>& line);
bool isSymmetricLanes(const std::vector<cv::Point>& leftLane, 
                     const std::vector<cv::Point>& rightLane,
                     float angleTolerance);
cv::Point findPointBySlope(float slope, const cv::Point& refPoint, int targetY);
cv::Point findPointOnLineByY(const std::vector<cv::Point>& line, int targetY);
void drawTextWithBorder(
    cv::Mat& img, 
    const std::string& text, 
    cv::Point position, 
    int fontFace, 
    double fontScale, 
    cv::Scalar textColor, 
    cv::Scalar borderColor, 
    int borderThickness,
    int lineType);
cv::Point findPointAtY(const std::vector<cv::Point>& polyline, int targetY);
void drawDebugText(
    cv::Mat& img, 
    const std::string& text, 
    cv::Point position);
void drawPoints(
    const cv::Point& point,
    int modelWidth,
    int modelHeight,
    int imgWidth,
    int imgHeight,
    cv::Mat& image, cv::Scalar color);
// DEPRECATED: Function using BoundingBox removed
// void drawLanePoints(
//     const BoundingBox& box,
//     int modelWidth,
//     int modelHeight,
//     int imgWidth,
//     int imgHeight,
//     cv::Mat& image,
//     cv::Scalar color);
void drawLanePoints(
    const std::vector<cv::Point>& leftPoints,
    const std::vector<cv::Point>& rightPoints,
    int modelWidth,
    int modelHeight,
    int imgWidth,
    int imgHeight,
    cv::Mat& image,
    cv::Scalar color);
void drawLanePolygon(
    const std::vector<cv::Point>& leftPoints,
    const std::vector<cv::Point>& rightPoints,
    int modelWidth,
    int modelHeight,
    int imgWidth,
    int imgHeight,
    cv::Mat& image, cv::Scalar color);

#if defined(CV28) || defined(CV28_SIMULATOR)
static int image_buffer_to_mat_yuv2bgr_nv12(const ea_env_image_buffer_t *image_buffer, cv::Mat &bgr);
static int image_buffer_to_mat_yuv2yuv_nv12(const ea_env_image_buffer_t *image_buffer, cv::Mat &yuv_nv12);
static int image_buffer_to_mat_bgr2bgr(const ea_env_image_buffer_t *image_buffer, cv::Mat &bgr);
static int image_buffer_to_mat_rgb2bgr(const ea_env_image_buffer_t *image_buffer, cv::Mat &bgr);
cv::Mat convertTensorToMat(ea_tensor_t* imgTensor);
#endif
}