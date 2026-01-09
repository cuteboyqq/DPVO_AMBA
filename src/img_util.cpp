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

#include "img_util.hpp"

namespace imgUtil
{
void calcLinearEquation(cv::Point& pA, cv::Point& pB, std::vector<float>& equation)
{
    // Calculate the slope (m)
    float slope = static_cast<float>(pB.y - pA.y) / (pB.x - pA.x);

    // Slope = -slope due to image is from top-left to bottom-right
    slope = -slope;

    // Calculate the y-intercept (b) using y = mx + b
    float intercept = pA.y - slope * pA.x;

    equation = {slope, intercept};
}

void calcLinearEquation(cv::Point& pA, cv::Point& pB, float& a, float& b,cv::Mat& image, cv::Scalar color)
{
    // Calculate the slope (m)
    a = static_cast<float>(pB.y - pA.y) / (pB.x - pA.x);

    // Calculate the y-intercept (b) using y = mx + b
    b = pA.y - a * pA.x;

    // Draw the line between pA and pB
    cv::line(image, pA, pB, color, 2);  // Blue line

    // Draw points pA and pB
    cv::circle(image, pA, 2, cv::Scalar(0, 255, 0), -1);  // Green circle for pA
    cv::circle(image, pB, 2, cv::Scalar(255, 255, 0), -1);  // Yellow circle for pB
}

void calcLinearEquation(cv::Point& pA, cv::Point& pB, float& a, float& b)
{
    // Calculate the slope (m)
    a = static_cast<float>(pB.y - pA.y) / (pB.x - pA.x);

    // Calculate the y-intercept (b) using y = mx + b
    b = pA.y - a * pA.x;
}

void calcLinearEquation(cv::Point& pA, cv::Point& pB, float& a, float& b,cv::Mat& image)
{
    // Calculate the slope (m)
    a = static_cast<float>(pB.y - pA.y) / (pB.x - pA.x);

    // Calculate the y-intercept (b) using y = mx + b
    b = pA.y - a * pA.x;

    // Draw the line between pA and pB
    cv::line(image, pA, pB, cv::Scalar(255,0,0), 2);  // Blue line

    // Draw points pA and pB
    cv::circle(image, pA, 2, cv::Scalar(0, 255, 0), -1);  // Green circle for pA
    cv::circle(image, pB, 2, cv::Scalar(255, 255, 0), -1);  // Yellow circle for pB
}

// Alister add 2025-01-24
void calcNonLinearEquation(std::vector<cv::Point>& pList, float& a, float& b, float& c, int imgWidth, int imgHeight, cv::Mat& image) 
{
    std::cout << "---------------pList :----------------------------\n";
    for (const auto& p : pList) {
        std::cout << "(" << p.x << "," << p.y << ")\n";
    }

    int n = pList.size();
    if (n < 3) {
        std::cerr << "Error: Insufficient points to calculate a quadratic fit.\n";
        return;
    }

    // Step 1: Prepare data in a suitable form (using vector of points)
    cv::Mat dataMat(n, 3, CV_32F);
    for (int i = 0; i < n; ++i) {
        dataMat.at<float>(i, 0) = pList[i].x * pList[i].x;
        dataMat.at<float>(i, 1) = pList[i].x;
        dataMat.at<float>(i, 2) = 1.0f;
    }

    // Step 2: Perform the polynomial fit for y = ax^2 + bx + c
    cv::Mat yMat(n, 1, CV_32F);
    for (int i = 0; i < n; ++i) {
        yMat.at<float>(i, 0) = pList[i].y;
    }

    // Fit the quadratic model
    cv::Mat coeffs;
    cv::solve(dataMat, yMat, coeffs, cv::DECOMP_SVD);

    // Retrieve the coefficients
    a = coeffs.at<float>(0, 0);  // x^2 coefficient
    b = coeffs.at<float>(1, 0);  // x coefficient
    c = coeffs.at<float>(2, 0);  // constant term

    std::cout << "Fitted Coefficients: a=" << a << ", b=" << b << ", c=" << c << "\n";
    std::cout << "Equation: y = " << a << "x^2 + " << b << "x + " << c << "\n";

    // Step 3: Compute new y for each x using the fitted equation and store as new points
    // std::vector<cv::Point> newPoints;
    // for (const auto& p : pList) {
    //     float newY = a * p.x * p.x + b * p.x + c;
    //     newPoints.push_back(cv::Point(p.x, static_cast<int>(newY)));
    // }

    // Step 3: Compute y for each x from the original input points using the fitted equation
    // std::vector<cv::Point> curvePoints;
    for (const auto& p : pList) {
        float fittedY = a * p.x * p.x + b * p.x + c;
        std::cout << "Original: (" << p.x << ", " << p.y << "), Fitted y: " << fittedY << "\n";
        // Draw the fitted point on the image
        cv::circle(image, cv::Point(p.x, static_cast<int>(fittedY)), 3, cv::Scalar(0, 0, 255), -1);
        // if (fittedY >= 0 && fittedY < imgHeight) {  // Ensure y is within image bounds
        //     curvePoints.push_back(cv::Point(p.x, static_cast<int>(fittedY)));
        // }
    }


    // Draw the fitted curve on the image
    // if (curvePoints.size() > 1) {
    //     cv::polylines(image, curvePoints, false, cv::Scalar(0, 255, 0), 2); // Green curve
    // }


    // Step 4: Solve for x from y using the quadratic formula: ax^2 + bx + (c - y) = 0
    std::cout << "Calculated x values from y:" << std::endl;
    for (const auto& p : pList) {
        float y_val = p.y;

        // Compute the discriminant
        float discriminant = b * b - 4 * a * (c - y_val);

        // If the discriminant is negative, no real solution exists
        if (discriminant < 0) {
            std::cerr << "Error: No real solutions for y = " << y_val << std::endl;
            continue; // Skip this point
        }

        // Calculate the two roots for x using the quadratic formula
        float x1 = (-b + sqrt(discriminant)) / (2 * a);
        float x2 = (-b - sqrt(discriminant)) / (2 * a);

        // Output both possible x values for the given y
        std::cout << "For y = " << y_val << ", possible x values: x1 = " << x1 << ", x2 = " << x2 << std::endl;

        // Choose the valid x value within range (0 to 575)
        float chosen_x = NAN;

        // Check if x1 is within the valid range
        if (x1 >= 0 && x1 <= 575) {
            chosen_x = x1;
        }

        // If x1 is not valid, try selecting x2 if it's within range
        if (std::isnan(chosen_x) && x2 >= 0 && x2 <= 575) {
            chosen_x = x2;
        }

        // If both are out of range, choose the closest valid value
        if (std::isnan(chosen_x)) {
            // Pick the closest valid value from x1 and x2 (if both are outside 0-575)
            if (x1 < 0) {
                chosen_x = 0;  // Choose the lower bound if x1 is less than 0
            } else if (x1 > 575) {
                chosen_x = 575;  // Choose the upper bound if x1 is greater than 575
            } else if (x2 < 0) {
                chosen_x = 0;  // Choose the lower bound if x2 is less than 0
            } else if (x2 > 575) {
                chosen_x = 575;  // Choose the upper bound if x2 is greater than 575
            }
        }

        std::cout << "Chosen x value for y = " << y_val << " is " << chosen_x << std::endl;
    }
}


// DEPRECATED: Function using Point removed
// int checkPointOnWhichLineSide(Point& p, std::vector<float>& linearEquation)
// {
//     // Slope (m)
//     float m = linearEquation[0];
//
//     // Intercept (b)
//     float b = linearEquation[1];
//
//     float res = p.x * m + b;
//
//     if (res > 0)
//         return 0; // left side
//     else if (res == 0)
//         return -1; // on the line
//     else
//         return 1; // right side
// }

int checkPointOnWhichLineSide(cv::Point& p, cv::Point& pA, cv::Point& pB)
{
    float tmpx = static_cast<float>(pA.x - pB.x) / (pA.y - pB.y) * (p.y - pB.y) + pB.x;

    return tmpx > static_cast<float>(p.x) ? 0 : 1; // 0 for left and 1 for right side
}

float findXGivenY(int y, float a, float b)
{
    // get y given x and line equation y = a*x + b;
    float x = (y - b) / a;
    return x;
}

cv::Point2f getLine(cv::Point& p1, cv::Point& p2)
{
    if (p1.x == p2.x)
        return cv::Point2f(0.0f, 0.0f); // Vertical line

    float slope     = static_cast<float>(p1.y - p2.y) / static_cast<float>(p1.x - p2.x);
    float intercept = static_cast<float>(p1.y) - slope * static_cast<float>(p1.x);

    return cv::Point2f(slope, intercept);
}

cv::Point getIntersectionPoint(cv::Point& pLeftBot, cv::Point& pLeftTop, cv::Point& pRightBot, cv::Point& pRightTop)
{
    cv::Point2f line1 = getLine(pLeftBot, pLeftTop);
    cv::Point2f line2 = getLine(pRightBot, pRightTop);

    if (line1.x == line2.x)
        return cv::Point(0, 0); // Parallel lines

    float x = (line1.y - line2.y) / (line2.x - line1.x);
    float y = line1.x * x + line1.y;

    return cv::Point(std::max(0.0f, x), std::max(0.0f, y));
}

cv::Point getIntersectionPoint(float m1, float c1, float m2, float c2)
{
    if (m1 == m2)
        return cv::Point(0, 0); // Parallel lines

    float x = (c2 - c1) / (m1 - m2);
    float y = m1 * x + c1;

    return cv::Point(std::max(0.0f, x), std::max(0.0f, y));
}

// DEPRECATED: Function using Point removed
// float calcEuclideanDistance(Point& pA, Point& pB)
// {
//     if (pB.x == -1 && pB.y == -1)
//         return -1.0f; // Invalid point
//
//     int dx = pA.x - pB.x;
//     int dy = pA.y - pB.y;
//
//     return std::sqrt(static_cast<float>(dx * dx + dy * dy));
// }

float calcEuclideanDistance(cv::Point& pA, cv::Point& pB)
{
    if (pB.x == -1 && pB.y == -1)
        return -1.0f; // Invalid point

    int dx = pA.x - pB.x;
    int dy = pA.y - pB.y;

    return std::sqrt(static_cast<float>(dx * dx + dy * dy));
}

float calcEuclideanDistance(cv::Point2f& pA, cv::Point2f& pB)
{
    float distance = 0;
    if ((pB.x == -1) && (pB.y == -1))
        distance = -1.0;
    else
    {
        int dx   = pA.x - pB.x;
        int dy   = pA.y - pB.y;
        distance = sqrt(pow(dx, 2) + pow(dy, 2));
    }

    return distance;
}

double angleBetweenPoints(cv::Point& pA, cv::Point& pB)
{
    double x1 = static_cast<double>(pA.x);
    double x2 = static_cast<double>(pB.x);
    double y1 = static_cast<double>(pA.y);
    double y2 = static_cast<double>(pB.y);

    double dot_product   = x1 * x2 + y1 * y2;
    double magnitude1_sq = x1 * x1 + y1 * y1;
    double magnitude2_sq = x2 * x2 + y2 * y2;

    if (magnitude1_sq == 0.0 || magnitude2_sq == 0.0)
        return 0.0; // Handle zero vectors

    double cosine = dot_product / std::sqrt(magnitude1_sq * magnitude2_sq);
    return std::acos(cosine);
}

void rotateVector(cv::Point2f& v, cv::Point2f& vRotated, double angle)
{
    double cosAngle = std::cos(angle);
    double sinAngle = std::sin(angle);
    vRotated.x      = v.x * cosAngle - v.y * sinAngle;
    vRotated.y      = v.x * sinAngle + v.y * cosAngle;
}

template <typename T>
std::vector<double> linspace(T startIn, T endIn, int numIn)
{
    std::vector<double> linspaced;
    if (numIn <= 0)
        return linspaced; // Handle invalid input

    double start = static_cast<double>(startIn);
    double end   = static_cast<double>(endIn);
    double num   = static_cast<double>(numIn);

    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);
    linspaced.reserve(numIn);

    for (int i = 0; i < num - 1; ++i)
        linspaced.push_back(start + delta * i);
    linspaced.push_back(end);

    return linspaced;
}

void polyFitX(const std::vector<double>& fity, std::vector<double>& fitx, const cv::Mat& line_fit)
{
    const float a = line_fit.at<float>(2, 0);
    const float b = line_fit.at<float>(1, 0);
    const float c = line_fit.at<float>(0, 0);
    fitx.reserve(fity.size());

    for (double y : fity)
    {
        double x = a * y * y + b * y + c;
        fitx.push_back(x);
    }
}

void polyFit(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst, int order)
{
    CV_Assert((src_x.rows > 0) && (src_y.rows > 0) && (src_x.cols == 1) && (src_y.cols == 1) && (dst.cols == 1)
              && (dst.rows == (order + 1)) && (order >= 1));

    cv::Mat X(src_x.rows, order + 1, CV_32FC1, cv::Scalar(0));
    for (int i = 0; i <= order; ++i)
    {
        cv::Mat copy = src_x.clone();
        cv::pow(copy, i, copy);
        copy.col(0).copyTo(X.col(i));
    }

    cv::Mat X_t, X_inv;
    transpose(X, X_t);
    cv::invert(X_t * X, X_inv);
    dst = X_inv * X_t * src_y;
}


bool sortByContourSize(const std::vector<cv::Point>& a, const std::vector<cv::Point>& b)
{
    return a.size() > b.size();
}

void findMaxContour(cv::Mat& src, cv::Mat& dst)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i>              hierarchy;
    cv::findContours(src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        dst = cv::Mat::zeros(src.size(), CV_8UC1);
        return;
    }
    std::vector<std::vector<cv::Point>>::iterator maxContourIter =
        std::max_element(contours.begin(), contours.end(), sortByContourSize);

    dst = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::drawContours(dst, contours, std::distance(contours.begin(), maxContourIter), cv::Scalar(255), -1, cv::LINE_8);
}

void findClosestContour(cv::Mat& src, int y, cv::Mat& dst)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i>              hierarchy;
    cv::findContours(src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        dst = cv::Mat::zeros(src.size(), CV_8UC1);
        return;
    }

    int minDiff = 1000;
    int idx     = 0;

    auto closestContourIter = std::min_element(contours.begin(), contours.end(),
                                               [y](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                                   cv::Moments M_a       = cv::moments(a);
                                                   cv::Moments M_b       = cv::moments(b);
                                                   int         centerY_a = static_cast<int>(M_a.m01 / M_a.m00);
                                                   int         centerY_b = static_cast<int>(M_b.m01 / M_b.m00);
                                                   return std::abs(centerY_a - y) < std::abs(centerY_b - y);
                                               });

    dst = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::drawContours(dst, contours, std::distance(contours.begin(), closestContourIter), cv::Scalar(255), -1,
                     cv::LINE_8);
}

void removeSmallContour(cv::Mat& src, cv::Mat& dst, int sizeThreshold)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i>              hierarchy;
    cv::findContours(src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> largeContours;
    std::copy_if(contours.begin(), contours.end(), std::back_inserter(largeContours),
                 [sizeThreshold](const std::vector<cv::Point>& contour) { return contour.size() >= sizeThreshold; });

    dst = cv::Mat::zeros(src.size(), CV_8UC1);
    for (const auto& contour : largeContours)
        cv::drawContours(dst, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), -1, cv::LINE_8);
}

// DEPRECATED: Function using BoundingBox removed
// float getBboxOverlapRatio(BoundingBox& boxA, BoundingBox& boxB)
// {
//     int iouX = std::max(boxA.x1, boxB.x1);
//     int iouY = std::max(boxA.y1, boxB.y1);
//     int iouW = std::min(boxA.x2, boxB.x2) - iouX;
//     int iouH = std::min(boxA.y2, boxB.y2) - iouY;
//     iouW     = std::max(iouW, 0);
//     iouH     = std::max(iouH, 0);
//
//     if (boxA.getArea() == 0)
//         return 0.0f;
//
//     float iouArea  = static_cast<float>(iouW * iouH);
//     float boxAArea = static_cast<float>(boxA.getArea());
//     return iouArea / boxAArea;
// }

void roundedRectangle(cv::Mat& src, const cv::Point& topLeft, const cv::Point& bottomRight, const cv::Scalar& lineColor,
                      int thickness, int lineType, int cornerRadius, bool filled)
{
    cv::Point p1 = topLeft;
    cv::Point p2(bottomRight.x, topLeft.y);
    cv::Point p3 = bottomRight;
    cv::Point p4(topLeft.x, bottomRight.y);
    const int cornerRadiusHalf = cornerRadius / 2;

    cv::line(src, cv::Point(p1.x + cornerRadiusHalf, p1.y), cv::Point(p2.x - cornerRadiusHalf, p2.y), lineColor,
             thickness, lineType);
    cv::line(src, cv::Point(p2.x, p2.y + cornerRadiusHalf), cv::Point(p3.x, p3.y - cornerRadiusHalf), lineColor,
             thickness, lineType);
    cv::line(src, cv::Point(p4.x + cornerRadiusHalf, p4.y), cv::Point(p3.x - cornerRadiusHalf, p3.y), lineColor,
             thickness, lineType);
    cv::line(src, cv::Point(p1.x, p1.y + cornerRadiusHalf), cv::Point(p4.x, p4.y - cornerRadiusHalf), lineColor,
             thickness, lineType);

    cv::ellipse(src, p1 + cv::Point(cornerRadius, cornerRadius), cv::Size(cornerRadius, cornerRadius), 180.0, 0, 90,
                lineColor, thickness, lineType);
    cv::ellipse(src, p2 + cv::Point(-cornerRadius, cornerRadius), cv::Size(cornerRadius, cornerRadius), 270.0, 0, 90,
                lineColor, thickness, lineType);
    cv::ellipse(src, p3 + cv::Point(-cornerRadius, -cornerRadius), cv::Size(cornerRadius, cornerRadius), 0.0, 0, 90,
                lineColor, thickness, lineType);
    cv::ellipse(src, p4 + cv::Point(cornerRadius, -cornerRadius), cv::Size(cornerRadius, cornerRadius), 90.0, 0, 90,
                lineColor, thickness, lineType);

    if (filled)
    {
        cv::Rect rect(topLeft + cv::Point(cornerRadiusHalf, cornerRadiusHalf),
                      bottomRight - cv::Point(cornerRadiusHalf, cornerRadiusHalf));
        cv::rectangle(src, rect, lineColor, -1, lineType);
    }
}

// void PoseKeyPoints(cv::Mat& src, std::vector<std::pair<int, int>> pose_kpts, 
//                    const cv::Scalar& lineColor, int thickness)
// {
//     // Create an overlay for blending
//     cv::Mat overlay;
//     src.copyTo(overlay);

//     // Draw pose keypoints on the overlay
//     for (const auto& kpt : pose_kpts)
//     {
//         const cv::Point keypoint(kpt.first, kpt.second);
//         // std::cout << "Draw pose keypoints..." << std::endl;
//         // std::cout << "Keypoint: (" << kpt.first << ", " << kpt.second << ")\n";
//         // Draw a circle on the overlay
//         cv::circle(overlay, keypoint, thickness, lineColor, -1); 
//     }

//     // Blend the overlay with the original image using alpha = 0.6 (60% opacity)
//     double alpha = 1.0; // Opacity of the keypoints
//     double beta = 1.0 - alpha; // Remaining weight for the source image
//     cv::addWeighted(overlay, alpha, src, beta, 0, src);
// }


void PoseKeyPoints(cv::Mat& src, std::vector<std::pair<int, int>> pose_kpts, 
                    const cv::Scalar& lineColor, int thickness)
{
    // Create an overlay for blending
    cv::Mat overlay;
    src.copyTo(overlay);

    // Define COCO keypoint colors (Ultralytics-like, BGR format for OpenCV)
    const std::vector<cv::Scalar> KPT_COLORS = {
        {255, 128, 0},   // 0: nose (orange)
        {255, 0, 0},     // 1: left eye (blue)
        {0, 0, 255},     // 2: right eye (red)
        {255, 85, 255},  // 3: left ear (pink)
        {170, 0, 255},   // 4: right ear (purple)
        {0, 255, 0},     // 5: left shoulder (green)
        {0, 128, 255},   // 6: right shoulder (cyan/orange)
        {0, 255, 128},   // 7: left elbow
        {0, 255, 255},   // 8: right elbow
        {128, 255, 128}, // 9: left wrist
        {128, 255, 255}, // 10: right wrist
        {128, 0, 255},   // 11: left hip
        {255, 0, 128},   // 12: right hip
        {85, 255, 170},  // 13: left knee
        {255, 85, 85},   // 14: right knee
        {0, 85, 255},    // 15: left ankle
        {85, 0, 255}     // 16: right ankle
    };

    // Draw keypoints with different colors
    for (size_t i = 0; i < pose_kpts.size(); i++)
    {
        const auto& kpt = pose_kpts[i];
        if (kpt.first <= 0 && kpt.second <= 0) continue;

        cv::Point keypoint(kpt.first, kpt.second);
        cv::Scalar color = KPT_COLORS[i % KPT_COLORS.size()];
        cv::circle(overlay, keypoint, thickness + 1, color, -1);
    }

    // Define COCO skeleton edges (same as before)
    const std::vector<std::pair<int, int>> COCO_SKELETON = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},  // legs/hips
        {5, 11}, {6, 12}, {5, 6},                          // torso (body)
        {5, 7}, {7, 9}, {6, 8}, {8, 10},                   // arms
        {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4},            // head
        {3, 5}, {4, 6}                                     // shoulders
    };

    const std::vector<cv::Scalar> EDGE_COLORS = {
        {255, 0, 0},   {255, 0, 0},   {0, 0, 255}, {0, 0, 255}, {0, 255, 255},
        {0, 255, 0},   {0, 255, 0},   {0, 255, 255},
        {255, 128, 0}, {255, 128, 0}, {0, 128, 255}, {0, 128, 255},
        {128, 0, 255}, {255, 0, 255}, {255, 0, 255}, {128, 128, 255}, {255, 128, 255},
        {128, 255, 128}, {128, 255, 128}
    };

    // Draw skeleton lines
    for (size_t i = 0; i < COCO_SKELETON.size(); i++)
    {
        int idx1 = COCO_SKELETON[i].first;
        int idx2 = COCO_SKELETON[i].second;

        if (idx1 < pose_kpts.size() && idx2 < pose_kpts.size())
        {
            cv::Point p1(pose_kpts[idx1].first, pose_kpts[idx1].second);
            cv::Point p2(pose_kpts[idx2].first, pose_kpts[idx2].second);

            if ((p1.x <= 0 && p1.y <= 0) || (p2.x <= 0 && p2.y <= 0))
                continue;

            cv::line(overlay, p1, p2, EDGE_COLORS[i % EDGE_COLORS.size()], thickness - 0);
        }
    }

    // Blend overlay
    cv::addWeighted(overlay, 1.0, src, 0.0, 0, src);
}

cv::Scalar getPoseColor(const std::string& action) 
{
    if (action == "fall_down") return {0,0,255}; // red
    if (action == "bending")   return {0,128,255}; // ??
    if (action == "sitting")   return {0,255,255}; // yellow
    if (action == "running")   return {255,0,0}; // blue
    if (action == "walking")   return {0,255,0}; // green
    return {255,255,255}; // default white
}

// void drawSkeletonAction(cv::Mat& src, BoundingBox& box, const cv::Scalar& color)
// {
//     if (box.skeletonAction.empty())
//     {   
//         // std::cout<<"No skeletonAction!!"<<std::endl;
//         return; // nothing to draw
//     } 

//     // Choose a position slightly above the top-left corner of the box
//     cv::Point textOrg(box.x1 + 20, box.y1 - 20); // 5 pixels above the box

//     cv::Scalar _color = getPoseColor(box.skeletonAction);

//     // Draw the text on m_displayImg
//     cv::putText(
//         src,               // Image
//         box.skeletonAction,         // Text to draw
//         textOrg,                    // Bottom-left corner of text
//         cv::FONT_HERSHEY_SIMPLEX,   // Font
//         0.7,                        // Font scale
//         _color,                      // Color
//         2,                          // Thickness
//         cv::LINE_AA                 // Line type
//     );
//     // === Fall Down Warning Overlay ===
//     bool draw_warning = false;
//     bool draw_red_rect = false;
//     if ( (box.skeletonAction == "fall_down" || box.skeletonAction == "falling") && draw_red_rect)
//     {
//         // Draw a red rectangle around the entire frame
//         cv::rectangle(src, cv::Point(50, 50), 
//                            cv::Point(src.cols - 50, src.rows - 50), 
//                            cv::Scalar(0, 0, 255), 50); // Thick red border

//         // Big warning text in center of frame
//         std::string warning = "!!! FALL DOWN WARNING !!!";
//         int fontFace = cv::FONT_HERSHEY_DUPLEX;
//         double fontScale = 4.0;
//         int thickness = 6;
//         int baseline = 0;

//         cv::Size textSize = cv::getTextSize(warning, fontFace, fontScale, thickness, &baseline);
//         cv::Point textOrg(
//             (src.cols - textSize.width) / 2,
//             (src.rows + textSize.height) / 2
//         );
//         if(draw_warning)
//         {
//              cv::putText(src, warning, textOrg, fontFace, fontScale,
//                     cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);
//         }
       
//     }
// }


void efficientRectangle(cv::Mat& src, const cv::Point& topLeft, const cv::Point& bottomRight,
                        const cv::Scalar& lineColor, int thickness, int lineType, int cornerRadius, bool filled)
{
    const int x1     = topLeft.x;
    const int y1     = topLeft.y;
    const int x2     = bottomRight.x;
    const int y2     = bottomRight.y;
    const int width  = x2 - x1;
    const int height = y2 - y1;
    const int d = std::max(1, static_cast<int>(std::max(width, height) * 0.1));
    const int r = std::max(1, static_cast<int>(std::max(width, height) * 0.1));
   

    // Optional border effect - draw the same shape with a different color first
    if (thickness > 1) {
        // Draw border with black (or any other contrasting color)
        cv::Scalar borderColor(0, 0, 0); // Black border
        int borderThickness = thickness + 2; // Slightly thicker
        
        // Draw top left arc with border
        cv::line(src, cv::Point(x1 + r, y1), cv::Point(x1 + r + d, y1), borderColor, borderThickness, lineType);
        cv::line(src, cv::Point(x1, y1 + r), cv::Point(x1, y1 + r + d), borderColor, borderThickness, lineType);
        cv::ellipse(src, cv::Point(x1 + r, y1 + r), cv::Size(r, r), 180, 0, 90, borderColor, borderThickness, lineType);
    

        // Draw top right arc with border
        cv::line(src, cv::Point(x2 - r, y1), cv::Point(x2 - r - d, y1), borderColor, borderThickness, lineType);
        cv::line(src, cv::Point(x2, y1 + r), cv::Point(x2, y1 + r + d), borderColor, borderThickness, lineType);
        cv::ellipse(src, cv::Point(x2 - r, y1 + r), cv::Size(r, r), 270, 0, 90, borderColor, borderThickness, lineType);

        // Draw bottom left arc with border
        cv::line(src, cv::Point(x1 + r, y2), cv::Point(x1 + r + d, y2), borderColor, borderThickness, lineType);
        cv::line(src, cv::Point(x1, y2 - r), cv::Point(x1, y2 - r - d), borderColor, borderThickness, lineType);
        cv::ellipse(src, cv::Point(x1 + r, y2 - r), cv::Size(r, r), 90, 0, 90, borderColor, borderThickness, lineType);

        // Draw bottom right arc with border
        cv::line(src, cv::Point(x2 - r, y2), cv::Point(x2 - r - d, y2), borderColor, borderThickness, lineType);
        cv::line(src, cv::Point(x2, y2 - r), cv::Point(x2, y2 - r - d), borderColor, borderThickness, lineType);
        cv::ellipse(src, cv::Point(x2 - r, y2 - r), cv::Size(r, r), 0, 0, 90, borderColor, borderThickness, lineType);
    }

    // Draw top left arc
    cv::line(src, cv::Point(x1 + r, y1), cv::Point(x1 + r + d, y1), lineColor, thickness, lineType);
    cv::line(src, cv::Point(x1, y1 + r), cv::Point(x1, y1 + r + d), lineColor, thickness, lineType);
    cv::ellipse(src, cv::Point(x1 + r, y1 + r), cv::Size(r, r), 180, 0, 90, lineColor, thickness, lineType);

    // Draw top right arc
    cv::line(src, cv::Point(x2 - r, y1), cv::Point(x2 - r - d, y1), lineColor, thickness, lineType);
    cv::line(src, cv::Point(x2, y1 + r), cv::Point(x2, y1 + r + d), lineColor, thickness, lineType);
    cv::ellipse(src, cv::Point(x2 - r, y1 + r), cv::Size(r, r), 270, 0, 90, lineColor, thickness, lineType);

    // Draw bottom left arc
    cv::line(src, cv::Point(x1 + r, y2), cv::Point(x1 + r + d, y2), lineColor, thickness, lineType);
    cv::line(src, cv::Point(x1, y2 - r), cv::Point(x1, y2 - r - d), lineColor, thickness, lineType);
    cv::ellipse(src, cv::Point(x1 + r, y2 - r), cv::Size(r, r), 90, 0, 90, lineColor, thickness, lineType);

    // Draw bottom right arc
    cv::line(src, cv::Point(x2 - r, y2), cv::Point(x2 - r - d, y2), lineColor, thickness, lineType);
    cv::line(src, cv::Point(x2, y2 - r), cv::Point(x2, y2 - r - d), lineColor, thickness, lineType);
    cv::ellipse(src, cv::Point(x2 - r, y2 - r), cv::Size(r, r), 0, 0, 90, lineColor, thickness, lineType);

    if (filled)
    {
        cv::Rect rect(topLeft + cv::Point(r, r), bottomRight - cv::Point(r, r));
        cv::rectangle(src, rect, lineColor, -1, lineType);
    }
}

void dilate(cv::Mat& img, int kernelSize)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kernelSize + 1, 2 * kernelSize + 1));
    cv::dilate(img, img, kernel);
}

void erode(cv::Mat& img, int kernelSize)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kernelSize + 1, 2 * kernelSize + 1));
    cv::erode(img, img, kernel);
}

// DEPRECATED: Function using BoundingBox removed
// void cropImages(cv::Mat& img, cv::Mat& imgCrop, BoundingBox& box)
// {
//     int x1 = std::max(box.x1, 0);
//     int y1 = std::max(box.y1, 0);
//     int x2 = std::min(box.x2, img.cols);
//     int y2 = std::min(box.y2, img.rows);
//
//     int w = x2 - x1;
//     int h = y2 - y1;
//
//     cv::Rect roi(x1, y1, w, h);
//     img(roi).copyTo(imgCrop);
// }

// DEPRECATED: Function using BoundingBox removed
// float calcDistanceToCamera(float focalLen, float camHeight, int yVanish, BoundingBox& rescaledBox)
// {
//     cv::Point pAnchor;
//     pAnchor.x = rescaledBox.getCenterPoint().x;
//     pAnchor.y = rescaledBox.getCenterPoint().y + static_cast<int>(rescaledBox.getHeight() / 2);
//     int yDiff = std::max(pAnchor.y - yVanish, 1);
//
//     // Geometric Based Formula
//     return (focalLen * camHeight) / static_cast<float>(yDiff);
// }

// Utility function to find orientation of ordered triplet (p, q, r).
// The function returns:
// 0 -> p, q, and r are collinear
// 1 -> Clockwise
// 2 -> Counterclockwise
int orientation(const cv::Point& p, const cv::Point& q, const cv::Point& r)
{
    float val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0)
        return COLLINEAR;
    return (val > 0) ? CLOCKWISE : COUNTERCLOCKWISE;
}

// Function to check if a point q lies on line segment 'pr'
bool onSegment(const cv::Point& p, const cv::Point& q, const cv::Point& r)
{
    int x1 = p.x, y1 = p.y, x2 = r.x, y2 = r.y;
    if (x1 > x2)
        std::swap(x1, x2);
    if (y1 > y2)
        std::swap(y1, y2);

    if (q.x <= x2 && q.x >= x1 && q.y <= y2 && q.y >= y1)
        return true;

    return false;
}

bool isLinesIntersected(const cv::Point& p1, const cv::Point& q1, const cv::Point& p2, const cv::Point& q2)
{
    // Special Cases
    // p1, q1, and p2 are collinear, and p2 lies on segment p1q1
    int o1 = orientation(p1, q1, p2);
    if (o1 == 0 && onSegment(p1, p2, q1))
        return true;

    // p1, q1, and q2 are collinear, and q2 lies on segment p1q1
    int o2 = orientation(p1, q1, q2);
    if (o2 == 0 && onSegment(p1, q2, q1))
        return true;

    // p2, q2, and p1 are collinear, and p1 lies on segment p2q2
    int o3 = orientation(p2, q2, p1);
    if (o3 == 0 && onSegment(p2, p1, q2))
        return true;

    // p2, q2, and q1 are collinear, and q1 lies on segment p2q2
    int o4 = orientation(p2, q2, q1);
    if (o4 == 0 && onSegment(p2, q1, q2))
        return true;

    // General case
    if (o1 != o2 && o3 != o4)
        return true;

    return false;
}

// DEPRECATED: Functions using BoundingBox removed
// int getIntersectArea(const BoundingBox& a, const BoundingBox& b)
// {
//     /*
//         calculate intersection area as the product of intersection_w and intersection_h
//
//         if intersection_w or intersection_h is 0, the intersection area and iou will always be 0.
//         speed up by returning area as 0 instantly
//     */
//
//     // don't change the ordering
//     // auto logger         = spdlog::get(m_loggerStr);
//     int intersection_w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
//     if (intersection_w <= 0)
//     {
//         // intersection_w is 0 so area will be 0
//         return 0;
//     }
//
//     // don't change the ordering
//     int intersection_h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
//     if (intersection_h <= 0)
//     {
//         // intersection_w is 0 so area will be 0
//         return 0;
//     }
//
//     // both intersection_w and intersection_h > 0, so intersection_area will be > 0
//     return intersection_w * intersection_h;
// }
//
// float getArea(const BoundingBox& box)
// {
//     return (box.x2 - box.x1) * (box.y2 - box.y1);
// }
//
// float iou(const BoundingBox& a, const BoundingBox& b)
// {
//     /***
//     *  iou: return the intersection over union ratio between rectangle a and b
//     ***/
//     // auto logger            = spdlog::get(m_loggerStr);
//     int intersection_area = getIntersectArea(a, b);
//     if (intersection_area == 0)
//     {
//         // when intersection_area is 0. speed up by returning iou 0.0 instantly
//         return 0.0;
//     }
//
//     // area is now a member of v8xyxy. no need to repeat area calculation
//     // float iou = (float)intersection_area / (float)(a.area + b.area - intersection_area);
//     float iou = (float)intersection_area / (float)(getArea(a) + getArea(b) - intersection_area);
//     // avoid division by zero
//     // float iou               = (float) intersection_area / (float) (a.area + b.area - intersection_area + 1 );
//     return iou;
// }


/**
 * A simple and robust function to estimate curvature with strong differentiation
 * between straight lines and curves.
 * 
 * This function uses multiple criteria to ensure straight lines get very low curvature values.
 * 
 * @param points Vector of points (size = 15)
 * @return Curvature value (close to 0 for straight lines, higher for curves)
 */
double robustCurvatureEstimation(const std::vector<cv::Point>& points)
{
    // Check input validity
    if (points.size() != 15) {
        return 0.0;
    }
    
    // Step 1: Check collinearity through linear regression
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (const auto& p : points) {
        sumX += p.x;
        sumY += p.y;
        sumXY += p.x * p.y;
        sumX2 += p.x * p.x;
    }
    
    int n = points.size();
    double avgX = sumX / n;
    double avgY = sumY / n;
    
    // Calculate regression line
    double slope = 0;
    if (std::abs(sumX2 - sumX * avgX) > 1e-10) {
        slope = (sumXY - sumX * avgY) / (sumX2 - sumX * avgX);
    }
    double intercept = avgY - slope * avgX;
    
    // Calculate R-squared to measure straightness
    double sumSqTotal = 0, sumSqResidual = 0;
    for (const auto& p : points) {
        double predicted = slope * p.x + intercept;
        sumSqTotal += (p.y - avgY) * (p.y - avgY);
        sumSqResidual += (p.y - predicted) * (p.y - predicted);
    }
    
    double rSquared = 1.0;
    if (sumSqTotal > 1e-10) {
        rSquared = 1.0 - (sumSqResidual / sumSqTotal);
    }
    
    // Step 2: Check angles between consecutive segments
    std::vector<double> angles;
    for (int i = 1; i < n - 1; i++) {
        cv::Point2d v1(points[i].x - points[i-1].x, points[i].y - points[i-1].y);
        cv::Point2d v2(points[i+1].x - points[i].x, points[i+1].y - points[i].y);
        
        double mag1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
        double mag2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
        
        if (mag1 > 1e-10 && mag2 > 1e-10) {
            double dotProduct = v1.x * v2.x + v1.y * v2.y;
            double cosAngle = dotProduct / (mag1 * mag2);
            
            // Clamp to valid range for acos
            cosAngle = std::max(-1.0, std::min(1.0, cosAngle));
            
            // Calculate the angle and convert to degrees
            double angle = std::acos(cosAngle) * 180.0 / CV_PI;
            angles.push_back(angle);
        }
    }
    
    // Calculate the max angle deviation from 180° (straight line)
    double maxAngleDeviation = 0;
    for (double angle : angles) {
        maxAngleDeviation = std::max(maxAngleDeviation, std::abs(angle - 180.0));
    }
    
    // Step 3: Calculate local curvature using circle fitting for middle portion
    const int centerIndex = 7; // Center point (8th point, 0-indexed)
    std::vector<cv::Point2f> centerPoints;
    for (int i = centerIndex - 2; i <= centerIndex + 2; i++) {
        if (i >= 0 && i < n) {
            centerPoints.push_back(cv::Point2f(points[i].x, points[i].y));
        }
    }
    
    // Fit circle to the points
    cv::Point2f center;
    float radius = 1e10f; // Default to very large radius (almost straight)
    
    try {
        cv::minEnclosingCircle(centerPoints, center, radius);
    } catch (...) {
        // If fitting fails, assume very large radius (nearly straight)
        radius = 1e10f;
    }
    
    // Basic curvature (1/radius)
    double baseCurvature = (radius > 1e-10) ? (1.0 / radius) : 0.0;
    
    // FINAL DECISION LOGIC
    
    // For VERY straight lines (R² > 0.999 AND max angle deviation < 2°)
    if (rSquared > 0.999 && maxAngleDeviation < 2.0) {
        return 0.001; // Return a very small value to indicate "essentially straight"
    }
    
    // For MOSTLY straight lines (R² > 0.99 OR max angle deviation < 5°)
    if (rSquared > 0.99 || maxAngleDeviation < 5.0) {
        return baseCurvature * 0.3; // Return a significantly reduced curvature
    }
    
    // For slightly curved lines
    if (rSquared > 0.97) {
        return baseCurvature * 0.7; // Return a moderately reduced curvature
    }
    
    // For obviously curved lines
    return baseCurvature;
}

cv::Point getCenterPoint(const cv::Point& p1, const cv::Point& p2)
{
    return cv::Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
}


cv::Point getSmoothedPoint(const cv::Point& prevPoint, const cv::Point& currPoint, float alpha)
{
    return cv::Point((1 - alpha) * prevPoint + alpha * currPoint);
}


bool areSectionsClose(
    const std::vector<cv::Point>& firstLine,
    const std::vector<cv::Point>& secondLine,
    int imageWidth,
    int idxStart,
    int idxEnd,
    int maxLineSize,
    float proximityThreshold = 0.02f)  // Default 2% of image width
{
    // Validate input - ensure both lines have enough points
    if (firstLine.size() < maxLineSize || secondLine.size() < maxLineSize)
    {
        return false;  // Not enough points to compare middle sections
    }

    if (idxStart > maxLineSize || idxEnd > maxLineSize)
    {
        return false;
    }
    
    // Sort both lines by y-coordinate (top to bottom)
    std::vector<cv::Point> sortedFirstLine = firstLine;
    std::vector<cv::Point> sortedSecondLine = secondLine;
    
    std::sort(sortedFirstLine.begin(), sortedFirstLine.end(),
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    std::sort(sortedSecondLine.begin(), sortedSecondLine.end(),
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    
    // Extract sections (points idxStart to idxEnd)
    std::vector<cv::Point> firstMiddle, secondMiddle;
    
    for (int i = idxStart; i <= idxEnd; i++) {
        if (i < sortedFirstLine.size()) {
            firstMiddle.push_back(sortedFirstLine[i]);
        }
        if (i < sortedSecondLine.size()) {
            secondMiddle.push_back(sortedSecondLine[i]);
        }
    }
    
    // If either middle section is empty, they can't be close
    if (firstMiddle.empty() || secondMiddle.empty()) {
        return false;
    }
    
    // Calculate the average distance between middle sections
    float totalDistance = 0.0f;
    int numComparisons = 0;
    
    // Compare points at similar y-positions
    for (const auto& firstPt : firstMiddle) {
        // Find the closest point in second line with similar y-coordinate
        for (const auto& secondPt : secondMiddle) {
            // Look for points at similar heights (within 5 pixels)
            if (std::abs(firstPt.y - secondPt.y) <= 5) {
                float distance = std::abs(firstPt.x - secondPt.x);
                totalDistance += distance;
                numComparisons++;
                break;  // Found a match for this y-level
            }
        }
    }
    
    // If no comparable y-positions were found, use mean absolute distance
    if (numComparisons == 0) {
        for (int i = 0; i < std::min(firstMiddle.size(), secondMiddle.size()); i++) {
            float distance = std::abs(firstMiddle[i].x - secondMiddle[i].x);
            totalDistance += distance;
            numComparisons++;
        }
    }
    
    // If still no comparisons, consider the lines not close
    if (numComparisons == 0) {
        return false;
    }
    
    // Calculate average distance
    float avgDistance = totalDistance / numComparisons;
    
    // Normalize distance as a fraction of image width
    float normalizedDistance = avgDistance / imageWidth;
    
    // Return true if middle sections are close enough
    return normalizedDistance <= proximityThreshold;
}

bool areUpperSectionsClose(
    const std::vector<cv::Point>& firstLine,
    const std::vector<cv::Point>& secondLine,
    int imageWidth,
    float proximityThreshold = 0.02f)  // Default 2% of image width
{
    return areSectionsClose(firstLine, secondLine, imageWidth, 0, 10, 15, proximityThreshold);
}


bool areLowerSectionsClose(
    const std::vector<cv::Point>& firstLine,
    const std::vector<cv::Point>& secondLine,
    int imageWidth,
    float proximityThreshold = 0.02f)  // Default 2% of image width
{
    return areSectionsClose(firstLine, secondLine, imageWidth, 10, 15, 15, proximityThreshold);
}


bool areMiddleSectionsClose(
    const std::vector<cv::Point>& firstLine,
    const std::vector<cv::Point>& secondLine,
    int imageWidth,
    float proximityThreshold = 0.02f)  // Default 2% of image width
{
    return areSectionsClose(firstLine, secondLine, imageWidth, 5, 10, 15, proximityThreshold);
}


std::vector<cv::Point> fitAndResampleCurve(
    const std::vector<cv::Point>& points, int imgWidth, int numSamples)
{
    // Check for minimum requirements
    if (points.size() < 3 || numSamples <= 0) {
        return points;
    }
    
    // Sort points by y-coordinate
    std::vector<cv::Point> sortedPoints = points;
    std::sort(sortedPoints.begin(), sortedPoints.end(), 
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    
    // Define the y range with strict bounds checking
    double minY = sortedPoints.front().y;
    double maxY = sortedPoints.back().y;
    
    // Validate Y range (critical)
    if (minY >= maxY) {
        return points; // Invalid range
    }
    
    // CRITICAL FIX - Preserve historical minimum Y but with safer implementation
    static double historicalMinY = minY;
    if (minY < historicalMinY) {
        historicalMinY = minY;
    } else {
        double currentRange = maxY - minY;
        double maxExtension = currentRange * 0.2;
        
        if (historicalMinY < minY && minY - historicalMinY < maxExtension) {
            minY = historicalMinY;
        }
    }
    
    // Ensure all points are valid
    for (auto& pt : sortedPoints) {
        pt.x = std::max(0, std::min(imgWidth - 1, pt.x));
    }
    
    double yStep = (maxY - minY) / std::max(1, numSamples - 1);
    
    // MAJOR CHANGE: Default to linear interpolation for stability
    std::vector<cv::Point> result;
    result.reserve(numSamples);
    
    // Check for zigzag patterns that would indicate S-curves
    bool hasConsistentDirection = true;
    if (sortedPoints.size() >= 5) {
        int directionChanges = 0;
        int lastDirection = 0;
        
        for (size_t i = 2; i < sortedPoints.size(); i++) {
            int currentDirection = sortedPoints[i].x - sortedPoints[i-1].x > 0 ? 1 : -1;
            if (lastDirection != 0 && currentDirection != lastDirection) {
                directionChanges++;
            }
            lastDirection = currentDirection;
        }
        
        hasConsistentDirection = (directionChanges <= 1);
    }
    
    // Only attempt polynomial fitting if points have consistent direction
    bool usePolynomialFit = hasConsistentDirection && sortedPoints.size() >= 4;
    cv::Mat xCoeffs;
    
    if (usePolynomialFit) {
        try {
            // Create matrix for QUADRATIC fitting
            cv::Mat A(sortedPoints.size(), 3, CV_64F);
            cv::Mat bx(sortedPoints.size(), 1, CV_64F);
            
            for (size_t i = 0; i < sortedPoints.size(); i++) {
                double y = sortedPoints[i].y;
                A.at<double>(i, 0) = 1.0;
                A.at<double>(i, 1) = y;
                A.at<double>(i, 2) = y * y;
                bx.at<double>(i) = sortedPoints[i].x;
            }
            
            if (!cv::solve(A, bx, xCoeffs, cv::DECOMP_SVD)) {
                usePolynomialFit = false;
            } else {
                // Validate coefficients
                for (int i = 0; i < xCoeffs.rows; i++) {
                    if (std::isnan(xCoeffs.at<double>(i)) || std::isinf(xCoeffs.at<double>(i))) {
                        usePolynomialFit = false;
                        break;
                    }
                }
                
                // Limit quadratic coefficient to prevent extreme curves
                if (usePolynomialFit) {
                    double quadraticCoeff = std::abs(xCoeffs.at<double>(2));
                    const double MAX_CURVATURE = 0.0005 * imgWidth;
                    
                    if (quadraticCoeff > MAX_CURVATURE) {
                        xCoeffs.at<double>(2) = (xCoeffs.at<double>(2) > 0) ? 
                                               MAX_CURVATURE : -MAX_CURVATURE;
                    }
                }
            }
        } catch (...) {
            usePolynomialFit = false;
        }
    }
    
    // Generate points
    for (int i = 0; i < numSamples; i++) {
        double y = minY + i * yStep;
        double x;
        
        if (usePolynomialFit) {
            // Try polynomial fit first
            x = xCoeffs.at<double>(0) + 
                xCoeffs.at<double>(1) * y + 
                xCoeffs.at<double>(2) * y * y;
                
            // Fall back to linear if polynomial gives strange results
            if (std::isnan(x) || std::isinf(x) || x < 0 || x >= imgWidth) {
                usePolynomialFit = false;
            } else if (y < sortedPoints.front().y) {
                // Limit extrapolation for points beyond the original range
                double topX = sortedPoints.front().x;
                double maxDeviation = 0.05 * imgWidth;
                
                if (std::abs(x - topX) > maxDeviation) {
                    x = topX + (x > topX ? maxDeviation : -maxDeviation);
                }
            }
        }
        
        if (!usePolynomialFit) {
            // Linear interpolation/extrapolation
            if (y <= sortedPoints.front().y) {
                x = sortedPoints.front().x;
            } else if (y >= sortedPoints.back().y) {
                x = sortedPoints.back().x;
            } else {
                // Find points to interpolate between
                size_t idx = 0;
                while (idx < sortedPoints.size() - 1 && sortedPoints[idx + 1].y < y) {
                    idx++;
                }
                
                // Linear interpolation
                double y1 = sortedPoints[idx].y;
                double y2 = sortedPoints[idx + 1].y;
                double x1 = sortedPoints[idx].x;
                double x2 = sortedPoints[idx + 1].x;
                
                // Avoid division by zero
                double yDiff = y2 - y1;
                if (std::abs(yDiff) < 1e-6) {
                    x = (x1 + x2) / 2;
                } else {
                    x = x1 + (x2 - x1) * (y - y1) / yDiff;
                }
            }
        }
        
        // CRITICAL: Ensure points are within image bounds
        x = std::max(0.0, std::min(double(imgWidth - 1), x));
        
        result.push_back(cv::Point(static_cast<int>(x), static_cast<int>(y)));
    }
    
    // Apply smoothing if needed to eliminate any remaining zigzags
    if (result.size() >= 5) {
        int directionChanges = 0;
        int lastDelta = 0;
        
        for (size_t i = 1; i < result.size(); i++) {
            int delta = result[i].x - result[i-1].x;
            if (lastDelta != 0 && delta * lastDelta < 0) {
                directionChanges++;
            }
            lastDelta = delta;
        }
        
        if (directionChanges > 1) {
            // Apply smoothing using moving average
            std::vector<cv::Point> smoothed = result;
            const int windowSize = 5;
            
            for (size_t i = windowSize/2; i < result.size() - windowSize/2; i++) {
                int sumX = 0;
                for (int j = -windowSize/2; j <= windowSize/2; j++) {
                    sumX += result[i+j].x;
                }
                smoothed[i].x = sumX / windowSize;
            }
            
            result = smoothed;
        }
    }
    
    return result;
}

void fitAndResampleCurveInPlace(
    const std::vector<cv::Point>& points,
    std::vector<cv::Point>& result,       // Pre-allocated output buffer
    std::vector<cv::Point>& sortedBuffer, // Pre-allocated buffer for sorting
    std::vector<cv::Point>& smoothBuffer, // Pre-allocated buffer for smoothing
    cv::Mat& A,                           // Pre-allocated matrix for polynomial fitting
    cv::Mat& bx,                          // Pre-allocated matrix for polynomial fitting
    cv::Mat& xCoeffs,                     // Pre-allocated matrix for coefficients
    int imgWidth, 
    int numSamples)
{
    // Check for minimum requirements
    if (points.size() < 3 || numSamples <= 0) {
        result = points; // This is a copy, but it's an edge case
        return;
    }
    
    // Reuse the sortedBuffer instead of creating a new vector
    sortedBuffer.clear();
    sortedBuffer.resize(points.size());
    std::copy(points.begin(), points.end(), sortedBuffer.begin());
    
    // Sort points by y-coordinate
    std::sort(sortedBuffer.begin(), sortedBuffer.end(), 
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    
    // Define the y range with strict bounds checking
    double minY = sortedBuffer.front().y;
    double maxY = sortedBuffer.back().y;
    
    // Validate Y range (critical)
    if (minY >= maxY) {
        result = points; // Invalid range
        return;
    }
    
    // CRITICAL FIX - Preserve historical minimum Y but with safer implementation
    static double historicalMinY = minY;
    if (minY < historicalMinY) {
        historicalMinY = minY;
    } else {
        double currentRange = maxY - minY;
        double maxExtension = currentRange * 0.2;
        
        if (historicalMinY < minY && minY - historicalMinY < maxExtension) {
            minY = historicalMinY;
        }
    }
    
    // Ensure all points are valid
    for (auto& pt : sortedBuffer) {
        pt.x = std::max(0, std::min(imgWidth - 1, pt.x));
    }
    
    double yStep = (maxY - minY) / std::max(1, numSamples - 1);
    
    // Prepare the result vector
    result.clear();
    result.reserve(numSamples);
    
    // Check for zigzag patterns that would indicate S-curves
    bool hasConsistentDirection = true;
    if (sortedBuffer.size() >= 5) {
        int directionChanges = 0;
        int lastDirection = 0;
        
        for (size_t i = 2; i < sortedBuffer.size(); i++) {
            int currentDirection = sortedBuffer[i].x - sortedBuffer[i-1].x > 0 ? 1 : -1;
            if (lastDirection != 0 && currentDirection != lastDirection) {
                directionChanges++;
            }
            lastDirection = currentDirection;
        }
        
        hasConsistentDirection = (directionChanges <= 1);
    }
    
    // Only attempt polynomial fitting if points have consistent direction
    bool usePolynomialFit = hasConsistentDirection && sortedBuffer.size() >= 4;
    
    if (usePolynomialFit) {
        try {
            // Reuse pre-allocated matrices, resizing if needed
            int numPoints = sortedBuffer.size();
            if (A.rows != numPoints || A.cols != 3) {
                A.create(numPoints, 3, CV_64F);
            }
            if (bx.rows != numPoints || bx.cols != 1) {
                bx.create(numPoints, 1, CV_64F);
            }
            
            // Fill matrices for QUADRATIC fitting
            for (size_t i = 0; i < numPoints; i++) {
                double y = sortedBuffer[i].y;
                A.at<double>(i, 0) = 1.0;
                A.at<double>(i, 1) = y;
                A.at<double>(i, 2) = y * y;
                bx.at<double>(i) = sortedBuffer[i].x;
            }
            
            if (!cv::solve(A, bx, xCoeffs, cv::DECOMP_SVD)) {
                usePolynomialFit = false;
            } else {
                // Validate coefficients
                for (int i = 0; i < xCoeffs.rows; i++) {
                    if (std::isnan(xCoeffs.at<double>(i)) || std::isinf(xCoeffs.at<double>(i))) {
                        usePolynomialFit = false;
                        break;
                    }
                }
                
                // Limit quadratic coefficient to prevent extreme curves
                if (usePolynomialFit) {
                    double quadraticCoeff = std::abs(xCoeffs.at<double>(2));
                    const double MAX_CURVATURE = 0.0005 * imgWidth;
                    
                    if (quadraticCoeff > MAX_CURVATURE) {
                        xCoeffs.at<double>(2) = (xCoeffs.at<double>(2) > 0) ? 
                                               MAX_CURVATURE : -MAX_CURVATURE;
                    }
                }
            }
        } catch (...) {
            usePolynomialFit = false;
        }
    }
    
    // Generate points
    for (int i = 0; i < numSamples; i++) {
        double y = minY + i * yStep;
        double x;
        
        if (usePolynomialFit) {
            // Try polynomial fit first
            x = xCoeffs.at<double>(0) + 
                xCoeffs.at<double>(1) * y + 
                xCoeffs.at<double>(2) * y * y;
                
            // Fall back to linear if polynomial gives strange results
            if (std::isnan(x) || std::isinf(x) || x < 0 || x >= imgWidth) {
                usePolynomialFit = false;
            } else if (y < sortedBuffer.front().y) {
                // Limit extrapolation for points beyond the original range
                double topX = sortedBuffer.front().x;
                double maxDeviation = 0.05 * imgWidth;
                
                if (std::abs(x - topX) > maxDeviation) {
                    x = topX + (x > topX ? maxDeviation : -maxDeviation);
                }
            }
        }
        
        if (!usePolynomialFit) {
            // Linear interpolation/extrapolation
            if (y <= sortedBuffer.front().y) {
                x = sortedBuffer.front().x;
            } else if (y >= sortedBuffer.back().y) {
                x = sortedBuffer.back().x;
            } else {
                // Find points to interpolate between
                size_t idx = 0;
                while (idx < sortedBuffer.size() - 1 && sortedBuffer[idx + 1].y < y) {
                    idx++;
                }
                
                // Linear interpolation
                double y1 = sortedBuffer[idx].y;
                double y2 = sortedBuffer[idx + 1].y;
                double x1 = sortedBuffer[idx].x;
                double x2 = sortedBuffer[idx + 1].x;
                
                // Avoid division by zero
                double yDiff = y2 - y1;
                if (std::abs(yDiff) < 1e-6) {
                    x = (x1 + x2) / 2;
                } else {
                    x = x1 + (x2 - x1) * (y - y1) / yDiff;
                }
            }
        }
        
        // CRITICAL: Ensure points are within image bounds
        x = std::max(0.0, std::min(double(imgWidth - 1), x));
        
        result.push_back(cv::Point(static_cast<int>(x), static_cast<int>(y)));
    }
    
    // Apply smoothing if needed to eliminate any remaining zigzags
    if (result.size() >= 5) {
        int directionChanges = 0;
        int lastDelta = 0;
        
        for (size_t i = 1; i < result.size(); i++) {
            int delta = result[i].x - result[i-1].x;
            if (lastDelta != 0 && delta * lastDelta < 0) {
                directionChanges++;
            }
            lastDelta = delta;
        }
        
        if (directionChanges > 1) {
            // Use smoothBuffer instead of creating a new vector
            smoothBuffer.clear();
            smoothBuffer.resize(result.size());
            std::copy(result.begin(), result.end(), smoothBuffer.begin());
            
            const int windowSize = 5;
            
            for (size_t i = windowSize/2; i < result.size() - windowSize/2; i++) {
                int sumX = 0;
                for (int j = -windowSize/2; j <= windowSize/2; j++) {
                    sumX += result[i+j].x;
                }
                smoothBuffer[i].x = sumX / windowSize;
            }
            
            // Swap rather than copy
            result.swap(smoothBuffer);
        }
    }
}

// Alternative implementation - specify min/max Y explicitly
std::vector<cv::Point> fitAndResampleCurveWithRange(
    const std::vector<cv::Point>& points, double minY, double maxY, int imgWidth, int numSamples)
{
    if (points.size() < 3 || numSamples <= 0)
    {
        return points;
    }
    
    // Sort points by y-coordinate
    std::vector<cv::Point> sortedPoints = points;
    std::sort(sortedPoints.begin(), sortedPoints.end(), 
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    
    double yStep = (maxY - minY) / std::max(1, numSamples - 1);
    
    // Fit a polynomial (3rd degree) to the points
    cv::Mat xCoeffs;
    bool fitSuccessful = false;
    if (sortedPoints.size() >= 4)
    {
        // Create the Vandermonde matrix for polynomial fitting
        cv::Mat A(sortedPoints.size(), 4, CV_64F);
        for (size_t i = 0; i < sortedPoints.size(); i++)
        {
            double y = sortedPoints[i].y;
            A.at<double>(i, 0) = 1.0;
            A.at<double>(i, 1) = y;
            A.at<double>(i, 2) = y * y;
            A.at<double>(i, 3) = y * y * y;
        }
        
        // Solve for x = f(y)
        cv::Mat bx(sortedPoints.size(), 1, CV_64F);
        for (size_t i = 0; i < sortedPoints.size(); i++)
        {
            bx.at<double>(i) = sortedPoints[i].x;
        }
        
        // Check if solve was successful
        try {
            fitSuccessful = cv::solve(A, bx, xCoeffs, cv::DECOMP_SVD);
        } catch (const cv::Exception& e) {
            // Handle OpenCV exceptions
            fitSuccessful = false;
        }
    }
    
    // Generate evenly spaced points along the curve
    std::vector<cv::Point> result;
    result.reserve(numSamples);
    
    for (int i = 0; i < numSamples; i++)
    {
        double y = minY + i * yStep;
        double x;
        
        if (fitSuccessful && sortedPoints.size() >= 4)
        {
            // Polynomial evaluation
            x = xCoeffs.at<double>(0) + 
                xCoeffs.at<double>(1) * y + 
                xCoeffs.at<double>(2) * y * y + 
                xCoeffs.at<double>(3) * y * y * y;
                
            // Apply extrapolation constraints for stability
            if (std::isnan(x) || std::isinf(x))
            {
                fitSuccessful = false;
            }
            else
            {
                if (y < sortedPoints.front().y || y > sortedPoints.back().y) {
                    // Limit extrapolation to reasonable values
                    double nearestX = (y < sortedPoints.front().y) ? 
                                    sortedPoints.front().x : sortedPoints.back().x;
                    double maxDeviation = 0.1 * imgWidth;
                    
                    if (std::abs(x - nearestX) > maxDeviation) {
                        x = nearestX + (x > nearestX ? maxDeviation : -maxDeviation);
                    }
                }
            }
        }
        else
        {
            // Linear interpolation/extrapolation
            if (!fitSuccessful || sortedPoints.size() < 4)
            {
                if (y <= sortedPoints.front().y) {
                    x = sortedPoints.front().x;
                }
                else if (y >= sortedPoints.back().y) {
                    x = sortedPoints.back().x;
                }
                else {
                    // Find surrounding points for interpolation
                    size_t idx = 0;
                    while (idx < sortedPoints.size() - 1 && sortedPoints[idx + 1].y < y) {
                        idx++;
                    }
                    
                    double y1 = sortedPoints[idx].y;
                    double y2 = sortedPoints[idx + 1].y;
                    double x1 = sortedPoints[idx].x;
                    double x2 = sortedPoints[idx + 1].x;
                    
                    x = x1 + (x2 - x1) * (y - y1) / (y2 - y1);
                }
            }
        }
        
        result.push_back(cv::Point(static_cast<int>(x), static_cast<int>(y)));
    }
    
    return result;
}

// Use this function when combining lanes from multiple frames
std::vector<cv::Point> mergeAndPreserveLaneExtent(
    const std::vector<cv::Point>& currentPoints, 
    const std::vector<cv::Point>& previousPoints,
    int imgWidth,
    int numSamples)
{
    if (currentPoints.empty()) return previousPoints;
    if (previousPoints.empty()) return currentPoints;
    
    // Sort both sets of points by y-coordinate
    std::vector<cv::Point> sortedCurrent = currentPoints;
    std::vector<cv::Point> sortedPrevious = previousPoints;
    
    std::sort(sortedCurrent.begin(), sortedCurrent.end(),
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    std::sort(sortedPrevious.begin(), sortedPrevious.end(),
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    
    // Determine the full range across both frames
    int minY = std::min(sortedCurrent.front().y, sortedPrevious.front().y);
    int maxY = std::max(sortedCurrent.back().y, sortedPrevious.back().y);
    
    // Merge points from both sets
    std::vector<cv::Point> mergedPoints;
    mergedPoints.reserve(sortedCurrent.size() + sortedPrevious.size());
    
    // Add all points from both sets
    mergedPoints.insert(mergedPoints.end(), sortedCurrent.begin(), sortedCurrent.end());
    mergedPoints.insert(mergedPoints.end(), sortedPrevious.begin(), sortedPrevious.end());
    
    // Sort and remove duplicates
    std::sort(mergedPoints.begin(), mergedPoints.end(),
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    
    // Remove points that are too close together
    if (mergedPoints.size() > 1) {
        std::vector<cv::Point> uniquePoints;
        uniquePoints.push_back(mergedPoints[0]);
        
        for (size_t i = 1; i < mergedPoints.size(); i++) {
            const cv::Point& lastPt = uniquePoints.back();
            const cv::Point& currentPt = mergedPoints[i];
            
            // Only add points that are at least 3 pixels away in either direction
            if (std::abs(currentPt.x - lastPt.x) > 3 || std::abs(currentPt.y - lastPt.y) > 3) {
                uniquePoints.push_back(currentPt);
            }
        }
        
        mergedPoints = uniquePoints;
    }
    
    // Use our explicit range version of curve fitting to maintain the full extent
    return fitAndResampleCurveWithRange(mergedPoints, minY, maxY, imgWidth, numSamples);
}

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
    int numSamples)
{
    // Handle edge cases
    if (points.size() < 3 || numSamples <= 0 || minY >= maxY)
    {
        result = points; // This is a copy but only happens in edge cases
        return;
    }
    
    // Reuse sortedBuffer instead of creating a new vector
    sortedBuffer.clear();
    sortedBuffer.resize(points.size());
    std::copy(points.begin(), points.end(), sortedBuffer.begin());
    
    // Sort points by y-coordinate
    std::sort(sortedBuffer.begin(), sortedBuffer.end(), 
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    
    double yStep = (maxY - minY) / std::max(1, numSamples - 1);
    
    // Fit a polynomial (3rd degree) to the points
    bool fitSuccessful = false;
    
    if (sortedBuffer.size() >= 4)
    {
        // Resize matrices if needed (avoid reallocation if already correct size)
        int numPoints = sortedBuffer.size();
        if (matrixA.rows != numPoints || matrixA.cols != 4) {
            matrixA.create(numPoints, 4, CV_64F);
        }
        if (matrixB.rows != numPoints || matrixB.cols != 1) {
            matrixB.create(numPoints, 1, CV_64F);
        }
        
        // Fill matrices for polynomial fitting
        for (size_t i = 0; i < numPoints; i++)
        {
            double y = sortedBuffer[i].y;
            matrixA.at<double>(i, 0) = 1.0;
            matrixA.at<double>(i, 1) = y;
            matrixA.at<double>(i, 2) = y * y;
            matrixA.at<double>(i, 3) = y * y * y;
            matrixB.at<double>(i) = sortedBuffer[i].x;
        }
        
        // Solve for x = f(y)
        try {
            fitSuccessful = cv::solve(matrixA, matrixB, coeffsMatrix, cv::DECOMP_SVD);
        } catch (const cv::Exception&) {
            fitSuccessful = false;
        }
    }
    
    // Clear and prepare result vector
    result.clear();
    result.reserve(numSamples);
    
    for (int i = 0; i < numSamples; i++)
    {
        double y = minY + i * yStep;
        double x;
        
        if (fitSuccessful && sortedBuffer.size() >= 4)
        {
            // Polynomial evaluation
            x = coeffsMatrix.at<double>(0) + 
                coeffsMatrix.at<double>(1) * y + 
                coeffsMatrix.at<double>(2) * y * y + 
                coeffsMatrix.at<double>(3) * y * y * y;
                
            // Apply extrapolation constraints for stability
            if (std::isnan(x) || std::isinf(x))
            {
                fitSuccessful = false;
            }
            else
            {
                if (y < sortedBuffer.front().y || y > sortedBuffer.back().y) {
                    // Limit extrapolation to reasonable values
                    double nearestX = (y < sortedBuffer.front().y) ? 
                                    sortedBuffer.front().x : sortedBuffer.back().x;
                    double maxDeviation = 0.1 * imgWidth;
                    
                    if (std::abs(x - nearestX) > maxDeviation) {
                        x = nearestX + (x > nearestX ? maxDeviation : -maxDeviation);
                    }
                }
            }
        }
        
        if (!fitSuccessful || sortedBuffer.size() < 4)
        {
            // Linear interpolation/extrapolation
            if (y <= sortedBuffer.front().y) {
                x = sortedBuffer.front().x;
            }
            else if (y >= sortedBuffer.back().y) {
                x = sortedBuffer.back().x;
            }
            else {
                // Find surrounding points for interpolation
                size_t idx = 0;
                while (idx < sortedBuffer.size() - 1 && sortedBuffer[idx + 1].y < y) {
                    idx++;
                }
                
                double y1 = sortedBuffer[idx].y;
                double y2 = sortedBuffer[idx + 1].y;
                double x1 = sortedBuffer[idx].x;
                double x2 = sortedBuffer[idx + 1].x;
                
                // Avoid division by zero
                double yDiff = y2 - y1;
                if (std::abs(yDiff) < 1e-6) {
                    x = (x1 + x2) / 2.0;
                } else {
                    x = x1 + (x2 - x1) * (y - y1) / yDiff;
                }
            }
        }
        
        // Ensure x is within image bounds
        x = std::max(0.0, std::min(double(imgWidth - 1), x));
        
        result.push_back(cv::Point(static_cast<int>(x), static_cast<int>(y)));
    }
}

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
    int numSamples)
{
    // Handle edge cases by modifying output directly
    if (currentPoints.empty() && previousPoints.empty()) {
        outputPoints.clear();
        return;
    }
    
    if (currentPoints.empty()) {
        outputPoints.assign(previousPoints.begin(), previousPoints.end());
        return;
    }
    
    if (previousPoints.empty()) {
        outputPoints.assign(currentPoints.begin(), currentPoints.end());
        return;
    }
    
    // Reuse sortedBuffer for current points
    sortedBuffer.clear();
    sortedBuffer.resize(currentPoints.size());
    std::copy(currentPoints.begin(), currentPoints.end(), sortedBuffer.begin());
    
    // Sort current points
    std::sort(sortedBuffer.begin(), sortedBuffer.end(),
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    
    // Determine min/max of current points
    int currentMinY = sortedBuffer.front().y;
    int currentMaxY = sortedBuffer.back().y;
    
    // Now use mergedBuffer for previous points
    mergedBuffer.clear();
    mergedBuffer.resize(previousPoints.size());
    std::copy(previousPoints.begin(), previousPoints.end(), mergedBuffer.begin());
    
    // Sort previous points
    std::sort(mergedBuffer.begin(), mergedBuffer.end(),
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    
    // Calculate overall min/max
    int minY = std::min(currentMinY, mergedBuffer.front().y);
    int maxY = std::max(currentMaxY, mergedBuffer.back().y);
    
    // Now reuse mergedBuffer as our combined buffer
    // Add current points first
    mergedBuffer.resize(0);  // Clear without freeing memory
    mergedBuffer.reserve(sortedBuffer.size() + previousPoints.size());
    mergedBuffer.insert(mergedBuffer.end(), sortedBuffer.begin(), sortedBuffer.end());
    
    // Now sortedBuffer is free to use for previous points again
    
    // Add previous points
    mergedBuffer.insert(mergedBuffer.end(), previousPoints.begin(), previousPoints.end());
    
    // Sort all points
    std::sort(mergedBuffer.begin(), mergedBuffer.end(),
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    
    // Remove points that are too close together
    if (mergedBuffer.size() > 1) {
        // Use sortedBuffer for unique points
        sortedBuffer.clear();
        sortedBuffer.reserve(mergedBuffer.size());
        sortedBuffer.push_back(mergedBuffer[0]);
        
        for (size_t i = 1; i < mergedBuffer.size(); i++) {
            const cv::Point& lastPt = sortedBuffer.back();
            const cv::Point& currentPt = mergedBuffer[i];
            
            // Only add points that are at least 3 pixels away in either direction
            if (std::abs(currentPt.x - lastPt.x) > 3 || std::abs(currentPt.y - lastPt.y) > 3) {
                sortedBuffer.push_back(currentPt);
            }
        }
        
        // Use the unique points for fitting
        mergedBuffer.swap(sortedBuffer);
    }
    
    // Use our explicit range version of curve fitting to maintain the full extent
    fitAndResampleCurveWithRangeInPlace(
        mergedBuffer,        // Input - merged and filtered points
        minY,                // Min Y from both sets
        maxY,                // Max Y from both sets
        outputPoints,        // Output - will contain the final result
        sortedBuffer,        // Reusable buffer for internal operations
        matrixA,             // Reusable matrices
        matrixB,
        coeffsMatrix,
        imgWidth,
        numSamples
    );
}

std::vector<cv::Point> weightedMergeAndPreserveLaneExtent(
    const std::vector<cv::Point>& currentPoints, 
    const std::vector<cv::Point>& previousPoints,
    double currentWeight, // Weight for current frame (e.g., 0.7)
    double previousWeight, // Weight for previous frame (e.g., 0.3)
    int imgWidth,
    int numSamples)
{
    if (currentPoints.empty()) return previousPoints;
    if (previousPoints.empty()) return currentPoints;
    
    // Ensure weights are positive
    currentWeight = std::max(0.0, currentWeight);
    previousWeight = std::max(0.0, previousWeight);
    
    // Make sure we have some weighting
    if (currentWeight == 0 && previousWeight == 0) {
        currentWeight = 0.5;
        previousWeight = 0.5;
    }
    
    // Determine y-range from both sets of points
    int minY = INT_MAX;
    int maxY = 0;
    
    for (const auto& pt : currentPoints) {
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }
    
    for (const auto& pt : previousPoints) {
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }
    
    // Generate a y-based lookup map for quick interpolation
    std::map<int, int> currentLookup;  // Maps y to x
    std::map<int, int> previousLookup; // Maps y to x
    
    // Convert current points to lookup
    for (const auto& pt : currentPoints) {
        currentLookup[pt.y] = pt.x;
    }
    
    // Convert previous points to lookup
    for (const auto& pt : previousPoints) {
        previousLookup[pt.y] = pt.x;
    }
    
    // Prepare result vector with weighted interpolation
    std::vector<cv::Point> result;
    result.reserve(numSamples);
    
    // Calculate y-step
    double yStep = (maxY - minY) / std::max(1, numSamples - 1);
    
    // LIGHTWEIGHT: Simple y-based weighted interpolation
    for (int i = 0; i < numSamples; i++) {
        int y = static_cast<int>(minY + i * yStep);
        double x = 0.0;
        double weightSum = 0.0;
        bool hasValue = false;
        
        // Try to find current y-level in current points
        if (currentWeight > 0) {
            auto itCurrent = currentLookup.lower_bound(y);
            auto itCurrentPrev = (itCurrent != currentLookup.begin()) ? std::prev(itCurrent) : currentLookup.end();
            
            if (itCurrent != currentLookup.end() && itCurrentPrev != currentLookup.end()) {
                // Interpolate between two points
                int y1 = itCurrentPrev->first;
                int y2 = itCurrent->first;
                int x1 = itCurrentPrev->second;
                int x2 = itCurrent->second;
                
                // Simple linear interpolation
                double xCurrent = x1 + (x2 - x1) * (y - y1) / (double)(y2 - y1);
                
                x += xCurrent * currentWeight;
                weightSum += currentWeight;
                hasValue = true;
            }
            else if (itCurrent != currentLookup.end()) {
                // Use exact match or next greater point
                x += itCurrent->second * currentWeight;
                weightSum += currentWeight;
                hasValue = true;
            }
            else if (itCurrentPrev != currentLookup.end()) {
                // Use previous point if we're beyond the end
                x += itCurrentPrev->second * currentWeight;
                weightSum += currentWeight;
                hasValue = true;
            }
        }
        
        // Try to find current y-level in previous points
        if (previousWeight > 0) {
            auto itPrevious = previousLookup.lower_bound(y);
            auto itPreviousPrev = (itPrevious != previousLookup.begin()) ? std::prev(itPrevious) : previousLookup.end();
            
            if (itPrevious != previousLookup.end() && itPreviousPrev != previousLookup.end()) {
                // Interpolate between two points
                int y1 = itPreviousPrev->first;
                int y2 = itPrevious->first;
                int x1 = itPreviousPrev->second;
                int x2 = itPrevious->second;
                
                // Simple linear interpolation
                double xPrevious = x1 + (x2 - x1) * (y - y1) / (double)(y2 - y1);
                
                x += xPrevious * previousWeight;
                weightSum += previousWeight;
                hasValue = true;
            }
            else if (itPrevious != previousLookup.end()) {
                // Use exact match or next greater point
                x += itPrevious->second * previousWeight;
                weightSum += previousWeight;
                hasValue = true;
            }
            else if (itPreviousPrev != previousLookup.end()) {
                // Use previous point if we're beyond the end
                x += itPreviousPrev->second * previousWeight;
                weightSum += previousWeight;
                hasValue = true;
            }
        }
        
        // Check if we found any values
        if (hasValue && weightSum > 0) {
            x /= weightSum;  // Normalize by total weight
        } else {
            // Fallback behavior if no points found for this y
            // Use either the closest point from either set
            auto closestCurrent = std::min_element(currentPoints.begin(), currentPoints.end(),
                [y](const cv::Point& a, const cv::Point& b) {
                    return std::abs(a.y - y) < std::abs(b.y - y);
                });
                
            auto closestPrevious = std::min_element(previousPoints.begin(), previousPoints.end(),
                [y](const cv::Point& a, const cv::Point& b) {
                    return std::abs(a.y - y) < std::abs(b.y - y);
                });
            
            if (closestCurrent != currentPoints.end() && closestPrevious != previousPoints.end()) {
                double distCurrent = std::abs(closestCurrent->y - y);
                double distPrevious = std::abs(closestPrevious->y - y);
                
                if (distCurrent <= distPrevious) {
                    x = closestCurrent->x;
                } else {
                    x = closestPrevious->x;
                }
            } else if (closestCurrent != currentPoints.end()) {
                x = closestCurrent->x;
            } else if (closestPrevious != previousPoints.end()) {
                x = closestPrevious->x;
            }
        }
        
        // Ensure x is within image bounds
        x = std::max(0.0, std::min(double(imgWidth - 1), x));
        
        result.push_back(cv::Point(static_cast<int>(x), y));
    }
    
    // SIMPLIFIED S-SHAPE PREVENTION: 
    // Check if we need to apply a simple smoothing pass
    bool needsSmoothing = false;
    int previousDelta = 0;
    int directionChanges = 0;
    
    for (size_t i = 1; i < result.size(); i++) {
        int delta = result[i].x - result[i-1].x;
        if (previousDelta != 0 && delta * previousDelta < 0) {
            directionChanges++;
        }
        previousDelta = delta;
    }
    
    needsSmoothing = (directionChanges > 1);
    
    // Apply light smoothing if needed
    if (needsSmoothing && result.size() >= 5) {
        // Simple 3-point moving average
        const int windowSize = 3;
        std::vector<cv::Point> smoothed = result;
        
        for (size_t i = 1; i < result.size() - 1; i++) {
            smoothed[i].x = (result[i-1].x + result[i].x + result[i+1].x) / 3;
        }
        
        result = smoothed;
    }
    
    return result;
}

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
    int numSamples)
{
    // Handle empty input cases
    if (currentPoints.empty() && previousPoints.empty()) {
        result.clear();
        return;
    }
    
    if (currentPoints.empty()) {
        result = previousPoints;  // Note: This is a copy, but it's an edge case
        return;
    }
    
    if (previousPoints.empty()) {
        result = currentPoints;   // Note: This is a copy, but it's an edge case
        return;
    }
    
    // Ensure weights are positive
    currentWeight = std::max(0.0, currentWeight);
    previousWeight = std::max(0.0, previousWeight);
    
    // Make sure we have some weighting
    if (currentWeight == 0 && previousWeight == 0) {
        currentWeight = 0.5;
        previousWeight = 0.5;
    }
    
    // Determine y-range from both sets of points
    int minY = INT_MAX;
    int maxY = 0;
    
    for (const auto& pt : currentPoints) {
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }
    
    for (const auto& pt : previousPoints) {
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }
    
    // Clear and reuse maps instead of creating new ones
    currentLookup.clear();
    previousLookup.clear();
    
    // Convert current points to lookup
    for (const auto& pt : currentPoints) {
        currentLookup[pt.y] = pt.x;
    }
    
    // Convert previous points to lookup
    for (const auto& pt : previousPoints) {
        previousLookup[pt.y] = pt.x;
    }
    
    // Prepare result vector with weighted interpolation
    result.clear();
    result.reserve(numSamples);
    
    // Calculate y-step
    double yStep = (maxY - minY) / std::max(1, numSamples - 1);
    
    // LIGHTWEIGHT: Simple y-based weighted interpolation
    for (int i = 0; i < numSamples; i++) {
        int y = static_cast<int>(minY + i * yStep);
        double x = 0.0;
        double weightSum = 0.0;
        bool hasValue = false;
        
        // Try to find current y-level in current points
        if (currentWeight > 0) {
            auto itCurrent = currentLookup.lower_bound(y);
            auto itCurrentPrev = (itCurrent != currentLookup.begin()) ? std::prev(itCurrent) : currentLookup.end();
            
            if (itCurrent != currentLookup.end() && itCurrentPrev != currentLookup.end()) {
                // Interpolate between two points
                int y1 = itCurrentPrev->first;
                int y2 = itCurrent->first;
                int x1 = itCurrentPrev->second;
                int x2 = itCurrent->second;
                
                // Simple linear interpolation
                double xCurrent = x1 + (x2 - x1) * (y - y1) / (double)(y2 - y1);
                
                x += xCurrent * currentWeight;
                weightSum += currentWeight;
                hasValue = true;
            }
            else if (itCurrent != currentLookup.end()) {
                // Use exact match or next greater point
                x += itCurrent->second * currentWeight;
                weightSum += currentWeight;
                hasValue = true;
            }
            else if (itCurrentPrev != currentLookup.end()) {
                // Use previous point if we're beyond the end
                x += itCurrentPrev->second * currentWeight;
                weightSum += currentWeight;
                hasValue = true;
            }
        }
        
        // Try to find current y-level in previous points
        if (previousWeight > 0) {
            auto itPrevious = previousLookup.lower_bound(y);
            auto itPreviousPrev = (itPrevious != previousLookup.begin()) ? std::prev(itPrevious) : previousLookup.end();
            
            if (itPrevious != previousLookup.end() && itPreviousPrev != previousLookup.end()) {
                // Interpolate between two points
                int y1 = itPreviousPrev->first;
                int y2 = itPrevious->first;
                int x1 = itPreviousPrev->second;
                int x2 = itPrevious->second;
                
                // Simple linear interpolation
                double xPrevious = x1 + (x2 - x1) * (y - y1) / (double)(y2 - y1);
                
                x += xPrevious * previousWeight;
                weightSum += previousWeight;
                hasValue = true;
            }
            else if (itPrevious != previousLookup.end()) {
                // Use exact match or next greater point
                x += itPrevious->second * previousWeight;
                weightSum += previousWeight;
                hasValue = true;
            }
            else if (itPreviousPrev != previousLookup.end()) {
                // Use previous point if we're beyond the end
                x += itPreviousPrev->second * previousWeight;
                weightSum += previousWeight;
                hasValue = true;
            }
        }
        
        // Check if we found any values
        if (hasValue && weightSum > 0) {
            x /= weightSum;  // Normalize by total weight
        } else {
            // Optimized fallback - find closest point more efficiently
            int minDistCurrentPt = INT_MAX;
            int closestCurrentX = 0;
            int minDistPreviousPt = INT_MAX;
            int closestPreviousX = 0;
            
            // Linear search for closest point (avoids min_element which requires full traversal)
            for (const auto& pt : currentPoints) {
                int dist = std::abs(pt.y - y);
                if (dist < minDistCurrentPt) {
                    minDistCurrentPt = dist;
                    closestCurrentX = pt.x;
                }
            }
            
            for (const auto& pt : previousPoints) {
                int dist = std::abs(pt.y - y);
                if (dist < minDistPreviousPt) {
                    minDistPreviousPt = dist;
                    closestPreviousX = pt.x;
                }
            }
            
            // Choose closest point based on distance
            if (minDistCurrentPt != INT_MAX && minDistPreviousPt != INT_MAX) {
                if (minDistCurrentPt <= minDistPreviousPt) {
                    x = closestCurrentX;
                } else {
                    x = closestPreviousX;
                }
            } else if (minDistCurrentPt != INT_MAX) {
                x = closestCurrentX;
            } else if (minDistPreviousPt != INT_MAX) {
                x = closestPreviousX;
            }
        }
        
        // Ensure x is within image bounds
        x = std::max(0.0, std::min(double(imgWidth - 1), x));
        
        result.push_back(cv::Point(static_cast<int>(x), y));
    }
    
    // SIMPLIFIED S-SHAPE PREVENTION: 
    // Check if we need to apply a simple smoothing pass
    bool needsSmoothing = false;
    int previousDelta = 0;
    int directionChanges = 0;
    
    for (size_t i = 1; i < result.size(); i++) {
        int delta = result[i].x - result[i-1].x;
        if (previousDelta != 0 && delta * previousDelta < 0) {
            directionChanges++;
        }
        previousDelta = delta;
    }
    
    needsSmoothing = (directionChanges > 1);
    
    // Apply light smoothing if needed
    if (needsSmoothing && result.size() >= 5) {
        // Reuse tempBuffer for smoothing
        tempBuffer = result;
        
        // Simple 3-point moving average
        for (size_t i = 1; i < result.size() - 1; i++) {
            tempBuffer[i].x = (result[i-1].x + result[i].x + result[i+1].x) / 3;
        }
        
        // Swap back to result
        result.swap(tempBuffer);
    }
}

std::vector<cv::Point> smoothPoints(
    const std::vector<cv::Point>& pointList, 
    int imageWidth, 
    float smoothingStrength = 0.5f)
{
    if (pointList.size() < 3) return pointList; // Not enough points to smooth
    
    std::vector<cv::Point> smoothedPoints = pointList;
    std::vector<cv::Point> originalPoints = pointList;
    
    // Window size for smoothing (odd number)
    int windowSize = 5;
    int halfWindow = windowSize / 2;
    
    // Gaussian-like weights (center point has highest weight)
    // Example weights for window size 5: [0.1, 0.2, 0.4, 0.2, 0.1]
    std::vector<float> weights(windowSize);
    float weightSum = 0.0f;
    for (int i = 0; i < windowSize; i++) {
        float distFromCenter = std::abs(i - halfWindow);
        weights[i] = std::exp(-0.5f * distFromCenter * distFromCenter);
        weightSum += weights[i];
    }
    
    // Normalize weights
    for (int i = 0; i < windowSize; i++) {
        weights[i] /= weightSum;
    }
    
    // Apply smoothing multiple times for stronger effect
    int iterations = 1 + static_cast<int>(smoothingStrength * 2.0f);
    
    for (int iter = 0; iter < iterations; iter++) {
        // Create a copy for this iteration
        std::vector<cv::Point> tempPoints = smoothedPoints;
        
        // Apply weighted average for each point
        for (size_t i = 0; i < pointList.size(); i++) {
            float weightedSumX = 0.0f;
            float totalWeight = 0.0f;
            
            // Apply weighted window
            for (int j = -halfWindow; j <= halfWindow; j++) {
                int idx = i + j;
                
                // Handle boundary conditions with clamping
                if (idx < 0) idx = 0;
                if (idx >= pointList.size()) idx = pointList.size() - 1;
                
                float weight = weights[j + halfWindow];
                weightedSumX += tempPoints[idx].x * weight;
                totalWeight += weight;
            }
            
            // Calculate smoothed X position
            int smoothedX = static_cast<int>(weightedSumX / totalWeight);
            
            // Constrain to image bounds
            smoothedX = std::max(0, std::min(imageWidth - 1, smoothedX));
            
            // Update the point (preserving Y coordinate)
            smoothedPoints[i].x = smoothedX;
        }
    }
    
    // Blend between original and smoothed based on smoothing strength
    // This helps preserve some of the original features
    for (size_t i = 0; i < pointList.size(); i++) {
        float blendFactor = smoothingStrength;
        
        // Preserve endpoints more (less smoothing at ends)
        if (i == 0 || i == pointList.size() - 1) {
            blendFactor *= 0.5f;
        }
        
        smoothedPoints[i].x = static_cast<int>(
            originalPoints[i].x * (1.0f - blendFactor) + 
            smoothedPoints[i].x * blendFactor);
    }
    
    return smoothedPoints;
}

// In-place version with output parameter
void smoothPointsInPlace(
    const std::vector<cv::Point>& inputPoints, 
    std::vector<cv::Point>& outputPoints,  // Pre-allocated output vector
    std::vector<cv::Point>& tempBuffer,    // Re-usable temporary buffer
    int imageWidth, 
    float smoothingStrength)
{
    if (inputPoints.size() < 3) {
        // Not enough points to smooth, just copy input to output
        outputPoints = inputPoints;
        return;
    }
    
    //
    tempBuffer.clear();

    // Resize output and temp buffers if needed
    outputPoints.resize(inputPoints.size());
    tempBuffer.resize(inputPoints.size());
    
    // Copy input points to output initially
    std::copy(inputPoints.begin(), inputPoints.end(), outputPoints.begin());
    
    // Window size for smoothing (odd number)
    int windowSize = 5;
    int halfWindow = windowSize / 2;
    
    // Gaussian-like weights (center point has highest weight)
    std::vector<float> weights(windowSize);
    float weightSum = 0.0f;
    for (int i = 0; i < windowSize; i++) {
        float distFromCenter = std::abs(i - halfWindow);
        weights[i] = std::exp(-0.5f * distFromCenter * distFromCenter);
        weightSum += weights[i];
    }
    
    // Normalize weights
    for (int i = 0; i < windowSize; i++) {
        weights[i] /= weightSum;
    }
    
    // Apply smoothing multiple times for stronger effect
    int iterations = 1 + static_cast<int>(smoothingStrength * 2.0f);
    
    for (int iter = 0; iter < iterations; iter++) {
        // Copy current state to temp buffer
        std::copy(outputPoints.begin(), outputPoints.end(), tempBuffer.begin());
        
        // Apply weighted average for each point
        for (size_t i = 0; i < inputPoints.size(); i++) {
            float weightedSumX = 0.0f;
            float totalWeight = 0.0f;
            
            // Apply weighted window
            for (int j = -halfWindow; j <= halfWindow; j++) {
                int idx = i + j;
                
                // Handle boundary conditions with clamping
                if (idx < 0) idx = 0;
                if (idx >= inputPoints.size()) idx = inputPoints.size() - 1;
                
                float weight = weights[j + halfWindow];
                weightedSumX += tempBuffer[idx].x * weight;
                totalWeight += weight;
            }
            
            // Calculate smoothed X position
            int smoothedX = static_cast<int>(weightedSumX / totalWeight);
            
            // Constrain to image bounds
            smoothedX = std::max(0, std::min(imageWidth - 1, smoothedX));
            
            // Update the point (preserving Y coordinate)
            outputPoints[i].x = smoothedX;
        }
    }
    
    // Blend between original and smoothed based on smoothing strength
    for (size_t i = 0; i < inputPoints.size(); i++) {
        float blendFactor = smoothingStrength;
        
        // Preserve endpoints more (less smoothing at ends)
        if (i == 0 || i == inputPoints.size() - 1) {
            blendFactor *= 0.5f;
        }
        
        outputPoints[i].x = static_cast<int>(
            inputPoints[i].x * (1.0f - blendFactor) + 
            outputPoints[i].x * blendFactor);
    }
}

bool arePolylinesIntersecting(
    const std::vector<cv::Point>& polyline1,
    const std::vector<cv::Point>& polyline2,
    double epsilon = 1e-9,
    bool debugOutput = false)
{
    // Input validation
    if (polyline1.size() < 2 || polyline2.size() < 2) {
        if (debugOutput) {
            std::cout << "Error: Polylines must have at least 2 points." << std::endl;
        }
        return false;
    }
    
    // Helper function to check if a point is on a line segment
    auto isPointOnSegment = [](const cv::Point& p, const cv::Point& start, const cv::Point& end) -> bool {
        // Check if point p is on the line segment from start to end
        // First check if p is collinear with start and end
        int crossProduct = (p.y - start.y) * (end.x - start.x) - (p.x - start.x) * (end.y - start.y);
        if (std::abs(crossProduct) > 0)
            return false;  // Not collinear
            
        // Check if point is within the bounding box of the segment
        if (p.x < std::min(start.x, end.x) || p.x > std::max(start.x, end.x))
            return false;
        if (p.y < std::min(start.y, end.y) || p.y > std::max(start.y, end.y))
            return false;
            
        return true;
    };
    
    // Helper function to compute orientation of triplet (p, q, r)
    auto orientation = [](const cv::Point& p, const cv::Point& q, const cv::Point& r) -> int {
        int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
        if (std::abs(val) < 1e-9) return 0;  // Collinear
        return (val > 0) ? 1 : 2;  // Clockwise or counterclockwise
    };
    
    // Helper function to check if two line segments intersect
    auto doSegmentsIntersect = [&](
        const cv::Point& p1, const cv::Point& q1, 
        const cv::Point& p2, const cv::Point& q2) -> bool {
        
        // Find the four orientations needed for general and special cases
        int o1 = orientation(p1, q1, p2);
        int o2 = orientation(p1, q1, q2);
        int o3 = orientation(p2, q2, p1);
        int o4 = orientation(p2, q2, q1);
        
        // General case: different orientations for both tests
        if (o1 != o2 && o3 != o4)
            return true;
        
        // Special Cases: collinear points
        // p1, q1 and p2 are collinear and p2 lies on segment p1q1
        if (o1 == 0 && isPointOnSegment(p2, p1, q1)) return true;
        
        // p1, q1 and q2 are collinear and q2 lies on segment p1q1
        if (o2 == 0 && isPointOnSegment(q2, p1, q1)) return true;
        
        // p2, q2 and p1 are collinear and p1 lies on segment p2q2
        if (o3 == 0 && isPointOnSegment(p1, p2, q2)) return true;
        
        // p2, q2 and q1 are collinear and q1 lies on segment p2q2
        if (o4 == 0 && isPointOnSegment(q1, p2, q2)) return true;
        
        return false;  // Doesn't fall in any of the above cases
    };
    
    // Check all segments from both polylines for intersection
    for (size_t i = 0; i < polyline1.size() - 1; i++) {
        const cv::Point& p1 = polyline1[i];
        const cv::Point& p2 = polyline1[i+1];
        
        for (size_t j = 0; j < polyline2.size() - 1; j++) {
            const cv::Point& q1 = polyline2[j];
            const cv::Point& q2 = polyline2[j+1];
            
            if (doSegmentsIntersect(p1, p2, q1, q2)) {
                if (debugOutput) {
                    std::cout << "Intersection found between segments: " 
                              << "(" << p1.x << "," << p1.y << ")-(" << p2.x << "," << p2.y << ") and "
                              << "(" << q1.x << "," << q1.y << ")-(" << q2.x << "," << q2.y << ")" << std::endl;
                }
                return true;
            }
        }
    }
    
    if (debugOutput) {
        std::cout << "No intersection found between polylines." << std::endl;
    }
    
    return false;
}

/**
 * Determines if two polylines intersect
 * 
 * @param polyline1 First polyline as vector of points
 * @param polyline2 Second polyline as vector of points
 * @param distanceTolerance Tolerance for distance between points
 * @return True if the polylines intersect, false otherwise
 */
bool isPolylinesIntersect(
    const std::vector<cv::Point>& polyline1,
    const std::vector<cv::Point>& polyline2,
    int idxStart,
    double distanceTolerance = 5.0)
{
    if (polyline1.size() < 2 || polyline2.size() < 2) {
        return false;
    }
    
    // Check for exact intersection first
    if (arePolylinesIntersecting(polyline1, polyline2)) {
        return true;
    }
    
    // Check minimum distance between the polylines
    float minDistance = std::numeric_limits<float>::max();
    
    // For each segment in polyline1
    for (size_t i = idxStart; i < polyline1.size() - 1; i++) {
        const cv::Point& p1 = polyline1[i];
        const cv::Point& p2 = polyline1[i+1];
        cv::Point2f segment1Dir(p2.x - p1.x, p2.y - p1.y);
        float segment1Length = std::sqrt(segment1Dir.x * segment1Dir.x + segment1Dir.y * segment1Dir.y);
        
        // Normalize direction vector
        if (segment1Length > 0) {
            segment1Dir.x /= segment1Length;
            segment1Dir.y /= segment1Length;
        }
        
        // For each segment in polyline2
        for (size_t j = idxStart; j < polyline2.size() - 1; j++) {
            const cv::Point& q1 = polyline2[j];
            const cv::Point& q2 = polyline2[j+1];
            cv::Point2f segment2Dir(q2.x - q1.x, q2.y - q1.y);
            float segment2Length = std::sqrt(segment2Dir.x * segment2Dir.x + segment2Dir.y * segment2Dir.y);

            // Skip zero-length segments
            if (segment2Length < 1e-6f)
            {
                continue;
            }

            // Normalize direction vector
            if (segment2Length > 0) {
                segment2Dir.x /= segment2Length;
                segment2Dir.y /= segment2Length;
            }
            
            // Calculate minimum distance between segments
            // Using algorithm for distance between two line segments
            cv::Point2f v1(p1);
            cv::Point2f v2(p2);
            cv::Point2f v3(q1);
            cv::Point2f v4(q2);
            
            cv::Point2f v13(v3.x - v1.x, v3.y - v1.y);
            cv::Point2f v21(v1.x - v2.x, v1.y - v2.y);
            cv::Point2f v43(v3.x - v4.x, v3.y - v4.y);
            
            // Check if segments are nearly parallel
            float dotProduct = segment1Dir.x * segment2Dir.x + segment1Dir.y * segment2Dir.y;
            bool nearlyParallel = std::abs(std::abs(dotProduct) - 1.0f) < 0.01f;
            
            if (nearlyParallel) {
                // For parallel segments, check distances between endpoints and segments
                float norm21 = cv::norm(v21);
                float norm43 = cv::norm(v43);

                // Avoid division by zero
                if (norm21 > 1e-6f && norm43 > 1e-6f)
                {
                    float d1 = cv::norm(v13.cross(v21)) / cv::norm(v21);
                    float d2 = cv::norm((v3 - v2).cross(v21)) / cv::norm(v21);
                    float d3 = cv::norm(v13.cross(v43)) / cv::norm(v43);
                    float d4 = cv::norm((v1 - v4).cross(v43)) / cv::norm(v43);
                    
                    minDistance = std::min(minDistance, std::min(std::min(d1, d2), std::min(d3, d4)));
                }
            } else {
                // For non-parallel segments, calculate closest point
                cv::Point2f dP = v1 - v3;
                
                float a = segment1Dir.dot(segment1Dir);
                float b = segment1Dir.dot(segment2Dir);
                float c = segment2Dir.dot(segment2Dir);
                float d = segment1Dir.dot(dP);
                float e = segment2Dir.dot(dP);
                
                float det = a * c - b * b;
                
                float s, t;
                
                if (det < 1e-9) {
                    // Nearly parallel segments
                    s = 0;
                    t = (b > c ? d / b : e / c);
                } else {
                    s = (b * e - c * d) / det;
                    t = (a * e - b * d) / det;
                }
                
                // Clamp parameters to segment boundaries
                s = std::max(0.0f, std::min(1.0f, s));
                t = std::max(0.0f, std::min(1.0f, t));
                
                // Calculate closest points on both segments
                cv::Point2f point1 = v1 + segment1Dir * (segment1Length * s);
                cv::Point2f point2 = v3 + segment2Dir * (segment2Length * t);
                
                // Calculate distance between closest points
                float distance = cv::norm(point1 - point2);
                minDistance = std::min(minDistance, static_cast<float>(distance));
            }
        }
    }
    
    return minDistance <= distanceTolerance;
}

// Detect intersection between two line segments
bool lineIntersection(const cv::Point& p1, const cv::Point& p2, 
                     const cv::Point& p3, const cv::Point& p4,
                     cv::Point& intersection) {
    // Line segment 1: p1 to p2
    // Line segment 2: p3 to p4
    
    // Convert to line form ax + by = c
    float a1 = p2.y - p1.y;
    float b1 = p1.x - p2.x;
    float c1 = a1 * p1.x + b1 * p1.y;
    
    float a2 = p4.y - p3.y;
    float b2 = p3.x - p4.x;
    float c2 = a2 * p3.x + b2 * p3.y;
    
    float determinant = a1 * b2 - a2 * b1;
    
    // If determinant is near zero, lines are parallel
    if (std::abs(determinant) < 1e-6) {
        return false;
    }
    
    // Calculate intersection point
    float x = (b2 * c1 - b1 * c2) / determinant;
    float y = (a1 * c2 - a2 * c1) / determinant;
    intersection = cv::Point(x, y);
    
    // Check if intersection is on both line segments
    bool onSegment1 = (std::min(p1.x, p2.x) <= x && x <= std::max(p1.x, p2.x)) &&
                      (std::min(p1.y, p2.y) <= y && y <= std::max(p1.y, p2.y));
                      
    bool onSegment2 = (std::min(p3.x, p4.x) <= x && x <= std::max(p3.x, p4.x)) &&
                      (std::min(p3.y, p4.y) <= y && y <= std::max(p3.y, p4.y));
    
    return onSegment1 && onSegment2;
}

// Find intersection point between two polylines
bool findPolylineIntersection(const std::vector<cv::Point>& polyline1, 
                              const std::vector<cv::Point>& polyline2,
                              cv::Point& intersection) {
    // Check each segment of polyline1 against each segment of polyline2
    for (size_t i = 1; i < polyline1.size(); i++) {
        for (size_t j = 1; j < polyline2.size(); j++) {
            if (lineIntersection(polyline1[i-1], polyline1[i],
                                polyline2[j-1], polyline2[j],
                                intersection)) {
                return true;
            }
        }
    }
    
    return false;
}

// Find all intersection points (in case there are multiple)
std::vector<cv::Point> findAllPolylineIntersections(
    const std::vector<cv::Point>& polyline1, 
    const std::vector<cv::Point>& polyline2) {
    
    std::vector<cv::Point> intersections;
    cv::Point intersection;
    
    // Check each segment of polyline1 against each segment of polyline2
    for (size_t i = 1; i < polyline1.size(); i++) {
        for (size_t j = 1; j < polyline2.size(); j++) {
            if (lineIntersection(polyline1[i-1], polyline1[i],
                                polyline2[j-1], polyline2[j],
                                intersection)) {
                intersections.push_back(intersection);
            }
        }
    }
    
    return intersections;
}

double cross_product(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.y - a.y * b.x;
}

cv::Point2f findLineIntersection(
    const cv::Point& p1_int, const cv::Point& p2_int,
    const cv::Point& p3_int, const cv::Point& p4_int)
{
    // Convert integer points to float points for calculation precision
    cv::Point2f p1(static_cast<float>(p1_int.x), static_cast<float>(p1_int.y));
    cv::Point2f p2(static_cast<float>(p2_int.x), static_cast<float>(p2_int.y));
    cv::Point2f p3(static_cast<float>(p3_int.x), static_cast<float>(p3_int.y));
    cv::Point2f p4(static_cast<float>(p4_int.x), static_cast<float>(p4_int.y));

    cv::Point2f v12 = p2 - p1; // Direction vector of line 1
    cv::Point2f v34 = p4 - p3; // Direction vector of line 2
    cv::Point2f v13 = p3 - p1; // Vector from p1 to p3

    double denominator = cross_product(v12, v34);

    // Check if lines are parallel (denominator is close to zero)
    if (std::abs(denominator) < std::numeric_limits<double>::epsilon()) {
        // Lines are parallel or collinear, no unique intersection
        return cv::Point2f(0, 0);
    }

    // Calculate the parameter t for line 1
    double t_num = cross_product(v13, v34);
    double t = t_num / denominator;

    // Calculate the intersection point using the parameter t for line 1
    cv::Point2f intersection = p1 + t * v12;

    return intersection;
}

void calculateLaneAnglesSimple(
    const cv::Point& pLeftFar,
    const cv::Point& pLeftCarhood,
    const cv::Point& pRightFar,
    const cv::Point& pRightCarhood,
    float& angleLeft,
    float& angleRight)
{
    // 1. Calculate vectors for both lane lines
    cv::Point2f leftVec(pLeftCarhood.x - pLeftFar.x, pLeftCarhood.y - pLeftFar.y);
    cv::Point2f rightVec(pRightCarhood.x - pRightFar.x, pRightCarhood.y - pRightFar.y);
    
    // 2. Normalize vectors
    float leftMag = std::sqrt(leftVec.x * leftVec.x + leftVec.y * leftVec.y);
    float rightMag = std::sqrt(rightVec.x * rightVec.x + rightVec.y * rightVec.y);
    
    if (leftMag > 1e-6f) {
        leftVec.x /= leftMag;
        leftVec.y /= leftMag;
    }
    
    if (rightMag > 1e-6f) {
        rightVec.x /= rightMag;
        rightVec.y /= rightMag;
    }
    
    // 3. Vertical vector (down)
    cv::Point2f verticalVec(0.0f, 1.0f);
    
    // 4. Calculate dot product (cosine value)
    float leftCosAngle = leftVec.x * verticalVec.x + leftVec.y * verticalVec.y;
    float rightCosAngle = rightVec.x * verticalVec.x + rightVec.y * verticalVec.y;
    
    // 5. Limit range to avoid floating point errors
    leftCosAngle = std::max(-1.0f, std::min(1.0f, leftCosAngle));
    rightCosAngle = std::max(-1.0f, std::min(1.0f, rightCosAngle));
    
    // 6. Calculate angle
    angleLeft = std::acos(leftCosAngle) * 180.0f / M_PI;
    angleRight = std::acos(rightCosAngle) * 180.0f / M_PI;

    // 7. Convert to angle with horizontal line (more intuitive)
    angleLeft = 90.0f - angleLeft;
    angleRight = 90.0f - angleRight;
}

cv::Point rotatePointClockwise(
    const cv::Point& point,
    float angleDegrees,
    const cv::Point& center)
{
    // Convert angle to radians and negate it (OpenCV rotation is positive for counter-clockwise)
    float angleRadians = -angleDegrees * CV_PI / 180.0f;
    
    // Calculate rotation matrix
    float cosA = std::cos(angleRadians);
    float sinA = std::sin(angleRadians);
    
    // Translate to rotation center
    int x = point.x - center.x;
    int y = point.y - center.y;
    
    // Rotate point
    int xNew = static_cast<int>(x * cosA - y * sinA);
    int yNew = static_cast<int>(x * sinA + y * cosA);
    
    // Translate back to original position
    return cv::Point(xNew + center.x, yNew + center.y);
}

float calculateVectorAngle(
    const cv::Point2f& vectorA,
    const cv::Point2f& vectorB,
    bool inDegrees = true)
{
    // Calculate dot product
    float dotProduct = vectorA.x * vectorB.x + vectorA.y * vectorB.y;
    
    // Calculate vector magnitudes
    float magnitudeA = std::sqrt(vectorA.x * vectorA.x + vectorA.y * vectorA.y);
    float magnitudeB = std::sqrt(vectorB.x * vectorB.x + vectorB.y * vectorB.y);
    
    // Calculate cosine of the angle
    float cosTheta = dotProduct / ((magnitudeA * magnitudeB) + 1e-6f);
    
    // Handle numerical errors
    cosTheta = std::max(-1.0f, std::min(1.0f, cosTheta));
    
    // Calculate angle
    float angleRadians = std::acos(cosTheta);
    
    // Convert to degrees or return radians
    return inDegrees ? angleRadians * 180.0f / CV_PI : angleRadians;
}

float calculateAngleBetweenPoints(
    const cv::Point& pointO,
    const cv::Point& pointA,
    const cv::Point& pointB,
    bool inDegrees = true)
{
    // Calculate vector OA and OB
    cv::Point2f vectorOA(pointA.x - pointO.x, pointA.y - pointO.y);
    cv::Point2f vectorOB(pointB.x - pointO.x, pointB.y - pointO.y);
    
    // Use vector angle function
    return calculateVectorAngle(vectorOA, vectorOB, inDegrees);
}

/**
 * Generates evenly spaced points along a straight line between two endpoints
 * 
 * @param startPoint The further point (top point with smaller y-coordinate)
 * @param endPoint The closer point (bottom point with larger y-coordinate)
 * @param numPoints Number of points to generate (default: 15)
 * @param resultPoints Vector of evenly spaced points along the line
 */
void generateLinearPoints(
    const cv::Point& startPoint, 
    const cv::Point& endPoint, 
    int numPoints,
    std::vector<cv::Point>& resultPoints) 
{    
    // Ensure startPoint is actually the top point (smaller y-coordinate)
    cv::Point topPoint = (startPoint.y < endPoint.y) ? startPoint : endPoint;
    cv::Point bottomPoint = (startPoint.y < endPoint.y) ? endPoint : startPoint;
    
    // Calculate the line parameters: y = mx + b or x = my + b
    // We'll use parametric form for better precision
    float dx = bottomPoint.x - topPoint.x;
    float dy = bottomPoint.y - topPoint.y;
    
    // Generate points along the line using linear interpolation
    for (int i = 0; i < numPoints; i++) {
        float t = static_cast<float>(i) / (numPoints - 1);  // Normalize to [0,1]
        
        int x = static_cast<int>(topPoint.x + t * dx);
        int y = static_cast<int>(topPoint.y + t * dy);
        
        resultPoints.push_back(cv::Point(x, y));
    }
}

bool isLaneShapeUnnatural(
    const std::vector<cv::Point>& lane,
    float slopeChangeThreshold = 0.0f)
{
    if (lane.size() < 5) {
        return false;
    }
    
    // Calculate direction changes between adjacent points
    int inflectionCount = 0;
    int maxInflectionThreshold = 1;
    std::vector<float> slopes;
    
    for (size_t i = 1; i < lane.size(); i++) {
        float dx = static_cast<float>(lane[i].x - lane[i-1].x);
        float dy = static_cast<float>(lane[i].y - lane[i-1].y);
        
        if (std::abs(dy) < 1e-5f) {
            dy = 1e-5f; // Avoid division by zero
        }
        
        slopes.push_back(dx / dy);
    }
    
    // Detect number of slope sign changes (inflection points)
    for (size_t i = 1; i < slopes.size(); i++) {
        // Original condition: slopes[i] * slopes[i-1] < 0
        // New condition: Also check if the absolute change in slope is greater than the threshold
        bool isSignChange = slopes[i] * slopes[i-1] < 0;
        
        // If there is a slope change threshold, also check the absolute change in slope
        bool isLargeChange = false;
        if (slopeChangeThreshold > 0.0f) {
            float slopeChange = std::abs(slopes[i] - slopes[i-1]);
            isLargeChange = slopeChange > slopeChangeThreshold;
        }
        
        // Also satisfy the sign change condition and large change condition (if a threshold is set)
        if (isSignChange && (slopeChangeThreshold <= 0.0f || isLargeChange)) {
            inflectionCount++;
        }
    }
    
    // Determine if the number of inflection points exceeds the threshold
    return inflectionCount > maxInflectionThreshold;
}

// Calculate the slope of a line
float calculateLineSlope(const std::vector<cv::Point>& line) {
    if (line.size() < 2) {
        return 0.0f;
    }
    
    // Use first and last points to calculate overall slope
    cv::Point first = line.front();
    cv::Point last = line.back();
    
    // Avoid division by zero
    if (last.x == first.x) {
        return 1000.0f; // Represent near-vertical line
    }
    
    return static_cast<float>(last.y - first.y) / (last.x - first.x);
}

cv::Point findPointAtY(const std::vector<cv::Point>& polyline, int targetY) {
    // Handle empty polyline case
    if (polyline.empty()) {
        return cv::Point(0, targetY); // Return default point with requested y
    }
    
    // Find min and max y values in the polyline
    int minY = INT_MAX;
    int maxY = INT_MIN;
    for (const auto& pt : polyline) {
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }
    
    // Handle case where targetY is outside the polyline's y-range
    if (targetY < minY) {
        // Find the point with minimum y
        for (const auto& pt : polyline) {
            if (pt.y == minY) {
                return cv::Point(pt.x, targetY);
            }
        }
        return cv::Point(polyline.front().x, targetY);
    }
    
    if (targetY > maxY) {
        // Find the point with maximum y
        for (const auto& pt : polyline) {
            if (pt.y == maxY) {
                return cv::Point(pt.x, targetY);
            }
        }
        return cv::Point(polyline.back().x, targetY);
    }
    
    // Find the two points that bound the target y
    for (size_t i = 1; i < polyline.size(); i++) {
        // Check if this segment contains the target y
        if ((polyline[i-1].y <= targetY && polyline[i].y >= targetY) || 
            (polyline[i-1].y >= targetY && polyline[i].y <= targetY)) {
            
            // Avoid division by zero
            if (polyline[i].y == polyline[i-1].y) {
                return cv::Point((polyline[i].x + polyline[i-1].x) / 2, targetY);
            }
            
            // Linear interpolation to find x
            float ratio = static_cast<float>(targetY - polyline[i-1].y) / 
                          (polyline[i].y - polyline[i-1].y);
            
            int x = polyline[i-1].x + ratio * (polyline[i].x - polyline[i-1].x);
            
            return cv::Point(x, targetY);
        }
    }
    
    // Fallback if no segment is found (shouldn't happen if y is in range)
    return cv::Point(polyline.back().x, targetY);
}

/**
 * Calculate the x-coordinate based on the slope, reference point, and target y-value, returning the complete coordinate
 * 
 * @param slope Line segment slope
 * @param refPoint Reference point (known point)
 * @param targetY Target y-coordinate
 * @return Complete Point containing the corresponding x-coordinate
 */
cv::Point findPointBySlope(float slope, const cv::Point& refPoint, int targetY) {
    // Check for vertical line case
    if (std::abs(slope) > 999.0f) {
        // For vertical lines, x-coordinate remains unchanged
        return cv::Point(refPoint.x, targetY);
    }
    
    // Using point-slope formula: (x - x1) = (y - y1) / slope
    // Transformed to: x = x1 + (y - y1) / slope
    
    // Difference between target y and reference point y
    int deltaY = targetY - refPoint.y;
    
    // Calculate the corresponding x-coordinate
    int targetX = refPoint.x + static_cast<int>(deltaY / slope);
    
    return cv::Point(targetX, targetY);
}

/**
 * Calculate the point on a line segment at a given y-value
 * 
 * @param line Collection of points on the line segment
 * @param targetY Target y-coordinate
 * @return Complete Point containing the corresponding x-coordinate
 */
cv::Point findPointOnLineByY(const std::vector<cv::Point>& line, int targetY) {
    if (line.size() < 2) {
        // Not enough points to form a line segment
        return cv::Point(0, targetY);
    }
    
    // For an array of line segments, check if the target y is within the line segment range
    int minY = std::numeric_limits<int>::max();
    int maxY = std::numeric_limits<int>::min();
    
    for (const auto& pt : line) {
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }
    
    // Check if targetY is within the y-range of the line segment
    if (targetY < minY || targetY > maxY) {
        // Target y is out of range, need to extend the line segment
        // Use the overall slope of the line segment for extrapolation
        float slope = calculateLineSlope(line);
        
        // Choose the closest endpoint as the reference point
        cv::Point refPoint = (std::abs(targetY - minY) < std::abs(targetY - maxY)) 
                              ? *std::min_element(line.begin(), line.end(), 
                                                [](const cv::Point& a, const cv::Point& b) { 
                                                    return a.y < b.y; 
                                                })
                              : *std::max_element(line.begin(), line.end(), 
                                                [](const cv::Point& a, const cv::Point& b) { 
                                                    return a.y < b.y; 
                                                });
                                                
        return findPointBySlope(slope, refPoint, targetY);
    }
    
    // Target y is within the line segment range, find the corresponding segment interval
    for (size_t i = 0; i < line.size() - 1; ++i) {
        int y1 = line[i].y;
        int y2 = line[i + 1].y;
        
        // Check if the target y is within the current segment interval
        if ((y1 <= targetY && targetY <= y2) || (y2 <= targetY && targetY <= y1)) {
            // Interval found, calculate the slope for this small segment
            int x1 = line[i].x;
            int x2 = line[i + 1].x;
            
            // Avoid division by zero
            if (y1 == y2) {
                // Horizontal line segment, x is the midpoint of x1 and x2
                return cv::Point((x1 + x2) / 2, targetY);
            }
            
            // Use linear interpolation to calculate x
            float localSlope = static_cast<float>(x2 - x1) / (y2 - y1);
            int targetX = x1 + static_cast<int>((targetY - y1) * localSlope);
            
            return cv::Point(targetX, targetY);
        }
    }
    
    // If no suitable interval is found (theoretically shouldn't happen)
    // Use the overall slope for calculation
    float slope = calculateLineSlope(line);
    return findPointBySlope(slope, line[0], targetY);
}
bool isSymmetricLanes(const std::vector<cv::Point>& leftLane, 
                     const std::vector<cv::Point>& rightLane,
                     float angleTolerance = 5.0f) {
    // Calculate the slope of both lines
    float leftSlope = calculateLineSlope(leftLane);
    float rightSlope = calculateLineSlope(rightLane);
    
    // Convert slopes to angles (relative to horizontal line)
    float leftAngle = std::atan(leftSlope) * 180.0f / CV_PI;
    float rightAngle = std::atan(rightSlope) * 180.0f / CV_PI;
    
    // On a straight road, the absolute values of these angles should be similar
    return std::abs(std::abs(leftAngle) - std::abs(rightAngle)) < angleTolerance;
}

void drawTextWithBorder(
    cv::Mat& img, 
    const std::string& text, 
    cv::Point position, 
    int fontFace, 
    double fontScale, 
    cv::Scalar textColor, 
    cv::Scalar borderColor, 
    int borderThickness = 1,
    int lineType = cv::LINE_AA)
{
    // Draw border by drawing text multiple times with offsets
    for(int i = -borderThickness; i <= borderThickness; i++) {
        for(int j = -borderThickness; j <= borderThickness; j++) {
            if(i == 0 && j == 0) continue; // Skip the center (will be drawn later)
            cv::putText(img, text, cv::Point(position.x + i, position.y + j), 
                      fontFace, fontScale, borderColor, borderThickness, lineType);
        }
    }
    
    // Then draw thinner text in main color
    int textThickness = std::max(borderThickness, 1);
    cv::putText(img, text, position, fontFace, fontScale, textColor, textThickness, lineType);
}

void drawDebugText(
    cv::Mat& img, 
    const std::string& text, 
    cv::Point position)
{
    drawTextWithBorder(img, text, position, 
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

void drawPoints(
    const cv::Point& point,
    int modelWidth,
    int modelHeight,
    int imgWidth,
    int imgHeight,
    cv::Mat& image,
    cv::Scalar color)
{
    // Calculate scaling factors
    float scaleX = static_cast<float>(imgWidth) / modelWidth;
    float scaleY = static_cast<float>(imgHeight) / modelHeight;
    
    // Scale and convert points
    std::vector<std::pair<int, int>> keypoints;
    
    int scaledX = static_cast<int>(point.x * scaleX);
    int scaledY = static_cast<int>(point.y * scaleY);
    keypoints.push_back(std::make_pair(scaledX, scaledY));

    imgUtil::PoseKeyPoints(image, keypoints, color, 5);    
}

// DEPRECATED: Function using BoundingBox removed
// void drawLanePoints(
//     const BoundingBox& box,
//     int modelWidth,
//     int modelHeight,
//     int imgWidth,
//     int imgHeight,
//     cv::Mat& image,
//     cv::Scalar color)
// {
//     // Calculate scaling factors
//     float scaleX = static_cast<float>(imgWidth) / modelWidth;
//     float scaleY = static_cast<float>(imgHeight) / modelHeight;
//     
//     // Scale and convert points
//     std::vector<std::pair<int, int>> scaledPoints;
//     
//     for (const auto& point : box.pose_kpts)
//     {
//         int scaledX = static_cast<int>(point.first * scaleX);
//         int scaledY = static_cast<int>(point.second * scaleY);
//         scaledPoints.emplace_back(scaledX, scaledY);
//     }
//     
//     imgUtil::PoseKeyPoints(image, scaledPoints, color, 3);
// }

void drawLanePoints(
    const std::vector<cv::Point>& leftPoints,
    const std::vector<cv::Point>& rightPoints,
    int modelWidth,
    int modelHeight,
    int imgWidth,
    int imgHeight,
    cv::Mat& image,
    cv::Scalar color)
{
    // Calculate scaling factors
    float scaleX = static_cast<float>(imgWidth) / modelWidth;
    float scaleY = static_cast<float>(imgHeight) / modelHeight;
    
    // Scale and convert points
    std::vector<std::pair<int, int>> leftKeypoints;
    std::vector<std::pair<int, int>> rightKeypoints;
    std::vector<cv::Point> scaledLeftPoints;
    std::vector<cv::Point> scaledRightPoints;
    
    for (const auto& point : leftPoints)
    {
        int scaledX = static_cast<int>(point.x * scaleX);
        int scaledY = static_cast<int>(point.y * scaleY);
        leftKeypoints.push_back(std::make_pair(scaledX, scaledY));
        scaledLeftPoints.push_back(cv::Point(scaledX, scaledY));
    }
    
    for (const auto& point : rightPoints)
    {
        int scaledX = static_cast<int>(point.x * scaleX);
        int scaledY = static_cast<int>(point.y * scaleY);
        rightKeypoints.push_back(std::make_pair(scaledX, scaledY));
        scaledRightPoints.push_back(cv::Point(scaledX, scaledY));
    }

    imgUtil::PoseKeyPoints(image, leftKeypoints, color, 3);
    imgUtil::PoseKeyPoints(image, rightKeypoints, color, 3);

    // Draw polylines connecting the points
    if (scaledLeftPoints.size() > 1) {
        std::vector<std::vector<cv::Point>> leftContours = {scaledLeftPoints};
        cv::polylines(image, leftContours, false, color, 2, cv::LINE_AA);
    }
    
    if (scaledRightPoints.size() > 1) {
        std::vector<std::vector<cv::Point>> rightContours = {scaledRightPoints};
        cv::polylines(image, rightContours, false, color, 2, cv::LINE_AA);
    }
}

void drawLanePolygon(
    const std::vector<cv::Point>& leftPoints,
    const std::vector<cv::Point>& rightPoints,
    int modelWidth,
    int modelHeight,
    int imgWidth,
    int imgHeight,
    cv::Mat& image,
    cv::Scalar fillColor)
{
    cv::Mat laneLineResult;
#ifndef SAV837
    laneLineResult.create(imgHeight, imgWidth, CV_8UC3);
#else
    laneLineResult.create(imgHeight, imgWidth, CV_8UC4);
#endif
    laneLineResult = cv::Scalar(0, 0, 0); // Set all values to zero


    // Calculate scaling factors
    float scaleX = static_cast<float>(imgWidth) / modelWidth;
    float scaleY = static_cast<float>(imgHeight) / modelHeight;
    
    // Scale and convert points
    std::vector<std::pair<int, int>> leftKeypoints;
    std::vector<std::pair<int, int>> rightKeypoints;
    std::vector<cv::Point> scaledLeftPoints;
    std::vector<cv::Point> scaledRightPoints;
    
    for (const auto& point : leftPoints)
    {
        int scaledX = static_cast<int>(point.x * scaleX);
        int scaledY = static_cast<int>(point.y * scaleY);
        leftKeypoints.push_back(std::make_pair(scaledX, scaledY));
        scaledLeftPoints.push_back(cv::Point(scaledX, scaledY));
    }
    
    for (const auto& point : rightPoints)
    {
        int scaledX = static_cast<int>(point.x * scaleX);
        int scaledY = static_cast<int>(point.y * scaleY);
        rightKeypoints.push_back(std::make_pair(scaledX, scaledY));
        scaledRightPoints.push_back(cv::Point(scaledX, scaledY));
    }

    if (!scaledLeftPoints.empty() && !scaledRightPoints.empty()) {
        // Create a closed polygon by connecting left and right lanes
        std::vector<cv::Point> lanePolygon;
        
        // Add all points from left lane (from top to bottom)
        lanePolygon.insert(lanePolygon.end(), scaledLeftPoints.begin(), scaledLeftPoints.end());
        
        // Add points from right lane in reverse order (from bottom to top)
        lanePolygon.insert(lanePolygon.end(), scaledRightPoints.rbegin(), scaledRightPoints.rend());
        
        // Fill the polygon
        std::vector<std::vector<cv::Point>> contours = {lanePolygon};
        cv::fillPoly(laneLineResult, contours, fillColor, cv::LINE_AA);
        cv::addWeighted(image, 1.0, laneLineResult, 0.3, 0, image);
    }
}

#if defined(CV28) || defined(CV28_SIMULATOR)
static int image_buffer_to_mat_yuv2bgr_nv12(const ea_env_image_buffer_t *image_buffer, cv::Mat &bgr)
{
	int rval = EA_SUCCESS;
	cv::Mat nv12(image_buffer->height * 3 / 2, image_buffer->width, CV_8UC1, image_buffer->buf, image_buffer->pitch);

	do {
#if CV_VERSION_MAJOR < 4
		cv::cvtColor(nv12, bgr, CV_YUV2BGR_NV12);
#else
		cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
#endif
		// EA_R_ASSERT(bgr.isContinuous());
	} while (0);

	return rval;
}

static int image_buffer_to_mat_yuv2yuv_nv12(const ea_env_image_buffer_t *image_buffer, cv::Mat &yuv_nv12)
{
	int rval = EA_SUCCESS;
	cv::Mat nv12(image_buffer->height * 3 / 2, image_buffer->width, CV_8UC1, image_buffer->buf, image_buffer->pitch);

	do {
		nv12.copyTo(yuv_nv12);
	} while (0);

	return rval;
}

static int image_buffer_to_mat_bgr2bgr(const ea_env_image_buffer_t *image_buffer, cv::Mat &bgr)
{
	int rval = EA_SUCCESS;
	void *c1_data = image_buffer->buf;
	void *c2_data = image_buffer->buf + image_buffer->height * image_buffer->pitch;
	void *c3_data = image_buffer->buf + image_buffer->height * image_buffer->pitch * 2;
	cv::Mat c1(image_buffer->height, image_buffer->width, CV_8UC1, c1_data, image_buffer->pitch);
	cv::Mat c2(image_buffer->height, image_buffer->width, CV_8UC1, c2_data, image_buffer->pitch);
	cv::Mat c3(image_buffer->height, image_buffer->width, CV_8UC1, c3_data, image_buffer->pitch);
	std::vector<cv::Mat> channels;

	do {

		if (image_buffer->type == EA_ENV_IMAGE_BUFFER_TYPE_GRAYSCALE) {
			channels.push_back(c1);
		}
		else {
			channels.push_back(c1);
			channels.push_back(c2);
			channels.push_back(c3);
		}

		cv::merge(channels, bgr);
		// EA_R_ASSERT(bgr.isContinuous());
	} while (0);

	return rval;
}

static int image_buffer_to_mat_rgb2bgr(const ea_env_image_buffer_t *image_buffer, cv::Mat &bgr)
{
	int rval = EA_SUCCESS;
	void *c1_data = image_buffer->buf;
	void *c2_data = image_buffer->buf + image_buffer->height * image_buffer->pitch;
	void *c3_data = image_buffer->buf + image_buffer->height * image_buffer->pitch * 2;
	cv::Mat c1(image_buffer->height, image_buffer->width, CV_8UC1, c1_data, image_buffer->pitch);
	cv::Mat c2(image_buffer->height, image_buffer->width, CV_8UC1, c2_data, image_buffer->pitch);
	cv::Mat c3(image_buffer->height, image_buffer->width, CV_8UC1, c3_data, image_buffer->pitch);
	std::vector<cv::Mat> channels;

	do {

		if (image_buffer->type == EA_ENV_IMAGE_BUFFER_TYPE_GRAYSCALE) {
			channels.push_back(c1);
		}
		else {
			channels.push_back(c1);
			channels.push_back(c2);
			channels.push_back(c3);
		}

		std::swap(channels[0], channels[2]);
		cv::merge(channels, bgr);
		// EA_R_ASSERT(bgr.isContinuous());
	} while (0);

	return rval;
}

cv::Mat convertTensorToMat(ea_tensor_t* tensor)
{
	int rval = EA_SUCCESS;
	ea_env_image_t *image = NULL;
	ea_env_image_buffer_t image_buffer;
	ea_tensor_t *related = NULL;
	uint8_t *data = NULL;
	uint8_t *buf = NULL;

    int color_mode = EA_TENSOR_COLOR_MODE_BGR;
    cv::Mat bgr_mat;
	do {

		memset(&image_buffer, 0, sizeof(ea_env_image_buffer_t));
		image_buffer.height = ea_tensor_shape(tensor)[EA_H];
		image_buffer.width = ea_tensor_shape(tensor)[EA_W];
		image_buffer.pitch = ea_tensor_pitch(tensor);

		switch (color_mode) {
            case EA_TENSOR_COLOR_MODE_YUV_NV12:
                image_buffer.type = EA_ENV_IMAGE_BUFFER_TYPE_YUV_NV12;

                related = ea_tensor_related(tensor);
                if (related != NULL) {
                    std::cout << "related != NULL" << std::endl;
                }
                else {
                    // EA_R_ASSERT((ea_tensor_shape(tensor)[EA_H] % 3) == 0);
                    image_buffer.height = ea_tensor_shape(tensor)[EA_H] * 2 / 3;
                }

                buf = (uint8_t *)malloc(image_buffer.height * image_buffer.pitch * 3 / 2);
                // EA_R_ASSERT(buf != NULL);
                data = (uint8_t *)ea_tensor_data_for_read(tensor, EA_CPU);
                memcpy(buf, data, ea_tensor_size(tensor));
                if (related != NULL) {
                    data = (uint8_t *)ea_tensor_data_for_read(related, EA_CPU);
                    memcpy(buf + ea_tensor_size(tensor), data, ea_tensor_size(related));
                }
                image_buffer.buf = buf;
                break;
            case EA_TENSOR_COLOR_MODE_RGB:
                image_buffer.type = EA_ENV_IMAGE_BUFFER_TYPE_RGB;
                image_buffer.buf = (uint8_t *)ea_tensor_data_for_read(tensor, EA_CPU);
                break;
            case EA_TENSOR_COLOR_MODE_BGR:
                image_buffer.type = EA_ENV_IMAGE_BUFFER_TYPE_BGR;
                image_buffer.buf = (uint8_t *)ea_tensor_data_for_read(tensor, EA_CPU);
                break;
            case EA_TENSOR_COLOR_MODE_GRAY:
                image_buffer.type = EA_ENV_IMAGE_BUFFER_TYPE_GRAYSCALE;
                image_buffer.buf = (uint8_t *)ea_tensor_data_for_read(tensor, EA_CPU);
                break;
            default :
                break;
		}

        do {
            int rval = EA_SUCCESS;
            switch (image_buffer.type)
            {
                case EA_ENV_IMAGE_BUFFER_TYPE_YUV_NV12:
                    rval = image_buffer_to_mat_yuv2bgr_nv12(&image_buffer, bgr_mat);
                    break;
                case EA_ENV_IMAGE_BUFFER_TYPE_RGB:
                    rval = image_buffer_to_mat_rgb2bgr(&image_buffer, bgr_mat);
                    break;
                case EA_ENV_IMAGE_BUFFER_TYPE_BGR:
                    rval = image_buffer_to_mat_bgr2bgr(&image_buffer, bgr_mat);
                    break;
                case EA_ENV_IMAGE_BUFFER_TYPE_GRAYSCALE:
                    rval = image_buffer_to_mat_bgr2bgr(&image_buffer, bgr_mat);
                    break;
                default :
                    break;
                }
        } while(0);
        
		if (buf) {
			free(buf);
			buf = NULL;
		}
	} while (0);


    // bool success = cv::imwrite("img.jpg", bgr_mat);
    //  if (!success) { std::cerr << "Failed to save img.jpg" << std::endl; }
    //  else { std::cout << "Saved img.jpg successfully." << std::endl; }

    return bgr_mat;
}
#endif
}