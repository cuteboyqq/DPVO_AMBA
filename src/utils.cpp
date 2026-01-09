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

#include "utils.hpp"

namespace utils
{
bool checkFileExists(const std::string& name) {
    // Check for basic invalid path patterns
    if (name.empty() || 
        name.find("/.") != std::string::npos || // Catches hidden files and invalid patterns like /./
        name.find("//") != std::string::npos)   // Catches double slashes
    {
        return false;
    }

    struct stat buffer;   
    return (stat(name.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

bool checkFolderExists(const std::string &folderName)
{
    struct stat info;
    return stat(folderName.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

bool createFolder(const std::string &folderName)
{
    if (!checkFolderExists(folderName))
    {
        // Create the folder using mkdir
        if (mkdir(folderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0)
        {
            std::cout << "Folder created successfully: " << folderName << std::endl;
            return true;
        }
        else
        {
            std::cerr << "Error creating folder: " << strerror(errno) << std::endl;
            return false;
        }
    }
    else
    {
        std::cout << "Folder already exists: " << folderName << std::endl;
    }

    return true;
}

bool createDirectory(const std::string &path)
{
    // Check if the directory already exists
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        // Directory does not exist, try to create it
        if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else if (info.st_mode & S_IFDIR)
    {
        // Directory already exists
        return true;
    }
    else
    {
        // Path exists but is not a directory
        return false;
    }
}

bool createDirectories(const std::string &path)
{
    std::vector<std::string> parts;
    std::istringstream       iss(path);
    std::string              part;

    // Split the path into parts
    while (std::getline(iss, part, '/'))
    {
        if (!part.empty())
        {
            parts.push_back(part);
        }
    }

    std::string currentPath = "";
    for (const std::string &part : parts)
    {
        currentPath += part + "/";
        if (!createDirectory(currentPath))
        {
            return false; // Error occurred while creating the directory
        }
    }

    return true; // All directories created successfully
}
//
//
//
void getDateTime(std::string &timeStr)
{
    // Get the current time point
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();

    // Convert the time point to a time_t object
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    // Convert the time_t object to a local time structure
    std::tm *now_tm = std::localtime(&now_time_t);

    // Extract date and time components
    int year   = now_tm->tm_year + 1900; // Years since 1900
    int month  = now_tm->tm_mon + 1;     // Months start from 0
    int day    = now_tm->tm_mday;        // Day of the month
    int hour   = now_tm->tm_hour;        // Hour of the day
    int minute = now_tm->tm_min;         // Minute of the hour

    timeStr = std::to_string(year) + "-" + std::to_string(month) + "-" + std::to_string(day) + "-"
              + std::to_string(hour) + "-" + std::to_string(minute);
}

//
//
//
bool sortbysec(const pair<int, int> &a, const pair<int, int> &b)
{
    return (a.second > b.second);
}

bool sortbysec_float(const pair<int, float> &a, const pair<int, float> &b)
{
    return (a.second > b.second);
}

bool sortBySecAscend(const pair<int, int> &a, const pair<int, int> &b)
{
    return (a.second < b.second);
}

// DEPRECATED: Function using BoundingBox removed
// bool sortByBBoxArea(BoundingBox &bboxA, BoundingBox &bboxB)
// {
//     return bboxA.getArea() > bboxB.getArea();
// }

// bool sortByObjBBoxArea(Object &objA, Object &objB)
// {
//     return objA.m_lastDetectBoundingBox.getArea() > objB.m_lastDetectBoundingBox.getArea();
// }

// DEPRECATED: Function using Point removed
// float calcVecDegree(Point &a, Point &b)
// {
//     float dot;
//     float det;
//     float angle;
//     float degree;
//
//     dot   = a.x * b.x + a.y * b.y; // dot product between [x1, y1] and [x2, y2]
//     det   = a.x * b.y - a.y * b.x; // determinant
//     angle = atan2(det, dot);
//
//     degree = abs(angle * 180 / M_PI);
//
//     return degree;
// }

float calcVecDegree(cv::Point &a, cv::Point &b)
{
    float dot;
    float det;
    float angle;
    float degree;

    dot   = a.x * b.x + a.y * b.y; // dot product between [x1, y1] and [x2, y2]
    det   = a.x * b.y - a.y * b.x; // determinant
    angle = atan2(det, dot);

    degree = abs(angle * 180 / M_PI);

    return degree;
}

float calcVecDegree(cv::Point a, cv::Point b)
{
    float dot;
    float det;
    float angle;
    float degree;

    dot   = a.x * b.x + a.y * b.y; // dot product between [x1, y1] and [x2, y2]
    det   = a.x * b.y - a.y * b.x; // determinant
    angle = atan2(det, dot);

    degree = abs(angle * 180 / M_PI);

    return degree;
}

void updateIntList(vector<int> &list, int x, int maxSize)
{
    if (list.size() < maxSize)
        list.push_back(x);
    else
    {
        list.erase(list.begin());
        list.push_back(x);
    }
}

//
//
//
void findMaxScoreKey(vector<pair<int, float>> &dict, int &key, float &score)
{
    int   idxMaxScore = 0;
    float maxScore    = 0.0;
    for (auto it = dict.begin(); it != dict.end(); it++)
    {
        int   keyId      = (*it).first;
        float matchScore = (*it).second;
        if (matchScore > maxScore)
        {
            maxScore    = matchScore;
            idxMaxScore = keyId;
        }
    }
    key   = idxMaxScore;
    score = maxScore;
}

//
//
//
int findListIdx(int key, vector<int> &dict)
{
    int idx = 0;
    for (int i = 0; i < dict.size(); i++)
    {
        if (key == dict[i])
        {
            idx = i;
            break;
        }
    }
    return idx;
}

int findListIdx(int key, vector<pair<int, vector<int>>> &dict)
{
    int idx = 0;
    for (int i = 0; i < dict.size(); i++)
    {
        if (key == dict[i].first)
        {
            idx = i;
            break;
        }
    }
    return idx;
}

int findListIdx(int key, vector<pair<int, vector<pair<int, float>>>> &dict)
{
    int idx = -1;

    for (int i = 0; i < static_cast<int>(dict.size()); i++)
    {
        if (key == dict[i].first)
        {
            idx = i;
            break;
        }
    }

    return idx;
}

int findListIdx(int key, vector<pair<int, pair<int, float>>> &dict)
{
    int idx = 0;
    for (int i = 0; i < dict.size(); i++)
    {
        if (key == dict[i].first)
        {
            idx = i;
            break;
        }
    }
    return idx;
}

// DEPRECATED: Function using Point removed
// int findListIdx(int key, vector<pair<int, vector<Point>>> &dict)
// {
//     int idx = 0;
//     for (int i = 0; i < dict.size(); i++)
//     {
//         if (key == dict[i].first)
//         {
//             idx = i;
//             break;
//         }
//     }
//     return idx;
// }

//
//
//
int findMedian(vector<int> a, int n)
{
    // If size of the arr[] is even
    if (n % 2 == 0)
    {
        // Applying nth_element
        // on n/2th index
        nth_element(a.begin(), a.begin() + n / 2, a.end());
        // Applying nth_element
        // on (n-1)/2 th index
        nth_element(a.begin(), a.begin() + (n - 1) / 2, a.end());
        // Find the average of value at
        // index N/2 and (N-1)/2
        return (int)(a[(n - 1) / 2] + a[n / 2]) / 2.0;
    }
    // If size of the arr[] is odd
    else
    {
        // Applying nth_element
        // on n/2
        nth_element(a.begin(), a.begin() + n / 2, a.end());
        // Value at index (N/2)th
        // is the median
        return (int)a[n / 2];
    }
}

int findMedian(std::vector<unsigned int> a, int n)
{
    // If size of the arr[] is even
    if (n % 2 == 0)
    {
        // Applying nth_element
        // on n/2th index
        nth_element(a.begin(), a.begin() + n / 2, a.end());
        // Applying nth_element
        // on (n-1)/2 th index
        nth_element(a.begin(), a.begin() + (n - 1) / 2, a.end());
        // Find the average of value at
        // index N/2 and (N-1)/2
        return (int)(a[(n - 1) / 2] + a[n / 2]) / 2.0;
    }
    // If size of the arr[] is odd
    else
    {
        // Applying nth_element
        // on n/2
        nth_element(a.begin(), a.begin() + n / 2, a.end());
        // Value at index (N/2)th
        // is the median
        return (int)a[n / 2];
    }
}

bool isKeyInDict(int key, vector<int> &dict)
{
    bool inDict = false;
    for (auto it = dict.begin(); it != dict.end(); it++)
    {
        if (key == (*it))
        {
            inDict = true;
            break;
        }
    }
    return inDict;
}

bool isKeyInDict(int key, vector<pair<int, vector<int>>> &dict)
{
    bool inDict = false;
    for (auto it = dict.begin(); it != dict.end(); it++)
    {
        if (key == (*it).first)
        {
            inDict = true;
            break;
        }
    }
    return inDict;
}

bool isKeyInDict(int key, vector<pair<int, vector<pair<int, float>>>> &dict)
{
    bool inDict = false;
    for (auto it = dict.begin(); it != dict.end(); it++)
    {
        if (key == (*it).first)
        {
            inDict = true;
            break;
        }
    }
    return inDict;
}

bool isKeyInDict(int key, vector<pair<int, pair<int, float>>> &dict)
{
    bool inDict = false;
    for (auto it = dict.begin(); it != dict.end(); it++)
    {
        if (key == (*it).first)
        {
            inDict = true;
            break;
        }
    }
    return inDict;
}

//
//
//
float getWidthRatio(int refWidth, int width)
{
    return (float)refWidth / (float)width;
}

float getWidthRatio(int xA, int xB, int width)
{
    int   dist       = abs(xA - xB);
    float widthRatio = (float)dist / (float)width;

    return widthRatio;
}

float getWidthRatio(int xA, int xB, int refWidth, int width)
{
    int   dist       = abs(xA - xB);
    float diff       = abs(refWidth - dist);
    float widthRatio = (float)diff / (float)width;

    return widthRatio;
}

//
//
//
std::pair<float, float> computeStdDevAndMean(std::vector<int> &v)
{
    double sum  = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev  = std::sqrt(sq_sum / v.size() - mean * mean);
    return std::make_pair(mean, stdev);
}

std::pair<double, double> computeStdDevAndMean(std::vector<double> &v)
{
    double sum  = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev  = std::sqrt(sq_sum / v.size() - mean * mean);
    return std::make_pair(mean, stdev);
}

//
//
//
vector<int> outlierRemoval(vector<int> &list)
{
    std::pair<float, float> meanAndStdDev = computeStdDevAndMean(list);
    int   median     = findMedian(list, list.size());
    float upperBound = (meanAndStdDev.first + 2.3 * meanAndStdDev.second);
    float lowerBound = (meanAndStdDev.first - 2.3 * meanAndStdDev.second);

    int mean = (int)meanAndStdDev.first;
    // cout << "upperBound = " << upperBound << endl;
    // cout << "lowerBound = " << lowerBound << endl;
    // cout << "mean = " << mean << endl;
    // cout << "median = " << median << endl;

    vector<int> filteredList;
    for (int i = 0; i < list.size(); i++)
    {
        if (list[i] < upperBound && list[i] > lowerBound)
            filteredList.push_back(list[i]);
        else
            filteredList.push_back(mean);
    }

    return filteredList;
}

//
//
//
void applyMovingAverage(vector<int> &list, vector<int> &listMA, int windowSize)
{
    float       s = 0;
    vector<int> cumsumList;
    vector<int> tmpCumsumList;
    int         listLen = list.size();

    // SMA- Simple Moving Average, the following is an efficient O(3n) algorithm,
    //.better than Brute Force algorithm (O(n^2))

    // Step1: calculate culmulate sum
    for (auto it = list.begin(); it != list.end(); it++)
    {
        s += (*it);
        tmpCumsumList.push_back(s);
    }

    // Step2: padding and subtracting
    for (int i = 0; i < windowSize; i++)
        cumsumList.push_back(tmpCumsumList[i]);
    for (int i = 0; i < listLen - windowSize; i++)
        cumsumList.push_back(tmpCumsumList[i + windowSize] - tmpCumsumList[i]);

    // Step3: calculate average
    for (int i = 0; i < listLen - windowSize + 1; i++)
        listMA.push_back(static_cast<int>(cumsumList[i + windowSize - 1] / windowSize));
}

    // Helper function to calculate moving average
float calculateMovingAverage(std::vector<float>& distances, size_t currentIdx, int windowSize) 
{
    int start = std::max(0, static_cast<int>(currentIdx) - windowSize + 1);
    int count = currentIdx - start + 1;
    float sum = 0.0;
    
    for (int i = start; i <= currentIdx; ++i) {
        sum += distances[i];
    }
    
    return sum / count;
}

//
//
//
std::string to_string_with_precision(float a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return std::move(out).str();
}

// DEPRECATED: Function using BoundingBox removed
// void rescaleBBox(BoundingBox &bbox, BoundingBox &rescaleBBox, int modelWidth, int modelHeight, int videoWidth,
//                  int videoHeight)
// {
//     float w_ratio = static_cast<float>(modelWidth) / static_cast<float>(videoWidth);
//     float h_ratio = static_cast<float>(modelHeight) / static_cast<float>(videoHeight);
//
//     rescaleBBox.x1 = static_cast<int>(static_cast<float>(bbox.x1) / w_ratio);
//     rescaleBBox.y1 = static_cast<int>(static_cast<float>(bbox.y1) / h_ratio);
//     rescaleBBox.x2 = static_cast<int>(static_cast<float>(bbox.x2) / w_ratio);
//     rescaleBBox.y2 = static_cast<int>(static_cast<float>(bbox.y2) / h_ratio);
//
//     rescaleBBox.confidence = bbox.confidence;
//     rescaleBBox.label      = bbox.label;
//      // Rescale pose detection keypoints, Alister add 2025-01-12
//     // cout<<"bbox.pose_kpts.size() = "<<bbox.pose_kpts.size()<<endl;
//     for(int i=0; i<bbox.pose_kpts.size();i++)
//     {
//         int pose_kpt_x = (int)((float)bbox.pose_kpts[i].first/ w_ratio);
//         int pose_kpt_y = (int)((float)bbox.pose_kpts[i].second/ h_ratio);
//         rescaleBBox.pose_kpts.push_back({pose_kpt_x, pose_kpt_y});
//     }
//     rescaleBBox.skeletonAction = bbox.skeletonAction;
//     // cout<<"Hi Alister !! rescaleBBox.pose_kpts.size() = "<<rescaleBBox.pose_kpts.size()<<endl;
//
//
//     if (rescaleBBox.x1 < 0)
//         rescaleBBox.x1 = 0;
//     if (rescaleBBox.x2 > videoWidth - 1)
//         rescaleBBox.x2 = videoWidth - 1;
//     if (rescaleBBox.y1 < 0)
//         rescaleBBox.y1 = 0;
//     if (rescaleBBox.y2 > videoHeight - 1)
//         rescaleBBox.y2 = videoHeight - 1;
// }

// =================================================================================================
// DEPRECATED: YOLOv8 function - removed as YOLOv8 code has been removed
// =================================================================================================
/*
void rescalev8xyxy(v8xyxy &bbox, BoundingBox &rescaleBBox, int modelWidth, int modelHeight, int videoWidth,
                   int videoHeight)
{
    float w_ratio = (float)modelWidth / (float)videoWidth;
    float h_ratio = (float)modelHeight / (float)videoHeight;

    rescaleBBox.x1         = (int)((float)bbox.x1 / w_ratio);
    rescaleBBox.y1         = (int)((float)bbox.y1 / h_ratio);
    rescaleBBox.x2         = (int)((float)bbox.x2 / w_ratio);
    rescaleBBox.y2         = (int)((float)bbox.y2 / h_ratio);
    rescaleBBox.confidence = bbox.c_prob;
    rescaleBBox.label      = bbox.c;

    if (rescaleBBox.x1 < 0)
        rescaleBBox.x1 = 0;
    if (rescaleBBox.x2 > videoWidth - 1)
        rescaleBBox.x2 = videoWidth - 1;
    if (rescaleBBox.y1 < 0)
        rescaleBBox.y1 = 0;
    if (rescaleBBox.y2 > videoHeight - 1)
        rescaleBBox.y2 = videoHeight - 1;
}
*/

void rescaleROI(ROI &srcROI, ROI &dstROI, int modelWidth, int modelHeight, int videoWidth, int videoHeight)
{
    float w_ratio = (float)modelWidth / (float)videoWidth;
    float h_ratio = (float)modelHeight / (float)videoHeight;

    dstROI.x1 = srcROI.x1 / w_ratio;
    dstROI.y1 = srcROI.y1 / h_ratio;

    dstROI.x2 = srcROI.x2 / w_ratio;
    dstROI.y2 = srcROI.y2 / h_ratio;
}


void rescalePoint(cv::Point &pSrc, cv::Point &pDst, float xRatio, float yRatio)
{
    pDst.x = (int)((float)pSrc.x / xRatio);
    pDst.y = (int)((float)pSrc.y / yRatio);
}

void rescalePoint(cv::Point &pSrc, cv::Point &pDst, int modelWidth, int modelHeight, int videoWidth, int videoHeight)
{
    float xRatio = (float)modelWidth / (float)videoWidth;
    float yRatio = (float)modelHeight / (float)videoHeight;
    pDst.x       = (int)((float)pSrc.x / xRatio);
    pDst.y       = (int)((float)pSrc.y / yRatio);
}

// DEPRECATED: Function using BoundingBox and Point removed
// void smoothBBoxes(vector<BoundingBox> &bboxList, vector<BoundingBox> &procBboxList, int windowSize,
//                   int maxFrameInterval)
// {
//     int numBbox        = bboxList.size();
//     int halfWindowSize = (int)(windowSize * 0.5);
//
//     for (int i = halfWindowSize; i < numBbox - halfWindowSize; i += 1)
//     {
//         int newX1 = 0;
//         int newY1 = 0;
//         int newX2 = 0;
//         int newY2 = 0;
//
//         vector<BoundingBox> tmpBboxList;
//         vector<int>         frameStampList;
//         for (int j = i - halfWindowSize; j <= i + halfWindowSize; j++)
//         {
//             tmpBboxList.push_back(bboxList[j]);
//             frameStampList.push_back(bboxList[j].frameStamp);
//         }
//
//         vector<int> frameIntervalList;
//         frameIntervalList.push_back(0); // 1st frame interval = 0
//         for (int i = 1; i < windowSize; i++)
//         {
//             int interval = frameStampList[i] - frameStampList[i - 1];
//             frameIntervalList.push_back(interval);
//         }
//
//         vector<vector<Point>> cornerPointList;
//         for (int i = 0; i < (int)tmpBboxList.size(); i++)
//         {
//             cornerPointList.push_back(tmpBboxList[i].getCornerPoints());
//         }
//
//         // 0:TL, 1:TR, 2:BL, 3:BR
//         vector<int> x1List;
//         vector<int> y1List;
//         vector<int> x2List;
//         vector<int> y2List;
//         for (int i = 0; i < (int)cornerPointList.size(); i++)
//         {
//             int frameInterval = frameIntervalList[i];
//             if (frameInterval > maxFrameInterval)
//             {
//                 break; // ignore remain bounding boxes ...
//             }
//             x1List.push_back(cornerPointList[i][0].x);
//             y1List.push_back(cornerPointList[i][0].y);
//             x2List.push_back(cornerPointList[i][3].x);
//             y2List.push_back(cornerPointList[i][3].y);
//         }
//
//         vector<int>::iterator x1It_min;
//         vector<int>::iterator x1It_max;
//
//         vector<int>::iterator x2It_min;
//         vector<int>::iterator x2It_max;
//
//         vector<int>::iterator y1It;
//         vector<int>::iterator y2It;
//
//         x1It_min = std::min_element(x1List.begin(), x1List.end());
//         x1It_max = std::max_element(x1List.begin(), x1List.end());
//
//         x2It_min = std::min_element(x2List.begin(), x2List.end());
//         x2It_max = std::max_element(x2List.begin(), x2List.end());
//
//         y1It = std::min_element(y1List.begin(), y1List.end());
//         y2It = std::min_element(y2List.begin(), y2List.end());
//
//         // newX1 = int(((*x1It_min) + (*x1It_max)) * 0.5);
//         // newX2 = int(((*x2It_min) + (*x2It_max)) * 0.5);
//         // // newX2 = (*x2It_min);
//
//         // newY1 = (*y1It);
//         // // newX2 = (*x2It);
//         // // newY2 = (*y2It);
//         newX1 = utils::findMedian(x1List, x1List.size());
//         newX2 = utils::findMedian(x2List, x2List.size());
//         newY1 = utils::findMedian(y1List, y1List.size());
//         newY2 = utils::findMedian(y2List, y2List.size());
//
//         BoundingBox bbox = BoundingBox(newX1, newY1, newX2, newY2, bboxList[0].label);
//         procBboxList.push_back(bbox);
//     }
// }

//
//
//
bool isAscendingList(vector<float> &list)
{
    float counter = 0;
    for (int i = 1; i < list.size(); i++)
    {
        if (list[i] > list[i - 1])
            counter += 1;
    }

    // cout << "isAscendingList counter = " << counter << endl;
    if ((counter / (float)list.size()) > 0.5)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool isDescendingList(std::vector<float> &list)
{
    float counter = 0;
    for (int i = 1; i < list.size(); i++)
    {
        if (list[i] < list[i - 1])
            counter += 1;
    }
    // cout << "isDescendingList counter = " << counter << endl;
    if ((counter / (float)list.size()) > 0.5)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool isZeroPoint(cv::Point &p)
{
    if (p.x == 0 && p.y == 0)
        return true;
    else
        return false;
}

cv::Point2f calculateRhoTheta(cv::Point point1, cv::Point point2)
{
    double rho, theta;

    // Calculate rho
    double numerator   = point1.x * point2.y - point2.x * point1.y;
    double denominator = std::sqrt(std::pow(point2.x - point1.x, 2) + std::pow(point2.y - point1.y, 2));
    rho                = abs(numerator / denominator);

    // Calculate theta
    if (point2.x == point1.x)
    {
        theta = (point2.y > point1.y) ? CV_PI / 2 : 3 * CV_PI / 2;
    }
    else
    {
        theta = std::atan2(point2.y - point1.y, point2.x - point1.x);
    }

    if (theta < 0)
    {
        theta += CV_PI / 2; // Adjust to [0, 2π]
    }
    return cv::Point2f(rho, theta);
}

bool checkVectorDifference(const std::vector<cv::Point> &pointVector, float threshold, float maxPercentage)
{
    if (pointVector.empty())
        return true;

    // Calculate mean x and y
    cv::Point sum(0, 0);
    for (const auto &point : pointVector)
    {
        sum += point;
    }
    cv::Point mean = sum * (1.0 / pointVector.size());

    // Count points exceeding threshold
    int countExceeding = 0;
    for (const auto &point : pointVector)
    {
        float distance = cv::norm(point - mean);
        if (distance > threshold)
        {
            countExceeding++;
        }
    }

    // Calculate percentage
    float percentage = (float)countExceeding / pointVector.size() * 100.0f;

    return percentage <= maxPercentage;
}

bool checkFloatVectorDifference(const std::vector<float> &floatVector, float threshold, float maxPercentage)
{
    if (floatVector.empty())
        return true;

    // Calculate mean
    float sum  = std::accumulate(floatVector.begin(), floatVector.end(), 0.0f);
    float mean = sum / floatVector.size();

    // Count values exceeding threshold
    int countExceeding = 0;
    for (const auto &value : floatVector)
    {
        if (std::abs(value - mean) > threshold)
        {
            countExceeding++;
        }
    }

    // Calculate percentage
    float percentage = (float)countExceeding / floatVector.size() * 100.0f;

    return percentage <= maxPercentage;
}

cv::Point calculateMeanPoint(const std::vector<cv::Point> &points)
{
    cv::Point sum(0, 0);
    for (const auto &point : points)
    {
        sum += point;
    }
    return sum * (1.0 / points.size());
}

float calculateMeanFloat(const std::vector<float> &values)
{
    return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
}

std::pair<std::string, float> extractDeviceInfoFromFilename(const std::string &filename)
{
    std::regex pattern1("REC\\d{8}-\\d{6}_([A-Z0-9]+)_CameraHeight=(\\d+\\.\\d+)(m?)");
    std::regex pattern2("(\\d+)_CameraHeight=(\\d+\\.\\d+)(m?)");
    std::regex pattern3("REC\\d{8}-\\d{6}_([A-Z0-9]+)_CameraHeight=(\\d+\\.\\d+)");
    std::regex pattern4("REC-\\d+_([A-Z0-9]+)_CameraHeight=(\\d+\\.\\d+)");
    std::regex pattern5("REC-[A-Za-z0-9]+_([A-Z0-9]+)_CameraHeight=(\\d+\\.\\d+)");
    std::regex pattern6("REC-Test\\d+-\\d+_([A-Z0-9]+)_CameraHeight=(\\d+\\.\\d+)");

    std::smatch matches;

    if (std::regex_search(filename, matches, pattern1) || std::regex_search(filename, matches, pattern3))
    {
        std::string suffix = matches[2].str() + matches[3].str();
        return {matches[1].str() + "_CameraHeight=" + suffix, std::stof(matches[2].str())};
    }
    else if (std::regex_search(filename, matches, pattern2))
    {
        std::string suffix = matches[2].str() + matches[3].str();
        return {matches[1].str() + "_CameraHeight=" + suffix, std::stof(matches[2].str())};
    }
    else if (std::regex_search(filename, matches, pattern4))
    {
        return {matches[1].str() + "_CameraHeight=" + matches[2].str(), std::stof(matches[2].str())};
    }
    else if (std::regex_search(filename, matches, pattern5))
    {
        return {matches[1].str() + "_CameraHeight=" + matches[2].str(), std::stof(matches[2].str())};
    }
    else if (std::regex_search(filename, matches, pattern6))
    {
        return {matches[1].str() + "_CameraHeight=" + matches[2].str(), std::stof(matches[2].str())};
    }
    
    return {"", 0.0f};
}

// DEPRECATED: Function using BoundingBox removed
// bool isLineIntersectingBox(const cv::Point& p1, const cv::Point& p2, const BoundingBox& box) {
//     // Convert box to line segments
//     cv::Point topLeft(box.x1, box.y1);
//     cv::Point topRight(box.x2, box.y1);
//     cv::Point bottomLeft(box.x1, box.y2);
//     cv::Point bottomRight(box.x2, box.y2);
//
//     // Check intersection with all 4 sides of the box
//     return (utils::doLinesIntersect(p1, p2, topLeft, topRight) ||     // Top
//             utils::doLinesIntersect(p1, p2, topRight, bottomRight) || // Right
//             utils::doLinesIntersect(p1, p2, bottomLeft, bottomRight) ||// Bottom
//             utils::doLinesIntersect(p1, p2, topLeft, bottomLeft));     // Left
// }

bool doLinesIntersect(const cv::Point& p1, const cv::Point& p2, 
                        const cv::Point& p3, const cv::Point& p4) {
    int d1 = direction(p3, p4, p1);
    int d2 = direction(p3, p4, p2);
    int d3 = direction(p1, p2, p3);
    int d4 = direction(p1, p2, p4);

    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)))
        return true;

    return false;
}

int direction(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3) {
    return ((p3.x - p1.x) * (p2.y - p1.y)) - ((p2.x - p1.x) * (p3.y - p1.y));
}

double _calculateDistance(const cv::Point& p1, const cv::Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

}
