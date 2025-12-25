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

#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "yolov8.hpp"
#include "yolov8_decoder.hpp"

using namespace cv;
using namespace std;

// static means this function is only visible in this file
static bool xyxyobj_compare(const v8xyxy &a, const v8xyxy &b)
{
    return a.c_prob > b.c_prob; // decreasing order
}

YOLOv8_Decoder::YOLOv8_Decoder(int inputH, int inputW, std::string loggerStr)
{
    m_loggerStr = loggerStr;

#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::syslog_logger_mt(m_loggerStr, "adas-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
    auto logger = spdlog::stdout_color_mt(m_loggerStr);
    logger->set_pattern("[%n] [%^%l%$] %v");
#endif

    m_inputH = inputH;
    m_inputW = inputW;
};

YOLOv8_Decoder::~YOLOv8_Decoder()
{
    spdlog::drop(m_loggerStr);
};

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
    // vector of vectors requres initialization
    if (classwisePicked.empty())
        classwisePicked = std::vector<std::vector<v8xyxy>>(num_Cls);

    // classwise bbox list before nms, vector<v8xyxy> bboxlist[i] for class i
    vector<vector<v8xyxy>> bboxlist(num_Cls);

    // YOLOv8 Decode
    getCandidates(
        m_detection_box_buff,
        m_detection_conf_buff,
        m_detection_class_buff,
        numBbox,
        confThreshold,
        bboxlist);

    // sending reference to doNMS
    doNMS(bboxlist, iouThreshold, classwisePicked, num_Cls);

    // handle the case when the total picked bbx number is larger than or equal to maximum allowable BBX Num
    // TODO
    // should be classwise and remove boxes with lower conf in each class

    int size = 0;
    for (int cls = 0; cls < num_Cls; cls++)
        size += classwisePicked[cls].size();

    return size;
}

unsigned int YOLOv8_Decoder::decodeBoxAndKpt(
    const float *m_detection_box_buff, 
    const float *m_detection_conf_buff,
    const float *m_detection_class_buff, 
    const float *m_keypoint_buff,
    int numBbox, 
    float confThreshold,
    float iouThreshold, 
    int num_Cls,
    std::vector<std::vector<v8xyxy>> &classwisePicked)
{
    auto logger = spdlog::get(m_loggerStr);
    // vector of vectors requres initialization
    if (classwisePicked.empty())
        classwisePicked = std::vector<std::vector<v8xyxy>>(num_Cls);

    // classwise bbox list before nms, vector<v8xyxy> bboxlist[i] for class i
    vector<vector<v8xyxy>> bboxlist(num_Cls);

    // YOLOv8 Decode
    // First stage: Confidence threshold filtering  
    getCandidatesWithKpt(
        m_detection_box_buff, 
        m_detection_conf_buff, 
        m_detection_class_buff, 
        m_keypoint_buff, 
        numBbox, 
        confThreshold, 
        bboxlist);

    // Second stage: Apply NMS (Non-Maximum Suppression)
    doNMS(bboxlist, iouThreshold, classwisePicked, num_Cls);

    // handle the case when the total picked bbx number is larger than or equal to maximum allowable BBX Num
    // TODO
    // should be classwise and remove boxes with lower conf in each class

    int size = 0;
    for (int cls = 0; cls < num_Cls; cls++)
        size += classwisePicked[cls].size();

    return size;
}

int YOLOv8_Decoder::getCandidates(const float *detectionBox, const float *detectionConf, const float *detectionClass,
                                    int numBbox, const float conf_thr, vector<vector<v8xyxy>> &bbox_list)
{
    auto logger            = spdlog::get(m_loggerStr);
    int  numThresholdedBbx = 0;

    if (numBbox <= 0)
    {
        logger->error("Invalid numBbox: {}", numBbox);
        return 0;
    }


    // NOTE: detection output is in raw shape due to failed compability with SNPE converter:
    // the bbox is serialized to first 4 * 5040 elements of (detection) and the class scores
    // is serialized to the remaining 10*5040 elements of (detection)

    for (int i = 0; i < numBbox; i++) // each box
    {
        if (detectionConf[i] > conf_thr) // If class score higher than theshold
        {
            v8xyxy box;
            // cast float dimensions to int
            box.x1     = static_cast<int>(detectionBox[i]);
            box.y1     = static_cast<int>(detectionBox[numBbox + i]);
            box.x2     = static_cast<int>(detectionBox[numBbox * 2 + i]);
            box.y2     = static_cast<int>(detectionBox[numBbox * 3 + i]);
            box.c      = detectionClass[i];
            box.c_prob = detectionConf[i];
            // calculate area only once
            box.area = (box.x2 - box.x1) * (box.y2 - box.y1);

            // in order to speed up, push box according to its class into classwise vector<v8xyxy> bbox_list[box.c]
            if (box.c >= 0 && box.c < bbox_list.size())
                bbox_list[box.c].push_back(box);
            else
                logger->error("Invalid class index: {}", box.c);
            numThresholdedBbx++;
        }
    }

    return numThresholdedBbx;
}


int YOLOv8_Decoder::getCandidatesWithKpt(
    const float *detectionBox,
    const float *detectionConf,
    const float *detectionClass,
    const float *detectionKeypoints,
    int numBbox,
    const float conf_thr,
    vector<vector<v8xyxy>> &bbox_list)
{
    auto logger = spdlog::get(m_loggerStr);
    int numThresholdedBbx = 0;

    if (numBbox <= 0)
    {
        logger->error("Invalid numBbox: {}", numBbox);
        return 0;
    }

    const int m_inputW = 512;
    const int m_inputH = 288;

    const float minRatio = 0.25f;
    const float maxRatio = 4.00f;
    const float maxArea = 0.60f;
    const float imgArea = static_cast<float>(m_inputW) * static_cast<float>(m_inputH);

    const int grid_8 = (m_inputH / 8) * (m_inputW / 8); // stride=8 grid
    const int grid_16 = (m_inputH / 16) * (m_inputW / 16); // stride=16 grid
    const int grid_32 = (m_inputH / 32) * (m_inputW / 32); // stride=32 grid

    const int range_8_end = grid_8 - 1;                          // 0 .. range_8_end
    const int range_16_end = grid_8 + grid_16 - 1;               // grid_8 .. range_16_end
    const int range_32_end = grid_8 + grid_16 + grid_32 - 1;     // ...

    // --- Safety check for keypoint buffer size if provided ---
    size_t expected_kpt_elems = static_cast<size_t>(numBbox) * static_cast<size_t>(m_numOfKeypoints) * 3u;
    if (detectionKeypoints != nullptr) {
        // If you know the input buffer length elsewhere, compare it here. At least check for overflow.
        if (expected_kpt_elems == 0) {
            logger->error("Expected keypoint elements == 0 (numBbox: {}, m_numOfKeypoints: {})", numBbox, m_numOfKeypoints);
            // continue but avoid indexing
        }
    }

    // Debug: log once
    logger->debug("YOLOv8 Grid Info - Input: {}x{}, numBbox: {}", m_inputW, m_inputH, numBbox);
    logger->debug("Grid sizes - 8x: {}, 16x: {}, 32x: {}, Total: {}", grid_8, grid_16, grid_32);
    logger->debug("Stride ranges - 8: [0-{}], 16: [{}-{}], 32: [{}-{}]", range_8_end, grid_8, range_16_end, grid_8 + grid_16, range_32_end);

    // Calculate last valid flattened index for detectionKeypoints
    // valid indices are 0 .. (expected_kpt_elems - 1)
    size_t max_index = (expected_kpt_elems == 0) ? 0 : (expected_kpt_elems - 1);

    for (int i = 0; i < numBbox; ++i)
    {
        if (detectionConf[i] < conf_thr) continue; // skip low-confidence

        v8xyxy box;
        // read bbox: detectionBox layout assumed: [x0..xn-1, y0..yn-1, x2.., y2..]
        box.x1     = static_cast<int>(detectionBox[i]);
        box.y1     = static_cast<int>(detectionBox[numBbox + i]);
        box.x2     = static_cast<int>(detectionBox[numBbox * 2 + i]);
        box.y2     = static_cast<int>(detectionBox[numBbox * 3 + i]);
        box.c      = static_cast<int>(detectionClass[i]);
        box.c_prob = detectionConf[i];

        float width  = static_cast<float>(box.x2 - box.x1);
        float height = static_cast<float>(box.y2 - box.y1);
        float area   = width * height;
        float ratio  = (height != 0.0f) ? (width / height) : 1000.0f; // avoid div0

        if (ratio < minRatio || ratio > maxRatio || area > maxArea * imgArea) continue;

        if (detectionKeypoints != nullptr && expected_kpt_elems > 0)
        {
            float kpt_stride = 8.0f;
            for (int k = 0; k < m_numOfKeypoints; ++k)
            {
                // flattened indexing: (numBbox * 3 * k) + (numBbox * coord) + i
                // coord: 0->x, 1->y, 2->c
                size_t base = static_cast<size_t>(numBbox) * 3u * static_cast<size_t>(k);
                size_t kx_index = base + static_cast<size_t>(0u * numBbox) + static_cast<size_t>(i);
                size_t ky_index = base + static_cast<size_t>(1u * numBbox) + static_cast<size_t>(i);
                size_t kc_index = base + static_cast<size_t>(2u * numBbox) + static_cast<size_t>(i);

                // check indices are within expected buffer
                if (kx_index > max_index || ky_index > max_index || kc_index > max_index) {
                    logger->error("Keypoint index out of bounds (i={}, k={}, kx_index={}, ky_index={}, kc_index={}, max_index={})",
                                  i, k, kx_index, ky_index, kc_index, max_index);
                    // skip this keypoint safely
                    continue;
                }

                if (i <= range_8_end) {
                    kpt_stride = 8.0f;
                } else if (i <= range_16_end) {
                    kpt_stride = 16.0f;
                } else { // i <= range_32_end
                    kpt_stride = 32.0f;
                }

                float raw_kx = detectionKeypoints[kx_index];
                float raw_ky = detectionKeypoints[ky_index];
                float kc     = detectionKeypoints[kc_index];

                if (!std::isfinite(raw_kx) || !std::isfinite(raw_ky)) {
                    logger->warn("Non-finite keypoint coordinate (i={}, k={}, raw_kx={}, raw_ky={})", i, k, raw_kx, raw_ky);
                    continue;
                }

                if (kc < conf_thr) continue;

                int kx = static_cast<int>(raw_kx * kpt_stride);
                int ky = static_cast<int>(raw_ky * kpt_stride);

                // clamp to image bounds
                kx = std::max(0, std::min(kx, m_inputW - 1));
                ky = std::max(0, std::min(ky, m_inputH - 1));

                box.pose_kpts.push_back({kx, ky});
            }
        }

        // class index check before push
        if (box.c >= 0 && box.c < static_cast<int>(bbox_list.size()))
            bbox_list[box.c].push_back(box);
        else
            logger->error("Invalid class index: {}", box.c);

        ++numThresholdedBbx;
    }

    return numThresholdedBbx;
}

// int YOLOv8_Decoder::getCandidatesWithKpt(
//     const float *detectionBox, 
//     const float *detectionConf, 
//     const float *detectionClass,
//     const float *detectionKeypoints,
//     int numBbox, 
//     const float conf_thr, 
//     vector<vector<v8xyxy>> &bbox_list)
// {
//     auto logger            = spdlog::get(m_loggerStr);
//     int numThresholdedBbx = 0;
//     int max_index = (numBbox * m_numOfKeypoints * 3) + numBbox - 1;

//     const int m_inputW = 512;
//     const int m_inputH = 288;

//     const float minRatio = 0.25f;
//     const float maxRatio = 4.00f;
//     const float maxArea = 0.60f;
//     const float imgArea = m_inputW * m_inputH;

//     const int grid_8 = (m_inputH / 8) * (m_inputW / 8); // stride=8 grid
//     const int grid_16 = (m_inputH / 16) * (m_inputW / 16); // stride=16 grid  
//     const int grid_32 = (m_inputH / 32) * (m_inputW / 32); // stride=32 grid
//     const int total_calculated = grid_8 + grid_16 + grid_32;

//     // Anchor ranges (cumulative) - calculated once
//     const int range_8_end = grid_8 - 1; // 0 to (grid_8-1)
//     const int range_16_end = grid_8 + grid_16 - 1; // grid_8 to (grid_8+grid_16-1)
//     const int range_32_end = grid_8 + grid_16 + grid_32 - 1; // (grid_8+grid_16) to (grid_8+grid_16+grid_32-1)

//     float kpt_stride = 8.0f; // default stride

//     // Debug: log grid information once
//     logger->debug("YOLOv8 Grid Info - Input: {}x{}, numBbox: {}", m_inputW, m_inputH, numBbox);
//     logger->debug("Grid sizes - 8x: {}, 16x: {}, 32x: {}, Total: {}", grid_8, grid_16, grid_32, total_calculated);
//     logger->debug("Stride ranges - 8: [0-{}], 16: [{}-{}], 32: [{}-{}]", range_8_end, grid_8, range_16_end,
//                   grid_8 + grid_16, range_32_end);
    
//     if (numBbox <= 0)
//     {
//         logger->error("Invalid numBbox: {}", numBbox);
//         return 0;
//     }

//     // NOTE: detection output is in raw shape due to failed compability with SNPE converter:
//     // the bbox is serialized to first 4 * 5040 elements of (detection) and the class scores
//     // is serialized to the remaining 10*5040 elements of (detection)

//     for (int i = 0; i < numBbox; i++) // each box
//     {
//         if (detectionConf[i] < conf_thr)continue; // If class score higher than theshold
        
//         v8xyxy box;
//         // cast float dimensions to int
//         box.x1     = static_cast<int>(detectionBox[i]);
//         box.y1     = static_cast<int>(detectionBox[numBbox + i]);
//         box.x2     = static_cast<int>(detectionBox[numBbox * 2 + i]);
//         box.y2     = static_cast<int>(detectionBox[numBbox * 3 + i]);
//         box.c      = detectionClass[i];
//         box.c_prob = detectionConf[i];
//         // calculate area only once
//         box.area = (box.x2 - box.x1) * (box.y2 - box.y1);

//         float width  = box.x2 - box.x1;
//         float height = box.y2 - box.y1;
//         float area   = width * height;
//         float ratio  = width / height;

//         if(ratio<minRatio || ratio>maxRatio || area > maxArea*imgArea)continue;

//         if(detectionKeypoints!=nullptr) // (51,3780) with m_numOfKeypoints=17 kpts, numBbox=3780
//         {
//             for (int k = 0; k < m_numOfKeypoints; k++)
//             {   
//                 // Pose kpts Format : x1x1x1x1x1..x1(numBbox) y1y1y1y1y1..y1(numBbox) c1c1c1c1c1..c1(numBbox)
//                 //                    x2x2x2x2x2..x2(numBbox) y2y2y2y2y2..y2(numBbox) c2c2c2c2c2..c2(numBbox)
//                 //                    x3x3x3x3x3..x3(numBbox) y3y3y3y3y3..y3(numBbox) c3c3c3c3c3..c3(numBbox)
//                 //                                          ...
//                 //                    xkxkxkxkxk..xk(numBbox) ykykykykyk..yk(numBbox) ckckckckck..ck(numBbox)
//                 int kx_index = (numBbox * 3 * k) + (numBbox * 0) + i;
//                 int ky_index = (numBbox * 3 * k) + (numBbox * 1) + i;
//                 int kc_index = (numBbox * 3 * k) + (numBbox * 2) + i;
            
//                 if (kx_index > max_index || ky_index > max_index)
//                 {
//                     logger->error("Keypoint index out of bounds");
//                     continue;
//                 }
//                 if(i<=range_8_end){
//                     kpt_stride = 8.0f;
//                 }else if(i<= range_16_end){
//                     kpt_stride = 16.0f;
//                 }else if(i<= range_32_end){
//                     kpt_stride = 32.0f;
//                 }
            
//                 int   kx = static_cast<int>(detectionKeypoints[kx_index] * kpt_stride);  // Get x-coordinate
//                 int   ky = static_cast<int>(detectionKeypoints[ky_index] * kpt_stride);  // Get y-coordinate
//                 float kc = static_cast<float>(detectionKeypoints[kc_index]);  // Get kpt confidence
            
//                 if (kc < conf_thr)
//                 {
//                     // logger->info("Keypoint confidence < conf_thr, skip this keypoint ...");
//                     continue;
//                 }

//                 // cout << k << ": kx:" << kx << " ky:" << ky << endl;
//                 // kx = std::max(0, std::min(kx, m_inputW - 1));
//                 // ky = std::max(0, std::min(ky, m_inputH - 1));
//                 box.pose_kpts.push_back({kx,ky});
//             }
//         }

//         // in order to speed up, push box according to its class into classwise vector<v8xyxy> bbox_list[box.c]
//         if (box.c >= 0 && box.c < bbox_list.size())
//             bbox_list[box.c].push_back(box);
//         else
//             logger->error("Invalid class index: {}", box.c);
//         numThresholdedBbx++;
        
//     }
// }

int YOLOv8_Decoder::getCandidatesWithKpt_v2(
    const float *detectionBox, 
    const float *detectionConf, 
    const float *detectionClass,
    const float *detectionKeypoints,
    int numBbox, 
    const float conf_thr, 
    vector<vector<v8xyxy>> &bbox_list)
{
    auto logger            = spdlog::get(m_loggerStr);
    int numThresholdedBbx = 0;
    int max_index = (numBbox * m_numOfKeypoints * 3) + numBbox - 1;

    if (numBbox <= 0)
    {
        logger->error("Invalid numBbox: {}", numBbox);
        return 0;
    }


    // NOTE: detection output is in raw shape due to failed compability with SNPE converter:
    // the bbox is serialized to first 4 * 5040 elements of (detection) and the class scores
    // is serialized to the remaining 10*5040 elements of (detection)

    for (int i = 0; i < numBbox; i++) // each box
    {
        if (detectionConf[i] > conf_thr) // If class score higher than theshold
        {
            v8xyxy box;
            // cast float dimensions to int
            box.x1     = static_cast<int>(detectionBox[i]);
            box.y1     = static_cast<int>(detectionBox[numBbox + i]);
            box.x2     = static_cast<int>(detectionBox[numBbox * 2 + i]);
            box.y2     = static_cast<int>(detectionBox[numBbox * 3 + i]);
            box.c      = detectionClass[i];
            box.c_prob = detectionConf[i];
            // calculate area only once
            box.area = (box.x2 - box.x1) * (box.y2 - box.y1);


            m_rawKeypointXList.clear();
            m_rawKeypointYList.clear();


            if (detectionKeypoints != nullptr) // (40,3780) with m_numOfKeypoints=20 kpts, numBbox=3780
            {
                for (int k = 0; k < m_numOfKeypoints; k++)
                {   
                    int kx_index = (numBbox * k * 3) + i;
                    int ky_index = (numBbox * k * 3) + numBbox + i;
                    int kc_index = (numBbox * k * 3) + (numBbox * 2) + i;
                    
                    if (kx_index > max_index || ky_index > max_index)
                    {
                        logger->error("Keypoint index out of bounds");
                        continue;
                    }

                    if (kc_index < conf_thr){
                        logger->info("Keypoint confidence is smaller than conf_thr");
                        continue;
                    }

                    int kx = static_cast<int>(detectionKeypoints[kx_index]);  // Get x-coordinate
                    int ky = static_cast<int>(detectionKeypoints[ky_index]);  // Get y-coordinate


                    m_rawKeypointXList.push_back(kx);
                    m_rawKeypointYList.push_back(ky);

                    // box.pose_kpts.push_back({kx,ky});

                }
                // Check if raw keypoint lists are not empty
                if (!m_rawKeypointXList.empty() && !m_rawKeypointYList.empty())
                {
                    // Find the minimum and maximum values in the raw X and Y keypoint lists
                    int minRawX = *std::min_element(m_rawKeypointXList.begin(), m_rawKeypointXList.end());
                    int maxRawX = *std::max_element(m_rawKeypointXList.begin(), m_rawKeypointXList.end());
                    int minRawY = *std::min_element(m_rawKeypointYList.begin(), m_rawKeypointYList.end());
                    int maxRawY = *std::max_element(m_rawKeypointYList.begin(), m_rawKeypointYList.end());
                    
                    // Calculate the range of raw coordinates for X and Y
                    int xRawRange = maxRawX - minRawX;
                    int yRawRange = maxRawY - minRawY;
                    
                    // Ensure the range is not zero to avoid division by zero
                    if (xRawRange == 0) xRawRange = 1;
                    if (yRawRange == 0) yRawRange = 1;
                    
                    int box_width = box.x2 - box.x1;
                    int box_height = box.y2 - box.y1;
                    int padding_x = box_width * 0.0;
                    int padding_y = box_height * 0.0;

                    int padded_x1 = box.x1 - padding_x;
                    int padded_y1 = box.y1 - padding_y;
                    int padded_x2 = box.x2 + padding_x;
                    int padded_y2 = box.y2 + padding_y;

                    int padded_box_width = padded_x2 - padded_x1;
                    int padded_box_height = padded_y2 - padded_y1;

                    // Iterate over each keypoint
                    for (int l = 0; l < m_numOfKeypoints; l++)
                    {
                        // Get the raw X and Y coordinates for the current keypoint
                        int kxRaw = m_rawKeypointXList[l];
                        int kyRaw = m_rawKeypointYList[l];
                        // Normalize the raw coordinates to fit within the bounding box dimensions
                        // int kx = static_cast<int>((kxRaw - minRawX) / xRawRange);
                        // int ky = static_cast<int>((kyRaw - minRawY) / yRawRange);
                        int kx = static_cast<int>(padded_x1 + padded_box_width * (kxRaw - minRawX) / xRawRange);
                        // int ky = static_cast<int>(padded_y1 + padded_box_height * (kyRaw - minRawY) / yRawRange);
                        int ky = static_cast<int> (kyRaw);
                       

                        // Ensure the normalized coordinates are within the image bounds
                        kx = std::max(0, std::min(kx, m_inputW - 1));
                        ky = std::max(0, std::min(ky, m_inputH - 1));
                        // Add the normalized keypoint to the pose keypoints list of the box
                        box.pose_kpts.push_back({kx, ky});
                        
                        
                    }
                }
            }

            // in order to speed up, push box according to its class into classwise vector<v8xyxy> bbox_list[box.c]
            if (box.c >= 0 && box.c < bbox_list.size())
                bbox_list[box.c].push_back(box);
            else
                logger->error("Invalid class index: {}", box.c);
            numThresholdedBbx++;
        }
    }

    return numThresholdedBbx;
}


int YOLOv8_Decoder::getIntersectArea(const v8xyxy &a, const v8xyxy &b)
{
    /*
        calculate intersection area as the product of intersection_w and intersection_h

        if intersection_w or intersection_h is 0, the intersection area and iou will always be 0.
        speed up by returning area as 0 instantly
    */

    // don't change the ordering
    auto logger         = spdlog::get(m_loggerStr);
    int  intersection_w = min(a.x2, b.x2) - max(a.x1, b.x1);
    if (intersection_w <= 0)
    {
        // intersection_w is 0 so area will be 0
        return 0;
    }

    // don't change the ordering
    int intersection_h = min(a.y2, b.y2) - max(a.y1, b.y1);
    if (intersection_h <= 0)
    {
        // intersection_w is 0 so area will be 0
        return 0;
    }

    // both intersection_w and intersection_h > 0, so intersection_area will be > 0
    return intersection_w * intersection_h;
}

float YOLOv8_Decoder::iou(const v8xyxy &a, const v8xyxy &b)
{
    /***
    *  iou: return the intersection over union ratio between rectangle a and b
    ***/
    auto logger            = spdlog::get(m_loggerStr);
    int  intersection_area = getIntersectArea(a, b);
    if (intersection_area == 0)
    {
        // when intersection_area is 0. speed up by returning iou 0.0 instantly
        return 0.0;
    }

    // area is now a member of v8xyxy. no need to repeat area calculation
    float iou = (float)intersection_area / (float)(a.area + b.area - intersection_area);
    // avoid division by zero
    // float iou               = (float) intersection_area / (float) (a.area + b.area - intersection_area + 1 );
    return iou;
}

float YOLOv8_Decoder::getBboxOverlapRatio(const v8xyxy &boxA, const v8xyxy &boxB)
{
    auto logger  = spdlog::get(m_loggerStr);
    int  iouArea = getIntersectArea(boxA, boxB);
    // area is now a member of v8xyxy. no need to repeat area calculation
    float ratio = (float)iouArea / (float)boxA.area;
    // avoid division by zero
    // float ratio   = (float) iouArea / (float) (boxA.area + 1);
    return ratio;
}

int YOLOv8_Decoder::doNMS(std::vector<std::vector<v8xyxy>> &bboxlist, const float iou_thr,
                            std::vector<std::vector<v8xyxy>> &classwisePicked, int num_Cls)
{
    /***
    *  do_nms : apply non-maximum-suppresion algorithm
    *    inputs:
    *      bboxlist: the content of bboxlist will be modified afer applying do_nms() (with side-effect)
    *      iou_thr: the iou threshold
    *
    *    outputs:
    *      picked: the picked bounding boxes after applying NMS
    *
    *    perform per-class NMS by including bbox with the highest detection confidence score
    *    for each candidante, compare iou and overlapping with boxes with higher confidence score
    *    if iou and overlapping is higher than threshold, exclude them from the list
    ***/
    auto logger = spdlog::get(m_loggerStr);

    for (int cls = 0; cls < num_Cls; cls++)
    {
        int size = bboxlist[cls].size();
        // logger->info("size of bboxlist[{}] {}", cls, size);

        // reference to bboxlist[cls]
        std::vector<v8xyxy> &currBboxList = bboxlist[cls];
        if (!classwisePicked[cls].empty())
            classwisePicked[cls].clear();
        std::vector<v8xyxy> &currPicked = classwisePicked[cls];

        // sort the bboxlist based on its class probability, in non-increasing order
        // speed up by sorting per class list only
        sort(currBboxList.begin(), currBboxList.end(), xyxyobj_compare);

        for (unsigned int i = 0; i < currBboxList.size(); ++i)
        {
            // check iou and overlapping
            bool keep = true;
            for (unsigned int j = 0; j < currPicked.size(); ++j)
            {
                float iouValue = iou(currBboxList[i], currPicked[j]);
                // logger->info("iou {} iou_thr {}", _iou, iou_thr);

                // checking iou with
                if (iouValue >= iou_thr)
                {
                    currBboxList[i].c_prob = 0.0; // reset score to 0
                    keep                   = false;
                    // break without pushing box to list
                    break;
                }

                // TODO:
                float bboxOverlapRatio = getBboxOverlapRatio(currBboxList[i], currPicked[j]);
                if (bboxOverlapRatio >= overlapThreshold)
                {
                    currBboxList[i].c_prob = 0.0; // reset score (obj) to 0
                    keep                   = false;
                    // break without pushing box to list
                    break;
                }
            }

            // a box passes iou can overlapping checks can be included to the list
            if (keep)
                currPicked.push_back(currBboxList[i]);
        }
    }

    int size = 0;
    for (int cls = 0; cls < num_Cls; cls++)
    {
        size += classwisePicked[cls].size();
    }

    return size;
}
