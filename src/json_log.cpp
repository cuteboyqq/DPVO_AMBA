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

#include "json_log.hpp"

using json = nlohmann::json;

JSON_LOG::JSON_LOG(std::string file, Config_S* _config) : m_jsonFilePath(file)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::syslog_logger_mt("adas-output", "adas-output", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
    auto logger = spdlog::stdout_color_mt("JSON");
    logger->set_pattern("[%n] [%^%l%$] %v");
#endif

    logger->set_level(_config->stDebugConfig.enableJson ? spdlog::level::debug : spdlog::level::info);
    m_bDebugProfiling = _config->stDebugProfiling; 
}

JSON_LOG::~JSON_LOG()
{
#ifdef SPDLOG_USE_SYSLOG
    spdlog::drop("adas-output");
#else
    spdlog::drop("JSON");
#endif
}

// DEPRECATED: Function using BoundingBox and Object removed
// std::string JSON_LOG::logInfo(
//     WNC_APP_Results appResult,
//     std::vector<BoundingBox> humanBBoxList,
//     std::vector<BoundingBox> vehicleBBoxList,
//     std::vector<Object> trackedObjList,
//     int frameIdx,
//     DebugProfile debugProfile,
//     char* version)
// {
//     auto logger = spdlog::get(
// #ifdef SPDLOG_USE_SYSLOG
//         "app-output"
// #else
//         "JSON"
// #endif
//         );
//
//     json        jsonData, jsonDataCurrentFrame, adasStr;
//     std::string strIndex = std::to_string(frameIdx);

//     if (m_bSaveToJSONFile)
//     {
//         std::ifstream inFile(m_jsonFilePath);
//
//         if (inFile.is_open())
//         {
//             inFile >> jsonData;
//             inFile.close();
//         }
//         else
//             logger->error("Unable to open the file");
//     }
//
//
// // #ifdef SAV837
// //     jsonDataCurrentFrame["frame_ID"][strIndex]["Seq"] = std::to_string(u32FrameSequenceNumber);
// // #endif
//  
//     jsonData["version"].push_back(version);
//     jsonDataCurrentFrame["version"].push_back(version);
//
//
// int FRAME_WIDTH = 1920;
// int FRAME_HEIGHT = 1080;
// int MODEL_WIDTH = 512;
// int MODEL_HEIGHT = 288;

//     if (m_bSaveTrackObj)
//     {
//         for (auto box : appResult.trackObjList)
//         {
//             if (box.bboxList.size() > 0)
//             {
//                 BoundingBox lastBox = box.bboxList.back();
//                 BoundingBox rescaleBox(-1, -1, -1, -1, -1);
//                 utils::rescaleBBox(lastBox, rescaleBox, MODEL_WIDTH, MODEL_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT);
//                 string classStr = lastBox.label == HUMAN ? "HUMAN" : "VEHICLE";
//                 json   track    = {{"trackObj.x1", rescaleBox.x1},
//                               {"trackObj.y1", rescaleBox.y1},
//                               {"trackObj.x2", rescaleBox.x2},
//                               {"trackObj.y2", rescaleBox.y2},
//                               {"trackObj.confidence", rescaleBox.confidence},
//                               {"trackObj.label", classStr},
//                               {"trackObj.distanceToCamera", box.distanceToCamera},
//                               {"trackObj.id", box.id}};
//                 jsonData["frame_ID"][strIndex]["trackObj"].push_back(track);
//                 jsonDataCurrentFrame["frame_ID"][strIndex]["trackObj"].push_back(track);
//             }
//         }
//     }
//
//     if (m_bSaveFaceObj)
//     {
//         for (auto box : appResult.faceObjList)
//         {   
//             BoundingBox rescaleBox;
//             utils::rescaleBBox(box, rescaleBox, MODEL_WIDTH, MODEL_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT);
//             json det = {{"face.x1", rescaleBox.x1}, {"face.y1", rescaleBox.y1},
//                         {"face.x2", rescaleBox.x2}, {"face.y2", rescaleBox.y2},
//                         {"face.label", "HUMAN"},  {"face.confidence", rescaleBox.confidence}};
//             jsonData["frame_ID"][strIndex]["face"]["FACE"].push_back(det);
//             jsonDataCurrentFrame["frame_ID"][strIndex]["face"]["FACE"].push_back(det);
//         }
//     }
//
//     // if (m_bSavePoseObj)
//     // {
//     //     for (auto box : appResult.poseObjList)
//     //     {   
//     //         BoundingBox rescaleBox;
//     //         utils::rescaleBBox(box, rescaleBox, MODEL_WIDTH, MODEL_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT);
//     //         json det = {{"pose.x1", rescaleBox.x1}, {"pose.y1", rescaleBox.y1},
//     //                     {"pose.x2", rescaleBox.x2}, {"pose.y2", rescaleBox.y2},
//     //                     {"pose.label", "HUMAN"},  {"pose.confidence", rescaleBox.confidence}};
//     //         jsonData["frame_ID"][strIndex]["pose"]["HUMAN"].push_back(det);
//     //         jsonDataCurrentFrame["frame_ID"][strIndex]["pose"]["HUMAN"].push_back(det);
//     //     }
//     // }
//
//     if (m_bSavePoseObj)
//     {
//         for (auto box : appResult.poseObjList)
//         {   
//             BoundingBox rescaleBox;
//             utils::rescaleBBox(box, rescaleBox,
//                             MODEL_WIDTH, MODEL_HEIGHT,
//                             FRAME_WIDTH, FRAME_HEIGHT);
//
//             // Prepare keypoints JSON array
//             json keypoints = json::array();
//             for (size_t i = 0; i < box.pose_kpts.size(); ++i)
//             {
//                 const auto& kp = box.pose_kpts[i];
//                 json kp_json = {
//                     // {"id", static_cast<int>(i)},   // index of keypoint
//                     {"x", kp.first},               // x coordinate
//                     {"y", kp.second}               // y coordinate
//                     // {kp.first, kp.second},
//                 };
//                 keypoints.push_back(kp_json);
//             }
//
//             json det = {
//                 {"pose.x1", rescaleBox.x1},
//                 {"pose.y1", rescaleBox.y1},
//                 {"pose.x2", rescaleBox.x2},
//                 {"pose.y2", rescaleBox.y2},
//                 {"pose.label", "HUMAN"},
//                 {"pose.confidence", rescaleBox.confidence},
//                 {"pose.keypoints", keypoints}
//             };
//
//             jsonData["frame_ID"][strIndex]["pose"]["HUMAN"].push_back(det);
//             jsonDataCurrentFrame["frame_ID"][strIndex]["pose"]["HUMAN"].push_back(det);
//         }
//     }
//
//
//     if (m_bSaveDetObjLog)
//     {
//         // VEHICLE
//         for (auto box : vehicleBBoxList)
//         {
//             BoundingBox rescaleBox;
//             utils::rescaleBBox(box, rescaleBox, MODEL_WIDTH, MODEL_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT);
//             json det = {{"detectObj.x1", rescaleBox.x1}, {"detectObj.y1", rescaleBox.y1},
//                         {"detectObj.x2", rescaleBox.x2}, {"detectObj.y2", rescaleBox.y2},
//                         {"detectObj.label", "VEHICLE"},  {"detectObj.confidence", rescaleBox.confidence}};
//             jsonData["frame_ID"][strIndex]["detectObj"]["VEHICLE"].push_back(det);
//             jsonDataCurrentFrame["frame_ID"][strIndex]["detectObj"]["VEHICLE"].push_back(det);
//         }
//
//         for (auto box : humanBBoxList)
//         {
//             BoundingBox rescaleBox;
//             utils::rescaleBBox(box, rescaleBox, MODEL_WIDTH, MODEL_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT);
//             json det = {{"detectObj.x1", rescaleBox.x1}, {"detectObj.y1", rescaleBox.y1},
//                         {"detectObj.x2", rescaleBox.x2}, {"detectObj.y2", rescaleBox.y2},
//                         {"detectObj.label", "HUMAN"},    {"detectObj.confidence", rescaleBox.confidence}};
//             jsonData["frame_ID"][strIndex]["detectObj"]["HUMAN"].push_back(det);
//             jsonDataCurrentFrame["frame_ID"][strIndex]["detectObj"]["HUMAN"].push_back(det);
//         }
//     }
//
//     if (m_bDebugProfiling)
//     {
//         json profile = {{"AI Inference Time", debugProfile.AIInfrerenceTime},
//                         {"AI Input Buffer Size", debugProfile.yoloADAS_InputBufferSize},
//                         {"AI Output Buffer Size", debugProfile.yoloADAS_OutputBufferSize},
//                         {"Post Processing Input Buffer Size", debugProfile.postProc_InputBufferSize},
//                         {"Post Processing Output Buffer Size", debugProfile.postProc_OutputBufferSize},
//                         {"Ashacam Result Buffer Size", debugProfile.adasResultBufferSize}};
//         jsonData["frame_ID"][strIndex]["debugProfile"].push_back(profile);
//         jsonDataCurrentFrame["frame_ID"][strIndex]["debugProfile"].push_back(profile);
//     }
//
//
// //     json statusCodeJSON = {{"statusCode", statusCode}};
// //     jsonData["frame_ID"][strIndex]["statusCode"].push_back(statusCodeJSON);
// //     jsonDataCurrentFrame["frame_ID"][strIndex]["statusCode"].push_back(statusCodeJSON);
//
// //     std::string jsonCurrentFrameString = jsonDataCurrentFrame.dump();
//
//     // TODO: Not implemented yet
//     std::string jsonCurrentFrameString = "";
//
//     // Convert the JSON object to a string with indentation
//     if (m_bShowJson)
//     {
// std::string jsonCurrentFrameString = jsonDataCurrentFrame.dump();
// #ifndef SPDLOG_USE_SYSLOG
//         logger->info("====================================================================================");
// #endif
//         logger->info("json:{}", jsonCurrentFrameString);
// #ifndef SPDLOG_USE_SYSLOG
//         logger->info("====================================================================================");
// #endif
//     }
//     if (m_bSaveToJSONFile)
//         _appendToJsonFile(strIndex, jsonDataCurrentFrame);
//     return jsonCurrentFrameString;
// }

void JSON_LOG::_initializeJsonFile()
{
    json rootStructure = {};

    std::ofstream outFile(m_jsonFilePath);
    if (outFile)
    {
        outFile << rootStructure.dump();
    }
    else
    {
        auto logger = spdlog::get("JSON");
        logger->error("Unable to initialize JSON file");
    }
}

void JSON_LOG::_appendToJsonFile(const std::string& frameIdx, const json& frameData)
{   
    auto logger = spdlog::get("JSON");
    logger->info("m_jsonFilePath = {}",m_jsonFilePath);
    std::ofstream outFile(m_jsonFilePath, std::ios::app);
    if (outFile)
    {
        outFile << frameData.dump() << '\n';
    }
    else
    {
        auto logger = spdlog::get("JSON");
        logger->error("Unable to open the file for writing");
    }
}


