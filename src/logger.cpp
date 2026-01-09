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

#include "logger.hpp"

// --- Base Logger --- //
BaseLogger::BaseLogger(const std::string& loggerName)
{
    m_loggerName = "[" + loggerName + "]";
}

BaseLogger::~BaseLogger()
{
}


// DEPRECATED: Object Detection Logger removed (used BoundingBox)
// --- Object Detection Logger --- //
// void ObjectDetectionLogger::logObject(const BoundingBox& box)
// {
//     std::ostringstream oss;
//     std::string        logStr   = "";
//     std::string        classStr = "";
//     std::string        boxStr   = "";
//     std::string        confStr  = "";
//
//     // Determine class string
//     switch (box.label)
//     {
//     case 0:
//         classStr = "Human";
//         break;
//     case 1:
//         classStr = "Big Vehicle";
//         break;
//     case 2:
//         classStr = "Road Sign";
//         break;
//     default:
//         classStr = "Unknown";
//         break;
//     }
//
//     oss << "Cls: " << classStr << " Box: (" << box.x1 << ", " << box.y1 << ", " << box.x2 << ", " << box.y2
//         << ") Conf: " << box.confidence;
//
//     log(oss.str());
// }
//
// void ObjectDetectionLogger::logObjects(const std::vector<BoundingBox>& boxList)
// {
//     m_logs.clear();
//     for (auto& obj : boxList)
//         logObject(obj);
// }

// DEPRECATED: Object Tracking Logger removed (used Object and BoundingBox)
// --- Object Tracking Logger --- //
// void ObjectTrackingLogger::logObject(const Object& obj)
// {
//     std::ostringstream oss;
//     std::string        classStr;
//     const BoundingBox& box = obj.bbox;
//
//     // Determine class string
//     switch (box.label)
//     {
//     case 0:
//         classStr = "Human";
//         break;
//     case 1:
//         classStr = "Big Vehicle";
//         break;
//     case 2:
//         classStr = "Road Sign";
//         break;
//     default:
//         classStr = "Unknown";
//         break;
//     }
//
//     oss << "Cls: " << classStr << " Box: (" << box.x1 << ", " << box.y1 << ", " << box.x2 << ", " << box.y2
//         << ") Conf: " << box.confidence << " Distance: " << obj.distanceToCamera << " TTC: " << obj.currTTC
//         << " TTCCounter: " << obj.ttcCounter;
//
//     log(oss.str());
// }
//
// void ObjectTrackingLogger::logObjects(const std::vector<Object>& objList)
// {
//     m_logs.clear();
//     for (const auto& obj : objList)
//     {
//         if (obj.getStatus() == 1)
//             logObject(obj);
//     }
// }

// DEPRECATED: Logger Manager updated (removed ObjectDetectionLogger and ObjectTrackingLogger)
// --- Logger Manager --- //
LoggerManager::LoggerManager()
{
    // DEPRECATED: ObjectDetectionLogger and ObjectTrackingLogger removed
    // m_objectDetectionLogger         = new ObjectDetectionLogger("ObjDetect");
    // m_objectTrackingLogger          = new ObjectTrackingLogger("ObjTrack ");
}

LoggerManager::~LoggerManager()
{
    // DEPRECATED: ObjectDetectionLogger and ObjectTrackingLogger removed
    // delete m_objectDetectionLogger;
    // delete m_objectTrackingLogger;
    //
    // m_objectDetectionLogger         = nullptr;
    // m_objectTrackingLogger          = nullptr;
}