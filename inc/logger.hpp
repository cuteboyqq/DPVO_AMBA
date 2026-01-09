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

#ifndef SPDLOG_COMPILED_LIB
#define SPDLOG_COMPILED_LIB
#endif

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/syslog_sink.h>

// Include necessary WNC headers
#include "dataStructures.h"
// DEPRECATED: bounding_box.hpp, object.hpp removed
// #include "bounding_box.hpp"
// #include "object.hpp"

// --- Base Logger --- //
class BaseLogger
{
public:
    explicit BaseLogger(const std::string& loggerName);
    virtual ~BaseLogger();

    void log(const std::string& message)
    {
        m_logs.push_back(message);
    }

    void log(const std::string& logType, const std::string& message)
    {
        m_logs.push_back(m_loggerName + " " + logType + ": " + message);
    }

    template <typename T>
    void log(const std::string& logType, const T& value)
    {
        m_logs.push_back(m_loggerName + " " + logType + ": " + std::to_string(value));
    }

    void log(const std::string& logType, float value)
    {
        std::string logStr = m_loggerName + " " + logType + ": " + std::to_string(value);
        m_logs.push_back(logStr);
    }

    void log(const std::string& logType, const cv::Point& point)
    {
        m_logs.push_back(m_loggerName + " " + logType + ": (" + std::to_string(point.x) + ", " + std::to_string(point.y)
                         + ")");
    }

    template <typename T>
    void logLineValue(const std::string& logType, const T& valLeft, const T& valRight)
    {
        m_logs.push_back(m_loggerName + " " + logType + ": Left: " + std::to_string(valLeft) + "\t Right: "
                         + std::to_string(valRight));
    }

    void logLineValue(const std::string& logType, const cv::Point& pLeft, const cv::Point& pRight)
    {
        m_logs.push_back(m_loggerName + " " + logType + ": Left: (" + std::to_string(pLeft.x) + ", "
                         + std::to_string(pLeft.y) + ")" + "\t Right: (" + std::to_string(pRight.x) + ", "
                         + std::to_string(pRight.y) + ")");
    }

    std::vector<std::string> m_logs;
    std::string              m_loggerName;
};

// DEPRECATED: Object Detection Logger removed (used BoundingBox)
// --- Object Detection Logger --- //
// class ObjectDetectionLogger : public BaseLogger
// {
// public:
//     explicit ObjectDetectionLogger(const std::string& loggerName) : BaseLogger(loggerName)
//     {
//     }
//     void logObject(const BoundingBox& bbox);
//     void logObjects(const std::vector<BoundingBox>& bboxList);
// };

// DEPRECATED: Object Tracking Logger removed (used Object)
// --- Object Tracking Logger --- //
// class ObjectTrackingLogger : public BaseLogger
// {
// public:
//     explicit ObjectTrackingLogger(const std::string& loggerName) : BaseLogger(loggerName)
//     {
//     }
//     void logObject(const Object& obj);
//     void logObjects(const std::vector<Object>& objList);
// };


// DEPRECATED: Logger Manager updated (removed ObjectDetectionLogger and ObjectTrackingLogger)
// --- Logger Manager --- //
class LoggerManager
{
public:
    LoggerManager();
    ~LoggerManager();
    // DEPRECATED: ObjectDetectionLogger and ObjectTrackingLogger removed
    // ObjectDetectionLogger*         m_objectDetectionLogger;
    // ObjectTrackingLogger*          m_objectTrackingLogger;
};
