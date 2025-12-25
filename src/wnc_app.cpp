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

#include "wnc_app.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

using namespace std;

// Add these helper functions before _createTensorDirectories()
bool _directoryExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

bool _createDirectory(const std::string& path) {
    if (mkdir(path.c_str(), 0777) == 0) {
        return true;
    }
    return errno == EEXIST;
}

WNC_APP::WNC_APP(const std::string configPath, std::string inputFile)
{
    _init(configPath, inputFile); // Init with config
};

// Helper functions
std::shared_ptr<spdlog::logger> WNC_APP::_initializeLogger()
{
#ifdef SPDLOG_USE_SYSLOG
    m_logger = spdlog::syslog_logger_mt("app", "app-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
    m_logger = spdlog::stdout_color_mt("App");
    m_logger->set_pattern("[%n] [%^%l%$] %v");

    m_logger->info(R"(
    =================================================
    =                   WNC APP                    =
    =================================================
    Version: v{}
    -------------------------------------------------
    )",
                   APP_VERSION);

    return m_logger;
#endif
}

void WNC_APP::_initializeConfig(const std::string& configPath, std::shared_ptr<spdlog::logger>& logger)
{
    m_config           = new Config_S();
    m_appConfigReader  = new AppConfigReader();
    m_appConfigReader->read(configPath);
    m_config = m_appConfigReader->getConfig();
    m_logger->set_level(m_config->stDebugConfig.App ? spdlog::level::debug : spdlog::level::info);

    _checkPaths(m_logger);
}

void WNC_APP::_checkPaths(std::shared_ptr<spdlog::logger>& logger)
{
    if (!utils::checkFileExists(m_config->modelPath))
    {
        m_logger->error("Model path {} not found", m_config->modelPath);
        exit(1);
    }
}

void WNC_APP::_initializeModules(std::shared_ptr<spdlog::logger>& logger, std::string inputFile)
{
    // ==== Live Mode ==== //

    // IO
    m_frameWidth         = m_config->frameWidth;
    m_frameHeight        = m_config->frameHeight;
    m_dbg_imgsDirPath    = m_config->stDebugConfig.imgsDirPath;
    m_dbg_rawImgsDirPath = m_config->stDebugConfig.rawImgsDirPath;

    // Detection ROI
    m_objectDetectionROI.x1    = m_frameWidth * 0.0;
    m_objectDetectionROI.y1    = m_frameHeight * 0.0;
    m_objectDetectionROI.x2    = m_frameWidth * 1.0;
    m_objectDetectionROI.y2    = m_frameHeight * 1.0;
    m_objectDetectionROI.label = -1;


    // Tracking
    m_maxTracking = m_config->stTrackerConifg.maxTracking;

    if (m_objectDetectionROI.getArea() <= 0)
    {
        m_logger->error("Detection ROI area <= 0");
        exit(1);
    }

    // FPS
    m_thresholdFps = m_config->procFrameRate / m_config->procFrameStep;  // 1 seconds long

    // YOLOv8
    m_yolov8 = new YOLOv8(m_config, [this]() { this->wakeProcessingThread(); });
    m_yolov8PostProc =
        new YOLOv8_POSTPROC(m_config, [this]() { this->wakeProcessingThread(); });
    m_logger->debug("Initialized Post Processing module");

    // Tracking ROI
    _initTrackingROI();

    // Log
    std::string baseDir = m_config->stDebugConfig.logsDirPath;
    std::string fileName = inputFile.substr(inputFile.find_last_of("/") + 1);
    system(("mkdir -p " + baseDir + "/" + fileName).c_str());
    inputFile = baseDir + "/" + fileName + ".log";

    m_logger->info("Output log file {}", inputFile);
    m_jsonLog = new JSON_LOG(inputFile, m_config);

    // Socket
    m_socket  = new SOCKET();

    // Tracker
    m_humanTracker   = new ObjectTracker(m_config, "human");
    m_vehicleTracker = new ObjectTracker(m_config, "vehicle");
    m_trackedObjList.reserve(m_humanTracker->m_maxTracking * 2);
    m_logger->debug("Initialized Human and Vehicle Tracker modules");

    m_logger->debug("Initialized AI model and other modules");
}

void WNC_APP::_initTrackingROI()
{   
    if(m_config->enableROI)
    {
        m_roi.x1  = MODEL_WIDTH * m_config->startXRatio;
        m_roi.y1  = MODEL_HEIGHT * m_config->startYRatio;
        m_roi.x2  = MODEL_WIDTH * m_config->endXRatio;
        m_roi.y2  = MODEL_HEIGHT * m_config->endYRatio;
        m_roiBBox = BoundingBox(m_roi.x1, m_roi.y1, int(m_roi.x2-m_roi.x1), int(m_roi.y2-m_roi.y1), -1);
    }
    else
    {
        m_roi.x1  = 0;
        m_roi.y1  = 0;
        m_roi.x2  = MODEL_WIDTH - 1;
        m_roi.y2  = MODEL_HEIGHT - 1;
        m_roiBBox = BoundingBox(0, 0, MODEL_WIDTH, MODEL_HEIGHT, -1);
    }

    // Print/log ROI
    m_logger->error("ROI: x1={}, y1={}, x2={}, y2={}", m_roi.x1, m_roi.y1, m_roi.x2, m_roi.y2);
    
}

void WNC_APP::_setupDebugAndDisplayConfigs()
{
    // Historical Feed (Add at 2024-07-06)
    m_counters            = m_config->HistoricalFeedModeConfig.imageModeStartFrame;
    m_max_counters        = m_config->HistoricalFeedModeConfig.imageModeEndFrame;
    m_historicalFrameName = m_config->HistoricalFeedModeConfig.imageModeFrameName;

    m_frameStep_wnc = m_config->procFrameStep;

    _readDebugConfig();
    _readDisplayConfig();
    m_estimateTime = m_config->stShowProcTimeConfig.App;
    m_logger->info("Set up Debug and Display Config");
}

void WNC_APP::_startThreads()
{
#if defined(CV28) || defined(CV28_SIMULATOR)
    startProcessingThread();
    m_processingThreadRunning = true;
#endif

    // TODO: Not implemented yet
    // if (m_config->stDebugConfig.saveRawImages)
    //     _startHistoricalFeedThread();

    m_yolov8->runThread();
    m_yolov8PostProc->runThread();

    if (m_config->HistoricalFeedModeConfig.visualizeMode == 0 ||
        m_config->HistoricalFeedModeConfig.visualizeMode == 1)
    {
        // m_jsonLog->m_bSaveDetObjLog = true;
        // m_jsonLog->m_bSaveLaneInfo  = true;
    }
    m_logger->info("Started threads");
}

void WNC_APP::_createTensorDirectories(std::string inputFile) {
    std::string baseDir = "tmp_tensors/";
    
    // Extract model name from full path
    std::string modelPath = m_config->modelPath;
    size_t lastSlash = modelPath.find_last_of("/\\");
    std::string modelName = (lastSlash != std::string::npos) ? 
                           modelPath.substr(lastSlash + 1) : modelPath;
    
    // Remove file extension if present
    size_t lastDot = modelName.find_last_of(".");
    if (lastDot != std::string::npos) {
        modelName = modelName.substr(0, lastDot);
    }
    
    // Create the full directory path
    std::string videoName = inputFile.substr(inputFile.find_last_of("/") + 1);
    std::string fullPath = baseDir + modelName + "/" + videoName;
    
    m_logger->debug("Creating tensor directories for model: {}, input: {}", modelName, videoName);
    
    if (utils::createDirectories(fullPath)) {
        m_logger->debug("Created directory chain: {}", fullPath);
        m_yolov8->updateTensorPath(fullPath); // Update the path in the YOLO ADAS module
    } else {
        m_logger->error("Failed to create directory chain: {}", fullPath);
    }
}

void WNC_APP::_init(std::string configPath, std::string inputFile)
{
    m_logger = _initializeLogger();

    // Get Date Time
    utils::getDateTime(m_dbg_dateTime);

    _initializeConfig(configPath, m_logger);
    _initializeModules(m_logger, inputFile);
    _setupDebugAndDisplayConfigs();
    _createTensorDirectories(inputFile);
    _startThreads();
    m_logger->info("Finished Init");
}

void WNC_APP::_readDebugConfig()
{
    auto logger = spdlog::get(
#ifdef SPDLOG_USE_SYSLOG
        "app"
#else
        "App"
#endif
        );

    const auto& debugConfig = m_config->stDebugConfig;
    m_dbg_saveImages        = debugConfig.saveImages;

    const std::string basePath = m_dbg_dateTime;
    m_dbg_imgsDirPath += "/" + basePath;
    m_dbg_rawImgsDirPath += "/" + basePath;

    const auto createDirIfNeeded = [&](const std::string& fullPath, const bool shouldCreate, const std::string& name) {
        if (!fullPath.empty() && shouldCreate)
        {
            // const std::string fullPath = dirPath + "/" + basePath;
            if (utils::createDirectories(fullPath))
                m_logger->debug("Folders created successfully: {}", fullPath);
            else
                m_logger->warn("Error creating folders: {}", fullPath);
        }
    };

#ifndef SHOW_IMAGES
    createDirIfNeeded(m_dbg_imgsDirPath, debugConfig.saveImages, "Images");
    createDirIfNeeded(m_dbg_rawImgsDirPath, debugConfig.saveRawImages, "Raw Images");
#endif
}

void WNC_APP::_readDisplayConfig()
{
    m_dsp_results           = m_config->stDisplayConfig.results;
    m_dsp_objectDetection   = m_config->stDisplayConfig.objectDetection;
    m_dsp_objectTracking    = m_config->stDisplayConfig.objectTracking;
    m_dsp_poseDetection     = m_config->stDisplayConfig.poseDetection;
    m_dsp_information       = m_config->stDisplayConfig.information;
}

WNC_APP::~WNC_APP()
{
    stopThread();

#ifdef SPDLOG_USE_SYSLOG
    spdlog::drop("app");
#else
    spdlog::drop("App");
#endif

    delete m_appConfigReader;
    delete m_config;
    delete m_yolov8;
    delete m_yolov8PostProc;
    delete m_humanTracker;
    delete m_vehicleTracker;
    delete m_jsonLog;
    delete m_socket;

    m_appConfigReader   = nullptr;
    m_config            = nullptr;
    m_yolov8            = nullptr;
    m_yolov8PostProc    = nullptr;
    m_humanTracker      = nullptr;
    m_vehicleTracker    = nullptr;
    m_jsonLog           = nullptr;
    m_socket            = nullptr;
};

void WNC_APP::stopThread()
{
    if (!m_yolov8->m_threadTerminated)
    {
        m_yolov8->stopThread();
    }

    if (!m_yolov8PostProc->m_threadTerminated)
    {
        m_yolov8PostProc->stopThread();
    }
    stopProcessingThread();
    m_logger->warn("Stop Threads sucessfully");
}


// ============================================
//               Live Mode Main
// ============================================
void WNC_APP::updateInput(ea_tensor_t* imgTensor)
{
    bool bufferCondition = false;
    int  frameStep       = m_frameStep_wnc;

    bufferCondition = m_yolov8->m_inputFrameBuffer.size() >= 2;

    // Skip if no image saving and frame skipping condition is not met
    bool skipFrame = (++m_frameNum % frameStep != 0);

    // Callback when buffer condition is met
    if (skipFrame || bufferCondition)
    {
        if (m_frameProcessedCallback)
        {
            m_frameProcessedCallback();
        }
        return;
    }

    // Callback when image is empty
    if (imgTensor == nullptr)
    {
        if (m_frameProcessedCallback)
        {
            m_frameProcessedCallback();
        }

        m_logger->warn("Input image is empty");
        return;
    }

    m_bDone = false;
    m_logger->debug(" ============ Source Frame Index: {} ============", m_frameIdx_wnc);
    m_logger->debug(" ============ Result Frame Index: {} ============", m_resultFrameIdx);

    // Handle ROI cropping if enabled //TODO: Not implemented yet
    // cv::Mat inputImage;
    // if (m_config->enableROI == 1)
    // {
    //     if (m_newXStart == -1)
    //         m_newXStart = static_cast<int>(imgFrame.cols * m_config->startXRatio);
    //     if (m_newXEnd == -1)
    //         m_newXEnd = static_cast<int>(imgFrame.cols * m_config->endXRatio);
    //     if (m_newYStart == -1)
    //         m_newYStart = static_cast<int>(imgFrame.rows * m_config->startYRatio);
    //     if (m_newYEnd == -1)
    //         m_newYEnd = static_cast<int>(imgFrame.rows * m_config->endYRatio);

    //     const cv::Rect roi(m_newXStart, m_newYStart, m_newXEnd - m_newXStart, m_newYEnd - m_newYStart);
    //     inputImage = imgFrame(roi).clone();
    // }
    // else
    // {
    //     inputImage = imgFrame.clone();
    // }

    updateFrameIndex(); // Update frame index

    // Update input frame (tensor)
    m_yolov8->updateInputFrame(imgTensor, m_frameIdx_wnc);

    // Run AI process
    wakeProcessingThread();
    m_logger->debug("Finished {}", __func__);
}

// TODO: Not implemented yet
#ifdef SAV837
// Alister add 2024-10-14
void WNC_APP::_sendImageOrLog(cv::Mat img, const std::string& jsonLog, const std::string& server_ip, int server_port,
                               int frameIdx)
{
    if (m_config->HistoricalFeedModeConfig.visualizeMode == 0)
    {
        if (!img.empty())
        {
            m_socket->send_image_and_log_live_mode(img, jsonLog.c_str(), server_ip.c_str(), server_port, frameIdx);
        }
    }
    else if (m_config->HistoricalFeedModeConfig.visualizeMode == 1)
    {
        m_socket->send_json_log(jsonLog.c_str(), server_ip.c_str(), server_port);
    }
}

// Alister add 2024-10-14
void WNC_APP::_sendDataLiveMode(int success, const std::string& jsonLog, const std::string& server_ip, int server_port,
                                 int frameIdx)
{
    cv::Mat img;
    if (m_config->HistoricalFeedModeConfig.visualizeMode == 2)
    {
        return;
    }

    if (m_config->stDebugConfig.saveRawImages || m_config->HistoricalFeedModeConfig.inputMode == 3)
    {
        img = m_displayImg; // Use current display image
    }
    else if (success == ADAS_SUCCESS)
    {
        img = m_yolov8->m_historical->getImageAndRemoveByFrameIdx(frameIdx); // Get historical image
    }

    _sendImageOrLog(img, jsonLog, server_ip, server_port, frameIdx);
}

void WNC_APP::_sendDataHistoricalMode(int visualizeMode, const std::string& imagePath, const std::string& jsonLog,
                                       const std::string& server_ip, int server_port, int frameId)
{
    if (visualizeMode == 0)
    {
        // Send image and log, including frameId and image path
        m_socket->send_image_and_log_and_frameIdx_and_imgPath(imagePath, jsonLog.c_str(), server_ip.c_str(),
                                                              server_port, frameId);
    }
    else if (visualizeMode == 1)
    {
        // Send only the JSON log
        m_socket->send_json_log(jsonLog.c_str(), server_ip.c_str(), server_port);
    }
}
#endif


void WNC_APP::printObjectList(const std::vector<Object>& objList, const std::shared_ptr<spdlog::logger>& logger) {
    if (logger) {
        logger->info("Tracking {} object(s)...", objList.size());
    } else {
        printf("Tracking %zu object(s)...\n", objList.size());
    }

    for (const auto& obj : objList) {
        if (logger) {
            logger->info(" Object ID: {}, Class: {}", obj.id, obj.classStr);
            logger->info("   ▸ BoundingBox: ({}, {}, {}, {}, label={})",
                         obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2, obj.bbox.label);
            logger->info("------------------------------------------------------");
        } else {
            printf(" Object ID: %d, Class: %s\n", obj.id, obj.classStr.c_str());
            printf("   ▸ BoundingBox: (%d, %d, %d, %d, label=%d)\n",
                   obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2, obj.bbox.label);
            printf("------------------------------------------------------\n");
        }
    }
}


// Example helper for BoundingBox
void WNC_APP::_printBoundingBox(const BoundingBox& box, int idx) 
{
    if (idx >= 0)
        std::cout << "  [" << idx << "] ";
    std::cout << "BBox(x1=" << box.x1
              << ", y1=" << box.y1
              << ", x2=" << box.x2
              << ", y2=" << box.y2 << ")\n";
}

// Example helper for Object
void WNC_APP::_printObject(const Object& obj, int idx)
{
    if (idx >= 0)
        std::cout << "  [" << idx << "] ";
    std::cout << "Object(id=" << obj.id
              << ", x1=" << obj.bbox.x1
              << ", y1=" << obj.bbox.y1
              << ", x2=" << obj.bbox.x2
              << ", y2=" << obj.bbox.y2 << ")\n";
}

// Main printer for WNC_APP_Results
void WNC_APP::_printAppResult(const WNC_APP_Results& res) 
{
    std::cout << "==== App Result ====\n";
    std::cout << "FrameID: " << res.frameID << "\n";
    std::cout << "EventType: " << res.eventType << "\n";
    std::cout << "isDetectLine: " << (res.isDetectLine ? "true" : "false") << "\n";

    std::cout << "\nDetected Objects (" << res.detectObjList.size() << "):\n";
    for (size_t i = 0; i < res.detectObjList.size(); ++i)
        _printBoundingBox(res.detectObjList[i], i);

    std::cout << "\nTracked Objects (" << res.trackObjList.size() << "):\n";
    for (size_t i = 0; i < res.trackObjList.size(); ++i)
        _printObject(res.trackObjList[i], i);

    std::cout << "\nFace Objects (" << res.faceObjList.size() << "):\n";
    for (size_t i = 0; i < res.faceObjList.size(); ++i)
        _printBoundingBox(res.faceObjList[i], i);

    std::cout << "\nPose Objects (" << res.poseObjList.size() << "):\n";
    for (size_t i = 0; i < res.poseObjList.size(); ++i)
        _printBoundingBox(res.poseObjList[i], i);

    std::cout << "\nVehicle Objects (" << res.vehicleObjList.size() << "):\n";
    for (size_t i = 0; i < res.vehicleObjList.size(); ++i)
        _printBoundingBox(res.vehicleObjList[i], i);

    std::cout << "\nRider Objects (" << res.riderObjList.size() << "):\n";
    for (size_t i = 0; i < res.riderObjList.size(); ++i)
        _printBoundingBox(res.riderObjList[i], i);

    std::cout << "====================\n";
}


int WNC_APP::appProcess(cv::Mat& resultMat)
{
    auto logger = spdlog::get(
    #ifdef SPDLOG_USE_SYSLOG
            "app-output"
    #else
            "JSON"
    #endif
            );
    int         success = APP_FAILURE;
    int         predFrameIdx;
    std::string server_ip   = m_config->HistoricalFeedModeConfig.serverIP;
    int         server_port = m_config->HistoricalFeedModeConfig.serverPort;

    auto time_0 = m_estimateTime ? std::chrono::high_resolution_clock::now()
                                 : std::chrono::time_point<std::chrono::high_resolution_clock>{};

    // Get last prediction
    YOLOv8_Prediction pred;
    if (m_yolov8->getLastestPrediction(pred, predFrameIdx))
    {
        m_yolov8PostProc->updatePredictionBuffer(pred, predFrameIdx);
    }
    if (m_yolov8PostProc->getLastestResult(m_procResult, m_resultFrameIdx))
    {   

        if (m_frameProcessedCallback)
        {
            m_frameProcessedCallback();
        }

        // Get m_resultImage from m_procResult
        m_resultImage = m_procResult.img;

        // Update m_displayImg following m_resultImage
        m_displayImg = m_resultImage.clone();

        bool isForCalibrationAI = false;
        success = _objectDetection() && _objectTracking() // && _poseDetection()
                      ? APP_SUCCESS
                      : APP_FAILURE;

        if (success == APP_FAILURE)
            m_logger->warn("One or more of App sub-tasks failed");

        _showDetectionResults();

        if (m_dsp_results && success == APP_SUCCESS)
        {
            cv::resize(m_procResult.img, m_displayImg, cv::Size(m_frameWidth, m_frameHeight), cv::INTER_LINEAR);
            _drawResults();
            if (m_dbg_saveImages)
                _saveDrawResults();
        }

        _getResultImage(resultMat);
        _updateDebugProfile();


        // // TODO:
        std::string jsonLog = m_jsonLog->logInfo(
            m_result_log, m_humanBBoxList, m_vehicleBBoxList,
            m_trackedObjList, m_resultFrameIdx,m_debugProfile,
            APP_VERSION);
        
        // cout<<jsonLog<<endl;

        _updateAppResult(); //TODO:

        WNC_APP_Results res;
        getAppResult(res);
        // printObjectList(res.trackObjList, nullptr);
        _printAppResult(res);
        // done — memory released automatically when function ends
/*   
#ifndef SPDLOG_USE_SYSLOG
        logger->info("====================================================================================");
#endif
        logger->info("log:frameID = {}", res.frameID);
        logger->info("log:res.trackObjList.size() = {}", res.trackObjList.size());
#ifndef SPDLOG_USE_SYSLOG
        logger->info("====================================================================================");
#endif
*/


#ifdef SAV837 //TODO: Not implemented yet
        // Transfer data through socket to visualize
        _sendDataLiveMode(success, jsonLog, server_ip, server_port, m_resultFrameIdx);
#endif

        success = APP_LOGGED;
        m_bDone = true;
    }

    if (m_estimateTime)
    {
        auto time_1 = std::chrono::high_resolution_clock::now();
        m_logger->info("Processing Time: {} ms",
                       std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
    }

    return success;
}


void WNC_APP::addFrame(ea_tensor_t* imgTensor)
{
    updateInput(imgTensor);
}

void WNC_APP::startProcessingThread()
{
    auto logger        = m_logger;
    m_processingThread = std::thread([this, logger]() {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        while (m_processingThreadRunning)
        {
            m_queueCV.wait_for(lock, std::chrono::milliseconds(1000), [this]() {
            return !m_processingThreadRunning || _hasWorkToDo();
            });
            if (!m_processingThreadRunning)
                break;

            if (_hasWorkToDo())
            {
                lock.unlock();
                m_bDone = false;
                cv::Mat resultMat;

                appProcess(resultMat);

                lock.lock();
            }
        }
        m_logger->debug("startProcessingThread is terminated");
    });
}

void WNC_APP::stopProcessingThread()
{
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_processingThreadRunning = false;
    }
    m_queueCV.notify_one();
    if (m_processingThread.joinable())
        m_processingThread.join();
}

bool WNC_APP::_hasWorkToDo()
{
    return !m_yolov8->isPredictionBufferEmpty() || !m_yolov8PostProc->isOutputBufferEmpty();
}

bool WNC_APP::isProcessingComplete()
{
    bool yoloInputEmpty      = m_yolov8->isInputBufferEmpty();
    bool yoloPredEmpty       = m_yolov8->isPredictionBufferEmpty();
    bool postProcInputEmpty  = m_yolov8PostProc->isInputBufferEmpty();
    bool postProcOutputEmpty = m_yolov8PostProc->isOutputBufferEmpty();

    m_logger->debug("isProcessingComplete: YOLO input empty: {}, YOLO pred empty: {}", yoloInputEmpty,
                    yoloPredEmpty);
    m_logger->debug("isProcessingComplete: PostProc input empty: {}, PostProc output empty: {}", postProcInputEmpty,
                    postProcOutputEmpty);
    m_logger->debug("isProcessingComplete: m_bDone: {}, YOLO done: {}, PostProc done: {}", m_bDone,
                    m_yolov8->m_bDone, m_yolov8PostProc->m_bDone);

    return yoloInputEmpty && yoloPredEmpty && postProcInputEmpty && postProcOutputEmpty && m_bDone
            && m_yolov8->m_bDone && m_yolov8PostProc->m_bDone;
}


bool WNC_APP::_objectDetection()
{
    m_humanBBoxList    = std::move(m_procResult.humanBBoxList);
    m_vehicleBBoxList  = std::move(m_procResult.vehicleBBoxList);
    m_faceBBoxList  = std::move(m_procResult.faceBBoxList);
    m_riderBBoxList = std::move(m_procResult.riderBBoxList);

    m_logger->debug("[{}] Num of Human Bounding Box: {}", __func__, static_cast<int>(m_humanBBoxList.size()));
    m_logger->debug("[{}] Num of Vehicle Bounding Box: {}", __func__, static_cast<int>(m_vehicleBBoxList.size()));
    m_logger->debug("[{}] Num of Face Bounding Box: {}", __func__, static_cast<int>(m_faceBBoxList.size()));
    m_logger->debug("[{}] Num of Rider Bounding Box: {}", __func__, static_cast<int>(m_riderBBoxList.size()));

    for (auto& obj : m_humanBBoxList)
        m_logger->debug("Human Object: {}, {}, {}, {}, label: {}, c: {}", obj.x1, obj.y1, obj.x2, obj.y2, obj.label,
                        obj.confidence);

    for (auto& obj : m_vehicleBBoxList)
        m_logger->debug("Vehicle Object: {}, {}, {}, {}, label: {}, c: {}", obj.x1, obj.y1, obj.x2, obj.y2, obj.label,
                        obj.confidence);
    
    for (auto& obj : m_faceBBoxList)
        m_logger->debug("Vehicle Object: {}, {}, {}, {}, label: {}, c: {}", obj.x1, obj.y1, obj.x2, obj.y2, obj.label,
                        obj.confidence);
    
    for (auto& obj : m_riderBBoxList)
        m_logger->debug("Vehicle Object: {}, {}, {}, {}, label: {}, c: {}", obj.x1, obj.y1, obj.x2, obj.y2, obj.label,
                        obj.confidence);

    return true;
}

bool WNC_APP::_poseDetection()
{
    m_logger->debug("[{}] Num of Skeleton Bounding Box: {}", __func__,
                    static_cast<int>(m_procResult.skeletonBBoxList.size()));

    // for (auto& obj : m_procResult.skeletonBBoxList)
    for (size_t i=0; i<m_procResult.skeletonBBoxList.size(); i++) 
    {
        // Classify skeleton action using std::pair<int,int> keypoints
        // obj.skeletonAction = m_skeleton_postproc.classifySkeleton(obj);
        m_procResult.skeletonBBoxList[i].skeletonAction = m_skeleton_postproc.classifySkeleton(m_procResult.skeletonBBoxList[i],
                                                                                               m_skeleton_postproc.m_prevPoseBoundingboxList[i]);

        // Logging
        BoundingBox obj = m_procResult.skeletonBBoxList[i];
        m_logger->error("Skeleton Object: {}, {}, {}, {}, label: {}, action: {}, c: {}",
                        obj.x1, obj.y1, obj.x2, obj.y2, obj.label,
                        obj.skeletonAction, obj.confidence);
    }
    m_skeleton_postproc.updatePrevBoundingboxList(m_procResult.skeletonBBoxList);
    return true;
}


bool WNC_APP::_objectTracking()
{
    m_f_humanMainLaneBboxList   = m_humanBBoxList;
    m_f_vehicleMainLaneBboxList = m_vehicleBBoxList;

    // === Define central ROI (288x512 image) ===
    int imgW = 512, imgH = 288;
    int roi_w = int(m_roi.x2 - m_roi.x1);
    int roi_h = int(m_roi.y2 - m_roi.y1);
    m_logger->debug("m_roi.x1={} , m_roi.y1={}, m_roi.x2={}, m_roi.y2={}",m_roi.x1,m_roi.y1,m_roi.x2,m_roi.y2);
    cv::Rect roi(m_roi.x1, m_roi.y1, roi_w, roi_h);
    // cv::Rect roi(int(imgW / 4.0), int(imgH / 4.0), int(imgW/2.0), int(imgH/2.0));  
    // central 256x144 region
    // === Keep only bboxes inside ROI ===
    auto filterInsideROI = [&](std::vector<BoundingBox>& bboxes, const cv::Rect& roi) {
        bboxes.erase(
            std::remove_if(bboxes.begin(), bboxes.end(),
                           [&](const BoundingBox& box) {
                               cv::Rect boxRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
                               return ((boxRect & roi).area() == 0);  // remove if no overlap
                           }),
            bboxes.end());
    };

    filterInsideROI(m_f_humanMainLaneBboxList, roi);

    // === Keep only top 5 largest inside ROI ===
    auto keepTopN = [](std::vector<BoundingBox>& bboxes, int N) {
        std::sort(bboxes.begin(), bboxes.end(),
                  [](const BoundingBox& a, const BoundingBox& b) {
                      return a.getArea() > b.getArea();
                  });
        if (bboxes.size() > static_cast<size_t>(N)) {
            bboxes.resize(N);
        }
    };

    keepTopN(m_f_humanMainLaneBboxList, m_config->stTrackerConifg.maxTracking);

    // Debug Logs
    m_logger->debug("[{}] Before Track: Human in ROI {}", __func__,
                    static_cast<int>(m_f_humanMainLaneBboxList.size()));

    // Run Object Tracking (only humans inside ROI)
    m_humanTracker->run(m_f_humanMainLaneBboxList, m_humanObjUmap, m_humanObjList);

    // Merge Tracked Objects
    m_trackedObjList.clear();
    m_trackedObjList.insert(m_trackedObjList.end(),
                            std::make_move_iterator(m_humanObjList.begin()),
                            std::make_move_iterator(m_humanObjList.end()));

    // Debug Logs
    m_logger->debug("[{}] After Track: {} Human", __func__,
                    static_cast<int>(m_humanObjList.size()));

    return true;
}


// ============================================
//                  Results
// ============================================
void WNC_APP::_showDetectionResults()
{
    // Show Tracked Object Information
    for (auto& obj : m_trackedObjList)
    {
        if (obj.getStatus() == 1)
        {
            std::string classType = "";
            if (obj.bbox.label == HUMAN)
                classType = "Pedestrian";
            else if (obj.bbox.label == BIG_VEHICLE)
                classType = "Vehicle";

            float ttc = obj.needWarn ? obj.currTTC : -1.0f;

            m_logger->debug(
                "[{}] Tracking Obj[{}]: Cls = {}, Conf = {:.2f}, needWarn = {}", __func__,
                obj.id, classType, obj.bbox.confidence, obj.needWarn);
        }
    }
}

void WNC_APP::_drawResults()
{
    std::vector<std::function<void()>> drawFunctions;

    if (m_dsp_objectDetection)
        drawFunctions.emplace_back([this]() { _drawBoundingBoxes(); });

    if (m_dsp_objectTracking)
        drawFunctions.emplace_back([this]() { _drawTrackedObjects(); });

    if (m_dsp_poseDetection)
        drawFunctions.emplace_back([this]() { _drawPosePoints(); });

    for (const auto& drawFunc : drawFunctions)
        drawFunc();

    cv::resize(m_displayImg, m_dsp_imgResize, cv::Size(m_frameWidth, m_frameHeight),cv::INTER_LINEAR); // Width must be mutiplication of 4

    // TODO: Not usr Ambarella using RGBA, so we need to convert it to RGB
    // #ifdef SAV837
    //     cv::cvtColor(m_dsp_imgResize, m_dsp_imgResize, cv::COLOR_RGBA2RGB);
    // #endif

    if (m_dsp_information)
        _drawInformation();

// #if defined(QCS6490) || defined(SHOW_IMAGES)
    if (m_dsp_results)
    {
        cv::imshow("WNC-APP", m_dsp_imgResize);
        cv::waitKey(20);
    }
// #endif
}

void WNC_APP::_drawBoundingBoxes()
{
    std::vector<std::vector<BoundingBox>*> boundingBoxLists = {&m_humanBBoxList, &m_vehicleBBoxList, &m_faceBBoxList};

    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 128, 255), // Orange for humans
        cv::Scalar(255, 0, 255),  // Magenta for vehicles
        cv::Scalar(0, 0, 255)  // Magenta for face
    };
    std::vector<int> bb_thicknesss = {1, 1, 2};

    for (size_t i = 0; i < boundingBoxLists.size(); ++i)
    {
        auto&       boundingBoxList = *boundingBoxLists[i];
        cv::Scalar& color           = colors[i];
        int bb_thickness = bb_thicknesss[i];

        for (auto& boundingBox : boundingBoxList)
            _drawRoundedRectangle(boundingBox, color, bb_thickness);
    }
}

void WNC_APP::_drawTrackedObjects()
{
    const float EPSILON_DISTANCE = 0.1f;

    cout << "m_trackedObjList.size() = " << m_trackedObjList.size() << endl;

    for (auto& obj : m_trackedObjList)
    {
        if (obj.getStatus() == 0 || obj.bboxList.empty())
            continue;

        BoundingBox& lastBox = obj.bboxList.back();
        BoundingBox  rescaleBox(-1, -1, -1, -1, -1);

        utils::rescaleBBox(lastBox, rescaleBox, MODEL_WIDTH, MODEL_HEIGHT, m_frameWidth, m_frameHeight);

        // Color based on track ID (BGR)
        int id = obj.id;
        cv::Scalar rectColor(
            static_cast<double>((id * 70) % 256),  // Blue
            static_cast<double>((id * 140) % 256), // Green
            static_cast<double>((id * 30) % 256)   // Red
        );


        imgUtil::efficientRectangle(m_displayImg, cv::Point(rescaleBox.x1, rescaleBox.y1),
            cv::Point(rescaleBox.x2, rescaleBox.y2), rectColor, 2, cv::LINE_AA, 10, false);

        // Draw text background
        cv::rectangle(m_displayImg, 
            cv::Point(rescaleBox.x1 + 10, rescaleBox.y1 - 20),
            cv::Point(rescaleBox.x1 + 30, rescaleBox.y1 - 3), 
            cv::Scalar(0, 0, 0), -1 /* fill */);

        // Draw track ID text
        cv::putText(m_displayImg,
            std::to_string(obj.id),
            cv::Point(int(rescaleBox.x1) + 10, int(rescaleBox.y1) - 10), 
            cv::FONT_HERSHEY_DUPLEX, 0.4,
            cv::Scalar(255, 255, 255), 1, 5, 0);
    }
}

void WNC_APP::_drawPosePoints()
{
    m_skeletonBBoxList  = std::move(m_procResult.skeletonBBoxList);
    // cout << "[_drawPosePoints] m_skeletonBBoxList size: " << m_skeletonBBoxList.size() << endl;
    std::vector<std::vector<BoundingBox>*> boundingBoxLists = {&m_skeletonBBoxList};
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255), // Orange for white line
        cv::Scalar(0, 255, 255), // Magenta for cross road line
        cv::Scalar(255, 255, 0),   // Green for yellow line
        cv::Scalar(0, 128, 255)    // Red for road curb line
    };
    cout<<"[_drawPoseKeyPoints] boundingBoxLists.size() = "<<boundingBoxLists.size()<<endl;
    for (size_t i = 0; i < boundingBoxLists.size(); ++i)
    {
        auto&       boundingBoxList = *boundingBoxLists[i];
        cv::Scalar& color           = colors[i]; 
        //Using a reference (&) makes it clear that no additional copies are being made 
        //and the function directly operates on colors[i].

        for (auto& boundingBox : boundingBoxList)
        {
            // cout << "[_drawPosePoints] BoundingBox [x1=" << boundingBox.x1 << ", y1=" << boundingBox.y1 
            //      << ", x2=" << boundingBox.x2 << ", y2=" << boundingBox.y2 << "]" << endl;

            // Log the pose keypoints for this bounding box
            // cout << "[_drawPosePoints] Pose keypoints: ";
            // if (boundingBox.pose_kpts.empty())
            // {
            //     cout << "No keypoints available.";
            // }
            // else
            // {   
            //     cout<<boundingBox.skeletonAction<<" ";
            //     for (const auto& kpt : boundingBox.pose_kpts)
            //     {
            //         cout << "(" << kpt.first << ", " << kpt.second << ") ";
            //     }
            // }
            // cout << endl;
            _drawPoseKeyPoints(boundingBox, color);
        }         
    }
}

void WNC_APP::_drawPoseKeyPoints(BoundingBox& box, const cv::Scalar& color)
{
    // BoundingBox rescaledBox(-1, -1, -1, -1, -1);
    std::vector<std::pair<int, int>> emptyKeypoints; // Empty vector
    std::string pose_type;
    BoundingBox rescaledBox(-1, -1, -1, -1, -1, emptyKeypoints, pose_type);
    utils::rescaleBBox(box, rescaledBox, MODEL_WIDTH, MODEL_HEIGHT, m_frameWidth, m_frameHeight);
    imgUtil::PoseKeyPoints(m_displayImg, rescaledBox.pose_kpts,color, 2);
    imgUtil::drawSkeletonAction(m_displayImg, rescaledBox, color);
    // imgUtil::roundedRectangle(m_displayImg, cv::Point(rescaledBox.x1, rescaledBox.y1),
    //                           cv::Point(rescaledBox.x2, rescaledBox.y2), color, 2, 0, 5, false);
}

void WNC_APP::_drawInformation()
{
    // Draw static information
    const int x_offset      = 10;
    const int y_offset      = 20;
    const int y_offsetStep  = 20;
    std::string versionStr  = "Version: v" + std::string(APP_VERSION);
    imgUtil::drawDebugText(m_dsp_imgResize, versionStr, cv::Point(x_offset, y_offset));

    std::string frameStr    = "Frame: " + std::to_string(m_resultFrameIdx);
    imgUtil::drawDebugText(m_dsp_imgResize, frameStr, cv::Point(x_offset, y_offset + y_offsetStep * 1));

    // // Camera Height
    // std::stringstream cameraHeightSS;
    // cameraHeightSS << std::fixed << std::setprecision(2) << m_config->stCameraConfig.height;
    // std::string cameraHeightStr = "Camera Height: " + cameraHeightSS.str();
    // imgUtil::drawDebugText(m_dsp_imgResize, cameraHeightStr, cv::Point(x_offset, y_offset + y_offsetStep * 2));
}

void WNC_APP::_saveDrawResults()
{
    string imgName = "frame_" + std::to_string(m_resultFrameIdx) + ".jpg";
    string imgPath = m_dbg_imgsDirPath + "/" + imgName;

    cv::imwrite(imgPath, m_dsp_imgResize);

    m_logger->debug("Save img to {}", imgPath);
}

void WNC_APP::_saveRawImages()
{
    string imgName = "frame_" + std::to_string(m_resultFrameIdx) + ".jpg";
    string imgPath = m_dbg_rawImgsDirPath + "/" + imgName;

    cv::imwrite(imgPath, m_resultImage);

    m_logger->debug("Save raw img to {}", imgPath);
}

void WNC_APP::_getResultImage(cv::Mat& imgResult)
{
    if (m_dsp_results)
        imgResult = std::move(m_dsp_imgResize);
}

bool WNC_APP::getAppResult(WNC_APP_Results& res)
{
    std::unique_lock<std::mutex> lock(m_resultBufferMutex);
    if (!m_resultBuffer.empty())
    {
        res = m_resultBuffer.front();
        m_resultBuffer.pop_front();
        return true;
    }
    else
        return false;
}

void WNC_APP::_updateAppResult()
{
    std::unique_lock<std::mutex> lock(m_resultBufferMutex);

    WNC_APP_Results result; //TODO: Not implemented yet

    // Save Event Results
    result.eventType = _getDetectEvents();

    result.detectObjList = std::move(m_humanBBoxList); // Save Detected Objects
    result.trackObjList = std::move(m_trackedObjList); // Save Tracked Objects
    result.faceObjList = std::move(m_faceBBoxList); // Save Face Objects
    result.riderObjList = std::move(m_riderBBoxList); // Save Rider Objects
    result.vehicleObjList = std::move(m_vehicleBBoxList); // Save Vehicle Objects
    result.poseObjList = std::move(m_skeletonBBoxList); // Save Pose Objects

    result.frameID = m_resultFrameIdx;

    m_result_log = result;
    m_resultBuffer.push_back(result);

    // Keep only the most recent MAX_RESULT_BUFFER_SIZE entries
    if (m_resultBuffer.size() > MAX_RESULT_BUFFER_SIZE) m_resultBuffer.pop_front();


    m_debugProfile.adasResultBufferSize = m_resultBuffer.size();
    lock.unlock();
}

void WNC_APP::_updateDebugProfile()
{
    float inferenceTime    = 0.0f;
    int   inputBufferSize  = 0;
    int   outputBufferSize = 0;
    m_yolov8->getDebugProfiles(m_debugProfile.AIInfrerenceTime, m_debugProfile.yoloADAS_InputBufferSize,
                                 m_debugProfile.yoloADAS_OutputBufferSize);
    m_yolov8PostProc->getDebugProfiles(m_debugProfile.postProc_InputBufferSize,
                                         m_debugProfile.postProc_OutputBufferSize);
}

int WNC_APP::_getDetectEvents()
{
    return 0; //TODO:
}

void WNC_APP::_drawRoundedRectangle(BoundingBox& box, const cv::Scalar& color, int bb_thickness)
{
    BoundingBox rescaledBox(-1, -1, -1, -1, -1);
    utils::rescaleBBox(box, rescaledBox, MODEL_WIDTH, MODEL_HEIGHT, m_frameWidth, m_frameHeight);
    imgUtil::roundedRectangle(m_displayImg, cv::Point(rescaledBox.x1, rescaledBox.y1),
                              cv::Point(rescaledBox.x2, rescaledBox.y2), color, bb_thickness, 0, 10, false); // 1,0,10
}

void WNC_APP::updateFrameIndex()
{
    m_frameIdx_wnc = (m_frameIdx_wnc % 4294967295) + 1;
}

void WNC_APP::_saveDetectionResult(std::vector<std::string>& logs)
{
    auto logger = spdlog::get("APP_DEBUG");

    for (auto& log : logs)
        logger->info(log);
}


void WNC_APP::wakeProcessingThread()
{
    m_queueCV.notify_one();
}

//TODO: Not implemented yet
#ifdef SAV837
int WNC_APP::run_sequential(WNC_APP_Results& appResult, cv::Mat& imgFrame, string& imagePath, const int frameId)
{
    auto time_0 = m_estimateTime ? std::chrono::high_resolution_clock::now()
                                 : std::chrono::time_point<std::chrono::high_resolution_clock>{};

    if (imgFrame.empty())
    {
        printf("imgFrame is empty \n");
        return -1;
    }

    // Alsiter add 2024-07-31
    std::string server_ip   = m_config->HistoricalFeedModeConfig.serverIP;
    int         server_port = m_config->HistoricalFeedModeConfig.serverPort;

    // =========================== //
    // Input Image Processing
    // =========================== //
    cv::cvtColor(imgFrame, imgFrame, cv::COLOR_RGB2RGBA);

    int im_height = imgFrame.rows;
    int im_width  = imgFrame.cols;

    if (im_height != MODEL_HEIGHT || im_width != MODEL_WIDTH)
    {
        cv::resize(imgFrame, imgFrame, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    }

    cv::Mat inputImage = imgFrame.clone();

    // =========================== //
    // AI Inference
    // =========================== //
    YOLOv8_Prediction pred;
    if (!m_yolov8->run_sequential(inputImage, pred))
    {
        return -1;
    }

    if (!m_yolov8PostProc->run_sequential(pred, m_procResult))
    {
        return -1;
    }

    //---------------------------------------------------------------------------------------------

    m_logger->debug("");
    m_logger->debug("========================================");
    m_logger->debug("Frame Index: {}", frameId);
    m_logger->debug("========================================");

    bool isForCalibrationAI = false;
    int success = _objectDetection() && _objectTracking()
                  ? APP_SUCCESS
                  : APP_FAILURE;

    if (success == APP_FAILURE)
        m_logger->warn("One or more of APP sub-tasks failed");

    _showDetectionResults();

    // Draw and Save Results
    if (m_dsp_results && m_dbg_saveImages && success == APP_SUCCESS)
    {
        cv::resize(m_procResult.img, m_displayImg, cv::Size(m_frameWidth, m_frameHeight), cv::INTER_LINEAR);
        _drawResults();
        if (m_dbg_saveImages)
            _saveDrawResults();
    }

    // =========================== //
    // Save Results to adasResult
    // =========================== //
    WNC_APP_Results result; //TODO:
    result.isDetectLine      = m_isDetectLine;
    // Save Tracked Objects
    result.objList = m_trackedObjList;
    m_debugProfile.adasResultBufferSize = 0;

    // =========================== //
    // Save Log to Json File
    // =========================== //
    // //TODO:
    // std::string jsonLog = m_jsonLog->logInfo(
    //     result, m_laneLineBox.dcaBBoxList, m_laneLineBox.dlaBBoxList, m_laneLineBox.dmaBBoxList,
    //     m_laneLineBox.duaBBoxList, m_humanBBoxList, m_vehicleBBoxList, m_roadSignBBoxList, m_tailingObject,
    //     m_trackedObjList, frameId, m_debugProfile, egoVelocity, m_smoothValue, m_statusCode,
    //     APP_VERSION);

    // Copy result to adasResult
    appResult = result;


    if (m_estimateTime)
    {
        auto time_1 = std::chrono::high_resolution_clock::now();
        m_logger->info("");
        m_logger->info("Processing Time: {} ms",
                       std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
    }

    return 0;
}
#endif


#ifdef SAV837
// TODO: not implemented yet
// ======================================= //
//          Historical Mode Entry          //
// ======================================= //
void WNC_APP::_initializeModulesHistorical(std::shared_ptr<spdlog::logger>& logger, std::string inputFile)
{
    m_logger->info("Init Historical feed");

    // in historical feed
    // calibration and yolo adas are EXCLUSIVE
    // inputMode == 1: ONLY run calibration
    // inputMode == 2: ONLY run yolo adas

    // IO
    m_frameWidth         = m_config->frameWidth;
    m_frameHeight        = m_config->frameHeight;
    m_dbg_imgsDirPath    = m_config->stDebugConfig.imgsDirPath;
    m_dbg_rawImgsDirPath = m_config->stDebugConfig.rawImgsDirPath;
    m_dbg_logsDirPath    = m_config->stDebugConfig.logsDirPath;

    m_objectDetectionROI.x1    = m_frameWidth * 0.0;
    m_objectDetectionROI.y1    = m_frameHeight * 0.0;
    m_objectDetectionROI.x2    = m_frameWidth * 1.0;
    m_objectDetectionROI.y2    = m_frameHeight * 1.0;
    m_objectDetectionROI.label = -1;

    if (m_objectDetectionROI.getArea() <= 0)
    {
        m_logger->error("Detection ROI area <= 0");
        exit(1);
    }

    m_logger->info("InputMode: {}", m_config->HistoricalFeedModeConfig.inputMode);
    
    if (m_config->HistoricalFeedModeConfig.inputMode == 2)  // Historical APP Mode
    {
        m_logger->info("APP mode");
        // InputMode == 2, ONLY run ADAS
        // Avoid initializing/calling m_calibrationAI and m_calibration

        std::string imagedir = m_config->HistoricalFeedModeConfig.rawImageDir;

        string image_name_prefix = imagedir + '/' + m_historicalFrameName + std::to_string(m_counters);
        string imagePath;
        string bmpPath = image_name_prefix + ".bmp";
        string pngPath = image_name_prefix + ".png";
        m_logger->info("supports .bmp and .png");
        m_logger->info("RawImageDir: {}\n", imagedir);
        m_logger->info("ImageModeFrameName: {}\n", m_historicalFrameName);
        m_logger->info("ImageModeStartFrame: {}\n", m_config->HistoricalFeedModeConfig.imageModeStartFrame);
        imagePath = bmpPath;
        std::ifstream file(imagePath);
        if (!file.is_open())
        {
            imagePath = pngPath;
            std::ifstream file1(imagePath);
            if (!file1.is_open())
            {
                m_logger->info("{} not found", bmpPath);
                m_logger->info("{} not found", pngPath);
                m_logger->info("exit historical mode\n");
                m_logger->info("\n");
                exit(1);
            }
        }

        m_yolov8 = new YOLOv8(m_config, [this]() { this->wakeProcessingThread(); });
        m_yolov8PostProc =
            new YOLOv8_POSTPROC(m_config, m_objectDetectionROI, [this]() { this->wakeProcessingThread(); });

    }
    m_logger->debug("Initialized Post Processing module");


    _initTrackingROI();

    inputFile += ".log";
    // Initialize other modules
    m_logger->warn("Output log file {}", inputFile);
    m_jsonLog = new JSON_LOG(inputFile, m_config);
    m_socket  = new SOCKET();
    m_logger->debug("Initialized AI model and other modules");

    m_humanTracker   = new ObjectTracker(m_config, "human");
    m_vehicleTracker = new ObjectTracker(m_config, "vehicle");
    m_trackedObjList.reserve(m_humanTracker->m_maxTracking * 2);
    m_logger->debug("Initialized Human and Vehicle Tracker modules");
}

// // This is IPU simulator helper function
// #if defined(CV28_SIMULATOR)
// int WNC_APP::ipuProcess(YOLOv8_Prediction& prediction, int frameIdx)
// {
//     int              success = APP_FAILURE;
//     WNC_APP_Results  appResult;
//     std::string      server_ip   = m_config->HistoricalFeedModeConfig.serverIP;
//     int              server_port = m_config->HistoricalFeedModeConfig.serverPort;

//     auto time_0 = m_estimateTime ? std::chrono::high_resolution_clock::now()
//                                  : std::chrono::time_point<std::chrono::high_resolution_clock>{};

//     // Get last prediction
//     m_yolov8PostProc->updatePredictionBuffer(prediction, frameIdx);

//     while (!m_yolov8PostProc->getLastestResult(m_procResult, m_resultFrameIdx))
//     {
//         std::this_thread::sleep_for(std::chrono::milliseconds(10));
//     }

//     m_displayImg = prediction.img.clone();

//     updateFrameIndex();

//     bool isForCalibrationAI = false;
//     success = _objectDetection() &&  _objectTracking()
//                   ? APP_SUCCESS
//                   : APP_FAILURE;

//     if (success == APP_FAILURE)
//         m_logger->warn("One or more of App sub-tasks failed");

//     _showDetectionResults();

//     if (success == APP_SUCCESS)
//     {
//         cv::resize(m_procResult.img, m_displayImg, cv::Size(m_frameWidth, m_frameHeight), cv::INTER_LINEAR);
//         _drawResults();
//         if (m_dbg_saveImages)
//             _saveDrawResults();
//     }

//     m_resultFrameIdx = frameIdx;
//     _updateDebugProfile();
//     _updateAppResult(); //TODO:

//     // //TODO:
//     // std::string jsonLog = m_jsonLog->logInfo(
//     //     m_resultBuffer.back(), m_laneLineBox.dcaBBoxList, m_laneLineBox.dlaBBoxList, m_laneLineBox.dmaBBoxList,
//     //     m_laneLineBox.duaBBoxList, m_humanBBoxList, m_vehicleBBoxList, m_roadSignBBoxList, m_tailingObject,
//     //     m_trackedObjList, m_resultFrameIdx, m_debugProfile, m_smoothValue, m_statusCode,
//     //     APP_VERSION);

//     // _sendDataUsingSocket(success, jsonLog, server_ip, server_port, m_resultFrameIdx);

//     success = APP_LOGGED;

//     if (m_estimateTime)
//     {
//         auto time_1 = std::chrono::high_resolution_clock::now();
//         m_logger->info("Processing Time: {} ms",
//                        std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
//     }

//     return success;
// }
// #endif
#endif
