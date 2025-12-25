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
#ifndef __APP__
#define __APP__

#include <vector>
#include <string>
#include <cstdio>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <fstream>

// Ambarella
#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

// WNC
#include "dataStructures.h"
#include "dla_config.hpp"
#include "json_log.hpp"
#include "yolov8.hpp"
#include "yolov8_postproc.hpp"
#include "utils.hpp"
#include "object_tracker.hpp"
#include "logger.hpp"
#include "socket.hpp"
#include "skeleton_postproc.hpp"

using namespace std;

#define APP_VERSION "0.0.6"
constexpr int APP_SUCCESS = -3;
constexpr int APP_FAILURE = -4;
constexpr int APP_LOGGED  = -5;


class WNC_APP
{
public:
    WNC_APP(const std::string configPath, std::string inputFile);
    ~WNC_APP();
    
    // Function to get the current ADAS result log (for the latest frame)
    const WNC_APP_Results& getResultLog() const {
        return m_result_log;
    }

    // WNC_APP app_instance;
    // const WNC_APP_Results& log = app_instance.getResultLog();
    // Use log as needed, e.g., print or inspect values

    // === Config === //
    AppConfigReader* m_appConfigReader;
    Config_S*     m_config;

    // Thread
    void stopThread();
    void addFrame(ea_tensor_t* imgTensor);
    void startProcessingThread();
    void stopProcessingThread();
    bool isProcessingComplete();
    bool getAppResult(WNC_APP_Results &result);
    int ipuProcess(YOLOv8_Prediction& prediction, int frameIdx);
    int appProcess(cv::Mat& resultMat);
    void printObjectList(const std::vector<Object>& objList, const std::shared_ptr<spdlog::logger>& logger);
  
    // === Work Flow === //
    bool _objectDetection();
    bool _objectTracking();
    bool _poseDetection();

    // === Output === //
    int  _getDetectEvents();
    void _getResultImage(cv::Mat& imgResult);

    // === Utils === //
    void updateFrameIndex();

    // === Results === //
    void _drawResults();
    void _saveRawImages();
    void _showDetectionResults();
    void _saveDrawResults();

    // TODO: Not finished yet
    int run_sequential(WNC_APP_Results& adasResult, cv::Mat& imgFrame, string& imagePath, const int frameId);
    void _sendImageOrLog(cv::Mat img, const std::string& jsonLog, const std::string& server_ip, int server_port, int frameIdx); 
    void _sendDataLiveMode(int success, const std::string& jsonLog, const std::string& server_ip, int server_port, int frameIdx);
    void _sendDataHistoricalMode(int visualizeMode, const std::string& imagePath, const std::string& jsonLog,
                                    const std::string& server_ip, int server_port, int frameId);

    void _printAppResult(const WNC_APP_Results& res);
    std::string m_dbg_imgsDirPath    = "";
	std::string m_dbg_logsDirPath;

    // Alister add 2025-05-09
    // unordered_map<int, Object> m_humanObjUmap;
    // unordered_map<int, Object> m_vehicleObjUmap;

#ifdef SAV837_SIMULATOR
    static void dropLogger() 
    {
        spdlog::drop("App");
    }
#endif

    std::function<void()> m_frameProcessedCallback;
    void setFrameProcessedCallback(std::function<void()> callback) 
    {
        m_frameProcessedCallback = std::move(callback);
    }

    std::condition_variable m_queueCV;
    void wakeProcessingThread(); 

    // === Frame === //
    unsigned long m_frameIdx_wnc        = 0;
    unsigned long m_frameNum            = 0;
    int           m_frameStep_wnc       = 1;
    int           m_resultFrameIdx      = 0;
    int           m_counters            = 0;
    int           m_max_counters        = 99999;
    std::string   m_historicalFrameName = "RawFrame_";

    cv::Mat m_displayImg;

    // === Display === //
    bool m_dsp_results       = true;
    bool m_dbg_saveImages    = false;
    bool m_dbg_saveRawImages = false;
    bool m_estimateTime      = false;
    
    DebugProfile m_debugProfile;

    cv::Mat             m_resultImage;
    YOLOv8_POSTPROC*    m_yolov8PostProc = nullptr;
    YOLOv8*             m_yolov8 = nullptr;
    JSON_LOG*           m_jsonLog;
    POST_PROC_RESULTS   m_procResult;
    Object              m_tailingObject;
    SOCKET*             m_socket;
    WNC_APP_Results     m_appResult;
    SKELETON_POSTPROC   m_skeleton_postproc;

    // Output Bounding Boxes from AI
    std::vector<BoundingBox> m_humanBBoxList;
    std::vector<BoundingBox> m_vehicleBBoxList;
    std::vector<BoundingBox> m_faceBBoxList;
    std::vector<BoundingBox> m_riderBBoxList;
    std::vector<BoundingBox> m_skeletonBBoxList;    
    // Tracked Objects
    std::vector<Object> m_trackedObjList;

protected:
// TODO: change to ambarella's method
#ifdef SAV837
    // === IPU related === //
    MI_S32 IpuInit();
    MI_S32 IpuDeInit();
    MI_S32 IpuGetSclOutputPortParam(void);
#endif

private:

    std::shared_ptr<spdlog::logger> _initializeLogger();
    void _initializeConfig(const std::string& configPath, std::shared_ptr<spdlog::logger>& logger);
    void _checkPaths(std::shared_ptr<spdlog::logger>& logger);
    void _initializeModules(std::shared_ptr<spdlog::logger>& logger, std::string inputFile);
    void _initializeModulesHistorical(std::shared_ptr<spdlog::logger>& logger, std::string inputFile);
    void _setupDebugAndDisplayConfigs();
    void _startThreads();

    void _init(std::string configPath, std::string inputFile);
    void _initTrackingROI();
    void _readDebugConfig();
    void _readDisplayConfig();
    void _saveDetectionResult(std::vector<std::string>& logs);
    // void _startHistoricalFeedThread(); // TODO: Not implemented yet

    void updateInput(ea_tensor_t* imgTensor);
    void _updateAppResult();
    void _updateDebugProfile();

    void _drawBoundingBoxes();
    void _drawTrackedObjects();
    void _drawPosePoints();
    void _drawPoseKeyPoints(BoundingBox& box, const cv::Scalar& color);
    void _drawInformation();
    void _drawRoundedRectangle(BoundingBox& box, const cv::Scalar& color, int bb_thickness);
    bool _hasWorkToDo();

    void _createTensorDirectories(std::string inputFile);

    void _printBoundingBox(const BoundingBox& box, int idx = -1);
    void _printObject(const Object& obj, int idx = -1);

    int m_frameWidth  = 0; // Alister
    int m_frameHeight = 0; // Alister

    bool m_bDone = false;
    

    std::mutex m_resultBufferMutex;

    std::mutex m_queueMutex;
    std::atomic<bool> m_processingThreadRunning{false};
    std::thread m_processingThread;

    std::vector<Object> m_humanObjList;
    std::vector<Object> m_vehicleObjList;

    unordered_map<int, Object> m_humanObjUmap;
    unordered_map<int, Object> m_vehicleObjUmap;

    // Output Bounding Boxes after filtering by ROI
    std::vector<BoundingBox> m_f_humanMainLaneBboxList;
    std::vector<BoundingBox> m_f_vehicleMainLaneBboxList;

    // === Object Tracker === //
    ObjectTracker* m_humanTracker;
    ObjectTracker* m_vehicleTracker;

    // === ROI === //
    ROI         m_roi;
    BoundingBox m_roiBBox;
    BoundingBox m_objectDetectionROI;
    BoundingBox m_objecttrackingROI;

    // === Result === //
    std::deque<WNC_APP_Results> m_resultBuffer;
    WNC_APP_Results m_result_log;

    // === Display === //
    cv::Mat m_dsp_imgResize;

    int m_newXStart              = -1;
    int m_newXEnd                = -1;
    int m_newYStart              = -1;
    int m_newYEnd                = -1;

    int m_maxTracking = 10;
    
    float m_thresholdFps         = 0.0f;
    bool m_dsp_objectDetection   = false;
    bool m_dsp_objectTracking    = false;
    bool m_dsp_information       = false;
    bool m_dsp_poseDetection     = true;
    
    // === Debug === //
    bool          m_dbg_app      = false;
    std::string   m_dbg_rawImgsDirPath = "";
    std::string   m_dbg_dateTime       = "";
    LoggerManager m_loggerManager;

    std::shared_ptr<spdlog::logger> m_logger; // Store logger as a member variable
    static constexpr size_t MAX_RESULT_BUFFER_SIZE = 10;  // configurable buffer limit
};
#endif
