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

#include "config_reader.hpp"

AppConfigReader::AppConfigReader()
{
    m_config = new Config_S();
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::syslog_logger_mt("config", "config", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
    auto logger = spdlog::stdout_color_mt("AppConfigReader");
    logger->set_pattern("[%n] [%^%l%$] %v");
#endif
};

AppConfigReader::~AppConfigReader()
{
#ifdef SPDLOG_USE_SYSLOG
    spdlog::drop("config");
#else
    spdlog::drop("AppConfigReader");
#endif
};

Config_S *AppConfigReader::getConfig()
{
    return m_config;
}

void AppConfigReader::read(std::string configPath)
{
    // Create object of the class AppConfigReader
    ConfigReader *configReader = ConfigReader::getInstance();
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("config");
#else
    auto logger = spdlog::get("AppConfigReader");
#endif

    if (utils::checkFileExists(configPath) && configReader->parseFile(configPath))
    {
        // Print divider on the console to understand the output properly
        logger->info("=================================================");

        // Historical Feed Settings
        int    inputMode           = 0; // 0:Sensor , 1:Video, 2:Image
        string videoPath           = "";
        string rawImageDir         = "";
        string calibRawImageDir    = "";
        int    imageModeStartFrame = 0;
        int    imageModeEndFrame   = 0;
        string imageModeFrameName  = "RawFrame_";
        int    serverPort          = 4800;
        string serverIP            = "";
        int    visualizeMode       = 0;

        // Model Information
        string modelPath        = "";
        string fnetModelPath    = "";  // DPVO FNet model path
        string inetModelPath    = "";  // DPVO INet model path
        string updateModelPath  = "";  // DPVO Update model path
        int    modelWidth           = 320;
        int    modelHeight          = 320;

        // Model ROI Information
        string startXRatio = "0.0";
        string endXRatio   = "1.0";
        string startYRatio = "0.0";
        string endYRatio   = "1.0";
        int    enableROI   = 0;

        // Camera Information
        string cameraHeight      = "";
        // string cameraFocalLength = "";
        string intrinsic_fx      = "";
        string intrinsic_fy      = "";
        string intrinsic_cx      = "";
        string intrinsic_cy      = "";
        string distortion_k1     = "";
        string distortion_k2     = "";
        string distortion_p1     = "";
        string distortion_p2     = "";
        int    frameWidth        = 320;
        int    frameHeight       = 320;

        // Calibration
        int yVanish           = 0;
        int forceYVanish      = 0;
        int calibrationMode   = 0;
        int forceCalibYVanish = 0;

        // Processing Time
        string procFrameRate = "";
        int    procFrameStep = 4;

        // Object Detection
        string humanConfidence    = "";
        string carConfidence      = "";

        // Object Tracking
        int maxTracking = 3;

        // Debug Information
        int debugConfig             = 0;
        int debugApp                = 0;
        int debugAIPostProc         = 0;
        int debugAIModel            = 0;
        int debugObjectDetection    = 0;
        int debugObjectTracking     = 0;
        int debugEnableJson         = 0;
        int debugSaveLogs           = 0;
        int debugSaveImages         = 0;
        int debugSaveRawImages      = 0;

        string debugLogsDirPath      = "";
        string debugImagesDirPath    = "";
        string debugRawImagesDirPath = "";

        // Display Results
        int displayResults           = 0;
        int displayObjectDetection   = 0;
        int displayObjectTracking    = 0;
        int displayPoseDetection     = 0;
        int displayInformation       = 0;

        // Show Processing Time
        int showProcTimeApp               = 0;
        int showProcTimeAIModel           = 0;
        int showProcTimeAIPostProc        = 0;
        int showProcTimeObjectTracking    = 0;

        // Debug Profiling
        int showDebugProfiling = 0;

        // local setting values
        float installHeight  = 0.0;
        float installOffset  = 0.0;

        // Historical Feed Settings
        configReader->getValue("InputMode", inputMode);
        configReader->getValue("RawImageDir", rawImageDir);
        configReader->getValue("ImageModeStartFrame", imageModeStartFrame);
        configReader->getValue("ImageModeEndFrame", imageModeEndFrame);
        configReader->getValue("ImageModeFrameName", imageModeFrameName);
        configReader->getValue("ServerPort", serverPort);
        configReader->getValue("ServerIP", serverIP);
        configReader->getValue("VisualizeMode", visualizeMode);

        // Model Information
        configReader->getValue("ModelPath", modelPath);
        configReader->getValue("FnetModelPath", fnetModelPath);
        configReader->getValue("InetModelPath", inetModelPath);
        configReader->getValue("UpdateModelPath", updateModelPath);
        
        // Auto-detect ONNX Runtime mode if model paths end with .onnx
        bool useOnnxRuntime = false;
        string useOnnxRuntimeStr = "";
        configReader->getValue("UseOnnxRuntime", useOnnxRuntimeStr);
        if (!useOnnxRuntimeStr.empty() && (useOnnxRuntimeStr == "true" || useOnnxRuntimeStr == "1" || useOnnxRuntimeStr == "True")) {
            useOnnxRuntime = true;
        } else {
            // Auto-detect: if any model path ends with .onnx, use ONNX Runtime
            if ((!fnetModelPath.empty() && fnetModelPath.substr(fnetModelPath.length() - 5) == ".onnx") ||
                (!inetModelPath.empty() && inetModelPath.substr(inetModelPath.length() - 5) == ".onnx") ||
                (!updateModelPath.empty() && updateModelPath.substr(updateModelPath.length() - 5) == ".onnx")) {
                useOnnxRuntime = true;
                logger->info("Auto-detected ONNX Runtime mode (model paths contain .onnx extension)");
            }
        }
        // Inference Cache
        bool enableInferenceCache = false;
        string enableInferenceCacheStr = "";
        configReader->getValue("EnableInferenceCache", enableInferenceCacheStr);
        if (!enableInferenceCacheStr.empty() && (enableInferenceCacheStr == "true" || enableInferenceCacheStr == "1" || enableInferenceCacheStr == "True")) {
            enableInferenceCache = true;
        }

        // Viewer Frame Saving
        bool saveViewerFrames = false;
        string saveViewerFramesStr = "";
        configReader->getValue("SaveViewerFrames", saveViewerFramesStr);
        if (!saveViewerFramesStr.empty() && (saveViewerFramesStr == "true" || saveViewerFramesStr == "1" || saveViewerFramesStr == "True")) {
            saveViewerFrames = true;
        }

        // Hidden State Reset Interval
        int netResetInterval = 0;
        configReader->getValue("NetResetInterval", netResetInterval);

        // Update Model Max Edges
        int maxEdges = 360;  // default: match current model
        configReader->getValue("MaxEdges", maxEdges);

        configReader->getValue("ModelWidth", modelWidth);
        configReader->getValue("ModelHeight", modelHeight);

        // Model ROI Information
        configReader->getValue("StartXRatio", startXRatio);
        configReader->getValue("EndXRatio", endXRatio);
        configReader->getValue("StartYRatio", startYRatio);
        configReader->getValue("EndYRatio", endYRatio);
        configReader->getValue("EnableROI", enableROI);
        
        // Camera Information
        configReader->getValue("CameraHeight", cameraHeight);
        // configReader->getValue("CameraFocalLength", cameraFocalLength);
        configReader->getValue("Intrinsic_fx", intrinsic_fx);
        configReader->getValue("Intrinsic_fy", intrinsic_fy);
        configReader->getValue("Intrinsic_cx", intrinsic_cx);
        configReader->getValue("Intrinsic_cy", intrinsic_cy);
        configReader->getValue("Distortion_k1", distortion_k1);
        configReader->getValue("Distortion_k2", distortion_k2);
        configReader->getValue("Distortion_p1", distortion_p1);
        configReader->getValue("Distortion_p2", distortion_p2);
        configReader->getValue("FrameWidth", frameWidth);
        configReader->getValue("FrameHeight", frameHeight);

        // Calibration
        configReader->getValue("VanishY", yVanish);

        // Processing Time
        configReader->getValue("ProcessingFrameRate", procFrameRate);
        configReader->getValue("ProcessingFrameStep", procFrameStep);

        // Object Detection
        configReader->getValue("HumanConfidence", humanConfidence);
        configReader->getValue("CarConfidence", carConfidence);

        // Object Tracking
        configReader->getValue("MaxTracking", maxTracking);

        // Debug Information
        configReader->getValue("DebugConfig", debugConfig);
        configReader->getValue("DebugApp", debugApp);
        configReader->getValue("DebugAI", debugAIModel);
        configReader->getValue("DebugAIPostProcessing", debugAIPostProc);
        configReader->getValue("DebugObjectDetection", debugObjectDetection);
        configReader->getValue("DebugObjectTracking", debugObjectTracking);
        configReader->getValue("DebugEnableJSON", debugEnableJson);
        configReader->getValue("DebugSaveLogs", debugSaveLogs);
        configReader->getValue("DebugSaveImages", debugSaveImages);
        configReader->getValue("DebugSaveRawImages", debugSaveRawImages);

        configReader->getValue("DebugLogsDirPath", debugLogsDirPath);
        configReader->getValue("DebugImagesDirPath", debugImagesDirPath);
        configReader->getValue("DebugRawImagesDirPath", debugRawImagesDirPath);

        // Display Results
        configReader->getValue("DisplayResults", displayResults);
        configReader->getValue("DisplayObjectDetection", displayObjectDetection);
        configReader->getValue("DisplayObjectTracking", displayObjectTracking);
        configReader->getValue("DisplayPoseDetection", displayPoseDetection);
        configReader->getValue("DisplayInformation", displayInformation);

        // Show Processing Time
        configReader->getValue("ShowProcTimeApp", showProcTimeApp);
        configReader->getValue("ShowProcTimeAI", showProcTimeAIModel);
        configReader->getValue("ShowProcTimeAIPostProcessing", showProcTimeAIPostProc);
        configReader->getValue("ShowProcTimeObjectTracking", showProcTimeObjectTracking);

        configReader->getValue("ShowDebugProfiling", showDebugProfiling);

        // Variables has been updated. Print them on the console.
        if (debugConfig)
        {
            // logger->info("-------------------------------------------------");
            // logger->info("[Platform Runtime Information]");
            // logger->info("-------------------------------------------------");
            // logger->info("Runtime \t\t= {}", runtime);
            // logger->info("Firmware \t\t= {}", firmwarePath);

            logger->info("-------------------------------------------------");
            logger->info("[Model Information]");
            logger->info("-------------------------------------------------");
            logger->info("ModelPath \t\t= {}",      modelPath);
            logger->info("FnetModelPath \t\t= {}",  fnetModelPath);
            logger->info("InetModelPath \t\t= {}",  inetModelPath);
            logger->info("UpdateModelPath \t= {}",  updateModelPath);
            logger->info("UseOnnxRuntime \t\t= {} (auto-detected: {})", useOnnxRuntime, 
                        (useOnnxRuntimeStr.empty() ? "yes" : "no"));
            logger->info("EnableInferenceCache \t= {}", enableInferenceCache);
            logger->info("SaveViewerFrames \t= {}", saveViewerFrames);
            logger->info("NetResetInterval \t= {} {}", netResetInterval,
                        (netResetInterval > 0 ? "(reset m_net every N frames)" : "(disabled)"));
            logger->info("MaxEdges \t\t= {} (update model input edge dimension)", maxEdges);
            logger->info("ModelWidth \t\t= {}",     modelWidth);
            logger->info("ModelHeight \t\t= {}",    modelHeight);
            
            logger->info("-------------------------------------------------");
            logger->info("[ROI Information]");
            logger->info("-------------------------------------------------");
            logger->info("StartXRatio \t\t= {}",    startXRatio);
            logger->info("EndXRatio \t\t= {}",      endXRatio);
            logger->info("StartYRatio \t\t= {}",    startYRatio);
            logger->info("EndYRatio \t\t= {}",      endYRatio);
            logger->info("EnableROI \t\t= {}",      enableROI);

            logger->info("-------------------------------------------------");
            logger->info("[Camera Information]");
            logger->info("-------------------------------------------------");
            logger->info("CameraHeight \t= {}",     cameraHeight);
            // logger->info("CameraFocalLength \t= {}",cameraFocalLength);
            logger->info("Intrinsic_fx \t= {}",      intrinsic_fx);
            logger->info("Intrinsic_fy \t= {}",     intrinsic_fy);
            logger->info("Intrinsic_cx \t= {}",     intrinsic_cx);
            logger->info("Intrinsic_cy \t= {}",     intrinsic_cy);
            logger->info("Distortion_k1 \t= {}",     distortion_k1);
            logger->info("Distortion_k2 \t= {}",     distortion_k2);
            logger->info("Distortion_p1 \t= {}",     distortion_p1);
            logger->info("Distortion_p2 \t= {}",     distortion_p2);
            logger->info("FrameWidth \t\t= {}",     frameWidth);
            logger->info("FrameHeight \t\t= {}",    frameHeight);

            logger->info("-------------------------------------------------");
            logger->info("[Processing Time]");
            logger->info("-------------------------------------------------");
            logger->info("ProcessingFrameRate \t= {}", procFrameRate);
            logger->info("ProcessingFrameStep \t= {}", procFrameStep);

            logger->info("-------------------------------------------------");
            logger->info("[Object Detection]");
            logger->info("-------------------------------------------------");
            logger->info("HumanConfidence \t= {}",  humanConfidence);
            logger->info("CarConfidence \t= {}",    carConfidence);

            logger->info("-------------------------------------------------");
            logger->info("[Object Tracking]");
            logger->info("-------------------------------------------------");
            logger->info("MaxTracking \t\t= {}", maxTracking);

            // Debug Information
            logger->info("-------------------------------------------------");
            logger->info("[Debug Information]");
            logger->info("-------------------------------------------------");
            logger->info("Config \t\t= {}",             debugConfig);
            logger->info("App \t\t= {}",                debugApp);
            logger->info("AIModel \t\t= {}",            debugAIModel);
            logger->info("AIPostProcessing \t= {}",     debugAIPostProc);
            logger->info("ObjectDetection \t= {}",      debugObjectDetection);
            logger->info("ObjectTracking \t= {}",       debugObjectTracking);
            logger->info("EnableJSON \t\t= {}",         debugEnableJson);
            logger->info("DebugSaveLogs \t\t= {}",      debugSaveLogs);
            logger->info("DebugSaveImages \t\t= {}",    debugSaveImages);
            logger->info("DebugSaveRawImages \t\t= {}", debugSaveRawImages);

            logger->info("DebugLogsDirPath \t= {}",     debugLogsDirPath);
            logger->info("DebugImagesDirPath \t= {}",   debugImagesDirPath);
            logger->info("DebugRawImagesDirPath \t= {}",debugRawImagesDirPath);

            // Display Results
            logger->info("-------------------------------------------------");
            logger->info("[Display Information]");
            logger->info("-------------------------------------------------");
            logger->info("Results \t\t= {}",        displayResults);
            logger->info("ObjectDetection \t= {}",  displayObjectDetection);
            logger->info("PoseDetection \t= {}",    displayPoseDetection);
            logger->info("ObjectTracking \t= {}",   displayObjectTracking);
            logger->info("Information \t\t= {}",    displayInformation);

            // Show Processing Time Information
            logger->info("-------------------------------------------------");
            logger->info("[Show Processing Time Information]");
            logger->info("-------------------------------------------------");
            logger->info("App = {}",                showProcTimeApp);
            logger->info("AIModel = {}",            showProcTimeAIModel);
            logger->info("AIPostProcessing = {}",   showProcTimeAIPostProc);
            logger->info("ObjectTracking = {}",     showProcTimeObjectTracking);
            logger->info("=================================================");

            logger->info("ShowDebugProfiling = {}", showDebugProfiling);
        }

        // Historical Feed Settings
        m_config->HistoricalFeedModeConfig.inputMode           = inputMode;
        m_config->HistoricalFeedModeConfig.rawImageDir         = rawImageDir;
        m_config->HistoricalFeedModeConfig.imageModeStartFrame = imageModeStartFrame;
        m_config->HistoricalFeedModeConfig.imageModeEndFrame   = imageModeEndFrame;
        m_config->HistoricalFeedModeConfig.imageModeFrameName  = imageModeFrameName;
        m_config->HistoricalFeedModeConfig.serverPort          = serverPort;
        m_config->HistoricalFeedModeConfig.serverIP            = serverIP;
        m_config->HistoricalFeedModeConfig.visualizeMode       = visualizeMode;
        m_config->HistoricalFeedModeConfig.calibRawImageDir    = calibRawImageDir;

        // Model Information
        m_config->modelPath      = modelPath;
        m_config->fnetModelPath   = fnetModelPath;
        m_config->inetModelPath  = inetModelPath;
        m_config->updateModelPath = updateModelPath;
        m_config->useOnnxRuntime = useOnnxRuntime;
        m_config->enableInferenceCache = enableInferenceCache;
        m_config->saveViewerFrames = saveViewerFrames;
        m_config->netResetInterval = netResetInterval;
        m_config->maxEdges       = maxEdges;
        m_config->modelWidth     = modelWidth;
        m_config->modelHeight    = modelHeight;

        // Model ROI Information
        m_config->startXRatio = std::stof(startXRatio);
        m_config->endXRatio   = std::stof(endXRatio);
        m_config->startYRatio = std::stof(startYRatio);
        m_config->endYRatio   = std::stof(endYRatio);
        m_config->enableROI   = enableROI;

      
        // Camera Information
        if (cameraHeight.size() > 0)
        {
            m_config->stCameraConfig.height = std::stof(cameraHeight);
        }
        if (installHeight > 0.0)
        {
            m_config->stCameraConfig.height = installHeight;
            logger->warn("stCameraConfig.height change to user setting value = {}", installHeight);
        }
        m_config->stCameraConfig.centrlOffset = installOffset;
        // if (cameraFocalLength.size() > 0)
        // {
        //     m_config->stCameraConfig.focalLength = std::stof(cameraFocalLength);
        // }
        if (intrinsic_fx.size() > 0)
        {
            m_config->stCameraConfig.intrinsic_fx = std::stof(intrinsic_fx);
        }
        if (intrinsic_fy.size() > 0)
        {
            m_config->stCameraConfig.intrinsic_fy = std::stof(intrinsic_fy);
        }
        if (intrinsic_cx.size() > 0)
        {
            m_config->stCameraConfig.intrinsic_cx = std::stof(intrinsic_cx);
        }
        if (intrinsic_cy.size() > 0)
        {
            m_config->stCameraConfig.intrinsic_cy = std::stof(intrinsic_cy);
        }
        if (distortion_k1.size() > 0)
        {
            m_config->stCameraConfig.distortion_k1 = std::stof(distortion_k1);
        }
        if (distortion_k2.size() > 0)
        {
            m_config->stCameraConfig.distortion_k2 = std::stof(distortion_k2);
        }
        if (distortion_p1.size() > 0)
        {
            m_config->stCameraConfig.distortion_p1 = std::stof(distortion_p1);
        }
        if (distortion_p2.size() > 0)
        {
            m_config->stCameraConfig.distortion_p2 = std::stof(distortion_p2);
        }
        m_config->frameWidth                  = frameWidth;
        m_config->frameHeight                 = frameHeight;

        // Processing Time
        m_config->procFrameRate = std::stof(procFrameRate);
        m_config->procFrameStep = procFrameStep;

        // Object Detection
        m_config->stOdConfig.humanConfidence    = std::stof(humanConfidence);
        m_config->stOdConfig.carConfidence      = std::stof(carConfidence);

        // Object Tracking
        m_config->stTrackerConifg.maxTracking = maxTracking;

        // Debug Information
        m_config->stDebugConfig.config             = debugConfig;
        m_config->stDebugConfig.App                = debugApp;
        m_config->stDebugConfig.AIModel            = debugAIModel;
        m_config->stDebugConfig.AIPostProcessing   = debugAIPostProc;
        m_config->stDebugConfig.objectDetection    = debugObjectDetection;
        m_config->stDebugConfig.objectTracking     = debugObjectTracking;
        m_config->stDebugConfig.enableJson         = debugEnableJson;
        m_config->stDebugConfig.saveImages         = debugSaveImages;
        m_config->stDebugConfig.saveRawImages      = debugSaveRawImages;

        m_config->stDebugConfig.imgsDirPath        = debugImagesDirPath;
        m_config->stDebugConfig.logsDirPath        = debugLogsDirPath;
        m_config->stDebugConfig.rawImgsDirPath     = debugRawImagesDirPath;

        // Display Results
        m_config->stDisplayConfig.results           = displayResults;
        m_config->stDisplayConfig.objectDetection   = displayObjectDetection;
        m_config->stDisplayConfig.poseDetection     = displayPoseDetection;
        m_config->stDisplayConfig.objectTracking    = displayObjectTracking;
        m_config->stDisplayConfig.information       = displayInformation;

        // Show Processing Time
        m_config->stShowProcTimeConfig.App                = showProcTimeApp;
        m_config->stShowProcTimeConfig.AIModel            = showProcTimeAIModel;
        m_config->stShowProcTimeConfig.AIPostProcessing   = showProcTimeAIPostProc;
        m_config->stShowProcTimeConfig.objectTracking     = showProcTimeObjectTracking;

        // Debug Profiling
        m_config->stDebugProfiling = showDebugProfiling;
    }
    else
    {
        logger->error("=================================================");
        logger->error("Read Config Failed! ===> Use default configs");
        logger->error("=================================================");
        // Pass configuration
        exit(1);
    }
    configReader = NULL;
}
