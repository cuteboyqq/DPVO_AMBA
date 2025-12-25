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

#include "yolov8_postproc.hpp"

YOLOv8_POSTPROC::YOLOv8_POSTPROC(Config_S* config, WakeCallback wakeFunc)
    : m_numAnchorBox((MODEL_WIDTH * MODEL_HEIGHT / 64) + (MODEL_WIDTH * MODEL_HEIGHT / 256)
                     + (MODEL_WIDTH * MODEL_HEIGHT / 1024)),
      m_boxBufferSize(4 * m_numAnchorBox),
      m_confBufferSize(m_numAnchorBox),
      m_classBufferSize(m_numAnchorBox),
      m_humanConfidence(config->stOdConfig.humanConfidence),
      m_estimateTime(config->stShowProcTimeConfig.AIPostProcessing),
      m_saveRawImage(config->stDebugConfig.saveRawImages),
      m_inputMode(config->HistoricalFeedModeConfig.inputMode)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::syslog_logger_mt("ai-post-processing", "adas-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
    auto logger = spdlog::stdout_color_mt("AI-PostProc");
    logger->set_pattern("[%n] [%^%l%$] %v");
#endif

    logger->set_level(config->stDebugConfig.AIPostProcessing ? spdlog::level::debug : spdlog::level::info);

    // Output Decoder
    m_decoder = new YOLOv8_Decoder(MODEL_HEIGHT, MODEL_WIDTH, "AI_Decoder");

    // Reserve capacity for vectors
    m_objectDetectionOut.reserve(NUM_OBJ_CLASSES);  // Traffic Object Detection

    // Wake function
    m_wakeFunc = wakeFunc;
};

YOLOv8_POSTPROC::~YOLOv8_POSTPROC()
{
#ifdef SPDLOG_USE_SYSLOG
    spdlog::drop("ai-post-processing");
#else
    spdlog::drop("AI-PostProc");
#endif
    spdlog::drop("AI_Decoder");

    delete m_decoder;

    m_decoder      = nullptr;
    m_objBoxBuff   = nullptr;
    m_objConfBuff  = nullptr;
    m_objClsBuff   = nullptr;
};


// ============================================
//           Check Output Buffers
// ============================================
bool YOLOv8_POSTPROC::_areObjectBuffersValid()
{
    return m_objBoxBuff && m_objConfBuff && m_objClsBuff;
}

// ============================================
//               Post Processing
// ============================================
bool YOLOv8_POSTPROC::_checkPrediction(const YOLOv8_Prediction& pred)
{
    return pred.objBoxBuff != nullptr && pred.objConfBuff != nullptr && pred.objClsBuff != nullptr;
}

bool YOLOv8_POSTPROC::_postProcessing(YOLOv8_Prediction& pred)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
#else
    auto logger = spdlog::get("AI-PostProc");
#endif

    logger->debug(" ======== Post-processing Frame Index: {} ========", m_frameIndex);

    if (!_checkPrediction(pred))
    {
        logger->error("Get NULL prediction ...");
        return false;
    }

    // STEP1. Get Prediction Results
    // Object Detection
    m_objBoxBuff   = pred.objBoxBuff;
    m_objConfBuff  = pred.objConfBuff;
    m_objClsBuff   = pred.objClsBuff;
    // Image
    m_img          = pred.img;

    // Lane Box Detection
    m_laneBoxBuff  = pred.laneBoxBuff;
    m_laneConfBuff = pred.laneConfBuff;
    m_laneClsBuff  = pred.laneClsBuff;

    // Lane Point Detection
    m_poseBoxBuff  = pred.poseBoxBuff;
    m_poseConfBuff = pred.poseConfBuff;
    m_poseClsBuff  = pred.poseClsBuff;
    m_poseKptsBuff = pred.poseKptsBuff;



    // STEP 2: Traffic Object Detection
    if (!_objectPostProcessing())
        return false;

    // STEP 3: Lane Line Detection
    if (!_lanePostProcessing())
        return false;

    logger->debug("End Post Processing");
    logger->debug("========================================");

    return true;
}

void YOLOv8_POSTPROC::_updateResultBuffer(int predFrameIdx)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
#else
    auto logger = spdlog::get("AI-PostProc");
#endif

    POST_PROC_RESULTS res;

    // Object Detection
    _getHumanBoundingBox(res.humanBBoxList, m_humanConfidence);
    _getVehicleBoundingBox(res.vehicleBBoxList,m_vehicleConfidence);
    _getFaceBoundingBox(res.faceBBoxList,m_faceConfidence);
    _getSkeletonBoundingBox(res.skeletonBBoxList,m_skeletonConfidence);
    _getRiderBoundingBox(res.riderBBoxList,m_riderConfidence);
    // Image
    res.img = m_img;

    // Push to result buffer
    std::unique_lock<std::mutex> result_lock(m_result_mutex);
    m_resultBuffer.emplace_back(predFrameIdx, res);
    result_lock.unlock();
    m_result_cond.notify_one();
    notifyProcessingComplete();

    logger->debug("Finished {}", __func__);
    m_bDone = true;
}

// ============================================
//        Post Processing (Multi-Thread)
// ============================================
void YOLOv8_POSTPROC::updatePredictionBuffer(YOLOv8_Prediction& pred, int frameIdx)
{
    if (!m_threadStarted)
        m_threadStarted = true;

    std::unique_lock<std::mutex> lock(m_mutex);
    m_predictionBuffer.emplace_back(frameIdx, pred);
    m_frameIndex = frameIdx;
    m_bDone      = false;
    lock.unlock();

    m_condition.notify_one();
}

void YOLOv8_POSTPROC::runThread()
{
    m_threadPostProcessing = std::thread(&YOLOv8_POSTPROC::_runProcessingFunc, this);
    return;
}

void YOLOv8_POSTPROC::stopThread()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_threadTerminated = true;
    }
    m_condition.notify_all(); // Wake up the thread if it's waiting
    
    if (m_threadPostProcessing.joinable())
        m_threadPostProcessing.join();
}

void YOLOv8_POSTPROC::_runProcessingFunc()
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
#else
    auto logger = spdlog::get("AI-PostProc");
#endif

    while (!m_threadTerminated)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        // Wait for work or termination signal
        m_condition.wait(lock,
                         [this]() { return m_threadTerminated || (!m_predictionBuffer.empty() && m_threadStarted); });

        if (m_threadTerminated)
        {
            break;
        }

        if (!m_predictionBuffer.empty())
        {
            auto time_0 = m_estimateTime ? std::chrono::high_resolution_clock::now()
                                         : std::chrono::time_point<std::chrono::high_resolution_clock>{};
            if (m_estimateTime)
            {
                logger->info("[AI Post-Processing Time]");
                logger->info("-----------------------------------------");
            }

            // YOLOv8_Prediction pred = m_predictionBuffer.front();
            auto                pair         = m_predictionBuffer.front();
            int                 predFrameIdx = pair.first;
            YOLOv8_Prediction pred         = pair.second;
            m_predictionBuffer.pop_front();
            lock.unlock();

            logger->debug("-----------------------------------------");
            logger->debug("[AI Post-Processing] Frame index: {}", predFrameIdx);
            logger->debug("[AI Post-Processing] Buffer size: {}", m_predictionBuffer.size());
            logger->debug("-----------------------------------------");

            // Perform Post Processing
            if (!_postProcessing(pred))
            {
                return;
            }

#if defined(CV28) || defined(CV28_SIMULATOR)
            // Clean Prediction Buffers

            delete[] pred.objBoxBuff;
            delete[] pred.objConfBuff;
            delete[] pred.objClsBuff;

            delete[] pred.laneBoxBuff;
            delete[] pred.laneConfBuff;
            delete[] pred.laneClsBuff;

            delete[] pred.poseBoxBuff;
            delete[] pred.poseConfBuff;
            delete[] pred.poseClsBuff;
            delete[] pred.poseKptsBuff;

            pred.objBoxBuff     = nullptr;
            pred.objConfBuff    = nullptr;
            pred.objClsBuff     = nullptr;

            pred.laneBoxBuff     = nullptr;
            pred.laneConfBuff    = nullptr;
            pred.laneClsBuff     = nullptr;

            pred.poseBoxBuff     = nullptr;
            pred.poseConfBuff    = nullptr;
            pred.poseClsBuff     = nullptr;
            pred.poseKptsBuff     = nullptr;
#endif

            // Add Processed Results into buffer
            _updateResultBuffer(predFrameIdx);

            if (m_estimateTime)
            {
                auto time_1 = std::chrono::high_resolution_clock::now();
                logger->info("[Total]: {} ms",
                             std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count()
                                 / (1000.0 * 1000));
                logger->debug("-----------------------------------------");
            }
        }
    }
}

bool YOLOv8_POSTPROC::run_sequential(YOLOv8_Prediction& pred, POST_PROC_RESULTS& res)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
#else
    auto logger = spdlog::get("AI-PostProc");
#endif
    if (!_checkPrediction(pred))
    {
        logger->error("Get NULL prediction ...");
        return false;
    }

    // STEP1. Get Prediction Results
    // Object Detection
    m_objBoxBuff   = pred.objBoxBuff;
    m_objConfBuff  = pred.objConfBuff;
    m_objClsBuff   = pred.objClsBuff;

    // Image
    m_img          = pred.img;

    // STEP 2: Traffic Object Detection
    if (!_objectPostProcessing())
        return false;

    delete[] pred.objBoxBuff;
    delete[] pred.objConfBuff;
    delete[] pred.objClsBuff;

    // Object Detection
    _getHumanBoundingBox(res.humanBBoxList, m_humanConfidence);

    // Image
    res.img = m_img;

    logger->debug("End Post Processing");
    logger->debug("========================================");

    return true;
}

bool YOLOv8_POSTPROC::getLastestResult(POST_PROC_RESULTS& res, int& resultFrameIdx)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
#else
    auto logger = spdlog::get("AI-PostProc");
#endif

    // std::unique_lock<std::mutex> lock(m_mutex);
    std::unique_lock<std::mutex> result_lock(m_result_mutex);
    int                          bufferSize = m_resultBuffer.size();

    if (bufferSize > 0)
    {
        auto pair      = m_resultBuffer.front();
        res            = pair.second;
        resultFrameIdx = pair.first;

        logger->debug("-----------------------------------------");
        logger->debug("[AI Post-Processing] Frame index: {}", resultFrameIdx);
        logger->debug("[AI Post-Processing] Buffer size: {}", m_resultBuffer.size());
        logger->debug("-----------------------------------------");

        // printf("[getLastestResult] ------------------------Get resultFrameIdx =
        // %d---------------------------\n",resultFrameIdx);
        m_resultBuffer.pop_front();
        result_lock.unlock();
        return true;
    }
    result_lock.unlock();
    return false;
}

// ============================================
//     Post Processing (Object Detection)
// ============================================
bool YOLOv8_POSTPROC::_objectPostProcessing()
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
#else
    auto logger = spdlog::get("AI-PostProc");
#endif

    auto time_0 = std::chrono::high_resolution_clock::now();
    auto time_1 = std::chrono::high_resolution_clock::now();

    // Check if all buffers are valid
    if (!_areObjectBuffersValid())
    {
        logger->error("Not all outputs of the network are available");
        return false;
    }

    m_objectDetectionOut.clear();

    logger->debug("Starting object detection post-processing......");
    m_numObjBox = m_decoder->decodeBox(
        m_objBoxBuff,
        m_objConfBuff,
        m_objClsBuff,
        m_numAnchorBox,
        m_confidenceThreshold,
        m_iouThreshold,
        NUM_OBJ_CLASSES,
        m_objectDetectionOut);

    if (m_estimateTime)
    {
        time_1 = std::chrono::high_resolution_clock::now();
        logger->debug("[Post-Proc OD]: {}",
                     std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
    }

    logger->debug("Number of raw object bounding boxes: {}", m_objectDetectionOut.size());

    if (logger->should_log(spdlog::level::debug))
    {
        int i = 0;
        for (int cls = 0; cls < NUM_OBJ_CLASSES; cls++)
        {
            for (const auto& obj : m_objectDetectionOut[cls])

                logger->debug("=> bbx {}: ({},{})-({},{}), c={}, conf={}", i++, obj.x1, obj.y1, obj.x2, obj.y2, obj.c,
                              obj.c_prob);
        }
    }

    logger->debug("Finished object detection post-processing");

    if (m_estimateTime)
    {
        logger->debug("[_objectPostProcessing]: {} ms",
                     std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
    }

    return true;
}

bool YOLOv8_POSTPROC::_areLaneBuffersValid()
{
    return m_laneBoxBuff && m_laneConfBuff && m_laneClsBuff;
}

bool YOLOv8_POSTPROC::_arePoseBuffersValid()
{
    return m_poseBoxBuff && m_poseConfBuff && m_poseClsBuff && m_poseKptsBuff;
}

// ============================================
//            Post Processing (Lane)
// ============================================
bool YOLOv8_POSTPROC::_lanePostProcessing()
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
#else
    auto logger = spdlog::get("AI-PostProc");
#endif

    auto time_0 = std::chrono::high_resolution_clock::now();
    auto time_1 = std::chrono::high_resolution_clock::now();
    auto time_2 = std::chrono::high_resolution_clock::now();

    // Check if all buffers are valid
    if (!_areLaneBuffersValid() || !_arePoseBuffersValid())
    {
        logger->error("Not all outputs of the network are available");
        return false;
    }

    //----------------------------------------------------------------------------------------------------------------------
    logger->debug("Starting face detection post-processing......");

    m_laneDetectionOut.clear();
    m_numLaneBox = m_decoder->decodeBox(
        m_laneBoxBuff,
        m_laneConfBuff,
        m_laneClsBuff,
        m_numAnchorBox,
        m_confidenceThreshold,
        m_iouThreshold,
        NUM_LANE_CLASSES,
        m_laneDetectionOut);

    if (m_estimateTime)
    {
        time_1 = std::chrono::high_resolution_clock::now();
        logger->info("[Post-Proc Lane]: {}",
                     std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
    }

    logger->debug("Number of raw lane bounding boxes: {}", m_laneDetectionOut.size());

    if (logger->should_log(spdlog::level::debug))
    {
        int i = 0;
        for (int cls = 0; cls < NUM_LANE_CLASSES; cls++)
        {
            for (const auto& obj : m_laneDetectionOut[cls])
                logger->debug("=> bbx {}: ({},{})-({},{}), c={}, conf={}", i++, obj.x1, obj.y1, obj.x2, obj.y2, obj.c,
                              obj.c_prob);
        }
    }

    time_1 = std::chrono::high_resolution_clock::now();
    logger->debug("Finished lane box detection post-processing");
    //----------------------------------------------------------------------------------------------------------------------

    logger->debug("Starting pose detection post-processing......(getCandidate & NMS)");

    m_poseDetectionOut.clear();
    m_numPoseBox = m_decoder->decodeBoxAndKpt(
        m_poseBoxBuff,
        m_poseConfBuff,
        m_poseClsBuff,
        m_poseKptsBuff, 
        m_numAnchorBox,
        m_confidenceThreshold,
        m_iouThreshold,
        NUM_POSE_CLASSES,
        m_poseDetectionOut);

    if (m_estimateTime)
    {
        time_2 = std::chrono::high_resolution_clock::now();
        logger->info("[Post-Proc pose]: {}",
                    std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_1).count() / (1000.0 * 1000));
    }

    logger->debug("Number of raw pose bounding boxes: {}", m_poseDetectionOut.size());

    if (logger->should_log(spdlog::level::debug))
    {
        int i = 0;
        for (int cls = 0; cls < NUM_POSE_CLASSES; cls++)
        {
            for (const auto& obj : m_poseDetectionOut[cls])
                logger->debug("=> bbx {}: ({},{})-({},{}), c={}, conf={}", i++, obj.x1, obj.y1, obj.x2, obj.y2, obj.c,
                            obj.c_prob);
        }
    }
    logger->debug("Finished lane point detection post-processing");
    //-------------------------------------------------------------------------------------------------------------------------------

    logger->debug("Finished lane post-processing");

    if (m_estimateTime)
    {
        time_2 = std::chrono::high_resolution_clock::now();
        logger->info("[_lanePostProcessing]: {} ms",
                     std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_0).count() / (1000.0 * 1000));
    }

    return true;
}



// ============================================
//                  Outputs
// ============================================
bool YOLOv8_POSTPROC::_getVehicleBoundingBox(vector<BoundingBox>& _outBboxList, float confidence)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
#else
    auto logger = spdlog::get("AI-PostProc");
#endif

    // Clear previous bounding boxes
    _outBboxList.clear();

    logger->debug("Get vehicle box => m_numBox = {}", m_numObjBox);

    float bboxWidth, bboxHeight, ratio;
    Point centerPoint, centerPointB;
    bool  bNeedToSkip = false;
    int   area, areaB, xDiff;

    std::vector<v8xyxy>& boxes = m_laneDetectionOut[BIG_VEHICLE];
    for (int i = 0; i < boxes.size(); ++i)
    {
        const v8xyxy& box = boxes[i];
        if (box.c_prob >= confidence)
        {
            logger->debug("Get vehicle box [{}] : ({}, {}, {}, {}, {}, {})", i, box.x1, box.y1, box.x2, box.y2, box.c,
                          box.c_prob);

            bboxWidth  = static_cast<float>(box.x2 - box.x1);
            bboxHeight = static_cast<float>(box.y2 - box.y1);

            // ratio = bboxWidth / bboxHeight;

            // if (ratio > 1.5)
            //     continue;
                
            // // filter out vehicles that out of ROI
            // centerPoint = Point(static_cast<int>((box.x1 + box.x2) / 2), static_cast<int>((box.y1 + box.y2) / 2));

            // // remove boxes
            // bNeedToSkip = false;
            // for (int j = 0; j < boxes.size(); ++j)
            // {
            //     if (j != i && boxes[j].c_prob >= confidence)
            //     {
            //         const v8xyxy& boxB = boxes[j];

            //         centerPointB =
            //             Point(static_cast<int>((boxB.x1 + boxB.x2) / 2), static_cast<int>((boxB.y1 + boxB.y2) / 2));
            //         xDiff = std::abs(centerPoint.x - centerPointB.x);

            //         if (xDiff < 10 && box.y1 < boxB.y1)
            //         {
            //             // Skip this box, due to overlap and y is smaller than another box
            //             bNeedToSkip = true;
            //             break;
            //         }
            //     }
            // }

            if (true)
            {
                _outBboxList.emplace_back(box.x1, box.y1, box.x2, box.y2, box.c);
                _outBboxList.back().confidence = box.c_prob;
            }
        }
        else
        {
            logger->debug("Exclude box [{}] : ({}, {}, {}, {}, {}, {})", i, box.x1, box.y1, box.x2, box.y2, box.c,
                          box.c_prob);
        }
    }

    return !_outBboxList.empty();
}

bool YOLOv8_POSTPROC::_getHumanBoundingBox(vector<BoundingBox>& _outBboxList, float confidence)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
#else
    auto logger = spdlog::get("AI-PostProc");
#endif

    // Clear previous bounding boxes
    _outBboxList.clear();

    logger->debug("Get pedestrian box => m_numBox = {}", m_numObjBox);

    std::vector<v8xyxy>& boxes = m_objectDetectionOut[HUMAN];
    for (int i = 0; i < boxes.size(); ++i)
    {
        const v8xyxy& box = boxes[i];
        if (box.c_prob >= confidence)
        {
            logger->debug("Get pedestrian box [{}] : ({}, {}, {}, {}, {}, {})", i, box.x1, box.y1, box.x2, box.y2,
                          box.c, box.c_prob);

            _outBboxList.emplace_back(box.x1, box.y1, box.x2, box.y2, box.c);
            _outBboxList.back().confidence = box.c_prob;
        }
        else
        {
            logger->debug("Exclude box [{}] : ({}, {}, {}, {}, {}, {})", i, box.x1, box.y1, box.x2, box.y2, box.c,
                          box.c_prob);
        }
    }

    return !_outBboxList.empty();
}

bool YOLOv8_POSTPROC::_getFaceBoundingBox(vector<BoundingBox>& _outBboxList, float confidence)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
#else
    auto logger = spdlog::get("AI-PostProc");
#endif

    // Clear previous bounding boxes
    _outBboxList.clear();

    logger->debug("Get pedestrian box => m_numBox = {}", m_numObjBox);

    std::vector<v8xyxy>& boxes = m_laneDetectionOut[FACE];
    for (int i = 0; i < boxes.size(); ++i)
    {
        const v8xyxy& box = boxes[i];
        if (box.c_prob >= confidence)
        {
            logger->debug("Get pedestrian box [{}] : ({}, {}, {}, {}, {}, {})", i, box.x1, box.y1, box.x2, box.y2,
                          box.c, box.c_prob);

            _outBboxList.emplace_back(box.x1, box.y1, box.x2, box.y2, box.c);
            _outBboxList.back().confidence = box.c_prob;
        }
        else
        {
            logger->debug("Exclude box [{}] : ({}, {}, {}, {}, {}, {})", i, box.x1, box.y1, box.x2, box.y2, box.c,
                          box.c_prob);
        }
    }

    return !_outBboxList.empty();
}

bool YOLOv8_POSTPROC::_getSkeletonBoundingBox(vector<BoundingBox>& _outBboxList, float confidence)
{
    #ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
    #else
    auto logger = spdlog::get("AI-PostProc");
    #endif

    // Clear previous bounding boxes
    _outBboxList.clear();
    // logger->info("Starting to process white line bounding boxes with confidence threshold: {}", confidence);

    // Log the total number of boxes
    // logger->info("Total detected white line boxes: {}", m_numPoseBox);

    // Access the boxes for WHITE_LINE detection
    std::vector<v8xyxy>& boxes = m_poseDetectionOut[SKELETON];

    for (int i = 0; i < boxes.size(); ++i)
    {
        const v8xyxy& box = boxes[i];
        // logger->info("Processing box [{}] : x1 = {}, y1 = {}, x2 = {}, y2 = {}, c = {}, c_prob = {}", 
        //               i, box.x1, box.y1, box.x2, box.y2, box.c, box.c_prob);

        if (box.c_prob >= confidence)
        {
            // logger->info("Box [{}] passed confidence threshold ({:.2f} >= {:.2f})",  i, box.c_prob, confidence);
            
            _outBboxList.emplace_back(box.x1, box.y1, box.x2, box.y2, box.c, box.pose_kpts); // Add pose keypoints
            _outBboxList.back().confidence = box.c_prob;

            // Log the added bounding box details
            const BoundingBox& addedBox = _outBboxList.back();
            // logger->info("Added bounding box [{}]: x1 = {}, y1 = {}, x2 = {}, y2 = {}, c = {}, confidence = {:.2f}", 
            //              i, addedBox.x1, addedBox.y1, addedBox.x2, addedBox.y2, addedBox.label, addedBox.confidence);

            // if (!addedBox.pose_kpts.empty())
            // {
            //     logger->info("Pose Keypoints for box [{}]:", i);
            //     for (size_t j = 0; j < addedBox.pose_kpts.size(); ++j)
            //     {
            //         logger->info("Skeleton Keypoint [{}]: x = {}, y = {}", j, addedBox.pose_kpts[j].first, addedBox.pose_kpts[j].second);
            //     }
            // }
        }
        // else
        // {
        //     logger->error("Box [{}] failed confidence threshold ({:.2f} < {:.2f})", 
        //                   i, box.c_prob, confidence);
        // }
    }

    // logger->info("Processed all boxes. Total valid bounding boxes: {}", _outBboxList.size());
    return !_outBboxList.empty();
}

bool YOLOv8_POSTPROC::_getRiderBoundingBox(vector<BoundingBox>& _outBboxList, float confidence)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::get("ai-post-processing");
#else
    auto logger = spdlog::get("AI-PostProc");
#endif

    // Clear previous bounding boxes
    _outBboxList.clear();

    logger->debug("Get pedestrian box => m_numBox = {}", m_numObjBox);

    std::vector<v8xyxy>& boxes = m_laneDetectionOut[RIDER];
    for (int i = 0; i < boxes.size(); ++i)
    {
        const v8xyxy& box = boxes[i];
        if (box.c_prob >= confidence)
        {
            logger->debug("Get pedestrian box [{}] : ({}, {}, {}, {}, {}, {})", i, box.x1, box.y1, box.x2, box.y2,
                          box.c, box.c_prob);

            _outBboxList.emplace_back(box.x1, box.y1, box.x2, box.y2, box.c);
            _outBboxList.back().confidence = box.c_prob;
        }
        else
        {
            logger->debug("Exclude box [{}] : ({}, {}, {}, {}, {}, {})", i, box.x1, box.y1, box.x2, box.y2, box.c,
                          box.c_prob);
        }
    }

    return !_outBboxList.empty();
}

void YOLOv8_POSTPROC::getDebugProfiles(int& inputBufferSize, int& outputBufferSize)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    inputBufferSize  = m_predictionBuffer.size();
    outputBufferSize = m_resultBuffer.size();
    lock.unlock();
}

bool YOLOv8_POSTPROC::isInputBufferEmpty() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_predictionBuffer.empty();
}

bool YOLOv8_POSTPROC::isOutputBufferEmpty() const
{
    std::lock_guard<std::mutex> lock(m_result_mutex);
    return m_resultBuffer.empty();
}

void YOLOv8_POSTPROC::notifyProcessingComplete()
{
    if (m_wakeFunc)
        m_wakeFunc();
}
