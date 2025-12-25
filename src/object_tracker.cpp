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

#include "object_tracker.hpp"

ObjectTracker::ObjectTracker(Config_S* _config, string _task)
    : m_task(_task == "human" ? TRACK_HUMAN : TRACK_CAR),
      m_frameWidth(_config->frameWidth),
      m_maxTracking(_config->stTrackerConifg.maxTracking),
      m_frameRate(static_cast<int>(_config->procFrameRate / _config->procFrameStep)),
      m_minWidth(m_task == TRACK_HUMAN ? 5 : 15),
      m_minHeight(m_task == TRACK_HUMAN ? 10 : 15),
      m_cameraHeight(_config->stCameraConfig.height),
      m_loggerStr(_task == "human" ? "HumanTracker" : "VehicleTracker"),
      m_estimateTime(_config->stShowProcTimeConfig.objectTracking)
{
#ifdef SPDLOG_USE_SYSLOG
    auto logger = spdlog::syslog_logger_mt(m_loggerStr, "adas-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
    auto logger = spdlog::stdout_color_mt(m_loggerStr);
    logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    logger->set_level(_config->stDebugConfig.objectTracking ? spdlog::level::debug : spdlog::level::info);

    // initialize BaseTracker
    const int maxTimeLost = m_frameRate / 2; // remove lost tracklet after maxTimeLost frames
    m_BaseTracker         = BaseTracker(m_tTrackThresh, m_tActivateThresh, m_tMatchThresh, maxTimeLost);
};

ObjectTracker::~ObjectTracker()
{
    spdlog::drop(m_loggerStr);
};

void ObjectTracker::run(std::vector<BoundingBox>& bboxList, unordered_map<int, Object>& objectUmap, vector<Object>& trackedObjList)
{
    auto logger    = spdlog::get(m_loggerStr);
    auto logToggle = logger->should_log(spdlog::level::debug);
    if (m_estimateTime)
    {
        logger->info("[{} Tracker Processing Time]", m_task == TRACK_HUMAN ? "Human" : "Vehicle");
        logger->info("-----------------------------------------");
    }
    auto time_0 = m_estimateTime ? std::chrono::high_resolution_clock::now()
                                 : std::chrono::time_point<std::chrono::high_resolution_clock>{};

    _tracking(bboxList);
    _updateObjectUmap(objectUmap);
    _distanceCalc(objectUmap, trackedObjList);
    // Update box position
    for (auto& obj : trackedObjList)
    {
        BoundingBox lastBox = obj.bboxList.back();
        for (auto& box : bboxList)
        {
            if (imgUtil::iou(lastBox, box) > 0.5f)
            {
                obj.bboxList.back().boxPosition = box.boxPosition;
                obj.bbox.boxPosition = box.boxPosition;
            }
        }
    }

    if (m_estimateTime)
    {
        auto time_1 = std::chrono::high_resolution_clock::now();
        logger->info("[Total]: {} ms",
                     std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
        logger->info("-----------------------------------------");
    }
}


void ObjectTracker::_distanceCalc(unordered_map<int, Object>& objectUmap, vector<Object>& trackedObjList)
{
    /*
        inputs:
        unordered_map<int,Object>& objectUmap: map id -> Object, a unordered_map as member in wnc_adas
        vector<Object>& trackedObjList: activated object

        Calculate distance and ttc
    */

    auto logger    = spdlog::get(m_loggerStr);
    auto logToggle = logger->should_log(spdlog::level::debug);
    auto time_0    = m_estimateTime ? std::chrono::high_resolution_clock::now()
                                 : std::chrono::time_point<std::chrono::high_resolution_clock>{};

    vector<int> exceed_max_time_ids;

    trackedObjList.clear();
    for (auto& item : objectUmap)
    {
        // get Object in the map. item.first: id, item.second: Object
        int     track_id = item.first;
        Object& obj      = item.second;
        // if object is activated and is in activeIdSet
        if (obj.getStatus() == 1 && this->activeIdSet.find(obj.id) != this->activeIdSet.end())
        {
            // if (m_yVanish > 0)
            // {
            //     // Following Distance
            //     m_distance->run(obj);
            // }
            // else
            // {
            //     if (logToggle)
            //     {
            //         logger->debug("Skipping calculating distance due to m_yVanish = {}", m_yVanish);
            //     }
            // }
            // _calcTTC(obj);
            trackedObjList.push_back(obj);
        }
        else
        {
            // exclude objects not in the activeIdSet
            // warning: could just be a non-activated object
            obj.updateStatus(0);

            // failsafe: check internal frame id again, and get obj that exceeds max_time_lost
            int frame_id = m_BaseTracker.get_frame_id();
            // in most cases, elapsed_frames should <= m_BaseTracker.get_max_time_lost()
            int elapsed_frames = frame_id - obj.bbox.frameStamp;
            if (frame_id < obj.bbox.frameStamp)
            {
                // frame_id exceeds INT_MAX and reset, but obj.bbox.frameStamp is still near to INT_MAX
                elapsed_frames = frame_id + (INT_MAX - obj.bbox.frameStamp);
            }
            if (elapsed_frames > m_BaseTracker.get_max_time_lost())
            {
                exceed_max_time_ids.push_back(track_id);
            }
        }
    }

    // remove obj that exceeds max_time_lost
    for (int& exceed_id : exceed_max_time_ids)
    {
        // if obj to remove is found
        if (objectUmap.find(exceed_id) != objectUmap.end())
        {
            // wipe Object content and erase
            objectUmap[exceed_id].init(-1);
            objectUmap.erase(exceed_id);
        }
    }

    // need to erase some items that disappeared but not removed
    if (m_estimateTime)
    {
        auto time_1 = std::chrono::high_resolution_clock::now();
        logger->info("[{}]: {} ms", __func__,
                     std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
    }
}


std::vector<STrack> ObjectTracker::_boundingBoxToSTrack(std::vector<BoundingBox>& bboxList, std::string logStr)
{
    /*
        inputs:
        std::vector<BoundingBox> &bboxList: filtered box list

        convert vector<BoundingBox> &bboxList to std::vector<STrack> STrackList
    */

    auto logger    = spdlog::get(m_loggerStr);
    auto logToggle = logger->should_log(spdlog::level::debug);

    if (logToggle)
    {
        logger->debug("[Tracker Input] {} vector size: {}", logStr, bboxList.size());
        for (size_t i = 0; i < bboxList.size(); ++i)
        {
            BoundingBox& box = bboxList[i];
            logger->debug("box[{}]: x1: {}, y1: {}, x2: {}, y2: {}, label: {}", i, box.x1, box.y1, box.x2, box.y2,
                          box.label);
        }
    }

    // Convert BoundingBox to STrack
    std::vector<STrack> STrackList;
    int                 numBox = min((int)bboxList.size(), ADAS_TRACKER_MAX_NUM_DETECT_BOX);
    for (int i = 0; i < numBox; i++)
    {
        BoundingBox&  bbox = bboxList[i];
        vector<float> tlbr = {(float)bbox.x1, (float)bbox.y1, (float)bbox.x2, (float)bbox.y2};
        STrack        strack(tlbr, bbox.confidence);
        STrackList.push_back(strack);
    }

    if (logToggle)
    {
        logger->debug("[Tracker Input] Converted STrack {} list vector size: {}", logStr, STrackList.size());
        for (size_t i = 0; i < STrackList.size(); ++i)
        {
            STrack& strack = STrackList[i];
            logger->debug("strack[{}]: x1: {}, y1: {}, x2: {}, y2: {}, conf {}", i, strack.tlbr[0], strack.tlbr[1],
                          strack.tlbr[2], strack.tlbr[3], (float)strack.get_score());
        }
    }

    return STrackList;
}

void ObjectTracker::_tracking(std::vector<BoundingBox>& bboxList)
{
    /*
        inputs:
        std::vector<BoundingBox> &bboxList: filtered box list

        convert vector<BoundingBox> &bboxList to std::vector<STrack> STrackList
        tracking by BaseTracker m_BaseTracker.update(STrackList);
        update tracklets in this->output_stracks
    */

    auto logger    = spdlog::get(m_loggerStr);
    auto logToggle = logger->should_log(spdlog::level::debug);
    auto time_0    = m_estimateTime ? std::chrono::high_resolution_clock::now()
                                 : std::chrono::time_point<std::chrono::high_resolution_clock>{};

    // Prepare a bbox list for object tracking
    std::vector<BoundingBox> bboxList2Track;
    bboxList2Track.reserve(bboxList.size());
    bboxList2Track = bboxList;


    std::vector<STrack> STrackList = _boundingBoxToSTrack(bboxList, "main bbox"); // bboxList2Track
    this->output_stracks = m_BaseTracker.update(STrackList);


    logToggle = true;
    if (logToggle)
    {
        logger->debug("[Tracker Output] Tracked STrack list vector size: {}", this->output_stracks.size());
        for (size_t i = 0; i < this->output_stracks.size(); ++i)
        {
            STrack& strack = this->output_stracks[i];
            logger->debug("strack[{}]: track_id {}, x1: {}, y1: {}, x2: {}, y2: {}, conf {}, is_activated {}, state {}",
                          i, strack.track_id, strack.tlbr[0], strack.tlbr[1], strack.tlbr[2], strack.tlbr[3],
                          (float)strack.get_score(), strack.is_activated, strack.state);
        }
    }

    if (m_estimateTime)
    {
        auto time_1 = std::chrono::high_resolution_clock::now();
        logger->info("[{}]: {} ms", __func__,
                     std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
    }
}

void ObjectTracker::_updateObjectUmap(unordered_map<int, Object>& objectUmap)
{
    /*
        inputs:
        unordered_map<int,Object>& objectUmap: map id -> Object, a unordered_map as member in wnc_adas

        Update unordered_map objectUmap with std::vector<STrack> this->output_stracks
        if strack track_id doesn't exist in objectUmap, create a new object.
        always update bounding box
        remove the objects in objectUmap given remove id
        disable the objects in objectUmap given lost id
    */

    auto logger    = spdlog::get(m_loggerStr);
    auto logToggle = logger->should_log(spdlog::level::debug);
    auto time_0    = m_estimateTime ? std::chrono::high_resolution_clock::now()
                                 : std::chrono::time_point<std::chrono::high_resolution_clock>{};

    // get remove ids and lost ids
    vector<int> removed_ids = m_BaseTracker.get_removed_ids();
    vector<int> lost_ids    = m_BaseTracker.get_lost_ids();

    if (logToggle)
    {
        for (STrack& strack : this->output_stracks)
        {
            int& track_id = strack.track_id;
            if (objectUmap.find(track_id) == objectUmap.end())
            {
                // create new object
                logger->debug("[Tracker Update]: New Object track_id {}", track_id);
            }
        }
    }

    this->activeIdSet.clear();
    // Update objectUmap with activated objects according to track id
    for (STrack& strack : this->output_stracks)
    {
        // activated objects
        int& track_id = strack.track_id;
        this->activeIdSet.insert(track_id);

        // convert STrack to BoundingBox
        BoundingBox bbox;
        if (strack.tlbr.size() >= 4)
        {
            // valid box
            bbox.x1 = strack.tlbr[0]; // Bounding Box x1
            bbox.y1 = strack.tlbr[1]; // Bounding Box y1
            bbox.x2 = strack.tlbr[2]; // Bounding Box x2
            bbox.y2 = strack.tlbr[3]; // Bounding Box y2
        }
        else
        {
            // something wrong
            continue;
        }

        bbox.label      = m_task;             // Bounding Box label
        bbox.confidence = strack.score;       // Confidence score
        bbox.frameStamp = strack.end_frame(); // internal timestamp
        bbox.objID      = -1;
        bbox.boxID      = -1;

        // track_id not found in objectUmap
        if (objectUmap.find(track_id) == objectUmap.end())
        {
            // create new object in objectUmap
            objectUmap[track_id] = Object();
            objectUmap[track_id].init(0);
            objectUmap[track_id].id = track_id;
        }
        // now track id must exist in objectUmap

        // setup tracked object & new object
        objectUmap[track_id].updateBoundingBox(bbox);
        objectUmap[track_id].updateStatus(1);

        // push bbox to bboxList
        vector<BoundingBox>& bboxlist = objectUmap[track_id].bboxList;
        bboxlist.push_back(bbox);

        // keep the last MAX_BBOX_LIST_SIZE elements
        int num = (int)bboxlist.size() - MAX_BBOX_LIST_SIZE;
        if (num > 0)
        {
            // erase from the beginning to keep the last MAX_BBOX_LIST_SIZE elements
            bboxlist.erase(bboxlist.begin(), bboxlist.begin() + num);
        }
    }

    if (logToggle)
    {
        logger->debug("[Tracker Update] removed ids list vector size: {}", removed_ids.size());
        for (int& remove_id : removed_ids)
        {
            logger->debug("remove id {}", remove_id);
        }
        logger->warn("[Tracker Update] lost ids list vector size: {}", lost_ids.size());
        for (int& lost_id : lost_ids)
        {
            logger->error("lost id {}", lost_id);
        }
    }

    for (int& remove_id : removed_ids)
    {
        // if obj to remove is found
        if (objectUmap.find(remove_id) != objectUmap.end())
        {
            // wipe Object content and erase
            objectUmap[remove_id].init(-1);
            objectUmap.erase(remove_id);
        }
    }

    for (int& lost_id : lost_ids)
    {
        // if lost obj is found
        if (objectUmap.find(lost_id) != objectUmap.end())
        {
            // update status as non-activated
            objectUmap[lost_id].updateStatus(0);
        }
    }

    if (logToggle)
    {
        logger->debug("[Tracker Update] Object map size {}", objectUmap.size());
        for (auto& item : objectUmap)
        {
            Object& obj = item.second;
            logger->debug("Obj id {} status {}", obj.id, (int)obj.getStatus());
        }
    }

    if (m_estimateTime)
    {
        auto time_1 = std::chrono::high_resolution_clock::now();
        logger->info("[{}]: {} ms", __func__,
                     std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
    }
}