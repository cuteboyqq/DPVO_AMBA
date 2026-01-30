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

#ifndef __JSON_LOG__
#define __JSON_LOG__

#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>  // Use system-installed version (3.12.0)
// DEPRECATED: bounding_box.hpp removed
// #include "bounding_box.hpp"
#include "dataStructures.h"
#include "config_reader.hpp"
constexpr int NUM_POINTS = 3;
using namespace std;
// Use the nlohmann::json type
using json = nlohmann::json;

class JSON_LOG
{
public:
    JSON_LOG(std::string file, Config_S* m_config);
    ~JSON_LOG();

    // DEPRECATED: Function using BoundingBox and Object removed
    // std::string logInfo(WNC_APP_Results appResult, 
    //                             std::vector<BoundingBox> humanBBoxList,
    //                             std::vector<BoundingBox> vehicleBBoxList, 
    //                             std::vector<Object> trackedObjList,
    //                             int frameIdx, 
    //                             DebugProfile debugProfile,
    //                             char* version);
    
    bool m_bSaveDetObjLog  = false;

protected:
    void _initializeJsonFile();
    void _appendToJsonFile(const std::string& frameIdx, const json& frameData);
    
    bool m_bShowJson       = true;
    bool m_bSaveTrackObj   = true;
    bool m_bSaveFaceObj    = false;
    bool m_bSavePoseObj    = false;
    bool m_bDebugProfiling = false;
    bool m_bSaveToJSONFile = true;
// #ifdef SAV837
//         false
// #else
//         true
// #endif
//         ;

    std::string m_jsonFilePath;
};

#endif
