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

#ifndef _DLA_CONFIG_
#define _DLA_CONFIG_

#include <map>
#include <string>
#include <iostream>

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    float humanConfidence;
    float carConfidence;
} OD_Config_S;

typedef struct
{
    int maxTracking; // How many object can be tracked at the same time
} TRACKER_Config_S;

typedef struct
{
    float height;
    float intrinsic_fx;   // Camera intrinsic fx (focal length X) in pixels - highest priority
    float intrinsic_fy;   // Camera intrinsic fy (focal length Y) in pixels - highest priority
    float intrinsic_cx;   // Camera intrinsic cx (principal point X) in pixels - highest priority
    float intrinsic_cy;   // Camera intrinsic cy (principal point Y) in pixels - highest priority
    float centrlOffset;    // Horizontal offset distance to car center in meter unit
} CAMERA_Config_S;


typedef struct
{
    bool        config;
    bool        App;
    bool        AIModel;
    bool        AIPostProcessing;
    bool        objectDetection;
    bool        objectTracking;
    bool        enableJson;
    bool        saveImages;
    bool        saveRawImages;

    std::string imgsDirPath;
    std::string rawImgsDirPath;
    std::string logsDirPath;

} Debug_Config_S;

typedef struct
{
    bool results;
    bool objectDetection;
    bool objectTracking;
    bool poseDetection;
    bool information;
} Display_Config_S;

typedef struct
{
    bool App;
    bool AIModel;
    bool AIPostProcessing;
    bool objectTracking;

} ShowProcTime_Config_S;

extern "C" {
typedef struct
{
    int         enableVideoMode;
    int         inputMode;
    // std::string videoPath;
    std::string rawImageDir;
    std::string calibRawImageDir;
    int         saveVideoImages;
    std::string decodeType;
    int         imageModeStartFrame;
    int         imageModeEndFrame;
    std::string imageModeFrameName;
    int         serverPort;
    std::string serverIP;
    int         visualizeMode;

} Historical_Feed_Mode_Config_S;
}

typedef struct
{
    std::string modelPath;
    
    // DPVO Model Paths
    std::string fnetModelPath;      // Path to FNet model
    std::string inetModelPath;      // Path to INet model
    std::string updateModelPath;    // Path to DPVO Update model

    // Model Input Size
    int modelWidth;
    int modelHeight;

    // Model ROI Ratio
    float startXRatio;
    float endXRatio;
    float startYRatio;
    float endYRatio;
    bool enableROI;
    
    // Frame Size
    int frameWidth;
    int frameHeight;

    // Enable ROI
    int EnableROI;

    // Enable Line Cross
    int EnableLineCross;

    // ROI coordinate
    int RoiX1;
    int RoiY1;
    int RoiX2;
    int RoiY2;

    // Line cross coordinate
    int LineCrossX1;
    int LineCrossY1;
    int LineCrossX2;
    int LineCrossY2;

    // Alarm Zone
    int AlarmZone;

    // Processing
    float procFrameRate;
    int   procFrameStep;

    // Other config
    OD_Config_S                   stOdConfig;
    TRACKER_Config_S              stTrackerConifg;
    CAMERA_Config_S               stCameraConfig;
    Debug_Config_S                stDebugConfig;
    Display_Config_S              stDisplayConfig;
    ShowProcTime_Config_S         stShowProcTimeConfig;
    Historical_Feed_Mode_Config_S HistoricalFeedModeConfig;
    bool                          stDebugProfiling;

} Config_S;


class ConfigReader
{
private:
    // Define the map to store data from the config file
    std::map<std::string, std::string> m_ConfigSettingMap;

    // Static pointer instance to make this class singleton.
    static ConfigReader* m_pInstance;

public:
    // Public static method getInstance(). This function is
    // responsible for object creation.
    static ConfigReader* getInstance();

    // Parse the config file.
    bool parseFile(std::string fileName = "/tmp/default_config");

    // Overloaded getValue() function.
    // Value of the tag in cofiguration file could be
    // string or integer. So the caller need to take care this.
    // Caller need to call appropiate function based on the
    // data type of the value of the tag.

    bool getValue(std::string tag, int& value);
    bool getValue(std::string tag, float& value);
    bool getValue(std::string tag, std::string& value);

    // Function dumpFileValues is for only debug purpose
    void dumpFileValues();

private:
    // Define constructor in the private section to make this
    // class as singleton.
    ConfigReader();

    // Define destructor in private section, so no one can delete
    // the instance of this class.
    ~ConfigReader();

    // Define copy constructor in the private section, so that no one can
    // voilate the singleton policy of this class
    ConfigReader(const ConfigReader& obj)
    {
    }
    // Define assignment operator in the private section, so that no one can
    // voilate the singleton policy of this class
    void operator=(const ConfigReader& obj)
    {
    }

    // Helper function to trim the tag and value. These helper function is
    // calling to trim the un-necessary spaces.
    std::string trim(const std::string& str, const std::string& whitespace = " \t");
    std::string reduce(const std::string& str, const std::string& fill = " ", const std::string& whitespace = " \t");
};

#ifdef __cplusplus
}

#endif
#endif
