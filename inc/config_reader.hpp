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
#ifndef __CONFIG_READER__
#define __CONFIG_READER__

#include <iostream>
#include <string>

#include "utils.hpp"
#include "dla_config.hpp"
#include "logger.hpp"

#define CONFIG_ENABLE 1
#define CONFIG_DISABLE 0

class AppConfigReader
{
public:
    AppConfigReader();
    ~AppConfigReader();

    void read(std::string configPath);
    void getConfig(Config_S* config);
    Config_S* getConfig();

    Config_S* m_config;
};

#endif