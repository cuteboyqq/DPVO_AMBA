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

#include <iostream>

class Point
{
public:
    Point(int _x, int _y);
    Point() : x(-1), y(-1)
    {
    }
    ~Point();
    int   x              = -1;
    int   y              = -1;
    int   behevior       = 0; // special use for human behavior
    int   needWarn       = 0;
    float visionDistance = 65535.0f;
    float radarDistance  = 65535.0f;
    int   objID          = -1;

private:
    int debugMode = false;
};
