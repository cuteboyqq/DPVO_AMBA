/*
  (C) 2023-2024 Wistron NeWeb Corporation (WNC) - All Rights Reserved
  Minimal Point class for YOLOv8/BoundingBox compatibility.
*/

#pragma once

#include <iostream>

class Point
{
public:
    Point(int _x, int _y);
    Point() : x(-1), y(-1) {}
    ~Point();
    int   x              = -1;
    int   y              = -1;
    int   behevior       = 0;
    int   needWarn       = 0;
    float visionDistance = 65535.0f;
    float radarDistance  = 65535.0f;
    int   objID          = -1;

private:
    int debugMode = false;
};
