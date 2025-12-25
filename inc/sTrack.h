#pragma once

#include <vector>

using namespace std;

enum TrackState
{
    New = 0,
    Tracked,
    Lost,
    Removed
};

class STrack
{
public:
    STrack(vector<float> tlbr_, float score);
    ~STrack();

    void mark_lost();
    void mark_removed();
    // int next_id();
    int end_frame();

    // void activate(int frame_id);
    void activate(int frame_id, int next_id);
    // void re_activate(STrack &new_track, int frame_id, bool new_id = false);
    void re_activate(STrack &new_track, int frame_id, bool new_id, int next_id);
    void update(STrack &new_track, int frame_id);
    float get_score();

public:
    bool is_activated;
    int  track_id;
    int  state;

    vector<float> tlbr;
    int           frame_id;
    int           tracklet_len;
    int           start_frame;
    float         score;
};