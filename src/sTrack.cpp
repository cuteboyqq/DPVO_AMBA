#include "sTrack.h"

STrack::STrack(vector<float> tlbr_, float score)
{
    is_activated = false;
    track_id     = 0;
    state        = TrackState::New;

    tlbr.resize(4);
    tlbr.assign(tlbr_.begin(), tlbr_.end());

    frame_id     = 0;
    tracklet_len = 0;
    this->score  = score;
    start_frame  = 0;
}

STrack::~STrack()
{
}

void STrack::activate(int frame_id, int next_id)
{
    this->track_id = next_id;
    // this->track_id = this->next_id();

    this->tracklet_len = 0;
    this->state        = TrackState::Tracked;
    if (frame_id == 1)
    {
        this->is_activated = true;
    }
    // this->is_activated = true;
    this->frame_id    = frame_id;
    this->start_frame = frame_id;
}

void STrack::re_activate(STrack &new_track, int frame_id, bool new_id, int next_id)
{
    tlbr.resize(4);
    tlbr.assign(new_track.tlbr.begin(), new_track.tlbr.end());

    this->tracklet_len = 0;
    this->state        = TrackState::Tracked;
    this->is_activated = true;
    this->frame_id     = frame_id;
    this->score        = new_track.score;
    if (new_id)
        this->track_id = next_id;
    // this->track_id = next_id();
}

void STrack::update(STrack &new_track, int frame_id)
{
    this->frame_id = frame_id;
    this->tracklet_len++;

    tlbr.resize(4);
    tlbr.assign(new_track.tlbr.begin(), new_track.tlbr.end());

    this->state        = TrackState::Tracked;
    this->is_activated = true;

    this->score = new_track.score;
}

void STrack::mark_lost()
{
    state = TrackState::Lost;
}

void STrack::mark_removed()
{
    state = TrackState::Removed;
}

// move next_id() to base tracker and avoid id collision
// maintaining a STrack static set will not distinguish multi-class tracking
// int STrack::next_id()
// {
//     static int _count = 0;
//     _count = (_count+1)%(INT_MAX);
//     return _count;
// }

int STrack::end_frame()
{
    return this->frame_id;
}

float STrack::get_score()
{
    return score;
}
