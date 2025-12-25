#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <unordered_set>
#include <iomanip>

#include "baseTracker.h"
#include "lapjv.h"

using std::map;
using std::cout;
using std::endl;

BaseTracker::BaseTracker(float _track_thresh, float _high_thresh, float _match_thresh, int _max_time_lost)
    : track_thresh(_track_thresh),
      high_thresh(_high_thresh),
      match_thresh(_match_thresh),
      max_time_lost(_max_time_lost),
      frame_id(0)
{
}

BaseTracker::BaseTracker(int frame_rate, int track_buffer)
{
    track_thresh = 0.1; // track thresh (for detections & detections_low) original is 0.3
    high_thresh  = 0.1; // high thresh (for activation) original is 0.3
    match_thresh = 0.4; // original is 0.7

    frame_id      = 0;
    max_time_lost = int(frame_rate / 30.0 * track_buffer);
}

BaseTracker::~BaseTracker()
{
}


vector<STrack> BaseTracker::update(const vector<STrack> &objects)
{
    ////////////////// Step 1: Get detections //////////////////
    this->frame_id = (this->frame_id + 1) % (INT_MAX);
    vector<STrack> activated_stracks;
    vector<STrack> refind_stracks;
    vector<STrack> removed_stracks;
    vector<STrack> lost_stracks;
    vector<STrack> detections;
#if defined(ADAS_TRACKER_LOW_SCORE_ASSIGN)
    // 1) match (main) detections with high score with tracklets
    // 2) match detections_low with the remained lost tracklets
    // priority of detections_low
    // (main) boxes with low score > secondary boxes with high score > secondary boxes with low score
    vector<STrack> detections_low;
#endif
    vector<STrack> detections_cp;
    vector<STrack> tracked_stracks_swap;
    vector<STrack> resa, resb;
    vector<STrack> output_stracks;

    vector<STrack *> unconfirmed;
    vector<STrack *> tracked_stracks;
    vector<STrack *> strack_pool;
    vector<STrack *> r_tracked_stracks;

    if (objects.size() > 0)
    {
        for (int i = 0; i < objects.size(); i++)
        {
            STrack strack = objects[i];
            float  score  = strack.get_score();
            if (score >= track_thresh)
            {
                detections.push_back(strack);
            }
#ifdef ADAS_TRACKER_LOW_SCORE_ASSIGN
            else
            {
                // (main) boxes with low score > secondary boxes with high score > secondary boxes with low score
                detections_low.push_back(strack);
            }
#endif
        }
#ifdef ADAS_TRACKER_LOW_SCORE_ASSIGN
// std::cout <<"main low score num " << detections_low.size() << std::endl;
#endif
    }

    // Add newly detected tracklets to tracked_stracks
    for (int i = 0; i < this->tracked_stracks.size(); i++)
    {
        if (!this->tracked_stracks[i].is_activated)
            unconfirmed.push_back(&this->tracked_stracks[i]);
        else
            tracked_stracks.push_back(&this->tracked_stracks[i]);
    }

    ////////////////// Step 2: First association, with IoU //////////////////
    strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);

    vector<vector<float>> dists;
    int                   dist_size = 0, dist_size_size = 0;
    dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

    vector<vector<int>> matches;
    vector<int>         u_track, u_detection;
    linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

    for (int i = 0; i < matches.size(); i++)
    {
        STrack *track = strack_pool[matches[i][0]];
        STrack *det   = &detections[matches[i][1]];
        if (track->state == TrackState::Tracked)
        {
            track->update(*det, this->frame_id);
            activated_stracks.push_back(*track);
        }
        else
        {
            track->re_activate(*det, this->frame_id, false, this->next_id());
            refind_stracks.push_back(*track);
        }
    }

    ////////////////// Step 3: Second association, using low score dets //////////////////
    for (int i = 0; i < u_detection.size(); i++)
    {
        detections_cp.push_back(detections[u_detection[i]]);
    }

#if defined(ADAS_TRACKER_LOW_SCORE_ASSIGN)

    detections.clear();
    int matches_size        = (int)matches.size();
    int detections_low_size = (int)detections_low.size();
    if (matches_size < ADAS_TRACKER_MAX_NUM_DETECT_BOX)
    {
        // matched tracklets haven't exceeded ADAS_TRACKER_MAX_NUM_DETECT_BOX
        // still have room for detection_low
        int num_low = min((int)ADAS_TRACKER_MAX_NUM_DETECT_BOX - matches_size, detections_low_size);
        detections.assign(detections_low.begin(), detections_low.begin() + num_low);
    }
#endif

    for (int i = 0; i < u_track.size(); i++)
    {
        if (strack_pool[u_track[i]]->state == TrackState::Tracked)
        {
            r_tracked_stracks.push_back(strack_pool[u_track[i]]);
        }
    }


#if defined(ADAS_TRACKER_LOW_SCORE_ASSIGN)

    dists.clear();
    dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

    matches.clear();
    u_track.clear();
    u_detection.clear();
    linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

    for (int i = 0; i < matches.size(); i++)
    {
        STrack *track = r_tracked_stracks[matches[i][0]];
        STrack *det   = &detections[matches[i][1]];
        if (track->state == TrackState::Tracked)
        {
            track->update(*det, this->frame_id);
            activated_stracks.push_back(*track);
        }
        else
        {
            track->re_activate(*det, this->frame_id, false, this->next_id());
            refind_stracks.push_back(*track);
        }
    }
#endif


#if defined(ADAS_TRACKER_LOW_SCORE_ASSIGN)

    for (int i = 0; i < u_track.size(); i++)
    {
        STrack *track = r_tracked_stracks[u_track[i]];
#else
    for (int i = 0; i < r_tracked_stracks.size(); i++)
    {
        STrack *track = r_tracked_stracks[i];
#endif
        if (track->state != TrackState::Lost)
        {
            track->mark_lost();
            lost_stracks.push_back(*track);
        }
    }

    // Deal with unconfirmed tracks, usually tracks with only one beginning frame
    detections.clear();
    detections.assign(detections_cp.begin(), detections_cp.end());

    dists.clear();
    dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

    matches.clear();
    vector<int> u_unconfirmed;
    u_detection.clear();
    linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

    for (int i = 0; i < matches.size(); i++)
    {
        unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
        activated_stracks.push_back(*unconfirmed[matches[i][0]]);
    }

    for (int i = 0; i < u_unconfirmed.size(); i++)
    {
        STrack *track = unconfirmed[u_unconfirmed[i]];
        track->mark_removed();
        removed_stracks.push_back(*track);
    }

    ////////////////// Step 4: Init new stracks //////////////////
    for (int i = 0; i < u_detection.size(); i++)
    {
        STrack *track = &detections[u_detection[i]];
        if (track->score < this->high_thresh)
            continue;
        track->activate(this->frame_id, this->next_id());
        activated_stracks.push_back(*track);
    }

    ////////////////// Step 5: Update state //////////////////
    for (int i = 0; i < this->lost_stracks.size(); i++)
    {
        // Normal case
        int elapsed_frames = this->frame_id - this->lost_stracks[i].end_frame();
        if (this->frame_id < this->lost_stracks[i].end_frame())
        {
            // frame_id exceeds INT_MAX and reset, but end_frame is still near to INT_MAX
            elapsed_frames = frame_id + (INT_MAX - this->lost_stracks[i].end_frame());
        }
        if (elapsed_frames > this->max_time_lost)
        {
            this->lost_stracks[i].mark_removed();
            removed_stracks.push_back(this->lost_stracks[i]);
        }
    }
    for (int i = 0; i < this->tracked_stracks.size(); i++)
    {
        if (this->tracked_stracks[i].state == TrackState::Tracked)
        {
            tracked_stracks_swap.push_back(this->tracked_stracks[i]);
        }
    }
    this->tracked_stracks.clear();
    this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

    this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
    this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

    // std::cout << activated_stracks.size() << std::endl;

    this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
    for (int i = 0; i < lost_stracks.size(); i++)
    {
        this->lost_stracks.push_back(lost_stracks[i]);
    }

    // use removed_stracks directly
    this->lost_stracks = sub_stracks(this->lost_stracks, removed_stracks);

    // // another way that ensure idendical result with original method
    // this->lost_stracks = sub_stracks1(this->lost_stracks, this->removed_stracks);
    // for (int i = 0; i < removed_stracks.size(); i++)
    // {
    //     this->removed_stracks.push_back(removed_stracks[i]);
    // }

    remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

    this->tracked_stracks.clear();
    this->tracked_stracks.assign(resa.begin(), resa.end());
    this->lost_stracks.clear();
    this->lost_stracks.assign(resb.begin(), resb.end());

    this->idSet.clear();
    this->lost_ids.clear();
    this->removed_ids.clear();
    for (auto &lost_strack : this->lost_stracks)
    {
        this->idSet.insert(lost_strack.track_id);
        this->lost_ids.push_back(lost_strack.track_id);
    }
    for (auto &removed_strack : removed_stracks)
    {
        this->removed_ids.push_back(removed_strack.track_id);
    }

    for (int i = 0; i < this->tracked_stracks.size(); i++)
    {
        this->idSet.insert(this->tracked_stracks[i].track_id);
        if (this->tracked_stracks[i].is_activated)
        {
            output_stracks.push_back(this->tracked_stracks[i]);
        }
    }

    return output_stracks;
}

vector<int> BaseTracker::get_removed_ids() const
{
    return this->removed_ids;
}

vector<int> BaseTracker::get_lost_ids() const
{
    return this->lost_ids;
}

int BaseTracker::get_max_time_lost() const
{
    return this->max_time_lost;
}

int BaseTracker::get_frame_id() const
{
    return this->frame_id;
}

vector<STrack *> BaseTracker::joint_stracks(vector<STrack *> &tlista, vector<STrack> &tlistb)
{
    map<int, int> exists;
    vector<STrack *> res;
    for (int i = 0; i < tlista.size(); i++)
    {
        exists.insert(pair<int, int>(tlista[i]->track_id, 1));
        res.push_back(tlista[i]);
    }
    for (int i = 0; i < tlistb.size(); i++)
    {
        int tid = tlistb[i].track_id;
        // is slightly redundant. Once you do exists[tid], the key is inserted anyway even if it wasn’t there.
        if (!exists[tid] || exists.count(tid) == 0)
        {
            exists[tid] = 1;
            res.push_back(&tlistb[i]);
        }
    }
    return res;
}

vector<STrack> BaseTracker::joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
    map<int, int> exists;
    vector<STrack> res;
    for (int i = 0; i < tlista.size(); i++)
    {
        exists.insert(pair<int, int>(tlista[i].track_id, 1));
        res.push_back(tlista[i]);
    }
    for (int i = 0; i < tlistb.size(); i++)
    {
        int tid = tlistb[i].track_id;
        if (!exists[tid] || exists.count(tid) == 0)
        {
            exists[tid] = 1;
            res.push_back(tlistb[i]);
        }
    }
    return res;
}

vector<STrack> BaseTracker::sub_stracks1(vector<STrack> &tlista, vector<STrack> &tlistb)
{
    map<int, STrack> stracks;
    for (int i = 0; i < tlista.size(); i++)
    {
        stracks.insert(pair<int, STrack>(tlista[i].track_id, tlista[i]));
    }

    vector<int> idsToRemove; // Track ids to remove from tlistb
    for (int i = 0; i < tlistb.size(); i++)
    {
        int tid = tlistb[i].track_id;
        if (stracks.count(tid) != 0)
        {
            stracks.erase(tid);

            // Record the index i to remove from tlistb later
            idsToRemove.push_back(i);
        }
    }

    // Remove items from tlistb based on recorded indices
    for (int i = idsToRemove.size() - 1; i >= 0; i--)
    {
        tlistb.erase(tlistb.begin() + idsToRemove[i]);
    }

    vector<STrack> res;
    std::map<int, STrack>::iterator it;
    for (it = stracks.begin(); it != stracks.end(); ++it)
    {
        res.push_back(it->second);
    }

    return res;
}

vector<STrack> BaseTracker::sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
    map<int, STrack> stracks;
    for (int i = 0; i < tlista.size(); i++)
    {
        stracks.insert(pair<int, STrack>(tlista[i].track_id, tlista[i]));
    }
    for (int i = 0; i < tlistb.size(); i++)
    {
        int tid = tlistb[i].track_id;
        if (stracks.count(tid) != 0)
        {
            stracks.erase(tid);
        }
    }

    vector<STrack> res;
    std::map<int, STrack>::iterator it;
    for (it = stracks.begin(); it != stracks.end(); ++it)
    {
        res.push_back(it->second);
    }

    return res;
}

void BaseTracker::remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa,
                                           vector<STrack> &stracksb)
{
    vector<vector<float>> pdist = iou_distance(stracksa, stracksb);
    vector<pair<int, int>> pairs;
    for (int i = 0; i < pdist.size(); i++)
    {
        for (int j = 0; j < pdist[i].size(); j++)
        {
            if (pdist[i][j] < 0.15)
            {
                pairs.push_back(pair<int, int>(i, j));
            }
        }
    }

    vector<int> dupa, dupb;
    for (int i = 0; i < pairs.size(); i++)
    {
        int timep = stracksa[pairs[i].first].frame_id - stracksa[pairs[i].first].start_frame;
        int timeq = stracksb[pairs[i].second].frame_id - stracksb[pairs[i].second].start_frame;
        if (timep > timeq)
            dupb.push_back(pairs[i].second);
        else
            dupa.push_back(pairs[i].first);
    }

    for (int i = 0; i < stracksa.size(); i++)
    {
        vector<int>::iterator iter = find(dupa.begin(), dupa.end(), i);
        if (iter == dupa.end())
        {
            resa.push_back(stracksa[i]);
        }
    }

    for (int i = 0; i < stracksb.size(); i++)
    {
        vector<int>::iterator iter = find(dupb.begin(), dupb.end(), i);
        if (iter == dupb.end())
        {
            resb.push_back(stracksb[i]);
        }
    }
}

void BaseTracker::linear_assignment(vector<vector<float>> &cost_matrix, 
                                    int cost_matrix_size, 
                                    int cost_matrix_size_size,
                                    float thresh, 
                                    vector<vector<int>> &matches, 
                                    vector<int> &unmatched_a,
                                    vector<int> &unmatched_b)
{
    if (cost_matrix.size() == 0)
    {
        for (int i = 0; i < cost_matrix_size; i++)
        {
            unmatched_a.push_back(i);
        }
        for (int i = 0; i < cost_matrix_size_size; i++)
        {
            unmatched_b.push_back(i);
        }
        return;
    }

    vector<int> rowsol;
    vector<int> colsol;

#ifdef ADAS_DEBUG_LAPJV_DENSE_MATCHING
    vector<int> rowsol_dense;
    vector<int> colsol_dense;
    lapjv(cost_matrix, rowsol_dense, colsol_dense, true, thresh);

#elif defined(ADAS_DEBUG_LAPMOD_SPARSE_MATCHING)
    vector<int> rowsol_sparse;
    vector<int> colsol_sparse;
    lapmod(cost_matrix, rowsol_sparse, colsol_sparse, true, thresh);
#endif

    // lapmod sparse matching with submatrix
    // lapmod_submatrix(cost_matrix, rowsol, colsol, true, thresh);
    lapmod(cost_matrix, rowsol, colsol, true, thresh);

    for (int i = 0; i < rowsol.size(); i++)
    {
        if (rowsol[i] >= 0)
        {
            vector<int> match;
            match.push_back(i);
            match.push_back(rowsol[i]);
            matches.push_back(match);
        }
        else
        {
            unmatched_a.push_back(i);
        }
    }

    for (int i = 0; i < colsol.size(); i++)
    {
        if (colsol[i] < 0)
        {
            unmatched_b.push_back(i);
        }
    }
}

vector<vector<float>> BaseTracker::ious(vector<vector<float>> &atlbrs, vector<vector<float>> &btlbrs)
{
    vector<vector<float>> ious;
    if (atlbrs.size() * btlbrs.size() == 0)
        return ious;

    ious.resize(atlbrs.size());
    for (int i = 0; i < ious.size(); i++)
    {
        ious[i].resize(btlbrs.size());
    }

    // bbox_ious
    for (int k = 0; k < btlbrs.size(); k++)
    {
        // vector<float> ious_tmp;
        float box_area = (btlbrs[k][2] - btlbrs[k][0] + 1) * (btlbrs[k][3] - btlbrs[k][1] + 1);
        for (int n = 0; n < atlbrs.size(); n++)
        {
            float iw = min(atlbrs[n][2], btlbrs[k][2]) - max(atlbrs[n][0], btlbrs[k][0]) + 1;
            if (iw > 0)
            {
                float ih = min(atlbrs[n][3], btlbrs[k][3]) - max(atlbrs[n][1], btlbrs[k][1]) + 1;
                if (ih > 0)
                {
                    float ua =
                        (atlbrs[n][2] - atlbrs[n][0] + 1) * (atlbrs[n][3] - atlbrs[n][1] + 1) + box_area - iw * ih;
                    ious[n][k] = iw * ih / ua;
                }
                else
                {
                    ious[n][k] = 0.0;
                }
            }
            else
            {
                ious[n][k] = 0.0;
            }
        }
    }

    return ious;
}

vector<vector<float>> BaseTracker::iou_distance(vector<STrack *> &atracks, vector<STrack> &btracks, int &dist_size,
                                                int &dist_size_size)
{
    vector<vector<float>> cost_matrix;
    if (atracks.size() * btracks.size() == 0)
    {
        dist_size      = atracks.size();
        dist_size_size = btracks.size();
        return cost_matrix;
    }
    vector<vector<float>> atlbrs, btlbrs;
    for (int i = 0; i < atracks.size(); i++)
    {
        atlbrs.push_back(atracks[i]->tlbr);
    }
    for (int i = 0; i < btracks.size(); i++)
    {
        btlbrs.push_back(btracks[i].tlbr);
    }

    dist_size      = atracks.size();
    dist_size_size = btracks.size();

    vector<vector<float>> _ious = ious(atlbrs, btlbrs);

    for (int i = 0; i < _ious.size(); i++)
    {
        vector<float> _iou;
        for (int j = 0; j < _ious[i].size(); j++)
        {
            _iou.push_back(1 - _ious[i][j]);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

vector<vector<float>> BaseTracker::iou_distance(vector<STrack> &atracks, vector<STrack> &btracks)
{
    vector<vector<float>> atlbrs, btlbrs;
    for (int i = 0; i < atracks.size(); i++)
    {
        atlbrs.push_back(atracks[i].tlbr);
    }
    for (int i = 0; i < btracks.size(); i++)
    {
        btlbrs.push_back(btracks[i].tlbr);
    }

    vector<vector<float>> _ious = ious(atlbrs, btlbrs);
    vector<vector<float>> cost_matrix;
    for (int i = 0; i < _ious.size(); i++)
    {
        vector<float> _iou;
        for (int j = 0; j < _ious[i].size(); j++)
        {
            _iou.push_back(1 - _ious[i][j]);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

// ADAS_DEBUG_LAPJV_DENSE_MATCHING
// dense matching with complete extended matrix, origin method
void BaseTracker::lapjv(const vector<vector<float>> &cost, vector<int> &rowsol, vector<int> &colsol, bool extend_cost,
                        float cost_limit, bool return_cost)
{
    vector<vector<float>> cost_c;
    cost_c.assign(cost.begin(), cost.end());

    vector<vector<float>> cost_c_extended;

    int n_rows = cost.size();
    int n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols)
    {
        n = n_rows;
    }
    /*else
    {
        if (!extend_cost)
        {
            cout << "set extend_cost=True" << endl;
            system("pause");
            exit(0);
        }
    }*/

    if (extend_cost || cost_limit < LONG_MAX)
    {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (int i = 0; i < cost_c_extended.size(); i++)
            cost_c_extended[i].resize(n);

        if (cost_limit < LONG_MAX)
        {
            for (int i = 0; i < cost_c_extended.size(); i++)
            {
                for (int j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        }
        else
        {
            float cost_max = -1;
            for (int i = 0; i < cost_c.size(); i++)
            {
                for (int j = 0; j < cost_c[i].size(); j++)
                {
                    if (cost_c[i][j] > cost_max)
                        cost_max = cost_c[i][j];
                }
            }
            for (int i = 0; i < cost_c_extended.size(); i++)
            {
                for (int j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (int i = n_rows; i < cost_c_extended.size(); i++)
        {
            for (int j = n_cols; j < cost_c_extended[i].size(); j++)
            {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double **cost_ptr;
    cost_ptr = new double *[n];
    for (int i      = 0; i < n; i++)
        cost_ptr[i] = new double[n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cost_ptr[i][j] = cost_c[i][j];
        }
    }

    int *x_c = new int[n];
    int *y_c = new int[n];

    // print cost matrix
    /*
    cout<<endl;
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            cout<<setprecision(2)<<std::setw(6)<<cost[i][j];
        }
        cout<<endl;
    }
    cout<<endl;
    */

    int ret = Tracker::lapjv_internal(n, cost_ptr, x_c, y_c);
    /*if (ret != 0)
    {
        cout << "Calculate Wrong!" << endl;
        system("pause");
        exit(0);
    }*/

    double opt = 0.0;

    if (n != n_rows)
    {
        for (int i = 0; i < n; i++)
        {
            if (x_c[i] >= n_cols)
                x_c[i] = -1;
            if (y_c[i] >= n_rows)
                y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++)
        {
            rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; i++)
        {
            colsol[i] = y_c[i];
        }

        /*if (return_cost)
        {
            for (int i = 0; i < rowsol.size(); i++)
            {
                if (rowsol[i] != -1)
                {
                    //cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] << endl;
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }*/
    }
    /*else if (return_cost)
    {
        for (int i = 0; i < rowsol.size(); i++)
        {
            opt += cost_ptr[i][rowsol[i]];
        }
    }*/

    for (int i = 0; i < n; i++)
    {
        delete[] cost_ptr[i];
    }
    delete[] cost_ptr;
    delete[] x_c;
    delete[] y_c;

    cost_ptr = nullptr;
    x_c      = nullptr;
    y_c      = nullptr;
}

// ADAS_DEBUG_LAPMOD_SPARSE_MATCHING
// sparse matching with complete extended matrix
void BaseTracker::lapmod(const vector<vector<float>> &cost, vector<int> &rowsol, vector<int> &colsol, bool extend_cost,
                         float cost_limit, bool return_cost)
{
    vector<vector<float>> cost_c;
    cost_c.assign(cost.begin(), cost.end());

    vector<vector<float>> cost_c_extended;

    int n_rows = cost.size();
    int n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols)
    {
        n = n_rows;
    }
    /*else
    {
        if (!extend_cost)
        {
            cout << "set extend_cost=True" << endl;
            system("pause");
            exit(0);
        }
    }*/

    if (extend_cost || cost_limit < LONG_MAX)
    {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (int i = 0; i < cost_c_extended.size(); i++)
            cost_c_extended[i].resize(n);

        if (cost_limit < LONG_MAX)
        {
            for (int i = 0; i < cost_c_extended.size(); i++)
            {
                for (int j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        }
        else
        {
            float cost_max = -1;
            for (int i = 0; i < cost_c.size(); i++)
            {
                for (int j = 0; j < cost_c[i].size(); j++)
                {
                    if (cost_c[i][j] > cost_max)
                        cost_max = cost_c[i][j];
                }
            }
            for (int i = 0; i < cost_c_extended.size(); i++)
            {
                for (int j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (int i = n_rows; i < cost_c_extended.size(); i++)
        {
            for (int j = n_cols; j < cost_c_extended[i].size(); j++)
            {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        // cost_c.clear();
        // cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    Tracker::cost_t *cost_ptr_sparse = nullptr;
    Tracker::uint_t *ii              = nullptr;
    Tracker::uint_t *kk              = nullptr;
    Tracker::int_t * x1              = nullptr;
    Tracker::int_t * y1              = nullptr;

    int n_extend = n_cols + n_rows;
    ii           = new Tracker::uint_t[(n_extend + 1)];
    x1           = new Tracker::int_t[n_extend];
    y1           = new Tracker::int_t[n_extend];

    // allocate oversized array first
    // the real indexing is determined by ii[]
    cost_ptr_sparse = new Tracker::cost_t[n_extend * n_extend];
    // the indexing of kk is determined by ii
    kk = new Tracker::uint_t[n_extend * n_extend];
    // kk must be sorted according to ii index
    // 0-th row
    // 1-th row

    for (int i = 0; i < n_extend; ++i)
    {
        x1[i] = -1;
        y1[i] = -1;
    }

    for (int i = 0; i <= n_extend; ++i)
    {
        ii[i] = 0;
    }

    // print cost matrix
    /*
    cout<<endl;
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            cout<<setprecision(2)<<std::setw(6)<<cost[i][j];
        }
        cout<<endl;
    }
    cout<<endl;
    */

    // cout<<endl;
    // cout<<"n_rows "<<n_rows<<" n_cols "<<n_cols<<" dense "<<(n_rows+n_cols)*(n_rows+n_cols)<<endl;
    // cout<<endl;

    // must run double loops in this ordering
    // https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
    Tracker::uint_t nonzero_count = 0;
    for (int i = 0; i < n_extend; i++) // rows
    {
        for (int j = 0; j < n_extend; j++) // cols
        {
            if (cost_c_extended[i][j] <= cost_limit) // 1-iou
            {
                // works for dense
                // int dest = j+(i*n_extend); ind;
                // also works
                int dest              = nonzero_count;
                kk[dest]              = j; // column id
                cost_ptr_sparse[dest] = cost_c_extended[i][j];
                ii[i]++; // row count
                nonzero_count++;
            }
        }
    }
    // cout<<(float)nonzero_count/(float)(n_extend*n_extend)<<endl;

    // Step 3: Accumulate counts to get row_ptr
    for (int i = 1; i <= n_extend; ++i)
    {
        ii[i] += ii[i - 1];
    }

    // Step 5: Fix row_ptr values (shift back)
    for (int i = n_extend; i > 0; --i)
    {
        ii[i] = ii[i - 1];
        // cout<<"i "<<i<<" ii[i] "<<ii[i]<<endl;
    }
    ii[0] = 0;
    // cout<<"i "<<0<<" ii[i] "<<ii[0]<<endl;

    int ret1 = Tracker::lapmod_internal(n_extend, cost_ptr_sparse, ii, kk, x1, y1, Tracker::FP_1);
    // fp_version
    // typedef enum fp_t { FP_1 = 1, FP_2 = 2, FP_DYNAMIC = 3 } fp_t;

    /*if (ret != 0)
    {
        cout << "Calculate Wrong!" << endl;
        system("pause");
        exit(0);
    }*/

    double opt = 0.0;

    if (n != n_rows)
    {
        for (int i = 0; i < n_extend; i++)
        {
            if (x1[i] >= n_cols)
                x1[i] = -1;
            if (y1[i] >= n_rows)
                y1[i] = -1;
        }
        for (int i = 0; i < n_rows; i++)
        {
            rowsol[i] = x1[i];
        }
        for (int i = 0; i < n_cols; i++)
        {
            colsol[i] = y1[i];
        }

        /*if (return_cost)
        {
            for (int i = 0; i < rowsol.size(); i++)
            {
                if (rowsol[i] != -1)
                {
                    //cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] << endl;
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }*/
    }
    /*else if (return_cost)
    {
        for (int i = 0; i < rowsol.size(); i++)
        {
            opt += cost_ptr[i][rowsol[i]];
        }
    }*/

    // debug output
    /*cout<<"x"<<endl;
    for (int i = 0; i < n_rows; ++i) {
        cout<<i<<" "<< x1[i]<<endl;
    }
    cout<<"y"<<endl;
    for (int i = 0; i < n_cols; ++i) {
        cout<< i<<" "<<y1[i]<<endl;
    }*/

    delete[] cost_ptr_sparse;
    delete[] ii;
    delete[] kk;
    delete[] x1;
    delete[] y1;

    cost_ptr_sparse = nullptr;
    ii              = nullptr;
    kk              = nullptr;
    x1              = nullptr;
    y1              = nullptr;
}

// sparse matching with submatrix
void BaseTracker::lapmod_submatrix(const vector<vector<float>> &cost, vector<int> &rowsol, vector<int> &colsol,
                                   bool extend_cost, float cost_limit, bool return_cost)
{
    int n_rows = cost.size();
    int n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n_extend = n_cols + n_rows;

// for debugging and compare with lapmod sparse matching
#ifdef ADAS_DEBUG_LAPMOD_SPARSE_COMPARE
    vector<vector<float>> cost_c;
    cost_c.assign(cost.begin(), cost.end());
    vector<vector<float>> cost_c_extended;

    int n = 0;
    if (n_rows == n_cols)
    {
        n = n_rows;
    }
    /*else
    {
        if (!extend_cost)
        {
            cout << "set extend_cost=True" << endl;
            system("pause");
            exit(0);
        }
    }*/

    if (extend_cost || cost_limit < LONG_MAX)
    {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (int i = 0; i < cost_c_extended.size(); i++)
            cost_c_extended[i].resize(n);

        if (cost_limit < LONG_MAX)
        {
            for (int i = 0; i < cost_c_extended.size(); i++)
            {
                for (int j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        }
        else
        {
            float cost_max = -1;
            for (int i = 0; i < cost_c.size(); i++)
            {
                for (int j = 0; j < cost_c[i].size(); j++)
                {
                    if (cost_c[i][j] > cost_max)
                        cost_max = cost_c[i][j];
                }
            }
            for (int i = 0; i < cost_c_extended.size(); i++)
            {
                for (int j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (int i = n_rows; i < cost_c_extended.size(); i++)
        {
            for (int j = n_cols; j < cost_c_extended[i].size(); j++)
            {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        //     cost_c.clear();
        //     cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    Tracker::cost_t *cost_ptr_sparse = nullptr;
    Tracker::uint_t *ii              = nullptr;
    Tracker::uint_t *kk              = nullptr;
    Tracker::int_t * x1              = nullptr;
    Tracker::int_t * y1              = nullptr;

    ii = new Tracker::uint_t[(n_extend + 1)];
    x1 = new Tracker::int_t[n_extend];
    y1 = new Tracker::int_t[n_extend];

    // allocate oversized array first
    // the real indexing is determined by ii[]
    cost_ptr_sparse = new Tracker::cost_t[n_extend * n_extend];
    // the indexing of kk is determined by ii
    kk = new Tracker::uint_t[n_extend * n_extend];
    // kk must be sorted according to ii index
    // 0-th row
    // 1-th row

    for (int i = 0; i < n_extend; ++i)
    {
        x1[i] = -1;
        y1[i] = -1;
    }

    for (int i = 0; i <= n_extend; ++i)
    {
        ii[i] = 0;
    }

    // https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
    Tracker::uint_t nonzero_count = 0;
    for (int i = 0; i < n_extend; i++) // rows
    {
        for (int j = 0; j < n_extend; j++) // cols
        {
            // if(cost_c_extended[i][j]<cost_limit) //1-iou
            //{
            // works for dense
            // int dest = j+(i*n_extend); ind;
            // also works
            int dest              = nonzero_count;
            kk[dest]              = j; // column id
            cost_ptr_sparse[dest] = cost_c_extended[i][j];
            ii[i]++; // row count
            nonzero_count++;
            //}
        }
    }

    // Accumulate counts to get row_ptr
    for (int i = 1; i <= n_extend; ++i)
    {
        ii[i] += ii[i - 1];
    }

    // Fix row_ptr values (shift back)
    for (int i = n_extend; i > 0; --i)
    {
        ii[i] = ii[i - 1];
        // cout<<"i "<<i<<" ii[i] "<<ii[i]<<endl;
    }
    ii[0] = 0;
    // cout<<"i "<<0<<" ii[i] "<<ii[0]<<endl;

    int ret1 = Tracker::lapmod_internal(n_extend, cost_ptr_sparse, ii, kk, x1, y1, Tracker::FP_1);
    // fp_version
    // typedef enum fp_t { FP_1 = 1, FP_2 = 2, FP_DYNAMIC = 3 } fp_t;
    for (int i = 0; i < n_extend; i++)
    {
        if (x1[i] >= n_cols)
            x1[i] = -1;
        if (y1[i] >= n_rows)
            y1[i] = -1;
    }
#endif

    Tracker::int_t * x2                  = nullptr;
    Tracker::int_t * y2                  = nullptr;
    Tracker::cost_t *cost_ptr_sparse_sub = nullptr;
    Tracker::uint_t *iis                 = nullptr;
    Tracker::uint_t *kks                 = nullptr;
    Tracker::int_t * xs                  = nullptr;
    Tracker::int_t * ys                  = nullptr;

    x2  = new Tracker::int_t[n_extend];
    y2  = new Tracker::int_t[n_extend];
    iis = new Tracker::uint_t[(n_extend + 1)];
    xs  = new Tracker::int_t[n_extend];
    ys  = new Tracker::int_t[n_extend];

    // allocate oversized array first
    // the real indexing is determined by ii[]
    // cost_ptr_sparse = new Tracker::cost_t [n_extend*n_extend];
    // the indexing of kk is determined by ii
    // kk = new Tracker::uint_t [n_extend*n_extend];
    // kk must be sorted according to ii index
    // 0-th row
    // 1-th row
    cost_ptr_sparse_sub = new Tracker::cost_t[n_extend * n_extend];
    kks                 = new Tracker::uint_t[n_extend * n_extend];

    for (int i = 0; i < n_extend; ++i)
    {
        x2[i] = -1;
        y2[i] = -1;
    }

    vector<vector<float>> matrix = cost;
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            if (matrix[i][j] > cost_limit)
            {
                matrix[i][j] = SPARSE_VAL;
            }
        }
    }

    // print cost matrix
    /*
    cout<<endl;
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            cout<<setprecision(2)<<std::setw(6)<<cost[i][j];
        }
        cout<<endl;
    }
    cout<<endl;
    */

    // partition the cost matrix into submatrices using union find algorithm
    SubMatrix sub(matrix);
    auto      subMatrices = sub.getSubMatrices();
    // sub.debug(matrix,true);

    // cout<<endl;
    // cout<<"n_rows "<<n_rows<<" n_cols "<<n_cols<<" dense "<<(n_rows+n_cols)*(n_rows+n_cols)<<endl;
    // cout<<endl;

    for (auto &subMatrix : subMatrices)
    {
        auto &vec = subMatrix.second;

        if (vec.size() == 1)
        {
            // vec size is 1,
            // it is the only entry in row "vec[0].first" and column "vec[0].second"
            if (matrix[vec[0].first][vec[0].second] <= cost_limit)
            {
                // assign the matching as long as the cost <= cost_limit
                x2[vec[0].first]  = vec[0].second;
                y2[vec[0].second] = vec[0].first;
            }
        }
        else
        {
            // vec size > 1,
            // more than one entries in row "vec[].first" or column "vec[].second"
            // prepare for sparse linear assignment with cost extension
            // https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
            // for (auto& elem : vec)
            // {
            //     cout << "["<<elem.first<<"]["<<elem.second<<"] "<<matrix[elem.first][elem.second] <<" ";
            // }
            // cout << endl;
            set<int> xSet;
            set<int> ySet;
            for (auto &item : vec)
            {
                xSet.insert(item.first);
                ySet.insert(item.second);
            }
            for (int i = 0; i < n_extend; ++i)
            {
                xs[i] = -1;
                ys[i] = -1;
            }
            for (int i = 0; i <= n_extend; ++i)
            {
                iis[i] = 0;
            }

            int nz_count = 0;
            for (int xVal : xSet)
            {
                for (int yyVal : ySet)
                {
                    int dest                  = nz_count;
                    kks[dest]                 = yyVal; // column id
                    cost_ptr_sparse_sub[dest] = cost[xVal][kks[dest]];
                    // cout<<setprecision(2)<<std::setw(6)<<cost_ptr_sparse_sub[dest];
                    iis[xVal]++; // row count
                    nz_count++;
                }
                for (int xxVal : xSet)
                {
                    int dest                  = nz_count;
                    kks[dest]                 = xxVal + n_cols; // column id
                    cost_ptr_sparse_sub[dest] = cost_limit / 2; // cost_c_extended[xVal][kk[dest]]
                    // cout<<setprecision(2)<<std::setw(6)<<cost_ptr_sparse_sub[dest];
                    iis[xVal]++; // row count
                    nz_count++;
                }
                // cout<<endl;
            }
            for (int yVal : ySet)
            {
                for (int yyVal : ySet)
                {
                    int dest                  = nz_count;
                    kks[dest]                 = yyVal;          // column id
                    cost_ptr_sparse_sub[dest] = cost_limit / 2; // cost_c_extended[yVal][kk[dest]];
                    // cout<<setprecision(2)<<std::setw(6)<<cost_ptr_sparse_sub[dest];
                    iis[yVal + n_rows]++; // row count
                    nz_count++;
                }
                for (int xxVal : xSet)
                {
                    int dest                  = nz_count;
                    kks[dest]                 = xxVal + n_cols; // column id
                    cost_ptr_sparse_sub[dest] = 0.0f;           // cost_c_extended[xVal][kk[dest]]
                    // cout<<setprecision(2)<<std::setw(6)<<cost_ptr_sparse_sub[dest];
                    iis[yVal + n_rows]++; // row count
                    nz_count++;
                }
                // cout<<endl;
            }
            // Accumulate counts to get row_ptr
            for (int i = 1; i <= n_extend; ++i)
            {
                iis[i] += iis[i - 1];
            }

            // Fix row_ptr values (shift back)
            for (int i = n_extend; i > 0; --i)
            {
                iis[i] = iis[i - 1];
                // cout<<"i "<<i<<" iis[i] "<<iis[i]<<endl;
            }
            iis[0] = 0;

            int ret1 = Tracker::lapmod_internal(n_extend, cost_ptr_sparse_sub, iis, kks, xs, ys, Tracker::FP_1);
            // cout<<"matching x"<<endl;
            for (int i = 0; i < n_rows; i++)
            {
                // cout << xs[i]<<" ";
                if (xs[i] != -1 && xs[i] < n_cols && xSet.find(i) != xSet.end() && ySet.find(xs[i]) != ySet.end())
                {
                    if (x2[i] == -1 || (x2[i] != -1 && cost[i][x2[i]] > cost[i][xs[i]]))
                    {
                        x2[i] = xs[i];
                    }
                }
            }
            // cout<<endl;
            // cout<<"matching y"<<endl;
            for (int i = 0; i < n_cols; i++)
            {
                // cout << ys[i]<<" ";
                if (ys[i] != -1 && ys[i] < n_rows && ySet.find(i) != ySet.end() && xSet.find(ys[i]) != xSet.end())
                {
                    if (y2[i] == -1 || (y2[i] != -1 && cost[y2[i]][i] > cost[ys[i]][i]))
                    {
                        y2[i] = ys[i];
                    }
                }
            }
            // cout<<endl;
        }
    }

    double opt = 0.0;

    for (int i = 0; i < n_extend; i++)
    {
        if (x2[i] >= n_cols)
        {
            x2[i] = -1;
        }
        if (y2[i] >= n_rows)
        {
            y2[i] = -1;
        }
    }
    for (int i = 0; i < n_rows; i++)
    {
        rowsol[i] = x2[i];
    }
    for (int i = 0; i < n_cols; i++)
    {
        colsol[i] = y2[i];
    }

// debug output
// cout<<"x"<<endl;
// for (int i = 0; i < n_rows; ++i) {
//     if(x1[i] != x2[i])
//         cout<<i<<" "<< x1[i] << " "<<x2[i] <<"--------------------------------------------------"<<endl;
// }
// cout<<"y"<<endl;
// for (int i = 0; i < n_cols; ++i) {
//     if(y1[i] != y2[i])
//         cout<< i<<" "<<y1[i] << " "<<y2[i] <<"--------------------------------------------------"<<endl;
// }

#ifdef ADAS_DEBUG_LAPMOD_SPARSE_COMPARE
    delete[] cost_ptr_sparse;
    delete[] ii;
    delete[] kk;
    delete[] x1;
    delete[] y1;

    cost_ptr_sparse = nullptr;
    ii              = nullptr;
    kk              = nullptr;
    x1              = nullptr;
    y1              = nullptr;
#endif

    delete[] cost_ptr_sparse_sub;
    delete[] kks;
    delete[] iis;
    delete[] xs;
    delete[] ys;
    delete[] x2;
    delete[] y2;

    cost_ptr_sparse_sub = nullptr;
    iis                 = nullptr;
    kks                 = nullptr;
    xs                  = nullptr;
    ys                  = nullptr;
    x2                  = nullptr;
    y2                  = nullptr;
}

int BaseTracker::next_id()
{
    static int _count      = 0;
    int        local_count = 0;

    _count = (_count + 1) % (ADAS_TRACKER_MAX_ID);

    // avoid id collision
    // loop while the id is found in idSet and the local_count increment doesn't exceed ADAS_TRACKER_MAX_ID
    while (this->idSet.find(_count) != this->idSet.end() && local_count < ADAS_TRACKER_MAX_ID)
    {
        // _count increment
        _count = (_count + 1) % (ADAS_TRACKER_MAX_ID);
        // local_count can avoid infinite loop where all ids are in idSet
        local_count++;
    }
    // the local_count increment exceed ADAS_TRACKER_MAX_ID
    if (local_count >= ADAS_TRACKER_MAX_ID)
    {
        // error! id collision still happened
    }
    return _count;
}

UnionFind::UnionFind(int n)
{
    parent.resize(n);
    rank.resize(n, 0);
    // initialize parent as the node itself
    for (int i = 0; i < n; ++i)
    {
        parent[i] = i;
    }
}

int UnionFind::find(int u)
{
    // find parent through recursive node traversal
    // if parent[u] == u, that means u is a root node
    if (parent[u] != u)
    {
        parent[u] = find(parent[u]);
    }
    return parent[u];
}

void UnionFind::unionSets(int u, int v)
{
    int rootU = find(u);
    int rootV = find(v);
    // union two nodes according to root ranks
    if (rootU != rootV)
    {
        // Union by rank
        if (rank[rootU] > rank[rootV])
        {
            parent[rootV] = rootU;
        }
        else if (rank[rootU] < rank[rootV])
        {
            parent[rootU] = rootV;
        }
        else
        {
            parent[rootV] = rootU;
            rank[rootU]++;
        }
    }
}

SubMatrix::SubMatrix(const vector<vector<float>> &matrix)
{
    // get rows and cols from matrix size
    rows = matrix.size();
    cols = matrix[0].size();

    // initialize UnionFind as individual row and column
    UnionFind uf(rows + cols);

    // Find union elements based on row and column indices
    // except for SPARSE_VAL (cost = 1 - iou, iou = 0, SPARSE_VAL = 1.0)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (matrix[i][j] != SPARSE_VAL)
            {
                uf.unionSets(i, rows + j); // Union row i with column j
            }
        }
    }

    // construct subMatrices according to roots
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (matrix[i][j] != SPARSE_VAL)
            {
                int root = uf.find(i);                     // Find the root of the current element's group
                this->subMatrices[root].push_back({i, j}); // Store the element in its group
            }
        }
    }
}

unordered_map<int, vector<pair<int, int>>> SubMatrix::getSubMatrices() const
{
    return this->subMatrices;
}

void SubMatrix::debug(const vector<vector<float>> &matrix, bool toggle) const
{
    // Output sub-matrices
    cout << "rows " << rows << " cols " << cols << " subMatrix size " << this->subMatrices.size() << endl;
    if (toggle)
    {
        for (auto &subMatrix : this->subMatrices)
        {
            cout << "Group " << subMatrix.first << ":\n";
            for (auto &elem : subMatrix.second)
            {
                cout << "[" << elem.first << "][" << elem.second << "] " << matrix[elem.first][elem.second] << " ";
            }
            cout << "\n";
        }
    }
}
