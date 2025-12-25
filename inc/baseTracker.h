#pragma once

#include <iostream>
#include <vector>
#include <climits>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <set>

#include "sTrack.h"

using namespace std;

// for two stage hungarian
// matches with two lists (main & secondary)
#define ADAS_TRACKER_SECONDARY_LIST
// matches with two lists (high scores & low scores)
// #define ADAS_TRACKER_LOW_SCORE_ASSIGN
// for submatrix
#define SPARSE_VAL (1.0f)

// caution! next_id() with small ADAS_TRACKER_MAX_ID will cause id collision or slow down the id selection
// ADAS_TRACKER_MAX_ID should be (several times) greater than (_max_time_lost*ADAS_TRACKER_MAX_NUM_DETECT_BOX)
// for tracking accuracy evaluation
// const int ADAS_TRACKER_MAX_ID = (INT_MAX);

// for production
const int ADAS_TRACKER_MAX_ID = 1000;

const int ADAS_TRACKER_MAX_NUM_DETECT_BOX = 300;

class BaseTracker
{
public:
    BaseTracker(int frame_rate = 30, int track_buffer = 30);
    BaseTracker(float _track_thresh, float _high_thresh, float _match_thresh, int _max_time_lost);
    ~BaseTracker();
// #ifdef ADAS_TRACKER_SECONDARY_LIST
    // 1) matches (primary) objects with tracklets
    // 2) matches lost tracklets with secondary objects
//     vector<STrack> update(const vector<STrack> &objects, const vector<STrack> &secondary_objects);
// #else
    // 1) matches objects with tracklets
    vector<STrack> update(const vector<STrack> &objects);
// #endif
    vector<int> get_lost_ids() const;
    vector<int> get_removed_ids() const;
    int         get_max_time_lost() const;
    int         get_frame_id() const;

private:
    vector<STrack *> joint_stracks(vector<STrack *> &tlista, vector<STrack> &tlistb);
    vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

    vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
    vector<STrack> sub_stracks1(vector<STrack> &tlista, vector<STrack> &tlistb);
    void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa,
                                  vector<STrack> &stracksb);

    void linear_assignment(vector<vector<float>> &cost_matrix, int cost_matrix_size, int cost_matrix_size_size,
                           float thresh, vector<vector<int>> &matches, vector<int> &unmatched_a,
                           vector<int> &unmatched_b);
    vector<vector<float>> iou_distance(vector<STrack *> &atracks, vector<STrack> &btracks, int &dist_size,
                                       int &dist_size_size);
    vector<vector<float>> iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
    vector<vector<float>> ious(vector<vector<float>> &atlbrs, vector<vector<float>> &btlbrs);

    void lapjv(const vector<vector<float>> &cost, vector<int> &rowsol, vector<int> &colsol, bool extend_cost = false,
               float cost_limit = LONG_MAX, bool return_cost = true);
    void lapmod(const vector<vector<float>> &cost, vector<int> &rowsol, vector<int> &colsol, bool extend_cost = false,
                float cost_limit = LONG_MAX, bool return_cost = true);
    void lapmod_submatrix(const vector<vector<float>> &cost, vector<int> &rowsol, vector<int> &colsol,
                          bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

    int next_id();

public:
    vector<int> removed_ids;
    vector<int> lost_ids;

private:
    float track_thresh;
    float high_thresh;
    float match_thresh;
    int   frame_id;
    int   max_time_lost;

    vector<STrack> tracked_stracks;
    vector<STrack> lost_stracks;
    // vector<STrack> removed_stracks;

    unordered_set<int> idSet;
};

class UnionFind
{
private:
    // parent of node
    vector<int> parent;
    // rank of node
    vector<int> rank;

public:
    UnionFind(int n);
    // find root node
    int find(int u);
    // union two nodes according to root ranks
    void unionSets(int u, int v);
};

class SubMatrix
{
public:
    unordered_map<int, vector<pair<int, int>>> getSubMatrices() const;

private:
    // size of row and column
    int rows;
    int cols;
    // Map to store sub-matrices
    unordered_map<int, vector<pair<int, int>>> subMatrices;

public:
    SubMatrix(const vector<vector<float>> &matrix);
    void debug(const vector<vector<float>> &matrix, bool toggle) const;
};
