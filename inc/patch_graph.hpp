#pragma once
#include "se.hpp"
#include <cstdint>

constexpr int BUFFER_SIZE = 4096;
constexpr int PATCHES_PER_FRAME = 4;
constexpr int PATCH_SIZE = 3;
constexpr int MAX_EDGES = 360;
constexpr int NET_DIM = 384;

struct Vec2 { float x, y; };
struct Vec3 { float x, y, z; };

class PatchGraph {
public:
    static constexpr int N = BUFFER_SIZE;
    static constexpr int M = PATCHES_PER_FRAME;
    static constexpr int P = PATCH_SIZE;

    // ---- counters ----
    int m_n;              // number of frames
    int m_m;              // number of patches
    int m_num_edges;
    int m_num_edges_inac;

    // ---- frame data ----
    int64_t m_tstamps[N];
    SE3 m_poses[N];

    // patches: (N, M, 3, P, P)
    float m_patches[N][M][3][P][P];

    int m_ix[N * M];   // patch → frame index Alsiter 2025-12-25 added

    // intrinsics: fx fy cx cy
    float m_intrinsics[N][4];

    // ---- map ----
    Vec3 m_points[N * M];
    uint8_t m_colors[N][M][3];

    // ---- index mapping ----
    // int m_index[N + 1];
    int m_index[N][M];
    int m_index_map[N + 1];

    // ---- active edges ----
    float m_net[MAX_EDGES][NET_DIM];
    int m_ii[MAX_EDGES];
    int m_jj[MAX_EDGES];
    int m_kk[MAX_EDGES];
    float m_weight[MAX_EDGES][2];  // [num_edges, 2] - weight channels (w0 for x, w1 for y, matching Python [1, M, 2])
    float m_target[MAX_EDGES * 2];  // [num_edges * 2] - target (x, y) per edge: target[e*2+0]=x, target[e*2+1]=y

    // ---- inactive edges ----
    int m_ii_inac[MAX_EDGES];
    int m_jj_inac[MAX_EDGES];
    int m_kk_inac[MAX_EDGES];
    float m_weight_inac[MAX_EDGES][2];  // [num_edges, 2] - weight channels (w0 for x, w1 for y, matching Python [1, M, 2])
    float m_target_inac[MAX_EDGES * 2];  // [num_edges * 2] - target (x, y) per edge

public:
    PatchGraph();

    void reset();
    void normalize();

private:
    void _normalizeDepth(float scale);
};
