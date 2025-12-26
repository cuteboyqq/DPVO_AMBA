#include "dpvo.hpp"
#include "net.hpp" // Patchifier
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include "projective_ops.hpp"
#include "correlation_kernel.hpp"
static_assert(sizeof(((PatchGraph*)0)->m_index[0]) ==
              sizeof(int) * PatchGraph::M,
              "PatchGraph layout mismatch");

// -------------------------------------------------------------
// Constructor
// -------------------------------------------------------------
DPVO::DPVO(const DPVOConfig& cfg, int ht, int wd)
    : DPVO(cfg, ht, wd, nullptr)
{
}

DPVO::DPVO(const DPVOConfig& cfg, int ht, int wd, Config_S* config)
    : m_cfg(cfg),
      m_ht(ht), m_wd(wd),
      m_counter(0),
      m_is_initialized(false),
      m_DIM(384),    // same as NET_DIM
      m_P(PatchGraph::P),
      m_pmem(cfg.BUFFER_SIZE),
      m_mem(cfg.BUFFER_SIZE)
{
    // fmap sizes
    m_fmap1_H = ht;
    m_fmap1_W = wd;
    m_fmap2_H = ht / 4;
    m_fmap2_W = wd / 4;

	const int M = cfg.PATCHES_PER_FRAME;
    // allocate float arrays
    m_imap  = new float[m_pmem * m_cfg.PATCHES_PER_FRAME * m_DIM]();
    m_gmap  = new float[m_pmem * m_cfg.PATCHES_PER_FRAME * 128 * m_P * m_P]();
    m_fmap1 = new float[1 * m_mem * 128 * m_fmap1_H * m_fmap1_W]();
    m_fmap2 = new float[1 * m_mem * 128 * m_fmap2_H * m_fmap2_W]();

	// -----------------------------
    // Zero-initialize (important!)
    // -----------------------------
    std::memset(m_imap,  0, sizeof(float) * m_pmem * M * m_DIM);
    std::memset(m_gmap,  0, sizeof(float) * m_pmem * M * 128 * m_P * m_P);
    std::memset(m_fmap1, 0, sizeof(float) * m_mem * 128 * m_fmap1_H * m_fmap1_W);
    std::memset(m_fmap2, 0, sizeof(float) * m_mem * 128 * m_fmap2_H * m_fmap2_W);

    // PatchGraph constructor will initialize internal arrays
	// Patchifier default patch size 3
    m_patchifier = Patchifier(3);
    
    // Initialize update model if config provided (no threading needed for sequential execution)
    if (config != nullptr) {
        m_updateModel = std::make_unique<DPVOUpdate>(config, nullptr);
    }
}

void DPVO::setUpdateModel(Config_S* config)
{
    if (config != nullptr && m_updateModel == nullptr) {
        m_updateModel = std::make_unique<DPVOUpdate>(config, nullptr);
    }
}

DPVO::~DPVO() {
    // Update model will be automatically destroyed by unique_ptr
    delete[] m_imap;
    delete[] m_gmap;
    delete[] m_fmap1;
    delete[] m_fmap2;
}

// -------------------------------------------------------------
// Main entry (equivalent to dpvo.py __call__)
// -------------------------------------------------------------
void DPVO::run(int64_t timestamp,
               const uint8_t* image,
               const float intrinsics[4],
               int H, int W)
{
    if (m_pg.m_n + 1 >= PatchGraph::N)
        throw std::runtime_error("PatchGraph buffer overflow");

    const int n = m_pg.m_n;
    const int pm = n % m_pmem;
    const int mm = n % m_mem;
    const int M  = m_cfg.PATCHES_PER_FRAME;
    const int P  = m_P;

    // -------------------------------------------------
    // 1. Patchify (WRITE DIRECTLY INTO BUFFERS)
    // -------------------------------------------------
    // float* imap_dst = &m_imap[imap_idx(pm, 0, 0)];
    // float* gmap_dst = &m_gmap[gmap_idx(pm, 0, 0, 0, 0)];
    // float* fmap1_dst = &m_fmap1[fmap1_idx(0, mm, 0, 0, 0)];
    // float* fmap2_dst = &m_fmap2[fmap2_idx(0, mm, 0, 0, 0)];
	m_cur_imap  = &m_imap[imap_idx(pm, 0, 0)];
	m_cur_gmap  = &m_gmap[gmap_idx(pm, 0, 0, 0, 0)];
	m_cur_fmap1 = &m_fmap1[fmap1_idx(0, mm, 0, 0, 0)];


    float patches[M * 3 * P * P];
    uint8_t clr[M * 3];

    m_patchifier.forward(
        image, H, W,
        m_cur_fmap1,     // full-res fmap
        m_cur_imap,
        m_cur_gmap,
        patches,
        clr,
        M
    );

    // -------------------------------------------------
    // 2. Bookkeeping
    // -------------------------------------------------
    m_tlist.push_back(timestamp);
    m_pg.m_tstamps[n] = timestamp;

    for (int k = 0; k < 4; k++)
        m_pg.m_intrinsics[n][k] = intrinsics[k];  // divide by RES if needed

    // -------------------------------------------------
    // 3. Pose initialization
    // -------------------------------------------------
    if (n > 0)
        m_pg.m_poses[n] = m_pg.m_poses[n - 1];

    // -------------------------------------------------
    // 4. Patch depth initialization (CRITICAL)
    // -------------------------------------------------
    for (int i = 0; i < M; i++) {
        int base = (i * 3 + 2) * P * P;
        for (int y = 0; y < P; y++)
            for (int x = 0; x < P; x++)
                patches[base + y * P + x] = 1.0f; // Python default
    }

    // -------------------------------------------------
    // 5. Store patches + colors into PatchGraph
    // -------------------------------------------------
    for (int i = 0; i < M; i++) {
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < P; y++) {
                for (int x = 0; x < P; x++) {
                    int idx = (i * 3 + c) * P * P + y * P + x;
                    m_pg.m_patches[n][i][c][y][x] = patches[idx];
                }
            }
        }
        for (int c = 0; c < 3; c++)
            m_pg.m_colors[n][i][c] = clr[i * 3 + c];
    }

    // -------------------------------------------------
    // 6. Downsample fmap → fmap2 (Python avg_pool2d)
    // -------------------------------------------------
    for (int c = 0; c < 128; c++) {
        for (int y = 0; y < m_fmap2_H; y++) {
            for (int x = 0; x < m_fmap2_W; x++) {
                float sum = 0.0f;
                for (int dy = 0; dy < 4; dy++)
                    for (int dx = 0; dx < 4; dx++)
                        sum += m_cur_fmap1[c * H * W +
                            (y * 4 + dy) * W +
                            (x * 4 + dx)];
                m_cur_fmap1[c * m_fmap2_H * m_fmap2_W +
                          y * m_fmap2_W + x] = sum / 16.0f;
            }
        }
    }

    // -------------------------------------------------
    // 7. Counters
    // -------------------------------------------------
    m_pg.m_n++;
    m_pg.m_m += M;
    m_counter++;

    // -------------------------------------------------
    // 8. Build edges
    // -------------------------------------------------
    std::vector<int> kk, jj;
    edgesForward(kk, jj);
    appendFactors(kk, jj);
    edgesBackward(kk, jj);
    appendFactors(kk, jj);

    // -------------------------------------------------
    // 9. Optimization
    // -------------------------------------------------
    if (m_is_initialized) {
        update();
        keyframe();
    } else if (m_pg.m_n >= 8) {
        m_is_initialized = true;
        for (int i = 0; i < 12; i++)
            update();
    }
}


// -------------------------------------------------------------
// Update (NN + BA stub)
// -------------------------------------------------------------
// void DPVO::update() {
//     if (m_pg.m_num_edges == 0) return;

//     // NN update + reprojection will go here
//     // BA_CV28(...) will go here
// }


void DPVO::update()
{
    const int num_active = m_pg.m_num_edges;
    if (num_active == 0)
        return;

    const int M = m_cfg.PATCHES_PER_FRAME;
    const int P = m_P;

    // -------------------------------------------------
    // 1. Reprojection
    // -------------------------------------------------
    std::vector<float> coords(num_active * 2 * P * P); // [num_active, 2, P, P]
    reproject(
        m_pg.m_ii, m_pg.m_jj, m_pg.m_kk,
        num_active,
        coords.data()
    );

    // -------------------------------------------------
    // 2. Correlation
    // -------------------------------------------------
    std::vector<float> corr(num_active * 2 * P * P); // flatten stacked corr1+corr2
    computeCorrelation(
		m_gmap,
		m_cur_fmap1,      // pyramid0
		m_fmap2,          // pyramid1
		coords.data(),
		m_pg.m_kk,        // ii
		m_pg.m_jj,        // jj
		num_active,
		M,
		P,
		m_fmap1_H, m_fmap1_W,
		128,
		corr.data()
	);


    // -------------------------------------------------
    // 3. Context slice from imap
    // -------------------------------------------------
    std::vector<float> ctx(num_active * m_DIM);
    for (int e = 0; e < num_active; e++) {
        int kk_idx = m_pg.m_kk[e] % (M * m_pmem);
        std::memcpy(
            &ctx[e * m_DIM],
            &m_imap[kk_idx * m_DIM],
            sizeof(float) * m_DIM
        );
    }

    // -------------------------------------------------
    // 4. Network update (DPVO Update Model Inference)
    // -------------------------------------------------
    std::vector<float> delta(num_active * 2);
    std::vector<float> weight(num_active);
    
    if (m_updateModel != nullptr) {
        // Model expects fixed shapes: [1, 384, 768, 1] for net/inp, [1, 882, 768, 1] for corr, [1, 768, 1] for indices
        const int MODEL_EDGE_COUNT = 768;
        const int CORR_DIM = 882; // Correlation feature dimension
        
        // Prepare input data - pad or truncate to MODEL_EDGE_COUNT
        const int num_edges_to_process = std::min(num_active, MODEL_EDGE_COUNT);
        
        // Allocate model input buffers
        std::vector<float> net_input(1 * 384 * MODEL_EDGE_COUNT * 1, 0.0f);
        std::vector<float> inp_input(1 * 384 * MODEL_EDGE_COUNT * 1, 0.0f);
        std::vector<float> corr_input(1 * CORR_DIM * MODEL_EDGE_COUNT * 1, 0.0f);
        std::vector<int32_t> ii_input(1 * MODEL_EDGE_COUNT * 1, 0);
        std::vector<int32_t> jj_input(1 * MODEL_EDGE_COUNT * 1, 0);
        std::vector<int32_t> kk_input(1 * MODEL_EDGE_COUNT * 1, 0);
        
        // Reshape and copy net data: [num_active, 384] -> [1, 384, 768, 1]
        // Layout: [batch, channels, spatial, 1] = [1, 384, 768, 1]
        for (int e = 0; e < num_edges_to_process; e++) {
            for (int d = 0; d < 384; d++) {
                // net: [1, 384, 768, 1] - channel major, then spatial
                net_input[0 * 384 * MODEL_EDGE_COUNT * 1 + d * MODEL_EDGE_COUNT * 1 + e * 1 + 0] = m_pg.m_net[e][d];
                // inp: [1, 384, 768, 1] - same layout
                inp_input[0 * 384 * MODEL_EDGE_COUNT * 1 + d * MODEL_EDGE_COUNT * 1 + e * 1 + 0] = ctx[e * 384 + d];
            }
        }
        
        // Reshape correlation: [num_active * 18] -> [1, 882, 768, 1]
        // Note: corr is [num_active, 2, P, P] = [num_active, 18] where P=3
        // Model expects [1, 882, 768, 1] - we'll pad the correlation dimension
        const int corr_per_edge = 2 * P * P; // 18
        for (int e = 0; e < num_edges_to_process; e++) {
            // Copy original correlation (18 values per edge)
            for (int c = 0; c < corr_per_edge && c < CORR_DIM; c++) {
                corr_input[0 * CORR_DIM * MODEL_EDGE_COUNT * 1 + c * MODEL_EDGE_COUNT * 1 + e * 1 + 0] = 
                    corr[e * corr_per_edge + c];
            }
            // Rest of CORR_DIM is zero-padded
        }
        
        // Copy indices: [num_active] -> [1, 768, 1]
        for (int e = 0; e < num_edges_to_process; e++) {
            ii_input[0 * MODEL_EDGE_COUNT * 1 + e * 1 + 0] = static_cast<int32_t>(m_pg.m_ii[e]);
            jj_input[0 * MODEL_EDGE_COUNT * 1 + e * 1 + 0] = static_cast<int32_t>(m_pg.m_jj[e]);
            kk_input[0 * MODEL_EDGE_COUNT * 1 + e * 1 + 0] = static_cast<int32_t>(m_pg.m_kk[e]);
        }
        
        // Call update model inference synchronously
        DPVOUpdate_Prediction pred;
        if (m_updateModel->runInference(
                net_input.data(),
                inp_input.data(),
                corr_input.data(),
                ii_input.data(),
                jj_input.data(),
                kk_input.data(),
                m_updateFrameCounter++,
                pred))
        {
            // Extract outputs: net_out [1, 384, 768, 1], d_out [1, 2, 768, 1], w_out [1, 2, 768, 1]
            // d_out contains delta: [1, 2, 768, 1] -> [num_edges, 2]
            // w_out contains weight: [1, 2, 768, 1] -> we'll use first channel
            
            if (pred.dOutBuff != nullptr && pred.wOutBuff != nullptr) {
                // Extract delta from d_out: [1, 2, 768, 1]
                for (int e = 0; e < num_edges_to_process; e++) {
                    // d_out layout: [batch, channels, spatial, 1] = [1, 2, 768, 1]
                    delta[e * 2 + 0] = pred.dOutBuff[0 * 2 * MODEL_EDGE_COUNT * 1 + 0 * MODEL_EDGE_COUNT * 1 + e * 1 + 0];
                    delta[e * 2 + 1] = pred.dOutBuff[0 * 2 * MODEL_EDGE_COUNT * 1 + 1 * MODEL_EDGE_COUNT * 1 + e * 1 + 0];
                    
                    // w_out layout: [1, 2, 768, 1] - use first channel for weight
                    weight[e] = pred.wOutBuff[0 * 2 * MODEL_EDGE_COUNT * 1 + 0 * MODEL_EDGE_COUNT * 1 + e * 1 + 0];
                }
                
                // Update m_pg.m_net with net_out if available
                if (pred.netOutBuff != nullptr) {
                    // net_out: [1, 384, 768, 1] -> [num_edges, 384]
                    for (int e = 0; e < num_edges_to_process; e++) {
                        for (int d = 0; d < 384; d++) {
                            m_pg.m_net[e][d] = pred.netOutBuff[0 * 384 * MODEL_EDGE_COUNT * 1 + d * MODEL_EDGE_COUNT * 1 + e * 1 + 0];
                        }
                    }
                }
            }
            
            // Free prediction buffers
            if (pred.netOutBuff) delete[] pred.netOutBuff;
            if (pred.dOutBuff) delete[] pred.dOutBuff;
            if (pred.wOutBuff) delete[] pred.wOutBuff;
        }
        
        // If we have more edges than processed, use zero delta/weight for remaining
        for (int e = num_edges_to_process; e < num_active; e++) {
            delta[e * 2 + 0] = 0.0f;
            delta[e * 2 + 1] = 0.0f;
            weight[e] = 0.0f;
        }
    } else {
        // Fallback: zero delta and weight if no update model
        std::fill(delta.begin(), delta.end(), 0.0f);
        std::fill(weight.begin(), weight.end(), 0.0f);
    }

    // -------------------------------------------------
    // 5. Compute target positions
    // -------------------------------------------------
    for (int e = 0; e < num_active; e++) {
        int center = (P / 2) * P + (P / 2);
        float cx = coords[e * 2 * P * P + center * 2 + 0];
        float cy = coords[e * 2 * P * P + center * 2 + 1];

        m_pg.m_target[e * 2 + 0] = cx + delta[e * 2 + 0];
        m_pg.m_target[e * 2 + 1] = cy + delta[e * 2 + 1];
        m_pg.m_weight[e] = weight[e];
    }

    // -------------------------------------------------
    // 6. Bundle Adjustment
    // -------------------------------------------------
    // try {
    //     bool run_global_ba = false;
    //     for (int e = 0; e < num_active; e++) {
    //         if (m_pg.m_ii[e] < m_pg.m_n - m_cfg.REMOVAL_WINDOW - 1) {
    //             run_global_ba = true;
    //             break;
    //         }
    //     }

    //     if (run_global_ba) {
    //         runGlobalBA();  // implement your global BA
    //     } else {
    //         int t0 = m_is_initialized ?
    //                  std::max(m_pg.m_n - m_cfg.OPTIMIZATION_WINDOW, 1) : 1;

    //         fastBA(
    //             m_pg.m_poses,
    //             m_pg.m_patches,
    //             m_pg.m_intrinsics,
    //             m_pg.m_target,
    //             m_pg.m_weight,
    //             m_pg.m_ii,
    //             m_pg.m_jj,
    //             m_pg.m_kk,
    //             t0,
    //             m_pg.m_n
    //         );
    //     }
    // } catch (...) {
    //     std::cerr << "Warning: BA failed...\n";
    // }

    // -------------------------------------------------
    // 7. Update point cloud
    // -------------------------------------------------
//     updatePointCloud(); // implement pops::point_cloud equivalent
}



// -------------------------------------------------------------
// Keyframe logic (minimal, safe version)
// -------------------------------------------------------------
// void DPVO::keyframe() {
//     int i = m_pg.m_n - m_cfg.KEYFRAME_INDEX - 1;
//     int j = m_pg.m_n - m_cfg.KEYFRAME_INDEX + 1;

//     if (i < 0 || j < 0) return;

//     float m = motionMagnitude(i, j) + motionMagnitude(j, i);
//     if (0.5f * m < m_cfg.KEYFRAME_THRESH) {
//         int k = m_pg.m_n - m_cfg.KEYFRAME_INDEX;

//         // shift frames left
//         for (int f = k; f < m_pg.m_n - 1; f++) {
//             m_pg.m_tstamps[f] = m_pg.m_tstamps[f + 1];
//             m_pg.m_poses[f]   = m_pg.m_poses[f + 1];
//         }

//         m_pg.m_n--;
//         m_pg.m_m -= PatchGraph::M;
//     }
// }
void DPVO::keyframe() {

    int n = m_pg.m_n;

    int i = n - m_cfg.KEYFRAME_INDEX - 1;
    int j = n - m_cfg.KEYFRAME_INDEX + 1;
    if (i < 0 || j < 0) return;

    float m = motionMagnitude(i, j) + motionMagnitude(j, i);

    // =============================================================
    // Phase A: Keyframe removal decision
    // =============================================================
    if (0.5f * m < m_cfg.KEYFRAME_THRESH) {

        int k = n - m_cfg.KEYFRAME_INDEX;

        // ---------------------------------------------------------
        // Phase B1: remove edges touching frame k
        // ---------------------------------------------------------
        bool remove[MAX_EDGES] = {false};

        int num_active = m_pg.m_num_edges;
        for (int e = 0; e < num_active; e++) {
            if (m_pg.m_ii[e] == k || m_pg.m_jj[e] == k)
                remove[e] = true;
        }

        removeFactors(remove, /*store=*/false);

        // ---------------------------------------------------------
        // Phase B2: reindex remaining edges
        // ---------------------------------------------------------
        num_active = m_pg.m_num_edges;
        for (int e = 0; e < num_active; e++) {

            if (m_pg.m_ii[e] > k) {
                m_pg.m_ii[e] -= 1;
                m_pg.m_kk[e] -= PatchGraph::M;
            }

            if (m_pg.m_jj[e] > k) {
                m_pg.m_jj[e] -= 1;
            }
        }

        // ---------------------------------------------------------
        // Phase B3: shift per-frame data
        // ---------------------------------------------------------
        for (int f = k; f < n - 1; f++) {

			m_pg.m_tstamps[f] = m_pg.m_tstamps[f + 1];
			m_pg.m_poses[f]   = m_pg.m_poses[f + 1];

			// ---- arrays → memcpy ----
			// void* std::memcpy(
			//     void*       dest,
			//     const void* src,
			//     std::size_t count
			// );
			// Copy all patch data of frame f+1 into frame f
			std::memcpy(
				m_pg.m_patches[f],
				m_pg.m_patches[f + 1],
				sizeof(m_pg.m_patches[0])
			);
			// Copy all patch colors from frame f+1 → frame f
			std::memcpy(
				m_pg.m_colors[f],
				m_pg.m_colors[f + 1],
				sizeof(m_pg.m_colors[0])
			);
			// Copy camera intrinsics of frame f+1 → frame f
			std::memcpy(
				m_pg.m_intrinsics[f],
				m_pg.m_intrinsics[f + 1],
				sizeof(m_pg.m_intrinsics[0])
			);

			// ---- ring buffers / lightweight objects ----
			m_imap[f % m_pmem] = m_imap[(f + 1) % m_pmem];
			m_gmap[f % m_pmem] = m_gmap[(f + 1) % m_pmem];
			m_fmap1[f % m_mem] = m_fmap1[(f + 1) % m_mem];
			m_fmap2[f % m_mem] = m_fmap2[(f + 1) % m_mem];
		}

        m_pg.m_n--;
        m_pg.m_m -= PatchGraph::M;
    }

    // =============================================================
    // Phase C: remove old edges outside optimization window
    // =============================================================
    {
        bool remove[MAX_EDGES] = {false};
        int num_active = m_pg.m_num_edges;

        for (int e = 0; e < num_active; e++) {
            if (m_pg.m_ix[m_pg.m_kk[e]] < m_pg.m_n - m_cfg.REMOVAL_WINDOW)
                remove[e] = true;
        }

        removeFactors(remove, /*store=*/true);
    }
}


// -------------------------------------------------------------
// // Edge construction (forward)
// 		- Forward edges connect:
// 		- all patches from recent frames
// 		- to the newest frame (n - 1)
// -------------------------------------------------------------
void DPVO::edgesForward(std::vector<int>& kk,
                        std::vector<int>& jj) {
    kk.clear();
    jj.clear();

    int r = m_cfg.PATCH_LIFETIME;
    int t0 = PatchGraph::M * std::max(m_pg.m_n - r, 0);
    int t1 = PatchGraph::M * std::max(m_pg.m_n - 1, 0);

    for (int k = t0; k < t1; k++) {
        kk.push_back(k);
        jj.push_back(m_pg.m_n - 1);
    }
}

// -------------------------------------------------------------
// Edge construction (backward)
// 		- Backward edges connect:
// 		- patches from the newest frame
// 		- to all frames in the lifetime window
// -------------------------------------------------------------
void DPVO::edgesBackward(std::vector<int>& kk,
                         std::vector<int>& jj) {
    kk.clear();
    jj.clear();

    int r = m_cfg.PATCH_LIFETIME;
    int t0 = PatchGraph::M * std::max(m_pg.m_n - 1, 0);
    int t1 = PatchGraph::M * m_pg.m_n;

    for (int k = t0; k < t1; k++) {
        for (int f = std::max(m_pg.m_n - r, 0); f < m_pg.m_n; f++) {
            kk.push_back(k);
            jj.push_back(f);
        }
    }
}

// -------------------------------------------------------------
// Append factors (CRITICAL FIXED VERSION)
// 1️⃣ What (ii, jj, kk) mean in DPVO
// In DPVO, each edge (factor) connects:
// patch (landmark)  ↔  pose (frame)
// The naming comes from factor-graph convention:
// Array	Meaning				Node type
// ii		patch index			Landmark node
// jj		frame index			Pose node
// kk		helper index		(frame, patch) linear ID
// -------------------------------------------------------------
void DPVO::appendFactors(const std::vector<int>& kk,
                         const std::vector<int>& jj) {
    int numNew = kk.size();
    if (m_pg.m_num_edges + numNew > MAX_EDGES) return;

    int base = m_pg.m_num_edges;

    for (int i = 0; i < numNew; i++) {
        int k = kk[i];
        int frame = k / PatchGraph::M;
        int patch = k % PatchGraph::M;

        m_pg.m_kk[base + i] = k;
        m_pg.m_jj[base + i] = jj[i];
        m_pg.m_ii[base + i] = m_pg.m_index[frame][patch];

        // zero NET_DIM manually
        for (int d = 0; d < NET_DIM; d++) {
            m_pg.m_net[base + i][d] = 0.0f;
        }
    }

    m_pg.m_num_edges += numNew;
}

// -------------------------------------------------------------
// Remove factors (already correct, kept)
// -------------------------------------------------------------
void DPVO::removeFactors(const bool* mask, bool store) {
    PatchGraph& pg = m_pg;

    const int num_active = pg.m_num_edges;
    if (num_active == 0) return;

    bool m[MAX_EDGES];

    for (int i = 0; i < num_active; i++) {
        m[i] = mask ? mask[i] : false;
    }

    // store inactive
    if (store) {
        int w = pg.m_num_edges_inac;
        for (int i = 0; i < num_active; i++) {
            if (!m[i]) continue;
            if (w >= MAX_EDGES) break;

            pg.m_ii_inac[w]     = pg.m_ii[i];
            pg.m_jj_inac[w]     = pg.m_jj[i];
            pg.m_kk_inac[w]     = pg.m_kk[i];
            pg.m_weight_inac[w]= pg.m_weight[i];
            pg.m_target_inac[w]= pg.m_target[i];
            w++;
        }
        pg.m_num_edges_inac = w;
    }

    // compact
    int write = 0;
    for (int read = 0; read < num_active; read++) {
        if (m[read]) continue;

        if (write != read) {
            pg.m_ii[write]     = pg.m_ii[read];
            pg.m_jj[write]     = pg.m_jj[read];
            pg.m_kk[write]     = pg.m_kk[read];
            pg.m_weight[write] = pg.m_weight[read];
            pg.m_target[write] = pg.m_target[read];
            std::memcpy(pg.m_net[write],
                        pg.m_net[read],
                        sizeof(float) * NET_DIM);
        }
        write++;
    }

    pg.m_num_edges = write;
}

// -------------------------------------------------------------
// Motion magnitude stub
// -------------------------------------------------------------
float DPVO::motionMagnitude(int, int) {
    return 1.0f;
}


void DPVO::reproject(
    const int* ii,
    const int* jj,
    const int* kk,
    int num_edges,
    float* coords_out)
{
    if (num_edges <= 0)
        return;

    // Flattened pointers to patches and intrinsics
    float* patches_flat = &m_pg.m_patches[0][0][0][0][0];
    float* intrinsics_flat = &m_pg.m_intrinsics[0][0];

    // Call flattened transform
    pops::transform(
        m_pg.m_poses,         // SE3 poses [N]
        patches_flat,         // flattened patches
        intrinsics_flat,      // flattened intrinsics
        ii, jj, kk,           // indices
        num_edges,            // number of edges
        m_cfg.PATCHES_PER_FRAME,
        m_P,
        coords_out            // output [num_edges][2][P][P] flattened
    );

    // Output layout already matches Python coords.permute(0,1,4,2,3)
    // Each edge: [2][P][P] → 2 channels: u,v
}



void DPVO::terminate() {}
