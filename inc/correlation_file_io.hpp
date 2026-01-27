#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <cstring>  // For std::memcpy
#include "logger.hpp"

/**
 * Utility functions for saving correlation input/output parameters to binary files
 * Used for comparison with Python correlation implementation
 */
namespace correlation_file_io {

/**
 * Save a float array to a binary file
 * @param filename Output filename
 * @param data Pointer to data array
 * @param num_elements Number of elements to write
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_float_array(const std::string& filename, const float* data, size_t num_elements, 
                            std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), sizeof(float) * num_elements);
    file.close();
    if (logger) logger->info("Saved {} floats to {}", num_elements, filename);
    return true;
}

/**
 * Save an int32 array to a binary file
 * @param filename Output filename
 * @param data Pointer to data array
 * @param num_elements Number of elements to write
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_int32_array(const std::string& filename, const int32_t* data, size_t num_elements,
                            std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), sizeof(int32_t) * num_elements);
    file.close();
    if (logger) logger->info("Saved {} int32s to {}", num_elements, filename);
    return true;
}

/**
 * Save correlation inputs and outputs for comparison with Python
 * @param frame_num Frame number (for filename)
 * @param coords Reprojected coordinates [num_active, 2, P, P]
 * @param kk Linear patch indices [num_active]
 * @param jj Target frame indices [num_active]
 * @param ii Patch indices [num_active] (not used in correlation, but saved for reference)
 * @param gmap Patch features [num_gmap_frames * M, 128, 3, 3]
 * @param fmap1 Pyramid level 0 [num_frames, 128, fmap1_H, fmap1_W]
 * @param fmap2 Pyramid level 1 [num_frames, 128, fmap2_H, fmap2_W]
 * @param corr Correlation output [num_active, D, D, P, P, 2]
 * @param num_active Number of active edges
 * @param M Patches per frame
 * @param P Patch size
 * @param D Correlation window diameter
 * @param num_frames Number of frames in pyramid buffers
 * @param num_gmap_frames Number of frames in gmap ring buffer
 * @param fmap1_H Height of fmap1
 * @param fmap1_W Width of fmap1
 * @param fmap2_H Height of fmap2
 * @param fmap2_W Width of fmap2
 * @param feature_dim Feature dimension (128)
 * @param logger Optional logger
 */
inline void save_correlation_data(
    int frame_num,
    const float* coords,
    const int* kk,
    const int* jj,
    const int* ii,
    const float* gmap,
    const float* fmap1,
    const float* fmap2,
    const float* corr,
    int num_active,
    int M,
    int P,
    int D,
    int num_frames,
    int num_gmap_frames,
    int fmap1_H,
    int fmap1_W,
    int fmap2_H,
    int fmap2_W,
    int feature_dim,
    std::shared_ptr<spdlog::logger> logger = nullptr
) {
    if (logger) {
        logger->info("Saving correlation data for frame {}:", frame_num);
    }
    
    // Save coords [num_active, 2, P, P]
    std::string coords_file = "corr_frame" + std::to_string(frame_num) + "_coords.bin";
    save_float_array(coords_file, coords, num_active * 2 * P * P, logger);
    
    // Save indices
    std::vector<int32_t> kk_int32(num_active);
    std::vector<int32_t> jj_int32(num_active);
    std::vector<int32_t> ii_int32(num_active);
    for (int i = 0; i < num_active; i++) {
        kk_int32[i] = static_cast<int32_t>(kk[i]);
        jj_int32[i] = static_cast<int32_t>(jj[i]);
        ii_int32[i] = static_cast<int32_t>(ii[i]);
    }
    
    std::string kk_file = "corr_frame" + std::to_string(frame_num) + "_kk.bin";
    save_int32_array(kk_file, kk_int32.data(), num_active, logger);
    
    std::string jj_file = "corr_frame" + std::to_string(frame_num) + "_jj.bin";
    save_int32_array(jj_file, jj_int32.data(), num_active, logger);
    
    std::string ii_file = "corr_frame" + std::to_string(frame_num) + "_ii.bin";
    save_int32_array(ii_file, ii_int32.data(), num_active, logger);
    
    // Save gmap - extract only the patches we need (based on kk indices)
    // gmap shape: [num_gmap_frames][M][128][3][3]
    // We need to extract patches for each kk[e] index
    // Mapping: ii1 = kk[e] % (M * num_gmap_frames), then frame_idx = ii1 / M, patch_idx = ii1 % M
    const int D_gmap = 3;  // gmap patch size (matches P=3)
    std::vector<float> gmap_slices(num_active * feature_dim * D_gmap * D_gmap);
    const int gmap_patch_stride = feature_dim * D_gmap * D_gmap;  // Size of one patch in gmap [128][3][3]
    const int gmap_frame_stride = M * gmap_patch_stride;  // Size of one frame in gmap [M][128][3][3]
    for (int e = 0; e < num_active; e++) {
        int kk_idx = kk[e];
        // Map kk to gmap index: ii1 = kk_idx % (M * num_gmap_frames)
        int ii1 = kk_idx % (M * num_gmap_frames);
        int frame_idx = ii1 / M;
        int patch_idx = ii1 % M;
        if (frame_idx >= 0 && frame_idx < num_gmap_frames && patch_idx >= 0 && patch_idx < M) {
            // Access: gmap[frame_idx][patch_idx][c][y][x]
            const float* src = gmap + frame_idx * gmap_frame_stride + patch_idx * gmap_patch_stride;
            float* dst = gmap_slices.data() + e * gmap_patch_stride;
            std::memcpy(dst, src, sizeof(float) * gmap_patch_stride);
        }
    }
    std::string gmap_file = "corr_frame" + std::to_string(frame_num) + "_gmap.bin";
    save_float_array(gmap_file, gmap_slices.data(), num_active * feature_dim * D_gmap * D_gmap, logger);
    
    // Save fmap1 slices - extract frames based on jj indices
    // fmap1 shape: [1][num_frames][128][fmap1_H][fmap1_W]
    // We need to extract the full feature map for each jj[e] frame
    // Mapping: jj1 = jj[e] % num_frames
    std::vector<float> fmap1_slices(num_active * feature_dim * fmap1_H * fmap1_W);
    const int fmap1_frame_stride = feature_dim * fmap1_H * fmap1_W;  // Size of one frame in fmap1 [128][H][W]
    for (int e = 0; e < num_active; e++) {
        int jj_idx = jj[e];
        // Map jj to fmap1 index: jj1 = jj_idx % num_frames
        int jj1 = jj_idx % num_frames;
        if (jj1 >= 0 && jj1 < num_frames) {
            // Access: fmap1[0][jj1][c][y][x] - batch dimension is always 0
            const float* src = fmap1 + jj1 * fmap1_frame_stride;
            float* dst = fmap1_slices.data() + e * fmap1_frame_stride;
            std::memcpy(dst, src, sizeof(float) * fmap1_frame_stride);
        }
    }
    std::string fmap1_file = "corr_frame" + std::to_string(frame_num) + "_fmap1.bin";
    save_float_array(fmap1_file, fmap1_slices.data(), num_active * feature_dim * fmap1_H * fmap1_W, logger);
    
    // Save fmap2 slices - extract frames based on jj indices
    // fmap2 shape: [1][num_frames][128][fmap2_H][fmap2_W]
    // Mapping: jj1 = jj[e] % num_frames
    std::vector<float> fmap2_slices(num_active * feature_dim * fmap2_H * fmap2_W);
    const int fmap2_frame_stride = feature_dim * fmap2_H * fmap2_W;  // Size of one frame in fmap2 [128][H][W]
    for (int e = 0; e < num_active; e++) {
        int jj_idx = jj[e];
        // Map jj to fmap2 index: jj1 = jj_idx % num_frames
        int jj1 = jj_idx % num_frames;
        if (jj1 >= 0 && jj1 < num_frames) {
            // Access: fmap2[0][jj1][c][y][x] - batch dimension is always 0
            const float* src = fmap2 + jj1 * fmap2_frame_stride;
            float* dst = fmap2_slices.data() + e * fmap2_frame_stride;
            std::memcpy(dst, src, sizeof(float) * fmap2_frame_stride);
        }
    }
    std::string fmap2_file = "corr_frame" + std::to_string(frame_num) + "_fmap2.bin";
    save_float_array(fmap2_file, fmap2_slices.data(), num_active * feature_dim * fmap2_H * fmap2_W, logger);
    
    // Save correlation output [num_active, D, D, P, P, 2]
    std::string corr_file = "corr_frame" + std::to_string(frame_num) + "_corr.bin";
    save_float_array(corr_file, corr, num_active * D * D * P * P * 2, logger);
    
    // Save metadata (parameters)
    std::string meta_file = "corr_frame" + std::to_string(frame_num) + "_meta.bin";
    std::ofstream meta(meta_file, std::ios::binary);
    if (meta.is_open()) {
        int32_t params[] = {
            static_cast<int32_t>(num_active),
            static_cast<int32_t>(M),
            static_cast<int32_t>(P),
            static_cast<int32_t>(D),
            static_cast<int32_t>(num_frames),
            static_cast<int32_t>(num_gmap_frames),
            static_cast<int32_t>(fmap1_H),
            static_cast<int32_t>(fmap1_W),
            static_cast<int32_t>(fmap2_H),
            static_cast<int32_t>(fmap2_W),
            static_cast<int32_t>(feature_dim)
        };
        meta.write(reinterpret_cast<const char*>(params), sizeof(params));
        meta.close();
        if (logger) logger->info("Saved metadata to {}", meta_file);
    }
    
    if (logger) {
        logger->info("Correlation data saved for frame {}", frame_num);
    }
}

} // namespace correlation_file_io

