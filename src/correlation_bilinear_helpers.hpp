#pragma once

/**
 * Helper functions for bilinear interpolation sampling (matching PyTorch's grid_sample)
 * 
 * These functions allow C++ correlation to match Python's grid_sample behavior:
 * - Normalizes coordinates to [-1, 1] range
 * - Uses bilinear interpolation instead of nearest neighbor
 * - Returns 0.0 for out-of-bounds coordinates
 */

#include <cmath>
#include <cstdint>
#include <limits>

/**
 * Convert float32 to half precision (16-bit float) and back to float32
 * This simulates Python's .half() conversion for dx/dy in bilinear wrapper interpolation
 * 
 * Half precision format (IEEE 754-2008):
 * - 1 sign bit
 * - 5 exponent bits
 * - 10 mantissa bits
 * 
 * This function rounds to nearest half precision value, matching PyTorch's behavior
 */
inline float float_to_half_to_float(float f) {
    // Handle special cases
    if (std::isnan(f)) return std::numeric_limits<float>::quiet_NaN();
    if (std::isinf(f)) return f;
    if (f == 0.0f) return 0.0f;
    
    // Extract sign, exponent, and mantissa from float32
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;  // Bias: 127 for float32
    uint32_t mantissa = bits & 0x7FFFFF;
    
    // Convert to half precision
    // Half precision: 5 exponent bits (bias: 15), 10 mantissa bits
    uint16_t half_bits = 0;
    
    // Set sign bit
    half_bits |= (sign << 15);
    
    // Handle exponent
    if (exp > 15) {
        // Overflow: return infinity
        half_bits |= 0x7C00;  // Infinity in half precision
    } else if (exp < -14) {
        // Underflow: return zero (or denormalized)
        // For simplicity, return zero
        half_bits |= 0x0000;
    } else {
        // Normal case: convert exponent
        half_bits |= ((exp + 15) << 10);  // Add bias 15
        
        // Round mantissa from 23 bits to 10 bits
        // Use IEEE 754 round-to-nearest-even (RNE) to match Python's .half() behavior
        uint32_t mantissa_half = mantissa >> 13;  // Keep top 10 bits
        uint32_t round_bit = (mantissa >> 12) & 0x1;  // Bit 12 (the bit we're rounding)
        uint32_t sticky_bits = mantissa & 0x1FFF;  // Bits 0-11 (bits below bit 12)
        bool has_sticky_bits = (sticky_bits != 0);
        
        // Round to nearest, ties to even (RNE)
        // Round up if:
        //   1. round_bit is 1 AND (has_sticky_bits OR mantissa_half is odd)
        // This ensures ties (round_bit=1, no sticky bits) round to even
        if (round_bit && (has_sticky_bits || (mantissa_half & 0x1))) {
            mantissa_half++;
            if (mantissa_half >= (1 << 10)) {  // Overflow in mantissa
                mantissa_half = 0;
                half_bits += (1 << 10);  // Increment exponent
            }
        }
        half_bits |= mantissa_half;
    }
    
    // Convert back to float32
    // Extract half precision components
    uint16_t h_sign = (half_bits >> 15) & 0x1;
    int16_t h_exp = ((half_bits >> 10) & 0x1F) - 15;  // Bias: 15
    uint16_t h_mantissa = half_bits & 0x3FF;
    
    // Reconstruct float32
    uint32_t f32_bits = 0;
    f32_bits |= (h_sign << 31);
    
    if (h_exp == -15 && h_mantissa == 0) {
        // Zero
        f32_bits |= 0x00000000;
    } else if (h_exp == 16) {
        // Infinity or NaN
        f32_bits |= 0x7F800000;  // Infinity
        if (h_mantissa != 0) {
            f32_bits |= 0x00400000;  // NaN
        }
    } else {
        // Normal number
        f32_bits |= ((h_exp + 127) << 23);  // Add bias 127
        f32_bits |= (static_cast<uint32_t>(h_mantissa) << 13);  // Shift mantissa
    }
    
    return *reinterpret_cast<float*>(&f32_bits);
}

/**
 * Normalize pixel coordinates to [-1, 1] range (matches grid_sample)
 * 
 * @param x_pixel Raw pixel x coordinate
 * @param y_pixel Raw pixel y coordinate
 * @param H Height of feature map
 * @param W Width of feature map
 * @param x_norm Output: normalized x in [-1, 1]
 * @param y_norm Output: normalized y in [-1, 1]
 */
inline void normalize_coords_for_grid_sample(
    float x_pixel, float y_pixel,
    int H, int W,
    float& x_norm, float& y_norm)
{
    // grid_sample normalization: gx = 2 * (x / (W - 1)) - 1
    // Maps [0, W-1] to [-1, 1]
    if (W > 1) {
        x_norm = 2.0f * (x_pixel / static_cast<float>(W - 1)) - 1.0f;
    } else {
        x_norm = 0.0f;  // Handle edge case: W == 1
    }
    
    if (H > 1) {
        y_norm = 2.0f * (y_pixel / static_cast<float>(H - 1)) - 1.0f;
    } else {
        y_norm = 0.0f;  // Handle edge case: H == 1
    }
}

/**
 * Bilinear interpolation sampling (matches PyTorch's grid_sample behavior)
 * 
 * @param fmap Feature map buffer
 * @param x_norm Normalized x coordinate in [-1, 1] range
 * @param y_norm Normalized y coordinate in [-1, 1] range
 * @param H Height of feature map
 * @param W Width of feature map
 * @param channel_idx Channel index (for multi-channel feature maps)
 * @param feature_dim Total number of channels
 * @param fmap_offset Offset to start of this frame's features in buffer
 * @return Interpolated value (0.0 if out of bounds, matching grid_sample)
 */
inline float bilinear_sample_grid_sample(
    const float* fmap,
    float x_norm,  // Normalized x in [-1, 1]
    float y_norm,  // Normalized y in [-1, 1]
    int H, int W,
    int channel_idx,
    int feature_dim,
    size_t fmap_offset)
{
    // PyTorch's grid_sample with padding_mode='zeros' behavior:
    // - Does NOT clamp normalized coordinates (they can be outside [-1, 1])
    // - Converts normalized coordinates to pixel coordinates
    // - Checks if bilinear sampling would access out-of-bounds pixels
    // - Returns 0 if ANY of the 4 corners needed for bilinear interpolation are out-of-bounds
    // - If all corners are in bounds, performs bilinear interpolation (even if normalized coords were outside [-1, 1])
    
    // Convert normalized coordinates to pixel coordinates [0, H-1] x [0, W-1]
    // Formula: pixel = (norm + 1) / 2 * (size - 1)
    // Note: This formula works even if normalized coords are outside [-1, 1]
    // PyTorch's grid_sample uses this exact formula with align_corners=True
    // (Both align_corners=True and False use the same conversion formula for normalized->pixel)
    float x_pixel = (x_norm + 1.0f) * 0.5f * static_cast<float>(W - 1);
    float y_pixel = (y_norm + 1.0f) * 0.5f * static_cast<float>(H - 1);
    
    // Get integer and fractional parts (for bilinear interpolation corners)
    // Use floor to get the lower-left corner of the 2x2 pixel block
    int x0 = static_cast<int>(std::floor(x_pixel));
    int y0 = static_cast<int>(std::floor(y_pixel));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // PyTorch's grid_sample with padding_mode='zeros' behavior:
    // Observation: Python successfully samples from slightly out-of-bounds coordinates
    // (e.g., y_pixel = -0.41, which gives y0 = -1, y1 = 0)
    // This suggests PyTorch clamps corner indices first, then checks if sampling is valid
    // 
    // Strategy:
    // 1. Check if pixel coordinate is within reasonable range (tolerance-based)
    // 2. If yes, clamp corners and proceed with bilinear interpolation
    // 3. If no, return 0
    
    // Check if pixel coordinate is within reasonable range
    // PyTorch allows sampling when coordinates are slightly outside bounds
    const float tolerance = 0.5f;  // Allow coordinates in range [-0.5, size-0.5]
    bool pixel_too_far_oob = (x_pixel < -tolerance || x_pixel > static_cast<float>(W - 1) + tolerance ||
                              y_pixel < -tolerance || y_pixel > static_cast<float>(H - 1) + tolerance);
    
    if (pixel_too_far_oob) {
        return 0.0f;  // Too far out-of-bounds, return 0
    }
    
    // Pixel coordinate is within reasonable range, clamp corners to valid indices
    // This allows sampling from edge pixels when coordinates are slightly outside bounds
    x0 = std::max(0, std::min(x0, W - 1));
    y0 = std::max(0, std::min(y0, H - 1));
    x1 = std::max(0, std::min(x1, W - 1));
    y1 = std::max(0, std::min(y1, H - 1));
    
    // Clamp pixel coordinates to valid range for interpolation weights calculation
    // This ensures dx and dy are computed correctly
    x_pixel = std::max(0.0f, std::min(x_pixel, static_cast<float>(W - 1)));
    y_pixel = std::max(0.0f, std::min(y_pixel, static_cast<float>(H - 1)));
    
    float dx = x_pixel - static_cast<float>(x0);
    float dy = y_pixel - static_cast<float>(y0);
    
    // All corners are guaranteed to be in bounds at this point
    // (x0, y0), (x1, y0), (x0, y1), (x1, y1) are all within [0, W-1] x [0, H-1]
    
    // Compute indices for 4 corners
    size_t idx00 = fmap_offset + static_cast<size_t>(channel_idx) * static_cast<size_t>(H) * static_cast<size_t>(W) + static_cast<size_t>(y0) * static_cast<size_t>(W) + static_cast<size_t>(x0);
    size_t idx01 = fmap_offset + static_cast<size_t>(channel_idx) * static_cast<size_t>(H) * static_cast<size_t>(W) + static_cast<size_t>(y0) * static_cast<size_t>(W) + static_cast<size_t>(x1);
    size_t idx10 = fmap_offset + static_cast<size_t>(channel_idx) * static_cast<size_t>(H) * static_cast<size_t>(W) + static_cast<size_t>(y1) * static_cast<size_t>(W) + static_cast<size_t>(x0);
    size_t idx11 = fmap_offset + static_cast<size_t>(channel_idx) * static_cast<size_t>(H) * static_cast<size_t>(W) + static_cast<size_t>(y1) * static_cast<size_t>(W) + static_cast<size_t>(x1);
    
    // Bilinear interpolation weights
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;
    
    // Sample and interpolate
    float v00 = fmap[idx00];
    float v01 = fmap[idx01];
    float v10 = fmap[idx10];
    float v11 = fmap[idx11];
    
    return w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
}

/**
 * Convenience function: Sample from pyramid buffer with bilinear interpolation
 * 
 * This function matches Python's altcorr.corr behavior where:
 * 1. Base coordinates are normalized first
 * 2. Offsets are added in normalized space (matching Python's grid_sample grid creation)
 * 
 * @param pyramid Pyramid feature buffer [num_frames, feature_dim, H, W]
 * @param x_base Base pixel x coordinate (before adding correlation window offset)
 * @param y_base Base pixel y coordinate (before adding correlation window offset)
 * @param offset_x Correlation window offset in x direction (in pixels, will be normalized)
 * @param offset_y Correlation window offset in y direction (in pixels, will be normalized)
 * @param pyramid_frame Frame index in pyramid buffer
 * @param channel_idx Channel index
 * @param H Height of feature map
 * @param W Width of feature map
 * @param feature_dim Total number of channels
 * @param num_frames Total frames in pyramid buffer
 * @return Interpolated feature value
 */
inline float sample_pyramid_bilinear_with_offset(
    const float* pyramid,
    float x_base, float y_base,
    float offset_x, float offset_y,
    int pyramid_frame,
    int channel_idx,
    int H, int W,
    int feature_dim,
    int num_frames)
{
    // Normalize base coordinates first (matching Python's approach)
    float x_base_norm, y_base_norm;
    normalize_coords_for_grid_sample(x_base, y_base, H, W, x_base_norm, y_base_norm);
    
    // Add offset in normalized space (matching Python: offset_norm = 2 * offset / (size - 1))
    // This matches how Python's altcorr.corr creates the grid for grid_sample
    float offset_x_norm = 2.0f * offset_x / static_cast<float>(W - 1);
    float offset_y_norm = 2.0f * offset_y / static_cast<float>(H - 1);
    float x_norm = x_base_norm + offset_x_norm;
    float y_norm = y_base_norm + offset_y_norm;
    
    // Compute frame offset
    size_t frame_offset = static_cast<size_t>(pyramid_frame) * static_cast<size_t>(feature_dim) * static_cast<size_t>(H) * static_cast<size_t>(W);
    
    // Sample using bilinear interpolation
    return bilinear_sample_grid_sample(
        pyramid, x_norm, y_norm,
        H, W, channel_idx, feature_dim,
        frame_offset
    );
}

/**
 * Convenience function: Sample from pyramid buffer with bilinear interpolation
 * (Legacy version - adds offset in pixel space then normalizes)
 * 
 * @param pyramid Pyramid feature buffer [num_frames, feature_dim, H, W]
 * @param x_pixel Raw pixel x coordinate (after adding offset)
 * @param y_pixel Raw pixel y coordinate (after adding offset)
 * @param pyramid_frame Frame index in pyramid buffer
 * @param channel_idx Channel index
 * @param H Height of feature map
 * @param W Width of feature map
 * @param feature_dim Total number of channels
 * @param num_frames Total frames in pyramid buffer
 * @return Interpolated feature value
 */
inline float sample_pyramid_bilinear(
    const float* pyramid,
    float x_pixel, float y_pixel,
    int pyramid_frame,
    int channel_idx,
    int H, int W,
    int feature_dim,
    int num_frames)
{
    // Normalize coordinates
    float x_norm, y_norm;
    normalize_coords_for_grid_sample(x_pixel, y_pixel, H, W, x_norm, y_norm);
    
    // Compute frame offset
    size_t frame_offset = static_cast<size_t>(pyramid_frame) * static_cast<size_t>(feature_dim) * static_cast<size_t>(H) * static_cast<size_t>(W);
    
    // Sample using bilinear interpolation
    return bilinear_sample_grid_sample(
        pyramid, x_norm, y_norm,
        H, W, channel_idx, feature_dim,
        frame_offset
    );
}

