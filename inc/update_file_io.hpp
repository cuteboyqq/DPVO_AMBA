#ifndef UPDATE_FILE_IO_HPP
#define UPDATE_FILE_IO_HPP

#include <string>
#include <vector>
#include <fstream>
#include <spdlog/spdlog.h>

namespace update_file_io {

/**
 * Save update model input data to binary file
 * 
 * @param filename Output filename
 * @param data Pointer to float data array
 * @param size Number of elements
 * @param logger Logger instance (optional)
 * @param name Description of the data being saved (for logging)
 */
inline void save_float_array(const std::string& filename, const float* data, size_t size,
                             std::shared_ptr<spdlog::logger> logger = nullptr,
                             const std::string& name = "") {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) {
            logger->error("[UpdateFileIO] Failed to open file for writing: {}", filename);
        }
        return;
    }
    
    file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    file.close();
    
    if (logger) {
        logger->info("[UpdateFileIO] Saved {} to {} ({} elements, {} bytes)", 
                    name.empty() ? "data" : name, filename, size, size * sizeof(float));
    }
}

/**
 * Save update model input index data (ii, jj, kk) to binary file
 * 
 * @param filename Output filename
 * @param data Pointer to int32 data array
 * @param size Number of elements
 * @param logger Logger instance (optional)
 * @param name Description of the data being saved (for logging)
 */
inline void save_int32_array(const std::string& filename, const int32_t* data, size_t size,
                             std::shared_ptr<spdlog::logger> logger = nullptr,
                             const std::string& name = "") {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) {
            logger->error("[UpdateFileIO] Failed to open file for writing: {}", filename);
        }
        return;
    }
    
    file.write(reinterpret_cast<const char*>(data), size * sizeof(int32_t));
    file.close();
    
    if (logger) {
        logger->info("[UpdateFileIO] Saved {} to {} ({} elements, {} bytes)", 
                    name.empty() ? "data" : name, filename, size, size * sizeof(int32_t));
    }
}

/**
 * Save update model output data to binary file
 * 
 * @param filename Output filename
 * @param data Pointer to float data array
 * @param size Number of elements
 * @param logger Logger instance (optional)
 * @param name Description of the data being saved (for logging)
 */
inline void save_output(const std::string& filename, const float* data, size_t size,
                       std::shared_ptr<spdlog::logger> logger = nullptr,
                       const std::string& name = "") {
    save_float_array(filename, data, size, logger, name);
}

/**
 * Save update model metadata to text file
 * 
 * @param filename Output filename
 * @param frame Frame number
 * @param num_active Number of active edges
 * @param max_edge Maximum number of edges (model input dimension)
 * @param dim Feature dimension (typically 384)
 * @param corr_dim Correlation dimension (typically 882)
 * @param logger Logger instance (optional)
 */
inline void save_metadata(const std::string& filename, int frame, int num_active, 
                         int max_edge, int dim, int corr_dim,
                         std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        if (logger) {
            logger->error("[UpdateFileIO] Failed to open metadata file for writing: {}", filename);
        }
        return;
    }
    
    file << "frame=" << frame << "\n";
    file << "num_active=" << num_active << "\n";
    file << "MAX_EDGE=" << max_edge << "\n";
    file << "DIM=" << dim << "\n";
    file << "CORR_DIM=" << corr_dim << "\n";
    file.close();
    
    if (logger) {
        logger->info("[UpdateFileIO] Saved metadata to {} (frame={}, num_active={}, MAX_EDGE={}, DIM={}, CORR_DIM={})",
                    filename, frame, num_active, max_edge, dim, corr_dim);
    }
}

} // namespace update_file_io

#endif // UPDATE_FILE_IO_HPP

