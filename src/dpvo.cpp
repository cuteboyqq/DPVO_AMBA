#include "dpvo.hpp"
#include "patchify.hpp" // Patchifier
#include "update.hpp"   // DPVOUpdate
#include <algorithm>
#include <cstring>
#include <cstdlib>  // For std::abort()
#include <stdexcept>
#include <chrono>
#include <random>   // For random depth initialization
#include "projective_ops.hpp"
#include "correlation_kernel.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
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
      m_mem(cfg.BUFFER_SIZE),
      m_patchifier(3, 384),  // Initialize with patch_size=3, DIM=384 (matches INet output channels)
      m_currentTimestamp(0),
      m_pg()  // Explicitly initialize PatchGraph (calls reset() which sets m_n=0)
{
    // Ensure PatchGraph is properly initialized - call reset() explicitly
    m_pg.reset();
    // Verify initialization
    if (m_pg.m_n != 0) {
        fprintf(stderr, "[DPVO] WARNING: PatchGraph.m_n is %d after reset, forcing to 0\n", m_pg.m_n);
        fflush(stderr);
        m_pg.m_n = 0;
        m_pg.m_m = 0;
    }
    // fmap sizes - Python uses RES=4, so work at 1/4 resolution
    // Python: ht = ht // RES, wd = wd // RES (where RES=4)
    // fmap1_: [ht // 1, wd // 1] = [ht/4, wd/4] in original coordinates
    // fmap2_: [ht // 4, wd // 4] = [ht/16, wd/16] in original coordinates
    const int RES = 4;
    const int res_ht = ht / RES;  // e.g., 480 / 4 = 120
    const int res_wd = wd / RES;  // e.g., 640 / 4 = 160
    
    m_fmap1_H = res_ht;      // 120 (1/4 resolution)
    m_fmap1_W = res_wd;      // 160 (1/4 resolution)
    m_fmap2_H = res_ht / 4;  // 30 (1/16 resolution)
    m_fmap2_W = res_wd / 4;  // 40 (1/16 resolution)

    // Validate dimensions to prevent bad_array_new_length
    if (m_fmap1_H <= 0 || m_fmap1_W <= 0 || m_fmap2_H <= 0 || m_fmap2_W <= 0) {
        throw std::runtime_error("Invalid fmap dimensions calculated from image size");
    }
    if (m_pmem <= 0 || m_mem <= 0 || cfg.PATCHES_PER_FRAME <= 0) {
        throw std::runtime_error("Invalid buffer configuration");
    }

	const int M = cfg.PATCHES_PER_FRAME;
    
    // Calculate array sizes and validate
    size_t imap_size = static_cast<size_t>(m_pmem) * static_cast<size_t>(M) * static_cast<size_t>(m_DIM);
    size_t gmap_size = static_cast<size_t>(m_pmem) * static_cast<size_t>(M) * 128 * static_cast<size_t>(m_P) * static_cast<size_t>(m_P);
    size_t fmap1_size = static_cast<size_t>(m_mem) * 128 * static_cast<size_t>(m_fmap1_H) * static_cast<size_t>(m_fmap1_W);
    size_t fmap2_size = static_cast<size_t>(m_mem) * 128 * static_cast<size_t>(m_fmap2_H) * static_cast<size_t>(m_fmap2_W);
    
    if (imap_size == 0 || gmap_size == 0 || fmap1_size == 0 || fmap2_size == 0) {
        throw std::runtime_error("Calculated array size is zero");
    }
    
    // allocate float arrays
    m_imap  = new float[imap_size]();
    m_gmap  = new float[gmap_size]();
    m_fmap1 = new float[fmap1_size]();
    m_fmap2 = new float[fmap2_size]();

	// -----------------------------
    // Zero-initialize (important!)
    // -----------------------------
    std::memset(m_imap,  0, sizeof(float) * imap_size);
    std::memset(m_gmap,  0, sizeof(float) * gmap_size);
    std::memset(m_fmap1, 0, sizeof(float) * fmap1_size);
    std::memset(m_fmap2, 0, sizeof(float) * fmap2_size);

    // Initialize intrinsics from config or use defaults
    if (config != nullptr) {
        setIntrinsicsFromConfig(config);
        m_updateModel = std::make_unique<DPVOUpdate>(config, nullptr);
    } else {
        // Default intrinsics (will be updated when frame dimensions are known)
        m_intrinsics[0] = static_cast<float>(wd) * 0.5f;  // fx
        m_intrinsics[1] = static_cast<float>(ht) * 0.5f;  // fy
        m_intrinsics[2] = static_cast<float>(wd) * 0.5f;  // cx
        m_intrinsics[3] = static_cast<float>(ht) * 0.5f;  // cy
    }
}

void DPVO::_startThreads()
{
#if defined(CV28) || defined(CV28_SIMULATOR)
    startProcessingThread();
    m_processingThreadRunning = true;
#endif
}

void DPVO::setUpdateModel(Config_S* config)
{
    if (config != nullptr && m_updateModel == nullptr) {
        m_updateModel = std::make_unique<DPVOUpdate>(config, nullptr);
    }
}

void DPVO::setPatchifierModels(Config_S* fnetConfig, Config_S* inetConfig)
{
    m_patchifier.setModels(fnetConfig, inetConfig);
    
    // Start threads after models are set (similar to WNC_APP::_init -> _startThreads)
    _startThreads();
}

void DPVO::setIntrinsics(const float intrinsics[4])
{
    std::memcpy(m_intrinsics, intrinsics, sizeof(float) * 4);
}

void DPVO::setIntrinsicsFromConfig(Config_S* config)
{
    if (config == nullptr) return;
    
    // Get focal length from config (in pixels)
    float focalLength = config->stCameraConfig.focalLength;
    int frameWidth = config->frameWidth;
    int frameHeight = config->frameHeight;
    
    // Calculate intrinsics: [fx, fy, cx, cy]
    // If focalLength is valid (> 0), use it; otherwise use frame dimensions as defaults
    if (focalLength > 0.0f) {
        m_intrinsics[0] = focalLength;  // fx
        m_intrinsics[1] = focalLength;  // fy (assuming square pixels)
    } else {
        // Default: use frame dimensions
        m_intrinsics[0] = static_cast<float>(frameWidth) * 0.5f;   // fx
        m_intrinsics[1] = static_cast<float>(frameHeight) * 0.5f;  // fy
    }
    
    // Principal point at image center
    m_intrinsics[2] = static_cast<float>(frameWidth) * 0.5f;   // cx
    m_intrinsics[3] = static_cast<float>(frameHeight) * 0.5f;  // cy
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
    // CRITICAL: Validate 'this' pointer FIRST - before accessing any members
    // The pattern 0x3e... in high bits indicates uninitialized/corrupted memory
    // On Linux x86_64, valid addresses have high 16 bits as:
    // - 0x0000 (heap addresses, lower 48 bits)
    // - 0x7fff/0xffff (stack addresses, canonical form)
    // The pattern 0x3e... is NOT a valid address pattern
    
    // Use both stdout and stderr to ensure output is visible
    fprintf(stderr, "[DPVO] ================ ENTERING run() ========================\n");
    fprintf(stderr, "[DPVO] DEBUG: Validating 'this' pointer at start of run(): %p\n", (void*)this);
    printf("[DPVO] ==================== ENTERING run() =============================\n");
    printf("[DPVO] DEBUG: Validating 'this' pointer at start of run(): %p\n", (void*)this);
    fflush(stdout);
    fflush(stderr);
    
    uintptr_t this_addr = reinterpret_cast<uintptr_t>(this);
    uint16_t high_bits = (this_addr >> 48) & 0xFFFF;
    
    printf("[DPVO] DEBUG: this_addr=0x%016lx, high_bits=0x%04x\n", this_addr, high_bits);
    fflush(stdout);
    
    // Check for NULL pointer
    if (this_addr == 0) {
        printf("[DPVO] CRITICAL: 'this' pointer is NULL at start of run()\n");
        fflush(stdout);
        std::abort();  // Cannot safely return - abort immediately
    }
    
    // Check for suspiciously small addresses (likely invalid)
    if (this_addr < 0x1000) {
        printf("[DPVO] CRITICAL: 'this' pointer is too small: %p (likely invalid)\n", (void*)this);
        fflush(stdout);
        std::abort();
    }
    
    // Check for the specific suspicious pattern we're seeing (0x3e...)
    // This pattern indicates uninitialized/corrupted memory
    uint16_t masked_bits = high_bits & 0xFF00;
    printf("[DPVO] DEBUG: Checking pattern: high_bits=0x%04x, masked=0x%04x, match=%d\n", 
            high_bits, masked_bits, (masked_bits == 0x3E00));
    fflush(stdout);
    
    if (masked_bits == 0x3E00) {
        printf("[DPVO] CRITICAL: 'this' pointer has corrupted pattern: %p (high bits: 0x%04x)\n", 
                (void*)this, high_bits);
        fflush(stdout);
        printf("[DPVO] CRITICAL: DPVO object is corrupted - pattern 0x3e... indicates uninitialized memory\n");
        fflush(stdout);
        printf("[DPVO] CRITICAL: This suggests the object was not properly constructed or has been destroyed\n");
        fflush(stdout);
        std::abort();  // Cannot safely return - abort immediately
    }
    
    // Additional check: verify that 'this' is in a reasonable address range
    // On x86_64 Linux, user-space addresses are typically:
    // - Heap: 0x000055... to 0x00007f... (lower 48 bits, high bits = 0x0000)
    // - Stack: 0x7fff... (canonical form, high bits = 0x7fff or 0xffff)
    // If high bits are not 0x0000, 0x7fff, or 0xffff, it's suspicious
    if (high_bits != 0x0000 && high_bits != 0x7fff && high_bits != 0xffff) {
        printf("[DPVO] CRITICAL: 'this' pointer has suspicious high bits: %p (high bits: 0x%04x)\n", 
                (void*)this, high_bits);
        fflush(stdout);
        printf("[DPVO] CRITICAL: Valid addresses should have high bits as 0x0000, 0x7fff, or 0xffff\n");
        fflush(stdout);
        std::abort();  // Cannot safely return - abort immediately
    }
    
    printf("[DPVO] DEBUG: 'this' pointer validation passed at start of run()\n");
    fflush(stdout);
    
    // Validate image pointer
    if (image == nullptr) {
        throw std::runtime_error("Null image pointer passed to DPVO::run");
    }
    
    // Note: H and W may differ from m_ht and m_wd (model input size)
    // fnet/inet models will resize internally, so we allow different input sizes
    // However, we validate that dimensions are reasonable
    if (H < 16 || W < 16) {
        throw std::runtime_error(
            "Image dimensions too small: " + std::to_string(H) + "x" + std::to_string(W) + 
            " (minimum 16x16)");
    }
    
    // CRITICAL: Validate and fix m_pg.m_n before using it
    // The value 1051635375 suggests uninitialized memory corruption
    fprintf(stderr, "[DPVO] DEBUG: m_pg.m_n = %d (before check), PatchGraph::N = %d\n", m_pg.m_n, PatchGraph::N);
    fflush(stderr);
    
    // CRITICAL: If m_pg.m_n is corrupted, try to reset m_pg and use n=0
    int n = 0;  // Default to 0
    if (m_pg.m_n < 0 || m_pg.m_n >= PatchGraph::N || m_pg.m_n > 1000) {
        fprintf(stderr, "[DPVO] CRITICAL: m_pg.m_n has invalid value: %d (expected 0-%d), RESETTING\n", 
                m_pg.m_n, PatchGraph::N - 1);
        fflush(stderr);
        // Try to reset m_pg - this might crash if m_pg is completely corrupted
        try {
            m_pg.reset();
            fprintf(stderr, "[DPVO] Reset PatchGraph, m_pg.m_n is now: %d\n", m_pg.m_n);
            fflush(stderr);
            // Verify reset worked
            if (m_pg.m_n < 0 || m_pg.m_n >= PatchGraph::N || m_pg.m_n > 1000) {
                fprintf(stderr, "[DPVO] CRITICAL: m_pg.m_n still corrupted after reset: %d, forcing to 0\n", m_pg.m_n);
                fflush(stderr);
                m_pg.m_n = 0;  // Force to 0
                m_pg.m_m = 0;
            }
            n = m_pg.m_n;  // Should be 0 now
        } catch (...) {
            fprintf(stderr, "[DPVO] CRITICAL: Cannot reset m_pg - object is completely corrupted, using n=0\n");
            fflush(stderr);
            n = 0;  // Use 0 as fallback
        }
    } else {
        n = m_pg.m_n;  // Use the valid value
    }
    
    if (n + 1 >= PatchGraph::N)
        throw std::runtime_error("PatchGraph buffer overflow");
    fprintf(stderr, "[DPVO] DEBUG: Using n = %d\n", n);
    fflush(stderr);
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

    // CRITICAL: Calculate the actual patch size D from patchify_cpu_safe
    // patchify_cpu_safe uses radius = m_patch_size / 2, and D = 2 * radius + 2
    // With m_patch_size = 3: radius = 1, D = 2 * 1 + 2 = 4
    // So patches array must be M * 3 * D * D, NOT M * 3 * P * P
    const int patch_radius = m_P / 2;  // m_P = 3, so radius = 1
    const int patch_D = 2 * patch_radius + 2;  // D = 4
    const int patches_size = M * 3 * patch_D * patch_D;  // 8 * 3 * 4 * 4 = 384 floats
    
    // Use heap allocation instead of stack to avoid stack overflow
    // Stack allocation of 384 floats (1536 bytes) might be okay, but let's be safe
    std::vector<float> patches_vec(patches_size);
    float* patches = patches_vec.data();
    
    uint8_t clr[M * 3];

    auto logger = spdlog::get("dpvo");
    if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
        logger = spdlog::stdout_color_mt("dpvo");
        logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    }
    
    if (logger) logger->info("DPVO::run: Starting patchifier.forward");
    printf("[DPVO] About to call patchifier.forward\n");
    fflush(stdout);
    
    // -------------------------------------------------
    // Normalize image: 2 * (image / 255.0) - 0.5 (matches Python)
    // Python: image = 2 * (image[None,None] / 255.0) - 0.5
    // This normalizes from uint8 [0, 255] to float [-0.5, 1.5]
    // -------------------------------------------------
    const int C = 3;  // RGB channels
    const int image_size = C * H * W;
    std::vector<float> normalized_image(image_size);
    
    if (logger) logger->info("DPVO::run: Normalizing image from uint8 [0, 255] to float [-0.5, 1.5], size={}", image_size);
    
    // Image format is [C, H, W] (channel-first)
    for (int i = 0; i < image_size; i++) {
        // Normalize: 2 * (image / 255.0) - 0.5
        normalized_image[i] = 2.0f * (static_cast<float>(image[i]) / 255.0f) - 0.5f;
    }
    
    if (logger) {
        float img_min = *std::min_element(normalized_image.begin(), normalized_image.end());
        float img_max = *std::max_element(normalized_image.begin(), normalized_image.end());
        logger->info("DPVO::run: Normalized image range: [{}, {}]", img_min, img_max);
    }
    
    // CRITICAL: Save 'this' pointer to a safe location (static/global) before calling patchifier.forward()
    // This protects against stack corruption that might overwrite the 'this' pointer on the stack
    static thread_local DPVO* saved_this_ptr = nullptr;
    saved_this_ptr = this;
    uintptr_t saved_this_addr = reinterpret_cast<uintptr_t>(this);
    
    printf("[DPVO] Saved 'this' pointer before patchifier.forward(): %p (saved_addr=0x%016lx)\n", 
           (void*)this, saved_this_addr);
    fflush(stdout);
    
    // Log where we're writing imap data
    int imap_write_offset = imap_idx(pm, 0, 0);
    if (logger) {
        logger->info("DPVO::run: Writing imap data to ring buffer slot pm={}, offset={}, n={}", 
                     pm, imap_write_offset, n);
        // Check a sample value before writing
        float before_val = m_imap[imap_write_offset];
        logger->info("DPVO::run: m_imap[{}] before patchifier.forward = {}", imap_write_offset, before_val);
    }
    
    // Pass normalized float image to patchifier
    // Note: We'll need to update Patchifier::forward() to accept float* and handle conversion to uint8 for models
    m_patchifier.forward(
        normalized_image.data(), H, W,  // Pass normalized float image
        m_cur_fmap1,     // full-res fmap
        m_cur_imap,
        m_cur_gmap,
        patches,
        clr,
        M
    );
    
    printf("[DPVO] patchifier.forward returned\n");
    fflush(stdout);
    
    // Verify data was written correctly
    if (logger) {
        float after_val = m_imap[imap_write_offset];
        float after_val_patch1 = m_imap[imap_write_offset + m_DIM];  // First element of patch 1
        logger->info("DPVO::run: m_imap[{}] after patchifier.forward = {}, m_imap[{}] (patch1) = {}", 
                     imap_write_offset, after_val, imap_write_offset + m_DIM, after_val_patch1);
        
        // Check all patches for this frame
        int nonzero_count = 0;
        int zero_count = 0;
        for (int p = 0; p < M; p++) {
            int patch_offset = imap_write_offset + p * m_DIM;
            float patch_val = m_imap[patch_offset];
            if (patch_val == 0.0f) zero_count++;
            else nonzero_count++;
        }
        logger->info("DPVO::run: Frame {} imap data - zero_count={}, nonzero_count={} (out of {} patches)", 
                     pm, zero_count, nonzero_count, M);
    }
    
    // CRITICAL: Re-validate 'this' pointer immediately after patchifier.forward() returns
    // If stack corruption occurred, 'this' might be corrupted, but saved_this_ptr should be safe
    uintptr_t current_this_addr = reinterpret_cast<uintptr_t>(this);
    uint16_t current_high_bits = (current_this_addr >> 48) & 0xFFFF;
    
    printf("[DPVO] After patchifier.forward: current 'this'=%p (0x%016lx, high_bits=0x%04x), saved='%p (0x%016lx)\n",
           (void*)this, current_this_addr, current_high_bits, 
           (void*)saved_this_ptr, saved_this_addr);
    fflush(stdout);
    
    // Check if 'this' was corrupted
    if (current_this_addr != saved_this_addr) {
        printf("[DPVO] CRITICAL: 'this' pointer was corrupted by patchifier.forward()!\n");
        printf("[DPVO] CRITICAL: Original: %p, Current: %p\n", (void*)saved_this_ptr, (void*)this);
        fflush(stdout);
        // Restore 'this' from saved value (this is a hack, but might work)
        // Actually, we can't restore 'this' - it's a parameter. We need to abort.
        std::abort();
    }
    
    // Check for corrupted pattern
    if ((current_high_bits & 0xFF00) == 0x3E00 || 
        (current_high_bits != 0x0000 && current_high_bits != 0x7fff && current_high_bits != 0xffff)) {
        printf("[DPVO] CRITICAL: 'this' pointer has corrupted pattern after patchifier.forward(): %p (high bits: 0x%04x)\n", 
                (void*)this, current_high_bits);
        fflush(stdout);
        printf("[DPVO] CRITICAL: Restoring from saved pointer: %p\n", (void*)saved_this_ptr);
        fflush(stdout);
        // We can't actually restore 'this', but we can use saved_this_ptr to access members
        // For now, just abort to prevent further corruption
        std::abort();
    }
    
    // Use saved_this_ptr to access members if 'this' is corrupted
    // But actually, if 'this' is corrupted, we can't safely continue
    // So we'll just abort for now
    
    printf("[DPVO] About to log patchifier completion\n");
    fflush(stdout);
    if (logger) {
        try {
            logger->info("DPVO::run: patchifier.forward completed");
        } catch (...) {
            fprintf(stderr, "[DPVO] EXCEPTION in logger->error\n");
            fflush(stderr);
        }
    }
    
    printf("[DPVO] About to validate n, current n=%d\n", n);
    fflush(stdout);
    
    // CRITICAL: The variable n was set earlier from m_pg.m_n, but it might be corrupted
    // Validate it before using for array indexing
    int n_use = n;  // Start with original n
    if (n_use < 0 || n_use >= PatchGraph::N || n_use > 1000) {
        fprintf(stderr, "[DPVO] CRITICAL: n=%d is corrupted! Using n_use=0 instead.\n", n);
        fflush(stderr);
        n_use = 0;
        // Don't try to fix m_pg.m_n here - just use n_use=0 for array indexing
        // We'll fix m_pg.m_n later after we've safely written data
    }
    
    printf("[DPVO] About to start bookkeeping, n=%d (original), n_use=%d (validated)\n", n, n_use);
    fflush(stdout);
    
    // For the rest of this function, we need to use n_use instead of n
    // But since n is const, we can't reassign it. Instead, we'll use n_use directly.
    // Actually, the simplest is to just replace n with n_use in the critical array accesses

    // -------------------------------------------------
    // 2. Bookkeeping
    // -------------------------------------------------
    if (logger) logger->info("DPVO::run: Starting bookkeeping, n={}", n_use);
    
    // Store timestamp in both m_tlist (for compatibility) and m_pg.m_tstamps (main storage)
    m_tlist.push_back(timestamp);
    m_pg.m_tstamps[n_use] = timestamp;
    
    // Store camera intrinsics (Python divides by RES=4)
    const float RES = 4.0f;
    float scaled_intrinsics[4];
    for (int i = 0; i < 4; i++) {
        scaled_intrinsics[i] = intrinsics[i] / RES;
    }
    std::memcpy(m_pg.m_intrinsics[n_use], scaled_intrinsics, sizeof(float) * 4);
    
    if (logger) logger->info("DPVO::run: Bookkeeping completed");

    // -------------------------------------------------
    // 3. Pose initialization (with motion model support)
    // -------------------------------------------------
    if (logger) logger->info("DPVO::run: Starting pose initialization");
    if (n_use > 1) {
        // Python: if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
        // For now, we'll implement simple copy (equivalent to Python's else branch)
        // TODO: Add MOTION_MODEL and MOTION_DAMPING to DPVOConfig if needed
        m_pg.m_poses[n_use] = m_pg.m_poses[n_use - 1];
    } else if (n_use > 0) {
        m_pg.m_poses[n_use] = m_pg.m_poses[n_use - 1];
    }
    if (logger) logger->info("DPVO::run: Pose initialization completed");

    // -------------------------------------------------
    // 4. Patch depth initialization (CRITICAL - matches Python logic)
    // -------------------------------------------------
    // Python: patches[:,:,2] = torch.rand_like(...) for first frames
    //         if self.is_initialized: s = torch.median(...); patches[:,:,2] = s
    // NOTE: patchify_cpu_safe writes patches of size D*D (where D=4), but we store them in m_pg.m_patches
    // which expects P*P (where P=3). We'll extract the center P*P region from the D*D patch.
    if (logger) logger->info("DPVO::run: Starting patch depth initialization, m_is_initialized={}", m_is_initialized);
    
    float depth_value = 1.0f;  // Default value
    
    if (m_is_initialized && n_use >= 3) {
        // Python: s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
        // Compute median of last 3 frames' depth values (center pixel of each patch)
        std::vector<float> depths;
        for (int f = std::max(0, n_use - 3); f < n_use; f++) {
            for (int i = 0; i < M; i++) {
                // Get center pixel depth: patches[f][i][2][P/2][P/2]
                int center_y = P / 2;
                int center_x = P / 2;
                float d = m_pg.m_patches[f][i][2][center_y][center_x];
                if (d > 0.0f) {  // Only include valid depths
                    depths.push_back(d);
                }
            }
        }
        if (!depths.empty()) {
            std::sort(depths.begin(), depths.end());
            depth_value = depths[depths.size() / 2];  // Median
            if (logger) logger->info("DPVO::run: Using median depth from last 3 frames: {}", depth_value);
        }
    } else {
        // Python: patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        // Random initialization for first frames
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.1f, 1.0f);
        depth_value = dis(gen);
        if (logger) logger->info("DPVO::run: Using random depth initialization: {}", depth_value);
    }
    
    // Initialize all patches with the computed depth value
    for (int i = 0; i < M; i++) {
        // patches layout from patchify_cpu_safe: [M][3][D][D] where D=4
        // We need to access the depth channel (c=2) and initialize center P*P region
        int base = (i * 3 + 2) * patch_D * patch_D;
        int center_offset = (patch_D - P) / 2;  // Center the P*P region within D*D
        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {
                int patch_idx = base + (center_offset + y) * patch_D + (center_offset + x);
                patches[patch_idx] = depth_value;
            }
        }
    }
    if (logger) logger->info("DPVO::run: Patch depth initialization completed");

    // -------------------------------------------------
    // 5. Store patches + colors into PatchGraph
    // -------------------------------------------------
    // NOTE: Extract center P*P region from D*D patches written by patchify_cpu_safe
    if (logger) logger->info("DPVO::run: Starting store patches, n_use={}, M={}, P={}, patch_D={}", n_use, M, P, patch_D);
    int center_offset = (patch_D - P) / 2;  // Center the P*P region within D*D
    for (int i = 0; i < M; i++) {
        for (int c = 0; c < 3; c++) {
            // patches layout: [M][3][D][D] where D=4
            int base = (i * 3 + c) * patch_D * patch_D;
            for (int y = 0; y < P; y++) {
                for (int x = 0; x < P; x++) {
                    // Extract center P*P region from D*D patch
                    int patch_idx = base + (center_offset + y) * patch_D + (center_offset + x);
                    m_pg.m_patches[n_use][i][c][y][x] = patches[patch_idx];
                }
            }
        }
        for (int c = 0; c < 3; c++)
            m_pg.m_colors[n_use][i][c] = clr[i * 3 + c];
    }
    if (logger) logger->info("DPVO::run: Store patches completed");

    // -------------------------------------------------
    // 6. Downsample fmap1 → fmap2 (Python avg_pool2d)
    // fmap1 is at 1/4 resolution (e.g., 120x160), fmap2 is at 1/16 resolution (e.g., 30x40)
    // -------------------------------------------------
    if (logger) logger->info("DPVO::run: Starting downsample fmap1->fmap2, fmap1_H={}, fmap1_W={}, fmap2_H={}, fmap2_W={}", 
                              m_fmap1_H, m_fmap1_W, m_fmap2_H, m_fmap2_W);
    for (int c = 0; c < 128; c++) {
        for (int y = 0; y < m_fmap2_H; y++) {
            for (int x = 0; x < m_fmap2_W; x++) {
                float sum = 0.0f;
                for (int dy = 0; dy < 4; dy++)
                    for (int dx = 0; dx < 4; dx++)
                        // fmap1 is at m_fmap1_H x m_fmap1_W (1/4 resolution)
                        sum += m_cur_fmap1[c * m_fmap1_H * m_fmap1_W +
                            (y * 4 + dy) * m_fmap1_W +
                            (x * 4 + dx)];
                // Store in fmap2 (at 1/16 resolution)
                m_fmap2[fmap2_idx(0, mm, c, y, x)] = sum / 16.0f;
            }
        }
    }
    if (logger) logger->info("DPVO::run: Downsample fmap1->fmap2 completed");

    // -------------------------------------------------
    // 7. Motion probe check (Python: if self.n > 0 and not self.is_initialized)
    // -------------------------------------------------
    if (n_use > 0 && !m_is_initialized) {
        if (logger) logger->info("DPVO::run: Running motion probe check before initialization");
        float motion_val = motionProbe();
        if (motion_val < 2.0f) {
            // Python: self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
            // For now, we'll just return early (delta handling can be added later if needed)
            if (logger) logger->info("DPVO::run: Motion probe returned {} < 2.0, skipping frame", motion_val);
            return;
        }
        if (logger) logger->info("DPVO::run: Motion probe returned {} >= 2.0, proceeding", motion_val);
    }

    // -------------------------------------------------
    // 8. Counters
    // -------------------------------------------------
    if (logger) logger->info("DPVO::run: Updating counters");
    // Use n_use instead of m_pg.m_n to ensure we're incrementing from a valid value
    // If n was corrupted, n_use will be 0, so we'll start counting from 0
    try {
        m_pg.m_n = n_use + 1;  // Set to n_use + 1 (next frame index)
        m_pg.m_m += M;
        m_counter++;
        if (logger) logger->info("DPVO::run: Counters updated, m_n={}, m_m={}", m_pg.m_n, m_pg.m_m);
    } catch (...) {
        fprintf(stderr, "[DPVO] EXCEPTION updating counters, m_pg might be corrupted\n");
        fflush(stderr);
        // Continue anyway - the data has been written using n_use
    }

    // -------------------------------------------------
    // 9. Build edges
    // -------------------------------------------------
    if (logger) logger->info("DPVO::run: Starting build edges");
    std::vector<int> kk, jj;
    edgesForward(kk, jj);
    if (logger) logger->info("DPVO::run: edgesForward completed, kk.size()={}, jj.size()={}", kk.size(), jj.size());
    appendFactors(kk, jj);
    if (logger) logger->info("DPVO::run: appendFactors (forward) completed");
    edgesBackward(kk, jj);
    if (logger) logger->info("DPVO::run: edgesBackward completed, kk.size()={}, jj.size()={}", kk.size(), jj.size());
    appendFactors(kk, jj);
    if (logger) logger->info("DPVO::run: appendFactors (backward) completed");

    // -------------------------------------------------
    // 10. Optimization
    // -------------------------------------------------
    if (logger) logger->info("DPVO::run: Starting optimization, m_is_initialized={}, m_n={}", m_is_initialized, m_pg.m_n);
    if (m_is_initialized) {
        if (logger) logger->info("DPVO::run: Calling update()");
        update();
        if (logger) logger->info("DPVO::run: update() completed");
        if (logger) logger->info("DPVO::run: Calling keyframe()");
        keyframe();
        if (logger) logger->info("DPVO::run: keyframe() completed");
    } else if (m_pg.m_n >= 8) {
        if (logger) logger->info("DPVO::run: Initializing with 12 update() calls");
        m_is_initialized = true;
        for (int i = 0; i < 12; i++) {
            if (logger) logger->info("DPVO::run: Initialization update() call {}/12", i+1);
            update();
        }
        if (logger) logger->info("DPVO::run: Initialization completed");
    }
    if (logger) logger->info("DPVO::run: Optimization completed");
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
    // CRITICAL: Pass full buffers (m_fmap1, m_fmap2), not single-frame pointers (m_cur_fmap1)
    // computeCorrelation needs to access multiple frames based on jj[e] indices
    // Correlation output shape: [num_active, D, D, P, P, 2] where D = 2*R + 2 = 8 (R=3)
    const int R = 3;  // Correlation radius
    const int D = 2 * R + 2;  // Correlation window diameter (D = 8)
    std::vector<float> corr(num_active * D * D * P * P * 2); // [num_active, D, D, P, P, 2] (channel last)
    
    auto logger = spdlog::get("dpvo");
    if (logger) {
        logger->info("DPVO::update: Starting correlation, num_active={}, M={}, P={}, D={}, m_mem={}, m_pmem={}", 
                     num_active, M, P, D, m_mem, m_pmem);
        logger->info("DPVO::update: fmap1 dimensions: {}x{}, fmap2 dimensions: {}x{}", 
                     m_fmap1_H, m_fmap1_W, m_fmap2_H, m_fmap2_W);
    }
    
    computeCorrelation(
		m_gmap,
		m_fmap1,          // pyramid0 - full buffer [m_mem][128][fmap1_H][fmap1_W]
		m_fmap2,          // pyramid1 - full buffer [m_mem][128][fmap2_H][fmap2_W]
		coords.data(),
		m_pg.m_ii,        // ii - patch indices (within frame)
		m_pg.m_jj,        // jj - frame indices (for pyramid/target frame)
		m_pg.m_kk,        // kk - linear patch indices (frame * M + patch, for gmap source frame)
		num_active,
		M,
		P,
		m_mem,            // num_frames - number of frames in pyramid buffers
		m_pmem,           // num_gmap_frames - number of frames in gmap ring buffer
		m_fmap1_H, m_fmap1_W,  // Dimensions for pyramid0 (fmap1)
		m_fmap2_H, m_fmap2_W,  // Dimensions for pyramid1 (fmap2) - CRITICAL: different from fmap1!
		128,
		corr.data()
	);
    
    if (logger) logger->info("DPVO::update: Correlation completed");


    // -------------------------------------------------
    // 3. Context slice from imap
    // -------------------------------------------------
    if (logger) logger->info("DPVO::update: Starting context slice from imap, num_active={}, m_DIM={}", num_active, m_DIM);
    
    // Check m_imap buffer statistics before slicing
    if (logger) {
        // Sample first few entries to check if m_imap is populated
        float imap_sample_min = std::numeric_limits<float>::max();
        float imap_sample_max = std::numeric_limits<float>::lowest();
        int imap_sample_zero = 0;
        int imap_sample_nonzero = 0;
        int sample_size = std::min(100, M * m_pmem * m_DIM);
        for (int i = 0; i < sample_size; i++) {
            float val = m_imap[i];
            if (val == 0.0f) imap_sample_zero++;
            else imap_sample_nonzero++;
            if (val < imap_sample_min) imap_sample_min = val;
            if (val > imap_sample_max) imap_sample_max = val;
        }
        logger->info("DPVO::update: m_imap buffer sample stats (first {} elements) - zero_count={}, nonzero_count={}, min={}, max={}",
                     sample_size, imap_sample_zero, imap_sample_nonzero, imap_sample_min, imap_sample_max);
        
        // Also check the most recent frame (should be frame 8 if n=9, so pm=8)
        int current_n = m_pg.m_n;
        int current_pm = (current_n - 1) % m_pmem;  // Most recently written frame
        int current_imap_offset = imap_idx(current_pm, 0, 0);
        float current_frame_sample = m_imap[current_imap_offset];
        logger->info("DPVO::update: Most recent frame n={}, pm={}, imap_offset={}, m_imap[{}]={}",
                     current_n, current_pm, current_imap_offset, current_imap_offset, current_frame_sample);
        
        // Check all frames in ring buffer
        int frames_with_data = 0;
        int frames_without_data = 0;
        for (int f = 0; f < m_pmem; f++) {
            int frame_offset = imap_idx(f, 0, 0);
            float frame_sample = m_imap[frame_offset];
            if (frame_sample != 0.0f) frames_with_data++;
            else frames_without_data++;
        }
        logger->info("DPVO::update: Ring buffer status - frames_with_data={}, frames_without_data={} (out of {} slots)",
                     frames_with_data, frames_without_data, m_pmem);
    }
    
    std::vector<float> ctx(num_active * m_DIM);
    for (int e = 0; e < num_active; e++) {
        // CRITICAL: kk is a linear patch index: kk = frame * M + patch
        // m_imap is a ring buffer indexed as [frame % m_pmem][patch][dim]
        // So we need to convert kk to (frame, patch) and then index correctly
        int kk_val = m_pg.m_kk[e];
        int frame = kk_val / M;  // Extract frame from linear index
        int patch = kk_val % M;  // Extract patch from linear index
        
        // Convert to ring buffer index: frame % m_pmem
        int imap_frame = frame % m_pmem;
        
        // Validate indices
        if (frame < 0 || patch < 0 || patch >= M) {
            if (logger && e < 10) logger->error("DPVO::update: Invalid kk={} -> frame={}, patch={} for edge e={}, M={}", 
                                                kk_val, frame, patch, e, M);
            // Fallback to frame 0, patch 0
            imap_frame = 0;
            patch = 0;
        }
        
        // Use imap_idx to get the correct offset: [imap_frame][patch][0]
        int imap_offset = imap_idx(imap_frame, patch, 0);
        
        // Validate source pointer before memcpy
        if (logger && e < 5) {
            float src_sample = m_imap[imap_offset];
            logger->info("DPVO::update: Edge e={}, kk={} -> frame={}, patch={}, imap_frame={}, imap_offset={}, m_imap[{}]={}", 
                         e, kk_val, frame, patch, imap_frame, imap_offset, imap_offset, src_sample);
        }
        
        std::memcpy(
            &ctx[e * m_DIM],
            &m_imap[imap_offset],
            sizeof(float) * m_DIM
        );
    }
    
    // Check ctx statistics after slicing
    if (logger) {
        float ctx_min = *std::min_element(ctx.begin(), ctx.end());
        float ctx_max = *std::max_element(ctx.begin(), ctx.end());
        int ctx_zero_count = 0;
        int ctx_nonzero_count = 0;
        for (size_t i = 0; i < ctx.size(); i++) {
            if (ctx[i] == 0.0f) ctx_zero_count++;
            else ctx_nonzero_count++;
        }
        logger->info("DPVO::update: Context (ctx) stats after slicing - zero_count={}, nonzero_count={}, min={}, max={}, size={}",
                     ctx_zero_count, ctx_nonzero_count, ctx_min, ctx_max, ctx.size());
    }
    
    if (logger) logger->info("DPVO::update: Context slice completed");

    // -------------------------------------------------
    // 4. Network update (DPVO Update Model Inference)
    // -------------------------------------------------
    if (logger) logger->info("DPVO::update: Starting network update, m_updateModel={}", (void*)m_updateModel.get());
    std::vector<float> delta(num_active * 2);
    std::vector<float> weight(num_active);
    
    if (m_updateModel != nullptr) {
        if (logger) logger->info("DPVO::update: m_updateModel is not null, preparing model inputs");
        // Model expects fixed shapes: [1, 384, 768, 1] for net/inp, 
        // [1, 882, 768, 1] for corr, [1, 768, 1] for indices
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
        
        // CRITICAL: YAML specifies [N, C, H, W] layout: 1,384,768,1
        // Where: N=1 (batch), C=384 (channels), H=768 (spatial/edges), W=1
        // Reshape and copy net data: [num_active, 384] -> [1, 384, 768, 1]
        // Layout: [batch, channels, height, width] = [1, 384, 768, 1]
        if (logger) logger->info("DPVO::update: Reshaping net/inp data, num_edges_to_process={}", num_edges_to_process);
        
        // Check net state before copying
        int net_zero_count = 0;
        int net_nonzero_count = 0;
        float net_min = std::numeric_limits<float>::max();
        float net_max = std::numeric_limits<float>::lowest();
        for (int e = 0; e < std::min(num_edges_to_process, num_active); e++) {
            for (int d = 0; d < 384; d++) {
                float val = m_pg.m_net[e][d];
                if (val == 0.0f) net_zero_count++;
                else net_nonzero_count++;
                if (val < net_min) net_min = val;
                if (val > net_max) net_max = val;
            }
        }
        if (logger) {
            logger->info("DPVO::update: Net state stats - zero_count={}, nonzero_count={}, min={}, max={}",
                         net_zero_count, net_nonzero_count, net_min, net_max);
        }
        
        // WORKAROUND: If net is all zeros, initialize it from context (inp) to break the cycle
        // This is a temporary fix - ideally the model should accept zero input or we need proper initialization
        bool net_all_zero = (net_nonzero_count == 0);
        if (net_all_zero && logger) {
            logger->warn("DPVO::update: Net state is all zeros - initializing from context (inp) as workaround");
            // Initialize net from context (inp) - this gives the model some initial state
            for (int e = 0; e < num_edges_to_process; e++) {
                if (e < 0 || e >= num_active) continue;
                for (int d = 0; d < 384; d++) {
                    // Use context as initial net state (scaled down to avoid large values)
                    m_pg.m_net[e][d] = ctx[e * 384 + d] * 0.1f;
                }
            }
            if (logger) logger->info("DPVO::update: Net state initialized from context for {} edges", num_edges_to_process);
        }
        
        for (int e = 0; e < num_edges_to_process; e++) {
            // Validate edge index
            if (e < 0 || e >= num_active) {
                if (logger && e < 10) logger->error("DPVO::update: Invalid edge index e={}, num_active={}", e, num_active);
                continue;
            }
            for (int d = 0; d < 384; d++) {
                // YAML layout: [N, C, H, W] = [1, 384, 768, 1]
                // Index calculation: n * C * H * W + c * H * W + h * W + w
                // For net/inp: n=0, c=d (channel), h=e (edge index), w=0
                int idx = 0 * 384 * MODEL_EDGE_COUNT * 1 + d * MODEL_EDGE_COUNT * 1 + e * 1 + 0;
                net_input[idx] = m_pg.m_net[e][d];
                inp_input[idx] = ctx[e * 384 + d];
            }
        }
        
        // Check input data statistics after copying
        float net_input_min = *std::min_element(net_input.begin(), net_input.end());
        float net_input_max = *std::max_element(net_input.begin(), net_input.end());
        float inp_input_min = *std::min_element(inp_input.begin(), inp_input.end());
        float inp_input_max = *std::max_element(inp_input.begin(), inp_input.end());
        float corr_input_min = *std::min_element(corr_input.begin(), corr_input.end());
        float corr_input_max = *std::max_element(corr_input.begin(), corr_input.end());
        
        if (logger) {
            logger->info("DPVO::update: Input data ranges - net=[{}, {}], inp=[{}, {}], corr=[{}, {}]",
                         net_input_min, net_input_max, inp_input_min, inp_input_max, corr_input_min, corr_input_max);
        }
        
        if (logger) logger->info("DPVO::update: Net/inp data reshaping completed");
        
        // Reshape correlation: [num_active, D, D, P, P, 2] -> [1, 882, 768, 1]
        // Python CUDA kernel outputs: [B, M, D, D, H, W] per channel
        // Python stacks: torch.stack([corr1, corr2], -1) -> [B, M, D, D, H, W, 2]
        // C++ output matches: [num_active, D, D, P, P, 2] (channel last, matching Python)
        // Total per edge: D * D * P * P * 2 = 8 * 8 * 3 * 3 * 2 = 1152
        // Model expects [1, 882, 768, 1] - we need to downsample from 8×8 to 7×7 window
        // 882 = 2 * 7 * 7 * 9 = 2 * 7 * 7 * 3 * 3 (2 channels, 7×7 window, 3×3 patch)
        const int target_corr_dim = CORR_DIM; // 882
        const int D_target = 7;  // Target window size for model (7×7 instead of 8×8)
        const int offset = (D - D_target) / 2;  // Center offset: (8-7)/2 = 0 (integer division)
        
        // Check correlation data before reshaping
        float corr_min = *std::min_element(corr.begin(), corr.end());
        float corr_max = *std::max_element(corr.begin(), corr.end());
        int corr_zero_count = 0;
        int corr_nonzero_count = 0;
        for (size_t i = 0; i < corr.size(); i++) {
            if (corr[i] == 0.0f) corr_zero_count++;
            else corr_nonzero_count++;
        }
        if (logger) {
            logger->info("DPVO::update: Correlation data stats - zero_count={}, nonzero_count={}, min={}, max={}, size={}",
                         corr_zero_count, corr_nonzero_count, corr_min, corr_max, corr.size());
        }
        
        if (logger) logger->info("DPVO::update: Reshaping correlation data, D={}, D_target={}, offset={}", D, D_target, offset);
        for (int e = 0; e < num_edges_to_process; e++) {
            // Validate edge index
            if (e < 0 || e >= num_active) {
                if (logger && e < 10) logger->error("DPVO::update: Invalid edge index e={} in correlation reshape, num_active={}", e, num_active);
                continue;
            }
            // Source layout: corr[e][di][dj][pi][pj][c] = [num_active, D, D, P, P, 2]
            // We need to extract center 7×7 region from 8×8 window
            for (int c = 0; c < 2; c++) {
                for (int di = 0; di < D_target && (di + offset) < D; di++) {
                    for (int dj = 0; dj < D_target && (dj + offset) < D; dj++) {
                        for (int pi = 0; pi < P; pi++) {
                            for (int pj = 0; pj < P; pj++) {
                                // Source: corr[e][di+offset][dj+offset][pi][pj][c]
                                // Layout: [num_active, D, D, P, P, 2]
                                int src_idx = e * D * D * P * P * 2 +
                                             (di + offset) * D * P * P * 2 +
                                             (dj + offset) * P * P * 2 +
                                             pi * P * 2 +
                                             pj * 2 +
                                             c;  // Channel last
                                
                                // YAML layout: [N, C, H, W] = [1, 882, 768, 1]
                                // Model expects: [1, 882, 768, 1] where 882 = 2 * 7 * 7 * 3 * 3
                                // Index calculation: n * C * H * W + c * H * W + h * W + w
                                // Where: n=0, c=dst_corr_idx (feature index), h=e (edge index), w=0
                                int dst_corr_idx = c * D_target * D_target * P * P +
                                                  di * D_target * P * P +
                                                  dj * P * P +
                                                  pi * P + pj;
                                
                                if (dst_corr_idx < target_corr_dim) {
                                    // [N, C, H, W] = [1, 882, 768, 1]
                                    int idx = 0 * CORR_DIM * MODEL_EDGE_COUNT * 1 + 
                                             dst_corr_idx * MODEL_EDGE_COUNT * 1 + 
                                             e * 1 + 
                                             0;
                                    corr_input[idx] = corr[src_idx];
                                }
                            }
                        }
                    }
                }
            }
            // Rest of CORR_DIM is zero-padded (already initialized to 0)
        }
        if (logger) logger->info("DPVO::update: Correlation reshaping completed");
        
        // Copy indices: [num_active] -> [1, 768, 1] (YAML specifies [N, H, W] = [1, 768, 1])
        // For int32 indices: [1, 768, 1] where N=1, H=768, W=1 (no channel dimension)
        if (logger) logger->info("DPVO::update: Copying indices");
        for (int e = 0; e < num_edges_to_process; e++) {
            // Validate edge index
            if (e < 0 || e >= num_active) {
                if (logger && e < 10) logger->error("DPVO::update: Invalid edge index e={} in index copy, num_active={}", e, num_active);
                continue;
            }
            // YAML layout: [N, H, W] = [1, 768, 1] (no channel dimension for indices)
            // Index: n * H * W + h * W + w
            // Where: n=0, h=e, w=0
            int idx = 0 * MODEL_EDGE_COUNT * 1 + e * 1 + 0;
            ii_input[idx] = static_cast<int32_t>(m_pg.m_ii[e]);
            jj_input[idx] = static_cast<int32_t>(m_pg.m_jj[e]);
            kk_input[idx] = static_cast<int32_t>(m_pg.m_kk[e]);
        }
        if (logger) logger->info("DPVO::update: Indices copied, calling runInference");
        
        // Call update model inference synchronously
        DPVOUpdate_Prediction pred;
        if (logger) {
            logger->info("DPVO::update: About to call m_updateModel->runInference");
            logger->info("DPVO::update: Input data ready - net_input size={}, inp_input size={}, corr_input size={}",
                         net_input.size(), inp_input.size(), corr_input.size());
        }
        bool inference_success = m_updateModel->runInference(
                net_input.data(),
                inp_input.data(),
                corr_input.data(),
                ii_input.data(),
                jj_input.data(),
                kk_input.data(),
                m_updateFrameCounter++,
                pred);
        
        if (logger) {
            logger->info("DPVO::update: runInference returned: {}", inference_success);
        }
        
        if (inference_success)
        {
            if (logger) logger->info("DPVO::update: runInference returned true, extracting outputs");
            // Extract outputs: net_out [1, 384, 768, 1], d_out [1, 2, 768, 1], w_out [1, 2, 768, 1]
            // d_out contains delta: [1, 2, 768, 1] -> [num_edges, 2]
            // w_out contains weight: [1, 2, 768, 1] -> we'll use first channel
            
            if (pred.dOutBuff != nullptr && pred.wOutBuff != nullptr) {
                // Extract delta from d_out: YAML layout [N, C, H, W] = [1, 2, 768, 1]
                for (int e = 0; e < num_edges_to_process; e++) {
                    // d_out layout: [N, C, H, W] = [1, 2, 768, 1]
                    // Index: n * C * H * W + c * H * W + h * W + w
                    // Where: n=0, c=0 or 1, h=e, w=0
                    int idx0 = 0 * 2 * MODEL_EDGE_COUNT * 1 + 0 * MODEL_EDGE_COUNT * 1 + e * 1 + 0;
                    int idx1 = 0 * 2 * MODEL_EDGE_COUNT * 1 + 1 * MODEL_EDGE_COUNT * 1 + e * 1 + 0;
                    delta[e * 2 + 0] = pred.dOutBuff[idx0];
                    delta[e * 2 + 1] = pred.dOutBuff[idx1];
                    
                    // w_out layout: [1, 2, 768, 1] - use first channel (c=0) for weight
                    weight[e] = pred.wOutBuff[idx0];
                }
                
                // Update m_pg.m_net with net_out if available
                if (pred.netOutBuff != nullptr) {
                    if (logger) logger->info("DPVO::update: Updating m_pg.m_net from net_out");
                    // net_out: YAML layout [N, C, H, W] = [1, 384, 768, 1]
                    float net_out_min = std::numeric_limits<float>::max();
                    float net_out_max = std::numeric_limits<float>::lowest();
                    for (int e = 0; e < num_edges_to_process; e++) {
                        for (int d = 0; d < 384; d++) {
                            // Index: n * C * H * W + c * H * W + h * W + w
                            // Where: n=0, c=d, h=e, w=0
                            int idx = 0 * 384 * MODEL_EDGE_COUNT * 1 + d * MODEL_EDGE_COUNT * 1 + e * 1 + 0;
                            float val = pred.netOutBuff[idx];
                            m_pg.m_net[e][d] = val;
                            if (val < net_out_min) net_out_min = val;
                            if (val > net_out_max) net_out_max = val;
                        }
                    }
                    if (logger) {
                        logger->info("DPVO::update: net_out range: [{}, {}], updated m_pg.m_net for {} edges",
                                     net_out_min, net_out_max, num_edges_to_process);
                    }
                } else {
                    if (logger) logger->warn("DPVO::update: pred.netOutBuff is null - m_pg.m_net will not be updated");
                }
            }
            
            // Free prediction buffers
            if (pred.netOutBuff) delete[] pred.netOutBuff;
            if (pred.dOutBuff) delete[] pred.dOutBuff;
            if (pred.wOutBuff) delete[] pred.wOutBuff;
        } else {
            if (logger) {
                logger->warn("DPVO::update: runInference returned false - using zero delta/weight fallback");
                logger->warn("DPVO::update: This means m_pg.m_net will remain unchanged (may stay zero)");
            }
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
    if (logger) logger->info("DPVO::update: Computing target positions, num_active={}", num_active);
    for (int e = 0; e < num_active; e++) {
        // Validate edge index
        if (e < 0 || e >= num_active) {
            if (logger && e < 10) logger->error("DPVO::update: Invalid edge index e={} in target positions, num_active={}", e, num_active);
            continue;
        }
        // Get center pixel coordinates (i0=1, j0=1 for P=3)
        int center_i0 = P / 2;  // 1 for P=3
        int center_j0 = P / 2;  // 1 for P=3
        // coords layout: [num_active][2][P][P] flattened
        // For edge e, channel c (0=x, 1=y), pixel (i0, j0): coords[e * 2 * P * P + c * P * P + i0 * P + j0]
        int coord_x_idx = e * 2 * P * P + 0 * P * P + center_i0 * P + center_j0;
        int coord_y_idx = e * 2 * P * P + 1 * P * P + center_i0 * P + center_j0;
        float cx = coords[coord_x_idx];
        float cy = coords[coord_y_idx];

        m_pg.m_target[e * 2 + 0] = cx + delta[e * 2 + 0];
        m_pg.m_target[e * 2 + 1] = cy + delta[e * 2 + 1];
        m_pg.m_weight[e] = weight[e];
    }
    if (logger) logger->info("DPVO::update: Target positions computed");

    // -------------------------------------------------
    // 6. Bundle Adjustment
    // -------------------------------------------------
    if (logger) logger->info("DPVO::update: Starting bundle adjustment");
    try {
        bundleAdjustment(1e-4f, 100.0f, false, 1);
        if (logger) logger->info("DPVO::update: Bundle adjustment completed");
    } catch (const std::exception& e) {
        if (logger) logger->error("DPVO::update: Bundle adjustment exception: {}", e.what());
    } catch (...) {
        if (logger) logger->error("DPVO::update: Bundle adjustment unknown exception");
    }
    if (logger) logger->info("DPVO::update: update() completed successfully");

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
    
    auto logger = spdlog::get("dpvo");
    if (logger) {
        logger->info("DPVO::keyframe: n={}, i={}, j={}, motion={}, threshold={}, will_remove={}", 
                     n, i, j, 0.5f * m, m_cfg.KEYFRAME_THRESH, (0.5f * m < m_cfg.KEYFRAME_THRESH));
    }

    // =============================================================
    // Phase A: Keyframe removal decision
    // =============================================================
    if (0.5f * m < m_cfg.KEYFRAME_THRESH) {
        int k = n - m_cfg.KEYFRAME_INDEX;
        if (logger) {
            logger->info("DPVO::keyframe: Removing keyframe k={}, m_pg.m_n will decrease from {} to {}", 
                         k, n, n - 1);
        }

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
// Motion magnitude (based on Python motionmag)
// -------------------------------------------------------------
float DPVO::motionMagnitude(int i, int j) {
    // Find active edges where ii == i and jj == j
    const int num_active = m_pg.m_num_edges;
    if (num_active == 0) {
        return 0.0f;
    }
    
    // Collect edges matching (i, j)
    std::vector<int> matching_ii, matching_jj, matching_kk;
    for (int e = 0; e < num_active; e++) {
        if (m_pg.m_ii[e] == i && m_pg.m_jj[e] == j) {
            matching_ii.push_back(m_pg.m_ii[e]);
            matching_jj.push_back(m_pg.m_jj[e]);
            matching_kk.push_back(m_pg.m_kk[e]);
        }
    }
    
    // If no matching edges, return 0.0
    if (matching_ii.empty()) {
        return 0.0f;
    }
    
    // Flattened pointers to patches and intrinsics
    float* patches_flat = &m_pg.m_patches[0][0][0][0][0];
    float* intrinsics_flat = &m_pg.m_intrinsics[0][0];
    
    // Allocate output for flow magnitudes
    std::vector<float> flow_out(matching_ii.size());
    
    // Call flow_mag with matching edges
    pops::flow_mag(
        m_pg.m_poses,
        patches_flat,
        intrinsics_flat,
        matching_ii.data(),
        matching_jj.data(),
        matching_kk.data(),
        static_cast<int>(matching_ii.size()),
        m_cfg.PATCHES_PER_FRAME,
        m_P,
        0.5f,  // beta = 0.5 (from Python default)
        flow_out.data(),
        nullptr  // valid_out not needed
    );
    
    // Return mean flow
    float sum = 0.0f;
    for (float f : flow_out) {
        sum += f;
    }
    return (matching_ii.size() > 0) ? (sum / static_cast<float>(matching_ii.size())) : 0.0f;
}

// -------------------------------------------------------------
// Motion probe (based on Python motion_probe)
// -------------------------------------------------------------
float DPVO::motionProbe() {
    // Python: kk = torch.arange(self.m-self.M, self.m, device="cuda")
    //         jj = self.n * torch.ones_like(kk)
    //         ii = self.ix[kk]
    //         net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
    //         coords = self.reproject(indicies=(ii, jj, kk))
    //         corr = self.corr(coords, indicies=(kk, jj))
    //         ctx = self.imap[:,kk % (self.M * self.pmem)]
    //         net, (delta, weight, _) = self.network.update(net, ctx, corr, None, ii, jj, kk)
    //         return torch.quantile(delta.norm(dim=-1).float(), 0.5)
    
    const int M = m_cfg.PATCHES_PER_FRAME;
    const int P = m_P;
    
    // Get patches from last frame: kk = [m-M, m-M+1, ..., m-1]
    int m = m_pg.m_m;
    int n = m_pg.m_n;
    
    if (m < M || n == 0) {
        return 0.0f;  // Not enough patches/frames
    }
    
    std::vector<int> kk_vec, jj_vec, ii_vec;
    for (int k = m - M; k < m; k++) {
        kk_vec.push_back(k);
        jj_vec.push_back(n);  // Target is current frame (n)
        
        // Extract frame and patch from linear index
        int frame = k / M;
        int patch = k % M;
        ii_vec.push_back(m_pg.m_index[frame][patch]);
    }
    
    int num_edges = static_cast<int>(kk_vec.size());
    if (num_edges == 0) {
        return 0.0f;
    }
    
    // Reproject
    std::vector<float> coords(num_edges * 2 * P * P);
    reproject(ii_vec.data(), jj_vec.data(), kk_vec.data(), num_edges, coords.data());
    
    // Correlation (simplified - we need correlation computation)
    // For now, we'll use a simplified version that just computes delta norm
    // In full implementation, we'd call computeCorrelation and then update model
    
    // Simplified: compute delta from reprojection error
    // Python computes delta from network update, but for motion probe we can use a simpler metric
    std::vector<float> delta_norms(num_edges);
    for (int e = 0; e < num_edges; e++) {
        // Get center pixel coordinates
        int center_i0 = P / 2;
        int center_j0 = P / 2;
        int coord_x_idx = e * 2 * P * P + 0 * P * P + center_i0 * P + center_j0;
        int coord_y_idx = e * 2 * P * P + 1 * P * P + center_i0 * P + center_j0;
        
        // For motion probe, we compute the magnitude of the reprojection
        // This is a simplified version - full version would use network update
        float dx = coords[coord_x_idx];
        float dy = coords[coord_y_idx];
        delta_norms[e] = std::sqrt(dx * dx + dy * dy);
    }
    
    // Compute median (quantile 0.5)
    std::sort(delta_norms.begin(), delta_norms.end());
    float median = delta_norms[delta_norms.size() / 2];
    
    return median;
}


void DPVO::reproject(
    const int* ii,
    const int* jj,
    const int* kk,
    int num_edges,
    float* coords_out,
    float* Ji_out,
    float* Jj_out,
    float* Jz_out,
    float* valid_out)
{
    if (num_edges <= 0)
        return;

    // Flattened pointers to patches and intrinsics
    float* patches_flat = &m_pg.m_patches[0][0][0][0][0];
    float* intrinsics_flat = &m_pg.m_intrinsics[0][0];

    const int P = m_P;
    
    // Allocate temporary buffers if Jacobians are not provided
    std::vector<float> Ji_temp, Jj_temp, Jz_temp, valid_temp;
    float* Ji_ptr = Ji_out;
    float* Jj_ptr = Jj_out;
    float* Jz_ptr = Jz_out;
    float* valid_ptr = valid_out;
    
    if (Ji_ptr == nullptr) {
        Ji_temp.resize(num_edges * 2 * P * P * 6);
        Ji_ptr = Ji_temp.data();
    }
    if (Jj_ptr == nullptr) {
        Jj_temp.resize(num_edges * 2 * P * P * 6);
        Jj_ptr = Jj_temp.data();
    }
    if (Jz_ptr == nullptr) {
        Jz_temp.resize(num_edges * 2 * P * P * 1);
        Jz_ptr = Jz_temp.data();
    }
    if (valid_ptr == nullptr) {
        valid_temp.resize(num_edges * P * P);
        valid_ptr = valid_temp.data();
    }

    // Call transformWithJacobians
    pops::transformWithJacobians(
        m_pg.m_poses,         // SE3 poses [N]
        patches_flat,         // flattened patches
        intrinsics_flat,      // flattened intrinsics
        ii, jj, kk,           // indices
        num_edges,            // number of edges
        m_cfg.PATCHES_PER_FRAME,
        m_P,
        coords_out,           // output [num_edges][2][P][P] flattened
        Ji_ptr,               // Jacobian w.r.t. pose i
        Jj_ptr,               // Jacobian w.r.t. pose j
        Jz_ptr,               // Jacobian w.r.t. inverse depth
        valid_ptr             // validity mask
    );

    // Output layout already matches Python coords.permute(0,1,4,2,3)
    // Each edge: [2][P][P] → 2 channels: u,v
    // Jacobians are stored if output buffers were provided
}




void DPVO::terminate() 
{
    stopProcessingThread();
}

// -------------------------------------------------------------
// Helper function to convert tensor to image data (used in updateInput)
// -------------------------------------------------------------
#if defined(CV28) || defined(CV28_SIMULATOR)
static bool convertTensorToImage(ea_tensor_t* imgTensor, std::vector<uint8_t>& image_out, int& H_out, int& W_out)
{
    if (imgTensor == nullptr) {
        return false;
    }
    
    // Get tensor data
    void* tensor_data_ptr = ea_tensor_data_for_read(imgTensor, EA_CPU);
    if (tensor_data_ptr == nullptr) {
        return false;
    }
    
    uint8_t* image_data = static_cast<uint8_t*>(tensor_data_ptr);
    
    // Get pitch and shape
    size_t pitch = ea_tensor_pitch(imgTensor);
    size_t tensor_size = ea_tensor_size(imgTensor);
    const size_t* shape = ea_tensor_shape(imgTensor);
    
    if (shape == nullptr) {
        return false;
    }
    
    int H = static_cast<int>(shape[EA_H]);
    int W = static_cast<int>(shape[EA_W]);
    int C = static_cast<int>(shape[EA_C]);
    
    // Validate dimensions
    if (H <= 0 || W <= 0 || C <= 0 || H > 10000 || W > 10000 || C > 10) {
        return false;
    }
    
    // Allocate output buffer [C, H, W] format
    image_out.resize(H * W * C);
    
    // Convert from [H, W, C] to [C, H, W]
    if (pitch == W * C) {
        // Contiguous memory
        for (int c = 0; c < C; c++) {
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int src_idx = y * W * C + x * C + c;
                    int dst_idx = c * H * W + y * W + x;
                    if (src_idx >= 0 && src_idx < H * W * C && 
                        dst_idx >= 0 && dst_idx < H * W * C) {
                        image_out[dst_idx] = image_data[src_idx];
                    }
                }
            }
        }
    } else {
        // Non-contiguous or planar format
        size_t buffer_size = static_cast<size_t>(H) * W * C;
        std::vector<uint8_t> image_contiguous(buffer_size);
        
        if (pitch < static_cast<size_t>(W * C)) {
            // Planar format
            size_t channel_size = static_cast<size_t>(H) * pitch;
            if (tensor_size < channel_size * C) {
                return false;
            }
            
            for (int c = 0; c < C; c++) {
                size_t channel_offset = static_cast<size_t>(c) * channel_size;
                uint8_t* channel_src = image_data + channel_offset;
                
                for (int y = 0; y < H; y++) {
                    size_t src_row_offset = static_cast<size_t>(y) * pitch;
                    size_t dst_offset = static_cast<size_t>(c) * H * W + y * W;
                    
                    if (channel_offset + src_row_offset + pitch <= tensor_size &&
                        dst_offset + W <= buffer_size) {
                        std::memcpy(image_out.data() + dst_offset, 
                                   channel_src + src_row_offset, 
                                   pitch);
                    }
                }
            }
        } else {
            // Non-contiguous (padded)
            for (int y = 0; y < H; y++) {
                size_t src_offset = static_cast<size_t>(y) * pitch;
                size_t dst_offset = static_cast<size_t>(y) * W * C;
                size_t total_src_size = static_cast<size_t>(H) * pitch;
                
                if (src_offset + W * C <= total_src_size && 
                    dst_offset + W * C <= buffer_size) {
                    std::memcpy(image_contiguous.data() + dst_offset, 
                               image_data + src_offset, 
                               W * C);
                }
            }
            
            // Convert from [H, W, C] to [C, H, W]
            for (int c = 0; c < C; c++) {
                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        int src_idx = y * W * C + x * C + c;
                        int dst_idx = c * H * W + y * W + x;
                        if (src_idx >= 0 && src_idx < H * W * C && 
                            dst_idx >= 0 && dst_idx < H * W * C) {
                            image_out[dst_idx] = image_contiguous[src_idx];
                        }
                    }
                }
            }
        }
    }
    
    H_out = H;
    W_out = W;
    return true;
}
#endif

// -------------------------------------------------------------
// Threading interface (similar to wnc_app)
// -------------------------------------------------------------
void DPVO::startProcessingThread()
{
    // Initialize logger if it doesn't exist
    auto logger = spdlog::get("dpvo");
    if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
        logger = spdlog::stdout_color_mt("dpvo");
        logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    }
    
    m_processingThreadRunning = true;
    m_processingThread = std::thread([this]() {
        // Get or create logger in thread
        auto logger = spdlog::get("dpvo");
        if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
            logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
            logger = spdlog::stdout_color_mt("dpvo");
            logger->set_pattern("[%n] [%^%l%$] %v");
#endif
        }
        
        std::unique_lock<std::mutex> lock(m_queueMutex);
        while (m_processingThreadRunning)
        {
            m_queueCV.wait_for(lock, std::chrono::milliseconds(1000), [this]() {
                return !m_processingThreadRunning || !m_inputFrameQueue.empty();
            });
            if (!m_processingThreadRunning)
                break;

            if (!m_inputFrameQueue.empty())
            {
                InputFrame frame = std::move(m_inputFrameQueue.front());
                m_inputFrameQueue.pop();
                lock.unlock();
                
                m_bDone = false;
                
                try {
                    // Update timestamp (increment for each frame)
                    m_currentTimestamp++;
                    
                    // Call the main processing function (use member variables for intrinsics and timestamp)
					logger->info("Start run DPVO run function");
                    run(m_currentTimestamp, frame.image.data(), m_intrinsics, frame.H, frame.W);
					logger->info("DPVO run function finished");
                } catch (const std::exception& e) {
                    if (logger) logger->error("Exception in frame processing: {}", e.what());
                } catch (...) {
                    if (logger) logger->error("Unknown exception in frame processing");
                }
                
                m_bDone = true;
                lock.lock();
            }
        }
        if (logger) logger->debug("startProcessingThread is terminated");
    });
}

void DPVO::stopProcessingThread()
{
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_processingThreadRunning = false;
    }
    m_queueCV.notify_one();
    if (m_processingThread.joinable())
        m_processingThread.join();
}

void DPVO::wakeProcessingThread()
{
    m_queueCV.notify_one();
}

#if defined(CV28) || defined(CV28_SIMULATOR)
void DPVO::updateInput(ea_tensor_t* imgTensor)
{
    if (imgTensor == nullptr) {
        auto logger = spdlog::get("dpvo");
        if (logger) logger->error("updateInput: imgTensor is nullptr");
        return;
    }
    
    // Get or create logger
    auto logger = spdlog::get("dpvo");
    if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
        logger = spdlog::stdout_color_mt("dpvo");
        logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    }
    
    if (logger) logger->debug("updateInput: Starting tensor conversion");
    
    // Convert tensor to image data (like WNC_APP processes tensor in updateInput)
    // IMPORTANT: Do this conversion BEFORE the tensor might be freed
    InputFrame frame;
    if (!convertTensorToImage(imgTensor, frame.image, frame.H, frame.W)) {
        if (logger) logger->error("updateInput: convertTensorToImage failed");
        return;  // Conversion failed
    }
    
    if (logger) logger->debug("updateInput: Conversion successful, H={}, W={}, image_size={}", 
                              frame.H, frame.W, frame.image.size());
    
    std::lock_guard<std::mutex> lock(m_queueMutex);
    
    m_inputFrameQueue.push(std::move(frame));
    
    if (logger) logger->debug("updateInput: Frame added to queue, queue_size={}", m_inputFrameQueue.size());
    
    // Limit queue size to prevent memory issues
    const size_t MAX_QUEUE_SIZE = 10;
    while (m_inputFrameQueue.size() > MAX_QUEUE_SIZE)
    {
        m_inputFrameQueue.pop();
    }
    
    wakeProcessingThread();
    
    if (logger) logger->debug("updateInput: Finished, notified processing thread");
}

void DPVO::addFrame(ea_tensor_t* imgTensor)
{
    updateInput(imgTensor);
}
#else
// Fallback implementation for non-CV28 platforms
void DPVO::updateInput(const uint8_t* image, int H, int W)
{
    if (image == nullptr)
        return;
        
    std::lock_guard<std::mutex> lock(m_queueMutex);
    
    InputFrame frame;
    frame.image.assign(image, image + H * W * 3);  // Copy image data (assuming RGB)
    frame.H = H;
    frame.W = W;
    
    m_inputFrameQueue.push(std::move(frame));
    
    // Limit queue size to prevent memory issues
    const size_t MAX_QUEUE_SIZE = 10;
    while (m_inputFrameQueue.size() > MAX_QUEUE_SIZE)
    {
        m_inputFrameQueue.pop();
    }
    
    wakeProcessingThread();
}

void DPVO::addFrame(const uint8_t* image, int H, int W)
{
    updateInput(image, H, W);
}
#endif

bool DPVO::_hasWorkToDo()
{
    return !m_inputFrameQueue.empty();
}

bool DPVO::isProcessingComplete()
{
    std::lock_guard<std::mutex> lock(m_queueMutex);
    return m_inputFrameQueue.empty() && m_bDone;
}

