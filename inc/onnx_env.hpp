#pragma once

#ifdef USE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#include <mutex>

// Singleton wrapper for Ort::Env
// ONNX Runtime requires a single global Ort::Env instance
class OnnxEnvSingleton {
public:
    static Ort::Env& getInstance() {
        static OnnxEnvSingleton instance;
        return instance.m_env;
    }
    
    // Prevent copying
    OnnxEnvSingleton(const OnnxEnvSingleton&) = delete;
    OnnxEnvSingleton& operator=(const OnnxEnvSingleton&) = delete;
    
private:
    OnnxEnvSingleton() 
        : m_env(ORT_LOGGING_LEVEL_WARNING, "DPVO_ONNX") {
    }
    
    ~OnnxEnvSingleton() = default;
    
    Ort::Env m_env;
};

#endif

