/*
  (C) 2025-2026 Wistron NeWeb Corporation (WNC) - All Rights Reserved
  YOLOv8 for DPVO: synchronous inference on processing thread.
*/

#include "yolov8.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <sys/stat.h>

#if defined(CV28) || defined(CV28_SIMULATOR)

YOLOv8::YOLOv8(Config_S* config, WakeCallback wakeFunc)
    : m_numAnchorBox((YOLOV8_MODEL_WIDTH * YOLOV8_MODEL_HEIGHT / 64) +
                     (YOLOV8_MODEL_WIDTH * YOLOV8_MODEL_HEIGHT / 256) +
                     (YOLOV8_MODEL_WIDTH * YOLOV8_MODEL_HEIGHT / 1024)),
      m_boxBufferSize(4 * m_numAnchorBox),
      m_confBufferSize(m_numAnchorBox),
      m_classBufferSize(m_numAnchorBox),
      m_wakeFunc(wakeFunc)
{
    auto logger = spdlog::get("dpvo");
    if (!logger) {
        logger = spdlog::stdout_color_mt("yolov8");
        logger->set_pattern("[%n] [%^%l%$] %v");
    }
    m_estimateTime = config && config->stShowProcTimeConfig.AIModel;
    m_saveRawImage = config && config->stDebugConfig.saveRawImages ? 1 : 0;
    m_inputMode = DETECTION_MODE_FILE;

    m_modelPathStr = (config && !config->yolov8ModelPath.empty()) ? config->yolov8ModelPath : "";
    m_ptrModelPath = m_modelPathStr.empty() ? nullptr : const_cast<char*>(m_modelPathStr.c_str());

    if (m_modelPathStr.empty()) {
        if (logger) logger->info("YOLOv8: no yolov8ModelPath set, YOLOv8 disabled");
        m_objBoxBuff = nullptr;
        m_objConfBuff = nullptr;
        m_objClsBuff = nullptr;
        return;
    }

    ea_net_params_t net_params;
    memset(&net_params, 0, sizeof(net_params));
    net_params.acinf_gpu_id = -1;
    m_model = ea_net_new(&net_params);
    if (!m_model) {
        if (logger) logger->error("YOLOv8: creating model failed");
        return;
    }
    m_inputTensor = nullptr;
    m_outputTensors.resize(m_outputTensorList.size());
    m_objBoxBuff = new float[m_boxBufferSize];
    m_objConfBuff = new float[m_confBufferSize];
    m_objClsBuff = new float[m_classBufferSize];
    _initModelIO();
}

YOLOv8::~YOLOv8()
{
#if defined(CV28) || defined(CV28_SIMULATOR)
    _releaseInputTensor();
    _releaseOutputTensor();
    _releaseTensorBuffers();
    _releaseModel();
#endif
}

bool YOLOv8::_releaseModel()
{
    if (m_model) {
        ea_net_free(m_model);
        m_model = nullptr;
    }
    return true;
}

bool YOLOv8::_releaseInputTensor()
{
    m_inputTensor = nullptr;
    return true;
}

bool YOLOv8::_releaseOutputTensor()
{
    for (size_t i = 0; i < m_outputTensors.size(); i++)
        m_outputTensors[i] = nullptr;
    return true;
}

bool YOLOv8::_releaseTensorBuffers()
{
    if (m_objBoxBuff)  { delete[] m_objBoxBuff;  m_objBoxBuff = nullptr; }
    if (m_objConfBuff) { delete[] m_objConfBuff; m_objConfBuff = nullptr; }
    if (m_objClsBuff)  { delete[] m_objClsBuff;  m_objClsBuff = nullptr; }
    return true;
}

void YOLOv8::_initModelIO()
{
    auto logger = spdlog::get("yolov8");
    if (!logger) logger = spdlog::get("dpvo");
    if (!m_model || m_modelPathStr.empty()) return;

    int rval = ea_net_config_input(m_model, m_inputTensorName.c_str());
    for (const auto& name : m_outputTensorList)
        ea_net_config_output(m_model, name.c_str());
    FILE* file = fopen(m_modelPathStr.c_str(), "r");
    if (!file) {
        if (logger) logger->error("YOLOv8: model file does not exist: {}", m_modelPathStr);
        return;
    }
    fclose(file);
    rval = ea_net_load(m_model, EA_NET_LOAD_FILE, (void*)m_ptrModelPath, 1);
    m_inputTensor = ea_net_input(m_model, m_inputTensorName.c_str());
    if (m_inputTensor) {
        m_inputHeight = static_cast<int>(ea_tensor_shape(m_inputTensor)[EA_H]);
        m_inputWidth  = static_cast<int>(ea_tensor_shape(m_inputTensor)[EA_W]);
        m_inputChannel = static_cast<int>(ea_tensor_shape(m_inputTensor)[EA_C]);
        if (logger) logger->info("YOLOv8: input {}x{}x{}", m_inputHeight, m_inputWidth, m_inputChannel);
    }
    for (size_t i = 0; i < m_outputTensorList.size(); i++)
        m_outputTensors[i] = ea_net_output_by_index(m_model, static_cast<int>(i));
}

bool YOLOv8::_checkSavedTensor(int /*frameIdx*/)
{
    return false;
}

bool YOLOv8::_loadInput(ea_tensor_t* imgTensor)
{
    if (_preProcessingMemory(imgTensor) != EA_SUCCESS) {
        auto logger = spdlog::get("yolov8");
        if (logger) logger->error("YOLOv8: load input failed");
        return false;
    }
    return true;
}

int YOLOv8::_preProcessingMemory(ea_tensor_t* imgTensor)
{
    int rval;
    if (m_inputMode == DETECTION_MODE_FILE) {
#if defined(CV28)
    rval = ea_cvt_color_resize(imgTensor, m_inputTensor, EA_COLOR_BGR2RGB, EA_VP);
#else
    rval = ea_cvt_color_resize(imgTensor, m_inputTensor, EA_COLOR_BGR2RGB, EA_CPU);
#endif
    } else {
#if defined(CV28)
    rval = ea_cvt_color_resize(imgTensor, m_inputTensor, EA_COLOR_YUV2RGB_NV12, EA_VP);
#else
    rval = ea_cvt_color_resize(imgTensor, m_inputTensor, EA_COLOR_YUV2RGB_NV12, EA_CPU);
#endif
    }
    return rval;
}

bool YOLOv8::runSync(ea_tensor_t* imgTensor, int frameIdx, YOLOv8_Prediction& out_pred)
{
    if (!m_model || !m_inputTensor) return false;
    auto logger = spdlog::get("yolov8");
    if (!logger) logger = spdlog::get("dpvo");

    if (out_pred.objBoxBuff) { delete[] out_pred.objBoxBuff; out_pred.objBoxBuff = nullptr; }
    if (out_pred.objConfBuff) { delete[] out_pred.objConfBuff; out_pred.objConfBuff = nullptr; }
    if (out_pred.objClsBuff) { delete[] out_pred.objClsBuff; out_pred.objClsBuff = nullptr; }

    if (!_loadInput(imgTensor)) return false;

    if (EA_SUCCESS != ea_net_forward(m_model, 1)) {
        if (logger) logger->error("YOLOv8: ea_net_forward failed");
        return false;
    }
#if defined(CV28)
    if (ea_tensor_sync_cache(m_outputTensors[0], EA_VP, EA_CPU) != EA_SUCCESS ||
        ea_tensor_sync_cache(m_outputTensors[1], EA_VP, EA_CPU) != EA_SUCCESS ||
        ea_tensor_sync_cache(m_outputTensors[2], EA_VP, EA_CPU) != EA_SUCCESS) {
        if (logger) logger->error("YOLOv8: sync cache failed");
    }
#endif
    m_outputTensors[0] = ea_net_output_by_index(m_model, 0);
    m_outputTensors[1] = ea_net_output_by_index(m_model, 1);
    m_outputTensors[2] = ea_net_output_by_index(m_model, 2);

    out_pred.objBoxBuff = new float[m_boxBufferSize];
    out_pred.objConfBuff = new float[m_confBufferSize];
    out_pred.objClsBuff = new float[m_classBufferSize];
    std::memcpy(out_pred.objBoxBuff, (float*)ea_tensor_data(m_outputTensors[0]), m_boxBufferSize * sizeof(float));
    std::memcpy(out_pred.objConfBuff, (float*)ea_tensor_data(m_outputTensors[1]), m_confBufferSize * sizeof(float));
    std::memcpy(out_pred.objClsBuff, (float*)ea_tensor_data(m_outputTensors[2]), m_classBufferSize * sizeof(float));
    out_pred.img = imgUtil::convertTensorToMat(imgTensor);
    out_pred.isProcessed = true;
    return true;
}

bool YOLOv8::_run(ea_tensor_t* imgTensor, int frameIdx)
{
    std::unique_lock<std::mutex> pred_lock(m_pred_mutex);
    if (!_loadInput(imgTensor)) return false;
    cv::Mat img = imgUtil::convertTensorToMat(imgTensor);
    if (_checkSavedTensor(frameIdx)) {
        m_pred.objBoxBuff = new float[m_boxBufferSize];
        m_pred.objConfBuff = new float[m_confBufferSize];
        m_pred.objClsBuff = new float[m_classBufferSize];
        std::memcpy(m_pred.objBoxBuff, m_objBoxBuff, m_boxBufferSize * sizeof(float));
        std::memcpy(m_pred.objConfBuff, m_objConfBuff, m_confBufferSize * sizeof(float));
        std::memcpy(m_pred.objClsBuff, m_objClsBuff, m_classBufferSize * sizeof(float));
    } else {
        if (EA_SUCCESS != ea_net_forward(m_model, 1)) return false;
#if defined(CV28)
        ea_tensor_sync_cache(m_outputTensors[0], EA_VP, EA_CPU);
        ea_tensor_sync_cache(m_outputTensors[1], EA_VP, EA_CPU);
        ea_tensor_sync_cache(m_outputTensors[2], EA_VP, EA_CPU);
#endif
        m_outputTensors[0] = ea_net_output_by_index(m_model, 0);
        m_outputTensors[1] = ea_net_output_by_index(m_model, 1);
        m_outputTensors[2] = ea_net_output_by_index(m_model, 2);
        m_pred.objBoxBuff = new float[m_boxBufferSize];
        m_pred.objConfBuff = new float[m_confBufferSize];
        m_pred.objClsBuff = new float[m_classBufferSize];
        std::memcpy(m_pred.objBoxBuff, (float*)ea_tensor_data(m_outputTensors[0]), m_boxBufferSize * sizeof(float));
        std::memcpy(m_pred.objConfBuff, (float*)ea_tensor_data(m_outputTensors[1]), m_confBufferSize * sizeof(float));
        std::memcpy(m_pred.objClsBuff, (float*)ea_tensor_data(m_outputTensors[2]), m_classBufferSize * sizeof(float));
    }
    m_pred.img = img;
    m_predictionBuffer.push_back({frameIdx, m_pred});
    m_bProcessed = true;
    notifyProcessingComplete();
    return true;
}

void YOLOv8::runThread()
{
    m_threadInference = std::thread(&YOLOv8::_runInferenceFunc, this);
}

void YOLOv8::stopThread()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_threadTerminated = true;
    }
    m_condition.notify_all();
    if (m_threadInference.joinable())
        m_threadInference.join();
}

void YOLOv8::_runInferenceFunc()
{
    while (!m_threadTerminated) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_condition.wait(lock, [this]() {
            return m_threadTerminated || (!m_inputFrameBuffer.empty() && m_threadStarted);
        });
        if (m_threadTerminated) break;
        auto pair = m_inputFrameBuffer.front();
        m_inputFrameBuffer.pop_front();
        int frameIdx = pair.first;
        ea_tensor_t* imgTensor = pair.second;
        lock.unlock();
        _run(imgTensor, frameIdx);
    }
}

void YOLOv8::notifyProcessingComplete()
{
    if (m_wakeFunc) m_wakeFunc();
}

void YOLOv8::updateInputFrame(ea_tensor_t* imgTensor, int frameIdx)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_inputFrameBuffer.emplace_back(frameIdx, imgTensor);
    m_threadStarted = true;
    m_bDone = false;
    m_condition.notify_one();
}

bool YOLOv8::getLastestPrediction(YOLOv8_Prediction& pred, int& frameIdx)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_predictionBuffer.empty()) return false;
    auto pair = m_predictionBuffer.front();
    m_predictionBuffer.pop_front();
    frameIdx = pair.first;
    pred = pair.second;
    return true;
}

bool YOLOv8::isInputBufferEmpty() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_inputFrameBuffer.empty();
}

bool YOLOv8::isPredictionBufferEmpty() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_predictionBuffer.empty();
}

#endif // CV28 || CV28_SIMULATOR
