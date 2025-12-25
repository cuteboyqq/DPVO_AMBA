/*
  (C) 2024-2025 Wistron NeWeb Corporation (WNC) - All Rights Reserved

  This software and its associated documentation are the confidential and
  proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
  may not be copied, modified, distributed, or otherwise disclosed to third
  parties without the express written consent of the Company.

  Unauthorized reproduction, distribution, or disclosure of this software and
  its associated documentation or the information contained herein is a
  violation of applicable laws and may result in severe legal penalties.
*/

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "spdlog/spdlog.h"
#include "opencv2/opencv.hpp"

class VideoHandler {
public:
    VideoHandler(const std::string& videoPath, spdlog::logger* logger) 
        : m_logger(logger)
        , m_frameBuffer(FRAME_BUFFER_SIZE)  // Pre-allocate buffer space
    {
			m_capture.open(videoPath);
			if (!m_capture.isOpened()) 
			{
				m_logger->error("Failed to open video file: {}", videoPath);
				throw std::runtime_error("Failed to open video file");
			}

			// Set optimal buffer size for video capture
			m_capture.set(cv::CAP_PROP_BUFFERSIZE, FRAME_BUFFER_SIZE);

			// Get video properties once instead of repeatedly
			m_frameWidth = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH));
			m_frameHeight = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
			
			// Pre-allocate frame buffer to avoid repeated memory allocations
			for (auto& frame : m_frameBuffer)
			{
				frame.create(m_frameHeight, m_frameWidth, CV_8UC3);
			}
    }

    ~VideoHandler() {
        if (m_capture.isOpened()) {
            m_capture.release();
        }
    }

    bool processNextFrame(unsigned int& frameCount, const std::string& dataFolderPath)
		{
			// Use pre-allocated buffer frame
			cv::Mat& currentFrame = m_frameBuffer[m_currentBufferIndex];
			
			if (!m_capture.read(currentFrame))
			{
				return false;
			}

			if (currentFrame.empty())
			{
				m_logger->info("End of video");
				return false;
			}

			// Save the current frame to the data folder with sequential naming
			std::string frameFilename = dataFolderPath + "/" + 
				std::string(5 - std::to_string(frameCount).length(), '0') + 
				std::to_string(frameCount) + ".jpg";
			
			try {
				cv::imwrite(frameFilename, currentFrame);
				m_logger->debug("Saved frame {} to {}", frameCount, frameFilename);
			} catch (const cv::Exception& e) {
				m_logger->error("Failed to save frame {}: {}", frameCount, e.what());
			}
			
			// Update frame counter and buffer index
			frameCount++;
			m_currentBufferIndex = (m_currentBufferIndex + 1) % FRAME_BUFFER_SIZE;

			return true;
    }

private:
    cv::VideoCapture m_capture;
    spdlog::logger* m_logger;
    static constexpr size_t FRAME_BUFFER_SIZE = 32; // Adjust based on memory constraints
    std::vector<cv::Mat> m_frameBuffer;
    size_t m_currentBufferIndex = 0;
    int m_frameWidth = 0;
    int m_frameHeight = 0;
};