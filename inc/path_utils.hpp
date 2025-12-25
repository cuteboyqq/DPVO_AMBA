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
#include <dirent.h>


//
enum class InputType { Video, ImageDirectory, Unknown };

bool is_directory(const std::string& path) {
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0) {
        return false;
    }
    return S_ISDIR(statbuf.st_mode);
}

bool is_file(const std::string& path) {
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0) {
        return false;
    }
    return S_ISREG(statbuf.st_mode);
}

bool is_image_file(const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    const std::vector<std::string> imageExts = {
        "jpg", "jpeg", "png", "bmp", "tiff"
    };
    return std::find(imageExts.begin(), imageExts.end(), ext) != imageExts.end();
}

InputType detectInputType(const std::string& path)
{
    if (is_directory(path))
    {
        // Check if directory contains image files
        DIR* dir;
        struct dirent* ent;
        if ((dir = opendir(path.c_str())) != NULL)
        {
            while ((ent = readdir(dir)) != NULL)
            {
                std::string filename = ent->d_name;
                if (is_image_file(filename))
                {
                    closedir(dir);
                    return InputType::ImageDirectory;
                }
            }
            closedir(dir);
        }
    }
    else if (is_file(path))
    {
        // Get lowercase extension
        std::string filename = path;
        std::string ext;
        size_t pos = filename.find_last_of('.');
        if (pos != std::string::npos) {
            ext = filename.substr(pos);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        }

        // Check against supported video extensions
        const std::vector<std::string> videoExts = {
            ".mp4", ".avi", ".mov", ".mkv", ".flv", 
            ".wmv", ".webm", ".m4v", ".mpeg", ".mpg"
        };

        if (std::find(videoExts.begin(), videoExts.end(), ext) != videoExts.end()) {
            return InputType::Video;
        }
    }
    
    return InputType::Unknown;
}

// Global constants
namespace constants {
	constexpr int FAILURE = 1;
	constexpr int SUCCESS = 0;
	
	// File extensions for video validation
	const std::vector<std::string> VIDEO_EXTENSIONS = {
			".mp4", ".avi", ".mov", ".mkv", ".flv", 
			".wmv", ".webm", ".m4v", ".mpeg", ".mpg"
	};
	
	// Image extensions for validation
	const std::vector<std::string> IMAGE_EXTENSIONS = {
			".jpg", ".jpeg", ".png", ".bmp", ".tiff"
	};
}

class PathUtils {
public:
    // Checks if a given extension is supported for the specified type
    static bool isSupportedExtension(const std::string& filename, const std::vector<std::string>& validExtensions) {
        std::string ext = getFileExtension(filename);
        return std::find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end();
    }

    static bool isVideoFile(const std::string& filename) {
        return isSupportedExtension(filename, constants::VIDEO_EXTENSIONS);
    }
    
    static bool isImageFile(const std::string& filename) {
        return isSupportedExtension(filename, constants::IMAGE_EXTENSIONS);
    }

    static std::string extractBaseName(const std::string& path) {
        // Remove trailing slashes
        std::string trimmedPath = path;
        while (!trimmedPath.empty() && isPathSeparator(trimmedPath.back())) {
            trimmedPath.pop_back();
        }
        
        // Find last directory separator
        size_t lastSlash = trimmedPath.find_last_of("/\\");
        std::string fileName = (lastSlash == std::string::npos) ? 
                             trimmedPath : trimmedPath.substr(lastSlash + 1);
        
        // Convert to lowercase for case-insensitive comparison
        std::string lowerFileName = toLowerCase(fileName);
        
        // Remove extension if it's a video file
        for (const auto& ext : constants::VIDEO_EXTENSIONS) {
            if (lowerFileName.length() > ext.length() && 
                lowerFileName.substr(lowerFileName.length() - ext.length()) == ext) {
                return fileName.substr(0, fileName.length() - ext.length());
            }
        }
        
        return fileName;
    }

    static std::string getFileExtension(const std::string& filename) {
        size_t pos = filename.find_last_of('.');
        if (pos == std::string::npos) return "";
        std::string ext = filename.substr(pos);
        return toLowerCase(ext);
    }
    
    static bool isPathSeparator(char c) {
        return c == '/' || c == '\\';
    }
    
    static std::string toLowerCase(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
};