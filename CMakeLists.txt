cmake_minimum_required(VERSION 3.10)
project(wnc-app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default for performance-critical correlation kernel
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Add compile definitions
add_definitions(-DCV28_SIMULATOR)
add_definitions(-DSAVE_OUTPUT_TENSOR)

# Optional: ONNX Runtime (for .onnx model inference; requires ONNXRUNTIME_ROOT or system install)
# When ONNX Runtime is not available, the build uses AMBA models (expects .json paths).
set(USE_ONNX FALSE)
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Path to ONNX Runtime (e.g. /tmp/onnxruntime-linux-x64-1.16.3)")
if(ONNXRUNTIME_ROOT)
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
    set(USE_ONNX TRUE)
    add_definitions(-DUSE_ONNX_RUNTIME)
    include_directories(${ONNXRUNTIME_ROOT}/include)
    link_directories(${ONNXRUNTIME_ROOT}/lib)
else()
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS /usr/local/lib /usr/lib)
    if(ONNXRUNTIME_LIB)
        message(STATUS "ONNX Runtime: found ${ONNXRUNTIME_LIB}")
        set(USE_ONNX TRUE)
        add_definitions(-DUSE_ONNX_RUNTIME)
    endif()
endif()

# Include directories
include_directories(
    ./lib/c/inc
    ./inc/
    /usr/include/opencv4
)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find spdlog and fmt
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

# Find Pangolin for 3D viewer (poses + point cloud)
find_package(Pangolin REQUIRED)
find_package(X11 REQUIRED)

# Add compile definitions
add_definitions(-DENABLE_PANGOLIN_VIEWER)

# Include Pangolin headers
include_directories(${Pangolin_INCLUDE_DIRS})

# Collect source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
# Exclude se.cpp: use se3.cpp (lietorch-matching implementation) for correct viewer/BA
list(FILTER SOURCES EXCLUDE REGEX "se\\.cpp$")

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${CMAKE_THREAD_LIBS_INIT}
    ${OpenCV_LIBS}
    spdlog::spdlog
    fmt::fmt
    ${Pangolin_LIBRARIES}
    ${X11_LIBRARIES}
)

# Link EazyAI library (if available)
target_link_directories(${PROJECT_NAME} PRIVATE ./lib/libeazyai/build)
target_link_libraries(${PROJECT_NAME} eazyai)

# Link ONNX Runtime if enabled
if(USE_ONNX)
    target_link_libraries(${PROJECT_NAME} onnxruntime)
endif()

# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../build
) 