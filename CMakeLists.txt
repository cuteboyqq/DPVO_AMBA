cmake_minimum_required(VERSION 3.10)
project(wnc-app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile definitions
add_definitions(-DCV28_SIMULATOR)
add_definitions(-DSAVE_OUTPUT_TENSOR)

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

# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../build
) 