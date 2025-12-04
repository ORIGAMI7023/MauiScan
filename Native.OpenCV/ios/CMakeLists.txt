cmake_minimum_required(VERSION 3.18.1)
project(opencv_scanner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum iOS deployment version")

# OpenCV iOS SDK 路径
set(OPENCV_IOS_SDK "${CMAKE_SOURCE_DIR}/../../OpenCVSDK/ios" CACHE PATH "Path to OpenCV iOS SDK")
set(OPENCV_INCLUDE_DIR "${OPENCV_IOS_SDK}/Headers")
set(OPENCV_LIBRARY "${OPENCV_IOS_SDK}/opencv2")

# 检查 OpenCV 是否存在
if(NOT EXISTS "${OPENCV_INCLUDE_DIR}")
    message(FATAL_ERROR "OpenCV headers not found at ${OPENCV_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${OPENCV_LIBRARY}")
    message(FATAL_ERROR "OpenCV library not found at ${OPENCV_LIBRARY}")
endif()

# 添加源文件
add_library(opencv_scanner STATIC
    ../src/opencv_scanner.cpp
)

# 设置包含目录
target_include_directories(opencv_scanner PRIVATE
    ${OPENCV_INCLUDE_DIR}
    ../src
)

# 链接 OpenCV 和系统框架
target_link_libraries(opencv_scanner
    ${OPENCV_LIBRARY}
    "-framework Foundation"
    "-framework CoreGraphics"
    "-framework UIKit"
    "-framework Accelerate"
    "-framework CoreMedia"
    "-framework CoreVideo"
    "-framework AVFoundation"
    "-framework QuartzCore"
)

# 设置编译选项
target_compile_options(opencv_scanner PRIVATE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
)
