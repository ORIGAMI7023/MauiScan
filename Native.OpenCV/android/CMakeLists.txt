cmake_minimum_required(VERSION 3.18.1)
project(opencv_scanner)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../../MauiScan/Platforms/Android/libs/${ANDROID_ABI})

# OpenCV 路径（从命令行参数或默认路径获取）
if(NOT DEFINED OPENCV_ANDROID_SDK)
    set(OPENCV_ANDROID_SDK "D:/OpenCV-android-sdk" CACHE PATH "Path to OpenCV Android SDK")
endif()
set(OpenCV_DIR "${OPENCV_ANDROID_SDK}/sdk/native/jni")

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ../src
)

# 源文件
add_library(
    opencv_scanner
    SHARED
    ../src/opencv_scanner.cpp
)

# 链接 OpenCV 库
target_link_libraries(
    opencv_scanner
    ${OpenCV_LIBS}
    log
)

# 设置编译选项
target_compile_options(opencv_scanner PRIVATE
    -fvisibility=hidden
    -Wall
    -Wextra
)

# 导出符号
set_target_properties(opencv_scanner PROPERTIES
    LINK_FLAGS "-Wl,--exclude-libs,ALL"
)
