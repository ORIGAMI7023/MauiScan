cmake_minimum_required(VERSION 3.18.1)
project(opencv_scanner)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../MauiScan/Platforms/Android/libs/${ANDROID_ABI})

# OpenCV 路径
if(NOT DEFINED OPENCV_ANDROID_SDK)
    set(OPENCV_ANDROID_SDK "D:/OpenCV-android-sdk" CACHE PATH "Path to OpenCV Android SDK")
endif()

# OpenCV 头文件
set(OPENCV_INCLUDE_DIR "${OPENCV_ANDROID_SDK}/sdk/native/jni/include")

# OpenCV 共享库
set(OPENCV_LIB_DIR "${OPENCV_ANDROID_SDK}/sdk/native/libs/${ANDROID_ABI}")
add_library(opencv_java4 SHARED IMPORTED)
set_target_properties(opencv_java4 PROPERTIES IMPORTED_LOCATION "${OPENCV_LIB_DIR}/libopencv_java4.so")

# 包含目录
include_directories(
    ${OPENCV_INCLUDE_DIR}
    ../src
)

# 源文件
add_library(
    opencv_scanner
    SHARED
    ../src/opencv_scanner.cpp
)

# 链接库（包含 C++ 标准库）
target_link_libraries(
    opencv_scanner
    opencv_java4
    log
    c++_shared
)

# 设置编译选项
target_compile_options(opencv_scanner PRIVATE
    -fvisibility=hidden
    -Wall
    -Wextra
)

# 复制 OpenCV 共享库到输出目录
add_custom_command(TARGET opencv_scanner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OPENCV_LIB_DIR}/libopencv_java4.so"
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libopencv_java4.so"
)
