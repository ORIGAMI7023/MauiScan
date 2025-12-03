#!/bin/bash
# Android 构建脚本 - 构建所有 ABI 架构

set -e

# 配置
ANDROID_NDK="/path/to/android-ndk"  # 需要用户配置
OPENCV_ANDROID_SDK="/path/to/opencv-android-sdk"  # 需要用户配置
MIN_SDK_VERSION=21

# 检查 NDK
if [ ! -d "$ANDROID_NDK" ]; then
    echo "错误: 未找到 Android NDK: $ANDROID_NDK"
    echo "请下载 Android NDK 并更新 ANDROID_NDK 路径"
    echo "下载地址: https://developer.android.com/ndk/downloads"
    exit 1
fi

# 检查 OpenCV
if [ ! -d "$OPENCV_ANDROID_SDK" ]; then
    echo "错误: 未找到 OpenCV Android SDK: $OPENCV_ANDROID_SDK"
    echo "请下载 OpenCV Android SDK 并更新 OPENCV_ANDROID_SDK 路径"
    echo "下载地址: https://opencv.org/releases/"
    exit 1
fi

# ABI 架构列表
ABIS=("arm64-v8a" "armeabi-v7a" "x86" "x86_64")

echo "开始构建 Android Native 库..."
echo "NDK: $ANDROID_NDK"
echo "OpenCV SDK: $OPENCV_ANDROID_SDK"
echo "目标架构: ${ABIS[@]}"
echo ""

# 构建每个 ABI
for ABI in "${ABIS[@]}"; do
    echo "========================================="
    echo "构建架构: $ABI"
    echo "========================================="

    BUILD_DIR="build-$ABI"

    # 创建构建目录
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # 运行 CMake
    cmake .. \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$ABI" \
        -DANDROID_PLATFORM="android-$MIN_SDK_VERSION" \
        -DANDROID_STL=c++_shared \
        -DOPENCV_ANDROID_SDK="$OPENCV_ANDROID_SDK" \
        -DCMAKE_BUILD_TYPE=Release

    # 编译
    cmake --build . --config Release

    cd ..

    echo "✓ $ABI 构建完成"
    echo ""
done

echo "========================================="
echo "所有架构构建完成！"
echo "========================================="
echo ""
echo "输出文件位置:"
for ABI in "${ABIS[@]}"; do
    LIB_PATH="../../MauiScan/Platforms/Android/libs/$ABI/libopencv_scanner.so"
    if [ -f "$LIB_PATH" ]; then
        echo "  ✓ $LIB_PATH"
    else
        echo "  ✗ $LIB_PATH (未找到)"
    fi
done
echo ""
echo "下一步: 在 Visual Studio 中构建 MAUI 项目"
