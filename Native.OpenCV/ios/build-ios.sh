#!/bin/bash
# iOS 构建脚本

set -e

# 配置
OPENCV_IOS_FRAMEWORK="/path/to/opencv2.framework"
SOURCE_DIR="../src"
OUTPUT_DIR="../../MauiScan/Platforms/iOS"

# 检查 OpenCV framework
if [ ! -d "$OPENCV_IOS_FRAMEWORK" ]; then
    echo "错误: 未找到 OpenCV framework: $OPENCV_IOS_FRAMEWORK"
    echo "请下载 OpenCV iOS framework 并更新路径"
    exit 1
fi

# 编译为通用库（支持真机 + 模拟器）
echo "构建 iOS 库..."

# 真机架构 (arm64)
xcodebuild -target opencv_scanner \
    -configuration Release \
    -arch arm64 \
    -sdk iphoneos \
    ONLY_ACTIVE_ARCH=NO \
    BUILD_DIR="./build" \
    OBJROOT="./build/obj" \
    SYMROOT="./build/sym" \
    CONFIGURATION_BUILD_DIR="./build/Release-iphoneos" \
    clean build

# 模拟器架构 (x86_64, arm64)
xcodebuild -target opencv_scanner \
    -configuration Release \
    -arch x86_64 -arch arm64 \
    -sdk iphonesimulator \
    ONLY_ACTIVE_ARCH=NO \
    BUILD_DIR="./build" \
    OBJROOT="./build/obj" \
    SYMROOT="./build/sym" \
    CONFIGURATION_BUILD_DIR="./build/Release-iphonesimulator" \
    clean build

# 合并为 XCFramework
echo "创建 XCFramework..."
xcodebuild -create-xcframework \
    -library "./build/Release-iphoneos/libopencv_scanner.a" \
    -library "./build/Release-iphonesimulator/libopencv_scanner.a" \
    -output "$OUTPUT_DIR/libopencv_scanner.xcframework"

echo "✓ iOS 库构建完成: $OUTPUT_DIR/libopencv_scanner.xcframework"
