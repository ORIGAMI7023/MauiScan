#!/bin/bash
# iOS Native 库构建脚本 (使用 CMake)

set -e

# 配置
OPENCV_IOS_SDK="../../OpenCVSDK/ios"
OUTPUT_DIR="../../MauiScan/Platforms/iOS"

echo "================================"
echo "构建 iOS Native OpenCV 扫描库"
echo "================================"

# 检查 OpenCV SDK
if [ ! -d "$OPENCV_IOS_SDK" ]; then
    echo "错误: 未找到 OpenCV iOS SDK: $OPENCV_IOS_SDK"
    exit 1
fi

echo "OpenCV SDK: $OPENCV_IOS_SDK"

# 清理旧的构建
echo "清理旧构建..."
rm -rf build-iphoneos build-iphonesimulator

# 1. 构建真机版本 (arm64)
echo ""
echo "=> 构建真机版本 (arm64)..."
cmake -S . -B build-iphoneos \
    -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DOPENCV_IOS_SDK="$OPENCV_IOS_SDK"

cmake --build build-iphoneos --config Release

# 2. 构建模拟器版本 (x86_64 + arm64)
echo ""
echo "=> 构建模拟器版本 (x86_64 + arm64)..."
cmake -S . -B build-iphonesimulator \
    -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
    -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
    -DCMAKE_OSX_SYSROOT=iphonesimulator \
    -DOPENCV_IOS_SDK="$OPENCV_IOS_SDK"

cmake --build build-iphonesimulator --config Release

# 3. 创建 XCFramework
echo ""
echo "=> 创建 XCFramework..."
rm -rf "$OUTPUT_DIR/libopencv_scanner.xcframework"

xcodebuild -create-xcframework \
    -library "./build-iphoneos/Release-iphoneos/libopencv_scanner.a" \
    -library "./build-iphonesimulator/Release-iphonesimulator/libopencv_scanner.a" \
    -output "$OUTPUT_DIR/libopencv_scanner.xcframework"

echo ""
echo "================================"
echo "✓ 构建完成!"
echo "输出: $OUTPUT_DIR/libopencv_scanner.xcframework"
echo "================================"

# 验证符号
echo ""
echo "=> 验证导出符号..."
nm -g "$OUTPUT_DIR/libopencv_scanner.xcframework/ios-arm64/libopencv_scanner.a" | grep " T " | grep scanner
