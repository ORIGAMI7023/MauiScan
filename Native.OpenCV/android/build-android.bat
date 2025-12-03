@echo off
REM Android 构建脚本 - Windows 版本

setlocal enabledelayedexpansion

REM 配置（需要用户修改）
set ANDROID_NDK=C:\Users\%USERNAME%\AppData\Local\Android\Sdk\ndk\26.1.10909125
set OPENCV_ANDROID_SDK=D:\OpenCV-android-sdk
set MIN_SDK_VERSION=21

REM 检查 NDK
if not exist "%ANDROID_NDK%" (
    echo 错误: 未找到 Android NDK: %ANDROID_NDK%
    echo 请下载 Android NDK 并更新 ANDROID_NDK 路径
    echo 下载地址: https://developer.android.com/ndk/downloads
    exit /b 1
)

REM 检查 OpenCV
if not exist "%OPENCV_ANDROID_SDK%" (
    echo 错误: 未找到 OpenCV Android SDK: %OPENCV_ANDROID_SDK%
    echo 请下载 OpenCV Android SDK 并更新 OPENCV_ANDROID_SDK 路径
    echo 下载地址: https://opencv.org/releases/
    exit /b 1
)

REM ABI 架构列表
set ABIS=arm64-v8a armeabi-v7a x86 x86_64

echo 开始构建 Android Native 库...
echo NDK: %ANDROID_NDK%
echo OpenCV SDK: %OPENCV_ANDROID_SDK%
echo 目标架构: %ABIS%
echo.

REM 构建每个 ABI
for %%A in (%ABIS%) do (
    echo =========================================
    echo 构建架构: %%A
    echo =========================================

    set BUILD_DIR=build-%%A

    REM 创建构建目录
    if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
    cd "!BUILD_DIR!"

    REM 运行 CMake
    cmake .. ^
        -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK%/build/cmake/android.toolchain.cmake" ^
        -DANDROID_ABI=%%A ^
        -DANDROID_PLATFORM=android-%MIN_SDK_VERSION% ^
        -DANDROID_STL=c++_shared ^
        -DOPENCV_ANDROID_SDK="%OPENCV_ANDROID_SDK%" ^
        -DCMAKE_BUILD_TYPE=Release ^
        -G "Ninja"

    if errorlevel 1 (
        echo 错误: CMake 配置失败
        cd ..
        exit /b 1
    )

    REM 编译
    cmake --build . --config Release

    if errorlevel 1 (
        echo 错误: 编译失败
        cd ..
        exit /b 1
    )

    cd ..

    echo √ %%A 构建完成
    echo.
)

echo =========================================
echo 所有架构构建完成！
echo =========================================
echo.
echo 输出文件位置:
for %%A in (%ABIS%) do (
    set LIB_PATH=..\..\MauiScan\Platforms\Android\libs\%%A\libopencv_scanner.so
    if exist "!LIB_PATH!" (
        echo   √ !LIB_PATH!
    ) else (
        echo   × !LIB_PATH! (未找到^)
    )
)
echo.
echo 下一步: 在 Visual Studio 中构建 MAUI 项目

endlocal
