# Android Native Library Build Guide

## 环境要求

- **Android NDK r27c**（或更高版本）
- **OpenCV Android SDK**
- **Visual Studio 2022**（包含 CMake 和 Ninja）

## 构建步骤

⚠️ **重要：必须使用 Developer Command Prompt for VS 2022 来执行构建**

1. 打开 **Developer Command Prompt for VS 2022**
   - 开始菜单 → Visual Studio 2022 → Developer Command Prompt for VS 2022

2. 进入构建目录：
   ```cmd
   cd D:\Programing\C#\MauiScan\Native.OpenCV\android
   ```

3. 运行构建脚本：
   ```cmd
   build-android.bat
   ```

## 输出文件

编译成功后，库文件会自动复制到：
```
D:\Programing\C#\MauiScan\MauiScan\Platforms\Android\libs\arm64-v8a\libopencv_scanner.so
D:\Programing\C#\MauiScan\MauiScan\Platforms\Android\libs\armeabi-v7a\libopencv_scanner.so
D:\Programing\C#\MauiScan\MauiScan\Platforms\Android\libs\x86\libopencv_scanner.so
D:\Programing\C#\MauiScan\MauiScan\Platforms\Android\libs\x86_64\libopencv_scanner.so
```

## 故障排除

### 错误：'cmake' 不是内部或外部命令

**原因：** 未使用 Developer Command Prompt for VS 2022

**解决：** 必须在 Developer Command Prompt for VS 2022 中运行构建脚本，普通 PowerShell 或 CMD 不包含 cmake 环境变量。

### 错误：未找到 Android NDK

修改 `build-android.bat` 中的 `ANDROID_NDK` 路径：
```batch
set "ANDROID_NDK=C:\Program Files (x86)\Android\AndroidNDK\android-ndk-r27c"
```

### 错误：未找到 OpenCV SDK

修改 `build-android.bat` 中的 `OPENCV_ANDROID_SDK` 路径：
```batch
set "OPENCV_ANDROID_SDK=D:\Programing\C#\MauiScan\OpenCVSDK\OpenCV-android-sdk"
```
