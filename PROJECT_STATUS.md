# MauiScan 项目状态

## 项目概述

.NET 10 MAUI 文档扫描应用，使用 Native C++ OpenCV 进行图像处理。

## 技术架构

- **UI 框架**: .NET 10 MAUI (Code-Behind 模式，非 MVVM)
- **图像处理**: Native C++ OpenCV + P/Invoke
- **不使用**: OpenCvSharp、OCR

## 已完成

### Android 构建 ✅
- Native C++ 库已编译完成（4 个架构）
- 输出位置: `Platforms/Android/libs/{abi}/`
  - `libopencv_scanner.so` - 我们的扫描库
  - `libopencv_java4.so` - OpenCV 共享库
  - `libc++_shared.so` - C++ 运行时库
- `.csproj` 已配置 AndroidNativeLibrary 引用

### 核心代码 ✅
- `Native.OpenCV/src/opencv_scanner.h` - C 接口定义
- `Native.OpenCV/src/opencv_scanner.cpp` - OpenCV 算法实现
- `MauiScan/Services/NativeImageProcessingService.cs` - P/Invoke 封装
- `MauiScan/Services/IImageProcessingService.cs` - 服务接口
- `MauiScan/Models/` - 数据模型

### OpenCV 算法
- 高斯模糊降噪
- Canny 边缘检测
- 轮廓检测与四边形识别
- 透视变换（文档矫正）
- 自适应阈值增强

## 待完成

### iOS 构建 ❌
需要在 Mac 上执行：

1. **OpenCV iOS SDK 已包含在项目中**
   - 位置: `OpenCVSDK/ios/`
   - 包含 Headers、Modules、opencv2 (582MB)

2. **构建步骤**
   ```bash
   cd Native.OpenCV/ios
   chmod +x build-ios.sh
   ./build-ios.sh
   ```

3. **预期输出**
   - `Platforms/iOS/libopencv_scanner.xcframework`

4. **构建脚本位置**
   - `Native.OpenCV/ios/build-ios.sh`（需检查并更新 OpenCV 路径）

### Windows 构建 ❌
- 未实现
- 需要编译 Windows DLL

### 功能测试 ❌
- Android/iOS 应用运行测试
- 相机拍照功能
- 边缘检测功能
- 文档矫正功能

## 目录结构

```
MauiScan/
├── MauiScan/                    # MAUI 主项目
│   ├── Models/                  # 数据模型
│   ├── Services/                # 服务层 (P/Invoke)
│   ├── Views/                   # 页面
│   └── Platforms/
│       └── Android/
│           └── libs/            # ✅ Android Native 库
│               ├── arm64-v8a/
│               ├── armeabi-v7a/
│               ├── x86/
│               └── x86_64/
├── Native.OpenCV/               # Native C++ 代码
│   ├── src/
│   │   ├── opencv_scanner.h     # C 接口
│   │   └── opencv_scanner.cpp   # OpenCV 实现
│   ├── android/
│   │   ├── CMakeLists.txt
│   │   └── build-android.bat    # Windows 构建脚本
│   └── ios/
│       └── build-ios.sh         # Mac 构建脚本
└── OpenCVSDK/
    └── ios/                     # ✅ iOS OpenCV Framework
        ├── Headers/
        ├── Modules/
        └── opencv2
```

## 构建环境

### Android (Windows) ✅
- Visual Studio 2026 Insiders
- Android NDK r27c: `C:\Program Files (x86)\Android\AndroidNDK\android-ndk-r27c`
- OpenCV Android SDK: `D:\OpenCV-android-sdk`（项目外部）
- CMake + Ninja（VS 自带）

### iOS (Mac) 需要
- Xcode
- OpenCV iOS Framework（已包含在 `OpenCVSDK/ios/`）

## Git 仓库

- GitHub: https://github.com/ORIGAMI7023/MauiScan
- 分支: main

## 下一步操作

在 Mac 上：
1. 克隆或复制项目到 Mac
2. 修改 `Native.OpenCV/ios/build-ios.sh` 中的 OpenCV 路径指向 `OpenCVSDK/ios`
3. 运行 iOS 构建脚本
4. 在 Xcode 或 VS for Mac 中测试运行
