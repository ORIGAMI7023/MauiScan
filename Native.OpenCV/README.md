# Native.OpenCV - 原生 C++ OpenCV 文档扫描库

该库为 MauiScan 应用提供原生 C++ OpenCV 图像处理功能，支持 Android 和 iOS 平台。

## 项目结构

```
Native.OpenCV/
├── src/
│   ├── opencv_scanner.h       # C 接口头文件
│   └── opencv_scanner.cpp     # OpenCV 算法实现
├── android/
│   ├── CMakeLists.txt         # Android CMake 配置
│   ├── build-android.sh       # Linux/Mac 构建脚本
│   └── build-android.bat      # Windows 构建脚本
└── ios/
    └── build-ios.sh           # iOS 构建脚本
```

## 依赖项

### 通用依赖
- CMake 3.18.1 或更高版本
- C++ 17 编译器

### Android 依赖
- Android NDK r21 或更高版本
- OpenCV Android SDK 4.x
- Ninja (可选，推荐用于 Windows)

### iOS 依赖
- Xcode 14 或更高版本
- OpenCV iOS Framework 4.x
- macOS 开发环境

## 下载 OpenCV SDK

### Android
1. 访问 https://opencv.org/releases/
2. 下载 **OpenCV Android** (例如: opencv-4.10.0-android-sdk.zip)
3. 解压到本地目录，例如 `D:\opencv-android-sdk`

### iOS
1. 访问 https://opencv.org/releases/
2. 下载 **OpenCV iOS Framework** (例如: opencv-4.10.0-ios-framework.zip)
3. 解压后将 `opencv2.framework` 放到本地目录

## 构建步骤

### Android 构建

#### Windows 环境

1. **安装 Android NDK**
   - 通过 Android Studio SDK Manager 安装
   - 或从 https://developer.android.com/ndk/downloads 下载

2. **安装 Ninja** (推荐)
   ```cmd
   choco install ninja
   ```
   或从 https://github.com/ninja-build/ninja/releases 下载

3. **配置路径**
   编辑 `android/build-android.bat`，修改以下路径：
   ```batch
   set ANDROID_NDK=C:\Users\%USERNAME%\AppData\Local\Android\Sdk\ndk\26.1.10909125
   set OPENCV_ANDROID_SDK=D:\opencv-android-sdk
   ```

4. **运行构建**
   ```cmd
   cd Native.OpenCV\android
   build-android.bat
   ```

#### Linux/Mac 环境

1. **安装 Android NDK**
   ```bash
   # 方式1: 通过 Android Studio SDK Manager
   # 方式2: 手动下载
   wget https://dl.google.com/android/repository/android-ndk-r26-linux.zip
   unzip android-ndk-r26-linux.zip
   ```

2. **配置路径**
   编辑 `android/build-android.sh`：
   ```bash
   ANDROID_NDK="/path/to/android-ndk"
   OPENCV_ANDROID_SDK="/path/to/opencv-android-sdk"
   ```

3. **运行构建**
   ```bash
   cd Native.OpenCV/android
   chmod +x build-android.sh
   ./build-android.sh
   ```

### iOS 构建 (仅 macOS)

1. **配置路径**
   编辑 `ios/build-ios.sh`：
   ```bash
   OPENCV_IOS_FRAMEWORK="/path/to/opencv2.framework"
   ```

2. **运行构建**
   ```bash
   cd Native.OpenCV/ios
   chmod +x build-ios.sh
   ./build-ios.sh
   ```

## 输出文件

### Android
构建完成后，`.so` 库会自动复制到：
```
MauiScan/Platforms/Android/libs/
├── arm64-v8a/libopencv_scanner.so
├── armeabi-v7a/libopencv_scanner.so
├── x86/libopencv_scanner.so
└── x86_64/libopencv_scanner.so
```

### iOS
构建完成后，XCFramework 会生成到：
```
MauiScan/Platforms/iOS/libopencv_scanner.xcframework
```

## 算法说明

核心文档扫描算法流程：

1. **灰度化** - 转换为单通道灰度图
2. **高斯模糊** - 降噪 (kernel size = 5)
3. **Canny 边缘检测** - 提取边缘 (threshold1 = 50, threshold2 = 150)
4. **轮廓查找** - 查找所有外部轮廓
5. **四边形筛选** - 查找面积最大的四边形 (最小面积比例 = 0.1)
6. **透视变换** - 将四边形变换为标准矩形
7. **可选增强** - 自适应阈值处理 (blockSize = 11, C = 2)
8. **JPEG 编码** - 输出压缩图像 (quality = 95)

## C 接口说明

### 主要函数

```c
// 获取默认参数
ScannerParams scanner_get_default_params(void);

// 处理文档扫描（完整流程）
int32_t scanner_process_scan(
    const uint8_t* input_data,
    int32_t input_size,
    int32_t apply_enhancement,
    const ScannerParams* params,
    ScanResult* result
);

// 仅检测文档边界
int32_t scanner_detect_bounds(
    const uint8_t* input_data,
    int32_t input_size,
    const ScannerParams* params,
    QuadPoints* quad
);

// 释放结果内存
void scanner_free_result(ScanResult* result);

// 获取版本信息
const char* scanner_get_version(void);
```

### 数据结构

```c
typedef struct {
    double canny_threshold1;      // Canny 阈值1
    double canny_threshold2;      // Canny 阈值2
    int32_t gaussian_kernel_size; // 高斯核大小
    double min_contour_area_ratio;// 最小轮廓面积比例
    int32_t jpeg_quality;         // JPEG 质量
} ScannerParams;

typedef struct {
    uint8_t* image_data;      // 图像数据
    int32_t image_size;       // 数据大小
    int32_t width;            // 宽度
    int32_t height;           // 高度
    QuadPoints quad;          // 四边形顶点
    int32_t success;          // 成功标志
    char error_message[256];  // 错误信息
} ScanResult;
```

## 故障排查

### Android 构建失败

**问题**: CMake 找不到 OpenCV
```
Could NOT find OpenCV (missing: OpenCV_DIR)
```
**解决**: 检查 `CMakeLists.txt` 中的 `OPENCV_ANDROID_SDK` 路径是否正确

**问题**: NDK 版本不兼容
```
error: C++ 17 is required
```
**解决**: 更新 NDK 到 r21 或更高版本

### iOS 构建失败

**问题**: 找不到 opencv2.framework
```
fatal error: 'opencv2/opencv.hpp' file not found
```
**解决**: 检查 `build-ios.sh` 中的 `OPENCV_IOS_FRAMEWORK` 路径

**问题**: 架构不匹配
```
ld: building for iOS Simulator, but linking in object file built for iOS
```
**解决**: 使用 XCFramework，它同时支持真机和模拟器

### 运行时错误

**问题**: P/Invoke 找不到库
```
System.DllNotFoundException: Unable to load DLL 'opencv_scanner'
```
**解决**:
- Android: 检查 `.so` 文件是否在正确的 `libs/{ABI}/` 目录
- iOS: 检查 XCFramework 是否正确引用

**问题**: 内存泄漏
```
Memory usage keeps increasing
```
**解决**: 确保 C# 代码中调用了 `scanner_free_result()`

## 性能优化建议

1. **图像尺寸** - 建议将输入图像缩放到 1920x1080 以下
2. **参数调优** - 根据实际场景调整 Canny 阈值
3. **多线程** - OpenCV 内部会自动使用多线程优化
4. **内存管理** - 及时释放 `ScanResult` 以避免内存泄漏

## 许可证

本项目使用 OpenCV，遵循 Apache 2.0 许可证。
详见: https://opencv.org/license/
