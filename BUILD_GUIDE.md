# MauiScan 构建指南

.NET MAUI 文档扫描应用 - 使用原生 C++ OpenCV 实现

## 项目架构

```
MauiScan/
├── MauiScan/                  # MAUI 应用主项目
│   ├── Models/                # 数据模型
│   ├── Services/              # 服务层（相机、剪贴板、图像处理）
│   ├── Views/                 # UI 界面（Code-Behind 模式）
│   └── Platforms/             # 平台特定实现
│       ├── Android/
│       │   ├── Services/      # Android 平台服务
│       │   └── libs/          # Native .so 库（构建后生成）
│       └── iOS/
│           ├── Services/      # iOS 平台服务
│           └── libopencv_scanner.xcframework  # Native 库（构建后生成）
└── Native.OpenCV/             # 原生 C++ OpenCV 库
    ├── src/                   # C++ 源代码
    ├── android/               # Android 构建配置
    └── ios/                   # iOS 构建配置
```

## 技术栈

- **UI 框架**: .NET 10 MAUI
- **UI 模式**: Code-Behind（非 MVVM）
- **图像处理**: 原生 C++ OpenCV 4.x
- **跨平台调用**: P/Invoke
- **目标平台**: Android + iOS

## 快速开始

### 1. 环境准备

#### Windows 开发环境
- Visual Studio 2022 (17.13+)
- .NET 10 SDK
- Android NDK (通过 Android Studio)
- CMake 3.18+
- Ninja (推荐)
- OpenCV Android SDK 4.x

#### macOS 开发环境（iOS 构建）
- macOS 13+
- Xcode 14+
- .NET 10 SDK
- CMake 3.18+
- OpenCV iOS Framework 4.x

### 2. 下载 OpenCV SDK

#### Android
```bash
# 下载 OpenCV Android SDK
https://opencv.org/releases/

# 解压到本地目录，例如:
D:\opencv-android-sdk
```

#### iOS
```bash
# 下载 OpenCV iOS Framework
https://opencv.org/releases/

# 解压后将 opencv2.framework 放到本地
/path/to/opencv2.framework
```

### 3. 构建原生库

#### 构建 Android 库

**Windows:**
```cmd
cd Native.OpenCV\android

REM 编辑 build-android.bat，配置路径:
REM set ANDROID_NDK=C:\Users\%USERNAME%\AppData\Local\Android\Sdk\ndk\26.1.10909125
REM set OPENCV_ANDROID_SDK=D:\opencv-android-sdk

build-android.bat
```

**Linux/Mac:**
```bash
cd Native.OpenCV/android

# 编辑 build-android.sh，配置路径
# ANDROID_NDK="/path/to/android-ndk"
# OPENCV_ANDROID_SDK="/path/to/opencv-android-sdk"

chmod +x build-android.sh
./build-android.sh
```

构建成功后，`.so` 文件会自动复制到 `MauiScan/Platforms/Android/libs/{ABI}/`

#### 构建 iOS 库（仅 macOS）

```bash
cd Native.OpenCV/ios

# 编辑 build-ios.sh，配置路径
# OPENCV_IOS_FRAMEWORK="/path/to/opencv2.framework"

chmod +x build-ios.sh
./build-ios.sh
```

构建成功后，XCFramework 会生成到 `MauiScan/Platforms/iOS/`

### 4. 构建 MAUI 应用

#### Android
```bash
cd MauiScan
dotnet build -t:Run -f net10.0-android
```

或在 Visual Studio 中选择 Android 模拟器/设备运行。

#### iOS
```bash
cd MauiScan
dotnet build -t:Run -f net10.0-ios
```

或在 Visual Studio/Xcode 中选择 iOS 模拟器/设备运行。

## 核心功能

1. **相机拍摄** - 调用系统相机拍摄文档照片
2. **边缘检测** - 自动检测文档四边形边界
3. **透视变换** - 将倾斜的文档矫正为标准矩形
4. **图像增强** - 可选的灰度化和对比度增强
5. **保存结果** - 输出 JPEG 格式图像
6. **自动复制** - 扫描完成后自动复制到系统剪贴板

## OpenCV 算法参数

默认参数配置（在 `opencv_scanner.cpp` 中定义）：

```c
canny_threshold1 = 50.0          // Canny 边缘检测低阈值
canny_threshold2 = 150.0         // Canny 边缘检测高阈值
gaussian_kernel_size = 5         // 高斯模糊核大小（奇数）
min_contour_area_ratio = 0.1     // 最小轮廓面积比例（相对图像面积）
jpeg_quality = 95                // JPEG 压缩质量（0-100）
```

## 项目文件说明

### 关键 C# 文件

| 文件 | 说明 |
|------|------|
| `Models/QuadrilateralPoints.cs` | 四边形顶点数据结构 |
| `Models/ScanResult.cs` | 扫描结果封装 |
| `Services/IImageProcessingService.cs` | 图像处理服务接口 |
| `Services/NativeImageProcessingService.cs` | Native C++ 的 P/Invoke 封装 |
| `Services/ICameraService.cs` | 相机服务接口 |
| `Services/IClipboardService.cs` | 剪贴板服务接口 |
| `Platforms/Android/Services/CameraService.cs` | Android 相机实现（MediaPicker） |
| `Platforms/Android/Services/ClipboardService.cs` | Android 剪贴板实现（FileProvider） |
| `Views/ScanPage.xaml` | 扫描界面 UI |
| `Views/ScanPage.xaml.cs` | 扫描界面 Code-Behind 逻辑 |
| `MauiProgram.cs` | 依赖注入配置 |

### 关键 C++ 文件

| 文件 | 说明 |
|------|------|
| `Native.OpenCV/src/opencv_scanner.h` | C 接口头文件 |
| `Native.OpenCV/src/opencv_scanner.cpp` | OpenCV 算法实现 |
| `Native.OpenCV/android/CMakeLists.txt` | Android CMake 配置 |
| `Native.OpenCV/android/build-android.{sh,bat}` | Android 构建脚本 |
| `Native.OpenCV/ios/build-ios.sh` | iOS 构建脚本 |

## 常见问题

### 1. 构建失败

**Q: CMake 找不到 OpenCV**
```
Could NOT find OpenCV (missing: OpenCV_DIR)
```
**A**: 检查构建脚本中的 `OPENCV_ANDROID_SDK` 或 `OPENCV_IOS_FRAMEWORK` 路径是否正确。

---

**Q: P/Invoke 找不到库**
```
System.DllNotFoundException: Unable to load DLL 'opencv_scanner'
```
**A**:
- Android: 检查 `MauiScan/Platforms/Android/libs/{ABI}/libopencv_scanner.so` 是否存在
- iOS: 检查 `MauiScan/Platforms/iOS/libopencv_scanner.xcframework` 是否存在
- 确保 `.csproj` 中正确配置了 Native 库引用

---

**Q: 运行时崩溃**
```
java.lang.UnsatisfiedLinkError: dlopen failed: library "opencv_scanner" not found
```
**A**: 构建脚本可能未正确复制 `.so` 文件。手动检查输出目录。

### 2. 算法调优

**Q: 检测不到文档边界**

**A**: 尝试调整以下参数（需修改 C++ 代码并重新编译）：
- 降低 `min_contour_area_ratio`（例如从 0.1 改为 0.05）
- 调整 Canny 阈值（例如 threshold1=30, threshold2=100）
- 增加高斯模糊核大小（例如从 5 改为 7）

---

**Q: 检测到错误的边界**

**A**:
- 提高 `min_contour_area_ratio`（例如从 0.1 改为 0.2）
- 确保拍摄时文档占据画面主要区域
- 背景尽量简洁，避免干扰

### 3. 性能优化

**Q: 处理速度慢**

**A**:
- 在拍摄时降低相机分辨率（推荐 1920x1080）
- 使用 Release 构建模式
- 确保 OpenCV 使用了 NEON 优化（ARM）

## 开发建议

### 修改算法参数

如需调整算法参数，编辑 `Native.OpenCV/src/opencv_scanner.cpp` 中的 `scanner_get_default_params()` 函数：

```cpp
ScannerParams scanner_get_default_params(void) {
    ScannerParams params;
    params.canny_threshold1 = 50.0;      // 修改这里
    params.canny_threshold2 = 150.0;     // 修改这里
    params.gaussian_kernel_size = 5;     // 修改这里
    params.min_contour_area_ratio = 0.1; // 修改这里
    params.jpeg_quality = 95;            // 修改这里
    return params;
}
```

修改后需重新构建 Native 库。

### 添加新功能

如需添加新的图像处理功能：

1. 在 `opencv_scanner.h` 中添加新的 C 接口函数声明
2. 在 `opencv_scanner.cpp` 中实现该函数
3. 在 `NativeImageProcessingService.cs` 中添加对应的 P/Invoke 声明
4. 在 `IImageProcessingService.cs` 中添加接口方法（如需要）
5. 重新构建 Native 库和 MAUI 应用

## 许可证

- **OpenCV**: Apache 2.0 License
- **MauiScan**: (根据项目实际情况填写)

## 参考资料

- [OpenCV 官方文档](https://docs.opencv.org/)
- [.NET MAUI 文档](https://learn.microsoft.com/dotnet/maui/)
- [P/Invoke 指南](https://learn.microsoft.com/dotnet/standard/native-interop/pinvoke)
- [Android NDK 文档](https://developer.android.com/ndk/guides)

## 后续改进方向

- [ ] 添加手动调整四边形顶点功能
- [ ] 支持多页文档批量扫描
- [ ] 添加 PDF 导出功能
- [ ] 优化低光环境下的检测效果
- [ ] 添加自动旋转功能（基于文字方向）
- [ ] 支持彩色扫描模式选择

---

**构建完成后，在 Visual Studio 中运行项目即可开始使用！**
