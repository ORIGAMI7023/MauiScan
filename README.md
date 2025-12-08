# MauiScan - 文档扫描应用

.NET 10 MAUI 跨平台文档扫描应用，使用深度学习 + OpenCV 实现高精度文档边界检测。

---

## 🎯 核心功能

- 📸 **拍照扫描** - 相机拍摄文档
- 🤖 **智能检测** - 深度学习模型 + OpenCV 双重检测文档边界
- ✂️ **自动裁剪** - 透视变换矫正文档
- 🔄 **实时同步** - SignalR 跨设备同步

---

## 📊 技术方案

### 文档边界检测（两阶段）

#### 阶段1：ML粗定位（ONNX模型）
- **模型**: 自训练角点检测模型（240张标注数据）
- **输入**: 1024×768 RGB图像
- **输出**: 4个角点坐标（TL, TR, BR, BL）
- **准确率**: 约85%

#### 阶段2：CV精修（OpenCV）
- **最优方案**: Baseline + 多尺度Canny (79.5%成功率)
  - 预处理: Gaussian5x5 + Contrast1.15
  - 边缘检测: 5尺度Canny融合 (kernel: 3,5,7,9,11)
- **改进**: 相比原始CV提升20.9%

详见: [CV预处理优化计划.md](CV预处理优化计划.md)

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────┐
│   .NET 10 MAUI (跨平台UI)           │
├─────────────────────────────────────┤
│  ML层: ONNX Runtime (角点检测)      │
│  CV层: Native C++ OpenCV (边缘精修) │
├─────────────────────────────────────┤
│  平台层: Android / iOS              │
└─────────────────────────────────────┘
```

### 核心组件
- **UI**: .NET 10 MAUI (Code-Behind模式)
- **ML**: ONNX Runtime + 自训练模型
- **CV**: Native C++ OpenCV + P/Invoke
- **同步**: ASP.NET Core + SignalR

---

## 📁 项目结构

```
MauiScan/
├── MauiScan/                      # MAUI主应用
│   ├── Resources/Raw/
│   │   └── ppt_corner_detector.onnx  # ML模型
│   ├── Services/
│   │   ├── OnnxInferenceService.cs   # ONNX推理
│   │   └── NativeImageProcessingService.cs  # OpenCV封装
│   └── Views/
│       ├── ScanPage.xaml          # 扫描页面
│       └── MLTestPage.xaml        # ML测试页面
│
├── MauiScan.ML/                   # ML模块
│   ├── Models/                    # 数据模型
│   └── Platforms/Android/         # Android特定实现
│       └── OnnxInferenceService.Android.cs
│
├── MauiScan.ML.Training/          # ML训练
│   ├── Program.cs                 # 训练入口
│   └── ppt_corner_detector.onnx   # 训练输出
│
├── Native.OpenCV/                 # Native C++ OpenCV
│   ├── src/
│   │   ├── opencv_scanner.h       # C接口
│   │   └── opencv_scanner.cpp     # OpenCV实现
│   ├── android/                   # Android构建
│   │   ├── CMakeLists.txt
│   │   └── build-android.bat
│   └── ios/                       # iOS构建
│       └── build-ios.sh
│
├── AnnotationTool/                # 标注工具
│   └── data/                      # 240张标注数据
│
├── scripts/                       # 测试脚本
│   ├── test_preprocessing_methods.py      # 30方法测试
│   ├── test_combination_methods.py        # 91组合测试
│   └── analyze_preprocessing_results.py   # 结果分析
│
└── docs/
    ├── CV预处理优化计划.md        # CV优化完整方案
    ├── CV精修优化方案.md          # CV精修分析
    └── ML_IMPLEMENTATION.md       # ML实现文档
```

---

## 🚀 快速开始

### 环境要求

**通用**:
- .NET 10 SDK
- Visual Studio 2022 (17.12+) 或 Rider

**Android**:
- Android NDK r27c
- CMake + Ninja

**iOS**:
- macOS + Xcode
- OpenCV iOS Framework

### 构建步骤

#### 1. Android Native库
```cmd
# 必须在 Developer Command Prompt for VS 2022 中运行
cd Native.OpenCV\android
build-android.bat
```

输出: `MauiScan/Platforms/Android/libs/{abi}/libopencv_scanner.so`

#### 2. iOS Native库 (在Mac上)
```bash
cd Native.OpenCV/ios
./build-ios.sh
```

#### 3. 运行应用
```bash
# Android
dotnet build -f net10.0-android

# iOS (在Mac上)
dotnet build -f net10.0-ios
```

详见: [BUILD_GUIDE.md](BUILD_GUIDE.md)

---

## 📈 性能指标

### ML模型
- 推理速度: ~200ms (CPU)
- 模型大小: 2.3MB
- 准确率: 约85%

### CV精修
- 成功率: 79.5% (优化后)
- 提升: +20.9% (相比原始CV)
- 处理速度: ~50ms

### 整体检测
- 端到端延迟: ~250ms
- 综合成功率: ~80%

---

## 🔬 研发历程

### 已完成
- ✅ ML模型训练 (240张标注数据)
- ✅ ONNX部署到.NET MAUI
- ✅ CV预处理优化 (30方法 + 91组合测试)
- ✅ Android Native库构建
- ✅ 跨平台P/Invoke封装

### 进行中
- 🔄 多尺度Canny部署到C++
- 🔄 iOS构建和测试

### 待优化
- ⏳ 多轮廓检测 (方案3，如需进一步提升)
- ⏳ 模型量化 (减小体积)
- ⏳ GPU加速推理

---

## 📚 文档

- [CV预处理优化计划](CV预处理优化计划.md) - CV优化完整方案和测试结果
- [CV精修优化方案](CV精修优化方案.md) - CV精修问题分析
- [ML实现文档](ML_IMPLEMENTATION.md) - ML训练和部署详情
- [构建指南](BUILD_GUIDE.md) - 跨平台构建说明

---

## 🛠️ 开发工具

### Python脚本
- `test_preprocessing_methods.py` - 30种预处理方法测试
- `test_combination_methods.py` - 91种组合方案测试
- `analyze_preprocessing_results.py` - 结果分析和可视化
- `cv_debug_visualizer.py` - CV调试可视化工具

### 使用示例
```bash
# 测试预处理方法
python test_preprocessing_methods.py

# 测试组合方案
python test_combination_methods.py

# 分析结果
python analyze_preprocessing_results.py
```

---

## 📝 版本历史

### v0.3.0 (2025-12-08)
- ✨ CV预处理优化：找到最佳组合 (79.5%成功率)
- ✨ 完成91种组合测试
- 📊 总测试量: 29,040个 (240图 × 121方法)

### v0.2.0 (2025-12-07)
- ✨ ML模型训练完成
- ✨ ONNX集成到MAUI
- 🐛 修复跨平台编译问题

### v0.1.0 (2025-12-03)
- 🎉 初始版本
- ✅ Android Native库构建
- ✅ 基础OpenCV算法

---

## 🤝 贡献

贡献者: ORIGAMI7023

---

## 📄 许可证

本项目仅供学习和研究使用。

---

**生成工具**: [Claude Code](https://claude.com/claude-code)
