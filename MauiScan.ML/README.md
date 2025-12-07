# MauiScan.ML - ML 推理服务

PPT 四角点智能检测的 ML 推理模块。

## 项目概述

本项目提供基于 ONNX Runtime 的 ML 推理服务，用于智能检测教室投屏 PPT 的四个角点。

## 架构设计

```
┌─────────────────────────────────────┐
│     MauiScan MAUI 应用              │
│  (IImageProcessingService)          │
└──────────────┬──────────────────────┘
               │
      ┌────────┴────────┐
      ↓                 ↓
┌──────────────┐  ┌─────────────────┐
│  传统 OpenCV │  │  ML 推理引擎    │
│    算法      │  │ (OnnxInference) │
└──────────────┘  └─────────────────┘
      ↓                 ↓
┌─────────────────────────────────────┐
│      混合调度服务 (TODO)            │
│  - 置信度评估                       │
│  - 算法选择/融合                    │
└─────────────────────────────────────┘
```

## 功能特性

- ✅ ONNX Runtime 推理引擎
- ✅ ML 检测结果模型定义
- ✅ 置信度评估（高/中/低质量）
- ⏳ 混合调度服务（ML + 传统算法）
- ⏳ Android NNAPI 加速
- ⏳ 模型版本管理

## 使用方法

### 1. 添加项目引用

在 `MauiScan.csproj` 中添加：

```xml
<ItemGroup>
  <ProjectReference Include="..\MauiScan.ML\MauiScan.ML.csproj" />
</ItemGroup>
```

### 2. 放置 ONNX 模型文件

将训练好的模型放置在：
```
MauiScan/Resources/Raw/ppt_corner_detector.onnx
```

### 3. 注册服务

在 `MauiProgram.cs` 中：

```csharp
using MauiScan.ML.Services;

// 注册 ML 推理服务
var modelPath = Path.Combine(FileSystem.AppDataDirectory, "ppt_corner_detector.onnx");
builder.Services.AddSingleton<IMLInferenceService>(
    new OnnxInferenceService(modelPath)
);
```

### 4. 使用 ML 检测

```csharp
using MauiScan.ML.Services;

public class MyService
{
    private readonly IMLInferenceService _mlService;

    public MyService(IMLInferenceService mlService)
    {
        _mlService = mlService;
    }

    public async Task DetectPPTCorners(byte[] imageBytes)
    {
        // 检查模型是否可用
        if (!await _mlService.IsModelAvailableAsync())
        {
            Console.WriteLine("模型文件不存在");
            return;
        }

        // 运行检测
        var result = await _mlService.DetectCornersAsync(imageBytes);

        // 根据置信度判断结果质量
        if (result.IsHighQuality)
        {
            Console.WriteLine($"高质量检测 (置信度: {result.Confidence:F2})");
            // 直接使用 ML 结果
            var corners = result.Corners;
        }
        else if (result.IsMediumQuality)
        {
            Console.WriteLine($"中等质量检测 (置信度: {result.Confidence:F2})");
            // 建议与传统算法融合
        }
        else
        {
            Console.WriteLine($"低质量检测 (置信度: {result.Confidence:F2})");
            // 降级使用传统算法
        }
    }
}
```

## API 文档

### IMLInferenceService

ML 推理服务接口。

#### DetectCornersAsync

```csharp
Task<MLDetectionResult> DetectCornersAsync(byte[] imageBytes)
```

检测图片中的 PPT 四个角点。

**参数**:
- `imageBytes`: 原始图片字节数据（JPG/PNG）

**返回**: `MLDetectionResult` 包含：
- `Corners`: 四个角点坐标（左上、右上、右下、左下）
- `Confidence`: 整体置信度 (0-1)
- `CornerConfidences`: 每个角点的置信度
- `IsHighQuality`: 是否为高质量结果 (≥ 0.85)
- `IsMediumQuality`: 是否为中等质量结果 (0.60-0.85)
- `IsLowQuality`: 是否为低质量结果 (< 0.60)

#### IsModelAvailableAsync

```csharp
Task<bool> IsModelAvailableAsync()
```

检查 ONNX 模型文件是否存在。

#### GetModelInfoAsync

```csharp
Task<ModelInfo> GetModelInfoAsync()
```

获取模型元数据（版本、训练信息等）。

#### WarmupAsync

```csharp
Task WarmupAsync()
```

预热模型（首次加载较慢，预热后加快后续推理）。

### MLDetectionResult

ML 检测结果。

**属性**:
- `QuadrilateralPoints Corners`: 四个角点坐标
- `float Confidence`: 整体置信度 (0-1)
- `float[] CornerConfidences`: 每个角点的置信度 [4]
- `bool IsHighQuality`: 置信度 ≥ 0.85
- `bool IsMediumQuality`: 0.60 ≤ 置信度 < 0.85
- `bool IsLowQuality`: 置信度 < 0.60

### QuadrilateralPoints

四边形角点坐标（ML.Models 版本）。

**属性**:
- `float TopLeftX/Y`: 左上角坐标
- `float TopRightX/Y`: 右上角坐标
- `float BottomRightX/Y`: 右下角坐标
- `float BottomLeftX/Y`: 左下角坐标

**方法**:
- `ToArray()`: 转换为数组 `[(x,y), ...]`
- `FromArray(points)`: 从数组创建

## 配置选项

### ModelConfig

模型配置参数。

```csharp
var config = new ModelConfig
{
    InputWidth = 512,                   // 模型输入宽度
    InputHeight = 512,                  // 模型输入高度
    HighQualityThreshold = 0.85f,       // 高质量阈值
    MediumQualityThreshold = 0.60f,     // 中等质量阈值
    EnableGpuAcceleration = true,       // GPU 加速
    InferenceTimeoutMs = 5000           // 推理超时(ms)
};

var service = new OnnxInferenceService(modelPath, config);
```

## 性能指标

### 目标性能

- **推理延迟**: < 300ms (中端 Android 设备)
- **模型大小**: < 10 MB
- **内存占用**: < 50 MB
- **准确率**: > 95% (误差 < 50px)

### 实际测试

（待补充实际测试数据）

## 已知限制

1. **模型尚未训练**: 当前仅提供推理框架，需要先训练模型
2. **GPU 加速未启用**: Android NNAPI / iOS CoreML 支持需要额外配置
3. **混合调度未实现**: 暂时只能单独使用 ML 或传统算法

## 下一步开发

### 短期（1-2 周）

- [ ] 收集训练数据（300-500 张）
- [ ] 训练初版模型
- [ ] 导出 ONNX 并集成测试
- [ ] 实现混合调度服务

### 中期（1 个月）

- [ ] 启用 Android NNAPI 加速
- [ ] 模型量化（INT8）减小体积
- [ ] 真机性能测试和优化
- [ ] 边缘案例处理（高光、低光、边界外角点）

### 长期（3 个月）

- [ ] 在线模型更新机制
- [ ] A/B 测试框架
- [ ] 用户反馈数据收集
- [ ] 持续模型迭代优化

## 故障排查

### 问题：模型加载失败

**症状**: `FileNotFoundException: 模型文件不存在`

**解决**:
1. 检查模型文件是否在 `Resources/Raw/` 目录
2. 确保文件在首次运行时复制到应用目录
3. 检查文件权限

### 问题：推理速度慢

**症状**: 推理耗时 > 1 秒

**解决**:
1. 减小输入尺寸（512 → 416 或 384）
2. 启用 GPU 加速
3. 检查是否在主线程运行（应在后台线程）

### 问题：置信度总是很低

**症状**: Confidence < 0.60

**可能原因**:
1. 模型未正确训练
2. 输入图片预处理有误
3. 场景与训练数据差异太大

## 贡献指南

本模块是 MauiScan 项目的核心组件，欢迎贡献：

1. 性能优化
2. 模型改进
3. Bug 修复
4. 文档完善

## 许可协议

本项目是 MauiScan 的一部分。

## 相关文档

- [训练环境说明](../MauiScan.ML.Training/README.md)
- [数据标注工具](../AnnotationTool/README.md)
- [项目总体架构](../.claude/CLAUDE.md)
