# 角点精修模块实现计划

## 背景

根据过拟合测试和GPT分析，得出结论：
- **ML模型精度上限**：3-5px (512×512图) = 25-40px (原图4000×3000)
- **透视变换需求**：< 1px (512图) = < 8px (原图)
- **问题本质**：任务级别错配，ML直接输出无法达到几何级精度

## 解决方案：两阶段架构

### 阶段1：ML粗定位（已完成 ✅）
- 当前Fixed模型：3.51px (512图)
- 功能：识别4个角点大致位置
- 输出：将搜索空间从4000×3000压缩到64×64小窗口

### 阶段2：传统CV精修（待实现 ⚡）
对ML给出的每个角点，在原图上精修到亚像素级

---

## 精修模块技术方案

### 输入
- ML预测的归一化坐标：`[x, y]` ∈ [0, 1]
- 原始图片（高分辨率，如4000×3000）

### 输出
- 精修后的亚像素坐标：`[x_refined, y_refined]`
- 精度目标：< 1px误差

### 算法流程

```
对每个ML预测角点 (x_ml, y_ml)：
  1. 反归一化到原图坐标
  2. 在原图裁剪 patch (64×64 或 96×96)
  3. Canny边缘检测
  4. 直线检测与拟合
  5. 求两条边交点
  6. 返回精修坐标
```

---

## 实现位置选择

### 方案A：C# 后处理（推荐 ⭐）
**位置**：`MauiScan.ML/Services/OnnxInferenceService.cs`

**优势**：
- ✅ 可以直接访问原图（避免传输512图）
- ✅ 使用 OpenCvSharp4（已有依赖）
- ✅ 跨平台（Android/iOS/Windows）
- ✅ 与ML推理在同一层，便于调试

**实现**：
```csharp
public async Task<MLDetectionResult> DetectCornersFromRgbAsync(
    float[] rgbData,
    int originalWidth,
    int originalHeight,
    byte[] originalImageBytes  // ⭐ 新增：原图数据
)
{
    // 1. ML粗定位
    var mlResult = await RunMLInference(rgbData);

    // 2. 传统CV精修（在原图上）
    var refinedCorners = RefineCorners(
        mlResult.Corners,
        originalImageBytes,
        originalWidth,
        originalHeight
    );

    return new MLDetectionResult { Corners = refinedCorners, ... };
}

private QuadrilateralPoints RefineCorners(
    QuadrilateralPoints mlCorners,
    byte[] imageBytes,
    int width,
    int height
)
{
    using var mat = Cv2.ImDecode(imageBytes, ImreadModes.Grayscale);

    // 对4个角点分别精修
    var refined = new QuadrilateralPoints();
    refined.TopLeft = RefineSingleCorner(mat, mlCorners.TopLeftX, mlCorners.TopLeftY);
    refined.TopRight = RefineSingleCorner(mat, mlCorners.TopRightX, mlCorners.TopRightY);
    // ... 其他角点

    return refined;
}

private (float x, float y) RefineSingleCorner(Mat image, float mlX, float mlY)
{
    // 1. 裁剪patch (64×64)
    int patchSize = 64;
    var roi = new Rect(
        (int)mlX - patchSize/2,
        (int)mlY - patchSize/2,
        patchSize,
        patchSize
    );
    var patch = new Mat(image, roi);

    // 2. Canny边缘检测
    var edges = new Mat();
    Cv2.Canny(patch, edges, 50, 150);

    // 3. 霍夫直线检测
    var lines = Cv2.HoughLinesP(
        edges,
        rho: 1,
        theta: Math.PI / 180,
        threshold: 30,
        minLineLength: 20,
        maxLineGap: 5
    );

    // 4. 直线拟合与交点计算
    // TODO: 实现直线聚类、拟合、求交点

    // 5. 返回精修后的坐标
    return (refinedX, refinedY);
}
```

### 方案B：C++ 原生层
**位置**：`Native.OpenCV/`

**优势**：
- ✅ 性能最优
- ✅ 直接使用OpenCV C++ API

**劣势**：
- ❌ 需要跨语言调用
- ❌ 需要重新编译Android/iOS原生库
- ❌ 调试困难

---

## 关键算法细节

### 1. Patch裁剪策略
- **尺寸**：64×64 或 96×96（根据实验调整）
- **边界处理**：如果角点靠近边缘，调整ROI避免越界
- **预处理**：灰度化、可选高斯模糊

### 2. 边缘检测参数
```csharp
Cv2.Canny(
    patch,
    edges,
    threshold1: 50,   // 低阈值
    threshold2: 150,  // 高阈值
    apertureSize: 3
);
```

### 3. 直线检测方法

**选项A：HoughLinesP（概率霍夫变换）**
```csharp
var lines = Cv2.HoughLinesP(
    edges,
    rho: 1,              // 距离分辨率（像素）
    theta: Math.PI/180,  // 角度分辨率（1度）
    threshold: 30,       // 累加器阈值
    minLineLength: 20,   // 最小线段长度
    maxLineGap: 5        // 最大间隙
);
```

**选项B：LSD（Line Segment Detector）** - 更精确
```csharp
var lsd = Cv2.CreateLineSegmentDetector();
var lines = lsd.Detect(patch);
```

### 4. 直线拟合与交点求解

**步骤**：
1. **聚类**：将检测到的线段按角度聚类（水平边 vs 垂直边）
2. **拟合**：对每簇线段进行最小二乘直线拟合
3. **求交点**：计算两条拟合直线的交点

```csharp
// 伪代码
var horizontalLines = lines.Where(l => IsNearHorizontal(l));
var verticalLines = lines.Where(l => IsNearVertical(l));

var hLine = FitLine(horizontalLines);  // y = k1*x + b1
var vLine = FitLine(verticalLines);     // y = k2*x + b2

// 求交点
float x = (b2 - b1) / (k1 - k2);
float y = k1 * x + b1;

// 转换回原图坐标
return (roi.X + x, roi.Y + y);
```

### 5. 失败回退策略
```csharp
if (refinedCorner is null || !IsValid(refinedCorner))
{
    // 精修失败，回退到ML预测结果
    return (mlX, mlY);
}
```

---

## 测试验证

### 单元测试
1. **合成数据测试**：使用人工生成的完美边缘图片
2. **真实数据测试**：使用3张过拟合测试图片

### 评估指标
- **精度提升**：精修前后的像素误差对比
- **成功率**：精修成功的角点比例
- **耗时**：单张图片的精修时间（目标 < 100ms）

### 可视化对比
```python
# 在validation脚本中
def visualize_refinement(image, ml_corners, refined_corners, gt_corners):
    """
    蓝色：ML预测
    红色：精修后
    绿色：真实标注
    """
```

---

## 参数调优空间

需要实验调整的参数：
- [ ] Patch尺寸：64/96/128
- [ ] Canny阈值：(50,150) / (30,100) / (70,200)
- [ ] 霍夫阈值：threshold, minLineLength, maxLineGap
- [ ] 直线聚类角度容差：±5° / ±10°
- [ ] 线段过滤：长度、置信度阈值

---

## 实现优先级

### Phase 1：基础版本（最小可行）
- [ ] 实现`RefineSingleCorner`基础逻辑
- [ ] Canny + HoughLinesP
- [ ] 简单的两条线求交点
- [ ] 在3张测试图上验证

### Phase 2：鲁棒性增强
- [ ] 直线聚类（处理多条候选线）
- [ ] 最小二乘拟合
- [ ] 失败回退策略
- [ ] 边界条件处理

### Phase 3：性能优化
- [ ] 参数调优
- [ ] 耗时分析与优化
- [ ] 可选：尝试LSD替代HoughLines

---

## 预期效果

**精修前（ML直接输出）**：
- 512图误差：3.51px
- 原图误差：~28px
- 透视效果：❌ 明显歪斜

**精修后（ML + 传统CV）**：
- 512图误差：< 1px（目标）
- 原图误差：< 8px（目标）
- 透视效果：✅ 可用

---

## 参考资料

### OpenCV文档
- Canny: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
- HoughLinesP: https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
- LSD: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga6b2ad2353c337c42551b521a73eeae7d

### OpenCvSharp4
- NuGet: https://www.nuget.org/packages/OpenCvSharp4
- 示例: https://github.com/shimat/opencvsharp

---

## 当前状态

- ✅ ML模型训练中（240张图，预计2-3小时或5-10分钟GPU）
- ⏳ 等待训练完成后实现精修模块
- ⏳ Phase 1实现 → Phase 2鲁棒性 → Phase 3优化

---

*创建时间: 2025-12-07*
*更新时间: 待训练完成后*
