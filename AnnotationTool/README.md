# AnnotationTool — 幕布边框标注与深度学习训练工具集

本文件夹包含两个部分：
1. **标注工具**（C# WinForms）：为幕布边框和 PPT 边框打标注
2. **深度学习流程**（Python）：基于标注数据训练角点回归模型

---

## 文件说明

### C# 标注工具

| 文件 | 说明 |
|------|------|
| `Form1.cs` | 主窗体逻辑，双模式标注（PPT 边框 / 幕布边缘）、鼠标交互、JSON 读写 |
| `Form1.Designer.cs` | 窗体设计器自动生成文件 |
| `Program.cs` | 程序入口 |
| `AnnotationTool.csproj` | .NET 10 WinForms 项目文件 |

### Python 深度学习流程

| 文件 | 说明 |
|------|------|
| `data_augmentation.py` | 数据增强脚本，读取 `ScreenCorners` 标注，生成训练/验证集（几何变换 + 光学增强 + 边框遮挡模拟）|
| `train.py` | 模型训练脚本，MobileNetV3-Small 骨干网 + 自定义回归头，WingLoss，输出归一化角点坐标 |
| `visualize_model.py` | 模型验证可视化，加载 `best_model.pth`，在验证集上推理并对比 GT vs 预测结果 |
| `test_homography.py` | 单应性矩阵后处理测试，对模型输出做几何约束修正（强制四边形满足透视矩形约束） |

### 数据目录（已加入 .gitignore，不提交）

| 目录 | 说明 |
|------|------|
| `data/` | 原始标注数据，包含照片 `.jpg` 和对应标注 `.json`（含 `ScreenCorners` / `PptCorners`）|
| `training_data/` | 增强后的训练数据集（`images/train`, `images/val`, `labels/train`, `labels/val`）|
| `test_homography/` | `test_homography.py` 输出的可视化对比图 |
| `test_results/` | `test_canny_params.py` 等传统算法对比测试的历史输出 |
| `visualization/` | `visualize_model.py` 输出的模型预测可视化图 |

### 模型文件（已加入 .gitignore）

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 训练过程中验证误差最低的模型权重（PyTorch checkpoint）|

---

## 数据格式

标注文件与图片同名，扩展名 `.json`，存放在 `data/` 目录下。

```json
{
  "ScreenCorners": [
    { "X": 628, "Y": 766 },
    { "X": 2590, "Y": 774 },
    { "X": 2525, "Y": 2068 },
    { "X": 494, "Y": 1936 }
  ],
  "PptCorners": [
    { "X": 700, "Y": 840 },
    { "X": 2510, "Y": 850 },
    { "X": 2450, "Y": 2000 },
    { "X": 560, "Y": 1870 }
  ]
}
```

- **`ScreenCorners`**：幕布边缘四角，当前深度学习训练使用的标注
- **`PptCorners`**：PPT 投影区域四角，旧版标注字段（保留数据，不用于训练）
- 角点顺序：左上 → 右上 → 右下 → 左下
- 坐标系：图片左上角为原点，像素单位，允许超出图片边界

---

## 标注工具使用方法

```cmd
cd D:\Programing\C#\MauiScan\AnnotationTool
dotnet run
```

### 操作说明

1. 点击"加载图片文件夹"，选择包含照片的目录
2. 从下拉框选择标注模式（PPT 边框 / 幕布边缘）
3. 依次点击四个角点（左上 → 右上 → 右下 → 左下）
4. 拖拽圆圈微调角点位置
5. 按 `Enter` 保存并跳转下一张

### 快捷键

| 按键 | 功能 |
|------|------|
| `Tab` | 切换标注模式 |
| `Enter` | 保存当前模式标注并下一张 |
| `Space` | 撤销上一个角点 |
| `Esc` | 清除当前模式所有角点 |
| `←` / `→` | 上一张 / 下一张 |
| `R` | 重置缩放视图 |
| 滚轮 | 缩放图片 |
| 中键拖拽 | 平移图片 |

---

## 深度学习训练流程

### 1. 生成训练数据

```powershell
# 增强倍数默认 200，可通过参数调整
python data_augmentation.py 10
```

输出到 `training_data/`（10 倍时约 85×10=850 训练样本）。

### 2. 训练模型

```powershell
python train.py
```

- 骨干网：MobileNetV3-Small（预训练权重）
- 损失函数：WingLoss（对小误差更敏感）
- 输出：8 个归一化坐标值（4 角点 × x/y），Sigmoid 激活
- 最佳模型自动保存到 `best_model.pth`

### 3. 验证结果

```powershell
python visualize_model.py
```

可视化图片保存到 `visualization/`。

### 4. 单应性矩阵后处理测试

```powershell
python test_homography.py
```

测试对模型输出四点做透视约束修正的效果，结果保存到 `test_homography/`（三栏对比：GT | 修正前 | 修正后）。

---

## 模型说明

**问题**：模型独立预测 4 个角点，轻微误差会导致四边形不满足透视矩形约束（对边不平行）。

**后处理方案**：`test_homography.py` 中实现了基于 DLT（直接线性变换）的单应性矩阵修正：
1. 估算四边形的宽高比（由四边平均长度推算）
2. 构造 DLT 方程组，SVD 求解最优单应性矩阵 H
3. 用 H 反投影得到满足透视矩形约束的修正角点

此方案**不假设固定宽高比**（如 16:9），适用于不同比例的幕布。
