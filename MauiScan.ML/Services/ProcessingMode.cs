namespace MauiScan.ML.Services;

/// <summary>
/// 图像处理模式
/// </summary>
public enum ProcessingMode
{
    /// <summary>
    /// 自动选择（优先 ML，根据置信度降级）
    /// </summary>
    Auto,

    /// <summary>
    /// 仅使用 ML 推理
    /// </summary>
    MLOnly,

    /// <summary>
    /// 仅使用传统 OpenCV 算法
    /// </summary>
    TraditionalOnly,

    /// <summary>
    /// 混合模式（ML + 传统算法融合）
    /// </summary>
    Hybrid
}
