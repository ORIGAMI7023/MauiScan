namespace MauiScan.Models;

/// <summary>
/// 两阶段检测结果
/// </summary>
public class TwoStageDetectionResult
{
    /// <summary>
    /// 第一阶段：幕布检测
    /// </summary>
    public StageResult ScreenStage { get; set; } = new();

    /// <summary>
    /// 第二阶段：PPT检测（可能为null）
    /// </summary>
    public StageResult? PptStage { get; set; }

    /// <summary>
    /// 两阶段是否都成功
    /// </summary>
    public bool BothStagesSuccess => ScreenStage.IsSuccess && PptStage?.IsSuccess == true;

    /// <summary>
    /// 原始图像字节
    /// </summary>
    public byte[] OriginalImageBytes { get; set; } = Array.Empty<byte>();

    /// <summary>
    /// 原始图像尺寸
    /// </summary>
    public (int Width, int Height) OriginalSize { get; set; }
}

/// <summary>
/// 单个阶段的检测结果
/// </summary>
public class StageResult
{
    /// <summary>
    /// 检测是否成功
    /// </summary>
    public bool IsSuccess { get; set; }

    /// <summary>
    /// 检测到的四边形顶点（绝对坐标）
    /// </summary>
    public QuadrilateralPoints? Quad { get; set; }

    /// <summary>
    /// 置信度 (0-1)
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// 错误消息（失败时）
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;
}

/// <summary>
/// 裁剪区域信息（用于坐标转换）
/// </summary>
public class CropRegion
{
    /// <summary>
    /// 原始四边形（在原图中的位置）
    /// </summary>
    public QuadrilateralPoints OriginalQuad { get; set; } = new(
        new Point2D(0, 0),
        new Point2D(0, 0),
        new Point2D(0, 0),
        new Point2D(0, 0)
    );

    /// <summary>
    /// 缩小比例
    /// </summary>
    public double ShrinkRatio { get; set; }

    /// <summary>
    /// 裁剪后的图像尺寸
    /// </summary>
    public (int Width, int Height) CroppedSize { get; set; }

    /// <summary>
    /// 透视变换矩阵的源点（缩小后的四边形）
    /// </summary>
    public QuadrilateralPoints ShrunkQuad { get; set; } = new(
        new Point2D(0, 0),
        new Point2D(0, 0),
        new Point2D(0, 0),
        new Point2D(0, 0)
    );
}
