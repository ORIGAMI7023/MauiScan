namespace MauiScan.ML.Models;

/// <summary>
/// ML 模型检测结果
/// </summary>
public class MLDetectionResult
{
    /// <summary>
    /// 检测到的四个角点坐标（左上、右上、右下、左下）- 最终结果（可能经过 CV 精修）
    /// </summary>
    public QuadrilateralPoints Corners { get; set; } = new();

    /// <summary>
    /// ML 原始输出的角点坐标（未经 CV 精修）
    /// </summary>
    public QuadrilateralPoints? MLRawCorners { get; set; }

    /// <summary>
    /// ML 输出的归一化坐标 [x1, y1, x2, y2, x3, y3, x4, y4]
    /// </summary>
    public float[]? NormalizedCoordinates { get; set; }

    /// <summary>
    /// 整体置信度 (0-1)，表示模型对检测结果的确信程度
    /// </summary>
    public float Confidence { get; set; }

    /// <summary>
    /// 每个角点的置信度 (0-1)，顺序与 Corners 一致
    /// </summary>
    public float[] CornerConfidences { get; set; } = new float[4];

    /// <summary>
    /// 是否为高质量检测结果 (Confidence >= 0.85)
    /// </summary>
    public bool IsHighQuality => Confidence >= 0.85f;

    /// <summary>
    /// 是否为中等质量检测结果 (0.60 <= Confidence < 0.85)
    /// </summary>
    public bool IsMediumQuality => Confidence >= 0.60f && Confidence < 0.85f;

    /// <summary>
    /// 是否为低质量检测结果 (Confidence < 0.60)，应该降级使用传统算法
    /// </summary>
    public bool IsLowQuality => Confidence < 0.60f;
}

/// <summary>
/// 四边形角点坐标
/// </summary>
public class QuadrilateralPoints
{
    /// <summary>
    /// 左上角 X 坐标（像素）
    /// </summary>
    public float TopLeftX { get; set; }

    /// <summary>
    /// 左上角 Y 坐标（像素）
    /// </summary>
    public float TopLeftY { get; set; }

    /// <summary>
    /// 右上角 X 坐标（像素）
    /// </summary>
    public float TopRightX { get; set; }

    /// <summary>
    /// 右上角 Y 坐标（像素）
    /// </summary>
    public float TopRightY { get; set; }

    /// <summary>
    /// 右下角 X 坐标（像素）
    /// </summary>
    public float BottomRightX { get; set; }

    /// <summary>
    /// 右下角 Y 坐标（像素）
    /// </summary>
    public float BottomRightY { get; set; }

    /// <summary>
    /// 左下角 X 坐标（像素）
    /// </summary>
    public float BottomLeftX { get; set; }

    /// <summary>
    /// 左下角 Y 坐标（像素）
    /// </summary>
    public float BottomLeftY { get; set; }

    /// <summary>
    /// 转换为数组格式 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    /// </summary>
    public (float x, float y)[] ToArray()
    {
        return new[]
        {
            (TopLeftX, TopLeftY),
            (TopRightX, TopRightY),
            (BottomRightX, BottomRightY),
            (BottomLeftX, BottomLeftY)
        };
    }

    /// <summary>
    /// 从数组创建 QuadrilateralPoints
    /// </summary>
    public static QuadrilateralPoints FromArray((float x, float y)[] points)
    {
        if (points.Length != 4)
            throw new ArgumentException("必须提供 4 个角点", nameof(points));

        return new QuadrilateralPoints
        {
            TopLeftX = points[0].x,
            TopLeftY = points[0].y,
            TopRightX = points[1].x,
            TopRightY = points[1].y,
            BottomRightX = points[2].x,
            BottomRightY = points[2].y,
            BottomLeftX = points[3].x,
            BottomLeftY = points[3].y
        };
    }
}
