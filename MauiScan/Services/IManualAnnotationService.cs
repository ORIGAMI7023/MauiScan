namespace MauiScan.Services;

/// <summary>
/// 手动标注服务接口（用于识别失败时手动指定文档边界）
/// </summary>
public interface IManualAnnotationService
{
    /// <summary>
    /// 启动手动标注界面
    /// </summary>
    /// <param name="imageBytes">原始图片数据</param>
    /// <returns>标注结果（包含4个角点坐标和处理后的图片）</returns>
    Task<ManualAnnotationResult?> AnnotateAsync(byte[] imageBytes);
}

/// <summary>
/// 手动标注结果
/// </summary>
public class ManualAnnotationResult
{
    /// <summary>
    /// 是否标注成功
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// 处理后的图片数据（经过透视变换）
    /// </summary>
    public byte[] ProcessedImageData { get; set; } = Array.Empty<byte>();

    /// <summary>
    /// 图片宽度
    /// </summary>
    public int Width { get; set; }

    /// <summary>
    /// 图片高度
    /// </summary>
    public int Height { get; set; }

    /// <summary>
    /// 用户标注的4个角点坐标 [topLeftX, topLeftY, topRightX, topRightY, bottomRightX, bottomRightY, bottomLeftX, bottomLeftY]
    /// </summary>
    public float[] Corners { get; set; } = Array.Empty<float>();
}
