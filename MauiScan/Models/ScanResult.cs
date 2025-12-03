namespace MauiScan.Models;

/// <summary>
/// 扫描结果封装
/// </summary>
public class ScanResult
{
    /// <summary>
    /// 处理后的图像数据（PNG/JPEG 字节）
    /// </summary>
    public byte[] ImageData { get; set; }

    /// <summary>
    /// 检测到的四边形顶点（原始坐标）
    /// </summary>
    public QuadrilateralPoints? DetectedQuad { get; set; }

    /// <summary>
    /// 是否成功检测到文档区域
    /// </summary>
    public bool IsSuccess { get; set; }

    /// <summary>
    /// 处理后的图像宽度
    /// </summary>
    public int Width { get; set; }

    /// <summary>
    /// 处理后的图像高度
    /// </summary>
    public int Height { get; set; }

    /// <summary>
    /// 错误信息（如果处理失败）
    /// </summary>
    public string? ErrorMessage { get; set; }

    public ScanResult(byte[] imageData, int width, int height, QuadrilateralPoints? detectedQuad = null)
    {
        ImageData = imageData;
        Width = width;
        Height = height;
        DetectedQuad = detectedQuad;
        IsSuccess = true;
    }

    public static ScanResult Failure(string errorMessage)
    {
        return new ScanResult(Array.Empty<byte>(), 0, 0)
        {
            IsSuccess = false,
            ErrorMessage = errorMessage
        };
    }
}
