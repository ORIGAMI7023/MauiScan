using MauiScan.Models;

namespace MauiScan.Services;

/// <summary>
/// 图像处理服务接口（边缘检测、透视变换、图像增强）
/// </summary>
public interface IImageProcessingService
{
    /// <summary>
    /// 处理扫描图像：检测文档边界 → 透视校正 → 可选增强
    /// </summary>
    /// <param name="imageBytes">原始图像字节数据</param>
    /// <param name="applyEnhancement">是否应用图像增强（灰度化、对比度调整）</param>
    /// <returns>扫描结果</returns>
    Task<ScanResult> ProcessScanAsync(byte[] imageBytes, bool applyEnhancement = false);

    /// <summary>
    /// 仅检测文档四边形边界（不做透视变换）
    /// </summary>
    Task<QuadrilateralPoints?> DetectDocumentBoundsAsync(byte[] imageBytes);
}
