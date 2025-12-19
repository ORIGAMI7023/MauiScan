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
    /// <param name="imageBytes">原始图像字节数据</param>
    /// <param name="minAreaRatio">最小轮廓面积占比（0.0-1.0），默认0.05（5%）</param>
    Task<QuadrilateralPoints?> DetectDocumentBoundsAsync(byte[] imageBytes, double minAreaRatio = 0.05);

    /// <summary>
    /// 基于亮度阈值检测PPT边界（用于第二阶段检测）
    /// 适用于PPT投影区域比幕布更亮的场景
    /// </summary>
    /// <param name="imageBytes">裁剪后的幕布区域图像字节数据</param>
    /// <param name="brightnessThreshold">亮度阈值增量（默认20，相对于平均亮度）</param>
    /// <param name="minAreaRatio">最小面积占比（默认0.5，即50%）</param>
    Task<QuadrilateralPoints?> DetectDocumentBoundsByBrightnessAsync(byte[] imageBytes, int brightnessThreshold = 20, double minAreaRatio = 0.5);
}
