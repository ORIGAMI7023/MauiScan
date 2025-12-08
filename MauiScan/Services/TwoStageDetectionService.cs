using MauiScan.Models;
using System.Diagnostics;

namespace MauiScan.Services;

/// <summary>
/// 两阶段文档检测服务
/// 第一阶段：检测投影幕布
/// 第二阶段：在幕布内检测PPT边界
/// </summary>
public class TwoStageDetectionService
{
    private readonly IImageProcessingService _nativeService;
    private readonly ImageCropService _cropService;

    public TwoStageDetectionService(IImageProcessingService nativeService, ImageCropService cropService)
    {
        _nativeService = nativeService;
        _cropService = cropService;
    }

    /// <summary>
    /// 执行两阶段检测
    /// </summary>
    /// <param name="imageBytes">原始图像字节</param>
    /// <param name="shrinkRatio">幕布缩小比例，默认0.08</param>
    /// <returns>两阶段检测结果</returns>
    public async Task<TwoStageDetectionResult> DetectAsync(byte[] imageBytes, double shrinkRatio = 0.08)
    {
        Debug.WriteLine("[TwoStageDetection] Starting two-stage detection");

        var result = new TwoStageDetectionResult
        {
            OriginalImageBytes = imageBytes
        };

        // 获取原始图像尺寸
        var imageSize = GetImageSize(imageBytes);
        result.OriginalSize = imageSize;

        Debug.WriteLine($"[TwoStageDetection] Image size: {imageSize.Width}x{imageSize.Height}");

        // 第一阶段：检测幕布
        Debug.WriteLine("[TwoStageDetection] Stage 1: Detecting screen...");

        var screenQuad = await _nativeService.DetectDocumentBoundsAsync(imageBytes);

        if (screenQuad == null)
        {
            Debug.WriteLine("[TwoStageDetection] Stage 1 failed: No screen detected");

            result.ScreenStage = new StageResult
            {
                IsSuccess = false,
                ErrorMessage = "幕布检测失败 - 未检测到文档边界"
            };

            return result;
        }

        Debug.WriteLine($"[TwoStageDetection] Stage 1 success: TL({screenQuad.TopLeft.X},{screenQuad.TopLeft.Y})");

        result.ScreenStage = new StageResult
        {
            IsSuccess = true,
            Quad = screenQuad,
            Confidence = CalculateConfidence(screenQuad, imageSize)
        };

        // 第二阶段：裁剪幕布区域并检测PPT
        Debug.WriteLine("[TwoStageDetection] Stage 2: Cropping screen region...");

        var cropResult = _cropService.CropAndShrink(imageBytes, screenQuad, shrinkRatio);

        if (cropResult == null)
        {
            Debug.WriteLine("[TwoStageDetection] Stage 2 failed: Crop failed");

            result.PptStage = new StageResult
            {
                IsSuccess = false,
                ErrorMessage = "裁剪失败 - 无法提取幕布区域"
            };

            return result;
        }

        Debug.WriteLine($"[TwoStageDetection] Cropped size: {cropResult.Value.Region.CroppedSize.Width}x{cropResult.Value.Region.CroppedSize.Height}");

        // 在裁剪后的区域中检测PPT
        Debug.WriteLine("[TwoStageDetection] Stage 2: Detecting PPT in cropped region...");

        var pptQuadRelative = await _nativeService.DetectDocumentBoundsAsync(cropResult.Value.CroppedBytes);

        if (pptQuadRelative == null)
        {
            Debug.WriteLine("[TwoStageDetection] Stage 2 failed: No PPT detected in cropped region");

            result.PptStage = new StageResult
            {
                IsSuccess = false,
                ErrorMessage = "PPT检测失败 - 在幕布内未检测到PPT边界"
            };

            return result;
        }

        Debug.WriteLine($"[TwoStageDetection] PPT detected (relative): TL({pptQuadRelative.TopLeft.X},{pptQuadRelative.TopLeft.Y})");

        // 转换相对坐标为绝对坐标
        var pptQuadAbsolute = _cropService.TransformToAbsolute(pptQuadRelative, cropResult.Value.Region);

        Debug.WriteLine($"[TwoStageDetection] PPT (absolute): TL({pptQuadAbsolute.TopLeft.X},{pptQuadAbsolute.TopLeft.Y})");

        result.PptStage = new StageResult
        {
            IsSuccess = true,
            Quad = pptQuadAbsolute,
            Confidence = CalculateConfidence(pptQuadAbsolute, imageSize)
        };

        Debug.WriteLine($"[TwoStageDetection] Both stages completed successfully");

        return result;
    }

    /// <summary>
    /// 计算置信度评分（基于面积比例和形状规则性）
    /// 参考 Native.OpenCV 中的 calculate_contour_score 逻辑
    /// </summary>
    private double CalculateConfidence(QuadrilateralPoints quad, (int Width, int Height) imageSize)
    {
        try
        {
            // 计算四边形面积（使用 Shoelace 公式）
            double area = Math.Abs(
                (quad.TopLeft.X * quad.TopRight.Y - quad.TopRight.X * quad.TopLeft.Y) +
                (quad.TopRight.X * quad.BottomRight.Y - quad.BottomRight.X * quad.TopRight.Y) +
                (quad.BottomRight.X * quad.BottomLeft.Y - quad.BottomLeft.X * quad.BottomRight.Y) +
                (quad.BottomLeft.X * quad.TopLeft.Y - quad.TopLeft.X * quad.BottomLeft.Y)
            ) / 2.0;

            // 图像总面积
            double imageArea = imageSize.Width * imageSize.Height;

            // 面积比例
            double areaRatio = area / imageArea;

            // 置信度主要基于面积比例（占比越大越好）
            // 0-30%面积 → 0-0.5分
            // 30-60%面积 → 0.5-0.8分
            // 60%以上 → 0.8-1.0分
            double confidence;
            if (areaRatio < 0.3)
            {
                confidence = areaRatio / 0.3 * 0.5; // 0-0.5
            }
            else if (areaRatio < 0.6)
            {
                confidence = 0.5 + (areaRatio - 0.3) / 0.3 * 0.3; // 0.5-0.8
            }
            else
            {
                confidence = 0.8 + Math.Min((areaRatio - 0.6) / 0.4 * 0.2, 0.2); // 0.8-1.0
            }

            return Math.Clamp(confidence, 0.0, 1.0);
        }
        catch
        {
            return 0.5; // 默认中等置信度
        }
    }

    /// <summary>
    /// 获取图像尺寸（从JPEG头部读取）
    /// </summary>
    private (int Width, int Height) GetImageSize(byte[] imageBytes)
    {
        try
        {
#if ANDROID
            using var bitmap = Android.Graphics.BitmapFactory.DecodeByteArray(imageBytes, 0, imageBytes.Length);
            if (bitmap != null)
            {
                return (bitmap.Width, bitmap.Height);
            }
#endif
            // 默认值
            return (0, 0);
        }
        catch
        {
            return (0, 0);
        }
    }
}
