using MauiScan.ML.Interop;
using MauiScan.ML.Models;

namespace MauiScan.ML.Services;

/// <summary>
/// OnnxInferenceService - macCatalyst 平台实现
/// </summary>
public partial class OnnxInferenceService
{
    /// <summary>
    /// 精修ML预测的角点（使用 Native OpenCV）
    /// </summary>
    private partial QuadrilateralPoints RefineCorners(
        QuadrilateralPoints mlCorners,
        byte[] imageBytes,
        int originalWidth,
        int originalHeight)
    {
        System.Diagnostics.Debug.WriteLine($"[Refinement] Starting Native OpenCV corner refinement on {originalWidth}x{originalHeight} image");

        try
        {
            var refined = new QuadrilateralPoints
            {
                TopLeftX = mlCorners.TopLeftX,
                TopLeftY = mlCorners.TopLeftY,
                TopRightX = mlCorners.TopRightX,
                TopRightY = mlCorners.TopRightY,
                BottomRightX = mlCorners.BottomRightX,
                BottomRightY = mlCorners.BottomRightY,
                BottomLeftX = mlCorners.BottomLeftX,
                BottomLeftY = mlCorners.BottomLeftY
            };

            // 精修4个角点
            int success = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.TopLeftX, mlCorners.TopLeftY, out float tlX, out float tlY);
            if (success == 1) { refined.TopLeftX = tlX; refined.TopLeftY = tlY; }

            success = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.TopRightX, mlCorners.TopRightY, out float trX, out float trY);
            if (success == 1) { refined.TopRightX = trX; refined.TopRightY = trY; }

            success = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.BottomRightX, mlCorners.BottomRightY, out float brX, out float brY);
            if (success == 1) { refined.BottomRightX = brX; refined.BottomRightY = brY; }

            success = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.BottomLeftX, mlCorners.BottomLeftY, out float blX, out float blY);
            if (success == 1) { refined.BottomLeftX = blX; refined.BottomLeftY = blY; }

            System.Diagnostics.Debug.WriteLine($"[Refinement] Native OpenCV refinement completed successfully");
            return refined;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[Refinement] Native OpenCV refinement failed: {ex.Message}");
            return mlCorners;
        }
    }
}
