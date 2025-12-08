using MauiScan.ML.Models;
using OpenCvSharp;

namespace MauiScan.ML.Services;

/// <summary>
/// OnnxInferenceService - Windows 平台实现
/// </summary>
public partial class OnnxInferenceService
{
    /// <summary>
    /// 精修ML预测的角点（使用传统CV方法达到亚像素精度）
    /// </summary>
    private partial QuadrilateralPoints RefineCorners(
        QuadrilateralPoints mlCorners,
        byte[] imageBytes,
        int originalWidth,
        int originalHeight)
    {
        System.Diagnostics.Debug.WriteLine($"[Refinement] Starting corner refinement on {originalWidth}x{originalHeight} image");

        // 加载原图为灰度图
        using var mat = Mat.FromImageData(imageBytes, ImreadModes.Grayscale);

        if (mat.Empty())
        {
            System.Diagnostics.Debug.WriteLine($"[Refinement] Failed to load image, returning ML corners");
            return mlCorners;
        }

        try
        {
            // 对4个角点分别精修
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

            // 精修左上角
            var (tlX, tlY) = RefineSingleCorner(mat, mlCorners.TopLeftX, mlCorners.TopLeftY);
            if (tlX >= 0)
            {
                refined.TopLeftX = tlX;
                refined.TopLeftY = tlY;
            }

            // 精修右上角
            var (trX, trY) = RefineSingleCorner(mat, mlCorners.TopRightX, mlCorners.TopRightY);
            if (trX >= 0)
            {
                refined.TopRightX = trX;
                refined.TopRightY = trY;
            }

            // 精修右下角
            var (brX, brY) = RefineSingleCorner(mat, mlCorners.BottomRightX, mlCorners.BottomRightY);
            if (brX >= 0)
            {
                refined.BottomRightX = brX;
                refined.BottomRightY = brY;
            }

            // 精修左下角
            var (blX, blY) = RefineSingleCorner(mat, mlCorners.BottomLeftX, mlCorners.BottomLeftY);
            if (blX >= 0)
            {
                refined.BottomLeftX = blX;
                refined.BottomLeftY = blY;
            }

            System.Diagnostics.Debug.WriteLine($"[Refinement] Refinement completed successfully");
            return refined;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[Refinement] Error during refinement: {ex.Message}");
            // 精修失败，返回ML原始结果
            return mlCorners;
        }
    }

    /// <summary>
    /// 精修单个角点
    /// </summary>
    /// <param name="image">灰度图</param>
    /// <param name="mlX">ML预测的X坐标</param>
    /// <param name="mlY">ML预测的Y坐标</param>
    /// <returns>精修后的坐标，失败返回(-1, -1)</returns>
    private (float x, float y) RefineSingleCorner(Mat image, float mlX, float mlY)
    {
        const int patchSize = 64; // 搜索窗口大小

        try
        {
            // 1. 裁剪patch（确保不越界）
            int centerX = (int)mlX;
            int centerY = (int)mlY;
            int halfPatch = patchSize / 2;

            int x1 = Math.Max(0, centerX - halfPatch);
            int y1 = Math.Max(0, centerY - halfPatch);
            int x2 = Math.Min(image.Width, centerX + halfPatch);
            int y2 = Math.Min(image.Height, centerY + halfPatch);

            if (x2 - x1 < 20 || y2 - y1 < 20)
            {
                // Patch太小，返回失败
                return (-1, -1);
            }

            var roi = new Rect(x1, y1, x2 - x1, y2 - y1);
            using var patch = new Mat(image, roi);

            // 2. Canny边缘检测
            using var edges = new Mat();
            Cv2.Canny(patch, edges, 50, 150, 3);

            // 3. 霍夫直线检测
            var lines = Cv2.HoughLinesP(
                edges,
                rho: 1,
                theta: Math.PI / 180,
                threshold: 30,
                minLineLength: 20,
                maxLineGap: 5
            );

            if (lines == null || lines.Length < 2)
            {
                // 检测到的直线太少
                return (-1, -1);
            }

            // 4. 直线聚类（水平 vs 垂直）
            var horizontalLines = new List<LineSegmentPoint>();
            var verticalLines = new List<LineSegmentPoint>();

            foreach (var line in lines)
            {
                float dx = Math.Abs(line.P2.X - line.P1.X);
                float dy = Math.Abs(line.P2.Y - line.P1.Y);

                if (dx > dy) // 更水平
                {
                    horizontalLines.Add(line);
                }
                else // 更垂直
                {
                    verticalLines.Add(line);
                }
            }

            if (horizontalLines.Count == 0 || verticalLines.Count == 0)
            {
                // 没有找到两组直线
                return (-1, -1);
            }

            // 5. 拟合直线（简化版：使用中位数斜率和截距）
            var hLine = FitLine(horizontalLines);
            var vLine = FitLine(verticalLines);

            if (hLine == null || vLine == null)
            {
                return (-1, -1);
            }

            // 6. 计算交点
            var intersection = ComputeIntersection(hLine.Value, vLine.Value);
            if (intersection == null)
            {
                return (-1, -1);
            }

            // 7. 转换回原图坐标
            float refinedX = x1 + intersection.Value.X;
            float refinedY = y1 + intersection.Value.Y;

            // 8. 验证精修结果是否合理（距离ML预测不能太远）
            float distance = (float)Math.Sqrt(
                Math.Pow(refinedX - mlX, 2) + Math.Pow(refinedY - mlY, 2)
            );

            if (distance > patchSize) // 超出搜索范围
            {
                return (-1, -1);
            }

            return (refinedX, refinedY);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[Refinement] Error in RefineSingleCorner: {ex.Message}");
            return (-1, -1);
        }
    }

    /// <summary>
    /// 拟合直线（返回直线方程参数）
    /// </summary>
    private (float k, float b)? FitLine(List<LineSegmentPoint> lines)
    {
        if (lines.Count == 0) return null;

        // 简化版：使用所有线段端点进行最小二乘拟合
        var points = new List<Point2f>();
        foreach (var line in lines)
        {
            points.Add(new Point2f(line.P1.X, line.P1.Y));
            points.Add(new Point2f(line.P2.X, line.P2.Y));
        }

        // 计算平均值
        float avgX = points.Average(p => p.X);
        float avgY = points.Average(p => p.Y);

        // 最小二乘法
        float numerator = 0;
        float denominator = 0;

        foreach (var p in points)
        {
            numerator += (p.X - avgX) * (p.Y - avgY);
            denominator += (p.X - avgX) * (p.X - avgX);
        }

        if (Math.Abs(denominator) < 1e-6)
        {
            // 垂直线，无法用 y=kx+b 表示
            return null;
        }

        float k = numerator / denominator;
        float b = avgY - k * avgX;

        return (k, b);
    }

    /// <summary>
    /// 计算两条直线的交点
    /// </summary>
    private Point2f? ComputeIntersection((float k, float b) line1, (float k, float b) line2)
    {
        // line1: y = k1*x + b1
        // line2: y = k2*x + b2
        // 交点: k1*x + b1 = k2*x + b2

        float k1 = line1.k, b1 = line1.b;
        float k2 = line2.k, b2 = line2.b;

        if (Math.Abs(k1 - k2) < 1e-6)
        {
            // 平行线
            return null;
        }

        float x = (b2 - b1) / (k1 - k2);
        float y = k1 * x + b1;

        return new Point2f(x, y);
    }
}
