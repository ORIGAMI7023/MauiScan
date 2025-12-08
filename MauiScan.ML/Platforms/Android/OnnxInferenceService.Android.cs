using MauiScan.ML.Interop;
using MauiScan.ML.Models;

namespace MauiScan.ML.Services;

/// <summary>
/// OnnxInferenceService - Android 平台实现
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
        System.Diagnostics.Debug.WriteLine($"[Refinement] Image bytes: {imageBytes.Length} bytes");

        try
        {
            // 测试库是否可以加载
            try
            {
                var versionPtr = NativeOpenCv.GetVersion();
                var version = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(versionPtr);
                System.Diagnostics.Debug.WriteLine($"[Refinement] Native OpenCV library loaded successfully, version: {version}");
            }
            catch (Exception libEx)
            {
                System.Diagnostics.Debug.WriteLine($"[Refinement] Failed to load Native OpenCV library: {libEx.GetType().Name}: {libEx.Message}");
                return mlCorners;
            }

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

            // 第 1 层：精修4个角点并收集置信度
            System.Diagnostics.Debug.WriteLine($"[Refinement] === Layer 1: CV Refinement with Confidence ===");

            System.Diagnostics.Debug.WriteLine($"[Refinement] Refining TL: ({mlCorners.TopLeftX:F1}, {mlCorners.TopLeftY:F1})");
            int confidenceTL = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.TopLeftX, mlCorners.TopLeftY, out float tlX, out float tlY);
            System.Diagnostics.Debug.WriteLine($"[Refinement] TL result: confidence={confidenceTL}, refined=({tlX:F1}, {tlY:F1})");

            System.Diagnostics.Debug.WriteLine($"[Refinement] Refining TR: ({mlCorners.TopRightX:F1}, {mlCorners.TopRightY:F1})");
            int confidenceTR = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.TopRightX, mlCorners.TopRightY, out float trX, out float trY);
            System.Diagnostics.Debug.WriteLine($"[Refinement] TR result: confidence={confidenceTR}, refined=({trX:F1}, {trY:F1})");

            System.Diagnostics.Debug.WriteLine($"[Refinement] Refining BR: ({mlCorners.BottomRightX:F1}, {mlCorners.BottomRightY:F1})");
            int confidenceBR = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.BottomRightX, mlCorners.BottomRightY, out float brX, out float brY);
            System.Diagnostics.Debug.WriteLine($"[Refinement] BR result: confidence={confidenceBR}, refined=({brX:F1}, {brY:F1})");

            System.Diagnostics.Debug.WriteLine($"[Refinement] Refining BL: ({mlCorners.BottomLeftX:F1}, {mlCorners.BottomLeftY:F1})");
            int confidenceBL = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.BottomLeftX, mlCorners.BottomLeftY, out float blX, out float blY);
            System.Diagnostics.Debug.WriteLine($"[Refinement] BL result: confidence={confidenceBL}, refined=({blX:F1}, {blY:F1})");

            // 第 1 层决策：只接受置信度 > 0 的结果
            if (confidenceTL > 0) { refined.TopLeftX = tlX; refined.TopLeftY = tlY; }
            if (confidenceTR > 0) { refined.TopRightX = trX; refined.TopRightY = trY; }
            if (confidenceBR > 0) { refined.BottomRightX = brX; refined.BottomRightY = brY; }
            if (confidenceBL > 0) { refined.BottomLeftX = blX; refined.BottomLeftY = blY; }

            // 第 2 层：相对一致性检测（防单点作妖）
            System.Diagnostics.Debug.WriteLine($"[Refinement] === Layer 2: Outlier Detection ===");

            // 计算4个点的偏移距离
            float[] distances = new float[4];
            distances[0] = Distance(mlCorners.TopLeftX, mlCorners.TopLeftY, refined.TopLeftX, refined.TopLeftY);
            distances[1] = Distance(mlCorners.TopRightX, mlCorners.TopRightY, refined.TopRightX, refined.TopRightY);
            distances[2] = Distance(mlCorners.BottomRightX, mlCorners.BottomRightY, refined.BottomRightX, refined.BottomRightY);
            distances[3] = Distance(mlCorners.BottomLeftX, mlCorners.BottomLeftY, refined.BottomLeftX, refined.BottomLeftY);

            System.Diagnostics.Debug.WriteLine($"[Refinement] Distances: TL={distances[0]:F1}px, TR={distances[1]:F1}px, BR={distances[2]:F1}px, BL={distances[3]:F1}px");

            // 检测离群值：某点偏移 > 2×其他平均 且 >100px
            for (int i = 0; i < 4; i++)
            {
                float avgOthers = (distances.Where((d, idx) => idx != i).Sum()) / 3f;
                bool isOutlier = distances[i] > 2 * avgOthers && distances[i] > 100;

                if (isOutlier)
                {
                    System.Diagnostics.Debug.WriteLine($"[Refinement] Point {i} is outlier: {distances[i]:F1}px > 2×{avgOthers:F1}px, reverting to ML");

                    // 回退到 ML 原始坐标
                    switch (i)
                    {
                        case 0: refined.TopLeftX = mlCorners.TopLeftX; refined.TopLeftY = mlCorners.TopLeftY; break;
                        case 1: refined.TopRightX = mlCorners.TopRightX; refined.TopRightY = mlCorners.TopRightY; break;
                        case 2: refined.BottomRightX = mlCorners.BottomRightX; refined.BottomRightY = mlCorners.BottomRightY; break;
                        case 3: refined.BottomLeftX = mlCorners.BottomLeftX; refined.BottomLeftY = mlCorners.BottomLeftY; break;
                    }
                }
            }

            // 第 3 层：四边形几何兜底（防整体崩盘）
            System.Diagnostics.Debug.WriteLine($"[Refinement] === Layer 3: Quadrilateral Geometry Validation ===");

            if (!IsValidQuadrilateral(refined))
            {
                System.Diagnostics.Debug.WriteLine($"[Refinement] Refined quadrilateral is INVALID, reverting ALL points to ML");
                return mlCorners;  // 整体回退
            }

            System.Diagnostics.Debug.WriteLine($"[Refinement] Refined quadrilateral is VALID");
            System.Diagnostics.Debug.WriteLine($"[Refinement] Native OpenCV refinement completed successfully");
            return refined;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[Refinement] Native OpenCV refinement exception: {ex.GetType().Name}");
            System.Diagnostics.Debug.WriteLine($"[Refinement] Exception message: {ex.Message}");
            System.Diagnostics.Debug.WriteLine($"[Refinement] Stack trace: {ex.StackTrace}");
            return mlCorners;
        }
    }

    /// <summary>
    /// 计算两点间的欧几里得距离
    /// </summary>
    private static float Distance(float x1, float y1, float x2, float y2)
    {
        float dx = x2 - x1;
        float dy = y2 - y1;
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }

    /// <summary>
    /// 验证四边形的几何合法性
    /// </summary>
    private static bool IsValidQuadrilateral(QuadrilateralPoints quad)
    {
        // 1. 检查凸性（使用叉积）
        if (!IsConvex(quad))
        {
            System.Diagnostics.Debug.WriteLine($"[Validation] Quadrilateral is NOT convex");
            return false;
        }

        // 2. 检查内角范围 (30° ~ 150°)
        var angles = GetInternalAngles(quad);
        for (int i = 0; i < 4; i++)
        {
            if (angles[i] < 30 || angles[i] > 150)
            {
                System.Diagnostics.Debug.WriteLine($"[Validation] Angle {i} is out of range: {angles[i]:F1}°");
                return false;
            }
        }

        // 3. 检查边长比例 (max/min < 5)
        float[] edges = new float[4];
        edges[0] = Distance(quad.TopLeftX, quad.TopLeftY, quad.TopRightX, quad.TopRightY);
        edges[1] = Distance(quad.TopRightX, quad.TopRightY, quad.BottomRightX, quad.BottomRightY);
        edges[2] = Distance(quad.BottomRightX, quad.BottomRightY, quad.BottomLeftX, quad.BottomLeftY);
        edges[3] = Distance(quad.BottomLeftX, quad.BottomLeftY, quad.TopLeftX, quad.TopLeftY);

        float maxEdge = edges.Max();
        float minEdge = edges.Min();
        float ratio = maxEdge / minEdge;

        if (ratio > 5)
        {
            System.Diagnostics.Debug.WriteLine($"[Validation] Edge ratio too large: {ratio:F2}");
            return false;
        }

        System.Diagnostics.Debug.WriteLine($"[Validation] All checks passed (convex, angles OK, ratio={ratio:F2})");
        return true;
    }

    /// <summary>
    /// 检查四边形是否为凸四边形（使用叉积）
    /// </summary>
    private static bool IsConvex(QuadrilateralPoints quad)
    {
        // 按顺序取4个点
        var points = new (float x, float y)[]
        {
            (quad.TopLeftX, quad.TopLeftY),
            (quad.TopRightX, quad.TopRightY),
            (quad.BottomRightX, quad.BottomRightY),
            (quad.BottomLeftX, quad.BottomLeftY)
        };

        // 计算每个顶点的叉积符号
        bool? firstSign = null;
        for (int i = 0; i < 4; i++)
        {
            var p1 = points[i];
            var p2 = points[(i + 1) % 4];
            var p3 = points[(i + 2) % 4];

            // 向量 p1->p2 和 p2->p3 的叉积
            float cross = (p2.x - p1.x) * (p3.y - p2.y) - (p2.y - p1.y) * (p3.x - p2.x);

            if (Math.Abs(cross) < 1e-6) continue; // 忽略共线

            bool currentSign = cross > 0;
            if (firstSign == null)
            {
                firstSign = currentSign;
            }
            else if (firstSign != currentSign)
            {
                return false; // 符号不一致，凹四边形
            }
        }

        return true;
    }

    /// <summary>
    /// 计算四边形的4个内角（度数）
    /// </summary>
    private static float[] GetInternalAngles(QuadrilateralPoints quad)
    {
        var points = new (float x, float y)[]
        {
            (quad.TopLeftX, quad.TopLeftY),
            (quad.TopRightX, quad.TopRightY),
            (quad.BottomRightX, quad.BottomRightY),
            (quad.BottomLeftX, quad.BottomLeftY)
        };

        float[] angles = new float[4];
        for (int i = 0; i < 4; i++)
        {
            var p1 = points[(i + 3) % 4]; // 前一个点
            var p2 = points[i];            // 当前点
            var p3 = points[(i + 1) % 4]; // 后一个点

            // 向量 p2->p1 和 p2->p3
            float v1x = p1.x - p2.x;
            float v1y = p1.y - p2.y;
            float v2x = p3.x - p2.x;
            float v2y = p3.y - p2.y;

            // 点积和模长
            float dot = v1x * v2x + v1y * v2y;
            float len1 = (float)Math.Sqrt(v1x * v1x + v1y * v1y);
            float len2 = (float)Math.Sqrt(v2x * v2x + v2y * v2y);

            // 夹角（弧度 → 度数）
            float cosAngle = dot / (len1 * len2);
            cosAngle = Math.Max(-1, Math.Min(1, cosAngle)); // 防止数值误差
            angles[i] = (float)(Math.Acos(cosAngle) * 180 / Math.PI);
        }

        return angles;
    }
}
