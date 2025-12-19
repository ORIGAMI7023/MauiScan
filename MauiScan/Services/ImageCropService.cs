using MauiScan.Models;
using System.Diagnostics;

namespace MauiScan.Services;

/// <summary>
/// 图像裁剪服务 - 支持四边形区域裁剪、向内缩小、坐标转换
/// </summary>
public class ImageCropService
{
    /// <summary>
    /// 裁剪四边形区域并向内缩小指定比例
    /// </summary>
    /// <param name="imageBytes">原始图像字节</param>
    /// <param name="quad">四边形顶点（原图坐标）</param>
    /// <param name="shrinkRatio">向内缩小比例（0-1），默认0.08表示向内缩8%</param>
    /// <returns>裁剪后的图像字节和裁剪区域信息，失败返回null</returns>
    public (byte[] CroppedBytes, CropRegion Region)? CropAndShrink(
        byte[] imageBytes,
        QuadrilateralPoints quad,
        double shrinkRatio = 0.08)
    {
        try
        {
#if ANDROID
            Debug.WriteLine($"[ImageCrop] Starting crop with shrink ratio: {shrinkRatio}");

            // 加载原始图像
            using var bitmap = Android.Graphics.BitmapFactory.DecodeByteArray(imageBytes, 0, imageBytes.Length);
            if (bitmap == null)
            {
                Debug.WriteLine("[ImageCrop] Failed to decode image");
                return null;
            }

            Debug.WriteLine($"[ImageCrop] Image size: {bitmap.Width}x{bitmap.Height}");

            // 向内缩小四边形
            var shrunkQuad = ShrinkQuad(quad, shrinkRatio);

            Debug.WriteLine($"[ImageCrop] Original quad TL: ({quad.TopLeft.X},{quad.TopLeft.Y})");
            Debug.WriteLine($"[ImageCrop] Shrunk quad TL: ({shrunkQuad.TopLeft.X},{shrunkQuad.TopLeft.Y})");

            // 计算目标矩形尺寸
            float width1 = Distance(shrunkQuad.TopLeft.X, shrunkQuad.TopLeft.Y, shrunkQuad.TopRight.X, shrunkQuad.TopRight.Y);
            float width2 = Distance(shrunkQuad.BottomLeft.X, shrunkQuad.BottomLeft.Y, shrunkQuad.BottomRight.X, shrunkQuad.BottomRight.Y);
            float height1 = Distance(shrunkQuad.TopLeft.X, shrunkQuad.TopLeft.Y, shrunkQuad.BottomLeft.X, shrunkQuad.BottomLeft.Y);
            float height2 = Distance(shrunkQuad.TopRight.X, shrunkQuad.TopRight.Y, shrunkQuad.BottomRight.X, shrunkQuad.BottomRight.Y);

            int dstWidth = (int)Math.Max(width1, width2);
            int dstHeight = (int)Math.Max(height1, height2);

            Debug.WriteLine($"[ImageCrop] Target size: {dstWidth}x{dstHeight}");

            // 源点（缩小后的四边形）
            float[] src = new float[]
            {
                shrunkQuad.TopLeft.X, shrunkQuad.TopLeft.Y,
                shrunkQuad.TopRight.X, shrunkQuad.TopRight.Y,
                shrunkQuad.BottomRight.X, shrunkQuad.BottomRight.Y,
                shrunkQuad.BottomLeft.X, shrunkQuad.BottomLeft.Y
            };

            // 目标点（矩形）
            float[] dst = new float[]
            {
                0, 0,
                dstWidth, 0,
                dstWidth, dstHeight,
                0, dstHeight
            };

            // 计算透视变换矩阵
            var matrix = new Android.Graphics.Matrix();
            if (!matrix.SetPolyToPoly(src, 0, dst, 0, 4))
            {
                Debug.WriteLine("[ImageCrop] Failed to compute perspective transform matrix");
                return null;
            }

            // 创建变换后的 Bitmap
            using var transformedBitmap = Android.Graphics.Bitmap.CreateBitmap(dstWidth, dstHeight, Android.Graphics.Bitmap.Config.Argb8888!);
            using var canvas = new Android.Graphics.Canvas(transformedBitmap);

            // 应用透视变换
            using var paint = new Android.Graphics.Paint { FilterBitmap = true, AntiAlias = true };
            canvas.DrawBitmap(bitmap, matrix, paint);

            // 转换为 JPEG 字节
            using var outputStream = new MemoryStream();
            transformedBitmap.Compress(Android.Graphics.Bitmap.CompressFormat.Jpeg!, 95, outputStream);

            var croppedBytes = outputStream.ToArray();

            Debug.WriteLine($"[ImageCrop] Cropped image size: {croppedBytes.Length / 1024.0:F1} KB");

            // 构造裁剪区域信息
            var cropRegion = new CropRegion
            {
                OriginalQuad = quad,
                ShrunkQuad = shrunkQuad,
                ShrinkRatio = shrinkRatio,
                CroppedSize = (dstWidth, dstHeight)
            };

            return (croppedBytes, cropRegion);
#else
            Debug.WriteLine("[ImageCrop] Not supported on non-Android platforms");
            return null;
#endif
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[ImageCrop] Error: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// 将裁剪后的相对坐标转换为原图绝对坐标
    /// </summary>
    /// <param name="relativeQuad">相对坐标（在裁剪后图像中的坐标）</param>
    /// <param name="cropRegion">裁剪区域信息</param>
    /// <returns>绝对坐标（在原图中的坐标）</returns>
    public QuadrilateralPoints TransformToAbsolute(
        QuadrilateralPoints relativeQuad,
        CropRegion cropRegion)
    {
        try
        {
#if ANDROID
            Debug.WriteLine($"[ImageCrop] Transforming relative to absolute");
            Debug.WriteLine($"[ImageCrop] Relative quad TL: ({relativeQuad.TopLeft.X},{relativeQuad.TopLeft.Y})");

            // 构造逆变换：从裁剪后的矩形坐标 → 原图中缩小后的四边形坐标
            var shrunkQuad = cropRegion.ShrunkQuad;
            var (dstWidth, dstHeight) = cropRegion.CroppedSize;

            // 源点（矩形，裁剪后图像）
            float[] src = new float[]
            {
                0, 0,
                dstWidth, 0,
                dstWidth, dstHeight,
                0, dstHeight
            };

            // 目标点（缩小后的四边形，原图坐标）
            float[] dst = new float[]
            {
                shrunkQuad.TopLeft.X, shrunkQuad.TopLeft.Y,
                shrunkQuad.TopRight.X, shrunkQuad.TopRight.Y,
                shrunkQuad.BottomRight.X, shrunkQuad.BottomRight.Y,
                shrunkQuad.BottomLeft.X, shrunkQuad.BottomLeft.Y
            };

            // 计算逆透视变换矩阵
            var inverseMatrix = new Android.Graphics.Matrix();
            if (!inverseMatrix.SetPolyToPoly(src, 0, dst, 0, 4))
            {
                Debug.WriteLine("[ImageCrop] Failed to compute inverse matrix, returning relative quad");
                return relativeQuad;
            }

            // 转换四个角点
            float[] points = new float[]
            {
                relativeQuad.TopLeft.X, relativeQuad.TopLeft.Y,
                relativeQuad.TopRight.X, relativeQuad.TopRight.Y,
                relativeQuad.BottomRight.X, relativeQuad.BottomRight.Y,
                relativeQuad.BottomLeft.X, relativeQuad.BottomLeft.Y
            };

            inverseMatrix.MapPoints(points);

            var absoluteQuad = new QuadrilateralPoints(
                new Point2D((int)points[0], (int)points[1]),
                new Point2D((int)points[2], (int)points[3]),
                new Point2D((int)points[4], (int)points[5]),
                new Point2D((int)points[6], (int)points[7])
            );

            Debug.WriteLine($"[ImageCrop] Absolute quad TL: ({absoluteQuad.TopLeft.X},{absoluteQuad.TopLeft.Y})");

            return absoluteQuad;
#else
            Debug.WriteLine("[ImageCrop] Transform not supported on non-Android platforms");
            return relativeQuad;
#endif
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[ImageCrop] Transform error: {ex.Message}");
            return relativeQuad;
        }
    }

    /// <summary>
    /// 计算四边形中心点
    /// </summary>
    private (float X, float Y) GetQuadCenter(QuadrilateralPoints quad)
    {
        float centerX = (quad.TopLeft.X + quad.TopRight.X + quad.BottomRight.X + quad.BottomLeft.X) / 4f;
        float centerY = (quad.TopLeft.Y + quad.TopRight.Y + quad.BottomRight.Y + quad.BottomLeft.Y) / 4f;
        return (centerX, centerY);
    }

    /// <summary>
    /// 向内缩小四边形
    /// </summary>
    /// <param name="quad">原始四边形</param>
    /// <param name="ratio">缩小比例（0-1），例如0.08表示向中心缩小8%</param>
    /// <returns>缩小后的四边形</returns>
    private QuadrilateralPoints ShrinkQuad(QuadrilateralPoints quad, double ratio)
    {
        var center = GetQuadCenter(quad);

        // 向中心移动指定比例
        Point2D ShrinkPoint(Point2D point)
        {
            float dx = center.X - point.X;
            float dy = center.Y - point.Y;

            int newX = (int)(point.X + dx * ratio);
            int newY = (int)(point.Y + dy * ratio);

            return new Point2D(newX, newY);
        }

        return new QuadrilateralPoints(
            ShrinkPoint(quad.TopLeft),
            ShrinkPoint(quad.TopRight),
            ShrinkPoint(quad.BottomRight),
            ShrinkPoint(quad.BottomLeft)
        );
    }

    /// <summary>
    /// 计算两点间距离
    /// </summary>
    private static float Distance(float x1, float y1, float x2, float y2)
    {
        float dx = x2 - x1;
        float dy = y2 - y1;
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }
}
