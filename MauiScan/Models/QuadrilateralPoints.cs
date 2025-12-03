namespace MauiScan.Models;

/// <summary>
/// 简单的 2D 点结构
/// </summary>
public struct Point2D
{
    public int X { get; set; }
    public int Y { get; set; }

    public Point2D(int x, int y)
    {
        X = x;
        Y = y;
    }
}

/// <summary>
/// 四边形的四个顶点（按顺时针顺序：左上、右上、右下、左下）
/// </summary>
public class QuadrilateralPoints
{
    public Point2D TopLeft { get; set; }
    public Point2D TopRight { get; set; }
    public Point2D BottomRight { get; set; }
    public Point2D BottomLeft { get; set; }

    public QuadrilateralPoints(Point2D topLeft, Point2D topRight, Point2D bottomRight, Point2D bottomLeft)
    {
        TopLeft = topLeft;
        TopRight = topRight;
        BottomRight = bottomRight;
        BottomLeft = bottomLeft;
    }

    /// <summary>
    /// 从轮廓点集自动排序生成四边形顶点
    /// </summary>
    public static QuadrilateralPoints FromContour(Point2D[] points)
    {
        if (points.Length != 4)
            throw new ArgumentException("必须是4个点", nameof(points));

        // 按 y 坐标排序，前两个是上方点，后两个是下方点
        var sorted = points.OrderBy(p => p.Y).ToArray();

        var topPoints = sorted.Take(2).OrderBy(p => p.X).ToArray();
        var bottomPoints = sorted.Skip(2).OrderBy(p => p.X).ToArray();

        return new QuadrilateralPoints(
            topLeft: topPoints[0],
            topRight: topPoints[1],
            bottomRight: bottomPoints[1],
            bottomLeft: bottomPoints[0]
        );
    }
}
