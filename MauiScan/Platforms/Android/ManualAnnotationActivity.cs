using Android.App;
using Android.Content;
using Android.Graphics;
using Android.OS;
using Android.Views;
using Android.Widget;
using MauiScan.Services;
using AndroidView = Android.Views.View;
using Paint = Android.Graphics.Paint;
using Color = Android.Graphics.Color;
using Path = Android.Graphics.Path;
using Button = Android.Widget.Button;

namespace MauiScan.Platforms.Android;

[Activity(Label = "手动标注文档边界", Theme = "@style/Maui.SplashTheme")]
public class ManualAnnotationActivity : Activity
{
    private ImageView? _imageView;
    private AnnotationViewOverlay? _overlayView;
    private Bitmap? _originalBitmap;
    private byte[]? _imageBytes;

    protected override void OnCreate(Bundle? savedInstanceState)
    {
        base.OnCreate(savedInstanceState);

        // 加载图片
        _imageBytes = Services.ManualAnnotationService.GetCurrentImageBytes();
        if (_imageBytes == null)
        {
            Finish();
            return;
        }

        _originalBitmap = BitmapFactory.DecodeByteArray(_imageBytes, 0, _imageBytes.Length);
        if (_originalBitmap == null)
        {
            Finish();
            return;
        }

        // 创建布局
        var layout = new RelativeLayout(this);

        // 按钮容器（需要先创建以获取 ID）
        var buttonContainerId = AndroidView.GenerateViewId();

        // 图片容器（包含图片和覆盖层）
        var imageContainer = new FrameLayout(this);
        var containerParams = new RelativeLayout.LayoutParams(
            ViewGroup.LayoutParams.MatchParent,
            ViewGroup.LayoutParams.MatchParent);
        containerParams.AddRule(LayoutRules.Above, buttonContainerId);
        imageContainer.LayoutParameters = containerParams;

        // 图片视图
        _imageView = new ImageView(this)
        {
            LayoutParameters = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent,
                ViewGroup.LayoutParams.MatchParent)
        };
        _imageView.SetScaleType(ImageView.ScaleType.FitCenter);
        _imageView.SetImageBitmap(_originalBitmap);
        imageContainer.AddView(_imageView);

        // 标注覆盖层
        _overlayView = new AnnotationViewOverlay(this, _originalBitmap.Width, _originalBitmap.Height)
        {
            LayoutParameters = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent,
                ViewGroup.LayoutParams.MatchParent)
        };
        imageContainer.AddView(_overlayView);

        layout.AddView(imageContainer);

        // 按钮容器
        var buttonContainer = new LinearLayout(this)
        {
            Id = buttonContainerId,
            Orientation = Orientation.Horizontal
        };
        var buttonContainerParams = new RelativeLayout.LayoutParams(
            ViewGroup.LayoutParams.MatchParent,
            ViewGroup.LayoutParams.WrapContent);
        buttonContainerParams.AddRule(LayoutRules.AlignParentBottom);
        buttonContainer.LayoutParameters = buttonContainerParams;
        buttonContainer.SetPadding(20, 20, 20, 20);

        // 取消按钮
        var cancelButton = new Button(this)
        {
            Text = "取消",
            LayoutParameters = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WrapContent, 1)
        };
        cancelButton.Click += (s, e) =>
        {
            Services.ManualAnnotationService.SetResult(null);
            Finish();
        };
        buttonContainer.AddView(cancelButton);

        // 间距
        var spacer = new Space(this)
        {
            LayoutParameters = new LinearLayout.LayoutParams(20, ViewGroup.LayoutParams.WrapContent)
        };
        buttonContainer.AddView(spacer);

        // 确认按钮
        var confirmButton = new Button(this)
        {
            Text = "确认",
            LayoutParameters = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WrapContent, 1)
        };
        confirmButton.Click += async (s, e) => await OnConfirmClickedAsync();
        buttonContainer.AddView(confirmButton);

        layout.AddView(buttonContainer);

        SetContentView(layout);
    }

    private async Task OnConfirmClickedAsync()
    {
        if (_overlayView == null || _imageBytes == null)
        {
            Services.ManualAnnotationService.SetResult(null);
            Finish();
            return;
        }

        try
        {
            // 获取用户标注的4个角点
            var corners = _overlayView.GetCorners();

            // 调用 native OpenCV 进行透视变换
            var imageProcessingService = new NativeImageProcessingService();
            var result = await Task.Run(() =>
            {
                // 创建 QuadPoints 结构
                var quadPoints = new NativeImageProcessingService.QuadPoints
                {
                    TopLeftX = corners[0],
                    TopLeftY = corners[1],
                    TopRightX = corners[2],
                    TopRightY = corners[3],
                    BottomRightX = corners[4],
                    BottomRightY = corners[5],
                    BottomLeftX = corners[6],
                    BottomLeftY = corners[7]
                };

                // 调用透视变换
                return imageProcessingService.ApplyPerspectiveTransform(_imageBytes, quadPoints, false);
            });

            if (result.IsSuccess && result.ImageData != null)
            {
                // 返回结果
                var annotationResult = new ManualAnnotationResult
                {
                    Success = true,
                    ProcessedImageData = result.ImageData,
                    Width = result.Width,
                    Height = result.Height,
                    Corners = corners
                };

                Services.ManualAnnotationService.SetResult(annotationResult);
            }
            else
            {
                Toast.MakeText(this, "透视变换失败", ToastLength.Short)?.Show();
                Services.ManualAnnotationService.SetResult(null);
            }

            Finish();
        }
        catch (Exception ex)
        {
            Toast.MakeText(this, $"处理失败: {ex.Message}", ToastLength.Short)?.Show();
            Services.ManualAnnotationService.SetResult(null);
            Finish();
        }
    }

    protected override void OnDestroy()
    {
        _originalBitmap?.Recycle();
        _originalBitmap = null;
        base.OnDestroy();
    }
}

/// <summary>
/// 标注视图覆盖层（绘制和处理4个角点）
/// </summary>
public class AnnotationViewOverlay : AndroidView
{
    private const float CornerRadius = 30f;
    private const float StrokeWidth = 3f;

    private readonly Paint _cornerPaint;
    private readonly Paint _linePaint;
    private readonly Paint _fillPaint;
    private readonly float[] _corners; // [topLeftX, topLeftY, topRightX, topRightY, bottomRightX, bottomRightY, bottomLeftX, bottomLeftY]
    private int _selectedCornerIndex = -1;
    private float _imageWidth;
    private float _imageHeight;

    public AnnotationViewOverlay(Context context, int imageWidth, int imageHeight) : base(context)
    {
        _imageWidth = imageWidth;
        _imageHeight = imageHeight;

        // 初始化4个角点（默认在图片的4个角）
        _corners = new[]
        {
            0f, 0f,                               // 左上
            imageWidth, 0f,                       // 右上
            imageWidth, imageHeight,              // 右下
            0f, imageHeight                       // 左下
        };

        // 角点画笔（蓝色圆圈）
        _cornerPaint = new Paint
        {
            Color = Color.Blue,
            AntiAlias = true
        };
        _cornerPaint.SetStyle(Paint.Style.Fill);

        // 线条画笔（蓝色边框）
        _linePaint = new Paint
        {
            Color = Color.Blue,
            StrokeWidth = StrokeWidth,
            AntiAlias = true
        };
        _linePaint.SetStyle(Paint.Style.Stroke);

        // 填充画笔（半透明蓝色）
        _fillPaint = new Paint
        {
            Color = Color.Argb(50, 0, 0, 255),
            AntiAlias = true
        };
        _fillPaint.SetStyle(Paint.Style.Fill);
    }

    protected override void OnDraw(Canvas? canvas)
    {
        base.OnDraw(canvas);

        if (canvas == null) return;

        // 计算缩放比例（图片显示尺寸 vs 实际尺寸）
        float scaleX = Width / _imageWidth;
        float scaleY = Height / _imageHeight;
        float scale = Math.Min(scaleX, scaleY);

        // 计算图片在视图中的偏移量
        float offsetX = (Width - _imageWidth * scale) / 2f;
        float offsetY = (Height - _imageHeight * scale) / 2f;

        // 转换角点坐标到屏幕坐标
        var screenCorners = new float[8];
        for (int i = 0; i < 4; i++)
        {
            screenCorners[i * 2] = _corners[i * 2] * scale + offsetX;
            screenCorners[i * 2 + 1] = _corners[i * 2 + 1] * scale + offsetY;
        }

        // 绘制四边形填充
        var path = new Path();
        path.MoveTo(screenCorners[0], screenCorners[1]);
        path.LineTo(screenCorners[2], screenCorners[3]);
        path.LineTo(screenCorners[4], screenCorners[5]);
        path.LineTo(screenCorners[6], screenCorners[7]);
        path.Close();
        canvas.DrawPath(path, _fillPaint);

        // 绘制四边形边框
        canvas.DrawPath(path, _linePaint);

        // 绘制4个角点
        for (int i = 0; i < 4; i++)
        {
            canvas.DrawCircle(screenCorners[i * 2], screenCorners[i * 2 + 1], CornerRadius, _cornerPaint);
        }
    }

    public override bool OnTouchEvent(MotionEvent? e)
    {
        if (e == null) return false;

        float scaleX = Width / _imageWidth;
        float scaleY = Height / _imageHeight;
        float scale = Math.Min(scaleX, scaleY);
        float offsetX = (Width - _imageWidth * scale) / 2f;
        float offsetY = (Height - _imageHeight * scale) / 2f;

        float touchX = e.GetX();
        float touchY = e.GetY();

        switch (e.Action)
        {
            case MotionEventActions.Down:
                // 检查是否点击了某个角点
                _selectedCornerIndex = FindNearestCorner(touchX, touchY, scale, offsetX, offsetY);
                break;

            case MotionEventActions.Move:
                if (_selectedCornerIndex >= 0)
                {
                    // 更新角点位置（转换回图片坐标）
                    _corners[_selectedCornerIndex * 2] = (touchX - offsetX) / scale;
                    _corners[_selectedCornerIndex * 2 + 1] = (touchY - offsetY) / scale;

                    // 限制在图片范围内
                    _corners[_selectedCornerIndex * 2] = Math.Clamp(_corners[_selectedCornerIndex * 2], 0, _imageWidth);
                    _corners[_selectedCornerIndex * 2 + 1] = Math.Clamp(_corners[_selectedCornerIndex * 2 + 1], 0, _imageHeight);

                    Invalidate();
                }
                break;

            case MotionEventActions.Up:
                _selectedCornerIndex = -1;
                break;
        }

        return true;
    }

    private int FindNearestCorner(float x, float y, float scale, float offsetX, float offsetY)
    {
        float minDistance = float.MaxValue;
        int nearestIndex = -1;

        for (int i = 0; i < 4; i++)
        {
            float cornerX = _corners[i * 2] * scale + offsetX;
            float cornerY = _corners[i * 2 + 1] * scale + offsetY;

            float distance = (float)Math.Sqrt(Math.Pow(x - cornerX, 2) + Math.Pow(y - cornerY, 2));

            if (distance < CornerRadius * 2 && distance < minDistance)
            {
                minDistance = distance;
                nearestIndex = i;
            }
        }

        return nearestIndex;
    }

    public float[] GetCorners() => _corners;
}
