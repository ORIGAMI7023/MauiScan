using MauiScan.Models;
using MauiScan.Services;

namespace MauiScan.Views;

public partial class ScanPreviewPage : ContentPage
{
    private enum PreviewMode { Result, ROI, FourPoint }

    private byte[] _imageData;
    private byte[]? _originalPhoto;
    private IImageProcessingService? _imageProcessingService;
    private PreviewMode _currentMode;

    // ROI相关
    private double _roiX, _roiY, _roiW, _roiH;
    private double _imageRenderW, _imageRenderH;
    private double _imageOffsetX, _imageOffsetY;
    private double _imageScale;

    // 四点相关
    private (double x, double y)[] _cornerPoints = new (double, double)[4];
    private PanGestureRecognizer?[] _cornerGestures = new PanGestureRecognizer?[4];

    public event Action<byte[]>? Confirmed;
    public event Action? Retake;

    public ScanPreviewPage(
        byte[] imageData,
        bool isAutoSuccess,
        IImageProcessingService imageProcessingService,
        byte[]? originalPhoto = null)
    {
        InitializeComponent();

        _imageData = imageData;
        _imageProcessingService = imageProcessingService;
        _originalPhoto = originalPhoto;

        PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(imageData));

        if (isAutoSuccess)
        {
            SetMode(PreviewMode.Result);
        }
        else
        {
            SetMode(PreviewMode.ROI);
            InitializeROI();
        }

        // 添加手势识别器到8个ROI手柄
        InitializeROIGestures();

        // 添加手势识别器到4个四点角点
        InitializeFourPointGestures();

        PreviewImage.SizeChanged += OnPreviewImageSizeChanged;
    }

    private void OnPreviewImageSizeChanged(object? sender, EventArgs e)
    {
        CalculateImageTransform();
        if (_currentMode == PreviewMode.ROI)
        {
            UpdateROIHandles();
        }
        else if (_currentMode == PreviewMode.FourPoint)
        {
            UpdateFourPointLines();
        }
    }

    /// <summary>
    /// 计算图片在AspectFit下的实际渲染大小和偏移
    /// </summary>
    private void CalculateImageTransform()
    {
        if (PreviewImage.Source is not StreamImageSource streamSource)
            return;

        // 获取图片原始尺寸（从内存流）
        try
        {
            var stream = streamSource.Stream.Invoke(CancellationToken.None).Result as MemoryStream;
            if (stream == null) return;

            stream.Seek(0, SeekOrigin.Begin);
            var image = new System.IO.MemoryStream(stream.ToArray());
            // 使用MAUI的Image加载来获取尺寸（简单做法）
            // 实际上我们可以从SizeAllocated来推算
        }
        catch { }

        // 从界面实际尺寸反推
        double controlW = ImageLayer.Width;
        double controlH = ImageLayer.Height;

        if (controlW <= 0 || controlH <= 0)
            return;

        // 根据图片数据估算宽高比（这里需要从MAUI Image获取实际尺寸）
        // 为简化，我们假设图片已经加载，从Image的ActualWidth/Height推算
        // 如果不行，就从_imageData字节推算（需要解析JPEG头）

        // 先用一个简单方案：从控件尺寸和AspectFit推算
        // 实际上更好的方案是在加载图片时就记录宽高

        double nativeW = 1920, nativeH = 1080; // 默认值，应该从图片元数据获取

        try
        {
            // 尝试从JPEG字节头读取宽高
            (nativeW, nativeH) = ExtractImageDimensions(_imageData);
        }
        catch { }

        _imageScale = Math.Min(controlW / nativeW, controlH / nativeH);
        _imageRenderW = nativeW * _imageScale;
        _imageRenderH = nativeH * _imageScale;
        _imageOffsetX = (controlW - _imageRenderW) / 2;
        _imageOffsetY = (controlH - _imageRenderH) / 2;
    }

    /// <summary>
    /// 从JPEG字节数据中提取宽高
    /// </summary>
    private (double width, double height) ExtractImageDimensions(byte[] imageData)
    {
        // JPEG SOF0标记 (0xFFC0) 之后的内容包含宽高
        for (int i = 0; i < imageData.Length - 8; i++)
        {
            if (imageData[i] == 0xFF && (imageData[i + 1] == 0xC0 || imageData[i + 1] == 0xC2))
            {
                int height = (imageData[i + 5] << 8) | imageData[i + 6];
                int width = (imageData[i + 7] << 8) | imageData[i + 8];
                return (width, height);
            }
        }
        return (1920, 1080);
    }

    /// <summary>
    /// 屏幕坐标转图片像素坐标
    /// </summary>
    private (double x, double y) ScreenToImagePixel(double screenX, double screenY)
    {
        double pixelX = (screenX - _imageOffsetX) / _imageScale;
        double pixelY = (screenY - _imageOffsetY) / _imageScale;
        return (pixelX, pixelY);
    }

    /// <summary>
    /// 图片像素坐标转屏幕坐标
    /// </summary>
    private (double x, double y) ImagePixelToScreen(double pixelX, double pixelY)
    {
        double screenX = pixelX * _imageScale + _imageOffsetX;
        double screenY = pixelY * _imageScale + _imageOffsetY;
        return (screenX, screenY);
    }

    private void SetMode(PreviewMode mode)
    {
        _currentMode = mode;

        ResultToolbar.IsVisible = (mode == PreviewMode.Result);
        ROIToolbar.IsVisible = (mode == PreviewMode.ROI);
        FourPointToolbar.IsVisible = (mode == PreviewMode.FourPoint);

        ROILayer.IsVisible = (mode == PreviewMode.ROI);
        FourPointLayer.IsVisible = (mode == PreviewMode.FourPoint);
    }

    /// <summary>
    /// 初始化ROI矩形框
    /// </summary>
    private void InitializeROI()
    {
        CalculateImageTransform();

        _roiW = _imageRenderW * 0.8;
        _roiH = _roiW / 1.5; // 3:2比例
        _roiX = _imageOffsetX + (_imageRenderW - _roiW) / 2;
        _roiY = _imageOffsetY + (_imageRenderH - _roiH) / 2;

        UpdateROIHandles();
    }

    /// <summary>
    /// 初始化四点角点（初始放在当前ROI矩形的四角）
    /// </summary>
    private void InitializeFourPointCorners()
    {
        var (tl_x, tl_y) = ScreenToImagePixel(_roiX, _roiY);
        var (tr_x, tr_y) = ScreenToImagePixel(_roiX + _roiW, _roiY);
        var (bl_x, bl_y) = ScreenToImagePixel(_roiX, _roiY + _roiH);
        var (br_x, br_y) = ScreenToImagePixel(_roiX + _roiW, _roiY + _roiH);

        _cornerPoints[0] = (tl_x, tl_y); // TL
        _cornerPoints[1] = (tr_x, tr_y); // TR
        _cornerPoints[2] = (br_x, br_y); // BR
        _cornerPoints[3] = (bl_x, bl_y); // BL

        UpdateFourPointLines();
    }

    /// <summary>
    /// 更新ROI手柄位置
    /// </summary>
    private void UpdateROIHandles()
    {
        // 四个遮罩
        AbsoluteLayout.SetLayoutBounds(MaskTop, new Rect(_imageOffsetX, _imageOffsetY, _imageRenderW, _roiY - _imageOffsetY));
        AbsoluteLayout.SetLayoutBounds(MaskBottom, new Rect(_imageOffsetX, _roiY + _roiH, _imageRenderW, ImageLayer.Height - (_roiY + _roiH)));
        AbsoluteLayout.SetLayoutBounds(MaskLeft, new Rect(_imageOffsetX, _roiY, _roiX - _imageOffsetX, _roiH));
        AbsoluteLayout.SetLayoutBounds(MaskRight, new Rect(_roiX + _roiW, _roiY, (ImageLayer.Width - (_roiX + _roiW)), _roiH));

        // 四个角手柄
        UpdateHandle(HandleTL, _roiX - 10, _roiY - 10);
        UpdateHandle(HandleTR, _roiX + _roiW - 10, _roiY - 10);
        UpdateHandle(HandleBL, _roiX - 10, _roiY + _roiH - 10);
        UpdateHandle(HandleBR, _roiX + _roiW - 10, _roiY + _roiH - 10);

        // 四个边中点手柄
        UpdateHandle(HandleT, _roiX + _roiW / 2 - 10, _roiY - 10);
        UpdateHandle(HandleB, _roiX + _roiW / 2 - 10, _roiY + _roiH - 10);
        UpdateHandle(HandleL, _roiX - 10, _roiY + _roiH / 2 - 10);
        UpdateHandle(HandleR, _roiX + _roiW - 10, _roiY + _roiH / 2 - 10);
    }

    private void UpdateHandle(BoxView handle, double x, double y)
    {
        AbsoluteLayout.SetLayoutBounds(handle, new Rect(x, y, 20, 20));
    }

    /// <summary>
    /// 更新四点连线和点位置
    /// </summary>
    private void UpdateFourPointLines()
    {
        var (tl_sx, tl_sy) = ImagePixelToScreen(_cornerPoints[0].x, _cornerPoints[0].y);
        var (tr_sx, tr_sy) = ImagePixelToScreen(_cornerPoints[1].x, _cornerPoints[1].y);
        var (br_sx, br_sy) = ImagePixelToScreen(_cornerPoints[2].x, _cornerPoints[2].y);
        var (bl_sx, bl_sy) = ImagePixelToScreen(_cornerPoints[3].x, _cornerPoints[3].y);

        // 绘制四条线
        DrawLine(LineT, tl_sx, tl_sy, tr_sx, tr_sy);
        DrawLine(LineR, tr_sx, tr_sy, br_sx, br_sy);
        DrawLine(LineB, br_sx, br_sy, bl_sx, bl_sy);
        DrawLine(LineL, bl_sx, bl_sy, tl_sx, tl_sy);

        // 更新点位置
        AbsoluteLayout.SetLayoutBounds(PointTL, new Rect(tl_sx - 15, tl_sy - 15, 30, 30));
        AbsoluteLayout.SetLayoutBounds(PointTR, new Rect(tr_sx - 15, tr_sy - 15, 30, 30));
        AbsoluteLayout.SetLayoutBounds(PointBR, new Rect(br_sx - 15, br_sy - 15, 30, 30));
        AbsoluteLayout.SetLayoutBounds(PointBL, new Rect(bl_sx - 15, bl_sy - 15, 30, 30));
    }

    private void DrawLine(BoxView line, double x1, double y1, double x2, double y2)
    {
        double dx = x2 - x1;
        double dy = y2 - y1;
        double length = Math.Sqrt(dx * dx + dy * dy);

        line.WidthRequest = length;
        line.Rotation = Math.Atan2(dy, dx) * 180 / Math.PI;

        double centerX = (x1 + x2) / 2 - length / 2;
        double centerY = (y1 + y2) / 2 - 1;

        AbsoluteLayout.SetLayoutBounds(line, new Rect(centerX, centerY, length, 2));
    }

    private void InitializeROIGestures()
    {
        var handles = new[] { HandleTL, HandleTR, HandleBL, HandleBR, HandleT, HandleB, HandleL, HandleR };
        for (int i = 0; i < handles.Length; i++)
        {
            var gesture = new PanGestureRecognizer();
            gesture.PanUpdated += (s, e) => OnROIHandlePan(i, e);
            handles[i].GestureRecognizers.Add(gesture);
        }
    }

    private void InitializeFourPointGestures()
    {
        var points = new[] { PointTL, PointTR, PointBR, PointBL };
        for (int i = 0; i < points.Length; i++)
        {
            var gesture = new PanGestureRecognizer();
            int index = i; // 闭包捕获
            gesture.PanUpdated += (s, e) => OnFourPointPan(index, e);
            points[i].GestureRecognizers.Add(gesture);
            _cornerGestures[i] = gesture;
        }
    }

    private void OnROIHandlePan(int handleIndex, PanUpdatedEventArgs e)
    {
        if (e.StatusType != GestureStatus.Running)
            return;

        double deltaX = e.TotalX;
        double deltaY = e.TotalY;

        // 简化处理：所有手柄都可以调整框的大小
        switch (handleIndex)
        {
            case 0: // TL
                _roiX += deltaX;
                _roiY += deltaY;
                _roiW -= deltaX;
                _roiH -= deltaY;
                break;
            case 1: // TR
                _roiY += deltaY;
                _roiW += deltaX;
                _roiH -= deltaY;
                break;
            case 2: // BL
                _roiX += deltaX;
                _roiW -= deltaX;
                _roiH += deltaY;
                break;
            case 3: // BR
                _roiW += deltaX;
                _roiH += deltaY;
                break;
            case 4: // T（上边中点）
                _roiY += deltaY;
                _roiH -= deltaY;
                break;
            case 5: // B（下边中点）
                _roiH += deltaY;
                break;
            case 6: // L（左边中点）
                _roiX += deltaX;
                _roiW -= deltaX;
                break;
            case 7: // R（右边中点）
                _roiW += deltaX;
                break;
        }

        // 约束框的最小尺寸
        if (_roiW < 100) _roiW = 100;
        if (_roiH < 100) _roiH = 100;

        UpdateROIHandles();
    }

    private void OnFourPointPan(int pointIndex, PanUpdatedEventArgs e)
    {
        if (e.StatusType != GestureStatus.Running)
            return;

        double deltaX = e.TotalX;
        double deltaY = e.TotalY;

        var (pixelX, pixelY) = ScreenToImagePixel(_cornerPoints[pointIndex].x + deltaX, _cornerPoints[pointIndex].y + deltaY);
        _cornerPoints[pointIndex] = (pixelX, pixelY);

        UpdateFourPointLines();
    }

    private async void OnRotateLeftClicked(object sender, EventArgs e)
    {
        await RotateImageAsync(-90);
    }

    private async void OnRotateRightClicked(object sender, EventArgs e)
    {
        await RotateImageAsync(90);
    }

    private async Task RotateImageAsync(int degrees)
    {
        try
        {
            var rotated = await Task.Run(() => RotateJpeg90(_imageData, degrees));
            if (rotated != null)
            {
                _imageData = rotated;
                PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(rotated));
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"旋转失败: {ex.Message}", "确定");
        }
    }

    private byte[]? RotateJpeg90(byte[] imageBytes, int degrees)
    {
#if ANDROID
        using var bitmap = Android.Graphics.BitmapFactory.DecodeByteArray(imageBytes, 0, imageBytes.Length);
        if (bitmap == null) return null;

        var matrix = new Android.Graphics.Matrix();
        matrix.PostRotate(degrees);

        using var rotatedBitmap = Android.Graphics.Bitmap.CreateBitmap(
            bitmap, 0, 0, bitmap.Width, bitmap.Height, matrix, true);

        using var stream = new MemoryStream();
        rotatedBitmap.Compress(Android.Graphics.Bitmap.CompressFormat.Jpeg, 90, stream);
        return stream.ToArray();
#else
        return imageBytes;
#endif
    }

    private async void OnConfirmClicked(object sender, EventArgs e)
    {
        var data = _imageData;
        await Navigation.PopAsync();
        Confirmed?.Invoke(data);
    }

    private async void OnRetakeClicked(object sender, EventArgs e)
    {
        await Navigation.PopAsync();
        Retake?.Invoke();
    }

    private void OnSwitchToFourPointClicked(object sender, EventArgs e)
    {
        if (_currentMode == PreviewMode.ROI)
        {
            InitializeFourPointCorners();
            SetMode(PreviewMode.FourPoint);
        }
    }

    private void OnSwitchToROIClicked(object sender, EventArgs e)
    {
        if (_currentMode == PreviewMode.FourPoint)
        {
            SetMode(PreviewMode.ROI);
        }
    }

    private async void OnROIDetectClicked(object sender, EventArgs e)
    {
        if (_imageProcessingService == null || _originalPhoto == null)
            return;

        try
        {
            // 显示加载指示器
            await MainThread.InvokeOnMainThreadAsync(() =>
            {
                // 这里应该显示加载动画，但预览页没有，暂时用DisplayAlert提示
            });

            // 将ROI屏幕坐标转为像素坐标
            var (roi_x, roi_y) = ScreenToImagePixel(_roiX, _roiY);
            var (roi_w, roi_h) = ScreenToImagePixel(_roiX + _roiW, _roiY + _roiH);
            roi_w -= roi_x;
            roi_h -= roi_y;

            // 裁剪ROI区域
            byte[] roiBytes = CropImageROI(_originalPhoto, (int)roi_x, (int)roi_y, (int)roi_w, (int)roi_h);

            // 在ROI区域内尝试识别
            var result = await _imageProcessingService.ProcessScanAsync(roiBytes, false);

            if (result.IsSuccess)
            {
                // 成功：切换到结果预览模式
                _imageData = result.ImageData;
                PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(result.ImageData));
                SetMode(PreviewMode.Result);
            }
            else
            {
                await DisplayAlert("未识别到", "未能在选定区域识别到边框，请调整范围或切换到四点模式手动标注", "确定");
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"识别失败: {ex.Message}", "确定");
        }
    }

    private async void OnFourPointApplyClicked(object sender, EventArgs e)
    {
        if (_imageProcessingService == null || _originalPhoto == null)
            return;

        try
        {
            // 将4个角点从像素坐标转为NativeImageProcessingService.QuadPoints
            var quad = new NativeImageProcessingService.QuadPoints
            {
                TopLeftX = (float)_cornerPoints[0].x,
                TopLeftY = (float)_cornerPoints[0].y,
                TopRightX = (float)_cornerPoints[1].x,
                TopRightY = (float)_cornerPoints[1].y,
                BottomRightX = (float)_cornerPoints[2].x,
                BottomRightY = (float)_cornerPoints[2].y,
                BottomLeftX = (float)_cornerPoints[3].x,
                BottomLeftY = (float)_cornerPoints[3].y,
            };

            // 获取NativeImageProcessingService实例来调用ApplyPerspectiveTransform
            var nativeService = _imageProcessingService as NativeImageProcessingService;
            if (nativeService == null)
            {
                await DisplayAlert("错误", "图像处理服务不可用", "确定");
                return;
            }

            var result = await Task.Run(() => nativeService.ApplyPerspectiveTransform(_originalPhoto, quad));

            if (result.IsSuccess)
            {
                _imageData = result.ImageData;
                PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(result.ImageData));
                SetMode(PreviewMode.Result);
            }
            else
            {
                await DisplayAlert("错误", "透视变换失败", "确定");
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"裁切失败: {ex.Message}", "确定");
        }
    }

    /// <summary>
    /// 裁剪图片ROI区域
    /// </summary>
    private byte[] CropImageROI(byte[] imageData, int x, int y, int width, int height)
    {
#if ANDROID
        using var bitmap = Android.Graphics.BitmapFactory.DecodeByteArray(imageData, 0, imageData.Length);
        if (bitmap == null) throw new Exception("无法解码图片");

        // 确保裁剪区域在图片范围内
        x = Math.Max(0, x);
        y = Math.Max(0, y);
        width = Math.Min(width, bitmap.Width - x);
        height = Math.Min(height, bitmap.Height - y);

        using var croppedBitmap = Android.Graphics.Bitmap.CreateBitmap(bitmap, x, y, width, height);
        using var stream = new MemoryStream();
        croppedBitmap.Compress(Android.Graphics.Bitmap.CompressFormat.Jpeg, 95, stream);
        return stream.ToArray();
#else
        // 其他平台暂不实现
        return imageData;
#endif
    }
}
