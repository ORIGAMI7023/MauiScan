using MauiScan.Models;
using MauiScan.Services;
using Microsoft.Maui.Layouts;

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

    // 手势拖拽的起始位置（用于累积delta）
    private double _roiHandleDragStartX, _roiHandleDragStartY;
    private double _roiDragStartRoiX, _roiDragStartRoiY, _roiDragStartRoiW, _roiDragStartRoiH;
    private double _cornerPointDragStartX, _cornerPointDragStartY;
    private int _draggingHandleIndex = -1;
    private int _draggingCornerIndex = -1;

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
        }

        // 添加手势识别器到8个ROI手柄
        InitializeROIGestures();

        // 添加手势识别器到4个四点角点
        InitializeFourPointGestures();

        PreviewImage.SizeChanged += OnPreviewImageSizeChanged;
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();

        // 在页面显示时初始化ROI（此时布局已完成）
        if (_currentMode == PreviewMode.ROI)
        {
            CalculateImageTransform();
            InitializeROI();
        }
    }

    private void OnDebugEnableROIClicked(object sender, EventArgs e)
    {
        SetMode(PreviewMode.ROI);
        CalculateImageTransform();
        InitializeROI();
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
        // 获取控件尺寸
        double controlW = ImageLayer.Width;
        double controlH = ImageLayer.Height;

        if (controlW <= 0 || controlH <= 0)
            return;

        // 从JPEG字节数据中提取宽高
        double nativeW = 1920, nativeH = 1080; // 默认值

        try
        {
            (nativeW, nativeH) = ExtractImageDimensions(_imageData);
        }
        catch { }

        // 计算AspectFit的缩放比例和渲染尺寸
        _imageScale = Math.Min(controlW / nativeW, controlH / nativeH);
        _imageRenderW = nativeW * _imageScale;
        _imageRenderH = nativeH * _imageScale;
        _imageOffsetX = (controlW - _imageRenderW) / 2;
        _imageOffsetY = (controlH - _imageRenderH) / 2;

        System.Diagnostics.Debug.WriteLine($"Image Transform: controlW={controlW}, controlH={controlH}, nativeW={nativeW}, nativeH={nativeH}, scale={_imageScale}");
        System.Diagnostics.Debug.WriteLine($"Render: w={_imageRenderW}, h={_imageRenderH}, offsetX={_imageOffsetX}, offsetY={_imageOffsetY}");
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
        // 确保已经计算过图片变换
        if (_imageRenderW <= 0)
        {
            CalculateImageTransform();
        }

        _roiW = _imageRenderW * 0.8;
        _roiH = _roiW / 1.5; // 3:2比例
        _roiX = _imageOffsetX + (_imageRenderW - _roiW) / 2;
        _roiY = _imageOffsetY + (_imageRenderH - _roiH) / 2;

        System.Diagnostics.Debug.WriteLine($"ROI Initialize: roiW={_roiW}, roiH={_roiH}, roiX={_roiX}, roiY={_roiY}");

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
        AbsoluteLayout.SetLayoutFlags(MaskTop, AbsoluteLayoutFlags.None);

        AbsoluteLayout.SetLayoutBounds(MaskBottom, new Rect(_imageOffsetX, _roiY + _roiH, _imageRenderW, ImageLayer.Height - (_roiY + _roiH)));
        AbsoluteLayout.SetLayoutFlags(MaskBottom, AbsoluteLayoutFlags.None);

        AbsoluteLayout.SetLayoutBounds(MaskLeft, new Rect(_imageOffsetX, _roiY, _roiX - _imageOffsetX, _roiH));
        AbsoluteLayout.SetLayoutFlags(MaskLeft, AbsoluteLayoutFlags.None);

        AbsoluteLayout.SetLayoutBounds(MaskRight, new Rect(_roiX + _roiW, _roiY, (ImageLayer.Width - (_roiX + _roiW)), _roiH));
        AbsoluteLayout.SetLayoutFlags(MaskRight, AbsoluteLayoutFlags.None);

        // ROI边框（4条白色线）
        AbsoluteLayout.SetLayoutBounds(BorderTop, new Rect(_roiX, _roiY, _roiW, 2));
        AbsoluteLayout.SetLayoutFlags(BorderTop, AbsoluteLayoutFlags.None);

        AbsoluteLayout.SetLayoutBounds(BorderBottom, new Rect(_roiX, _roiY + _roiH - 2, _roiW, 2));
        AbsoluteLayout.SetLayoutFlags(BorderBottom, AbsoluteLayoutFlags.None);

        AbsoluteLayout.SetLayoutBounds(BorderLeft, new Rect(_roiX, _roiY, 2, _roiH));
        AbsoluteLayout.SetLayoutFlags(BorderLeft, AbsoluteLayoutFlags.None);

        AbsoluteLayout.SetLayoutBounds(BorderRight, new Rect(_roiX + _roiW - 2, _roiY, 2, _roiH));
        AbsoluteLayout.SetLayoutFlags(BorderRight, AbsoluteLayoutFlags.None);

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

        System.Diagnostics.Debug.WriteLine($"UpdateROIHandles: roiX={_roiX}, roiY={_roiY}, roiW={_roiW}, roiH={_roiH}");
    }

    private void UpdateHandle(BoxView handle, double x, double y)
    {
        AbsoluteLayout.SetLayoutBounds(handle, new Rect(x, y, 20, 20));
        AbsoluteLayout.SetLayoutFlags(handle, AbsoluteLayoutFlags.None);
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
        AbsoluteLayout.SetLayoutFlags(PointTL, AbsoluteLayoutFlags.None);

        AbsoluteLayout.SetLayoutBounds(PointTR, new Rect(tr_sx - 15, tr_sy - 15, 30, 30));
        AbsoluteLayout.SetLayoutFlags(PointTR, AbsoluteLayoutFlags.None);

        AbsoluteLayout.SetLayoutBounds(PointBR, new Rect(br_sx - 15, br_sy - 15, 30, 30));
        AbsoluteLayout.SetLayoutFlags(PointBR, AbsoluteLayoutFlags.None);

        AbsoluteLayout.SetLayoutBounds(PointBL, new Rect(bl_sx - 15, bl_sy - 15, 30, 30));
        AbsoluteLayout.SetLayoutFlags(PointBL, AbsoluteLayoutFlags.None);
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
        AbsoluteLayout.SetLayoutFlags(line, AbsoluteLayoutFlags.None);
    }

    private void InitializeROIGestures()
    {
        var handles = new[] { HandleTL, HandleTR, HandleBL, HandleBR, HandleT, HandleB, HandleL, HandleR };
        for (int i = 0; i < handles.Length; i++)
        {
            var gesture = new PanGestureRecognizer();
            int index = i; // closure capture by value
            gesture.PanUpdated += (s, e) => OnROIHandlePan(index, e);
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
        switch (e.StatusType)
        {
            case GestureStatus.Started:
                _draggingHandleIndex = handleIndex;
                // 保存拖拽开始时的完整ROI状态
                _roiDragStartRoiX = _roiX;
                _roiDragStartRoiY = _roiY;
                _roiDragStartRoiW = _roiW;
                _roiDragStartRoiH = _roiH;
                break;

            case GestureStatus.Running:
                if (_draggingHandleIndex != handleIndex) return;

                double deltaX = e.TotalX;
                double deltaY = e.TotalY;

                // 从起始状态恢复，然后应用delta
                _roiX = _roiDragStartRoiX;
                _roiY = _roiDragStartRoiY;
                _roiW = _roiDragStartRoiW;
                _roiH = _roiDragStartRoiH;
                double newRoiW = _roiW;
                double newRoiH = _roiH;

                // 应用拖拽改变
                switch (handleIndex)
                {
                    case 0: // TL
                        _roiX += deltaX;
                        _roiY += deltaY;
                        newRoiW -= deltaX;
                        newRoiH -= deltaY;
                        break;
                    case 1: // TR
                        _roiY += deltaY;
                        newRoiW += deltaX;
                        newRoiH -= deltaY;
                        break;
                    case 2: // BL
                        _roiX += deltaX;
                        newRoiW -= deltaX;
                        newRoiH += deltaY;
                        break;
                    case 3: // BR
                        newRoiW += deltaX;
                        newRoiH += deltaY;
                        break;
                    case 4: // T
                        _roiY += deltaY;
                        newRoiH -= deltaY;
                        break;
                    case 5: // B
                        newRoiH += deltaY;
                        break;
                    case 6: // L
                        _roiX += deltaX;
                        newRoiW -= deltaX;
                        break;
                    case 7: // R
                        newRoiW += deltaX;
                        break;
                }

                // 约束最小尺寸
                if (newRoiW >= 100) _roiW = newRoiW;
                if (newRoiH >= 100) _roiH = newRoiH;

                // 批量更新UI，减少重绘
                MainThread.BeginInvokeOnMainThread(() => UpdateROIHandles());
                break;

            case GestureStatus.Completed:
            case GestureStatus.Canceled:
                _draggingHandleIndex = -1;
                break;
        }
    }

    private void OnFourPointPan(int pointIndex, PanUpdatedEventArgs e)
    {
        switch (e.StatusType)
        {
            case GestureStatus.Started:
                _draggingCornerIndex = pointIndex;
                _cornerPointDragStartX = _cornerPoints[pointIndex].x;
                _cornerPointDragStartY = _cornerPoints[pointIndex].y;
                break;

            case GestureStatus.Running:
                if (_draggingCornerIndex != pointIndex)
                    return;

                double deltaX = e.TotalX;
                double deltaY = e.TotalY;

                // 从屏幕坐标转换为像素坐标（当前屏幕位置 = 起始屏幕位置 + delta）
                var (startScreenX, startScreenY) = ImagePixelToScreen(_cornerPointDragStartX, _cornerPointDragStartY);
                var (currentScreenX, currentScreenY) = (startScreenX + deltaX, startScreenY + deltaY);
                var (pixelX, pixelY) = ScreenToImagePixel(currentScreenX, currentScreenY);

                _cornerPoints[pointIndex] = (pixelX, pixelY);
                MainThread.BeginInvokeOnMainThread(() => UpdateFourPointLines());
                break;

            case GestureStatus.Completed:
            case GestureStatus.Canceled:
                _draggingCornerIndex = -1;
                break;
        }
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
        {
            await DisplayAlert("错误", $"服务状态 - Service: {(_imageProcessingService?.GetType().Name ?? "null")}, Photo: {(_originalPhoto != null ? _originalPhoto.Length + " bytes" : "null")}", "确定");
            return;
        }

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
        {
            await DisplayAlert("错误", "服务不可用，无法进行裁切", "确定");
            return;
        }

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
