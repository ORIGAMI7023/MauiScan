using MauiScan.Models;
using MauiScan.Services;
using Microsoft.Maui.Layouts;

#if IOS || MACCATALYST
using CoreGraphics;
using UIKit;
using Foundation;
#endif

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

    // 原图像素尺寸（可靠来源）
    private int _nativeImageW, _nativeImageH;

    // 四点相关
    private (double x, double y)[] _cornerPoints = new (double, double)[4];

    // 手势拖拽状态
    private int _draggingHandleIndex = -1;   // 当前拖动的ROI手柄索引（-1=空闲）
    private int _draggingCornerIndex = -1;   // 当前拖动的四点角点索引（-1=空闲）

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

        // 在背景层上绑定统一的Pan手势（避免在移动的子View上绑定导致TotalX/Y跳变）
        InitializeROILayerGesture();
        InitializeFourPointLayerGesture();

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
        // DEBUG模式下，如果没有原始照片，使用当前图片数据作为原图
        if (_originalPhoto == null)
            _originalPhoto = _imageData;

        // 重置尺寸缓存，强制重新解码
        _nativeImageW = 0;
        _nativeImageH = 0;

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

        // 获取原图像素尺寸（使用已缓存的值或重新解码）
        if (_nativeImageW <= 0 || _nativeImageH <= 0)
        {
            var sourceData = _originalPhoto ?? _imageData;
            (_nativeImageW, _nativeImageH) = DecodeImageDimensions(sourceData);
        }

        double nativeW = _nativeImageW;
        double nativeH = _nativeImageH;

        // 计算AspectFit的缩放比例和渲染尺寸
        _imageScale = Math.Min(controlW / nativeW, controlH / nativeH);
        _imageRenderW = nativeW * _imageScale;
        _imageRenderH = nativeH * _imageScale;
        _imageOffsetX = (controlW - _imageRenderW) / 2;
        _imageOffsetY = (controlH - _imageRenderH) / 2;

        System.Diagnostics.Debug.WriteLine($"[Coord] Image Transform: control=({controlW:F0}x{controlH:F0}), native=({nativeW}x{nativeH}), scale={_imageScale:F4}");
        System.Diagnostics.Debug.WriteLine($"[Coord] Render: size=({_imageRenderW:F0}x{_imageRenderH:F0}), offset=({_imageOffsetX:F0},{_imageOffsetY:F0})");
    }

    /// <summary>
    /// 可靠地获取图片像素尺寸（使用平台API解码header）
    /// </summary>
    private (int width, int height) DecodeImageDimensions(byte[] imageData)
    {
#if ANDROID
        var options = new Android.Graphics.BitmapFactory.Options { InJustDecodeBounds = true };
        Android.Graphics.BitmapFactory.DecodeByteArray(imageData, 0, imageData.Length, options);
        int w = options.OutWidth;
        int h = options.OutHeight;
        System.Diagnostics.Debug.WriteLine($"[Coord] DecodeImageDimensions (Android): {w}x{h}");
        return (w > 0 ? w : 1920, h > 0 ? h : 1080);
#elif IOS || MACCATALYST
        using var nsData = NSData.FromArray(imageData);
        using var image = UIImage.LoadFromData(nsData);
        if (image != null)
        {
            int w = (int)image.Size.Width;
            int h = (int)image.Size.Height;
            System.Diagnostics.Debug.WriteLine($"[Coord] DecodeImageDimensions (iOS): {w}x{h}");
            return (w > 0 ? w : 1920, h > 0 ? h : 1080);
        }
        return (1920, 1080);
#else
        // 回退：扫描JPEG SOF标记
        for (int i = 0; i < imageData.Length - 8; i++)
        {
            if (imageData[i] == 0xFF && (imageData[i + 1] == 0xC0 || imageData[i + 1] == 0xC2))
            {
                int height = (imageData[i + 5] << 8) | imageData[i + 6];
                int width = (imageData[i + 7] << 8) | imageData[i + 8];
                if (width > 0 && height > 0) return (width, height);
            }
        }
        return (1920, 1080);
#endif
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
        double layerW = ImageLayer.Width;
        double layerH = ImageLayer.Height;

        // 四个遮罩（clamp尺寸避免负值导致不可见）
        double maskTopH = Math.Max(0, _roiY);
        AbsoluteLayout.SetLayoutBounds(MaskTop, new Rect(0, 0, layerW, maskTopH));
        AbsoluteLayout.SetLayoutFlags(MaskTop, AbsoluteLayoutFlags.None);

        double maskBottomY = _roiY + _roiH;
        double maskBottomH = Math.Max(0, layerH - maskBottomY);
        AbsoluteLayout.SetLayoutBounds(MaskBottom, new Rect(0, maskBottomY, layerW, maskBottomH));
        AbsoluteLayout.SetLayoutFlags(MaskBottom, AbsoluteLayoutFlags.None);

        double maskLeftW = Math.Max(0, _roiX);
        AbsoluteLayout.SetLayoutBounds(MaskLeft, new Rect(0, _roiY, maskLeftW, _roiH));
        AbsoluteLayout.SetLayoutFlags(MaskLeft, AbsoluteLayoutFlags.None);

        double maskRightX = _roiX + _roiW;
        double maskRightW = Math.Max(0, layerW - maskRightX);
        AbsoluteLayout.SetLayoutBounds(MaskRight, new Rect(maskRightX, _roiY, maskRightW, _roiH));
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

    /// <summary>
    /// 在ROILayer背景层绑定Pan手势，避免在移动的手柄View上绑定导致Android TotalX/Y跳变。
    /// 同时绑定PointerGestureRecognizer以在Pressed时获取精确触摸坐标，用于命中检测。
    /// </summary>
    private void InitializeROILayerGesture()
    {
        // 每个手柄绑完整的 Pan 处理（Started+Running+Completed 全在手柄自身处理）
        // 用增量模式（每帧 delta = 本帧Total - 上帧Total）避免手柄移动导致的 TotalX/Y 跳变
        var handles = new[] { HandleTL, HandleTR, HandleBL, HandleBR, HandleT, HandleB, HandleL, HandleR };
        for (int i = 0; i < handles.Length; i++)
        {
            int index = i;
            handles[i].InputTransparent = false;
            double lastTotalX = 0, lastTotalY = 0;
            var handlePan = new PanGestureRecognizer();
            handlePan.PanUpdated += (s, args) =>
            {
                switch (args.StatusType)
                {
                    case GestureStatus.Started:
                        if (_draggingHandleIndex != -1) return;
                        _draggingHandleIndex = index;
                        lastTotalX = 0;
                        lastTotalY = 0;
                        System.Diagnostics.Debug.WriteLine($"[Handle Pan] Started index={index}");
                        break;
                    case GestureStatus.Running:
                        if (_draggingHandleIndex != index) return;
                        double dX = args.TotalX - lastTotalX;
                        double dY = args.TotalY - lastTotalY;
                        lastTotalX = args.TotalX;
                        lastTotalY = args.TotalY;
                        ApplyROIHandleDelta(index, dX, dY);
                        break;
                    case GestureStatus.Completed:
                    case GestureStatus.Canceled:
                        if (_draggingHandleIndex == index)
                            _draggingHandleIndex = -1;
                        break;
                }
            };
            handles[i].GestureRecognizers.Add(handlePan);
        }
    }

    private void InitializeFourPointLayerGesture()
    {
        // 每个角点绑完整的 Pan，增量模式
        var points = new View[] { PointTL, PointTR, PointBR, PointBL };
        for (int i = 0; i < points.Length; i++)
        {
            int index = i;
            points[i].InputTransparent = false;
            double lastTotalX = 0, lastTotalY = 0;
            var pointPan = new PanGestureRecognizer();
            pointPan.PanUpdated += (s, args) =>
            {
                switch (args.StatusType)
                {
                    case GestureStatus.Started:
                        if (_draggingCornerIndex != -1) return;
                        _draggingCornerIndex = index;
                        lastTotalX = 0;
                        lastTotalY = 0;
                        System.Diagnostics.Debug.WriteLine($"[Corner Pan] Started index={index}");
                        break;
                    case GestureStatus.Running:
                        if (_draggingCornerIndex != index) return;
                        double dX = args.TotalX - lastTotalX;
                        double dY = args.TotalY - lastTotalY;
                        lastTotalX = args.TotalX;
                        lastTotalY = args.TotalY;
                        var (sx, sy) = ImagePixelToScreen(_cornerPoints[index].x, _cornerPoints[index].y);
                        var (px, py) = ScreenToImagePixel(sx + dX, sy + dY);
                        _cornerPoints[index] = (px, py);
                        UpdateFourPointLines();
                        break;
                    case GestureStatus.Completed:
                    case GestureStatus.Canceled:
                        if (_draggingCornerIndex == index)
                            _draggingCornerIndex = -1;
                        break;
                }
            };
            points[i].GestureRecognizers.Add(pointPan);
        }
    }

    /// <summary>
    /// 根据增量 dX/dY 更新 ROI（增量模式，不依赖 TotalX/Y 的绝对值）
    /// </summary>
    private void ApplyROIHandleDelta(int handleIndex, double dX, double dY)
    {
        double newX = _roiX, newY = _roiY, newW = _roiW, newH = _roiH;

        switch (handleIndex)
        {
            case 0: newX += dX; newY += dY; newW -= dX; newH -= dY; break; // TL
            case 1: newY += dY; newW += dX; newH -= dY; break;             // TR
            case 2: newX += dX; newW -= dX; newH += dY; break;             // BL
            case 3: newW += dX; newH += dY; break;                         // BR
            case 4: newY += dY; newH -= dY; break;                         // T
            case 5: newH += dY; break;                                     // B
            case 6: newX += dX; newW -= dX; break;                         // L
            case 7: newW += dX; break;                                     // R
        }

        if (newW >= 50 && newH >= 50)
            { _roiX = newX; _roiY = newY; _roiW = newW; _roiH = newH; }
        else if (newW >= 50)
            { _roiX = newX; _roiW = newW; }
        else if (newH >= 50)
            { _roiY = newY; _roiH = newH; }

        UpdateROIHandles();
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
#elif IOS || MACCATALYST
        using var nsData = NSData.FromArray(imageBytes);
        using var image = UIImage.LoadFromData(nsData);
        if (image == null) return null;

        // CoreGraphics 的旋转角度是弧度，顺时针为正
        float radians = (float)(degrees * Math.PI / 180.0);

        // 计算旋转后的尺寸
        nfloat imgW = image.Size.Width;
        nfloat imgH = image.Size.Height;
        nfloat newWidth, newHeight;
        if (degrees == 90 || degrees == -90 || degrees == 270 || degrees == -270)
        {
            newWidth = imgH;
            newHeight = imgW;
        }
        else
        {
            newWidth = imgW;
            newHeight = imgH;
        }

        using var colorSpace = CGColorSpace.CreateDeviceRGB();
        using var context = new CGBitmapContext(IntPtr.Zero, (int)newWidth, (int)newHeight, 8, 0, colorSpace, CGImageAlphaInfo.PremultipliedLast);
        context.TranslateCTM(newWidth / 2, newHeight / 2);
        context.RotateCTM(radians);
        context.DrawImage(new CGRect(-imgW / 2, -imgH / 2, imgW, imgH), image.CGImage);

        using var resultCgImage = context.ToImage();
        using var resultImage = new UIImage(resultCgImage);
        using var resultNsData = resultImage.AsJPEG(0.9f);
        var bytes = new byte[resultNsData.Length];
        System.Runtime.InteropServices.Marshal.Copy(resultNsData.Bytes, bytes, 0, (int)resultNsData.Length);
        return bytes;
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
        // 如果没有原始照片，使用当前显示的图片（失败路径中 _imageData 就是原图）
        var sourcePhoto = _originalPhoto ?? _imageData;
        System.Diagnostics.Debug.WriteLine($"OnROIDetectClicked: service={_imageProcessingService != null}, photo={sourcePhoto != null}");

        if (_imageProcessingService == null || sourcePhoto == null)
        {
            await DisplayAlert("错误", "服务或原始照片不可用", "确定");
            return;
        }

        try
        {
            // 将ROI屏幕坐标转为像素坐标
            double pixelLeft = (_roiX - _imageOffsetX) / _imageScale;
            double pixelTop = (_roiY - _imageOffsetY) / _imageScale;
            double pixelRight = (_roiX + _roiW - _imageOffsetX) / _imageScale;
            double pixelBottom = (_roiY + _roiH - _imageOffsetY) / _imageScale;

            // 限制在图片范围内
            int cropX = Math.Max(0, (int)pixelLeft);
            int cropY = Math.Max(0, (int)pixelTop);
            int cropW = Math.Min(_nativeImageW, (int)pixelRight) - cropX;
            int cropH = Math.Min(_nativeImageH, (int)pixelBottom) - cropY;

            System.Diagnostics.Debug.WriteLine($"[ROI] Screen: x={_roiX:F0}, y={_roiY:F0}, w={_roiW:F0}, h={_roiH:F0}");
            System.Diagnostics.Debug.WriteLine($"[ROI] Pixel crop: x={cropX}, y={cropY}, w={cropW}, h={cropH} (image={_nativeImageW}x{_nativeImageH})");

            if (cropW <= 50 || cropH <= 50)
            {
                await DisplayAlert("范围太小", "请调大选择范围后重试", "确定");
                return;
            }

            // 裁剪ROI区域
            byte[] roiBytes = CropImageROI(sourcePhoto, cropX, cropY, cropW, cropH);

            // 第一步：在ROI区域内检测文档边界（不裁切）
            var bounds = await _imageProcessingService.DetectDocumentBoundsAsync(roiBytes, 0.05);

            if (bounds == null)
            {
                await DisplayAlert("未识别到", "未能在选定区域识别到文档边框，请调整范围或切换到四点模式手动标注", "确定");
                return;
            }

            System.Diagnostics.Debug.WriteLine($"[ROI] 检测到边界: TL=({bounds.TopLeft.X:F0},{bounds.TopLeft.Y:F0}) BR=({bounds.BottomRight.X:F0},{bounds.BottomRight.Y:F0})");

            // 第二步：使用检测到的边界进行透视变换
            var nativeService = _imageProcessingService as NativeImageProcessingService;
            if (nativeService == null)
            {
                await DisplayAlert("错误", "图像处理服务不可用", "确定");
                return;
            }

            var quad = new NativeImageProcessingService.QuadPoints
            {
                TopLeftX = (float)bounds.TopLeft.X,
                TopLeftY = (float)bounds.TopLeft.Y,
                TopRightX = (float)bounds.TopRight.X,
                TopRightY = (float)bounds.TopRight.Y,
                BottomRightX = (float)bounds.BottomRight.X,
                BottomRightY = (float)bounds.BottomRight.Y,
                BottomLeftX = (float)bounds.BottomLeft.X,
                BottomLeftY = (float)bounds.BottomLeft.Y,
            };

            var result = await Task.Run(() => nativeService.ApplyPerspectiveTransform(roiBytes, quad));

            if (result.IsSuccess)
            {
                _imageData = result.ImageData;
                PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(result.ImageData));
                SetMode(PreviewMode.Result);
            }
            else
            {
                await DisplayAlert("裁切失败", result.ErrorMessage ?? "透视变换失败", "确定");
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"识别失败: {ex.Message}", "确定");
        }
    }

    private async void OnFourPointApplyClicked(object sender, EventArgs e)
    {
        var sourcePhoto = _originalPhoto ?? _imageData;
        System.Diagnostics.Debug.WriteLine($"OnFourPointApplyClicked: service={_imageProcessingService != null}, photo={sourcePhoto != null}");

        if (_imageProcessingService == null || sourcePhoto == null)
        {
            await DisplayAlert("错误", "服务或原始照片不可用", "确定");
            return;
        }

        try
        {
            // _cornerPoints 存的是像素坐标，直接传给 native
            System.Diagnostics.Debug.WriteLine($"[4Point] TL=({_cornerPoints[0].x:F0},{_cornerPoints[0].y:F0}) TR=({_cornerPoints[1].x:F0},{_cornerPoints[1].y:F0})");
            System.Diagnostics.Debug.WriteLine($"[4Point] BR=({_cornerPoints[2].x:F0},{_cornerPoints[2].y:F0}) BL=({_cornerPoints[3].x:F0},{_cornerPoints[3].y:F0})");
            System.Diagnostics.Debug.WriteLine($"[4Point] Image size: {_nativeImageW}x{_nativeImageH}");

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

            var result = await Task.Run(() => nativeService.ApplyPerspectiveTransform(sourcePhoto, quad));

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
#elif IOS || MACCATALYST
        using var nsData = NSData.FromArray(imageData);
        using var image = UIImage.LoadFromData(nsData);
        if (image == null) throw new Exception("无法解码图片");

        // 确保裁剪区域在图片范围内
        x = Math.Max(0, x);
        y = Math.Max(0, y);
        width = Math.Min(width, (int)image.Size.Width - x);
        height = Math.Min(height, (int)image.Size.Height - y);

        // CoreGraphics 坐标系 Y 轴从下往上，需要翻转 Y
        nfloat cgY = image.Size.Height - y - height;

        using var cgImage = image.CGImage!.WithImageInRect(new CGRect(x, cgY, width, height));
        using var resultImage = new UIImage(cgImage!);
        using var resultNsData = resultImage.AsJPEG(0.95f);
        var bytes = new byte[resultNsData.Length];
        System.Runtime.InteropServices.Marshal.Copy(resultNsData.Bytes, bytes, 0, (int)resultNsData.Length);
        return bytes;
#else
        return imageData;
#endif
    }
}
