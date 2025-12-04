using MauiScan.Models;
using MauiScan.Services;

namespace MauiScan.Views;

public partial class ScanPage : ContentPage
{
    private readonly ICameraService _cameraService;
    private readonly IImageProcessingService _imageProcessingService;
    private readonly IClipboardService _clipboardService;
    private readonly IDragDropService? _dragDropService;

    private byte[]? _currentImageData;
    private int _currentRotation = 0;
    private string _debugLog = "调试日志:\n";

    public ScanPage(
        ICameraService cameraService,
        IImageProcessingService imageProcessingService,
        IClipboardService clipboardService,
        IDragDropService? dragDropService = null)
    {
        InitializeComponent();

        _cameraService = cameraService;
        _imageProcessingService = imageProcessingService;
        _clipboardService = clipboardService;
        _dragDropService = dragDropService;

        // 设置相机服务的日志回调
        if (_cameraService is Platforms.iOS.Services.CameraService ioscamera)
        {
            ioscamera.OnDebugLog += AddDebugLog;
        }

        // 添加长按手势用于拖放
        var longPressGesture = new TapGestureRecognizer();
        // 使用 PointerGestureRecognizer 来处理长按（MAUI 没有内置长按手势）
        // 改用 DragGestureRecognizer
        var dragGesture = new DragGestureRecognizer();
        dragGesture.DragStarting += OnDragStarting;
        PreviewImage.GestureRecognizers.Add(dragGesture);
    }

    private void OnDragStarting(object? sender, DragStartingEventArgs e)
    {
        if (_currentImageData == null || _dragDropService == null)
        {
            e.Cancel = true;
            return;
        }

        // 使用自定义拖放服务来支持跨应用拖放
        _ = StartCrossAppDragAsync();

        // 取消 MAUI 默认的拖放行为，使用我们自己的实现
        e.Cancel = true;
    }

    private async Task StartCrossAppDragAsync()
    {
        if (_currentImageData == null || _dragDropService == null)
            return;

        var result = await _dragDropService.StartDragImageAsync(PreviewImage, _currentImageData);
        if (result)
        {
            StatusLabel.Text = "拖动图片到其他应用...";
        }
    }

    private async void OnCaptureClicked(object sender, EventArgs e)
    {
        try
        {
            AddDebugLog("OnCaptureClicked called");
            StatusLabel.Text = "正在拍摄...";
            SetLoading(true);

            // 1. 拍照
            byte[]? photoBytes = null;
            try
            {
                AddDebugLog("Calling TakePhotoAsync...");
                photoBytes = await _cameraService.TakePhotoAsync();

                if (photoBytes == null)
                {
                    AddDebugLog("TakePhotoAsync returned null");
                    StatusLabel.Text = "拍摄已取消";
                    return;
                }

                AddDebugLog($"TakePhotoAsync returned {photoBytes.Length} bytes");
                StatusLabel.Text = $"收到 {photoBytes.Length} 字节";
            }
            catch (Exception ex)
            {
                AddDebugLog($"ERROR in TakePhotoAsync: {ex.GetType().Name}: {ex.Message}");
                StatusLabel.Text = $"ERROR: {ex.GetType().Name}";
                return;
            }

            StatusLabel.Text = "正在处理图像...";

            // 2. 处理图像（边缘检测 + 透视变换）
            var result = await _imageProcessingService.ProcessScanAsync(photoBytes!, false);

            if (!result.IsSuccess)
            {
                // 识别失败：清除预览，显示错误状态
                _currentImageData = null;
                PreviewImage.IsVisible = false;
                PlaceholderLabel.IsVisible = true;
                SaveButton.IsEnabled = false;
                RotateButtonsGrid.IsVisible = false;
                StatusLabel.Text = $"识别失败: {result.ErrorMessage ?? "未知错误"}";
                return;
            }

            // 3. 显示结果
            _currentImageData = result.ImageData;
            _currentRotation = 0;
            PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(result.ImageData));
            PreviewImage.IsVisible = true;
            PlaceholderLabel.IsVisible = false;
            SaveButton.IsEnabled = true;
            RotateButtonsGrid.IsVisible = true;

            // 4. 自动复制到剪贴板
            var copied = await _clipboardService.CopyImageToClipboardAsync(result.ImageData);
            if (copied)
            {
                StatusLabel.Text = $"✓ 扫描成功 ({result.Width}×{result.Height}) | 已复制到剪贴板";
            }
            else
            {
                StatusLabel.Text = $"✓ 扫描成功 ({result.Width}×{result.Height})";
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"发生异常: {ex.Message}", "确定");
            StatusLabel.Text = "发生错误";
        }
        finally
        {
            SetLoading(false);
        }
    }

    private async void OnSaveClicked(object sender, EventArgs e)
    {
        if (_currentImageData == null)
            return;

        try
        {
            var fileName = $"Scan_{DateTime.Now:yyyyMMdd_HHmmss}.jpg";

#if ANDROID
            // Android: 保存到系统相册
            await SaveToGalleryAndroidAsync(fileName, _currentImageData);
            StatusLabel.Text = $"✓ 已保存到相册: {fileName}";
#else
            // 其他平台: 保存到应用目录
            var filePath = Path.Combine(FileSystem.AppDataDirectory, fileName);
            await File.WriteAllBytesAsync(filePath, _currentImageData);
            StatusLabel.Text = $"✓ 已保存: {fileName}";
#endif
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"保存失败: {ex.Message}";
        }
    }

#if ANDROID
    private async Task<string> SaveToGalleryAndroidAsync(string fileName, byte[] imageData)
    {
        var context = Android.App.Application.Context;
        var contentResolver = context.ContentResolver;

        var contentValues = new Android.Content.ContentValues();
        contentValues.Put(Android.Provider.MediaStore.IMediaColumns.DisplayName, fileName);
        contentValues.Put(Android.Provider.MediaStore.IMediaColumns.MimeType, "image/jpeg");
        contentValues.Put(Android.Provider.MediaStore.IMediaColumns.RelativePath, "Pictures/MauiScan");

        var uri = contentResolver!.Insert(Android.Provider.MediaStore.Images.Media.ExternalContentUri!, contentValues);
        if (uri == null)
            throw new Exception("无法创建媒体文件");

        using var outputStream = contentResolver.OpenOutputStream(uri);
        if (outputStream == null)
            throw new Exception("无法打开输出流");

        await outputStream.WriteAsync(imageData, 0, imageData.Length);
        await outputStream.FlushAsync();

        return uri.ToString()!;
    }
#endif

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
        if (_currentImageData == null)
            return;

        try
        {
            SetLoading(true);
            StatusLabel.Text = "正在旋转...";

            _currentRotation = (_currentRotation + degrees + 360) % 360;

            var rotatedData = await Task.Run(() => RotateJpegBytes(_currentImageData, degrees));
            if (rotatedData != null)
            {
                _currentImageData = rotatedData;
                PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(rotatedData));

                // 更新剪贴板
                var copied = await _clipboardService.CopyImageToClipboardAsync(rotatedData);
                if (copied)
                {
                    StatusLabel.Text = $"已旋转 {(degrees > 0 ? "右" : "左")} 90° | 已复制到剪贴板";
                }
                else
                {
                    StatusLabel.Text = $"已旋转 {(degrees > 0 ? "右" : "左")} 90°";
                }
            }
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"旋转失败: {ex.Message}";
        }
        finally
        {
            SetLoading(false);
        }
    }

    private byte[]? RotateJpegBytes(byte[] imageBytes, int degrees)
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
        // 其他平台暂不支持
        return imageBytes;
#endif
    }

    private void SetLoading(bool isLoading)
    {
        LoadingIndicator.IsRunning = isLoading;
        LoadingIndicator.IsVisible = isLoading;
        CaptureButton.IsEnabled = !isLoading;
        SaveButton.IsEnabled = !isLoading && _currentImageData != null;
    }

    private void AddDebugLog(string message)
    {
        _debugLog += $"\n[{DateTime.Now:HH:mm:ss}] {message}";
        Console.WriteLine($"[DEBUG] {message}");
        MainThread.BeginInvokeOnMainThread(() =>
        {
            DebugLabel.Text = _debugLog;
            // 每次输出日志都自动复制到剪贴板
            try
            {
                Clipboard.Default.SetTextAsync(_debugLog);
            }
            catch { }
        });
    }

    private async void OnCopyLogClicked(object sender, EventArgs e)
    {
        try
        {
            await Clipboard.Default.SetTextAsync(_debugLog);
            await DisplayAlert("成功", "日志已复制到剪贴板", "确定");
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"复制失败: {ex.Message}", "确定");
        }
    }
}
