using MauiScan.Models;
using MauiScan.Services;
using MauiScan.Services.Sync;

namespace MauiScan.Views;

public partial class ScanPage : ContentPage
{
    private readonly ICameraService _cameraService;
    private readonly IImageProcessingService _imageProcessingService;
    private readonly IClipboardService _clipboardService;
    private readonly IDragDropService? _dragDropService;
    private readonly ScanSyncService _syncService;

    private byte[]? _currentImageData;
    private int _currentRotation = 0;

    public ScanPage(
        ICameraService cameraService,
        IImageProcessingService imageProcessingService,
        IClipboardService clipboardService,
        ScanSyncService syncService,
        IDragDropService? dragDropService = null)
    {
        InitializeComponent();

        _cameraService = cameraService;
        _imageProcessingService = imageProcessingService;
        _clipboardService = clipboardService;
        _dragDropService = dragDropService;
        _syncService = syncService;

        // 监听来自其他设备的新扫描
        _syncService.NewScanReceived += OnNewScanReceived;

        // 监听连接状态变化
        _syncService.ConnectionStateChanged += OnConnectionStateChanged;

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
            SetLoading(true);
            StatusLabel.Text = "正在拍摄...";

            // 1. 拍照
            var photoBytes = await _cameraService.TakePhotoAsync();
            if (photoBytes == null)
            {
                StatusLabel.Text = "拍摄已取消";
                return;
            }

            StatusLabel.Text = "正在处理图像...";

            // 2. 处理图像（边缘检测 + 透视变换）
            var result = await _imageProcessingService.ProcessScanAsync(photoBytes, false);

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

            // 5. 自动上传到服务器
            _ = UploadToServerAsync(result.ImageData, result.Width, result.Height);
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

    protected override async void OnAppearing()
    {
        base.OnAppearing();

        // 页面显示时自动连接到服务器
        if (!_syncService.IsConnected)
        {
            await _syncService.ConnectAsync();
        }
    }

    protected override async void OnDisappearing()
    {
        base.OnDisappearing();

        // 页面隐藏时断开连接以节省资源（可选）
        // await _syncService.DisconnectAsync();
    }

    private async Task UploadToServerAsync(byte[] imageData, int width, int height)
    {
        try
        {
            var success = await _syncService.UploadScanAsync(imageData, width, height);
            if (success)
            {
                System.Diagnostics.Debug.WriteLine("✓ 图片已上传到服务器");
            }
            else
            {
                System.Diagnostics.Debug.WriteLine("✗ 图片上传失败");
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"上传异常: {ex.Message}");
        }
    }

    private async void OnNewScanReceived(ScanImageDto scanImage)
    {
        try
        {
            // 在主线程上更新 UI
            await MainThread.InvokeOnMainThreadAsync(async () =>
            {
                System.Diagnostics.Debug.WriteLine($"收到新扫描: {scanImage.FileName}");

                // 下载图片
                var imageData = await _syncService.DownloadScanAsync(scanImage.DownloadUrl);
                if (imageData != null)
                {
                    // 显示图片
                    _currentImageData = imageData;
                    _currentRotation = 0;
                    PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(imageData));
                    PreviewImage.IsVisible = true;
                    PlaceholderLabel.IsVisible = false;
                    SaveButton.IsEnabled = true;
                    RotateButtonsGrid.IsVisible = true;

                    StatusLabel.Text = $"✓ 收到新扫描: {scanImage.FileName} ({scanImage.Width}×{scanImage.Height})";

                    // 自动复制到剪贴板
                    await _clipboardService.CopyImageToClipboardAsync(imageData);
                }
            });
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"处理新扫描失败: {ex.Message}");
        }
    }

    private void OnConnectionStateChanged(bool isConnected)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            if (isConnected)
            {
                ConnectionStatusDot.Fill = new SolidColorBrush(Colors.Green);
                ConnectionStatusLabel.Text = "已连接";
            }
            else
            {
                ConnectionStatusDot.Fill = new SolidColorBrush(Colors.Red);
                ConnectionStatusLabel.Text = "未连接";
            }
        });
    }
}
