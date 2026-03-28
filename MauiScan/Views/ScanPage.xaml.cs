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
    private readonly IConfigService _configService;

    private byte[]? _currentImageData;
    private int _currentRotation = 0;
    private byte[]? _originalPhotoBytes; // 保存原始拍照图片（用于识别失败时的手动标注）
    private string? _lastUploadedFileName; // 记录最近上传的文件名，避免收到自己上传的通知

    public ScanPage(
        ICameraService cameraService,
        IImageProcessingService imageProcessingService,
        IClipboardService clipboardService,
        ScanSyncService syncService,
        IConfigService configService,
        IDragDropService? dragDropService = null)
    {
        InitializeComponent();

        _cameraService = cameraService;
        _imageProcessingService = imageProcessingService;
        _clipboardService = clipboardService;
        _dragDropService = dragDropService;
        _syncService = syncService;
        _configService = configService;

        // 监听来自其他设备的新扫描
        _syncService.NewScanReceived += OnNewScanReceived;

        // 监听连接状态变化
        _syncService.ConnectionStateChanged += OnConnectionStateChanged;

        // 监听错误事件
        _syncService.ErrorOccurred += OnErrorOccurred;

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
        await OnCaptureClickedAsync();
    }

    /// <summary>
    /// 处理系统相机拍回来的照片
    /// </summary>
    private async Task ProcessCapturedPhotoAsync(byte[] photoBytes)
    {
        try
        {
            StatusLabel.Text = "正在处理图像...";
            SetLoading(true);

            // 保存原图（用于手动标注）
            _originalPhotoBytes = photoBytes;

            // 处理图像（边缘检测 + 透视变换）
            var result = await _imageProcessingService.ProcessScanAsync(photoBytes, false);

            if (!result.IsSuccess)
            {
                // 识别失败：进入预览页面的ROI模式
                StatusLabel.Text = $"自动识别失败 - {result.ErrorMessage ?? "未知错误"}";

                // 上传失败的原图到 training 目录
                _ = UploadToServerAsync(photoBytes, photoBytes, 0, 0);

                // 进入预览页的ROI/四点模式，让用户手动调整
                await ShowPreviewAsync(photoBytes, isAutoSuccess: false);
                return;
            }

            System.Diagnostics.Debug.WriteLine($"扫描结果 - Width: {result.Width}, Height: {result.Height}");

            // 跳转预览页
            await ShowPreviewAsync(result.ImageData, isAutoSuccess: true);
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

    private async void OnRefreshClicked(object sender, EventArgs e)
    {
        try
        {
            SetLoading(true);
            StatusLabel.Text = "正在获取最新扫描...";

            if (!_syncService.IsConnected)
                await _syncService.ConnectAsync();

            await LoadLatestScanFromServerAsync();
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"刷新失败: {ex.Message}";
        }
        finally
        {
            SetLoading(false);
        }
    }

    private async Task ShowPreviewAsync(byte[] imageData, bool isAutoSuccess = true)
    {
        var previewPage = new ScanPreviewPage(
            imageData: imageData,
            isAutoSuccess: isAutoSuccess,
            imageProcessingService: _imageProcessingService,
            originalPhoto: isAutoSuccess ? null : _originalPhotoBytes
        );

        previewPage.Confirmed += async (finalData) =>
        {
            // 用户确认使用：更新主页显示、复制、上传
            _currentImageData = finalData;
            _currentRotation = 0;
            PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(finalData));
            PreviewImage.IsVisible = true;
            PlaceholderLabel.IsVisible = false;
            SaveButton.IsEnabled = true;

            var copied = await _clipboardService.CopyImageToClipboardAsync(finalData);
            StatusLabel.Text = copied
                ? "✓ 扫描成功 | 已复制到剪贴板"
                : "✓ 扫描成功";

            _ = UploadToServerAsync(_originalPhotoBytes!, finalData, 0, 0);
        };

        previewPage.Retake += async () =>
        {
            // 用户重拍：直接再次启动相机
            await OnCaptureClickedAsync();
        };

        await Navigation.PushAsync(previewPage);
    }

    private async Task OnCaptureClickedAsync()
    {
        try
        {
            var photoBytes = await _cameraService.TakePhotoAsync();
            if (photoBytes != null)
                await ProcessCapturedPhotoAsync(photoBytes);
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"启动相机失败: {ex.Message}", "确定");
            StatusLabel.Text = "发生错误";
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

    private async void OnManualAnnotationClicked(object sender, EventArgs e)
    {
#if ANDROID
        if (_originalPhotoBytes == null)
        {
            StatusLabel.Text = "没有可标注的图片";
            return;
        }

        try
        {
            // 启动 Android 平台特定的手动标注 Activity
            var manualAnnotationService = Handler?.MauiContext?.Services.GetService<IManualAnnotationService>();
            if (manualAnnotationService == null)
            {
                await DisplayAlert("错误", "手动标注服务不可用", "确定");
                return;
            }

            StatusLabel.Text = "启动手动标注...";

            // 调用手动标注服务
            var annotationResult = await manualAnnotationService.AnnotateAsync(_originalPhotoBytes);

            if (annotationResult != null && annotationResult.Success)
            {
                // 标注成功：显示扣出的文档
                _currentImageData = annotationResult.ProcessedImageData;
                _currentRotation = 0;
                PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(annotationResult.ProcessedImageData));
                PreviewImage.IsVisible = true;
                PlaceholderLabel.IsVisible = false;
                SaveButton.IsEnabled = true;
                ManualAnnotationButton.IsVisible = false;

                // 自动复制到剪贴板
                var copied = await _clipboardService.CopyImageToClipboardAsync(annotationResult.ProcessedImageData);
                if (copied)
                {
                    StatusLabel.Text = $"✓ 手动标注成功 ({annotationResult.Width}×{annotationResult.Height}) | 已复制到剪贴板";
                }
                else
                {
                    StatusLabel.Text = $"✓ 手动标注成功 ({annotationResult.Width}×{annotationResult.Height})";
                }

                // 上传到服务器
                _ = UploadToServerAsync(_originalPhotoBytes, annotationResult.ProcessedImageData, annotationResult.Width, annotationResult.Height);

                // 上传训练数据（原图 + 标注信息）
                _ = UploadTrainingDataAsync(_originalPhotoBytes, annotationResult.Corners, annotationResult.ProcessedImageData);
            }
            else
            {
                StatusLabel.Text = "手动标注已取消";
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"手动标注失败: {ex.Message}", "确定");
            StatusLabel.Text = "手动标注失败";
        }
#endif
    }

    private async Task UploadTrainingDataAsync(byte[] originalImage, float[] corners, byte[] processedImage)
    {
        try
        {
            System.Diagnostics.Debug.WriteLine("开始上传训练数据...");
            var success = await _syncService.UploadTrainingDataAsync(originalImage, corners, processedImage);

            if (success)
            {
                System.Diagnostics.Debug.WriteLine("✓ 训练数据上传成功");
            }
            else
            {
                System.Diagnostics.Debug.WriteLine("✗ 训练数据上传失败");
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"上传训练数据异常: {ex.Message}");
        }
    }


    private void SetLoading(bool isLoading)
    {
        LoadingIndicator.IsRunning = isLoading;
        LoadingIndicator.IsVisible = isLoading;
        CaptureButton.IsEnabled = !isLoading;
        RefreshButton.IsEnabled = !isLoading;
        SaveButton.IsEnabled = !isLoading && _currentImageData != null;
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();

        // 初始化配置
        try
        {
            var config = await _configService.LoadConfigAsync();
            System.Diagnostics.Debug.WriteLine($"配置加载成功: ServerUrl={config.ServerUrl}, ApiKey={(string.IsNullOrEmpty(config.ApiKey) ? "未设置" : "已设置")}");
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"配置加载失败: {ex.Message}");
        }

        // 页面显示时自动连接到服务器
        if (!_syncService.IsConnected)
        {
            await _syncService.ConnectAsync();

            // 连接成功后，自动获取服务器上的最新图片
            await LoadLatestScanFromServerAsync();
        }
    }

    private async Task LoadLatestScanFromServerAsync()
    {
        try
        {
            var recentScans = await _syncService.GetRecentScansAsync(1);

            if (recentScans.Count > 0)
            {
                var latestScan = recentScans[0];

                // 下载最新的图片
                var imageData = await _syncService.DownloadScanAsync(latestScan.DownloadUrl);

                if (imageData != null)
                {
                    // 显示图片
                    _currentImageData = imageData;
                    _currentRotation = 0;
                    PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(imageData));
                    PreviewImage.IsVisible = true;
                    PlaceholderLabel.IsVisible = false;
                    SaveButton.IsEnabled = true;
    
                    StatusLabel.Text = $"✓ 已加载最新扫描: {latestScan.Width}×{latestScan.Height}";

                    // 自动复制到剪贴板
                    await _clipboardService.CopyImageToClipboardAsync(imageData);
                }
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"加载最新扫描失败: {ex.Message}");
            // 静默失败，不影响用户体验
        }
    }

    protected override async void OnDisappearing()
    {
        base.OnDisappearing();

        // 页面隐藏时断开连接以节省资源（可选）
        // await _syncService.DisconnectAsync();
    }

    private async Task UploadToServerAsync(byte[] originalImageData, byte[] processedImageData, int width, int height)
    {
        try
        {
            var uploadedFileName = await _syncService.UploadScanAsync(originalImageData, processedImageData, width, height);
            if (uploadedFileName != null)
            {
                // 记录上传的文件名，用于过滤自己上传的通知
                _lastUploadedFileName = uploadedFileName;
                System.Diagnostics.Debug.WriteLine($"✓ 图片已上传到服务器: {uploadedFileName}");
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
            // 忽略自己刚上传的图片
            if (_lastUploadedFileName != null && scanImage.FileName == _lastUploadedFileName)
            {
                System.Diagnostics.Debug.WriteLine($"忽略自己上传的图片: {scanImage.FileName}");
                _lastUploadedFileName = null; // 重置
                return;
            }

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

    private async void OnErrorOccurred(string errorMessage)
    {
        await MainThread.InvokeOnMainThreadAsync(async () =>
        {
            await DisplayAlert("同步错误", errorMessage, "确定");
        });
    }
}
