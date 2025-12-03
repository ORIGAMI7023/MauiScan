using MauiScan.Models;
using MauiScan.Services;

namespace MauiScan.Views;

public partial class ScanPage : ContentPage
{
    private readonly ICameraService _cameraService;
    private readonly IImageProcessingService _imageProcessingService;
    private readonly IClipboardService _clipboardService;

    private byte[]? _currentImageData;
    private int _currentRotation = 0;

    public ScanPage(
        ICameraService cameraService,
        IImageProcessingService imageProcessingService,
        IClipboardService clipboardService)
    {
        InitializeComponent();

        _cameraService = cameraService;
        _imageProcessingService = imageProcessingService;
        _clipboardService = clipboardService;
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
                await DisplayAlert("处理失败", result.ErrorMessage ?? "未知错误", "确定");
                StatusLabel.Text = "处理失败";
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
            var savedPath = await SaveToGalleryAndroidAsync(fileName, _currentImageData);
            StatusLabel.Text = $"✓ 已保存到相册: {fileName}";
            await DisplayAlert("保存成功", "图片已保存到系统相册", "确定");
#else
            // 其他平台: 保存到应用目录
            var filePath = Path.Combine(FileSystem.AppDataDirectory, fileName);
            await File.WriteAllBytesAsync(filePath, _currentImageData);
            StatusLabel.Text = $"✓ 已保存: {fileName}";
            await DisplayAlert("保存成功", $"文件已保存到:\n{filePath}", "确定");
#endif
        }
        catch (Exception ex)
        {
            await DisplayAlert("保存失败", ex.Message, "确定");
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
        contentValues.Put(Android.Provider.MediaStore.IMediaColumns.RelativePath, Android.OS.Environment.DirectoryPictures + "/MauiScan");

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
}
