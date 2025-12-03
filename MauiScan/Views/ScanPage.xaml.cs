using MauiScan.Models;
using MauiScan.Services;

namespace MauiScan.Views;

public partial class ScanPage : ContentPage
{
    private readonly ICameraService _cameraService;
    private readonly IImageProcessingService _imageProcessingService;
    private readonly IClipboardService _clipboardService;

    private byte[]? _currentImageData;
    private bool _enhancementEnabled = false;
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
            var result = await _imageProcessingService.ProcessScanAsync(photoBytes, _enhancementEnabled);

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

            StatusLabel.Text = $"✓ 扫描成功 ({result.Width}×{result.Height})";

            // 4. 自动复制到剪贴板
            var copied = await _clipboardService.CopyImageToClipboardAsync(result.ImageData);
            if (copied)
            {
                StatusLabel.Text += " | 已复制到剪贴板";
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
            // 保存到 Pictures 目录
            var fileName = $"Scan_{DateTime.Now:yyyyMMdd_HHmmss}.jpg";
            var filePath = Path.Combine(FileSystem.AppDataDirectory, fileName);

            await File.WriteAllBytesAsync(filePath, _currentImageData);

            StatusLabel.Text = $"✓ 已保存: {fileName}";
            await DisplayAlert("保存成功", $"文件已保存到:\n{filePath}", "确定");
        }
        catch (Exception ex)
        {
            await DisplayAlert("保存失败", ex.Message, "确定");
        }
    }

    private void OnEnhanceToggled(object sender, EventArgs e)
    {
        _enhancementEnabled = !_enhancementEnabled;

        if (_enhancementEnabled)
        {
            EnhanceButton.Text = "✨ 增强模式: 开";
            EnhanceButton.BackgroundColor = Color.FromArgb("#FF5722");
        }
        else
        {
            EnhanceButton.Text = "✨ 增强模式: 关";
            EnhanceButton.BackgroundColor = Color.FromArgb("#FFC107");
        }

        StatusLabel.Text = _enhancementEnabled ? "增强模式已启用（灰度+对比度）" : "增强模式已关闭";
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
                StatusLabel.Text = $"已旋转 {(degrees > 0 ? "右" : "左")} 90°";

                // 更新剪贴板
                await _clipboardService.CopyImageToClipboardAsync(rotatedData);
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
        EnhanceButton.IsEnabled = !isLoading;
    }
}
