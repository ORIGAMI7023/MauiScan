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
            PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(result.ImageData));
            PreviewImage.IsVisible = true;
            PlaceholderLabel.IsVisible = false;
            SaveButton.IsEnabled = true;

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

    private void SetLoading(bool isLoading)
    {
        LoadingIndicator.IsRunning = isLoading;
        LoadingIndicator.IsVisible = isLoading;
        CaptureButton.IsEnabled = !isLoading;
        SaveButton.IsEnabled = !isLoading && _currentImageData != null;
        EnhanceButton.IsEnabled = !isLoading;
    }
}
