using MauiScan.Controls;
#if ANDROID
using MauiScan.Platforms.Android.Services;
#endif

namespace MauiScan.Views;

public partial class CameraPage : ContentPage
{
    private bool _isCaptured = false;

    public CameraPage()
    {
        InitializeComponent();
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();
#if ANDROID
        MainActivity.VolumeKeyPressed += OnVolumeKeyPressed;
#endif
    }

    protected override void OnDisappearing()
    {
#if ANDROID
        MainActivity.VolumeKeyPressed -= OnVolumeKeyPressed;
#endif
        base.OnDisappearing();
    }

    private void OnVolumeKeyPressed(object? sender, EventArgs e)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            OnCaptureClicked(sender, e);
        });
    }

    private void OnCaptureClicked(object? sender, EventArgs e)
    {
        if (_isCaptured) return;

        LoadingIndicator.IsVisible = true;
        LoadingIndicator.IsRunning = true;

        CameraPreview.CapturePhoto();
    }

    private async void OnPhotoCaptured(object? sender, byte[] imageData)
    {
        if (_isCaptured) return;
        _isCaptured = true;

        System.Diagnostics.Debug.WriteLine($"[CameraPage] 收到图像: {imageData.Length} 字节");

#if ANDROID
        CameraPageService.SetResult(imageData);
#endif

        await MainThread.InvokeOnMainThreadAsync(async () =>
        {
            await Shell.Current.GoToAsync("..");
        });
    }

    private async void OnCameraError(object? sender, string error)
    {
        System.Diagnostics.Debug.WriteLine($"[CameraPage] 相机错误: {error}");

        await MainThread.InvokeOnMainThreadAsync(async () =>
        {
            await DisplayAlert("相机错误", error, "确定");
            await OnCancelAsync();
        });
    }

    private async void OnCancelClicked(object? sender, EventArgs e)
    {
        await OnCancelAsync();
    }

    private async Task OnCancelAsync()
    {
        if (_isCaptured) return;
        _isCaptured = true;

#if ANDROID
        CameraPageService.Cancel();
#endif

        await Shell.Current.GoToAsync("..");
    }

    protected override bool OnBackButtonPressed()
    {
        MainThread.BeginInvokeOnMainThread(async () => await OnCancelAsync());
        return true;
    }
}
