using MauiScan.Controls;
#if ANDROID
using MauiScan.Platforms.Android.Services;
#endif

namespace MauiScan.Views;

public partial class CameraPage : ContentPage
{
    private bool _isCaptured = false;
    private bool _isLandscape = false;

    public CameraPage()
    {
        InitializeComponent();
    }

    private void OnPageSizeChanged(object? sender, EventArgs e)
    {
        var isLandscape = Width > Height;
        if (isLandscape == _isLandscape)
            return;

        _isLandscape = isLandscape;
        UpdateLayout(isLandscape);
    }

    private void UpdateLayout(bool isLandscape)
    {
        if (isLandscape)
        {
            // 横屏：控制栏在右侧
            MainGrid.RowDefinitions.Clear();
            MainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Star });

            MainGrid.ColumnDefinitions.Clear();
            MainGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Star });
            MainGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

            Grid.SetRow(CameraPreview, 0);
            Grid.SetColumn(CameraPreview, 0);

            Grid.SetRow(ControlBar, 0);
            Grid.SetColumn(ControlBar, 1);

            ControlBar.HeightRequest = -1; // 自动高度
            ControlBar.WidthRequest = 120;

            // 调整按钮布局为垂直
            CancelButton.HorizontalOptions = LayoutOptions.Center;
            CancelButton.VerticalOptions = LayoutOptions.Start;
            CaptureButton.HorizontalOptions = LayoutOptions.Center;
            CaptureButton.VerticalOptions = LayoutOptions.Center;
        }
        else
        {
            // 竖屏：控制栏在底部
            MainGrid.RowDefinitions.Clear();
            MainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Star });
            MainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

            MainGrid.ColumnDefinitions.Clear();
            MainGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Star });

            Grid.SetRow(CameraPreview, 0);
            Grid.SetColumn(CameraPreview, 0);

            Grid.SetRow(ControlBar, 1);
            Grid.SetColumn(ControlBar, 0);

            ControlBar.HeightRequest = 120;
            ControlBar.WidthRequest = -1; // 自动宽度

            // 调整按钮布局为水平
            CancelButton.HorizontalOptions = LayoutOptions.Start;
            CancelButton.VerticalOptions = LayoutOptions.Center;
            CaptureButton.HorizontalOptions = LayoutOptions.Center;
            CaptureButton.VerticalOptions = LayoutOptions.Center;
        }
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
