using MauiScan.Services;

namespace MauiScan.Platforms.Android.Services;

/// <summary>
/// Android 相机服务实现（使用自定义相机页面）
/// </summary>
public class CameraService : ICameraService
{
    public async Task<byte[]?> TakePhotoAsync()
    {
        // 检查权限
        if (!await CheckPermissionsAsync())
        {
            var granted = await RequestPermissionsAsync();
            if (!granted)
                throw new Exception("相机权限被拒绝");
        }

        // 启动自定义相机页面并等待结果
        return await CameraPageService.CapturePhotoAsync();
    }

    public async Task<bool> CheckPermissionsAsync()
    {
        var status = await Permissions.CheckStatusAsync<Permissions.Camera>();
        return status == PermissionStatus.Granted;
    }

    public async Task<bool> RequestPermissionsAsync()
    {
        var status = await Permissions.RequestAsync<Permissions.Camera>();
        return status == PermissionStatus.Granted;
    }
}

/// <summary>
/// 相机页面服务 - 管理相机页面的导航和结果
/// </summary>
public static class CameraPageService
{
    private static TaskCompletionSource<byte[]?>? _captureCompletionSource;

    public static async Task<byte[]?> CapturePhotoAsync()
    {
        _captureCompletionSource = new TaskCompletionSource<byte[]?>();

        // 导航到相机页面
        await Shell.Current.GoToAsync("camera");

        // 等待拍照结果
        return await _captureCompletionSource.Task;
    }

    public static void SetResult(byte[]? imageData)
    {
        _captureCompletionSource?.TrySetResult(imageData);
    }

    public static void Cancel()
    {
        _captureCompletionSource?.TrySetResult(null);
    }
}
