using Android.Content;
using Android.Provider;
using MauiScan.Services;

namespace MauiScan.Platforms.Android.Services;

/// <summary>
/// Android 相机服务实现（使用系统相机）
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

        // 启动系统相机并等待结果
        return await CameraPageService.LaunchSystemCameraAsync();
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
/// 系统相机拍照服务 - 管理拍照状态和临时文件
/// </summary>
public static class CameraPageService
{
    public const int REQUEST_CODE = 1001;

    private static TaskCompletionSource<byte[]?>? _captureCompletionSource;
    private static string? _tempFilePath;

    /// <summary>
    /// 启动系统相机并异步等待结果
    /// </summary>
    public static Task<byte[]?> LaunchSystemCameraAsync()
    {
        _captureCompletionSource = new TaskCompletionSource<byte[]?>();

        var activity = Platform.CurrentActivity!;
        var cacheDir = activity.CacheDir!;
        var timeStamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var tempFile = new Java.IO.File(cacheDir, $"MauiScan_capture_{timeStamp}.jpg");
        _tempFilePath = tempFile.AbsolutePath;

        var uri = AndroidX.Core.Content.FileProvider.GetUriForFile(activity, activity.PackageName + ".fileprovider", tempFile);
        var intent = new Intent(MediaStore.ActionImageCapture);
        intent.PutExtra(MediaStore.ExtraOutput, uri);
        intent.AddFlags(ActivityFlags.GrantReadUriPermission | ActivityFlags.GrantWriteUriPermission);
        intent.AddFlags(ActivityFlags.NoAnimation);

        activity.StartActivityForResult(intent, REQUEST_CODE);

        return _captureCompletionSource.Task;
    }

    /// <summary>
    /// 由 MainActivity.OnActivityResult 在拍照成功后调用
    /// </summary>
    public static void CompleteCapture()
    {
        if (_captureCompletionSource == null) return;

        var filePath = _tempFilePath;
        _tempFilePath = null;

        if (filePath != null && System.IO.File.Exists(filePath))
        {
            try
            {
                var bytes = System.IO.File.ReadAllBytes(filePath);
                System.IO.File.Delete(filePath);
                _captureCompletionSource.TrySetResult(bytes);
                return;
            }
            catch { }
        }

        _captureCompletionSource.TrySetResult(null);
    }

    /// <summary>
    /// 由 MainActivity.OnActivityResult 在取消时调用
    /// </summary>
    public static void CancelCapture()
    {
        if (_tempFilePath != null)
        {
            try { System.IO.File.Delete(_tempFilePath); } catch { }
            _tempFilePath = null;
        }
        _captureCompletionSource?.TrySetResult(null);
    }

}
