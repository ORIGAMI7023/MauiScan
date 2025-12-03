using MauiScan.Services;

namespace MauiScan.Platforms.Android.Services;

/// <summary>
/// Android 相机服务实现（使用 MAUI MediaPicker）
/// </summary>
public class CameraService : ICameraService
{
    public async Task<byte[]?> TakePhotoAsync()
    {
        try
        {
            // 检查权限
            if (!await CheckPermissionsAsync())
            {
                var granted = await RequestPermissionsAsync();
                if (!granted)
                    return null;
            }

            // 使用 MAUI MediaPicker 拍照
            var photo = await MediaPicker.Default.CapturePhotoAsync(new MediaPickerOptions
            {
                Title = "拍摄文档"
            });

            if (photo == null)
                return null;

            // 读取图像字节数据
            using var stream = await photo.OpenReadAsync();
            using var memoryStream = new MemoryStream();
            await stream.CopyToAsync(memoryStream);

            return memoryStream.ToArray();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"拍照失败: {ex.Message}");
            return null;
        }
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
