namespace MauiScan.Services;

/// <summary>
/// 相机服务接口（跨平台）
/// </summary>
public interface ICameraService
{
    /// <summary>
    /// 拍摄照片并返回图像字节数据
    /// </summary>
    /// <returns>图像数据（JPEG/PNG 格式），失败返回 null</returns>
    Task<byte[]?> TakePhotoAsync();

    /// <summary>
    /// 检查相机权限是否已授予
    /// </summary>
    Task<bool> CheckPermissionsAsync();

    /// <summary>
    /// 请求相机权限
    /// </summary>
    Task<bool> RequestPermissionsAsync();
}
